#include "models/qwen4_exp/program/prefix/prefix_cache.h"

#include "core/device.h"
#include "core/device_guard.h"

#include <algorithm>
#include <exception>
#include <stdexcept>
#include <string>
#include <utility>

namespace infernix::models::qwen4_exp::prefix {

namespace pc = runtime::prefix_cache;

namespace {

// Largest single pinned allocation of the slab pool: one very large pinned allocation can fail or
// stall under WDDM where several smaller ones succeed. A slab never crosses chunks.
constexpr std::size_t kHostChunkBytes = std::size_t{4} << 30U;
constexpr std::size_t kSlabAlignment  = 4096;

std::size_t align_slab(std::size_t bytes) { return (bytes + kSlabAlignment - 1U) / kSlabAlignment * kSlabAlignment; }

} // namespace

PrefixHostLayout plan_prefix_host_layout(const KvPageGeometry& pages, const StateImageLayout& image,
                                         std::uint64_t host_bytes) {
    PrefixHostLayout out;
    out.page        = plan_host_kv_page_layout(pages.geometry);
    out.slab_bytes  = align_slab(out.page.page_stride);
    out.image_bytes = image.image_bytes;
    out.image_slabs = static_cast<std::uint32_t>((image.image_bytes + out.slab_bytes - 1U) / out.slab_bytes);
    const std::uint64_t slabs = host_bytes / out.slab_bytes;
    if (slabs > pc::kNoId / 4U) { throw std::invalid_argument("Qwen4Exp prefix cache: the Host tier is too large"); }
    out.slabs           = static_cast<std::uint32_t>(slabs);
    out.slabs_per_chunk = static_cast<std::uint32_t>(std::max<std::size_t>(1U, kHostChunkBytes / out.slab_bytes));
    return out;
}

PrefixCache::PrefixCache(const PrefixCacheConfig& config, DeviceKVPagePool& leases, const KvPageGeometry& pages,
                         std::span<const PrefixSegment> segments, const LaneStateImage& lanes,
                         std::uint64_t host_bytes)
    : taps_(config.taps), leases_(&leases), geometry_(pages), lanes_(&lanes),
      host_(plan_prefix_host_layout(pages, lanes.layout(), host_bytes)) {
    // The segments split the whole model's page group by KV layer, each over the head pool's pages.
    if (segments.empty() || segments.size() != lanes.segments()) {
        throw std::invalid_argument("Qwen4Exp prefix cache: the segments do not match the lane image's");
    }
    const std::uint32_t attention_layers = pages.kv_layers - (segments.back().mtp ? 1U : 0U);
    std::uint32_t kv_layers              = 0;
    for (const PrefixSegment& segment : segments) {
        const bool has_pool = segment.pool != nullptr;
        if (has_pool != (segment.pages.kv_layers != 0) || (segment.mtp && &segment != &segments.back()) ||
            (has_pool && (segment.pool->plane_count() != segment.pages.geometry.planes.size() ||
                          segment.pool->capacity_pages() != leases.capacity_pages())) ||
            segment.attention_base + segment.pages.kv_layers - (segment.mtp ? 1U : 0U) > attention_layers) {
            throw std::invalid_argument("Qwen4Exp prefix cache: a segment's page geometry does not match its pool");
        }
        kv_layers += segment.pages.kv_layers;
    }
    if (kv_layers != pages.kv_layers || leases.plane_count() != segments.back().pages.geometry.planes.size()) {
        throw std::invalid_argument("Qwen4Exp prefix cache: the page geometry does not match the pool");
    }
    if (host_.slabs == 0) {
        // Host-born snapshots are the only snapshot source: without a Host tier there is no cache.
        throw std::invalid_argument("Qwen4Exp prefix cache needs a Host tier (--host-context-mib)");
    }
    if (host_.slabs < host_.image_slabs + 2U) {
        const std::uint64_t mib = (static_cast<std::uint64_t>(host_.image_slabs + 2U) * host_.slab_bytes + (1U << 20U) - 1U) >> 20U;
        throw std::invalid_argument("--host-context-mib is too small for the Qwen4Exp prefix cache: one snapshot needs " +
                                    std::to_string(mib) + " MiB");
    }
    pc::PrefixIndexConfig index = config.index;
    index.max_nodes             = config.kv_pages + host_.slabs + 1U;
    index.max_snapshots         = config.device_snapshot_slots + host_.slabs / host_.image_slabs + 1U;
    index.host_slabs            = host_.slabs;
    index.image_slabs           = host_.image_slabs;
    index.device_snapshot_slots = config.device_snapshot_slots;
    index.block_bytes           = host_.page.page_stride;
    index.image_bytes           = host_.image_bytes;
    index.host_blocks           = true;
    if (config.device_snapshot_slots != 0) {
        throw std::invalid_argument("Qwen4Exp prefix cache: Device snapshot slots are not supported");
    }
    const std::uint64_t total = static_cast<std::uint64_t>(host_.slabs) * host_.slab_bytes;
    std::uint64_t pinned      = 0;
    for (std::uint32_t first = 0; first < host_.slabs; first += host_.slabs_per_chunk) {
        const std::uint32_t slabs = std::min(host_.slabs_per_chunk, host_.slabs - first);
        const std::size_t bytes   = static_cast<std::size_t>(slabs) * host_.slab_bytes;
        try {
            chunks_.emplace_back(bytes);
        } catch (const std::exception& error) {
            throw std::runtime_error("pinning the Qwen4Exp prefix cache Host tier failed after " +
                                     std::to_string(pinned >> 20U) + " of " + std::to_string(total >> 20U) + " MiB (" +
                                     error.what() + "); lower --host-context-mib");
        }
        pinned += bytes;
    }
    const StateImageLayout& layout = lanes.layout();
    segments_.reserve(segments.size());
    for (std::size_t s = 0; s < segments.size(); ++s) {
        Segment& segment = segments_.emplace_back();
        segment.spec     = segments[s];
        if (segment.spec.pool != nullptr) {
            segment.host = segment_host_page_layout(host_.page, segment.spec.pages, segment.spec.attention_base,
                                                    attention_layers, segment.spec.mtp);
        }
        segment.last_group = s + 1 == segments.size() ? layout.mtp_group() : 0U;
    }
    for (std::uint32_t g = 0; g + 1 < layout.groups(); ++g) {
        Segment& segment = segments_[lanes.group_segment(g)];
        if (&segment != &segments_.back()) { segment.last_group = std::max(segment.last_group, g); }
    }
    node_mtp_next_.assign(index.max_nodes, -1);
    snapshot_meta_.assign(index.max_snapshots, StateImageHeader{});
    blocks_.reserve(index.max_nodes);
    index_.emplace(index, *this);
    // The streams last: a constructor that throws runs no destructor, so the ones made before a
    // failure are destroyed here.
    try {
        for (Segment& segment : segments_) {
            const DeviceGuard on(segment.spec.device);
            CUDA_CHECK(cudaStreamCreateWithFlags(&segment.transfer, cudaStreamNonBlocking));
            CUDA_CHECK(cudaStreamCreateWithFlags(&segment.restore, cudaStreamNonBlocking));
        }
    } catch (...) {
        for (Segment& segment : segments_) {
            try {
                const DeviceGuard on(segment.spec.device);
                for (cudaStream_t stream : {segment.transfer, segment.restore}) {
                    if (stream != nullptr) { (void)cudaStreamDestroy(stream); }
                }
            } catch (...) {}
            segment.transfer = segment.restore = nullptr;
        }
        index_.reset();
        throw;
    }
}

PrefixCache::~PrefixCache() {
    // Only CUDA objects are released here, each with its segment's device current; leases go back
    // to the pool with blocks_.
    for (Segment& segment : segments_) {
        try {
            const DeviceGuard on(segment.spec.device);
            for (cudaStream_t stream : {segment.transfer, segment.restore}) {
                if (stream != nullptr) {
                    (void)cudaStreamSynchronize(stream);
                    (void)cudaStreamDestroy(stream);
                }
            }
        } catch (...) {}
    }
    for (PendingCapture& c : captures_) { recycle_groups(c.events); }
    for (PendingWrite& w : writes_) {
        for (std::uint32_t s = 0; s < w.done.size(); ++s) { recycle(s, w.done[s]); }
    }
    recycle_groups(restore_.events);
    for (RestoreBatch& batch : landing_) { recycle_groups(batch.events); }
    for (Segment& segment : segments_) {
        try {
            const DeviceGuard on(segment.spec.device);
            for (cudaEvent_t e : segment.spare) { (void)cudaEventDestroy(e); }
        } catch (...) {}
        segment.spare.clear();
    }
    index_.reset();
}

std::uint64_t PrefixCache::pinned_bytes() const noexcept {
    std::uint64_t bytes = 0;
    for (const PinnedHostBuffer& chunk : chunks_) { bytes += chunk.size(); }
    return bytes;
}

// ---- ids, events, slabs ----------------------------------------------------------------------------

std::uint32_t PrefixCache::allocate_block_id(DeviceKVPageLease&& lease) {
    if (!free_blocks_.empty()) {
        const std::uint32_t id = free_blocks_.back();
        free_blocks_.pop_back();
        blocks_[id].emplace(std::move(lease));
        return id;
    }
    if (blocks_.size() >= pc::kNoId - 1U) { throw std::overflow_error("Qwen4Exp prefix cache block ids exhausted"); }
    blocks_.emplace_back(std::move(lease));
    return static_cast<std::uint32_t>(blocks_.size() - 1U);
}

DeviceKVPageHandle PrefixCache::page(std::uint32_t device_id) const {
    if (device_id >= blocks_.size() || !blocks_[device_id]) {
        throw std::logic_error("Qwen4Exp prefix cache block id is not registered");
    }
    return blocks_[device_id]->handle();
}

std::byte* PrefixCache::slab(std::uint32_t id) const noexcept {
    const std::uint32_t chunk = id / host_.slabs_per_chunk;
    return static_cast<std::byte*>(chunks_[chunk].data()) +
           static_cast<std::size_t>(id % host_.slabs_per_chunk) * host_.slab_bytes;
}

cudaEvent_t PrefixCache::take_event(std::uint32_t segment) {
    std::vector<cudaEvent_t>& spare = segments_[segment].spare;
    if (!spare.empty()) {
        cudaEvent_t event = spare.back();
        spare.pop_back();
        return event;
    }
    cudaEvent_t event = nullptr;
    CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    return event;
}

void PrefixCache::recycle(std::uint32_t segment, cudaEvent_t event) noexcept {
    if (event != nullptr) { segments_[segment].spare.push_back(event); }
}

void PrefixCache::recycle_groups(std::vector<cudaEvent_t>& events) noexcept {
    for (std::uint32_t g = 0; g < events.size(); ++g) {
        try {
            recycle(lanes_->group_segment(g), events[g]);
        } catch (...) {}
    }
    events.clear();
}

bool PrefixCache::landed(std::span<const cudaEvent_t> events) const {
    for (const Segment& segment : segments_) {
        const cudaError_t status = cudaEventQuery(events[segment.last_group]);
        if (status == cudaErrorNotReady) { return false; }
        CUDA_CHECK(status);
    }
    return true;
}

void PrefixCache::synchronize(std::span<const cudaEvent_t> events) const {
    for (const Segment& segment : segments_) { CUDA_CHECK(cudaEventSynchronize(events[segment.last_group])); }
}

void PrefixCache::order_after(std::span<const cudaEvent_t> events, std::span<const cudaStream_t> consumers) const {
    check_streams(consumers);
    if (events.empty()) { return; }
    for (std::size_t s = 0; s < segments_.size(); ++s) {
        CUDA_CHECK(cudaStreamWaitEvent(consumers[s], events[segments_[s].last_group], 0));
    }
}

void PrefixCache::check_streams(std::span<const cudaStream_t> streams) const {
    if (streams.size() != segments_.size()) {
        throw std::invalid_argument("Qwen4Exp prefix cache: one stream per segment is needed");
    }
}

std::optional<std::uint32_t> PrefixCache::group_kv_layer(std::uint32_t group, std::uint32_t attention) const {
    const StateImageLayout& layout = lanes_->layout();
    const PrefixSegment& segment   = segments_[lanes_->group_segment(group)].spec;
    if (group == layout.mtp_group()) {
        if (!segment.mtp) { return std::nullopt; }
        return segment.pages.kv_layers - 1U;
    }
    if (layout.spec.layer_types[group] != MixerKind::Attention) { return std::nullopt; }
    return attention - segment.attention_base;
}

// ---- blocks --------------------------------------------------------------------------------------

pc::InsertResult PrefixCache::insert_block(pc::NodeRef parent, std::uint64_t lookup_hash,
                                           std::span<const TokenId> tokens, std::uint64_t extra,
                                           DeviceKVPageLease& lease, bool attach, std::int32_t mtp_next) {
    if (!lease.valid()) { throw std::invalid_argument("Qwen4Exp prefix cache: a block needs its page"); }
    const std::uint32_t id = allocate_block_id(std::move(lease));
    pc::InsertResult result;
    try {
        result = index_->insert_block(parent, lookup_hash, tokens, extra, id, attach, true);
    } catch (...) {
        lease = std::move(*blocks_[id]);
        blocks_[id].reset();
        free_blocks_.push_back(id);
        throw;
    }
    if (!result.device_attached) {
        // The index kept an existing copy: the page stays the caller's private page.
        lease = std::move(*blocks_[id]);
        blocks_[id].reset();
        free_blocks_.push_back(id);
        ++counters_.blocks_duplicate;
    } else {
        node_mtp_next_[result.node.index] = mtp_next;
        ++(result.inserted ? counters_.blocks_inserted : counters_.blocks_reattached);
    }
    return result;
}

std::int32_t PrefixCache::mtp_next(pc::NodeRef node) const {
    if (!index_->valid(node)) { throw std::logic_error("Qwen4Exp prefix cache: stale node"); }
    return node_mtp_next_[node.index];
}

std::uint32_t PrefixCache::adopt_tail(DeviceKVPageLease&& lease) { return allocate_block_id(std::move(lease)); }

const StateImageHeader& PrefixCache::meta(pc::SnapshotRef snapshot) const {
    if (!index_->valid(snapshot)) { throw std::logic_error("Qwen4Exp prefix cache: stale snapshot"); }
    return snapshot_meta_[snapshot.index];
}

// ---- captures and write-through ------------------------------------------------------------------

bool PrefixCache::capture(std::uint32_t lane, Capture&& capture, std::span<const cudaStream_t> producers) {
    const bool tail = !capture.tail.empty();
    if (tail != capture.tail_page.has_value() || capture.tail.size() >= pc::kBlockTokens) {
        throw std::invalid_argument("Qwen4Exp prefix cache: a capture's tail does not match its page");
    }
    check_streams(producers);
    // Before the image's slabs are reserved and the anchor pinned, which a throw would leak.
    const std::vector<std::int32_t> tail_page =
        tail ? KVExecutionTablePool::indices_of(std::span<const DeviceKVPageHandle>(&*capture.tail_page, 1))
             : std::vector<std::int32_t>{};
    auto image = index_->reserve_host_image(tail, capture.claim, true);
    if (!image) {
        ++counters_.taps_skipped;
        return false;
    }
    // The anchor must outlive the copy: the capturing lane may release its path before it lands.
    if (capture.anchor.valid()) { index_->pin_node(capture.anchor); }
    std::vector<std::byte*> slabs;
    slabs.reserve(host_.image_slabs);
    for (std::uint32_t i = 0; i < host_.image_slabs; ++i) { slabs.push_back(slab(image->slabs[i])); }
    write_state_image_header(lanes_->layout(), capture.meta, slabs.front());
    const HostImage destination{slabs, host_.slab_bytes};
    const StateImageLayout& layout = lanes_->layout();
    std::byte* const tail_record[1]   = {tail ? slab(image->slabs.back()) : nullptr};
    const std::uint32_t tail_group[1] = {tail ? slab_group(image->slabs.back()) : 0U};
    PendingCapture pending{std::move(capture), std::move(*image), {}};
    pending.events.reserve(layout.groups());
    try {
        // Each segment's transfer stream starts once its compute stream has enqueued the frontier's call.
        for (std::uint32_t s = 0; s < segments_.size(); ++s) {
            const DeviceGuard on(segments_[s].spec.device);
            cudaEvent_t start = take_event(s);
            CUDA_CHECK(cudaEventRecord(start, producers[s]));
            CUDA_CHECK(cudaStreamWaitEvent(segments_[s].transfer, start, 0));
            recycle(s, start);
        }
        std::uint32_t attention = 0;
        for (std::uint32_t g = 0; g < layout.groups(); ++g) {
            const std::uint32_t s = lanes_->group_segment(g);
            const Segment& segment = segments_[s];
            const DeviceGuard on(segment.spec.device);
            if (const auto kv_layer = group_kv_layer(g, attention); kv_layer && tail) {
                // The layer's planes of the partial page (columns past the frontier are don't-care).
                const PlaneRange planes = kv_layer_planes(segment.spec.pages, *kv_layer);
                segment.spec.pool->copy_indices_to_host_records(tail_page, tail_record, tail_group, segment.host,
                                                                planes.begin, planes.end, segment.transfer);
            }
            if (g != layout.mtp_group() && layout.spec.layer_types[g] == MixerKind::Attention) { ++attention; }
            lanes_->copy_group_to_host(g, lane, destination, segment.transfer);
            cudaEvent_t event = take_event(s);
            CUDA_CHECK(cudaEventRecord(event, segment.transfer));
            pending.events.push_back(event);
        }
    } catch (...) {
        // Nothing waits on a partial capture: its copies are drained, its slabs given back.
        for (const Segment& segment : segments_) { (void)cudaStreamSynchronize(segment.transfer); }
        recycle_groups(pending.events);
        index_->release_host_image(std::move(pending.image));
        if (pending.capture.anchor.valid()) { index_->unpin_node(pending.capture.anchor); }
        throw;
    }
    counters_.host_write_bytes += host_.image_bytes + (tail ? host_.page.page_stride : 0U);
    ++counters_.host_image_writes;
    captures_.push_back(std::move(pending));
    return true;
}

std::span<const cudaEvent_t> PrefixCache::capture_events(std::uint64_t id) const noexcept {
    for (const PendingCapture& c : captures_) {
        if (c.capture.id == id) { return c.events; }
    }
    return {};
}

void PrefixCache::order_after_capture(std::uint64_t id, std::span<const cudaStream_t> consumers) const {
    order_after(capture_events(id), consumers);
}

void PrefixCache::write_blocks(std::span<const pc::NodeRef> nodes, std::span<const cudaStream_t> producers) {
    check_streams(producers);
    std::vector<pc::NodeRef> written;
    std::vector<DeviceKVPageHandle> pages;
    std::vector<std::byte*> records;
    std::vector<std::uint32_t> groups;
    for (const pc::NodeRef node : nodes) {
        if (!index_->valid(node)) { continue; }
        const pc::NodeView view = index_->node(node);
        if (view.device != pc::CopyState::Resident || view.host != pc::CopyState::Absent) { continue; }
        index_->pin_node(node);
        // A block's Host copy serves only the snapshots on its path: it takes free slabs, dead KV
        // and superseded snapshots, never a retained snapshot (the lineage's resume source when
        // its endpoint did not fit). Blocks the Host tier cannot back stay on the Device.
        const auto slab_id = index_->begin_host_fill(node, 0.0);
        if (!slab_id) {
            index_->unpin_node(node);
            break; // the Host tier cannot free a slab for anything worth less
        }
        written.push_back(node);
        pages.push_back(page(view.device_id));
        records.push_back(slab(*slab_id));
        groups.push_back(slab_group(*slab_id));
    }
    if (written.empty()) { return; }
    PendingWrite write{std::move(written), std::vector<cudaEvent_t>(segments_.size(), nullptr)};
    try {
        const std::vector<std::int32_t> indices = KVExecutionTablePool::indices_of(pages);
        // Each segment copies its planes of the pages after its compute stream's queued work.
        for (std::uint32_t s = 0; s < segments_.size(); ++s) {
            Segment& segment = segments_[s];
            if (segment.spec.pool == nullptr) { continue; }
            const DeviceGuard on(segment.spec.device);
            cudaEvent_t start = take_event(s);
            CUDA_CHECK(cudaEventRecord(start, producers[s]));
            CUDA_CHECK(cudaStreamWaitEvent(segment.transfer, start, 0));
            recycle(s, start);
            segment.spec.pool->copy_indices_to_host_records(indices, records, groups, segment.host, 0,
                                                            segment.spec.pages.geometry.planes.size(),
                                                            segment.transfer);
            write.done[s] = take_event(s);
            CUDA_CHECK(cudaEventRecord(write.done[s], segment.transfer));
        }
    } catch (...) {
        // A partial write: the copies already enqueued drain, the nodes' Host fills are abandoned
        // (their slabs go back) and their pins dropped, so nothing stays Filling or pinned.
        for (const Segment& segment : segments_) { (void)cudaStreamSynchronize(segment.transfer); }
        for (const pc::NodeRef node : write.nodes) {
            try {
                index_->abort_host_fill(node);
                index_->unpin_node(node);
            } catch (...) {}
        }
        for (std::uint32_t s = 0; s < write.done.size(); ++s) { recycle(s, write.done[s]); }
        throw;
    }
    counters_.host_block_writes += write.nodes.size();
    counters_.host_write_bytes += write.nodes.size() * host_.page.page_stride;
    writes_.push_back(std::move(write));
}

void PrefixCache::poll() {
    while (!captures_.empty()) {
        PendingCapture& c = captures_.front();
        if (!landed(c.events)) { break; }
        std::optional<std::uint32_t> tail_device;
        if (c.capture.tail_lease) { tail_device = adopt_tail(std::move(*c.capture.tail_lease)); }
        const pc::PublishResult result =
            index_->publish_host_snapshot(c.capture.anchor, c.capture.frontier, c.capture.tail, tail_device,
                                          std::move(c.image), c.capture.kind);
        if (result.created) {
            snapshot_meta_[result.snapshot.index] = c.capture.meta;
            ++(c.capture.kind == pc::SnapshotKind::Endpoint ? counters_.endpoints_created : counters_.taps_created);
        }
        if (c.capture.supersedes && index_->valid(*c.capture.supersedes) &&
            (!result.snapshot.valid() || *c.capture.supersedes != result.snapshot)) {
            index_->supersede(*c.capture.supersedes);
        }
        if (c.capture.anchor.valid()) { index_->unpin_node(c.capture.anchor); }
        published_[c.capture.id] = result;
        recycle_groups(c.events);
        captures_.pop_front();
    }
    while (!writes_.empty()) {
        PendingWrite& w = writes_.front();
        bool done       = true;
        for (const cudaEvent_t event : w.done) {
            if (event == nullptr) { continue; }
            const cudaError_t status = cudaEventQuery(event);
            if (status == cudaErrorNotReady) {
                done = false;
                break;
            }
            CUDA_CHECK(status);
        }
        if (!done) { break; }
        for (const pc::NodeRef node : w.nodes) {
            index_->complete_host_fill(node);
            index_->unpin_node(node);
        }
        for (std::uint32_t s = 0; s < w.done.size(); ++s) { recycle(s, w.done[s]); }
        writes_.pop_front();
    }
    while (!landing_.empty()) {
        RestoreBatch& batch = landing_.front();
        if (!landed(batch.events)) { break; }
        finish_restore(batch);
        landing_.pop_front();
    }
}

void PrefixCache::drain() {
    for (const Segment& segment : segments_) {
        CUDA_CHECK(cudaStreamSynchronize(segment.transfer));
        CUDA_CHECK(cudaStreamSynchronize(segment.restore));
    }
    poll();
}

bool PrefixCache::await_extended_endpoint(std::span<const pc::NodeRef> path, std::span<const TokenId> tokens) {
    for (const PendingCapture& c : captures_) {
        if (c.capture.kind != pc::SnapshotKind::Endpoint || tokens.size() <= c.capture.frontier) { continue; }
        const std::uint32_t k = c.capture.frontier / pc::kBlockTokens;
        if (k > 0 && (path.size() < k || path[k - 1] != c.capture.anchor)) { continue; }
        if (!std::equal(c.capture.tail.begin(), c.capture.tail.end(),
                        tokens.begin() + static_cast<std::ptrdiff_t>(k) * pc::kBlockTokens)) {
            continue;
        }
        synchronize(c.events);
        poll();
        return true;
    }
    return false;
}

std::optional<pc::PublishResult> PrefixCache::capture_result(std::uint64_t id) const {
    const auto found = published_.find(id);
    if (found == published_.end()) { return std::nullopt; }
    return found->second;
}

void PrefixCache::forget_capture(std::uint64_t id) noexcept { published_.erase(id); }

// ---- restores --------------------------------------------------------------------------------------

void PrefixCache::open_restore(std::span<const cudaStream_t> producers) {
    if (restore_.open) { throw std::logic_error("Qwen4Exp prefix cache: a restore is already open"); }
    check_streams(producers);
    restore_        = RestoreBatch{};
    restore_.open   = true;
    restore_.ticket = RestoreTicket{next_ticket_++};
    for (std::uint32_t s = 0; s < segments_.size(); ++s) {
        const DeviceGuard on(segments_[s].spec.device);
        cudaEvent_t start = take_event(s);
        CUDA_CHECK(cudaEventRecord(start, producers[s]));
        CUDA_CHECK(cudaStreamWaitEvent(segments_[s].restore, start, 0));
        recycle(s, start);
    }
}

void PrefixCache::restore_block(pc::NodeRef node, DeviceKVPageLease&& destination) {
    if (!restore_.open) { throw std::logic_error("Qwen4Exp prefix cache: no restore is open"); }
    const pc::NodeView view = index_->node(node);
    if (view.host != pc::CopyState::Resident || view.device != pc::CopyState::Absent) {
        throw std::logic_error("Qwen4Exp prefix cache: restored block is not Host-only");
    }
    const DeviceKVPageHandle handle = destination.handle();
    const std::uint32_t id          = allocate_block_id(std::move(destination));
    index_->begin_device_fill(node, id);
    index_->pin_node(node); // until the batch lands
    restore_.pins.push_back(node);
    restore_.nodes.push_back(node);
    restore_.pages.push_back(PageCopy{handle, slab(view.host_slab), slab_group(view.host_slab)});
    ++counters_.host_block_restores;
}

void PrefixCache::restore_node_page(pc::NodeRef node, DeviceKVPageHandle destination) {
    if (!restore_.open) { throw std::logic_error("Qwen4Exp prefix cache: no restore is open"); }
    const pc::NodeView view = index_->node(node);
    if (view.host != pc::CopyState::Resident) { throw std::logic_error("Qwen4Exp prefix cache: copied block has no Host copy"); }
    index_->pin_node(node);
    restore_.pins.push_back(node);
    restore_.pages.push_back(PageCopy{destination, slab(view.host_slab), slab_group(view.host_slab)});
}

void PrefixCache::restore_tail_page(pc::SnapshotRef snapshot, DeviceKVPageHandle destination) {
    if (!restore_.open) { throw std::logic_error("Qwen4Exp prefix cache: no restore is open"); }
    const pc::SnapshotView view = index_->snapshot(snapshot);
    if (view.host != pc::CopyState::Resident || view.tail_len == 0) {
        throw std::logic_error("Qwen4Exp prefix cache: restored tail has no Host copy");
    }
    index_->pin_snapshot(snapshot);
    restore_.snapshot_pins.push_back(snapshot);
    const std::uint32_t tail_slab = view.host_slabs.back();
    restore_.pages.push_back(PageCopy{destination, slab(tail_slab), slab_group(tail_slab)});
    ++counters_.host_tail_restores;
}

void PrefixCache::restore_image(pc::SnapshotRef snapshot, std::uint32_t lane) {
    if (!restore_.open || restore_.image) { throw std::logic_error("Qwen4Exp prefix cache: image restore is not open"); }
    const pc::SnapshotView view = index_->snapshot(snapshot);
    if (view.host != pc::CopyState::Resident) { throw std::logic_error("Qwen4Exp prefix cache: restored image is not on the Host"); }
    index_->pin_snapshot(snapshot);
    restore_.snapshot_pins.push_back(snapshot);
    restore_.image      = snapshot;
    restore_.image_lane = lane;
    ++counters_.host_image_restores;
}

RestoreTicket PrefixCache::submit_restore() {
    if (!restore_.open) { throw std::logic_error("Qwen4Exp prefix cache: no restore is open"); }
    const StateImageLayout& layout = lanes_->layout();
    std::vector<const std::byte*> slabs;
    if (restore_.image) {
        const pc::SnapshotView view = index_->snapshot(*restore_.image);
        for (std::uint32_t i = 0; i < host_.image_slabs; ++i) { slabs.push_back(slab(view.host_slabs[i])); }
    }
    const HostImageConst image{slabs, host_.slab_bytes};
    // The destination pages by index (the head's pool hands them out; every segment holds its
    // planes of the same pages) and their Host records.
    std::vector<DeviceKVPageHandle> handles;
    std::vector<const std::byte*> records;
    std::vector<std::uint32_t> groups;
    for (const PageCopy& c : restore_.pages) {
        handles.push_back(c.page);
        records.push_back(c.record);
        groups.push_back(c.group);
    }
    try {
        const std::vector<std::int32_t> pages = KVExecutionTablePool::indices_of(handles);
        std::uint32_t attention               = 0;
        for (std::uint32_t g = 0; g < layout.groups(); ++g) {
            const std::uint32_t s  = lanes_->group_segment(g);
            const Segment& segment = segments_[s];
            const DeviceGuard on(segment.spec.device);
            if (const auto kv_layer = group_kv_layer(g, attention); kv_layer && !pages.empty()) {
                const PlaneRange planes = kv_layer_planes(segment.spec.pages, *kv_layer);
                segment.spec.pool->copy_indices_from_host_records(records, groups, pages, segment.host, planes.begin,
                                                                  planes.end, segment.restore);
            }
            if (g != layout.mtp_group() && layout.spec.layer_types[g] == MixerKind::Attention) { ++attention; }
            if (restore_.image) { lanes_->copy_group_from_host(g, image, restore_.image_lane, segment.restore); }
            cudaEvent_t event = take_event(s);
            CUDA_CHECK(cudaEventRecord(event, segment.restore));
            restore_.events.push_back(event);
        }
    } catch (...) {
        // A partial batch: what was enqueued drains, then abort_restore returns the destinations
        // and drops the pins, so the next admission can open a restore.
        for (const Segment& segment : segments_) { (void)cudaStreamSynchronize(segment.restore); }
        abort_restore();
        throw;
    }
    counters_.host_restore_bytes += restore_.pages.size() * host_.page.page_stride + (restore_.image ? host_.image_bytes : 0U);
    restore_.open             = false;
    const RestoreTicket ticket = restore_.ticket;
    landing_.push_back(std::move(restore_));
    restore_ = RestoreBatch{};
    return ticket;
}

std::span<const cudaEvent_t> PrefixCache::restore_events(RestoreTicket ticket) const noexcept {
    for (const RestoreBatch& batch : landing_) {
        if (batch.ticket.id == ticket.id) { return batch.events; }
    }
    return {};
}

void PrefixCache::order_after_restore(RestoreTicket ticket, std::span<const cudaStream_t> consumers) const {
    order_after(restore_events(ticket), consumers);
}

void PrefixCache::await_restore(RestoreTicket ticket) {
    // Each segment's restore stream completes batches in submission order.
    const auto events = restore_events(ticket);
    if (!events.empty()) { synchronize(events); }
    poll();
}

void PrefixCache::finish_restore(RestoreBatch& batch) noexcept {
    for (const pc::NodeRef node : batch.nodes) { index_->complete_device_fill(node); }
    for (const pc::NodeRef node : batch.pins) { index_->unpin_node(node); }
    for (const pc::SnapshotRef snapshot : batch.snapshot_pins) { index_->unpin_snapshot(snapshot); }
    recycle_groups(batch.events);
}

void PrefixCache::abort_restore() noexcept {
    if (!restore_.open) { return; }
    // Nothing was enqueued, or a failed submit_restore drained what it enqueued: the destinations go
    // back, the pins are dropped and its events are recycled.
    recycle_groups(restore_.events);
    for (const pc::NodeRef node : restore_.nodes) { index_->abort_device_fill(node); }
    for (const pc::NodeRef node : restore_.pins) { index_->unpin_node(node); }
    for (const pc::SnapshotRef snapshot : restore_.snapshot_pins) { index_->unpin_snapshot(snapshot); }
    restore_ = RestoreBatch{};
}

void PrefixCache::clear() noexcept {
    try {
        drain();
    } catch (...) {}
    abort_restore(); // an admission that failed between open_restore and submit_restore
    for (RestoreBatch& batch : landing_) { recycle_groups(batch.events); }
    for (PendingCapture& c : captures_) { recycle_groups(c.events); }
    for (PendingWrite& w : writes_) {
        for (std::uint32_t s = 0; s < w.done.size(); ++s) { recycle(s, w.done[s]); }
    }
    landing_.clear();
    captures_.clear();
    writes_.clear();
    published_.clear();
    pc::PrefixIndexConfig config = index_->config();
    index_.reset();
    blocks_.clear();
    free_blocks_.clear();
    std::fill(node_mtp_next_.begin(), node_mtp_next_.end(), -1);
    std::fill(snapshot_meta_.begin(), snapshot_meta_.end(), StateImageHeader{});
    index_.emplace(config, *this);
}

// ---- PrefixIndexBackend ------------------------------------------------------------------------------

void PrefixCache::release_device_block(std::uint32_t device_id) noexcept {
    if (device_id >= blocks_.size() || !blocks_[device_id]) { std::terminate(); }
    blocks_[device_id].reset(); // the lease returns its page to the pool
    free_blocks_.push_back(device_id);
}

void PrefixCache::release_snapshot(pc::SnapshotRef snapshot) noexcept {
    if (snapshot.index < snapshot_meta_.size()) { snapshot_meta_[snapshot.index] = StateImageHeader{}; }
}

void PrefixCache::drop_snapshot_device_image(pc::SnapshotRef) noexcept {
    // Snapshots are Host-born; there are no Device images to drop.
}

} // namespace infernix::models::qwen4_exp::prefix
