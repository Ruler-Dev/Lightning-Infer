// The layer walk of Qwen4Exp's prefill (docs/maintainer/qwen3_8-flash-next-design.md §19.3.8 F4):
// a span of consecutive streamed calls of one prompt runs layer-major, so each decoder layer's
// non-resident experts cross PCIe once per span instead of once per call. A chunk-major prompt of
// 4,096-token calls re-streams nearly every non-resident expert for every call (~51 GB at the x8
// link's 27.6 GB/s, ~1.8 s per call), more than the call's compute; a span of 16 calls streams them
// once and is bound by compute.
//
// The span keeps the prompt's call grid: chunk c of layer l reads exactly what the chunk-major
// call c reads at layer l (its own residual columns, and the layer's KV, GDN, PLE and QSA state
// after chunks 0..c-1), so every output bit equals the chunk-major prefill's. Only placement
// differs (resident, streamed or staged experts), which the MoE's arithmetic does not see.
//
// The span's residual stream (BF16 [S*H, tokens]) and every chunk's staged io image live in the
// stream lease after the ring. Each advance_prefill call enqueues about one call's work (a few
// layers of every chunk) and synchronizes, so cancellation and other lanes interleave as between
// chunk-major calls; the walk's own state stays valid across them. A span ends where a prefix tap
// is realized (the captured state must be the boundary's in every layer), before a call too narrow
// to stream, and at kWalkMaxTokens.
//
// A layer split (multi-GPU WP16) walks on every segment: each segment's lease holds its own walk
// area (the span is as long as every segment can lend), every chunk's io image is uploaded to each
// segment's area (the n-gram rows to the PLE segment only), and layer l runs on the segment that
// owns it. After a segment's last layer of chunk c, the boundary's BulkHandoff moves chunk c's
// residual columns into the next segment's walk residual, ordered on both streams without a host
// wait, so the crossing of chunk c overlaps the sender's later chunks. Each segment streams its own
// layers' experts. On two segments of one GPU their kernels never overlap (SegmentWork). A 3.1 GB/s
// x4 segment streams each non-resident expert once per span instead of once per 4,096-token call,
// which bounded its chunk-major prefill (~7 s of link per call at K = 28).
//
// MoE offload (multi-GPU WP23, a split of two segments): the first segment's MoE runs on the second,
// which streams the routed experts over its own link while it would otherwise wait for its part of
// the span. Per layer and chunk the first segment runs the layer up to the MoE's input and its
// routing (Forward::layer_front), its boundary crossing carries those (~5 KiB per column) to the
// second segment, whose Forward::moe_remote runs the dispatch, the routed experts, the shared expert
// (a copy of the layer's projections in the second segment's walk area) and the combine, and the
// result (5 KiB per column) comes back for Forward::layer_back. The units pipeline (the first
// segment fronts the rest of a layer's chunks while a chunk's experts run on the second), and side
// streams carry every copy, so neither compute stream waits for a link. The first segment then
// streams no expert at all (its lease holds no ring), and a span of one call is worth walking. Its
// link (one copy engine for both directions on these GPUs under WDDM) carries ~10 KiB per column
// and layer, which bounds the first part of a long span with the first segment's compute. The bits
// are layer()'s: the same kernels see the same inputs.

#include "models/qwen4_exp/program/program_impl.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <deque>
#include <stdexcept>

namespace infernix::models::qwen4_exp::detail {

namespace {

constexpr std::size_t kWalkAlign = 256;

std::size_t walk_align(std::size_t bytes) { return (bytes + kWalkAlign - 1) / kWalkAlign * kWalkAlign; }

} // namespace

std::size_t ProgramImpl::walk_bytes(std::uint32_t tokens, std::size_t chunks) const noexcept {
    return walk_align(2ULL * static_cast<std::size_t>(width_) * tokens) + chunks * walk_align(io_layout_.bytes);
}

DeviceSegment& ProgramImpl::layer_segment(std::uint32_t layer) {
    for (const auto& segment : segments_) {
        if (segment->spec.layers.contains(layer)) { return *segment; }
    }
    throw std::out_of_range("Qwen4Exp: no segment runs layer " + std::to_string(layer));
}

// Every tap ends a span, as on one GPU, so a split takes the unsplit Program's snapshots. The
// measurement knob SplitWalk::ExactTaps ends a split's span at exact taps only: a flexible one (a
// history ladder tap, which asks for a call boundary, not a split) is realized where the span ends,
// saving each extra span's fixed cost (the rings' first layers, the offload's pipeline filling).
bool ProgramImpl::walk_tap_due(const Lane& lane, std::size_t next_call) const {
    return prefix_tap_due(lane, next_call, segments_.size() > 1 && split_walk_ == SplitWalk::ExactTaps);
}

std::byte* ProgramImpl::walk_slot(const DeviceSegment& seg, std::size_t chunk) const noexcept {
    return walk_area(seg) + walk_.span.residual_bytes + chunk * walk_.span.io_stride;
}

namespace {

// The MoE offload's copies (walk_offload_layers). These GPUs have one copy engine each under WDDM,
// which runs a device's copies in the order they were enqueued: a copy enqueued before its input is
// ready holds every later one of that device, whatever its stream (measured: an upload of a chunk's
// result enqueued right after its experts kept the first segment's link idle while they ran).
//
// Staging slots of the first segment: a chunk's MoE inputs wait in one until its side stream has
// sent them, while the compute stream fronts the next chunks.
constexpr std::size_t kOffloadSlots = 8;
// The crossings' pinned rings, of 16 MiB pieces: the inputs (~20 MiB per 4,096-token chunk) can run
// ahead of the second segment by several chunks; the results ride kResultLag + 1 chunks deep. These
// are the least piece counts: a wider chunk takes more (prepare_moe_offload).
constexpr std::size_t kCrossingPiece  = 16ULL << 20;
constexpr std::uint32_t kInputPieces  = 16;
constexpr std::uint32_t kResultPieces = 8;
// A chunk's result is uploaded to the first segment after the next kResultLag chunks' inputs were
// downloaded there, by when it has normally arrived, but at least two chunks before its back needs
// it (a span of n chunks: n - 2 at most). 16K tokens (n = 4): 2,875 tok/s uploading each at once,
// 3,275 one chunk later, 3,406 two later; with the paced ring copies below, 3,621-3,646 two later
// against 2,987-3,031 three later.
constexpr std::size_t kResultLag = 2;
// The second segment's ring copies of the first segment's layers: in runs of at most this, enqueued
// in shares over the chunks of the layer before (ExpertStream::set_paced), so a layer's 1.35 GB do
// not hold the crossings for ~50 ms (16K tokens: 3,525 tok/s copied at once, 3,640 paced).
constexpr std::size_t kRemoteRun = 64ULL << 20;

} // namespace

// The bytes of one chunk's MoE inputs (x, routing ids, weights, shared gate) at `columns`.
std::size_t ProgramImpl::offload_input_bytes(std::size_t columns) const noexcept {
    const std::size_t H = c_.hidden_size, K = c_.moe.top_k;
    return walk_align(2 * H * columns) + 2 * walk_align(4 * K * columns) + walk_align(4 * columns);
}

// The MoE offload's part of the first segment's walk area, after the span's io images: per chunk
// the result of its routed experts (BF16 [H, chunk]), then per chunk its inject (FP32 [S, chunk]),
// then the staging slots of its MoE inputs.
std::size_t ProgramImpl::offload_bytes(std::size_t chunks) const noexcept {
    const auto columns = static_cast<std::size_t>(chunk_);
    return chunks * (walk_align(2ULL * c_.hidden_size * columns) + walk_align(4ULL * c_.hc.streams * columns)) +
           kOffloadSlots * offload_input_bytes(columns);
}

execution::ForwardBatch ProgramImpl::walk_batch(const DeviceSegment& seg, std::size_t c) {
    execution::ForwardBatch fb = io_batch(walk_slot(seg, c), 1, walk_.span.widths[c], 1);
    fb.host_slots              = std::span<const std::int32_t>(&walk_.span.host_slot, 1);
    fb.host_table_rows         = fb.host_slots;
    fb.mtp_chunk               = seg.spec.head && walk_.span.mtp[c] ? &*walk_.span.mtp[c] : nullptr;
    fb.vision                  = seg.spec.embed && walk_.span.vision[c] ? &*walk_.span.vision[c] : nullptr;
    // Chunk 0 waits per layer for the lane's restore and in-flight capture. Resolved here, not at the
    // walk's start: the walk spans engine steps, and a poll between them retires a landed batch and
    // frees its events (a landed batch needs no wait). Each segment's layers wait for their own
    // entries, recorded on that segment's streams.
    if (c == 0) { fb.layer_waits = prefix_waits(lanes_[walk_.span.lane]); }
    if (dflash2_) {
        const DeviceLayout& plan = head().plan;
        const std::int32_t total = walk_.span.offsets.back() + walk_.span.widths.back();
        const std::int32_t from  = total - std::min(total, dim(plan.dflash2_config->window));
        const std::int32_t lo    = walk_.span.offsets[c];
        const std::int32_t hi    = lo + walk_.span.widths[c];
        const std::int32_t begin = std::max(lo, from);
        if (begin < hi) {
            fb.draft_features = dflash_feature_columns(
                seg, plan.dflash_round_columns + begin - from, hi - begin);
        }
    }
    fb.stream         = true;
    fb.release_stream = c + 1 == walk_.span.widths.size();
    return fb;
}

void ProgramImpl::prepare_moe_offload() {
    DeviceSegment& a = *segments_[0];
    DeviceSegment& b = *segments_[1];
    // A ring piece is sent again only after it was received (BulkHandoff::send), so each ring holds
    // what is in flight at the widest chunk: one chunk's inputs (received once they were sent) and
    // kResultLag + 1 results (one sent while the kResultLag before it wait for their uploads). The
    // least counts hold the 4,096-token chunk's with room; from 6,656 tokens a result (5 KiB per
    // column) takes three pieces, and kResultLag + 1 of them outgrow kResultPieces.
    const auto columns   = static_cast<std::size_t>(chunk_);
    const auto pieces_of = [](std::size_t bytes) {
        return static_cast<std::uint32_t>((bytes + kCrossingPiece - 1) / kCrossingPiece);
    };
    const std::uint32_t per_result    = pieces_of(2ULL * c_.hidden_size * columns);
    const std::uint32_t input_pieces  = std::max(kInputPieces, pieces_of(offload_input_bytes(columns)));
    const std::uint32_t result_pieces = std::max(kResultPieces, static_cast<std::uint32_t>(kResultLag + 1) * per_result);
    moe_bulk_.push_back(std::make_unique<BulkHandoff>(a.device->device, b.device->device, kCrossingPiece, input_pieces));
    a.moe_out = moe_bulk_.back().get();
    moe_bulk_.push_back(std::make_unique<BulkHandoff>(b.device->device, a.device->device, kCrossingPiece, result_pieces));
    a.moe_back = moe_bulk_.back().get();
    for (DeviceSegment* seg : {&a, &b}) {
        const DeviceGuard on(seg->device->device);
        CUDA_CHECK(cudaStreamCreateWithFlags(&seg->side.stream, cudaStreamNonBlocking));
    }
    {
        const DeviceGuard on(a.device->device);
        CUDA_CHECK(cudaStreamCreateWithFlags(&a.side.out, cudaStreamNonBlocking));
        for (auto* list : {&a.side.staged, &a.side.sent}) {
            for (std::size_t k = 0; k < kOffloadSlots; ++k) {
                cudaEvent_t event = nullptr;
                CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
                list->push_back(event);
            }
        }
    }
    const LayerRange layers           = a.spec.layers;
    const std::uint32_t E             = c_.moe.experts;
    const std::uint64_t record_stride = a.plan.record_stride;
    std::vector<ops::offloaded_moe::ExpertScales> scales(static_cast<std::size_t>(layers.size()) * E);
    {
        const DeviceGuard on(a.device->device);
        for (std::uint32_t l = layers.begin; l < layers.end; ++l) {
            const auto& moe = a.parameters->layers[l].moe;
            b.remote_records.push_back(reinterpret_cast<const std::uint8_t*>(moe.bank->planes.records));
            CUDA_CHECK(cudaMemcpy(scales.data() + static_cast<std::size_t>(l - layers.begin) * E, moe.device_scales,
                                  E * sizeof(ops::offloaded_moe::ExpertScales), cudaMemcpyDeviceToHost));
        }
    }
    // The layers' shared-expert projections: each parent's bytes (the Weight's planes lie in it),
    // aligned in one pinned image, and Weights pointing into the image.
    {
        std::vector<const execution::LinearParameters*> sources;
        for (std::uint32_t l = layers.begin; l < layers.end; ++l) {
            const auto& moe = a.parameters->layers[l].moe;
            sources.push_back(&moe.shared_gate_up);
            sources.push_back(&moe.shared_down);
        }
        std::vector<std::size_t> offsets;
        std::size_t bytes = 0;
        for (const auto* p : sources) {
            const Weight& w = p->weight;
            if (w.payload == nullptr || w.payload_bytes == 0) {
                throw std::logic_error("Qwen4Exp MoE offload: a shared expert projection has no payload");
            }
            offsets.push_back(bytes);
            bytes += walk_align(w.payload_bytes);
        }
        b.remote_shared_host  = std::make_unique<PinnedHostBuffer>(bytes);
        b.remote_shared_bytes = bytes;
        auto* image           = static_cast<std::byte*>(b.remote_shared_host->data());
        const DeviceGuard on(a.device->device);
        for (std::size_t i = 0; i < sources.size(); ++i) {
            const Weight& w = sources[i]->weight;
            CUDA_CHECK(cudaMemcpy(image + offsets[i], w.payload, w.payload_bytes, cudaMemcpyDeviceToHost));
            const auto* begin = static_cast<const std::byte*>(w.payload);
            const auto moved  = [&](const void* p) -> const void* {
                if (p == nullptr) { return nullptr; }
                const auto* q = static_cast<const std::byte*>(p);
                if (q < begin || q >= begin + w.payload_bytes) {
                    throw std::logic_error("Qwen4Exp MoE offload: a shared expert plane lies outside its parent");
                }
                return image + offsets[i] + (q - begin);
            };
            execution::LinearParameters copy = *sources[i];
            copy.weight.payload = image + offsets[i];
            copy.weight.qdata   = moved(w.qdata);
            copy.weight.qhigh   = moved(w.qhigh);
            copy.weight.scales  = moved(w.scales);
            b.remote_shared.push_back(copy);
        }
    }
    const DeviceGuard on(b.device->device);
    b.remote_scales = DeviceBuffer(scales.size() * sizeof(ops::offloaded_moe::ExpertScales));
    CUDA_CHECK(cudaMemcpy(b.remote_scales.p, scales.data(), scales.size() * sizeof(ops::offloaded_moe::ExpertScales),
                          cudaMemcpyHostToDevice));
    b.remote_frames = DeviceBuffer(E * sizeof(std::int32_t));
    CUDA_CHECK(cudaMemset(b.remote_frames.p, 0xFF, E * sizeof(std::int32_t))); // -1: none resident
    b.remote_table.assign(static_cast<std::size_t>(layers.size()) * E, -1);
    execution::ExpertStream::RecordOf record_of = [records = b.remote_records, record_stride](std::uint32_t layer,
                                                                                            std::uint32_t expert) {
        return records[layer] + static_cast<std::size_t>(expert) * record_stride;
    };
    b.remote_stream = std::make_unique<execution::ExpertStream>(layers.size(), E, record_stride, std::move(record_of),
                                                                /*one_allocation=*/true);
    b.remote_stream->set_paced(kRemoteRun);
}

// The exclusive end of the span that starts at lane.next_call; lane.next_call when the call runs
// chunk-major. Lends each segment's stream lease (with the walk's area) when none is held, or lends
// it again with the area when the held one has none for the span (another lane's chunk-major calls
// lent it). Another prefilling lane waits while the span runs (context_blocks), so two long prompts
// walk span by span instead of both re-streaming every expert per call. A split's span is as long
// as every segment's area allows, and ends at the prefix taps (walk_tap_due); INFERNIX_SPLIT_WALK=0
// keeps a split's prompts chunk-major.
std::size_t ProgramImpl::walk_span_end(const Lane& lane, std::uint32_t index) {
    const std::size_t first = lane.next_call;
    if (walk_.span.active || (segments_.size() > 1 && split_walk_ == SplitWalk::ChunkMajor)) { return first; }
    for (const auto& segment : segments_) {
        if (!segment->expert_stream || segment->residency->frames() == 0) { return first; }
    }
    std::size_t end     = first;
    std::uint32_t from  = lane.state_tokens;
    std::uint32_t total = 0;
    std::vector<std::uint32_t> tokens{0}; // tokens of the span [first, first + i)
    while (end < lane.calls.size()) {
        const std::uint32_t width = lane.calls[end] - from;
        if (width < static_cast<std::uint32_t>(kStreamMinColumns) || total + width > kWalkMaxTokens) { break; }
        total += width;
        from = lane.calls[end];
        ++end;
        tokens.push_back(total);
        if (walk_tap_due(lane, end)) { break; }
    }
    // With the MoE offload one call is worth a span: the first segment's experts run on the second.
    const std::size_t least = moe_offload_ ? 1 : 2;
    if (end - first < least) { return first; }
    for (std::size_t s = 0; s < segments_.size(); ++s) {
        DeviceSegment& seg = *segments_[s];
        const SegmentWork on(*this, seg);
        // The offload's first segment streams nothing (no ring) and keeps its experts' results in
        // its area; its second streams the first one's layers through its ring, every expert of a
        // layer at once, and holds their shared experts in its area.
        const bool front     = moe_offload_ && s == 0;
        const auto area_of   = [&](std::uint32_t tokens, std::size_t chunks) {
            return walk_bytes(tokens, chunks) + (front ? offload_bytes(chunks) : 0) +
                   (moe_offload_ && s == 1 ? seg.remote_shared_bytes : 0);
        };
        const std::size_t probe = std::min<std::size_t>(least == 1 ? 1 : 2, end - first);
        if (seg.stream_lease.valid() &&
            (area_of(tokens[probe], probe) > seg.stream_walk_bytes || (front && seg.stream_ring_bytes > 0))) {
            // Between calls nothing reads the lease; its frames go back and come out again with the area.
            return_stream_lease(seg);
        }
        if (!seg.stream_lease.valid()) {
            // Sized for the prompt's largest span: as many tokens as remain (at most the cap) in calls
            // of the configured chunk, plus a remainder call.
            const std::uint32_t remaining = std::min(lane.prefill_end - lane.state_tokens, kWalkMaxTokens);
            const std::size_t chunks      = remaining / static_cast<std::uint32_t>(chunk_) + 2;
            stream_chunk(seg, static_cast<std::int32_t>(lane.calls[first] - lane.state_tokens), area_of(remaining, chunks),
                         /*ring_wanted=*/!front, moe_offload_ && s == 1 ? c_.moe.experts : 0U);
            if (!seg.stream_lease.valid() || (!front && seg.stream_ring_bytes == 0)) { return first; }
        }
        while (end - first >= least && area_of(tokens[end - first], end - first) > seg.stream_walk_bytes) { --end; }
    }
    return end - first >= least ? end : first;
}

// Chunk c's io image (staged at `host_slot`) into `seg`'s walk area on its stream: all of it to the
// PLE segment, which alone reads the n-gram rows, and the rest around the rows to the others
// (IoLayout, upload_io).
void ProgramImpl::walk_upload(DeviceSegment& seg, std::size_t chunk, const std::byte* host_slot) {
    const SegmentWork on(*this, seg);
    const cudaStream_t s = seg.device->stream;
    std::byte* slot      = walk_slot(seg, chunk);
    if (seg.spec.ple) {
        CUDA_CHECK(cudaMemcpyAsync(slot, host_slot, io_layout_.bytes, cudaMemcpyHostToDevice, s));
        return;
    }
    CUDA_CHECK(cudaMemcpyAsync(slot, host_slot, io_layout_.ngram, cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaMemcpyAsync(slot + io_layout_.mtp_ids, host_slot + io_layout_.mtp_ids,
                               io_layout_.bytes - io_layout_.mtp_ids, cudaMemcpyHostToDevice, s));
}

void ProgramImpl::walk_begin(Lane& lane, std::uint32_t index, std::size_t end) {
    walk_.span            = WalkState::LayerWalk{};
    walk_.span.lane       = index;
    walk_.span.first_call = lane.next_call;
    walk_.span.end_call   = end;
    walk_.span.began      = Clock::now();
    const std::size_t n   = end - lane.next_call;
    std::int32_t begin = static_cast<std::int32_t>(lane.state_tokens), offset = 0;
    for (std::size_t c = lane.next_call; c < end; ++c) {
        const auto width = static_cast<std::int32_t>(lane.calls[c]) - begin;
        walk_.span.begins.push_back(begin);
        walk_.span.widths.push_back(width);
        walk_.span.offsets.push_back(offset);
        begin += width;
        offset += width;
    }
    walk_.span.io_stride      = walk_align(io_layout_.bytes);
    walk_.span.residual_bytes = walk_align(2ULL * static_cast<std::size_t>(width_) * static_cast<std::size_t>(offset));
    walk_.span.layers_per_step =
        static_cast<std::uint32_t>((c_.num_hidden_layers + n - 1) / n); // about one call's work per step
    walk_.span.mtp.assign(n, std::nullopt);
    walk_.span.vision.assign(n, std::nullopt);
    walk_.span.mtp_vision.assign(n, {});
    walk_.span.host_slot = host_lanes_[0] = static_cast<std::int32_t>(index);
    if (walk_.host.size() < n * walk_.span.io_stride) {
        walk_.host     = PinnedHostBuffer(n * walk_.span.io_stride);
        walk_.prefetch = WalkState::WalkPrefetch{};
    }
    walk_dflash_frontier_ = lane.dflash_frontier;
    walk_.span.active = true;

    auto* host_area = static_cast<std::byte*>(walk_.host.data());
    // A tensor (or word pointer) into `seg`'s io image, moved to chunk c's copy in its walk area: the
    // MTP cells point into the head's io, the image columns into the first segment's.
    const auto rebase = [&](const void* p, const DeviceSegment& seg, std::size_t c) -> std::byte* {
        const auto* io = static_cast<const std::byte*>(seg.io_device.p);
        const auto* b  = static_cast<const std::byte*>(p);
        return b >= io && b < io + io_layout_.bytes ? walk_slot(seg, c) + (b - io) : const_cast<std::byte*>(b);
    };
    const auto rebase_tensor = [&](Tensor t, const DeviceSegment& seg, std::size_t c) {
        if (t.data != nullptr) { t.data = rebase(t.data, seg, c); }
        return t;
    };
    for (std::size_t s = 0; s < segments_.size(); ++s) {
        DeviceSegment& seg = *segments_[s];
        const SegmentWork on(*this, seg);
        if (segments_.size() > 1) { CUDA_CHECK(cudaEventRecord(seg.walk_first.event, seg.device->stream)); }
        seg.residency->before_round(seg.device->stream);
        if (!moe_offload_) {
            seg.expert_stream->begin(ring_span(seg), seg.residency->host_table(), seg.device->stream);
        } else if (s == 1) {
            // The first segment's layers through this ring until its part ends (walk_offload_cross),
            // and their shared experts into the area after the span's io images.
            seg.remote_stream->begin(ring_span(seg), seg.remote_table.data(), seg.device->stream);
            std::byte* shared = walk_slot(seg, n);
            CUDA_CHECK(cudaMemcpyAsync(shared, seg.remote_shared_host->data(), seg.remote_shared_bytes,
                                       cudaMemcpyHostToDevice, seg.device->stream));
            const auto* image = static_cast<const std::byte*>(seg.remote_shared_host->data());
            const auto moved  = [&](const void* p) -> const void* {
                return p == nullptr ? nullptr : shared + (static_cast<const std::byte*>(p) - image);
            };
            seg.remote_shared_bound = seg.remote_shared;
            for (auto& p : seg.remote_shared_bound) {
                p.weight.payload = moved(p.weight.payload);
                p.weight.qdata   = moved(p.weight.qdata);
                p.weight.qhigh   = moved(p.weight.qhigh);
                p.weight.scales  = moved(p.weight.scales);
            }
        }
        if (moe_offload_) {
            while (seg.side.events.size() < n) {
                cudaEvent_t event = nullptr;
                CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
                seg.side.events.push_back(event);
            }
        }
    }
    DeviceSegment& embedder = layer_segment(0);
    for (std::size_t c = 0; c < n; ++c) {
        const std::int32_t b = walk_.span.begins[c], w = walk_.span.widths[c];
        // Rows the previous span prefetched already stand at this chunk's slot.
        const bool prefetched = walk_prefetched(index, walk_.span.first_call, c, b, w);
        {
            const nvtx::ScopedRange rows_range(nvtx::Name::PrefillPleRows, nvtx::Category::Prefill,
                                               static_cast<std::uint64_t>(w));
            stage_sequence(index, b, w, lane.history, !prefetched);
        }
        std::optional<execution::MtpChunk> chunk = stage_mtp_chunk(lane, index, b, w);
        const execution::VisionInput* vision     = stage_vision(lane, static_cast<std::uint32_t>(b), w);
        std::byte* host_slot = host_area + c * walk_.span.io_stride;
        if (prefetched) {
            const std::size_t rows = static_cast<std::size_t>(w) * hash_.heads() * c_.ple.table.row_bytes;
            const std::size_t tail = io_layout_.ngram + rows;
            std::memcpy(host_slot, io_host_.data(), io_layout_.ngram);
            std::memcpy(host_slot + tail, static_cast<const std::byte*>(io_host_.data()) + tail, io_layout_.bytes - tail);
            ++walk_.prefetched_chunks;
        } else {
            std::memcpy(host_slot, io_host_.data(), io_layout_.bytes);
        }
        for (const auto& segment : segments_) { walk_upload(*segment, c, host_slot); }
        if (chunk) {
            const DeviceSegment& h  = head();
            chunk->ids              = rebase_tensor(chunk->ids, h, c);
            chunk->positions        = rebase_tensor(chunk->positions, h, c);
            chunk->rope_positions   = reinterpret_cast<std::int32_t*>(rebase(chunk->rope_positions, h, c));
            chunk->block_start_rope = reinterpret_cast<std::int32_t*>(rebase(chunk->block_start_rope, h, c));
            if (chunk->vision != nullptr) {
                const std::int32_t cells     = (chunk->prepend ? 1 : 0) + chunk->columns;
                const std::int32_t subchunks = (cells + execution::kMtpChunkColumns - 1) / execution::kMtpChunkColumns;
                auto& copies                 = walk_.span.mtp_vision[c];
                copies.assign(chunk->vision, chunk->vision + subchunks);
                for (auto& input : copies) { input.columns = rebase_tensor(input.columns, h, c); }
                chunk->vision = copies.data();
            }
            walk_.span.mtp[c] = *chunk;
        }
        if (vision != nullptr) {
            walk_.span.vision[c] =
                execution::VisionInput{vision->embeddings, rebase_tensor(vision->columns, first(), c)};
        }
        // The chunk's embedding and first layer run while the host stages the next chunk; with the
        // MoE offload the embedding alone (its layers pipeline over the span's chunks).
        const SegmentWork on(*this, embedder);
        const WideWork wide(*this, embedder, w);
        if (moe_offload_) {
            execution::ForwardBatch fb = walk_batch(embedder, c);
            Tensor residual(walk_area(embedder) + 2ULL * static_cast<std::size_t>(width_) *
                                                      static_cast<std::size_t>(walk_.span.offsets[c]),
                            DType::BF16, {width_, w});
            embedder.forward->begin_call(fb, nullptr);
            embedder.forward->embed(fb, residual);
        } else {
            walk_pass(c, 0);
        }
    }
    walk_.span.next_layer = moe_offload_ ? 0 : 1;
    walk_.prefetch        = WalkState::WalkPrefetch{}; // consumed; the slots now hold this span's chunks
    ++walk_.spans_run;
    walk_.calls_run += n;
}

std::uint64_t ProgramImpl::window_key() const noexcept {
    std::uint64_t h = 1469598103934665603ULL; // FNV-1a over the window's token ids
    for (const std::int32_t t : window_) {
        h ^= static_cast<std::uint32_t>(t);
        h *= 1099511628211ULL;
    }
    return h;
}

bool ProgramImpl::walk_prefetched(std::uint32_t index, std::size_t first_call, std::size_t c, std::int32_t begin,
                                  std::int32_t width) {
    const WalkState::WalkPrefetch& f = walk_.prefetch;
    if (!f.valid || f.lane != index || f.first_call != first_call || c >= f.widths.size() || f.begins[c] != begin ||
        f.widths[c] != width) {
        return false;
    }
    ngram_window(lanes_[index].history, begin, width);
    return window_key() == f.keys[c];
}

// Reads up to `chunks` more chunks of the span after the current one (calls from walk_.span.end_call,
// as walk_span_end would group them) into walk_.host's slots. Only after the span's first device
// synchronization: until then its own slots may still be uploading.
void ProgramImpl::walk_prefetch_rows(Lane& lane, std::uint32_t index, std::size_t chunks) {
    WalkState::WalkPrefetch& f = walk_.prefetch;
    if (!f.valid || f.lane != index || f.first_call != walk_.span.end_call) {
        f            = WalkState::WalkPrefetch{};
        f.valid      = true;
        f.lane       = index;
        f.first_call = walk_.span.end_call;
    }
    const std::size_t slots = walk_.host.size() / walk_.span.io_stride;
    std::uint32_t tokens    = 0;
    for (const auto w : f.widths) { tokens += static_cast<std::uint32_t>(w); }
    for (std::size_t added = 0; added < chunks && !f.done; ++added) {
        const std::size_t k = f.widths.size(), call = f.first_call + k;
        if (k >= slots || call >= lane.calls.size()) {
            f.done = true;
            break;
        }
        const std::uint32_t from  = lane.calls[call - 1];
        const std::uint32_t width = lane.calls[call] - from;
        if (width < static_cast<std::uint32_t>(kStreamMinColumns) || tokens + width > kWalkMaxTokens) {
            f.done = true;
            break;
        }
        const nvtx::ScopedRange rows_range(nvtx::Name::PrefillPleRows, nvtx::Category::Prefill,
                                           static_cast<std::uint64_t>(width));
        ngram_window(lane.history, static_cast<std::int32_t>(from), static_cast<std::int32_t>(width));
        auto* slot = static_cast<std::byte*>(walk_.host.data()) + k * walk_.span.io_stride;
        read_ngram_to(lane, static_cast<std::int32_t>(width), slot + io_layout_.ngram);
        f.begins.push_back(static_cast<std::int32_t>(from));
        f.widths.push_back(static_cast<std::int32_t>(width));
        f.keys.push_back(window_key());
        tokens += width;
        if (walk_tap_due(lane, call + 1)) { f.done = true; }
    }
}

// Layer l of chunk c on the segment that runs it (with the embedding before layer 0, the MTP cells
// and, for the prompt's last chunk, the logits after the last layer, and the crossing into the next
// segment after a segment's last layer). The caller holds the segment's SegmentWork and WideWork.
void ProgramImpl::walk_pass(std::size_t c, std::uint32_t l) {
    const std::size_t n   = walk_.span.widths.size();
    const std::uint32_t L = c_.num_hidden_layers;
    const bool prompt_end = walk_.span.end_call == lanes_[walk_.span.lane].calls.size();
    DeviceSegment& seg    = layer_segment(l);
    auto* area            = walk_area(seg);
    const std::int32_t w  = walk_.span.widths[c];
    execution::ForwardBatch fb = walk_batch(seg, c);
    const std::size_t column_bytes = 2ULL * static_cast<std::size_t>(width_);
    Tensor residual(area + column_bytes * static_cast<std::size_t>(walk_.span.offsets[c]), DType::BF16, {width_, w});
    const bool logits_here = l + 1 == L && prompt_end && c + 1 == n;
    Tensor logits(head().logits32.p, DType::FP32, {vocab_, 1});
    seg.forward->begin_call(fb, logits_here ? &logits : nullptr);
    if (l == 0) { seg.forward->embed(fb, residual); }
    seg.forward->layer(l, fb, residual);
    if (l + 1 == L) { seg.forward->finish(fb, residual, logits_here ? &logits : nullptr); }
    if (l + 1 == seg.spec.layers.end && !seg.spec.head && !moe_offload_) {
        // The boundary: chunk c's residual columns into the next segment's walk residual, at the
        // same columns; the receiver's first layer of chunk c is ordered after them.
        DeviceSegment& next = *segments_[seg.spec.index + 1];
        seg.bulk_out->transfer(residual.data,
                               walk_area(next) + column_bytes * static_cast<std::size_t>(walk_.span.offsets[c]),
                               residual.bytes(), seg.device->stream, next.device->stream);
        if (dflash2_ && fb.draft_features.data != nullptr) {
            const DeviceLayout& plan = head().plan;
            const std::int32_t total = walk_.span.offsets.back() + walk_.span.widths.back();
            const std::int32_t from  = total - std::min(total, dim(plan.dflash2_config->window));
            const std::int32_t lo    = walk_.span.offsets[c];
            const std::int32_t begin = std::max(lo, from);
            Tensor destination = dflash_feature_columns(
                next, plan.dflash_round_columns + begin - from, fb.draft_features.ne[1]);
            seg.bulk_out->transfer(fb.draft_features.data, destination.data, fb.draft_features.bytes(),
                                   seg.device->stream, next.device->stream);
        }
    }
}

namespace {

// Submits a stream's queued work without waiting (the batched driver model holds it until then).
void submit(cudaStream_t stream) {
    const cudaError_t status = cudaStreamQuery(stream);
    if (status == cudaErrorNotReady) {
        (void)cudaGetLastError();
        return;
    }
    CUDA_CHECK(status);
}

} // namespace

// Layers [from, to) of the first segment over the span's chunks, their MoE on the second (the MoE
// offload): per layer l and chunk c, the first segment's front (Forward::layer_front), its four
// tensors copied into a staging slot and sent from there by its side stream through moe_out (one
// run) into chunk c's columns of the second segment's walk residual (unused until the crossing),
// received on the second segment's side stream; the second segment's moe_remote once they landed
// and its result sent through moe_back on its compute stream, uploaded into the first segment's
// area on that one's side stream only after the next chunks' inputs were downloaded (kResultLag of
// them, fewer in a short span; a device's copies run in the order they were enqueued); then a share
// of the ring's next layer. The first segment's back (Forward::layer_back) comes as late as it can:
// just before chunk c's front of layer l + 1 (it reads chunk c's residual), so a layer's other
// fronts run while chunk c's experts and its two crossings do; a step's last backs wait for the next
// step's fronts. A chunk's front needs layer l's state after the earlier chunks' fronts and its own
// residual, nothing of another chunk's back, so the order changes no bit. Neither compute stream
// carries a copy over the first segment's link (its 20 MiB of inputs per 4,096-token chunk take
// ~6.5 ms at 3.1 GB/s, the 20 MiB result as long). On one GPU the two segments' kernels never
// overlap (SegmentWork), so each step waits for the other's.
void ProgramImpl::walk_offload_layers(std::uint32_t from, std::uint32_t to, std::int32_t widest) {
    DeviceSegment& a               = *segments_[0];
    DeviceSegment& b               = *segments_[1];
    const std::size_t n            = walk_.span.widths.size();
    const std::uint32_t E          = c_.moe.experts;
    const auto H                   = static_cast<std::int32_t>(c_.hidden_size);
    const auto K                   = static_cast<std::int32_t>(c_.moe.top_k);
    const auto S                   = static_cast<std::int32_t>(c_.hc.streams);
    const std::size_t column_bytes = 2ULL * static_cast<std::size_t>(width_);
    // The first segment's part of its walk area (offload_bytes), after the span's io images.
    const std::size_t y_stride      = walk_align(2ULL * c_.hidden_size * static_cast<std::size_t>(chunk_));
    const std::size_t inject_stride = walk_align(4ULL * c_.hc.streams * static_cast<std::size_t>(chunk_));
    const std::size_t slot_bytes    = offload_input_bytes(static_cast<std::size_t>(chunk_));
    std::byte* results              = walk_slot(a, n);
    std::byte* staging_slots        = results + n * (y_stride + inject_stride);
    // A share of the ring's next layer per chunk (ExpertStream::pump), and the results' lag.
    const std::size_t pace       = (static_cast<std::size_t>(b.remote_stream->half()) * a.plan.record_stride + n - 1) / n;
    const std::size_t result_lag = n > 2 ? std::min(kResultLag, n - 2) : 0;
    const auto residual_of          = [&](std::size_t c) {
        return Tensor(walk_area(a) + column_bytes * static_cast<std::size_t>(walk_.span.offsets[c]), DType::BF16,
                      {width_, walk_.span.widths[c]});
    };
    const auto y_of = [&](std::size_t c) {
        return Tensor(results + c * y_stride, DType::BF16, {H, walk_.span.widths[c]});
    };
    const auto inject_of = [&](std::size_t c) {
        return Tensor(results + n * y_stride + c * inject_stride, DType::FP32, {S, walk_.span.widths[c]});
    };
    // Results sent by the second segment whose uploads to the first are not enqueued yet, oldest
    // first: (layer, chunk, pieces).
    struct SentResult {
        std::uint32_t layer;
        std::size_t chunk;
        std::vector<BulkHandoff::Piece> pieces;
    };
    std::deque<SentResult> results_sent;
    const auto receive_result = [&] {
        const SentResult r = std::move(results_sent.front());
        results_sent.pop_front();
        const DeviceGuard on(a.device->device);
        a.moe_back->receive(r.pieces, y_of(r.chunk).data, a.side.stream);
        CUDA_CHECK(cudaEventRecord(a.side.events[r.chunk], a.side.stream));
        submit(a.side.stream);
    };
    // Chunk c's MoE inputs at `at`, in the order they cross: x, routing ids, weights, gate.
    const auto inputs_at = [&](std::byte* at, std::int32_t w) {
        std::array<Tensor, 4> t{Tensor(nullptr, DType::BF16, {H, w}), Tensor(nullptr, DType::I32, {K, w}),
                                Tensor(nullptr, DType::FP32, {K, w}), Tensor(nullptr, DType::FP32, {w})};
        for (Tensor& tensor : t) {
            tensor.data = at;
            at += walk_align(tensor.bytes());
        }
        return t;
    };
    // Layer l's front of chunk c, its inputs' crossing and its routed experts on the second segment.
    const auto front_of = [&](std::uint32_t l, std::size_t c) {
        std::vector<BulkHandoff::Piece> inputs;
        const std::int32_t w      = walk_.span.widths[c];
        const std::size_t slot    = walk_.span.fronts++ % kOffloadSlots;
        const std::uint32_t local = l - a.spec.layers.begin;
        const std::array<Tensor, 4> staged = inputs_at(staging_slots + slot * slot_bytes, w);
        // Into chunk c's columns of the second segment's walk residual, unused until the crossing.
        const std::array<Tensor, 4> landed =
            inputs_at(walk_area(b) + column_bytes * static_cast<std::size_t>(walk_.span.offsets[c]), w);
        {
            const SegmentWork on(*this, a);
            const WideWork wide(*this, a, widest);
            execution::ForwardBatch fb = walk_batch(a, c);
            Tensor residual            = residual_of(c);
            Tensor inject              = inject_of(c);
            execution::MoeFront front;
            a.forward->begin_call(fb, nullptr);
            a.forward->layer_front(l, fb, residual, inject, front);
            // The slot's previous inputs (kOffloadSlots fronts ago) have been sent.
            CUDA_CHECK(cudaStreamWaitEvent(a.device->stream, a.side.sent[slot], 0));
            const Tensor* produced[] = {&front.x, &front.routing.ids, &front.routing.weights,
                                        &front.routing.shared_gate};
            for (std::size_t i = 0; i < staged.size(); ++i) {
                CUDA_CHECK(cudaMemcpyAsync(staged[i].data, produced[i]->data, produced[i]->bytes(),
                                           cudaMemcpyDeviceToDevice, a.device->stream));
            }
            CUDA_CHECK(cudaEventRecord(a.side.staged[slot], a.device->stream));
            submit(a.device->stream);
            CUDA_CHECK(cudaStreamWaitEvent(a.side.out, a.side.staged[slot], 0));
            // The four tensors lie in one run at the same offsets on both sides.
            const auto* first = static_cast<const std::byte*>(staged.front().data);
            const auto* end   = static_cast<const std::byte*>(staged.back().data) + staged.back().bytes();
            inputs = a.moe_out->send(first, static_cast<std::size_t>(end - first), a.side.out);
            CUDA_CHECK(cudaEventRecord(a.side.sent[slot], a.side.out));
            submit(a.side.out);
        }
        {
            const DeviceGuard on(b.device->device);
            a.moe_out->receive(inputs, landed.front().data, b.side.stream);
            CUDA_CHECK(cudaEventRecord(b.side.events[c], b.side.stream));
            submit(b.side.stream);
        }
        {
            const SegmentWork on(*this, b);
            const WideWork wide(*this, b, widest);
            CUDA_CHECK(cudaStreamWaitEvent(b.device->stream, b.side.events[c], 0));
            const execution::RemoteMoe remote{
                .frames        = static_cast<const std::int32_t*>(b.remote_frames.p),
                .host_records  = b.remote_records[local],
                .record_stride = a.plan.record_stride,
                .scales        = static_cast<const ops::offloaded_moe::ExpertScales*>(b.remote_scales.p) +
                          static_cast<std::size_t>(local) * E,
                .shared_gate_up = &b.remote_shared_bound.at(2ULL * local),
                .shared_down    = &b.remote_shared_bound.at(2ULL * local + 1),
                .stream       = b.remote_stream.get(),
                .stream_layer = local};
            const Tensor y = b.forward->moe_remote(remote, landed[0], {landed[1], landed[2], landed[3]});
            results_sent.push_back({l, c, a.moe_back->send(y.data, y.bytes(), b.device->stream)});
            // The layer's half of the ring is free after its last chunk's result has left (the next
            // layer but one is listed), and a share of the listed copies follows each chunk's.
            if (c + 1 == n) { b.remote_stream->after_experts(local, b.device->stream); }
            b.remote_stream->pump(pace);
            submit(b.device->stream);
        }
        // The upload of the result result_lag chunks back follows this chunk's downloads.
        while (results_sent.size() > result_lag) { receive_result(); }
    };
    // Fronts whose backs are not enqueued yet, oldest first: (layer, chunk), carried across steps.
    auto& pending   = walk_.span.pending;
    const auto back = [&] {
        const auto [l, c] = pending.front();
        pending.pop_front();
        // Its result's upload first (and every older one's).
        while (std::any_of(results_sent.begin(), results_sent.end(),
                           [&](const SentResult& r) { return r.layer == l && r.chunk == c; })) {
            receive_result();
        }
        const SegmentWork on(*this, a);
        CUDA_CHECK(cudaStreamWaitEvent(a.device->stream, a.side.events[c], 0));
        Tensor residual = residual_of(c);
        a.forward->layer_back(l, residual, inject_of(c), y_of(c));
        if (dflash2_) { dflash2_walk_features(c, c + 1, l, residual); }
    };
    for (std::uint32_t l = from; l < to; ++l) {
        for (std::size_t c = 0; c < n; ++c) {
            // Chunk c's residual must be through layer l - 1: its back, and every older one, first.
            while (!pending.empty() && pending.front().first + 1 < l) { back(); }
            while (!pending.empty() && pending.front().first + 1 == l && pending.front().second <= c) { back(); }
            front_of(l, c);
            pending.emplace_back(l, c);
        }
    }
    while (!results_sent.empty()) { receive_result(); }
    // The last layer's backs before the crossing; a step's last layer leaves them to the next step,
    // whose fronts they precede, so the step's end drains nothing.
    if (to == a.spec.layers.end) {
        while (!pending.empty()) { back(); }
    }
    const DeviceGuard on(a.device->device);
    submit(a.device->stream);
}

// After the first segment's last layer with the MoE offload: every chunk's residual crosses into
// the second segment's walk residual (its MoE inputs there are spent: its stream ran their experts
// first), and the second segment's own layers start streaming through its ring, after the first
// one's last experts were read.
void ProgramImpl::walk_offload_cross() {
    DeviceSegment& a               = *segments_[0];
    DeviceSegment& b               = *segments_[1];
    const std::size_t column_bytes = 2ULL * static_cast<std::size_t>(width_);
    for (std::size_t c = 0; c < walk_.span.widths.size(); ++c) {
        const std::size_t at = column_bytes * static_cast<std::size_t>(walk_.span.offsets[c]);
        a.bulk_out->transfer(walk_area(a) + at, walk_area(b) + at,
                             column_bytes * static_cast<std::size_t>(walk_.span.widths[c]), a.device->stream,
                             b.device->stream);
    }
    if (dflash2_) {
        const DeviceLayout& plan = head().plan;
        const std::int32_t total = walk_.span.offsets.back() + walk_.span.widths.back();
        const std::int32_t n     = std::min(total, dim(plan.dflash2_config->window));
        const Tensor from = dflash_feature_columns(a, plan.dflash_round_columns, n);
        const Tensor to   = dflash_feature_columns(b, plan.dflash_round_columns, n);
        a.bulk_out->transfer(from.data, to.data, from.bytes(), a.device->stream, b.device->stream);
    }
    {
        const DeviceGuard on(a.device->device);
        a.device->flush();
    }
    const SegmentWork on(*this, b);
    b.remote_stream->end();
    b.expert_stream->begin(ring_span(b), b.residency->host_table(), b.device->stream);
    b.device->flush();
}

// Layers [next_layer, to_layer) of every chunk, grouped by segment: each group under its segment's
// SegmentWork (its device current; on a shared GPU after the other segments' kernels) and wide arena.
// A split's groups are submitted at once, as the next segment's stream waits for their crossings.
void ProgramImpl::walk_enqueue(std::uint32_t to_layer) {
    const std::int32_t widest = *std::max_element(walk_.span.widths.begin(), walk_.span.widths.end());
    std::uint32_t l           = walk_.span.next_layer;
    while (l < to_layer) {
        DeviceSegment& seg      = layer_segment(l);
        const std::uint32_t end = std::min(to_layer, seg.spec.layers.end);
        const bool split        = segments_.size() > 1;
        if (moe_offload_ && seg.spec.index == 0) {
            // The first segment's layers with their experts on the second (each step under its own
            // segment's SegmentWork and wide arena); it starts at layer 0, recorded in walk_begin.
            walk_offload_layers(l, end, widest);
            l = end;
            if (end == seg.spec.layers.end) {
                {
                    const DeviceGuard on(seg.device->device);
                    CUDA_CHECK(cudaEventRecord(seg.walk_last.event, seg.device->stream));
                }
                walk_offload_cross();
            }
            continue;
        }
        const SegmentWork on(*this, seg);
        const WideWork wide(*this, seg, widest);
        if (split && l == seg.spec.layers.begin && l > 0) {
            CUDA_CHECK(cudaEventRecord(seg.walk_first.event, seg.device->stream));
        }
        for (; l < end; ++l) {
            for (std::size_t c = 0; c < walk_.span.widths.size(); ++c) { walk_pass(c, l); }
        }
        if (split && end == seg.spec.layers.end) {
            CUDA_CHECK(cudaEventRecord(seg.walk_last.event, seg.device->stream));
        }
        if (split) { seg.device->flush(); }
    }
    walk_.span.next_layer = std::max(walk_.span.next_layer, to_layer);
}

PrefillProgress ProgramImpl::walk_step(Lane& lane, std::uint32_t index, Clock::time_point start) {
    const std::uint32_t L = c_.num_hidden_layers;
    std::uint32_t tokens  = 0;
    for (const auto w : walk_.span.widths) { tokens += static_cast<std::uint32_t>(w); }
    // The first step already ran layer 0 while staging (not with the MoE offload).
    const std::uint32_t to = std::min(
        L, walk_.span.next_layer + walk_.span.layers_per_step - (walk_.span.steps == 0 && walk_.span.next_layer > 0 ? 1U : 0U));
    {
        const nvtx::ScopedRange step_range(nvtx::Name::PrefillChunk, nvtx::Category::Prefill,
                                           static_cast<std::uint64_t>(tokens));
        walk_enqueue(to);
    }
    ++walk_.span.steps;
    if (to < L) {
        // While the device runs this step, read the next span's n-gram rows (one span's chunks spread
        // over the remaining steps; ~35-70 ms of NVMe reads per 4,096-token chunk).
        if (walk_.span.steps > (segments_.size() > 1 ? 2U : 1U) && walk_.span.end_call < lane.calls.size()) {
            const std::size_t remaining = (L - to + walk_.span.layers_per_step - 1) / walk_.span.layers_per_step + 1;
            const std::size_t slots     = walk_.host.size() / walk_.span.io_stride;
            const std::size_t have      = walk_.prefetch.valid ? walk_.prefetch.widths.size() : 0;
            walk_prefetch_rows(lane, index, slots > have ? (slots - have + remaining - 1) / remaining : 0);
        }
        {
            const nvtx::ScopedRange wait_range(nvtx::Name::DeviceWait, nvtx::Category::Prefill);
            if (segments_.size() > 1) {
                // A split waits for the previous step's end: the device runs this step meanwhile.
                const std::size_t k = walk_.span.steps % 2;
                for (const auto& segment : segments_) {
                    const DeviceGuard on(segment->device->device);
                    CUDA_CHECK(cudaEventRecord(segment->walk_steps[k].event, segment->device->stream));
                    segment->device->flush();
                }
                if (walk_.span.steps > 1) {
                    for (const auto& segment : segments_) {
                        const DeviceGuard on(segment->device->device);
                        CUDA_CHECK(cudaEventSynchronize(segment->walk_steps[1 - k].event));
                    }
                }
            } else {
                synchronize();
            }
        }
        for (const auto& segment : segments_) {
            const SegmentWork on(*this, *segment);
            segment->residency->tier_step(segment->device->stream);
        }
        PrefillProgress out;
        out.summary                 = prefill_summary(lane);
        out.processed_prompt_tokens = 0;
        out.complete                = false;
        out.timing.submit_host_ns   = elapsed_ns(start);
        lane.prefill_ns += out.timing.submit_host_ns;
        return out;
    }
    // The span is enqueued: the lane's state is at its end in every layer.
    const std::int32_t begin = walk_.span.begins.back(), width = walk_.span.widths.back();
    const bool last          = walk_.span.end_call == lane.calls.size();
    lane.state_tokens = lane.calls[walk_.span.end_call - 1];
    lane.next_call    = walk_.span.end_call;
    for (const auto& segment : segments_) {
        const SegmentWork on(*this, *segment);
        segment->residency->enqueue_route_download(segment->device->stream, width);
        segment->expert_stream->end();
    }
    if (dflash2_) {
        const SegmentWork on_head(*this, head());
        dflash2_walk_context(lane, index);
    }
    walk_.span.active = false;
    host_lanes_[0]    = static_cast<std::int32_t>(index);
    if (last && lane.vision) { vision_release(lane); } // ahead of the stream leases (vision_release)
    if (last && !prefilling_besides(index)) { return_leases(); }
    prefix_after_prefill_call(lane, index, last);
    if (!last) {
        {
            const nvtx::ScopedRange wait_range(nvtx::Name::DeviceWait, nvtx::Category::Prefill);
            synchronize();
        }
        walk_report(walk_.span.began);
        const nvtx::ScopedRange residency_range(nvtx::Name::PrefillResidency, nvtx::Category::Moe,
                                                static_cast<std::uint64_t>(width));
        // The route log holds the span's last chunk (every chunk routes nearly every expert).
        trace_round(RouteTraceKind::PrefillChunk, 1, width, tokens, 0, static_cast<std::uint32_t>(begin));
        after_call(width, std::vector<std::size_t>(segments_.size(), 0));
        apply_vram_target(false);
    } else {
        walk_report(walk_.span.began);
    }
    // A walk promotes no experts (its routes still count toward the cache's scores). Between spans
    // the stream lease holds most frames, so promotions would churn through the rest (~12,000
    // records, ~1.2 s of link per 64K span) and the next span would wait for them. After the
    // prompt they would fill the returned frames with the prompt's experts while the answer's own
    // staged misses land there anyway: 16 per layer and chunk measured slower than none for every
    // answer length (128K + 512 tokens: 23.16 -> 21.10 s; design §19.3.17).
    const std::vector<std::size_t> none(segments_.size(), 0);
    return prefill_progress(lane, index, begin, width, tokens, last, none, start);
}

// A split's span report (debug): its tokens and wall time, and each segment's time on its GPU from
// its first to its last pass (with the host steps between them), so where the span's time went.
// Waits for each segment's last pass (the end of the span's work).
void ProgramImpl::walk_report(Clock::time_point began) noexcept {
    if (segments_.size() < 2) { return; }
    try {
        std::uint32_t tokens = 0;
        for (const auto w : walk_.span.widths) { tokens += static_cast<std::uint32_t>(w); }
        std::string line = "layer walk span of " + std::to_string(walk_.span.widths.size()) + " chunks (" +
                           std::to_string(tokens) + " tokens from " + std::to_string(walk_.span.begins.front()) +
                           "): ";
        for (const auto& segment : segments_) {
            const DeviceGuard on(segment->device->device);
            CUDA_CHECK(cudaEventSynchronize(segment->walk_last.event));
            float ms = 0.0F;
            CUDA_CHECK(cudaEventElapsedTime(&ms, segment->walk_first.event, segment->walk_last.event));
            line += segment_name(*segment) + std::to_string(static_cast<int>(ms)) + " ms; ";
        }
        line += "wall " + std::to_string(static_cast<long long>(elapsed_ns(began) / 1000000)) + " ms";
        diagnostic(line, DiagnosticLevel::Debug);
    } catch (...) {}
}

void ProgramImpl::walk_abandon() noexcept {
    if (!walk_.span.active) { return; }
    try {
        synchronize();
    } catch (...) {}
    for (const auto& segment : segments_) {
        try {
            const DeviceGuard on(segment->device->device);
            segment->expert_stream->end();
            if (segment->remote_stream) {
                (void)cudaStreamSynchronize(segment->remote_stream->stream());
                segment->remote_stream->end();
            }
            for (cudaStream_t side : {segment->side.stream, segment->side.out}) {
                if (side != nullptr) { (void)cudaStreamSynchronize(side); }
            }
        } catch (...) {}
    }
    walk_.span     = WalkState::LayerWalk{};
    walk_.prefetch = WalkState::WalkPrefetch{};
}

} // namespace infernix::models::qwen4_exp::detail
