// Layer-split KV (multi-GPU WP10). Each segment of a layer-split Program owns the KV planes and the
// exact window of its own attention layers (and MTP's on the head segment) over the head pool's page
// indices, and its own block tables on its own device:
//  - for every split of Qwen3.8-Flash-Next into one, two or three segments and every KV storage, a
//    segment's page planes and window are the matching slices of the whole model's, and the
//    segments' planes map onto the whole model's exactly once, in the single-GPU order (GPU-free);
//  - a publication by page index is validated, copies the caller's indices into the row's pinned
//    shadow and fences them like publish();
//  - the head's table pool and a mirror segment's, on two streams of one GPU and then on two GPUs,
//    hold the same matrix as a Host oracle after random publications by index and by lease,
//    repeats, truncations, releases and lockstep backed-limit changes.

#include "core/arena.h"
#include "core/device.h"
#include "core/layout.h"
#include "core/paged_kv_cache.h"
#include "models/qwen4_exp/program/prefix/state_image.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <optional>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace infernix;
using namespace infernix::models::qwen4_exp;
using namespace infernix::models::qwen4_exp::prefix;

namespace {

[[noreturn]] void fail(const std::string& message) { throw std::runtime_error(message); }

void require(bool value, const std::string& message) {
    if (!value) { fail(message); }
}

template <class F>
void require_throws(F&& f, const std::string& message) {
    try {
        f();
    } catch (const std::exception&) { return; }
    throw std::runtime_error("expected an exception: " + message);
}

// Qwen3.8-Flash-Next's text geometry: 48 layers, three GDN layers then one attention layer.
TextConfig flash_next() {
    TextConfig c;
    c.hidden_size       = 2560;
    c.num_hidden_layers = 48;
    for (std::uint32_t l = 0; l < c.num_hidden_layers; ++l) {
        c.layer_types.push_back(l % 4 == 3 ? MixerKind::Attention : MixerKind::Gdn);
    }
    c.attention_layers   = 12;
    c.gdn_layers         = 36;
    c.attention.heads    = 16;
    c.attention.kv_heads = 2;
    c.attention.head_dim = 256;
    c.gdn.key_heads      = 16;
    c.gdn.key_head_dim   = 128;
    c.gdn.value_heads    = 48;
    c.gdn.value_head_dim = 128;
    c.gdn.conv_kernel    = 4;
    c.hc.streams         = 4;
    c.qsa.index_head_dim = 128;
    c.qsa.compress_ratio = 4;
    c.ple.layer          = 1;
    c.ple.conv_kernel    = 4;
    c.ple.conv_dilation  = 3;
    return c;
}

constexpr std::array kStorages = {KvCacheStorage::BFloat16,      KvCacheStorage::Int8Group64,
                                  KvCacheStorage::Fp8E4M3Row256, KvCacheStorage::Nvfp4Group16,
                                  KvCacheStorage::Fp8KeyNvfp4Value, KvCacheStorage::Vq2,
                                  KvCacheStorage::Q4KeyVq2Value};

// A segment's share of the attention layers: decoder layers [previous end, end), MTP on the last.
struct SegmentLayers {
    std::uint32_t attention_base = 0;
    std::uint32_t attention      = 0;
    bool mtp                     = false;

    [[nodiscard]] std::uint32_t kv_layers() const noexcept { return attention + (mtp ? 1U : 0U); }
};

std::vector<SegmentLayers> split(const TextConfig& c, std::span<const std::uint32_t> ends, bool mtp) {
    std::vector<SegmentLayers> out;
    std::uint32_t begin = 0, attention = 0;
    for (std::size_t s = 0; s < ends.size(); ++s) {
        SegmentLayers segment;
        segment.attention_base = attention;
        for (std::uint32_t l = begin; l < ends[s]; ++l) {
            segment.attention += c.layer_types[l] == MixerKind::Attention ? 1U : 0U;
        }
        segment.mtp = mtp && s + 1 == ends.size();
        attention += segment.attention;
        begin = ends[s];
        out.push_back(segment);
    }
    return out;
}

std::string describe(KvCacheStorage storage, bool mtp, std::span<const std::uint32_t> ends) {
    std::string out = "storage " + std::to_string(static_cast<int>(storage)) + (mtp ? " mtp" : " plain") + " split";
    for (const std::uint32_t end : ends) { out += " " + std::to_string(end); }
    return out;
}

void check_split(const TextConfig& c, KvCacheStorage storage, bool mtp, std::span<const std::uint32_t> ends,
                 const KvPageGeometry& full, const KvWindowGeometry& full_window, std::uint32_t lanes) {
    const std::string what = describe(storage, mtp, ends);
    const auto& planes     = full.geometry.planes;
    std::vector<std::uint32_t> covered(planes.size(), 0);
    std::size_t window_bytes = 0;
    for (const SegmentLayers& s : split(c, ends, mtp)) {
        const KvPageGeometry g        = kv_page_geometry(c, storage, s.attention, s.mtp);
        const std::uint32_t kv_layers = s.kv_layers();
        require(g.kv_layers == kv_layers && g.geometry.page_tokens == full.geometry.page_tokens &&
                    g.geometry.device_plane_order == full.geometry.device_plane_order &&
                    g.planes_per_layer == (kv_layers == 0 ? 0U : full.planes_per_layer) &&
                    g.geometry.planes.size() == static_cast<std::size_t>(kv_layers) * full.planes_per_layer,
                what + ": the segment's page group has its own KV layers of the model's planes");
        if (!g.geometry.planes.empty()) {
            // A contiguous slice: the head segment's attention layers end where MTP's planes begin.
            const std::size_t first = global_kv_plane(g, s.attention_base, c.attention_layers, 0, s.mtp);
            require(first + g.geometry.planes.size() <= planes.size() &&
                        std::equal(g.geometry.planes.begin(), g.geometry.planes.end(), planes.begin() + first),
                    what + ": the segment's planes are a slice of the whole model's");
        }
        for (std::size_t p = 0; p < g.geometry.planes.size(); ++p) {
            const std::size_t q = global_kv_plane(g, s.attention_base, c.attention_layers, p, s.mtp);
            if (q >= planes.size() || g.geometry.planes[p] != planes[q]) {
                fail(what + ": plane " + std::to_string(p));
            }
            ++covered[q];
        }
        for (std::uint32_t k = 0; k < kv_layers; ++k) {
            const std::uint32_t layer = k < s.attention ? s.attention_base + k : c.attention_layers;
            const PlaneRange local    = kv_layer_planes(g, k);
            const PlaneRange global   = kv_layer_planes(full, layer);
            if (global_kv_plane(g, s.attention_base, c.attention_layers, local.begin, s.mtp) != global.begin ||
                local.end - local.begin != global.end - global.begin) {
                fail(what + ": KV layer " + std::to_string(k) + " does not map onto its global KV layer");
            }
        }
        const KvWindowGeometry w = kv_window_geometry(c, storage, s.attention, s.mtp, lanes);
        require(w.present() == (full_window.present() && kv_layers > 0),
                what + ": a window exactly where KV layers are");
        if (!w.present()) { continue; }
        require(w.kv_layers == kv_layers && w.lanes == lanes && w.kv_heads == full_window.kv_heads &&
                    w.layer_bytes() == full_window.layer_bytes(),
                what + ": the segment's window has its own KV layers of the model's");
        for (std::uint32_t k = 0; k < kv_layers; ++k) {
            const std::uint32_t global = k < s.attention ? s.attention_base + k : c.attention_layers;
            for (int p = 0; p < KvWindowGeometry::kPlanes; ++p) {
                if (w.lane_bytes(p) != full_window.lane_bytes(p) ||
                    w.plane_offset(k, p) - w.plane_offset(k, 0) !=
                        full_window.plane_offset(global, p) - full_window.plane_offset(global, 0)) {
                    fail(what + ": window plane " + std::to_string(p) + " of KV layer " + std::to_string(k));
                }
            }
        }
        window_bytes += w.bytes();
    }
    require(std::all_of(covered.begin(), covered.end(), [](std::uint32_t n) { return n == 1; }),
            what + ": every plane of the whole model belongs to exactly one segment");
    require(window_bytes == (full_window.present() ? full_window.bytes() : 0U),
            what + ": the segments' windows add up to the whole model's");
}

void test_subset_geometry(const TextConfig& c) {
    constexpr std::uint32_t kLanes = 3;
    std::size_t splits             = 0;
    for (const KvCacheStorage storage : kStorages) {
        for (const bool mtp : {false, true}) {
            const KvPageGeometry full = kv_page_geometry(c, storage, mtp);
            const KvPageGeometry same = kv_page_geometry(c, storage, c.attention_layers, mtp);
            require(full.geometry == same.geometry && full.kv_layers == same.kv_layers &&
                        full.planes_per_layer == same.planes_per_layer,
                    "the whole-model page geometry is the subset of every attention layer");
            const KvWindowGeometry full_window = kv_window_geometry(c, storage, mtp, kLanes);
            const KvWindowGeometry same_window = kv_window_geometry(c, storage, c.attention_layers, mtp, kLanes);
            require(full_window.kv_heads == same_window.kv_heads && full_window.kv_layers == same_window.kv_layers &&
                        full_window.lanes == same_window.lanes,
                    "the whole-model window is the subset of every attention layer");
            const std::uint32_t layers = c.num_hidden_layers;
            const std::uint32_t whole[1] = {layers};
            check_split(c, storage, mtp, whole, full, full_window, kLanes);
            ++splits;
            for (std::uint32_t a = 1; a < layers; ++a) {
                const std::uint32_t two[2] = {a, layers};
                check_split(c, storage, mtp, two, full, full_window, kLanes);
                ++splits;
                for (std::uint32_t b = a + 1; b < layers; ++b) {
                    const std::uint32_t three[3] = {a, b, layers};
                    check_split(c, storage, mtp, three, full, full_window, kLanes);
                    ++splits;
                }
            }
        }
    }
    const KvCacheStorage storage = KvCacheStorage::Int8Group64;
    require_throws([&] { (void)kv_page_geometry(c, storage, c.attention_layers + 1U, false); },
                   "more attention layers than the model has");
    require_throws([&] { (void)kv_window_geometry(c, KvCacheStorage::Vq2, c.attention_layers + 1U, false, 1); },
                   "a window of more attention layers than the model has");
    const KvPageGeometry six = kv_page_geometry(c, storage, 6, false);
    require_throws([&] { (void)global_kv_plane(six, 7, c.attention_layers, 0, false); },
                   "a segment past the model's attention layers");
    require_throws([&] { (void)global_kv_plane(six, 0, c.attention_layers, six.geometry.planes.size(), false); },
                   "a plane past the segment's");
    const KvPageGeometry none = kv_page_geometry(c, storage, 0, false);
    require(none.kv_layers == 0 && none.geometry.planes.empty() &&
                !kv_window_geometry(c, KvCacheStorage::Vq2, 0, false, 1).present(),
            "a segment without attention layers or MTP has no KV");
    require_throws([&] { (void)global_kv_plane(none, 0, c.attention_layers, 0, false); },
                   "a segment without KV planes");
    require_throws([&] { (void)global_kv_plane(none, 0, c.attention_layers, 0, true); },
                   "an MTP segment without KV layers");
    require(global_kv_plane(kv_page_geometry(c, storage, 0, true), 0, c.attention_layers, 0, true) ==
                static_cast<std::size_t>(c.attention_layers) * six.planes_per_layer,
            "an MTP-only segment maps onto the MTP planes");
    std::printf("subset geometry: %zu splits over %zu storages, plain and MTP: planes and windows are slices of "
                "the whole model's, each global plane owned once\n",
                splits / (2 * kStorages.size()), kStorages.size());
}

void CUDART_CB hold_stream(void* milliseconds) {
    std::this_thread::sleep_for(std::chrono::milliseconds(*static_cast<const int*>(milliseconds)));
}

void hold(cudaStream_t stream, const int& milliseconds) {
    CUDA_CHECK(cudaLaunchHostFunc(stream, hold_stream, const_cast<int*>(&milliseconds)));
}

// One segment's KV: a page pool over its own planes (the head's page count), its tables and its
// stream, all on its own device. Only the head segment's pool hands out pages.
class Segment {
public:
    Segment(int device, const KVPageGeometry& geometry, std::uint32_t page_count, std::uint32_t logical,
            std::int32_t rows)
        : device_(device) {
        bind();
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        LayoutBuilder builder;
        const auto pool_layout =
            plan_device_kv_page_pool(builder, {.page_group_count = page_count, .geometry = geometry});
        const auto table_layout =
            plan_kv_execution_tables(builder, {.logical_page_capacity = logical, .table_rows = rows});
        backing_ = std::make_unique<DeviceBuffer>(builder.finish(256));
        backing_->fill(0); // the oracle starts from zero tables
        const DeviceSpan span{backing_->p, backing_->bytes};
        pages  = std::make_unique<DeviceKVPagePool>(span, pool_layout);
        tables = std::make_unique<KVExecutionTablePool>(span, table_layout, *pages);
    }

    ~Segment() {
        (void)cudaSetDevice(device_);
        tables.reset();
        pages.reset();
        backing_.reset();
        if (stream != nullptr) { (void)cudaStreamDestroy(stream); }
    }

    Segment(const Segment&)            = delete;
    Segment& operator=(const Segment&) = delete;

    void bind() const { CUDA_CHECK(cudaSetDevice(device_)); }

    // The whole block-table matrix once this segment's stream has drained.
    [[nodiscard]] std::vector<std::int32_t> matrix() const {
        bind();
        CUDA_CHECK(cudaStreamSynchronize(stream));
        const Tensor& m = tables->matrix();
        std::vector<std::int32_t> out(m.bytes() / sizeof(std::int32_t));
        CUDA_CHECK(cudaMemcpy(out.data(), m.data, m.bytes(), cudaMemcpyDeviceToHost));
        return out;
    }

    cudaStream_t stream = nullptr;
    std::unique_ptr<DeviceKVPagePool> pages;
    std::unique_ptr<KVExecutionTablePool> tables;

private:
    int device_ = 0;
    std::unique_ptr<DeviceBuffer> backing_;
};

std::vector<DeviceKVPageLease> materialize(DeviceKVPagePool& pool, std::uint32_t count) {
    std::optional<DeviceKVPageReservation> reservation = pool.reserve(count);
    require(reservation.has_value(), "page reservation");
    std::vector<DeviceKVPageLease> out;
    out.reserve(count); // materialize never reallocates its destination
    pool.materialize(*reservation, count, out);
    return out;
}

std::vector<DeviceKVPageHandle> handles_of(std::span<const DeviceKVPageLease> pages) {
    std::vector<DeviceKVPageHandle> out;
    out.reserve(pages.size());
    for (const DeviceKVPageLease& page : pages) { out.push_back(page.handle()); }
    return out;
}

void test_index_publication(int device) {
    static constexpr int kHoldMs = 300;
    Segment segment(device, KVPageGeometry{.planes = {{DType::I8, 8, 2, 256}}}, 4, 4, 1);
    KVExecutionTablePool& tables = *segment.tables;
    const cudaStream_t stream    = segment.stream;

    // Validation, before anything is written.
    KVExecutionRowLease row        = tables.acquire(0);
    const std::int32_t negative[1] = {-1}, two[2] = {0, 1}, third[1] = {2};
    require_throws([&] { tables.publish_indices(row.handle(), 0, negative, stream); }, "a negative page index");
    segment.pages->set_backed_pages(2);
    require_throws([&] { tables.publish_indices(row.handle(), 0, third, stream); }, "an index past the backed pages");
    require_throws([&] { tables.publish_repeated_index(row.handle(), 2, 1, stream); },
                   "a repeated index past the backed pages");
    segment.pages->set_backed_pages(4);
    require_throws([&] { tables.publish_indices(row.handle(), 3, two, stream); }, "a publication past the row");
    require_throws([&] { tables.publish_repeated_index(row.handle(), 0, 5, stream); }, "a repeat past the row");
    const KVExecutionRowHandle stale = row.handle();
    row.release();
    require_throws([&] { tables.publish_indices(stale, 0, two, stream); }, "a released row");

    // indices_of reads each page's own pool and refuses empty or released pages.
    {
        std::vector<DeviceKVPageLease> leases = materialize(*segment.pages, 3);
        require(KVExecutionTablePool::indices_of(leases) == std::vector<std::int32_t>({0, 1, 2}), "lease indices");
        const std::vector<DeviceKVPageHandle> handles = handles_of(leases);
        require(KVExecutionTablePool::indices_of(handles) == std::vector<std::int32_t>({0, 1, 2}), "handle indices");
        leases[1].release();
        require_throws([&] { (void)KVExecutionTablePool::indices_of(handles); }, "a released page's handle");
        require_throws([&] { (void)KVExecutionTablePool::indices_of(std::span<const DeviceKVPageLease>(leases)); },
                       "an empty lease");
        const DeviceKVPageHandle none[1] = {};
        require_throws([&] { (void)KVExecutionTablePool::indices_of(none); }, "an empty handle");
    }

    // The caller's indices die before the stream reaches the copy; the next owner of the released
    // row rewrites the same entries, so it waits for that copy, and work queued behind the first
    // publication sees its mapping.
    DeviceBuffer observed(2 * sizeof(std::int32_t));
    KVExecutionRowLease owner = tables.acquire(0);
    const Tensor device_row   = tables.row(owner.handle());
    hold(stream, kHoldMs);
    {
        const std::vector<std::int32_t> first{0, 1};
        tables.publish_indices(owner.handle(), 0, first, stream);
    }
    CUDA_CHECK(cudaMemcpyAsync(observed.p, device_row.data, 2 * sizeof(std::int32_t), cudaMemcpyDeviceToDevice,
                               stream));
    owner.release();
    KVExecutionRowLease next = tables.acquire(0);
    tables.publish_indices(next.handle(), 0, std::vector<std::int32_t>{2, 0}, stream);
    std::vector<std::int32_t> seen(2);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    observed.copy_to_host(seen.data(), seen.size() * sizeof(std::int32_t));
    require(seen == std::vector<std::int32_t>({0, 1}), "queued work saw the next owner's mapping instead of its own");
    require(segment.matrix() == std::vector<std::int32_t>({2, 0, 0, 0}), "the next owner's publication landed");

    // A disjoint extension does not wait for a queued copy of other entries.
    hold(stream, kHoldMs);
    tables.publish_indices(next.handle(), 0, std::vector<std::int32_t>{3, 1}, stream);
    const auto started = std::chrono::steady_clock::now();
    tables.publish_indices(next.handle(), 2, std::vector<std::int32_t>{2}, stream);
    const auto waited = std::chrono::steady_clock::now() - started;
    require(waited < std::chrono::milliseconds(100), "a disjoint publication waited for an unrelated queued copy");
    require(segment.matrix() == std::vector<std::int32_t>({3, 1, 2, 0}), "the disjoint extension landed");
    tables.publish_repeated_index(next.handle(), 1, 3, stream);
    require(segment.matrix() == std::vector<std::int32_t>({1, 1, 1, 0}), "a repeated index landed");
    std::printf("index publication on device %d: validated, copied into the pinned shadow, fenced like publish()\n",
                device);
}

// The head segment hands out pages; the mirror segment publishes the same indices to its own
// tables on its own stream and follows the head's backed limit without ever allocating.
void test_lockstep_tables(const TextConfig& c, int mirror_device, int head_device) {
    static constexpr std::uint32_t kPages = 40, kLogical = 12;
    static constexpr std::int32_t kRows   = 4;
    static constexpr int kSteps           = 1200;
    static constexpr int kStallMs[3]      = {1, 3, 6};
    const std::uint32_t ends[2]    = {24, c.num_hidden_layers};
    const std::vector<SegmentLayers> layers = split(c, ends, true);
    const KvCacheStorage storage            = KvCacheStorage::Int8Group64;
    Segment mirror(mirror_device, kv_page_geometry(c, storage, layers[0].attention, false).geometry, kPages, kLogical,
                   kRows);
    Segment head(head_device, kv_page_geometry(c, storage, layers[1].attention, true).geometry, kPages, kLogical,
                 kRows);

    struct Lane {
        KVExecutionRowLease head_row;
        KVExecutionRowLease mirror_row;
        std::vector<DeviceKVPageLease> pages; // the head pool's
    };
    std::vector<std::int32_t> oracle(static_cast<std::size_t>(kRows) * kLogical, 0);
    std::vector<Lane> lanes(kRows);
    std::mt19937 rng(0x5EC7105U);
    const auto below = [&](std::uint32_t n) { return static_cast<std::uint32_t>(rng() % n); };
    std::size_t counts[8] = {};

    const auto check = [&](const std::string& when) {
        require(head.matrix() == oracle, "the head's tables match the oracle " + when);
        require(mirror.matrix() == oracle, "the mirror's tables match the oracle " + when);
    };
    // The same indices to both segments' tables, each on its own stream; the head sometimes
    // publishes its leases directly, which must agree with indices_of.
    const auto publish = [&](std::int32_t r, std::uint32_t begin, std::span<const DeviceKVPageLease> pages) {
        Lane& lane                             = lanes[static_cast<std::size_t>(r)];
        const bool by_lease                    = below(2) == 0;
        const std::vector<std::int32_t> indices = below(2) == 0 ? KVExecutionTablePool::indices_of(handles_of(pages))
                                                                : KVExecutionTablePool::indices_of(pages);
        head.bind();
        if (by_lease) {
            head.tables->publish(lane.head_row.handle(), begin, pages, head.stream);
        } else {
            head.tables->publish_indices(lane.head_row.handle(), begin, indices, head.stream);
        }
        mirror.bind();
        mirror.tables->publish_indices(lane.mirror_row.handle(), begin, indices, mirror.stream);
        std::copy(indices.begin(), indices.end(), oracle.begin() + static_cast<std::ptrdiff_t>(r * kLogical + begin));
    };
    const auto grow = [&](Lane& lane, std::uint32_t count) {
        if (count == 0 || head.pages->available_pages() < count) { return false; }
        std::vector<DeviceKVPageLease> fresh = materialize(*head.pages, count);
        lane.pages.reserve(lane.pages.size() + count);
        for (DeviceKVPageLease& page : fresh) { lane.pages.push_back(std::move(page)); }
        return true;
    };

    for (int step = 0; step < kSteps; ++step) {
        const auto r     = static_cast<std::int32_t>(below(kRows));
        Lane& lane       = lanes[static_cast<std::size_t>(r)];
        const bool bound = lane.head_row.valid();
        const std::uint32_t op = below(8);
        switch (op) {
        case 0:
        case 1: { // admit, or extend an admitted lane
            if (!bound) {
                lane.head_row   = head.tables->acquire(r);
                lane.mirror_row = mirror.tables->acquire(r);
            }
            const auto have = static_cast<std::uint32_t>(lane.pages.size());
            if (have == kLogical || !grow(lane, 1 + below(kLogical - have))) { break; }
            publish(r, have, std::span<const DeviceKVPageLease>(lane.pages).subspan(have));
            ++counts[op];
            break;
        }
        case 2: // republish the whole row
            if (!bound || lane.pages.empty()) { break; }
            publish(r, 0, lane.pages);
            ++counts[op];
            break;
        case 3: { // repeat one page over a prefix of the row
            if (!bound || lane.pages.empty()) { break; }
            const DeviceKVPageLease& page = lane.pages[below(static_cast<std::uint32_t>(lane.pages.size()))];
            const std::uint32_t count     = 1 + below(kLogical);
            const std::int32_t index      = KVExecutionTablePool::indices_of(std::span(&page, 1)).front();
            head.bind();
            if (below(2) == 0) {
                head.tables->publish_repeated(lane.head_row.handle(), page.handle(), count, head.stream);
            } else {
                head.tables->publish_repeated_index(lane.head_row.handle(), index, count, head.stream);
            }
            mirror.bind();
            mirror.tables->publish_repeated_index(lane.mirror_row.handle(), index, count, mirror.stream);
            std::fill_n(oracle.begin() + static_cast<std::ptrdiff_t>(r * kLogical), count, index);
            ++counts[op];
            break;
        }
        case 4: // truncate: the dropped pages go back to the head pool; the row keeps stale entries
            if (lane.pages.empty()) { break; }
            lane.pages.erase(lane.pages.begin() + below(static_cast<std::uint32_t>(lane.pages.size())),
                             lane.pages.end());
            ++counts[op];
            break;
        case 5: // release the row in lockstep, with its pages
            if (!bound) { break; }
            lane.pages.clear();
            lane.head_row.release();
            lane.mirror_row.release();
            ++counts[op];
            break;
        case 6: { // move the backed limit in lockstep; the head decides, the mirror always can
            const std::uint32_t target = below(kPages + 1);
            if (!head.pages->can_back(target)) { break; }
            require(mirror.pages->can_back(target), "a mirror that never allocates follows any backed limit");
            head.pages->set_backed_pages(target);
            mirror.pages->set_backed_pages(target);
            require(head.pages->backed_pages() == mirror.pages->backed_pages() &&
                        mirror.pages->available_pages() == target && mirror.pages->allocated_pages() == 0,
                    "the backed limits move in lockstep");
            if (bound && target < kPages) {
                const std::int32_t past[1] = {static_cast<std::int32_t>(target)};
                require_throws(
                    [&] { mirror.tables->publish_indices(lane.mirror_row.handle(), 0, past, mirror.stream); },
                    "a mirror publication past the backed pages");
            }
            ++counts[op];
            break;
        }
        default: { // stall one stream so the two drift apart
            const Segment& stalled = below(2) == 0 ? head : mirror;
            stalled.bind();
            hold(stalled.stream, kStallMs[below(3)]);
            ++counts[op];
            break;
        }
        }
        if (step % 100 == 99) { check("after step " + std::to_string(step + 1)); }
    }
    check("at the end");

    if (mirror_device != head_device) {
        // A fence never spans two devices: publishing on the other device's stream is refused
        // before anything is written.
        Lane& lane = lanes[0];
        if (!lane.mirror_row.valid()) {
            lane.head_row   = head.tables->acquire(0);
            lane.mirror_row = mirror.tables->acquire(0);
        }
        require_throws([&] { mirror.tables->publish_indices(lane.mirror_row.handle(), 0, {}, head.stream); },
                       "a publication on another device's stream");
        check("after a refused cross-device publication");
    }
    std::printf("lockstep tables (mirror on device %d, head on device %d): %d steps (admit/extend %zu, republish %zu, "
                "repeat %zu, truncate %zu, release %zu, backed %zu, stall %zu) match the oracle\n",
                mirror_device, head_device, kSteps, counts[0] + counts[1], counts[2], counts[3], counts[4], counts[5],
                counts[6], counts[7]);
}

} // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    try {
        const TextConfig c = flash_next();
        test_subset_geometry(c);
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
            std::printf("SKIP: no usable CUDA device\n");
            return 77;
        }
        test_index_publication(0);
        test_lockstep_tables(c, 0, 0);
        if (devices >= 2) {
            test_lockstep_tables(c, 0, 1);
        } else {
            std::printf("two-GPU lockstep tables skipped: one CUDA device\n");
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
