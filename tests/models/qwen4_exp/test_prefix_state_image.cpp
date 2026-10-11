// Qwen4Exp prefix-cache state images and page records (design §19.3.1, test X8), on synthetic Device
// pools of the real Qwen3.8-Flash-Next geometry with no model: random lane state is captured to
// packed Device slots and to Host images split over shuffled slabs, restored into other lanes group
// by group, and must come back byte for byte without touching any other lane. A KV page moved to a
// Host record and back into a different page per KV layer must also be byte-exact.

#include "core/arena.h"
#include "core/device.h"
#include "core/host_kv_arena.h"
#include "core/layout.h"
#include "core/linear_attention_state.h"
#include "core/paged_kv_cache.h"
#include "models/qwen4_exp/program/prefix/state_image.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace infernix;
using namespace infernix::models::qwen4_exp;
using namespace infernix::models::qwen4_exp::prefix;

namespace {

void require(bool value, const std::string& message) {
    if (!value) { throw std::runtime_error(message); }
}

template <class F>
void require_throws(F&& f, const std::string& message) {
    try {
        f();
    } catch (const std::exception&) { return; }
    throw std::runtime_error("expected an exception: " + message);
}

bool cuda_unavailable() {
    int n               = 0;
    const cudaError_t e = cudaGetDeviceCount(&n);
    return e != cudaSuccess || n == 0;
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

void fill_random(void* device, std::size_t bytes, std::uint32_t seed) {
    std::mt19937_64 rng(seed);
    std::vector<std::uint64_t> host((bytes + 7) / 8);
    for (auto& v : host) { v = rng(); }
    CUDA_CHECK(cudaMemcpy(device, host.data(), bytes, cudaMemcpyHostToDevice));
}

// Every non-header part of a lane, concatenated.
std::vector<std::byte> lane_bytes(const LaneStateImage& image, std::uint32_t lane) {
    std::vector<std::byte> out;
    for (std::size_t p = 1; p < image.layout().parts.size(); ++p) {
        const StateImagePart& part = image.layout().parts[p];
        const std::size_t at       = out.size();
        out.resize(at + part.bytes);
        CUDA_CHECK(cudaMemcpy(out.data() + at, image.device_part(part, lane), part.bytes, cudaMemcpyDeviceToHost));
    }
    return out;
}

// Pinned slabs handed out in a shuffled order, as the prefix cache's pool does.
struct HostSlabs {
    HostSlabs(std::size_t slabs, std::size_t slab_bytes, std::size_t used, std::uint32_t seed)
        : pool(slabs * slab_bytes), bytes(slab_bytes) {
        std::vector<std::size_t> order(slabs);
        std::iota(order.begin(), order.end(), 0);
        std::shuffle(order.begin(), order.end(), std::mt19937(seed));
        for (std::size_t i = 0; i < used; ++i) {
            segments.push_back(static_cast<std::byte*>(pool.data()) + order[i] * slab_bytes);
        }
        const_segments.assign(segments.begin(), segments.end());
    }

    HostImage image() const { return HostImage{segments, bytes}; }
    HostImageConst const_image() const { return HostImageConst{const_segments, bytes}; }

    PinnedHostBuffer pool;
    std::size_t bytes = 0;
    std::vector<std::byte*> segments;
    std::vector<const std::byte*> const_segments;
};

void test_layout(const TextConfig& c) {
    const StateImageLayout layout = plan_state_image(state_image_spec(c, true, KvCacheStorage::Int8Group64));
    // Design §19.3.1: 115,673,088 bytes per image with MTP (124 slabs of a 931,840-byte block).
    require(layout.image_bytes == 115'673'088ULL, "image bytes " + std::to_string(layout.image_bytes));
    require(layout.groups() == c.num_hidden_layers + 1U, "one group per decoder layer plus MTP");
    require(layout.parts.front().kind == StateImagePart::Kind::Header && layout.parts.front().offset == 0,
            "the header leads the image");
    std::size_t end = 0;
    for (const StateImagePart& part : layout.parts) {
        require(part.offset % kStateImageAlignment == 0 && part.offset >= end, "parts are aligned and ordered");
        end = part.offset + part.bytes;
    }
    require(end <= layout.image_bytes, "parts fit the image");
    std::uint32_t gdn = 0, attention = 0;
    for (std::uint32_t g = 0; g < c.num_hidden_layers; ++g) {
        std::uint32_t p = layout.group_begin[g];
        if (g == c.ple.layer) {
            require(layout.parts[p].kind == StateImagePart::Kind::Ple, "the PLE history leads its layer's group");
            ++p;
        }
        if (c.layer_types[g] == MixerKind::Gdn) {
            require(layout.group_begin[g + 1] - p == 2 && layout.parts[p].kind == StateImagePart::Kind::GdnRecurrent &&
                        layout.parts[p + 1].kind == StateImagePart::Kind::GdnConv && layout.parts[p].index == gdn &&
                        layout.parts[p + 1].index == gdn,
                    "a GDN layer's group is its recurrent then conv state");
            ++gdn;
        } else {
            require(layout.group_begin[g + 1] - p == 1 && layout.parts[p].kind == StateImagePart::Kind::QsaTail &&
                        layout.parts[p].index == attention,
                    "an attention layer's group is its QSA tail");
            ++attention;
        }
    }
    const std::uint32_t m = layout.group_begin[layout.mtp_group()];
    require(layout.group_begin[layout.mtp_group() + 1] - m == 2 && layout.parts[m].kind == StateImagePart::Kind::MtpSaved &&
                layout.parts[m + 1].kind == StateImagePart::Kind::MtpTail,
            "the MTP group is the saved residual then the MTP tail");
    const StateImageLayout plain = plan_state_image(state_image_spec(c, false, KvCacheStorage::Int8Group64));
    require(plain.group_begin[plain.mtp_group()] == plain.group_begin[plain.mtp_group() + 1] &&
                plain.fingerprint != layout.fingerprint && plain.image_bytes < layout.image_bytes,
            "without MTP the MTP group is empty and the layout differs");
}

void test_round_trips(const TextConfig& c) {
    constexpr std::uint32_t kLanes = 3;
    const StateImageLayout layout  = plan_state_image(state_image_spec(c, true, KvCacheStorage::Int8Group64));
    const StateImageSpec& spec     = layout.spec;

    LayoutBuilder builder;
    const LinearAttentionStatePoolSpec gdn_spec{.layers         = c.gdn_layers,
                                                .conv_channels  = static_cast<std::int32_t>(c.gdn.conv_channels()),
                                                .conv_width     = static_cast<std::int32_t>(c.gdn.conv_kernel - 1),
                                                .value_heads    = static_cast<std::int32_t>(c.gdn.value_heads),
                                                .value_head_dim = static_cast<std::int32_t>(c.gdn.value_head_dim),
                                                .key_head_dim   = static_cast<std::int32_t>(c.gdn.key_head_dim),
                                                .slot_count     = static_cast<std::int32_t>(kLanes)};
    const auto gdn_layout = plan_linear_attention_state_pool(builder, gdn_spec);
    DeviceBuffer state(builder.finish(256));
    LinearAttentionStatePool gdn(DeviceSpan{state.p, state.bytes}, gdn_layout);
    DeviceBuffer ple(spec.ple_bytes * kLanes);
    DeviceBuffer tails(spec.qsa_tail_bytes * kLanes * (c.attention_layers + 1U));
    DeviceBuffer saved(spec.mtp_saved_bytes * kLanes);
    const auto randomize = [&](std::uint32_t seed) {
        fill_random(state.p, state.bytes, seed);
        fill_random(ple.p, ple.bytes, seed + 1);
        fill_random(tails.p, tails.bytes, seed + 2);
        fill_random(saved.p, saved.bytes, seed + 3);
    };
    randomize(100);
    const LaneStateBuffers whole{.layer_begin = 0,
                                 .layer_end   = c.num_hidden_layers,
                                 .head        = true,
                                 .gdn         = &gdn,
                                 .lanes       = kLanes,
                                 .ple         = static_cast<std::byte*>(ple.p),
                                 .tails       = static_cast<std::byte*>(tails.p),
                                 .mtp_saved   = static_cast<std::byte*>(saved.p)};
    const LaneStateImage image(layout, std::span<const LaneStateBuffers>(&whole, 1));
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    // Lane 0 to a Host image over shuffled slabs, then into lane 1 group by group, last group first.
    constexpr std::size_t kSlab = 933'888; // one 931,840-byte block bundle, 4 KiB aligned
    const std::size_t used      = (layout.image_bytes + kSlab - 1) / kSlab;
    const auto want             = lane_bytes(image, 0);
    const auto lane2            = lane_bytes(image, 2);
    HostSlabs a(used + 9, kSlab, used, 7);
    image.copy_lane_to_host(0, a.image(), stream);
    StateImageHeader header;
    header.frontier     = 4242;
    header.mtp_written  = true;
    header.mtp_next     = 151643;
    header.lineage_echo = true;
    header.opener       = true;
    header.mtp_accept   = {0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2};
    write_state_image_header(layout, header, a.segments[0]);
    for (std::uint32_t g = layout.groups(); g-- > 0;) { image.copy_group_from_host(g, a.const_image(), 1, stream); }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    require(lane_bytes(image, 1) == want, "a Host image restores the lane byte for byte");
    require(lane_bytes(image, 2) == lane2, "a restore touches no other lane");
    const auto read = read_state_image_header(layout, a.segments[0]);
    require(read && read->frontier == header.frontier && read->mtp_written && read->mtp_next == header.mtp_next &&
                read->lineage_echo && read->opener && read->mtp_accept == header.mtp_accept,
            "the header round-trips");

    // Lane 0 to a packed Device slot, written through to a second Host image, restored into lane 2.
    DeviceBuffer packed(layout.image_bytes);
    packed.fill(0);
    image.copy_lane_to_packed(0, static_cast<std::byte*>(packed.p), stream);
    HostSlabs b(used + 3, kSlab, used, 11);
    image.copy_packed_to_host(static_cast<const std::byte*>(packed.p), b.image(), stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    require(!read_state_image_header(layout, b.segments[0]), "a packed slot carries no header");
    for (std::uint32_t g = 0; g < layout.groups(); ++g) { image.copy_group_from_host(g, b.const_image(), 2, stream); }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    require(lane_bytes(image, 2) == want, "slot write-through restores byte for byte");

    // A packed slot straight into a lane whose state changed meanwhile.
    fill_random(ple.p, ple.bytes, 999);
    fill_random(saved.p, saved.bytes, 998);
    const auto lane0_now = lane_bytes(image, 0);
    for (std::uint32_t g = 0; g < layout.groups(); ++g) {
        image.copy_group_from_packed(g, static_cast<const std::byte*>(packed.p), 1, stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    require(lane_bytes(image, 1) == want, "a packed slot restores byte for byte");
    require(lane_bytes(image, 0) == lane0_now, "the source lane is untouched");

    // Header validation and argument checks.
    write_state_image_header(layout, header, b.segments[0]);
    require(read_state_image_header(layout, b.segments[0]).has_value(), "a written header is read back");
    b.segments[0][8] ^= std::byte{1};
    require(!read_state_image_header(layout, b.segments[0]), "a damaged header is rejected");
    require(!read_state_image_header(plan_state_image(state_image_spec(c, false, KvCacheStorage::Int8Group64)), a.segments[0]),
            "a header of another layout is rejected");
    const HostSlabs short_image(used - 1, kSlab, used - 1, 3);
    require_throws([&] { image.copy_lane_to_host(0, short_image.image(), stream); }, "too few segments");
    require_throws([&] { (void)image.device_part(layout.parts[1], kLanes); }, "lane out of range");
    require_throws([&] { image.copy_group_from_host(layout.groups(), a.const_image(), 0, stream); }, "group out of range");
    CUDA_CHECK(cudaStreamDestroy(stream));
    std::printf("state image: %zu bytes in %zu parts, %u groups, %zu slabs; Host, packed and write-through "
                "round trips byte-exact\n",
                layout.image_bytes, layout.parts.size(), layout.groups(), used);
}

// One segment's lane state for decoder layers [begin, end) (a layer split's; the head holds MTP).
struct SegmentState {
    SegmentState(const TextConfig& c, const StateImageSpec& spec, std::uint32_t begin, std::uint32_t end,
                 std::uint32_t lanes) {
        for (std::uint32_t l = 0; l < begin; ++l) {
            ++(c.layer_types[l] == MixerKind::Gdn ? buffers.gdn_base : buffers.attention_base);
        }
        std::uint32_t gdn_layers = 0, attention_layers = 0;
        for (std::uint32_t l = begin; l < end; ++l) { ++(c.layer_types[l] == MixerKind::Gdn ? gdn_layers : attention_layers); }
        const bool head     = end == c.num_hidden_layers;
        buffers.layer_begin = begin;
        buffers.layer_end   = end;
        buffers.head        = head;
        buffers.lanes       = lanes;
        if (gdn_layers > 0) {
            LayoutBuilder builder;
            const auto gdn_layout = plan_linear_attention_state_pool(
                builder, {.layers         = gdn_layers,
                          .conv_channels  = static_cast<std::int32_t>(c.gdn.conv_channels()),
                          .conv_width     = static_cast<std::int32_t>(c.gdn.conv_kernel - 1),
                          .value_heads    = static_cast<std::int32_t>(c.gdn.value_heads),
                          .value_head_dim = static_cast<std::int32_t>(c.gdn.value_head_dim),
                          .key_head_dim   = static_cast<std::int32_t>(c.gdn.key_head_dim),
                          .slot_count     = static_cast<std::int32_t>(lanes)});
            state       = DeviceBuffer(builder.finish(256));
            gdn         = std::make_unique<LinearAttentionStatePool>(DeviceSpan{state.p, state.bytes}, gdn_layout);
            buffers.gdn = gdn.get();
        }
        if (c.ple.layer >= begin && c.ple.layer < end) {
            ple         = DeviceBuffer(spec.ple_bytes * lanes);
            buffers.ple = static_cast<std::byte*>(ple.p);
        }
        if (const std::uint32_t kv = attention_layers + (head && spec.mtp ? 1U : 0U); kv > 0) {
            tails         = DeviceBuffer(spec.qsa_tail_bytes * lanes * kv);
            buffers.tails = static_cast<std::byte*>(tails.p);
        }
        if (head && spec.mtp) {
            saved             = DeviceBuffer(spec.mtp_saved_bytes * lanes);
            buffers.mtp_saved = static_cast<std::byte*>(saved.p);
        }
    }

    DeviceBuffer state, ple, tails, saved;
    std::unique_ptr<LinearAttentionStatePool> gdn;
    LaneStateBuffers buffers;
};

// A layer split's lane image (segments [0, K) and [K, 48), MTP on the second) is the unsplit image
// byte for byte: an unsplit lane's Host image restored group by group into the split lane, and
// captured back from it, gives the same bytes, and each group lies in the segment that runs it.
void test_split_image(const TextConfig& c, std::uint32_t K) {
    constexpr std::uint32_t kLanes = 2;
    const StateImageLayout layout  = plan_state_image(state_image_spec(c, true, KvCacheStorage::Int8Group64));
    SegmentState all(c, layout.spec, 0, c.num_hidden_layers, kLanes);
    SegmentState first(c, layout.spec, 0, K, kLanes);
    SegmentState second(c, layout.spec, K, c.num_hidden_layers, kLanes);
    std::uint32_t seed = 40 + K;
    for (SegmentState* segment : {&all, &first, &second}) {
        for (DeviceBuffer* buffer : {&segment->state, &segment->ple, &segment->tails, &segment->saved}) {
            if (buffer->p != nullptr) { fill_random(buffer->p, buffer->bytes, ++seed); }
        }
    }
    const LaneStateImage unsplit(layout, std::span<const LaneStateBuffers>(&all.buffers, 1));
    const LaneStateBuffers parts[] = {first.buffers, second.buffers};
    const LaneStateImage split(layout, parts);
    require(split.segments() == 2 && split.group_segment(K - 1) == 0 && split.group_segment(K) == 1 &&
                split.group_segment(layout.mtp_group()) == 1,
            "groups belong to the segment that runs their layer, the MTP group to the head");
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    constexpr std::size_t kSlab = 933'888;
    const std::size_t used      = (layout.image_bytes + kSlab - 1) / kSlab;
    HostSlabs a(used + 4, kSlab, used, 3 + K);
    HostSlabs b(used + 4, kSlab, used, 5 + K);
    unsplit.copy_lane_to_host(1, a.image(), stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const auto other = lane_bytes(split, 0);
    for (std::uint32_t g = 0; g < layout.groups(); ++g) { split.copy_group_from_host(g, a.const_image(), 1, stream); }
    for (std::uint32_t g = 0; g < layout.groups(); ++g) { split.copy_group_to_host(g, 1, b.image(), stream); }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    require(lane_bytes(split, 1) == lane_bytes(unsplit, 1), "the split lane holds the unsplit lane's state");
    require(lane_bytes(split, 0) == other, "a split restore touches no other lane");
    bool same = true;
    for (std::size_t p = 1; p < layout.parts.size(); ++p) {
        const StateImagePart& part = layout.parts[p];
        for (std::size_t o = part.offset; same && o < part.offset + part.bytes; ++o) {
            same = a.segments[o / kSlab][o % kSlab] == b.segments[o / kSlab][o % kSlab];
        }
    }
    require(same, "the split lane's Host image is the unsplit one byte for byte");
    // Each part lies in the buffers of the segment that runs its group.
    for (std::uint32_t g = 0; g < layout.groups(); ++g) {
        const SegmentState& owner = split.group_segment(g) == 0 ? first : second;
        for (std::uint32_t p = layout.group_begin[g]; p < layout.group_begin[g + 1]; ++p) {
            const std::byte* at = split.device_part(layout.parts[p], 1);
            bool inside         = false;
            for (const DeviceBuffer* buffer : {&owner.state, &owner.ple, &owner.tails, &owner.saved}) {
                const auto* base = static_cast<const std::byte*>(buffer->p);
                inside = inside || (base != nullptr && at >= base && at + layout.parts[p].bytes <= base + buffer->bytes);
            }
            require(inside, "group " + std::to_string(g) + "'s part " + std::to_string(p) + " is its segment's");
        }
    }
    // Segments that do not cover the layers in order are refused.
    require_throws([&] { (void)LaneStateImage(layout, std::span<const LaneStateBuffers>(&first.buffers, 1)); },
                   "a split that misses layers");
    const LaneStateBuffers swapped[] = {second.buffers, first.buffers};
    require_throws([&] { (void)LaneStateImage(layout, swapped); }, "segments out of order");
    CUDA_CHECK(cudaStreamDestroy(stream));
    std::printf("state image split at K = %u: Host round trips byte-exact against the unsplit lane\n", K);
}

// A layer split's KV pools (one page count, each over its own KV layers) move pages of the whole
// model's record layout by page index: an unsplit page's record restored into the segments' pages one
// KV layer at a time, and copied back out, is the same record byte for byte.
void test_split_page_records(const TextConfig& c, std::uint32_t K) {
    constexpr std::uint32_t kPages = 6;
    const KvCacheStorage storage   = KvCacheStorage::Int8Group64;
    const KvPageGeometry model     = kv_page_geometry(c, storage, true);
    const HostKVPageLayout host    = plan_host_kv_page_layout(model.geometry);
    std::uint32_t attention        = 0;
    for (std::uint32_t l = 0; l < K; ++l) { attention += c.layer_types[l] == MixerKind::Attention ? 1U : 0U; }
    struct Pool {
        KvPageGeometry pages;
        DeviceBuffer backing;
        std::unique_ptr<DeviceKVPagePool> pool;
        HostKVPageLayout host;
    };
    const auto make = [&](std::uint32_t attention_layers, bool mtp, std::uint32_t base, std::uint32_t seed) {
        Pool p;
        p.pages = kv_page_geometry(c, storage, attention_layers, mtp);
        LayoutBuilder builder;
        const auto layout = plan_device_kv_page_pool(builder, {.page_group_count = kPages, .geometry = p.pages.geometry});
        p.backing = DeviceBuffer(builder.finish(256));
        fill_random(p.backing.p, p.backing.bytes, seed);
        p.pool = std::make_unique<DeviceKVPagePool>(DeviceSpan{p.backing.p, p.backing.bytes}, layout);
        p.host = segment_host_page_layout(host, p.pages, base, c.attention_layers, mtp);
        return p;
    };
    const Pool whole = make(c.attention_layers, true, 0, 21);
    require(whole.host == host, "the unsplit pool's record layout is the model's");
    std::vector<Pool> segments;
    if (attention > 0) { segments.push_back(make(attention, false, 0, 22)); }
    segments.push_back(make(c.attention_layers - attention, true, attention, 23));
    PinnedHostBuffer records(2 * host.page_stride);
    std::memset(records.data(), 0, 2 * host.page_stride);
    std::byte* const rec[2] = {static_cast<std::byte*>(records.data()),
                               static_cast<std::byte*>(records.data()) + host.page_stride};
    const std::int32_t source[1] = {4};
    const std::int32_t target[1] = {1};
    whole.pool->copy_indices_to_host_records(source, std::span(&rec[0], 1), {}, whole.host, 0,
                                             whole.pages.geometry.planes.size(), nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    const std::byte* const from[1] = {rec[0]};
    for (const Pool& segment : segments) {
        // One KV layer at a time, the last first, as a restore lands them.
        for (std::uint32_t k = segment.pages.kv_layers; k-- > 0;) {
            const PlaneRange planes = kv_layer_planes(segment.pages, k);
            segment.pool->copy_indices_from_host_records(from, {}, target, segment.host, planes.begin, planes.end,
                                                         nullptr);
        }
        segment.pool->copy_indices_to_host_records(target, std::span(&rec[1], 1), {}, segment.host, 0,
                                                   segment.pages.geometry.planes.size(), nullptr);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    require(std::memcmp(rec[0], rec[1], host.page_stride) == 0,
            "the segments' planes of a page make the unsplit record (K = " + std::to_string(K) + ")");
    require_throws([&] { segments.back().pool->copy_page_index(0, static_cast<std::int32_t>(kPages), nullptr); },
                   "a page index past the backed pages");
    const std::int32_t twice[2]    = {2, 2};
    const std::byte* const both[2] = {rec[0], rec[1]};
    require_throws(
        [&] { segments.back().pool->copy_indices_from_host_records(both, {}, twice, segments.back().host, 0, 1, nullptr); },
        "duplicate destination pages");
    std::printf("page records split at K = %u (%zu segment(s) with KV): byte-exact against the unsplit record\n", K,
                segments.size());
}

void test_page_records(const TextConfig& c, KvCacheStorage storage, std::size_t expected_stride) {
    const KvPageGeometry geometry = kv_page_geometry(c, storage, true);
    require(geometry.kv_layers == c.attention_layers + 1U &&
                geometry.geometry.planes.size() == static_cast<std::size_t>(geometry.kv_layers) * geometry.planes_per_layer,
            "every KV layer has the same planes");
    LayoutBuilder builder;
    const auto pool_layout = plan_device_kv_page_pool(builder, {.page_group_count = 8, .geometry = geometry.geometry});
    DeviceBuffer backing(builder.finish(256));
    fill_random(backing.p, backing.bytes, 5);
    DeviceKVPagePool pool(DeviceSpan{backing.p, backing.bytes}, pool_layout);
    const HostKVPageLayout host = plan_host_kv_page_layout(geometry.geometry);
    require(host.page_stride == expected_stride, "page record bytes " + std::to_string(host.page_stride));
    auto reservation = pool.reserve(8);
    require(reservation.has_value(), "pages");
    std::vector<DeviceKVPageLease> leases;
    leases.reserve(8); // materialize never reallocates its destination
    pool.materialize(*reservation, 8, leases);
    PinnedHostBuffer records(3 * host.page_stride);
    std::byte* const rec[3] = {static_cast<std::byte*>(records.data()),
                               static_cast<std::byte*>(records.data()) + host.page_stride,
                               static_cast<std::byte*>(records.data()) + 2 * host.page_stride};
    const DeviceKVPageHandle source = leases[2].handle(), target = leases[5].handle(), other = leases[6].handle();
    pool.copy_to_host_records(std::span(&source, 1), std::span(&rec[0], 1), {}, host);
    pool.copy_to_host_records(std::span(&other, 1), std::span(&rec[2], 1), {}, host);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<std::byte> other_before(rec[2], rec[2] + host.page_stride);
    // Into a different page one KV layer at a time, MTP first.
    const std::byte* const from[1] = {rec[0]};
    for (std::uint32_t k = geometry.kv_layers; k-- > 0;) {
        const PlaneRange planes = kv_layer_planes(geometry, k);
        pool.copy_from_host_records(std::span(from, 1), {}, std::span(&target, 1), host, planes.begin, planes.end, nullptr);
    }
    pool.copy_to_host_records(std::span(&target, 1), std::span(&rec[1], 1), {}, host);
    pool.copy_to_host_records(std::span(&other, 1), std::span(&rec[2], 1), {}, host);
    CUDA_CHECK(cudaDeviceSynchronize());
    require(std::memcmp(rec[0], rec[1], host.page_stride) == 0, "a page moved by KV layers is byte-exact");
    require(std::memcmp(rec[2], other_before.data(), host.page_stride) == 0, "other pages are untouched");
    pool.dematerialize(*reservation, 0, leases);
    std::printf("page records (%s): %zu bytes, %u KV layers of %u planes, byte-exact per layer\n",
                storage == KvCacheStorage::Int8Group64 ? "int8" : "bf16", host.page_stride, geometry.kv_layers,
                geometry.planes_per_layer);
}

} // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    try {
        const TextConfig c = flash_next();
        test_layout(c);
        if (cuda_unavailable()) {
            std::printf("SKIP: no usable CUDA device\n");
            return 77;
        }
        test_round_trips(c);
        for (const std::uint32_t K : {1U, 2U, 24U, 47U}) {
            test_split_image(c, K);
            test_split_page_records(c, K);
        }
        // 13 KV layers x 64 positions: INT8 K/V with FP16 group scales, pooled BF16 index keys.
        test_page_records(c, KvCacheStorage::Int8Group64, 931'840);
        // BF16 K and FP16 V: 2 x 2 heads x 256 x 2 bytes, plus 64 bytes of pooled keys, per position.
        test_page_records(c, KvCacheStorage::BFloat16, 13ULL * 64 * (2 * 2 * 256 * 2 + 64));
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
    std::printf("qwen4_exp prefix state checks passed\n");
    return 0;
}
