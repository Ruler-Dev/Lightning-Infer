#include "models/qwen4_exp/program/prefix/state_image.h"

#include "core/device.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

namespace infernix::models::qwen4_exp::prefix {
namespace {

constexpr std::uint64_t kHeaderMagic   = 0x31474D4934515751ULL; // "QWQ4IMG1"
constexpr std::uint32_t kHeaderVersion = 2;

// Header bytes: magic, version, fingerprint, frontier, mtp_next, flags, mtp_accept, draft_frontier.
struct PackedHeader {
    std::uint64_t magic       = 0;
    std::uint32_t version     = 0;
    std::uint32_t flags       = 0; // bit 0 mtp_written, bit 1 lineage_echo, bit 2 opener
    std::uint64_t fingerprint = 0;
    std::uint64_t image_bytes = 0;
    std::uint32_t frontier    = 0;
    std::int32_t mtp_next     = -1;
    std::array<double, 8> mtp_accept{};
    std::uint32_t draft_frontier = 0;
};
static_assert(sizeof(PackedHeader) <= kStateImageHeaderBytes);

std::size_t align_up(std::size_t bytes) {
    return (bytes + kStateImageAlignment - 1U) / kStateImageAlignment * kStateImageAlignment;
}

std::uint64_t fnv1a(std::uint64_t hash, std::uint64_t value) {
    for (int i = 0; i < 8; ++i) {
        hash ^= (value >> (8 * i)) & 0xFFU;
        hash *= 0x100000001B3ULL;
    }
    return hash;
}

// One copy per contiguous run between a Device range and the segmented Host image.
template <class Copy>
void for_each_segment_run(std::size_t offset, std::size_t bytes, std::size_t segment_bytes, Copy&& copy) {
    std::size_t done = 0;
    while (done < bytes) {
        const std::size_t at      = offset + done;
        const std::size_t segment = at / segment_bytes;
        const std::size_t within  = at % segment_bytes;
        const std::size_t run     = std::min(bytes - done, segment_bytes - within);
        copy(segment, within, done, run);
        done += run;
    }
}

} // namespace

KvWindowGeometry kv_window_geometry(const TextConfig& c, KvCacheStorage storage, bool mtp, std::uint32_t lanes) {
    return kv_window_geometry(c, storage, c.attention_layers, mtp, lanes);
}

KvWindowGeometry kv_window_geometry(const TextConfig& c, KvCacheStorage storage, std::uint32_t attention_layers,
                                    bool mtp, std::uint32_t lanes) {
    if (attention_layers > c.attention_layers) {
        throw std::invalid_argument("Qwen4Exp window: more attention layers than the model has");
    }
    KvWindowGeometry g;
    const std::uint32_t kv_layers = attention_layers + (mtp ? 1U : 0U);
    if (!kv_storage_has_exact_window(storage) || kv_layers == 0) { return g; }
    g.kv_heads  = static_cast<std::int32_t>(c.attention.kv_heads);
    g.kv_layers = kv_layers;
    g.lanes     = lanes;
    return g;
}

std::size_t KvWindowGeometry::lane_bytes(int plane) const {
    const std::size_t rows = static_cast<std::size_t>(kKVWindowSlots) * static_cast<std::size_t>(kv_heads);
    switch (plane) {
    case 0:
    case 1: return rows * kD256KVCacheHeadDim;              // K, V codes
    case 2:
    case 3: return rows * kKVWindowGroups * sizeof(std::uint16_t); // K, V group scales
    default: return rows * 2 * sizeof(std::int32_t);         // tags
    }
}

std::size_t KvWindowGeometry::layer_bytes() const {
    std::size_t bytes = 0;
    for (int p = 0; p < kPlanes; ++p) { bytes += lane_bytes(p) * lanes; }
    return bytes;
}

std::size_t KvWindowGeometry::plane_offset(std::uint32_t kv_layer, int plane) const {
    std::size_t offset = layer_bytes() * kv_layer;
    for (int p = 0; p < plane; ++p) { offset += lane_bytes(p) * lanes; }
    return offset;
}

PagedKVWindowView kv_window_view(const KvWindowGeometry& g, void* base, std::uint32_t kv_layer) {
    if (!g.present()) { return {}; }
    if (kv_layer >= g.kv_layers) { throw std::out_of_range("Qwen4Exp window: KV layer out of range"); }
    auto* bytes            = static_cast<std::byte*>(base);
    const std::int32_t s   = kKVWindowSlots, h = g.kv_heads, l = static_cast<std::int32_t>(g.lanes);
    const auto plane       = [&](int p) { return bytes + g.plane_offset(kv_layer, p); };
    return PagedKVWindowView{.k_codes  = Tensor(plane(0), DType::I8, {kD256KVCacheHeadDim, s, h, l}),
                             .v_codes  = Tensor(plane(1), DType::I8, {kD256KVCacheHeadDim, s, h, l}),
                             .k_scales = Tensor(plane(2), DType::FP16, {kKVWindowGroups, s, h, l}),
                             .v_scales = Tensor(plane(3), DType::FP16, {kKVWindowGroups, s, h, l}),
                             .tags     = Tensor(plane(4), DType::I32, {2, s, h, l}),
                             .slots    = {}};
}

StateImageSpec state_image_spec(const TextConfig& c, bool mtp, KvCacheStorage storage, DraftRingGeometry draft_ring) {
    StateImageSpec spec;
    spec.layer_types         = c.layer_types;
    spec.ple_layer           = c.ple.layer;
    spec.gdn_recurrent_bytes = static_cast<std::size_t>(c.gdn.key_head_dim) * c.gdn.value_head_dim * c.gdn.value_heads *
                               sizeof(float);
    spec.gdn_conv_bytes = static_cast<std::size_t>(c.gdn.conv_channels()) * (c.gdn.conv_kernel - 1U) * 2U;
    spec.ple_bytes      = static_cast<std::size_t>(c.residual_width()) * c.ple.conv_span() * 2U;
    spec.qsa_tail_bytes = static_cast<std::size_t>(c.qsa.index_head_dim) * (c.qsa.compress_ratio - 1U) * 2U;
    spec.mtp            = mtp;
    spec.draft_ring     = draft_ring;
    if (mtp && draft_ring.layers != 0) { throw std::invalid_argument("Qwen4Exp state image: one drafter"); }
    if (mtp) {
        spec.mtp_saved_bytes = static_cast<std::size_t>(c.residual_width()) * 2U;
        spec.mtp_tail_bytes  = spec.qsa_tail_bytes;
    }
    const KvWindowGeometry window = kv_window_geometry(c, storage, mtp, 1);
    if (window.present()) {
        for (int p = 0; p < KvWindowGeometry::kPlanes; ++p) { spec.window_bytes[p] = window.lane_bytes(p); }
    }
    return spec;
}

StateImageLayout plan_state_image(const StateImageSpec& spec) {
    if (spec.layer_types.empty() || spec.ple_layer >= spec.layer_types.size()) {
        throw std::invalid_argument("Qwen4Exp state image: the PLE layer is outside the decoder");
    }
    StateImageLayout layout;
    layout.spec           = spec;
    std::size_t offset    = 0;
    const auto add_part   = [&](StateImagePart::Kind kind, std::uint32_t index, std::size_t bytes) {
        if (bytes == 0) { throw std::invalid_argument("Qwen4Exp state image: a part has no bytes"); }
        layout.parts.push_back(StateImagePart{kind, index, offset, bytes});
        offset = align_up(offset + bytes);
    };
    // A KV layer's exact-window planes (vector-quantized storages) join its group.
    const auto add_window = [&](std::uint32_t kv_layer) {
        if (spec.window_bytes[0] == 0) { return; }
        for (std::uint32_t p = 0; p < KvWindowGeometry::kPlanes; ++p) {
            add_part(StateImagePart::Kind::QsaWindow, kv_layer * KvWindowGeometry::kPlanes + p, spec.window_bytes[p]);
        }
    };
    add_part(StateImagePart::Kind::Header, 0, kStateImageHeaderBytes);
    std::uint32_t gdn = 0, attention = 0;
    for (std::uint32_t layer = 0; layer < spec.layer_types.size(); ++layer) {
        layout.group_begin.push_back(static_cast<std::uint32_t>(layout.parts.size()));
        if (layer == spec.ple_layer) { add_part(StateImagePart::Kind::Ple, 0, spec.ple_bytes); }
        if (spec.layer_types[layer] == MixerKind::Gdn) {
            add_part(StateImagePart::Kind::GdnRecurrent, gdn, spec.gdn_recurrent_bytes);
            add_part(StateImagePart::Kind::GdnConv, gdn, spec.gdn_conv_bytes);
            ++gdn;
        } else {
            add_part(StateImagePart::Kind::QsaTail, attention, spec.qsa_tail_bytes);
            add_window(attention);
            ++attention;
        }
    }
    layout.group_begin.push_back(static_cast<std::uint32_t>(layout.parts.size()));
    if (spec.mtp) {
        add_part(StateImagePart::Kind::MtpSaved, 0, spec.mtp_saved_bytes);
        add_part(StateImagePart::Kind::MtpTail, 0, spec.mtp_tail_bytes);
        add_window(attention);
    }
    for (std::uint32_t layer = 0; layer < spec.draft_ring.layers; ++layer) {
        add_part(StateImagePart::Kind::DraftRingK, layer, spec.draft_ring.k_bytes);
        add_part(StateImagePart::Kind::DraftRingV, layer, spec.draft_ring.v_bytes);
    }
    layout.group_begin.push_back(static_cast<std::uint32_t>(layout.parts.size()));
    layout.image_bytes = offset;
    std::uint64_t hash = 0xCBF29CE484222325ULL;
    for (const StateImagePart& part : layout.parts) {
        hash = fnv1a(hash, static_cast<std::uint64_t>(part.kind));
        hash = fnv1a(hash, part.index);
        hash = fnv1a(hash, part.offset);
        hash = fnv1a(hash, part.bytes);
    }
    layout.fingerprint = hash;
    return layout;
}

void write_state_image_header(const StateImageLayout& layout, const StateImageHeader& header,
                              std::byte* image_begin) {
    PackedHeader packed;
    packed.magic       = kHeaderMagic;
    packed.version     = kHeaderVersion;
    packed.flags       = (header.mtp_written ? 1U : 0U) | (header.lineage_echo ? 2U : 0U) | (header.opener ? 4U : 0U);
    packed.fingerprint = layout.fingerprint;
    packed.image_bytes = layout.image_bytes;
    packed.frontier    = header.frontier;
    packed.mtp_next    = header.mtp_next;
    packed.mtp_accept  = header.mtp_accept;
    packed.draft_frontier = header.draft_frontier;
    std::memset(image_begin, 0, kStateImageHeaderBytes);
    std::memcpy(image_begin, &packed, sizeof(packed));
}

std::optional<StateImageHeader> read_state_image_header(const StateImageLayout& layout,
                                                        const std::byte* image_begin) {
    PackedHeader packed;
    std::memcpy(&packed, image_begin, sizeof(packed));
    if (packed.magic != kHeaderMagic || packed.version != kHeaderVersion ||
        packed.fingerprint != layout.fingerprint || packed.image_bytes != layout.image_bytes ||
        (packed.flags & ~7U) != 0) {
        return std::nullopt;
    }
    StateImageHeader header;
    header.frontier     = packed.frontier;
    header.mtp_written  = (packed.flags & 1U) != 0;
    header.lineage_echo = (packed.flags & 2U) != 0;
    header.opener       = (packed.flags & 4U) != 0;
    header.mtp_next     = packed.mtp_next;
    header.mtp_accept   = packed.mtp_accept;
    header.draft_frontier = packed.draft_frontier;
    return header;
}

LaneStateImage::LaneStateImage(const StateImageLayout& layout, std::span<const LaneStateBuffers> segments)
    : layout_(&layout) {
    const StateImageSpec& spec = layout.spec;
    const auto layers          = static_cast<std::uint32_t>(spec.layer_types.size());
    if (segments.empty() || segments.front().lanes == 0) {
        throw std::invalid_argument("Qwen4Exp state image: lane buffers are missing");
    }
    lanes_                 = segments.front().lanes;
    std::uint32_t next     = 0;
    std::uint32_t gdn      = 0;
    std::uint32_t attention = 0;
    for (std::size_t s = 0; s < segments.size(); ++s) {
        const LaneStateBuffers& b = segments[s];
        if (b.layer_begin != next || b.layer_end <= b.layer_begin || b.layer_end > layers ||
            b.head != (s + 1 == segments.size()) || b.gdn_base != gdn || b.attention_base != attention ||
            b.lanes != lanes_) {
            throw std::invalid_argument("Qwen4Exp state image: the segments do not cover the decoder layers in order");
        }
        Segment segment{.buffers = b};
        for (std::uint32_t l = b.layer_begin; l < b.layer_end; ++l) {
            ++(spec.layer_types[l] == MixerKind::Gdn ? segment.gdn_count : segment.attention_count);
        }
        const bool ple                = spec.ple_layer >= b.layer_begin && spec.ple_layer < b.layer_end;
        const bool mtp                = b.head && spec.mtp;
        const std::uint32_t kv_layers = segment.attention_count + (mtp ? 1U : 0U);
        if ((ple && b.ple == nullptr) || (kv_layers > 0 && b.tails == nullptr) || (mtp && b.mtp_saved == nullptr)) {
            throw std::invalid_argument("Qwen4Exp state image: lane buffers are missing");
        }
        if (spec.window_bytes[0] != 0 && kv_layers > 0) {
            const KvWindowGeometry& w = b.window_geometry;
            bool same                 = b.window != nullptr && w.lanes >= lanes_ && w.kv_layers == kv_layers;
            for (int p = 0; same && p < KvWindowGeometry::kPlanes; ++p) {
                same = w.lane_bytes(p) == spec.window_bytes[p];
            }
            if (!same) {
                throw std::invalid_argument("Qwen4Exp state image: the exact window does not match the layout");
            }
        }
        if (segment.gdn_count > 0) {
            if (b.gdn == nullptr || b.gdn->layer_count() != segment.gdn_count ||
                b.gdn->slot_count() < static_cast<std::int32_t>(lanes_) ||
                b.gdn->recurrent_slot(0, 0).bytes() != spec.gdn_recurrent_bytes ||
                b.gdn->conv_slot(0, 0).bytes() != spec.gdn_conv_bytes) {
                throw std::invalid_argument("Qwen4Exp state image: the GDN state pool does not match the layout");
            }
        }
        if (b.head) {
            const bool ring_matches =
                b.ring_k.size() == spec.draft_ring.layers &&
                b.ring_v.size() == spec.draft_ring.layers &&
                std::none_of(b.ring_k.begin(), b.ring_k.end(),
                             [](const std::byte* p) { return p == nullptr; }) &&
                std::none_of(b.ring_v.begin(), b.ring_v.end(),
                             [](const std::byte* p) { return p == nullptr; });
            if (!ring_matches) {
                throw std::invalid_argument(
                    "Qwen4Exp state image: the DFlash2 ring does not match the layout");
            }
        }
        next = b.layer_end;
        gdn += segment.gdn_count;
        attention += segment.attention_count;
        segments_.push_back(segment);
    }
    if (next != layers) {
        throw std::invalid_argument("Qwen4Exp state image: the segments do not cover the decoder layers in order");
    }
}

std::uint32_t LaneStateImage::group_segment(std::uint32_t group) const {
    if (group >= layout_->groups()) { throw std::out_of_range("Qwen4Exp state image: group out of range"); }
    for (std::uint32_t s = 0; s + 1 < segments_.size(); ++s) {
        if (group < segments_[s].buffers.layer_end) { return s; }
    }
    return static_cast<std::uint32_t>(segments_.size()) - 1U; // the head: its layers and the MTP group
}

const LaneStateImage::Segment& LaneStateImage::part_segment(const StateImagePart& part) const {
    const auto holds = [&](auto&& contains) -> const Segment& {
        for (const Segment& segment : segments_) {
            if (contains(segment)) { return segment; }
        }
        throw std::out_of_range("Qwen4Exp state image: no segment holds the part");
    };
    const auto attention_of = [&](std::uint32_t index) -> const Segment& {
        return holds([&](const Segment& s) {
            return index >= s.buffers.attention_base && index < s.buffers.attention_base + s.attention_count;
        });
    };
    switch (part.kind) {
    case StateImagePart::Kind::Header: break;
    case StateImagePart::Kind::Ple: return segments_[group_segment(layout_->spec.ple_layer)];
    case StateImagePart::Kind::GdnRecurrent:
    case StateImagePart::Kind::GdnConv:
        return holds([&](const Segment& s) {
            return part.index >= s.buffers.gdn_base && part.index < s.buffers.gdn_base + s.gdn_count;
        });
    case StateImagePart::Kind::QsaTail: return attention_of(part.index);
    case StateImagePart::Kind::MtpSaved:
    case StateImagePart::Kind::MtpTail: return segments_.back();
    case StateImagePart::Kind::QsaWindow: {
        // A KV layer past the model's attention layers is the MTP block's, on the head.
        const std::uint32_t kv_layer = part.index / KvWindowGeometry::kPlanes;
        const Segment& last          = segments_.back();
        return kv_layer < last.buffers.attention_base + last.attention_count ? attention_of(kv_layer) : last;
    }
    case StateImagePart::Kind::DraftRingK:
    case StateImagePart::Kind::DraftRingV: return segments_.back();
    }
    throw std::logic_error("Qwen4Exp state image: the header has no Device bytes");
}

std::byte* LaneStateImage::device_part(const StateImagePart& part, std::uint32_t lane) const {
    check_lane(lane);
    const StateImageSpec& spec = layout_->spec;
    const Segment& segment     = part_segment(part);
    const LaneStateBuffers& b  = segment.buffers;
    const auto slot            = static_cast<std::int32_t>(lane);
    switch (part.kind) {
    case StateImagePart::Kind::Header: break;
    case StateImagePart::Kind::Ple: return b.ple + static_cast<std::size_t>(lane) * spec.ple_bytes;
    case StateImagePart::Kind::GdnRecurrent:
        return static_cast<std::byte*>(b.gdn->recurrent_slot(part.index - b.gdn_base, slot).data);
    case StateImagePart::Kind::GdnConv:
        return static_cast<std::byte*>(b.gdn->conv_slot(part.index - b.gdn_base, slot).data);
    case StateImagePart::Kind::QsaTail:
        return b.tails +
               (static_cast<std::size_t>(part.index - b.attention_base) * lanes_ + lane) * spec.qsa_tail_bytes;
    case StateImagePart::Kind::MtpSaved: return b.mtp_saved + static_cast<std::size_t>(lane) * spec.mtp_saved_bytes;
    case StateImagePart::Kind::MtpTail:
        return b.tails + (static_cast<std::size_t>(segment.attention_count) * lanes_ + lane) * spec.qsa_tail_bytes;
    case StateImagePart::Kind::QsaWindow: {
        const KvWindowGeometry& w    = b.window_geometry;
        const int plane              = static_cast<int>(part.index % KvWindowGeometry::kPlanes);
        const std::uint32_t kv_layer = part.index / KvWindowGeometry::kPlanes;
        // The segment's KV layer: its attention layers from 0, then the MTP block's on the head.
        const std::uint32_t local = kv_layer < b.attention_base + segment.attention_count ? kv_layer - b.attention_base
                                                                                          : segment.attention_count;
        return b.window + w.plane_offset(local, plane) + static_cast<std::size_t>(lane) * w.lane_bytes(plane);
    }
    case StateImagePart::Kind::DraftRingK:
        return b.ring_k.at(part.index) + static_cast<std::size_t>(lane) * spec.draft_ring.k_bytes;
    case StateImagePart::Kind::DraftRingV:
        return b.ring_v.at(part.index) + static_cast<std::size_t>(lane) * spec.draft_ring.v_bytes;
    }
    throw std::logic_error("Qwen4Exp state image: the header has no Device bytes");
}

void LaneStateImage::check_lane(std::uint32_t lane) const {
    if (lane >= lanes_) { throw std::out_of_range("Qwen4Exp state image: lane out of range"); }
}

void LaneStateImage::check_image(std::size_t segments, std::size_t segment_bytes) const {
    if (segment_bytes == 0 || segments * segment_bytes < layout_->image_bytes) {
        throw std::invalid_argument("Qwen4Exp state image: the Host segments do not hold an image");
    }
}

void LaneStateImage::copy_lane_to_host(std::uint32_t lane, const HostImage& image, cudaStream_t stream) const {
    check_image(image.segments.size(), image.segment_bytes);
    for (std::size_t p = 1; p < layout_->parts.size(); ++p) {
        const StateImagePart& part = layout_->parts[p];
        const std::byte* source    = device_part(part, lane);
        for_each_segment_run(part.offset, part.bytes, image.segment_bytes,
                             [&](std::size_t segment, std::size_t within, std::size_t done, std::size_t run) {
                                 CUDA_CHECK(cudaMemcpyAsync(image.segments[segment] + within, source + done, run,
                                                            cudaMemcpyDeviceToHost, stream));
                             });
    }
}

void LaneStateImage::copy_group_to_host(std::uint32_t group, std::uint32_t lane, const HostImage& image,
                                        cudaStream_t stream) const {
    check_image(image.segments.size(), image.segment_bytes);
    if (group >= layout_->groups()) { throw std::out_of_range("Qwen4Exp state image: group out of range"); }
    for (std::uint32_t p = layout_->group_begin[group]; p < layout_->group_begin[group + 1U]; ++p) {
        const StateImagePart& part = layout_->parts[p];
        const std::byte* source    = device_part(part, lane);
        for_each_segment_run(part.offset, part.bytes, image.segment_bytes,
                             [&](std::size_t segment, std::size_t within, std::size_t done, std::size_t run) {
                                 CUDA_CHECK(cudaMemcpyAsync(image.segments[segment] + within, source + done, run,
                                                            cudaMemcpyDeviceToHost, stream));
                             });
    }
}

void LaneStateImage::copy_lane_to_packed(std::uint32_t lane, std::byte* packed, cudaStream_t stream) const {
    for (std::size_t p = 1; p < layout_->parts.size(); ++p) {
        const StateImagePart& part = layout_->parts[p];
        CUDA_CHECK(cudaMemcpyAsync(packed + part.offset, device_part(part, lane), part.bytes,
                                   cudaMemcpyDeviceToDevice, stream));
    }
}

void LaneStateImage::copy_packed_to_host(const std::byte* packed, const HostImage& image, cudaStream_t stream) const {
    check_image(image.segments.size(), image.segment_bytes);
    for_each_segment_run(0, layout_->image_bytes, image.segment_bytes,
                         [&](std::size_t segment, std::size_t within, std::size_t done, std::size_t run) {
                             CUDA_CHECK(cudaMemcpyAsync(image.segments[segment] + within, packed + done, run,
                                                        cudaMemcpyDeviceToHost, stream));
                         });
}

void LaneStateImage::copy_group_from_host(std::uint32_t group, const HostImageConst& image, std::uint32_t lane,
                                          cudaStream_t stream) const {
    check_image(image.segments.size(), image.segment_bytes);
    if (group >= layout_->groups()) { throw std::out_of_range("Qwen4Exp state image: group out of range"); }
    for (std::uint32_t p = layout_->group_begin[group]; p < layout_->group_begin[group + 1U]; ++p) {
        const StateImagePart& part = layout_->parts[p];
        std::byte* destination     = device_part(part, lane);
        for_each_segment_run(part.offset, part.bytes, image.segment_bytes,
                             [&](std::size_t segment, std::size_t within, std::size_t done, std::size_t run) {
                                 CUDA_CHECK(cudaMemcpyAsync(destination + done, image.segments[segment] + within, run,
                                                            cudaMemcpyHostToDevice, stream));
                             });
    }
}

void LaneStateImage::copy_group_from_packed(std::uint32_t group, const std::byte* packed, std::uint32_t lane,
                                            cudaStream_t stream) const {
    if (group >= layout_->groups()) { throw std::out_of_range("Qwen4Exp state image: group out of range"); }
    for (std::uint32_t p = layout_->group_begin[group]; p < layout_->group_begin[group + 1U]; ++p) {
        const StateImagePart& part = layout_->parts[p];
        CUDA_CHECK(cudaMemcpyAsync(device_part(part, lane), packed + part.offset, part.bytes,
                                   cudaMemcpyDeviceToDevice, stream));
    }
}

KvPageGeometry kv_page_geometry(const TextConfig& c, KvCacheStorage storage, bool mtp) {
    return kv_page_geometry(c, storage, c.attention_layers, mtp);
}

KvPageGeometry kv_page_geometry(const TextConfig& c, KvCacheStorage storage, std::uint32_t attention_layers, bool mtp) {
    if (attention_layers > c.attention_layers) {
        throw std::invalid_argument("Qwen4Exp KV geometry: more attention layers than the model has");
    }
    const auto layout   = paged_kv_storage_layout(storage, static_cast<std::int32_t>(c.attention.head_dim));
    const auto kv_heads = static_cast<std::int32_t>(c.attention.kv_heads);
    KvPageGeometry out;
    out.kv_layers = attention_layers + (mtp ? 1U : 0U);
    for (std::uint32_t l = 0; l < out.kv_layers; ++l) {
        out.geometry.planes.push_back({layout.key.data_dtype, layout.key.data_leading_extent, kv_heads});
        if (layout.key.has_scale()) {
            out.geometry.planes.push_back({layout.key.scale_dtype, layout.key.scale_leading_extent, kv_heads});
        }
        out.geometry.planes.push_back({layout.value.data_dtype, layout.value.data_leading_extent, kv_heads});
        if (layout.value.has_scale()) {
            out.geometry.planes.push_back({layout.value.scale_dtype, layout.value.scale_leading_extent, kv_heads});
        }
        out.geometry.planes.push_back(
            {DType::BF16, static_cast<std::int32_t>(c.qsa.index_head_dim / c.qsa.compress_ratio), 1});
    }
    out.planes_per_layer = out.kv_layers == 0 ? 0U : static_cast<std::uint32_t>(out.geometry.planes.size()) / out.kv_layers;
    return out;
}

PlaneRange kv_layer_planes(const KvPageGeometry& geometry, std::uint32_t kv_layer) {
    if (kv_layer >= geometry.kv_layers) { throw std::out_of_range("Qwen4Exp KV layer out of range"); }
    const std::size_t begin = static_cast<std::size_t>(kv_layer) * geometry.planes_per_layer;
    return PlaneRange{begin, begin + geometry.planes_per_layer};
}

std::size_t global_kv_plane(const KvPageGeometry& segment, std::uint32_t attention_base,
                            std::uint32_t global_attention_layers, std::size_t segment_plane, bool segment_has_mtp) {
    if (segment.geometry.planes.size() != static_cast<std::size_t>(segment.kv_layers) * segment.planes_per_layer ||
        (segment_has_mtp && segment.kv_layers == 0)) {
        throw std::invalid_argument("Qwen4Exp KV geometry: the segment's planes do not match its KV layers");
    }
    const std::uint32_t attention = segment.kv_layers - (segment_has_mtp ? 1U : 0U);
    if (attention_base > global_attention_layers || attention > global_attention_layers - attention_base) {
        throw std::out_of_range("Qwen4Exp KV geometry: the segment's attention layers are outside the model");
    }
    if (segment_plane >= segment.geometry.planes.size()) { throw std::out_of_range("Qwen4Exp KV plane out of range"); }
    const auto local          = static_cast<std::uint32_t>(segment_plane / segment.planes_per_layer);
    const std::uint32_t layer = local < attention ? attention_base + local : global_attention_layers;
    return static_cast<std::size_t>(layer) * segment.planes_per_layer + segment_plane % segment.planes_per_layer;
}

HostKVPageLayout segment_host_page_layout(const HostKVPageLayout& model, const KvPageGeometry& segment,
                                          std::uint32_t attention_base, std::uint32_t global_attention_layers,
                                          bool segment_has_mtp) {
    HostKVPageLayout out;
    out.geometry    = segment.geometry;
    out.page_stride = model.page_stride;
    if (segment.geometry.page_tokens != model.geometry.page_tokens ||
        segment.geometry.device_plane_order != model.geometry.device_plane_order) {
        throw std::invalid_argument("Qwen4Exp KV geometry: the segment's pages differ from the model's");
    }
    for (std::size_t p = 0; p < segment.geometry.planes.size(); ++p) {
        const std::size_t global =
            global_kv_plane(segment, attention_base, global_attention_layers, p, segment_has_mtp);
        if (global >= model.planes.size() || model.geometry.planes[global] != segment.geometry.planes[p]) {
            throw std::invalid_argument("Qwen4Exp KV geometry: a segment's plane is not the model's");
        }
        out.planes.push_back(model.planes[global]);
    }
    return out;
}

} // namespace infernix::models::qwen4_exp::prefix
