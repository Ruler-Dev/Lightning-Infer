#pragma once

// The layer split of a Qwen4Exp model across devices: segment s runs the contiguous decoder layers
// [begin, end) on one device, and the model's single components are owned by fixed segments. The
// first segment embeds the prompt; the segment holding the PLE layer computes the PLE and its
// n-gram gate; the last runs the final mixer, the head, sampling, the MTP drafter (whose draft for
// round r + 1 depends on round r's acceptance) and allocates the KV pages. Only the residual
// crosses a segment boundary, once per call.
//
// Each segment's GDN and attention layers are contiguous slices of the model's compact indices
// (TextConfig::compact_layer_indices counts each kind in layer order), so a segment's state, KV
// planes and tails are indexed by compact - gdn_base or compact - attention_base.
//
// Same-ordinal test mode: several segments may name one device (--devices 0,0), each with its own
// DeviceContext; they then split that device's expert frames statically in proportion to their
// layers (frame_share), and the runtime VRAM monitor stays off.
//
// Pure host data, no CUDA.

#include "models/qwen4_exp/config.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

namespace infernix::models::qwen4_exp {

// Decoder layers [begin, end).
struct LayerRange {
    std::uint32_t begin = 0, end = 0;

    [[nodiscard]] constexpr bool contains(std::uint32_t l) const noexcept {
        return l >= begin && l < end;
    }

    [[nodiscard]] constexpr std::uint32_t size() const noexcept { return end - begin; }
};

// The segment tag of an allocation every segment reads (the pinned expert banks, the embedding).
inline constexpr std::uint8_t kSharedSegment = 0xFF;

struct SegmentSpec {
    std::uint32_t index = 0;
    int device          = 0; // CUDA ordinal (PCI_BUS_ID order)
    LayerRange layers;
    // The model's single components this segment owns; kv_primary == head (the KV page allocator).
    bool embed = false, ple = false, head = false, mtp = false, kv_primary = false;
    // Compact GDN indices [gdn_base, gdn_base + gdn_count), compact attention indices likewise.
    std::uint32_t gdn_base = 0, gdn_count = 0;
    std::uint32_t attention_base = 0, attention_count = 0;
    std::uint32_t kv_layers = 0; // attention_count + (mtp ? 1 : 0)
    bool shares_device      = false;
    double frame_share      = 1.0; // same-ordinal test mode: the share of the device's frames
};

class SegmentMap {
public:
    // The whole model on one device: today's single-GPU placement.
    static SegmentMap single(const TextConfig& c, bool mtp, int device);
    // Segment i runs layers [ends[i-1], ends[i]) (from 0) on devices[i]: `ends` ascend to the layer
    // count L, each segment owning at least one layer. `mtp`: the MTP drafter is bound (its KV
    // layer joins the last segment). Throws std::invalid_argument on malformed ends or on a config
    // whose layer kinds and compact indices disagree.
    static SegmentMap contiguous(const TextConfig& c, bool mtp, std::span<const int> devices,
                                 std::span<const std::uint32_t> ends);

    [[nodiscard]] std::size_t size() const noexcept { return segments_.size(); }

    [[nodiscard]] const SegmentSpec& operator[](std::size_t i) const { return segments_.at(i); }

    [[nodiscard]] std::span<const SegmentSpec> segments() const noexcept { return segments_; }

    // The segment that runs decoder layer `layer`; throws std::out_of_range past the last layer.
    [[nodiscard]] std::uint32_t segment_of(std::uint32_t layer) const;

    [[nodiscard]] const SegmentSpec& first() const { return segments_.front(); }

    [[nodiscard]] const SegmentSpec& last() const { return segments_.back(); }

    [[nodiscard]] const SegmentSpec& ple() const { return segments_[ple_]; }

    // Some device runs more than one segment (the same-ordinal test mode). One segment never does.
    [[nodiscard]] bool same_device() const noexcept {
        return std::any_of(segments_.begin(), segments_.end(),
                           [](const SegmentSpec& s) { return s.shares_device; });
    }

private:
    SegmentMap() = default;

    std::vector<SegmentSpec> segments_; // never empty
    std::uint32_t ple_ = 0;             // the segment holding the PLE layer
};

inline SegmentMap SegmentMap::single(const TextConfig& c, bool mtp, int device) {
    const int devices[]        = {device};
    const std::uint32_t ends[] = {c.num_hidden_layers};
    return contiguous(c, mtp, devices, ends);
}

inline SegmentMap SegmentMap::contiguous(const TextConfig& c, bool mtp,
                                         std::span<const int> devices,
                                         std::span<const std::uint32_t> ends) {
    const std::uint32_t layers = c.num_hidden_layers;
    if (layers == 0 || c.layer_types.size() != layers || c.compact_layer_indices.size() != layers) {
        throw std::invalid_argument(
            "segment map: the config must give every layer a kind and a compact index");
    }
    if (c.ple.layer >= layers) {
        throw std::invalid_argument("segment map: the PLE layer is past the last layer");
    }
    if (devices.empty() || devices.size() != ends.size()) {
        throw std::invalid_argument("segment map: every segment needs one device and one end");
    }
    // Segment indices are u8 tags in the load bindings, kSharedSegment excluded.
    if (devices.size() >= kSharedSegment) {
        throw std::invalid_argument("segment map: too many segments");
    }

    SegmentMap map;
    map.segments_.reserve(devices.size());
    std::uint32_t begin = 0, gdn = 0, attention = 0; // running counts of each kind before `begin`
    for (std::size_t i = 0; i < devices.size(); ++i) {
        if (devices[i] < 0) {
            throw std::invalid_argument("segment map: device ordinals must be non-negative");
        }
        if (ends[i] <= begin || ends[i] > layers) {
            throw std::invalid_argument(
                "segment map: segment ends must ascend to the layer count, at least one "
                "layer per segment");
        }
        SegmentSpec s;
        s.index          = static_cast<std::uint32_t>(i);
        s.device         = devices[i];
        s.layers         = {begin, ends[i]};
        s.gdn_base       = gdn;
        s.attention_base = attention;
        for (std::uint32_t l = begin; l < ends[i]; ++l) {
            std::uint32_t& kind = c.layer_types[l] == MixerKind::Gdn ? gdn : attention;
            // Compact indices are running counters: a contiguous range is a contiguous slice.
            if (c.compact_layer_indices[l] != kind) {
                throw std::invalid_argument(
                    "segment map: compact layer indices must count each kind in layer order");
            }
            ++kind;
        }
        s.gdn_count       = gdn - s.gdn_base;
        s.attention_count = attention - s.attention_base;
        map.segments_.push_back(s);
        begin = ends[i];
    }
    if (begin != layers) {
        throw std::invalid_argument("segment map: the last segment must end at the last layer");
    }
    if (gdn != c.gdn_layers || attention != c.attention_layers) {
        throw std::invalid_argument(
            "segment map: the config's layer kind counts disagree with its layer kinds");
    }

    map.segments_.front().embed = true;
    map.ple_                    = map.segment_of(c.ple.layer);
    map.segments_[map.ple_].ple = true;
    SegmentSpec& head           = map.segments_.back();
    head.head = head.kv_primary = true;
    head.mtp                    = mtp;
    for (SegmentSpec& s : map.segments_) {
        s.kv_layers = s.attention_count + (s.mtp ? 1U : 0U);
        // Segments on one ordinal split its frames by their layers.
        std::uint32_t on_device = 0, sharing = 0;
        for (const SegmentSpec& other : map.segments_) {
            if (other.device != s.device) { continue; }
            on_device += other.layers.size();
            ++sharing;
        }
        s.shares_device = sharing > 1;
        s.frame_share   = s.shares_device ? double(s.layers.size()) / double(on_device) : 1.0;
    }
    return map;
}

inline std::uint32_t SegmentMap::segment_of(std::uint32_t layer) const {
    const auto it =
        std::partition_point(segments_.begin(), segments_.end(),
                             [layer](const SegmentSpec& s) { return s.layers.end <= layer; });
    if (it == segments_.end()) {
        throw std::out_of_range("segment map: layer past the last segment");
    }
    return it->index;
}

namespace execution {
using qwen4_exp::LayerRange;
}

} // namespace infernix::models::qwen4_exp
