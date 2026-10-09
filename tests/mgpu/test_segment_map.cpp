// The layer split's segment map (host only). On Qwen3.8-Flash-Next's layer pattern (48 layers,
// attention at 3, 7, ..., 47, the PLE at layer 1): every layer belongs to exactly one segment, each
// segment's GDN and attention layers are a contiguous slice of the compact indices, the embedding,
// PLE and head (with the MTP layer's KV) land on their segments for every split K, segments on one
// device ordinal share its frames by layers, and malformed splits are refused.

#include "models/qwen4_exp/segment_map.h"

#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace infernix::models::qwen4_exp;

namespace {

constexpr std::uint32_t kLayers = 48;

void require(bool value, const std::string& message) {
    if (!value) { throw std::runtime_error(message); }
}

template <typename Error, typename F>
void require_throws(F&& f, const std::string& message) {
    try {
        f();
    } catch (const Error&) { return; }
    throw std::runtime_error("not refused: " + message);
}

// The decoder layout parse_config records for Qwen3.8-Flash-Next: every fourth layer (3, 7, ...,
// 47) is attention, the rest GDN; compact indices count each kind in layer order.
TextConfig flash_next(std::uint32_t layers = kLayers) {
    TextConfig c;
    c.num_hidden_layers = layers;
    for (std::uint32_t l = 0; l < layers; ++l) {
        const bool attention = l % 4 == 3;
        c.layer_types.push_back(attention ? MixerKind::Attention : MixerKind::Gdn);
        c.compact_layer_indices.push_back(attention ? c.attention_layers++ : c.gdn_layers++);
    }
    c.ple.layer = 1;
    return c;
}

// Layers below `k` of each kind in this pattern: attention layers 3, 7, ... below k number k / 4.
std::uint32_t attention_below(std::uint32_t k) { return k / 4; }

std::uint32_t gdn_below(std::uint32_t k) { return k - attention_below(k); }

// Every layer has exactly one owner, and its compact index lies in that owner's slice of its kind.
void check_slices(const TextConfig& c, const SegmentMap& map) {
    std::uint32_t gdn = 0, attention = 0, next = 0;
    for (std::size_t i = 0; i < map.size(); ++i) {
        const SegmentSpec& s = map[i];
        require(s.index == i, "segment indices follow their order");
        require(&map.segments()[i] == &s, "segments() and operator[] see the same specs");
        require(s.layers.begin == next && s.layers.size() >= 1,
                "segments tile the layers in order");
        require(s.gdn_base == gdn && s.attention_base == attention,
                "a segment's slices start where the last ended");
        next = s.layers.end;
        gdn += s.gdn_count;
        attention += s.attention_count;
        require(s.kv_layers == s.attention_count + (s.mtp ? 1U : 0U),
                "KV layers are attention plus MTP");
    }
    require(next == kLayers && gdn == c.gdn_layers && attention == c.attention_layers,
            "the segments cover every layer of each kind");
    for (std::uint32_t l = 0; l < kLayers; ++l) {
        const std::uint32_t owner = map.segment_of(l);
        for (std::size_t i = 0; i < map.size(); ++i) {
            require(map[i].layers.contains(l) == (i == owner),
                    "layer " + std::to_string(l) + " has one owner");
        }
        const SegmentSpec& s        = map[owner];
        const std::uint32_t compact = c.compact_layer_indices[l];
        if (c.layer_types[l] == MixerKind::Gdn) {
            require(compact >= s.gdn_base && compact - s.gdn_base < s.gdn_count,
                    "GDN layer in its owner's slice");
        } else {
            require(compact >= s.attention_base && compact - s.attention_base < s.attention_count,
                    "attention layer in its owner's slice");
        }
    }
    require_throws<std::out_of_range>([&] { (void)map.segment_of(kLayers); },
                                      "a layer past the model");
    require_throws<std::out_of_range>([&] { (void)map[map.size()]; }, "a segment past the map");
}

void test_single() {
    const TextConfig c = flash_next();
    for (const bool mtp : {false, true}) {
        const SegmentMap map = SegmentMap::single(c, mtp, 3);
        require(map.size() == 1 && !map.same_device(), "one segment, not the same-ordinal mode");
        const SegmentSpec& s = map.first();
        require(&s == &map.last() && &s == &map.ple(), "one segment owns everything");
        require(s.device == 3 && s.layers.begin == 0 && s.layers.end == kLayers,
                "the whole model on the device");
        require(s.embed && s.ple && s.head && s.kv_primary && s.mtp == mtp,
                "every role on the one segment");
        require(s.gdn_base == 0 && s.gdn_count == 36 && s.attention_base == 0 &&
                    s.attention_count == 12,
                "36 GDN and 12 attention layers");
        require(s.kv_layers == (mtp ? 13U : 12U), "12 attention KV layers plus the MTP layer");
        require(!s.shares_device && s.frame_share == 1.0, "all of the device's frames");
        check_slices(c, map);
    }
}

void test_two_segments() {
    const TextConfig c  = flash_next();
    const int devices[] = {0, 1};
    for (const std::uint32_t k : {1U, 2U, 3U, 4U, 24U, 28U, 44U, 47U}) {
        const std::uint32_t ends[] = {k, kLayers};
        const SegmentMap map       = SegmentMap::contiguous(c, true, devices, ends);
        const std::string at       = " (K=" + std::to_string(k) + ")";
        require(map.size() == 2 && !map.same_device(), "two devices" + at);
        const SegmentSpec& a = map[0];
        const SegmentSpec& b = map[1];
        require(a.device == 0 && b.device == 1, "devices in order" + at);
        require(a.layers.begin == 0 && a.layers.end == k && b.layers.begin == k &&
                    b.layers.end == kLayers,
                "segment 0 = [0,K), segment 1 = [K,48)" + at);
        require(a.gdn_base == 0 && a.gdn_count == gdn_below(k) && a.attention_base == 0 &&
                    a.attention_count == attention_below(k),
                "segment 0's slices" + at);
        require(b.gdn_base == gdn_below(k) && b.gdn_count == 36 - gdn_below(k) &&
                    b.attention_base == attention_below(k) &&
                    b.attention_count == 12 - attention_below(k),
                "segment 1's slices" + at);
        require((a.attention_count == 0) == (k <= 3),
                "segment 0 has attention layers exactly from K=4" + at);
        // The PLE layer 1 runs on segment 0 for every K >= 2; K = 1 leaves segment 0 only layer 0.
        require(map.segment_of(1) == (k == 1 ? 1U : 0U) && a.ple == (k != 1) && b.ple == (k == 1) &&
                    &map.ple() == &map[k == 1 ? 1 : 0],
                "the PLE on segment_of(1)" + at);
        require(a.embed && !b.embed && &map.first() == &a,
                "the embedding on the first segment" + at);
        require(!a.head && !a.mtp && !a.kv_primary && b.head && b.mtp && b.kv_primary &&
                    &map.last() == &b,
                "the head, the MTP drafter and the KV allocator on the last segment" + at);
        require(a.kv_layers == attention_below(k) && b.kv_layers == 13 - attention_below(k),
                "segment 1 keeps the MTP layer's KV" + at);
        require(!a.shares_device && !b.shares_device && a.frame_share == 1.0 &&
                    b.frame_share == 1.0,
                "each device's own frames" + at);
        check_slices(c, map);
    }
    // The design's placements: K=24 puts 6 of the 13 KV layers on segment 0, K=28 puts 7.
    const std::uint32_t k24[] = {24, kLayers}, k28[] = {28, kLayers};
    require(SegmentMap::contiguous(c, true, devices, k24)[0].kv_layers == 6,
            "K=24: 6 KV layers on segment 0");
    require(SegmentMap::contiguous(c, true, devices, k28)[0].kv_layers == 7,
            "K=28: 7 KV layers on segment 0");
    // Without the drafter the last segment has no MTP layer.
    const SegmentMap plain = SegmentMap::contiguous(c, false, devices, k24);
    require(!plain[1].mtp && plain[1].head && plain[1].kv_layers == 6,
            "no MTP KV layer without the drafter");
}

void test_three_segments() {
    const TextConfig c         = flash_next();
    const int devices[]        = {2, 0, 1};
    const std::uint32_t ends[] = {10, 30, kLayers};
    const SegmentMap map       = SegmentMap::contiguous(c, true, devices, ends);
    require(map.size() == 3 && map[0].device == 2 && map[1].device == 0 && map[2].device == 1,
            "devices follow the given order, not their ordinals");
    require(map[0].embed && map[0].ple && !map[1].embed && !map[1].ple && !map[1].head &&
                map[2].head,
            "the middle segment owns only layers");
    require(map[0].attention_count == 2 && map[1].attention_base == 2 &&
                map[1].attention_count == 5 && map[2].attention_base == 7 &&
                map[2].attention_count == 5,
            "attention 3, 7 | 11..27 | 31..47");
    require(map[1].kv_layers == 5 && map[2].kv_layers == 6,
            "only the last segment adds the MTP layer");
    check_slices(c, map);
}

// --devices 0,0: two segments on one ordinal split its frames by their layers.
void test_same_device() {
    const TextConfig c = flash_next();
    {
        const int devices[] = {0, 0};
        for (const std::uint32_t k : {1U, 24U, 47U}) {
            const std::uint32_t ends[] = {k, kLayers};
            const SegmentMap map       = SegmentMap::contiguous(c, true, devices, ends);
            require(map.same_device() && map[0].shares_device && map[1].shares_device,
                    "one ordinal, two segments");
            require(map[0].frame_share == double(k) / kLayers &&
                        map[1].frame_share == double(kLayers - k) / kLayers,
                    "frame shares by layers (K=" + std::to_string(k) + ")");
            check_slices(c, map);
        }
    }
    {
        // Segments 0 and 2 share ordinal 0 (12 + 12 layers); segment 1 has ordinal 1 to itself.
        const int devices[]        = {0, 1, 0};
        const std::uint32_t ends[] = {12, 36, kLayers};
        const SegmentMap map       = SegmentMap::contiguous(c, true, devices, ends);
        require(map.same_device(), "an ordinal holds two segments");
        require(map[0].shares_device && map[2].shares_device && map[0].frame_share == 0.5 &&
                    map[2].frame_share == 0.5,
                "the sharers split their ordinal's frames");
        require(!map[1].shares_device && map[1].frame_share == 1.0,
                "a lone segment keeps all of its device");
    }
    {
        const int devices[]        = {1, 1, 1};
        const std::uint32_t ends[] = {6, 18, kLayers};
        const SegmentMap map       = SegmentMap::contiguous(c, true, devices, ends);
        require(map[0].frame_share == 6.0 / 48 && map[1].frame_share == 12.0 / 48 &&
                    map[2].frame_share == 30.0 / 48,
                "three sharers by layers");
    }
}

void test_refused() {
    const TextConfig c = flash_next();
    const auto refuse  = [&](std::vector<int> devices, std::vector<std::uint32_t> ends,
                            const std::string& why) {
        require_throws<std::invalid_argument>(
            [&] { (void)SegmentMap::contiguous(c, true, devices, ends); }, why);
    };
    refuse({}, {}, "no segment");
    refuse({0, 1}, {kLayers}, "more devices than ends");
    refuse({0}, {24, kLayers}, "more ends than devices");
    refuse({0, 1}, {0, kLayers}, "an empty first segment");
    refuse({0, 1}, {24, 24}, "an empty last segment");
    refuse({0, 1, 2}, {24, 24, kLayers}, "an empty middle segment");
    refuse({0, 1}, {30, 24}, "descending ends");
    refuse({0, 1}, {24, 47}, "a last end short of the layer count");
    refuse({0, 1}, {24, 49}, "an end past the layer count");
    refuse({0, -1}, {24, kLayers}, "a negative ordinal");

    // Segment indices are u8 tags with kSharedSegment reserved: 254 segments, not 255.
    const TextConfig deep = flash_next(kSharedSegment);
    std::vector<std::uint32_t> deep_ends(kSharedSegment);
    for (std::uint32_t i = 0; i < kSharedSegment; ++i) { deep_ends[i] = i + 1; }
    require_throws<std::invalid_argument>(
        [&] {
            (void)SegmentMap::contiguous(deep, true, std::vector<int>(kSharedSegment, 0),
                                         deep_ends);
        },
        "as many segments as the shared tag");
    deep_ends.erase(deep_ends.begin());
    require(SegmentMap::contiguous(deep, true, std::vector<int>(kSharedSegment - 1, 0), deep_ends)
                    .size() == 254,
            "254 segments");

    const int devices[]        = {0, 1};
    const std::uint32_t ends[] = {24, kLayers};
    TextConfig bad             = c;
    std::swap(bad.compact_layer_indices[0], bad.compact_layer_indices[1]);
    require_throws<std::invalid_argument>(
        [&] { (void)SegmentMap::contiguous(bad, true, devices, ends); },
        "compact indices out of layer order");
    bad = c;
    bad.compact_layer_indices.pop_back();
    require_throws<std::invalid_argument>([&] { (void)SegmentMap::single(bad, true, 0); },
                                          "a layer without an index");
    bad = c;
    ++bad.gdn_layers;
    require_throws<std::invalid_argument>([&] { (void)SegmentMap::single(bad, true, 0); },
                                          "kind counts that disagree with the kinds");
    bad           = c;
    bad.ple.layer = kLayers;
    require_throws<std::invalid_argument>([&] { (void)SegmentMap::single(bad, true, 0); },
                                          "a PLE layer past the model");
}

} // namespace

int main() {
    try {
        test_single();
        test_two_segments();
        test_three_segments();
        test_same_device();
        test_refused();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
    std::printf("qwen4_exp segment map checks passed\n");
    return 0;
}
