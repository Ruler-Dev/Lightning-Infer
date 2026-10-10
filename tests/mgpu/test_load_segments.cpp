// Segment-aware load bindings and per-segment Parameters (multi-GPU WP7) on the real artifact
// (INFERNIX_QWEN4_ARTIFACT; skips with 77 without it, and after the plan checks without a GPU),
// with the MTP drafter, the proposal head and the Vision tower bound when the artifact has them:
//  - plan: segment ends that do not ascend to the layer count are refused; {} and {L} give the same
//    one-segment plan, every device object in segment 0; with {24, L} every device object of
//    text/layers/i lies in segment_of(i), the head, the final mixer, the MTP drafter and the
//    proposal head in the last segment and the Vision tower in the first; the pinned placements
//    are the one-segment plan's, the token embedding tagged shared and each expert bank tagged
//    with its layer's segment; the per-part device bytes of each segment equal the sum of its
//    objects' bytes and alignment padding, which bounds its arena;
//  - materialization on two contexts of device 0, or with --second-device D segment 1 on device D
//    (the expert banks streamed, so only the embedding is pinned): one device for a two-segment
//    plan is refused without consuming it; each arena lies in its device's memory; every bound
//    weight of layer i points into arena segment_of(i), the head's into the last;
//  - Parameters, each built under its segment's device: [0,24) prepares layers 0-23 and no head,
//    MTP or draft head, and has null payloads, no bank and no scales elsewhere; [24,L) the head,
//    the MTP drafter and layers 24-47; every prepared operand lies in its segment's arena;
//    device_scales point into each segment's own scale buffer on its device, at its local layer,
//    and hold the banks' scales; a range across segments and roles that disagree with the range
//    are refused.
// With --second-device D it skips (77) when device D does not exist.

#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "core/device.h"
#include "core/device_guard.h"
#include "models/qwen4_exp/execution/parameters.h"
#include "models/qwen4_exp/load.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace {

using namespace infernix;
namespace q4 = models::qwen4_exp;
namespace ex = models::qwen4_exp::execution;
using q4::LayerRange;

constexpr std::uint32_t kSplit = 24; // segment 0 = [0, 24), segment 1 = [24, L)

int g_failures = 0;

void expect(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << '\n';
        ++g_failures;
    }
}

template <class Exception, class Call>
bool throws(Call&& call) {
    try {
        call();
    } catch (const Exception&) { return true; } catch (...) {
        return false;
    }
    return false;
}

// The segment a parameter's device objects belong to, by its name; nullopt for a parameter that is
// never device-resident here (the pinned embedding, the expert banks, their input scales).
std::optional<std::uint32_t> expected_segment(std::string_view name, std::uint32_t last) {
    constexpr std::string_view layers = "text/layers/";
    if (name == "text/token_embedding" || name.ends_with("/moe/experts") ||
        name.ends_with("/moe/expert_input_scales")) {
        return std::nullopt;
    }
    if (name.starts_with(layers)) {
        const std::uint32_t layer =
            static_cast<std::uint32_t>(std::stoul(std::string(name.substr(layers.size()))));
        return layer < kSplit ? 0U : last;
    }
    if (name.starts_with("vision/")) { return 0U; }
    // text/output_head, text/final_mixer/*, mtp/*, proposal/*.
    return last;
}

std::optional<std::uint32_t> layer_of(std::string_view name) {
    constexpr std::string_view layers = "text/layers/";
    if (!name.starts_with(layers)) { return std::nullopt; }
    return static_cast<std::uint32_t>(std::stoul(std::string(name.substr(layers.size()))));
}

bool inside(const void* pointer, DeviceSpan arena) {
    const auto* p = static_cast<const std::byte*>(pointer);
    const auto* a = static_cast<const std::byte*>(arena.data);
    return p != nullptr && a != nullptr && p >= a && p < a + arena.bytes;
}

// The load options of the run: MTP, the proposal head and the Vision tower (on the device) when the
// artifact has them.
models::LoadOptions base_options(const artifact::Reader& reader) {
    models::LoadOptions options;
    const auto& directory = reader.directory();
    if (directory.components.contains("mtp")) {
        options.speculative  = SpeculativeBackend::Mtp;
        const auto& proposal = directory.component("text").proposal;
        if (proposal && proposal->indexed && proposal->rows != 0) {
            options.proposal_head = ProposalHead::Optimized;
        }
    }
    options.vision         = directory.components.contains("vision");
    options.vision_offload = false;
    return options;
}

// The object handles each bound parameter name reads.
std::map<std::size_t, std::string> object_names(const artifact::Reader& reader) {
    std::map<std::size_t, std::string> out;
    for (const auto& [name, binding] : reader.directory().bindings) {
        for (const auto& part : binding.parts) {
            // A tied head shares the embedding's object: the device demand is the head's.
            auto [it, inserted] = out.emplace(part.object.index, name);
            if (!inserted && it->second == "text/token_embedding") { it->second = name; }
        }
    }
    return out;
}

void check_plans(const artifact::Reader& reader) {
    const models::LoadOptions base = base_options(reader);
    auto one                       = q4::plan_load(reader, base);
    const std::uint32_t layers     = one.config().text.num_hidden_layers;
    expect(layers > kSplit, "the model has more than 24 layers");

    // Malformed ends.
    for (const std::vector<std::uint32_t>& ends :
         {std::vector<std::uint32_t>{0, layers}, std::vector<std::uint32_t>{kSplit},
          std::vector<std::uint32_t>{30, kSplit, layers},
          std::vector<std::uint32_t>{kSplit, layers + 1},
          std::vector<std::uint32_t>{kSplit, kSplit, layers}}) {
        auto options         = base;
        options.segment_ends = ends;
        expect(throws<std::invalid_argument>([&] { (void)q4::plan_load(reader, options); }),
               "malformed segment ends are refused (" + std::to_string(ends.size()) + " ends)");
    }

    // One segment: {} and {L} agree, all in segment 0.
    auto whole_options         = base;
    whole_options.segment_ends = {layers};
    auto whole                 = q4::plan_load(reader, whole_options);
    const auto& a              = one.materialization();
    const auto& b              = whole.materialization();
    expect(one.segment_count() == 1 && whole.segment_count() == 1, "one segment without a split");
    expect(a.device_segments.size() == 1 && one.device_bytes() == a.device_capacity_bytes &&
               one.device_bytes(0) == a.device_capacity_bytes,
           "one segment: device_bytes() is the single arena");
    expect(throws<std::out_of_range>([&] { (void)one.device_bytes(1); }), "no device segment 1");
    bool same = a.device_objects.size() == b.device_objects.size();
    for (std::size_t i = 0; same && i < a.device_objects.size(); ++i) {
        const auto &x = a.device_objects[i], &y = b.device_objects[i];
        same = x.object == y.object && x.offset == y.offset && x.bytes == y.bytes &&
               x.alignment == y.alignment && x.segment == 0 && y.segment == 0;
    }
    expect(same, "{} and {L} give the same device placements, all in segment 0");

    // Two segments.
    auto split_options         = base;
    split_options.segment_ends = {kSplit, layers};
    auto split                 = q4::plan_load(reader, split_options);
    const auto& s              = split.materialization();
    expect(split.segment_count() == 2 && s.device_segments.size() == 2, "two segments");
    expect(split.device_bytes() == split.device_bytes(0) + split.device_bytes(1),
           "device_bytes() sums the arenas");

    const auto names         = object_names(reader);
    std::uint64_t checked[2] = {0, 0};
    std::uint64_t parts[2]   = {0, 0}; // bytes + alignment - 1 of each segment's objects
    for (const auto& placement : s.device_objects) {
        const auto found = names.find(placement.object.index);
        if (found == names.end()) {
            expect(false, "every device object is bound by a parameter");
            continue;
        }
        const auto want = expected_segment(found->second, 1);
        expect(want.has_value() && placement.segment == *want,
               found->second + ": device segment " + std::to_string(placement.segment));
        if (placement.segment < 2) {
            ++checked[placement.segment];
            parts[placement.segment] += placement.bytes + placement.alignment - 1;
        }
    }
    expect(checked[0] > 0 && checked[1] > 0, "both segments hold device objects");

    // The pinned space is the one-segment plan's; the tags name the reading segments.
    bool pinned_same = a.pinned_objects.size() == s.pinned_objects.size();
    for (std::size_t i = 0; pinned_same && i < a.pinned_objects.size(); ++i) {
        const auto &x = a.pinned_objects[i], &y = s.pinned_objects[i];
        pinned_same = x.object == y.object && x.offset == y.offset && x.bytes == y.bytes;
    }
    expect(pinned_same, "the split keeps the pinned placements");
    std::map<std::size_t, std::uint8_t> pinned_segment;
    for (const auto& placement : s.pinned_objects) {
        pinned_segment.emplace(placement.object.index, placement.segment);
    }
    const auto pinned_tag = [&](const std::string& name) -> std::optional<std::uint8_t> {
        const auto& binding = reader.directory().bindings.at(name);
        const auto found    = pinned_segment.find(binding.parts.front().object.index);
        if (found == pinned_segment.end()) { return std::nullopt; }
        return found->second;
    };
    expect(pinned_tag("text/token_embedding") == artifact::kSharedSegment,
           "the token embedding is pinned and shared");
    for (std::uint32_t l = 0; l < layers; ++l) {
        const std::string bank = "text/layers/" + std::to_string(l) + "/moe/experts";
        expect(pinned_tag(bank) == std::uint8_t(l < kSplit ? 0 : 1),
               bank + ": pinned for its layer's segment");
    }

    // The parts' device bytes: each segment's sum is exactly its objects' bytes plus padding.
    const auto per_layer = split.layer_device_bytes();
    expect(per_layer.size() == layers, "one device byte count per layer");
    std::uint64_t sums[2] = {split.vision_device_bytes(),
                             split.head_device_bytes() + split.mtp_device_bytes()};
    for (std::uint32_t l = 0; l < per_layer.size(); ++l) {
        expect(per_layer[l] > 0, "layer " + std::to_string(l) + " has device weights");
        sums[l < kSplit ? 0 : 1] += per_layer[l];
    }
    for (std::uint32_t seg = 0; seg < 2; ++seg) {
        expect(sums[seg] == parts[seg],
               "segment " + std::to_string(seg) + ": the parts' bytes cover its objects");
        expect(split.device_bytes(seg) <= sums[seg],
               "segment " + std::to_string(seg) + ": the arena fits its parts");
    }
    expect(split.head_device_bytes() > 0, "the head has device weights");
    expect((split.mtp_device_bytes() > 0) == split.config().mtp,
           "MTP device bytes exactly with the drafter");
    expect((split.vision_device_bytes() > 0) == split.config().vision.has_value(),
           "Vision device bytes exactly with the tower");
    // The per-part counts do not depend on the split.
    expect(one.layer_device_bytes() == per_layer &&
               one.head_device_bytes() == split.head_device_bytes() &&
               one.mtp_device_bytes() == split.mtp_device_bytes(),
           "the parts' bytes are the same without the split");
    std::cout << "plan: " << layers << " layers; segment 0 " << (split.device_bytes(0) >> 20)
              << " MiB, segment 1 " << (split.device_bytes(1) >> 20) << " MiB (one segment "
              << (one.device_bytes() >> 20) << " MiB)\n";
}

void check_operands(const ex::Parameters& p, std::uint32_t segment, DeviceSpan arena,
                    const q4::Model& model, int device) {
    const auto& c          = model.config().text;
    const LayerRange range = p.layer_range();
    const std::string tag  = "segment " + std::to_string(segment) + ": ";
    expect(p.layers.size() == c.num_hidden_layers, tag + "layers are indexed by decoder layer");
    expect(p.device() == device, tag + "built on its segment's device");
    expect(p.token_embedding.payload != nullptr, tag + "the token embedding is bound");

    const std::size_t per_layer = sizeof(ops::offloaded_moe::ExpertScales) * c.moe.experts;
    const DeviceSpan scales     = p.expert_scales();
    expect(scales.data != nullptr && scales.bytes == per_layer * range.size(),
           tag + "one scale row per own layer");
    std::vector<std::byte> host(scales.bytes);
    {
        const DeviceGuard on(device);
        CUDA_CHECK(cudaMemcpy(host.data(), scales.data, scales.bytes, cudaMemcpyDeviceToHost));
    }
    const auto banks = model.expert_banks();

    for (std::uint32_t l = 0; l < c.num_hidden_layers; ++l) {
        const auto& layer = p.layers[l];
        const bool gdn    = c.layer_types[l] == q4::MixerKind::Gdn;
        const void* project =
            std::visit([](const auto& m) { return m.projection.weight.payload; }, layer.mixer);
        const std::string at = tag + "layer " + std::to_string(l) + ": ";
        expect(std::holds_alternative<ex::GdnParameters>(layer.mixer) == gdn,
               at + "the mixer of its kind");
        if (!range.contains(l)) {
            expect(layer.attn_hc.down.weight.payload == nullptr &&
                       layer.attn_hc.norm.data == nullptr &&
                       layer.mlp_hc.up.weight.payload == nullptr && project == nullptr &&
                       layer.moe.router.data == nullptr &&
                       layer.moe.shared_down.weight.payload == nullptr,
                   at + "null payloads outside the range");
            expect(layer.moe.bank == nullptr && layer.moe.device_scales == nullptr && !layer.ple,
                   at + "no bank, scales or PLE outside the range");
            continue;
        }
        expect(inside(layer.attn_hc.down.weight.payload, arena) &&
                   inside(layer.attn_hc.norm.data, arena) &&
                   inside(layer.mlp_hc.up.weight.payload, arena) && inside(project, arena) &&
                   inside(layer.moe.router.data, arena) &&
                   inside(layer.moe.shared_gate_up.weight.payload, arena),
               at + "operands in the segment's arena");
        expect(layer.moe.bank == &banks[l], at + "its expert bank");
        const auto* expected =
            static_cast<const std::byte*>(scales.data) + per_layer * (l - range.begin);
        expect(static_cast<const void*>(layer.moe.device_scales) == expected,
               at + "scales at its local row");
        expect(std::memcmp(host.data() + per_layer * (l - range.begin), banks[l].scales.data(),
                           per_layer) == 0,
               at + "the scales are the bank's");
        expect(layer.ple.has_value() == (l == c.ple.layer), at + "PLE exactly on the PLE layer");
        if (layer.ple) {
            expect(inside(layer.ple->projection.weight.payload, arena), at + "PLE in the arena");
        }
    }
    cudaPointerAttributes attributes{};
    CUDA_CHECK(cudaPointerGetAttributes(&attributes, scales.data));
    expect(attributes.type == cudaMemoryTypeDevice && attributes.device == device,
           tag + "scales in its device's memory");

    if (p.role().head) {
        const void* head = p.output_head_q8 ? p.output_head_q8->weight.payload : p.output_head.data;
        expect(inside(head, arena) && inside(p.final_mixer.norm.data, arena) &&
                   inside(p.final_mixer.down.weight.payload, arena),
               tag + "the head in the arena");
        expect(p.mtp.has_value() == model.config().mtp, tag + "the MTP drafter exactly when bound");
        if (p.mtp) {
            expect(inside(p.mtp->embedding_projection.weight.payload, arena) &&
                       inside(p.mtp->attention.projection.weight.payload, arena) &&
                       inside(p.mtp->experts_gate_up.payload, arena),
                   tag + "the MTP drafter in the arena");
        }
        expect(p.draft_head.rows.has_value() == (model.config().proposal_rows != 0),
               tag + "the proposal head");
        if (p.draft_head.rows) {
            expect(inside(p.draft_head.rows->weight.payload, arena) &&
                       inside(p.draft_head.token_ids.data, arena),
                   tag + "the proposal head in the arena");
        }
    } else {
        expect(p.output_head.data == nullptr && !p.output_head_q8 &&
                   p.final_mixer.norm.data == nullptr &&
                   p.final_mixer.down.weight.payload == nullptr && !p.mtp && !p.draft_head.rows,
               tag + "no head, MTP or draft head");
    }
    expect(p.vision.has_value() == (p.role().vision && model.config().vision.has_value()),
           tag + "the Vision tower");
    if (p.vision) {
        expect(inside(p.vision->patch_embedding.weight.payload, arena) &&
                   inside(p.vision->merger_fc2.weight.payload, arena),
               tag + "the Vision tower in the arena");
    }
}

// Segment 0 on device 0, segment 1 on `second_device` (0: two contexts of one device).
void check_model(const artifact::Reader& reader, int second_device) {
    DeviceContext first(0), second(second_device);
    auto options           = base_options(reader);
    options.stream_experts = true; // pins only the embedding
    const std::uint32_t layers =
        static_cast<std::uint32_t>(q4::plan_load(reader, options).config().text.num_hidden_layers);
    options.segment_ends           = {kSplit, layers};
    auto plan                      = q4::plan_load(reader, options);
    const std::uint64_t planned[2] = {plan.device_bytes(0), plan.device_bytes(1)};

    DeviceContext* const one[] = {&first};
    expect(
        throws<std::invalid_argument>([&] { (void)q4::materialize_model(std::move(plan), one); }),
        "a two-segment plan needs two devices");
    DeviceContext* const both[] = {&first, &second};
    const auto model            = q4::materialize_model(std::move(plan), both);
    first.synchronize();
    second.synchronize();

    const auto& segments = model->segments();
    expect(segments.size() == 2 && segments[0].layers.begin == 0 &&
               segments[0].layers.end == kSplit && segments[1].layers.begin == kSplit &&
               segments[1].layers.end == layers,
           "the model's segments are the plan's");
    const int device_of[2] = {0, second_device};
    expect(segments[0].device == 0 && segments[1].device == second_device &&
               segments.same_device() == (second_device == 0),
           "the segments on devices 0 and " + std::to_string(second_device));
    const DeviceSpan arena[2] = {model->segment_arena(0), model->segment_arena(1)};
    for (std::uint32_t s = 0; s < 2; ++s) {
        expect(arena[s].data != nullptr && arena[s].bytes == planned[s],
               "segment " + std::to_string(s) + "'s arena is the planned size");
        cudaPointerAttributes attributes{};
        CUDA_CHECK(cudaPointerGetAttributes(&attributes, arena[s].data));
        expect(attributes.type == cudaMemoryTypeDevice && attributes.device == device_of[s],
               "segment " + std::to_string(s) + "'s arena in device " +
                   std::to_string(device_of[s]) + "'s memory");
    }
    const auto* a0 = static_cast<const std::byte*>(arena[0].data);
    const auto* a1 = static_cast<const std::byte*>(arena[1].data);
    expect(a0 + arena[0].bytes <= a1 || a1 + arena[1].bytes <= a0, "the arenas are disjoint");

    // Every bound weight of layer i points into arena segment_of(i), the head's into the last.
    std::uint32_t bound[2] = {0, 0};
    for (const auto& weight : model->weight_data()) {
        const auto want = expected_segment(weight.name, 1);
        if (!want) { continue; }
        bool in = !weight.view.parts.empty();
        for (const auto& part : weight.view.parts) {
            in = in && inside(part.parent->data, arena[*want]);
        }
        expect(in, weight.name + ": in segment " + std::to_string(*want) + "'s arena");
        if (const auto layer = layer_of(weight.name)) {
            expect(segments.segment_of(*layer) == *want, weight.name + ": segment_of agrees");
        }
        ++bound[*want];
    }
    expect(bound[0] > 0 && bound[1] > 0, "both segments hold bound weights");

    // Ranges and roles a segment's Parameters refuse.
    const DeviceGuard guard(0);
    const auto refused = [&](LayerRange range, ex::SegmentRole role, const std::string& what) {
        expect(throws<std::invalid_argument>([&] { const ex::Parameters p(*model, range, role); }),
               what + " is refused");
    };
    refused({0, layers}, {}, "the whole model across two segments");
    refused({20, 30}, {.embed = false, .head = false, .vision = false}, "a range across segments");
    refused({0, 0}, {.embed = true, .head = false, .vision = true}, "an empty range");
    refused({kSplit, layers + 1}, {.embed = false, .head = true, .vision = false},
            "a range past the last layer");
    refused({0, kSplit}, {.embed = true, .head = true, .vision = true},
            "a head without the last layer");
    refused({kSplit, layers}, {.embed = false, .head = false, .vision = false},
            "the last layer without the head");
    refused({kSplit, layers}, {.embed = true, .head = true, .vision = false},
            "embedding without layer 0");
    refused({kSplit, layers}, {.embed = false, .head = true, .vision = true},
            "the tower without embedding");
    refused({0, kSplit}, {.embed = false, .head = false, .vision = false},
            "layer 0 without embedding");

    const ex::Parameters head_less(*model, segments[0].layers, ex::segment_role(segments[0]));
    // Under its segment's device, as the Program builds each segment's Parameters.
    const auto built = [&] {
        const DeviceGuard on(second_device);
        return std::make_unique<const ex::Parameters>(*model, segments[1].layers,
                                                      ex::segment_role(segments[1]));
    }();
    const ex::Parameters& head = *built;
    expect(head_less.role().embed && !head_less.role().head && head.role().head &&
               !head.role().embed,
           "segment_role: the first embeds, the last holds the head");
    check_operands(head_less, 0, arena[0], *model, 0);
    check_operands(head, 1, arena[1], *model, second_device);
    expect(head_less.token_embedding.payload == head.token_embedding.payload,
           "one shared token embedding");
    expect(head_less.expert_scales().data != head.expert_scales().data,
           "each segment has its own scales");
    // A range inside one segment is allowed (its weights still share a device).
    const ex::Parameters part(*model, {0, kSplit / 2},
                              {.embed = true, .head = false, .vision = false});
    expect(part.layer_range().end == kSplit / 2 &&
               part.expert_scales().bytes == head_less.expert_scales().bytes / 2 &&
               part.layers[kSplit / 2 - 1].moe.bank != nullptr &&
               part.layers[kSplit / 2].moe.bank == nullptr && !part.vision,
           "a range inside one segment");
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

int main(int argc, char** argv) {
    int second_device = 0; // --second-device D: segment 1's device
    for (int i = 1; i < argc; ++i) {
        if (std::string_view(argv[i]) == "--second-device" && i + 1 < argc) {
            second_device = std::atoi(argv[++i]);
        } else {
            std::cerr << "usage: infernix_qwen4_exp_load_segments_test [--second-device D]\n";
            return 2;
        }
    }
    const char* path = std::getenv("INFERNIX_QWEN4_ARTIFACT");
    if (path == nullptr || !std::filesystem::exists(path)) {
        std::cout << "SKIP: INFERNIX_QWEN4_ARTIFACT is not set\n";
        return 77;
    }
    try {
        const artifact::Reader reader(path);
        if (!q4::is_qwen4_exp(reader)) {
            std::cerr << "FAIL: INFERNIX_QWEN4_ARTIFACT is not a Qwen4Exp artifact\n";
            return 1;
        }
        check_plans(reader);
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
            (void)cudaGetLastError();
            std::cout << (g_failures == 0 ? "plan checks passed; " : "")
                      << "SKIP: no CUDA device\n";
            return g_failures == 0 ? 77 : 1;
        }
        if (second_device >= count || second_device < 0) {
            std::cout << (g_failures == 0 ? "plan checks passed; " : "") << "SKIP: no CUDA device "
                      << second_device << '\n';
            return g_failures == 0 ? 77 : 1;
        }
        check_model(reader, second_device);
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    if (g_failures != 0) {
        std::cerr << g_failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "load segments: all checks passed\n";
    return 0;
}
