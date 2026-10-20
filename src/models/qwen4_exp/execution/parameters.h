#pragma once

// Native execution parameters of a loaded Qwen4Exp model: one Weight per fused projection (the
// converter packs every projection group that shares an input into one parent), BF16 vectors for
// norms and convolutions, and each MoE layer's expert source.
//
// Layer split: each segment has its own Parameters over its layer range, built on its device (the
// expert scales are its only allocation); the head segment's also hold the head, the MTP drafter
// and the draft head, the first segment's the Vision tower. An offloaded tower's parameters are
// views of pinned host memory every GPU reads, so the Program may run it on another segment (the
// one with the fastest link, design §19.3.2 "On a layer split"). Tables stay indexed by decoder
// layer.

#include "core/arena.h"
#include "core/tensor.h"
#include "models/qwen3_5/execution/parameters.h"
#include "models/qwen4_exp/model.h"
#include "models/qwen4_exp/segment_map.h"
#include "infernix/ops/offloaded_sparse_moe.h"
#include "infernix/ops/rope.h"
#include "infernix/ops/weight_input.h"

#include <memory>
#include <optional>
#include <variant>
#include <vector>

namespace infernix::models::qwen4_exp::execution {

using LinearParameters = ops::SingleProjectionWeight;

struct HyperConnectionParameters {
    Tensor norm;           // BF16 [S*H]
    LinearParameters down; // [rank (+ S), S*H]: down rows, then the injection rows when combining
    LinearParameters up;   // [S*H, rank]
    bool combine = false;
};

struct AttentionParameters {
    LinearParameters projection; // [q | gate | k | v | index q | index k, H]
    Tensor query_norm, key_norm, index_query_norm, index_key_norm;
    LinearParameters output;
};

struct GdnParameters {
    LinearParameters projection; // [q | k | v | z, H]
    LinearParameters control;    // [a | b, H]
    Tensor a_log, dt_bias;       // FP32 [value heads]
    Tensor convolution;          // BF16 [C, K]
    Tensor norm;                 // BF16 [value head dim]
    LinearParameters output;
};

struct MoeParameters {
    Tensor router;                   // BF16 [H, E]: one expert row per column (FP32 logits)
    Tensor shared_score;             // BF16 [H, 1]: the shared-expert gate row
    LinearParameters shared_gate_up; // [2 * I_shared, H]: gate rows, then up rows
    LinearParameters shared_down;    // [H, I_shared]
    const ExpertBank* bank = nullptr;
    const ops::offloaded_moe::ExpertScales* device_scales = nullptr; // [E]
};

struct PleParameters {
    LinearParameters projection; // [key (S*H) | value (H), embed]
    Tensor key_norm, query_norm, conv_norm; // BF16 [S*H]
    Tensor convolution;                     // BF16 [S*H, K]
    Tensor ngram_scale;                     // BF16 [1]
};

// The MTP drafter (design §11.2).
struct MtpParameters {
    Tensor embedding_norm, hidden_norm; // BF16 [H], [S*H]
    LinearParameters embedding_projection, hidden_projection;
    HyperConnectionParameters attn_hc, mlp_hc, final_mixer;
    AttentionParameters attention;
    Tensor router, shared_score; // BF16, as MoeParameters
    LinearParameters shared_gate_up, shared_down;
    Weight experts_gate_up; // [E * 2I, H]: expert e's gate rows, then its up rows
    Weight experts_down;    // [E * H, I]
};

// One DFlash2 layer (docs/maintainer/dflash.md "DFlash2 dynamic convolution").
struct DFlash2LayerParameters {
    Tensor input_norm, post_attention_norm; // BF16 [H]
    Weight query_key_value;                 // Q8 [q | k | v, H] for ops::attn_input_proj
    LinearParameters context_key, context_value; // [kv, H] (the K/V words, shared)
    Tensor query_norm, key_norm;                 // BF16 [head_dim]
    LinearParameters output;                     // [H, q]
    LinearParameters gate_up, down;              // [2I, H], [H, I]
    Tensor attention_base, mlp_base;             // BF16 [H, taps, sides]
    LinearParameters attention_projection, mlp_projection; // BF16 [2 * taps * H / group, H]
};

// The DFlash2 drafter of a companion artifact (--draft-model).
struct DFlash2Parameters {
    ops::PreparedRope rope;              // full-head 1-D, the drafter's theta
    LinearParameters feature_projection; // [H, taps * H]
    Tensor context_norm, final_norm;     // BF16 [H]
    std::vector<DFlash2LayerParameters> layers;
    LinearParameters selector_hidden;                 // [rank, H]
    Tensor predecessor_codebook, successor_codebook;  // BF16 [rank, vocab]
};

// The draft head: the text head, or the proposal head's rows and their token ids.
struct DraftHeadParameters {
    std::optional<LinearParameters> rows; // the proposal head (q4/q8 [rows, H]); empty = text head
    Tensor token_ids;                     // I32 [rows]
};

struct BlockParameters {
    HyperConnectionParameters attn_hc, mlp_hc;
    std::variant<AttentionParameters, GdnParameters> mixer;
    MoeParameters moe;
    std::optional<PleParameters> ple;
};

// The model's single components a segment's Parameters prepare besides its layers. embed: the
// segment embeds the prompt, so it starts at layer 0; head: the final mixer, the output head, the
// MTP drafter and the draft head, so it ends at the last layer; vision: the Vision tower (the
// load places it with the first segment). The token embedding is bound whatever the role (pinned,
// read in place): the head segment's MTP drafter embeds its tokens with it too.
struct SegmentRole {
    bool embed  = true;
    bool head   = true;
    bool vision = true;
};

// The role of a segment of the map: the first embeds and holds the tower, the last the head.
[[nodiscard]] constexpr SegmentRole segment_role(const SegmentSpec& segment) noexcept {
    return {.embed = segment.embed, .head = segment.head, .vision = segment.embed};
}

class Parameters {
public:
    // The whole model: Parameters(model, {0, L}, {}).
    explicit Parameters(const Model& model);
    // Decoder layers `layers` and the components of `role`. The range lies inside one of the
    // model's device segments (Model::segments()), so its weights share one device; its expert
    // scales are allocated on the current device, so it is constructed under that segment's
    // DeviceGuard. Throws std::invalid_argument for an empty range, one past the last layer or
    // across segments, and a role that disagrees with its range: embed exactly when it starts at
    // layer 0, head exactly when it ends at the last layer, vision only with embed.
    Parameters(const Model& model, LayerRange layers, SegmentRole role);
    ~Parameters();
    Parameters(const Parameters&)            = delete;
    Parameters& operator=(const Parameters&) = delete;

    [[nodiscard]] LayerRange layer_range() const noexcept { return range_; }

    [[nodiscard]] SegmentRole role() const noexcept { return role_; }

    // The device current at construction, which holds the expert scales.
    [[nodiscard]] int device() const noexcept { return device_; }

    // The device copy of the range's ExpertScales, [layer - begin][E].
    [[nodiscard]] DeviceSpan expert_scales() const noexcept { return {scales_.p, scales_.bytes}; }

    const Model& model;
    Weight token_embedding; // BF16 rows in pinned host memory, read zero-copy
    Tensor output_head; // BF16 [H, V]: one vocabulary row per column (FP32 logits)
    // Recipe B's 8-bit head (q8_g32_fp16 [V, H], FP32 logits); output_head is then empty.
    std::optional<LinearParameters> output_head_q8;
    HyperConnectionParameters final_mixer;
    // Indexed by decoder layer (size L). A layer outside the range is default: null payloads, no
    // bank, scales or PLE, and the mixer alternative of its kind.
    std::vector<BlockParameters> layers;
    std::optional<MtpParameters> mtp;
    std::optional<DFlash2Parameters> dflash2;
    DraftHeadParameters draft_head;
    std::optional<qwen3_5::execution::VisionParameters> vision; // --vision (shared tower types)

private:
    LayerRange range_;
    SegmentRole role_;
    int device_ = 0;
    // Device copy of the range's ExpertScales, [layers in range][E].
    DeviceBuffer scales_;
};

} // namespace infernix::models::qwen4_exp::execution
