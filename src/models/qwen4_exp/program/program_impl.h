#pragma once

#include "models/qwen4_exp/program/program.h"

#include "core/arena.h"
#include "core/cuda_vmm.h"
#include "core/vmm_arena.h"
#include "core/decode_graph.h"
#include "core/device.h"
#include "core/device_guard.h"
#include "core/gdn_replay_records.h"
#include "core/handoff.h"
#include "core/layout.h"
#include "core/link_probe.h"
#include "core/linear_attention_state.h"
#include "core/nvtx.h"
#include "core/paged_kv_cache.h"
#include "core/paged_kv_storage.h"
#include "core/tensor.h"
#include "core/vram_budget.h"
#include "core/weight_view.h"
#include "models/qwen3_5/frontend/prepared_prompt.h"
#include "models/qwen3_5/program/ngram_proposer.h"
#include "models/qwen4_exp/execution/forward.h"
#include "models/qwen4_exp/execution/parameters.h"
#include "models/qwen4_exp/frontend/ngram_hash.h"
#include "models/qwen4_exp/memory_plan.h"
#include "models/qwen4_exp/program/expert_residency.h"
#include "models/qwen4_exp/program/device_segment.h"
#include "models/qwen4_exp/program/host_expert_tier.h"
#include "models/qwen4_exp/program/expert_cache/expert_state.h"
#include "models/qwen4_exp/program/ngram_volume.h"
#include "models/qwen4_exp/program/prefill_walk.h"
#include "models/qwen4_exp/program/prefix/call_plan.h"
#include "models/qwen4_exp/program/prefix/prefix_cache.h"
#include "models/qwen4_exp/program/prefix/prefix_state.h"
#include "models/qwen4_exp/program/prefix/state_image.h"
#include "models/qwen4_exp/program/rope_positions.h"
#include "models/qwen4_exp/program/vision_window.h"
#include "models/qwen3_5/execution/vision_tower.h"
#include "models/qwen3_5/execution/vision_weight_stream.h"
#include "models/qwen3_5/program/vision_control.h"
#include "models/qwen4_exp/program/route_trace.h"
#include "models/qwen4_exp/program/split_capabilities.h"
#include "models/qwen4_exp/program/vram_monitor.h"
#include "models/qwen4_exp/segment_map.h"
#include "ops/offloaded_sparse_moe/cpu/fetch_channel.h"
#include "ops/offloaded_sparse_moe/cpu/miss_service.h"
#include "infernix/ops/argmax.h"
#include "infernix/ops/cast.h"
#include "infernix/ops/gdn_replay.h"
#include "infernix/ops/ple.h"
#include "infernix/ops/qsa.h"
#include "infernix/ops/sampling.h"
#include "infernix/ops/speculative_round.h"
#include "infernix/ops/token_constraint.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstdio>
#include <cerrno>
#include <cstdlib>
#include <cmath>
#include <thread>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace infernix::models::qwen4_exp {
namespace detail {

// A prompt with media: its items' control plan and the encode window it needs (design §19.3.2).
struct VisionAdmission {
    qwen3_5::VisionControlPlan control;
    VisionWindowPlan window;
};

// A TokenConstraint compiled for ops::constrain_logits: its distinct multi-token sets (written into
// the lane's set columns when it is first constrained) and each output step's descriptor relative to
// the lane (a set index, or the forced-token encoding -2 - token).
struct ConstraintPlan {
    std::vector<std::vector<TokenId>> sets;
    std::vector<std::int32_t> steps;
    std::uint64_t serial = 0; // unique per planned request: a lane's uploaded sets name their plan
};

struct BasePlanImpl {
    runtime::RequestPlanSummary summary;
    ops::SamplingConfig sampling;
    std::vector<TokenId> readout_tokens;               // read at the prompt's frontier (/v1/decide)
    std::shared_ptr<const ConstraintPlan> constraint;  // null: unconstrained
    std::uint32_t pages     = 0; // KV page groups the request needs at most (its extent)
    std::uint32_t positions = 0; // the extent: prompt + effective output, at most max_context
    bool reuse              = false; // the prefix cache may resume and publish this request
    std::shared_ptr<const VisionAdmission> vision;
    std::shared_ptr<const PreparedPrompt> prompt; // the request's prompt, moved in by plan_request
};

// The copy-on-write source of a resume (design §19.3.1, rule MR6): the snapshot's tail page (the
// frontier inside a block), the anchor block when its last MTP cell encodes another continuation,
// or none.
enum class PrefixCow : std::uint8_t { None, Tail, Anchor };

// A resume source chosen by the quote: a snapshot at `frontier` over `path`, or the root.
struct PrefixSelection {
    std::optional<runtime::prefix_cache::SnapshotRef> snapshot;
    std::vector<runtime::prefix_cache::NodeRef> path; // matched blocks up to the snapshot's anchor
    std::uint32_t frontier = 0;
    std::uint32_t shared   = 0; // leading blocks mapped from the cache
    PrefixCow cow          = PrefixCow::None;
    std::uint32_t need     = 0; // pages the admission newly takes (private + restored)
    bool image_restore     = false;
    std::vector<std::uint32_t> existing; // frontiers of every matched snapshot (tap spacing)
    std::uint32_t cached_tokens = 0;     // the prompt prefix held as cached blocks (diagnostics)
    prefix::CallPlan plan;               // the suffix's prefill calls and taps from `frontier`
};

// One lane's prefix-cache state.
struct LanePrefix {
    bool reuse = false;
    std::vector<runtime::prefix_cache::NodeRef> path; // pinned once each: mapped, then published blocks
    std::uint32_t page_base = 0;                       // logical block of lane.pages[0]
    std::uint32_t deepest   = 0;                       // frontier of the lineage's deepest snapshot
    std::uint32_t reused    = 0;
    std::optional<runtime::prefix_cache::SnapshotRef> resume;
    std::optional<prefix::RestoreTicket> restore;     // the first call waits on it per layer
    std::uint64_t capture          = 0;               // its copy-out gates the next call per layer
    // The current request's latest capture: its snapshot is the one the next capture supersedes.
    // `capture` may still name an earlier request's capture on this slot (its copy-out must land
    // before the slot is rewritten), which belongs to another lineage.
    std::uint64_t lineage_capture  = 0;
    std::uint64_t resident_capture = 0;               // the endpoint the slot still holds
    // The latest tap or boundary capture that copies one of the lane's own pages (its partial tail
    // page, which the lane keeps): release orders the streams after it before the page is freed.
    std::uint64_t tail_capture = 0;
    std::vector<runtime::prefix_cache::PlannedTap> taps;
    std::vector<runtime::prefix_cache::TapHint> hints;
    std::vector<runtime::prefix_cache::TapExclusion> exclusions; // Vision spans: no snapshot inside
    // Block lookup hashes: the prompt's, extended by publication over the generated blocks; the
    // prompt's cumulative Vision keys, and the key every block after the prompt carries.
    std::vector<std::uint64_t> hashes, extras;
    std::uint64_t trailing_extra = 0;
    // The admission's diagnostics: the prompt prefix held as cached blocks, and the bytes its
    // restore copied back from the Host tier.
    std::uint32_t cached_tokens  = 0;
    std::uint64_t restored_bytes = 0;
};

struct QuoteImpl {
    runtime::RequestPlanSummary summary;
    ops::SamplingConfig sampling;
    std::uint32_t pages = 0;
    std::uint32_t lane  = 0;
    std::optional<PrefixSelection> prefix;
    std::shared_ptr<const VisionAdmission> vision;
};

// The encode window's GPU time (Lane::vision_seconds): two events on the tower's stream around the
// pass, read when the handoff is released. The prompt's last call may be enqueued right behind the
// encode, so the read waits for the stop event, and a failed wait or read leaves the time at 0.
struct VisionTimer {
    VisionTimer() = default;
    VisionTimer(const VisionTimer&)            = delete;
    VisionTimer& operator=(const VisionTimer&) = delete;
    ~VisionTimer() {
        for (cudaEvent_t event : {start, stop}) {
            if (event != nullptr) { (void)cudaEventDestroy(event); }
        }
    }
    cudaEvent_t start = nullptr, stop = nullptr;
};

// A media prompt's encode window and handoff (design §19.3.2): filled at reserve, encoded over
// admission steps, released once the prompt's last call is enqueued or the lane is released. The
// window, the lease and the handoff are the tower's segment's (ProgramImpl::vision_seg).
struct LaneVision {
    std::shared_ptr<const VisionAdmission> admission;
    qwen3_5::VisionControl control;            // the encoded items' controls
    std::vector<std::shared_ptr<const qwen3_5::PreparedMediaPayload>> payloads; // per encoded item
    std::vector<std::uint32_t> visual;         // the encoded visual tokens' positions, ascending
    ExpertResidency::FrameLease lease;         // the handoff (W), or handoff and window (L)
    Tensor handoff;                            // BF16 [out, V_encoded]
    // A layer split: the handoff's copy for each segment that reads it (the first segment's scatter,
    // the head's MTP prompt cells) on another GPU than the tower, in frames its own cache lends with
    // the tower's lease (ProgramImpl::vision_lend), filled through the pinned crossing once the
    // encode ends (ProgramImpl::vision_mirror).
    struct Mirror {
        std::size_t segment = 0;
        ExpertResidency::FrameLease lease;
    };
    std::vector<Mirror> mirrors;
    // A lend that waits for another lane's leases (vision_lend): the prefill steps counted when it
    // last saw them advance, the polls since, and whether the wait was reported.
    std::uint64_t wait_steps = ~0ULL;
    std::uint32_t wait_polls = 0;
    bool wait_reported       = false;
    std::unique_ptr<qwen3_5::execution::VisionWeightStream> stream;
    std::optional<qwen3_5::execution::VisionParameters> weights; // the stream's rebased view
    std::unique_ptr<qwen3_5::execution::VisionPassLayout> layout;
    std::unique_ptr<qwen3_5::execution::VisionTowerPass> pass;
    std::unique_ptr<VisionTimer> timer;
    bool encoded = false;
};

struct ContractAccess {
    static SequenceHandle make_sequence(const void* owner, std::uint32_t lane, std::uint64_t epoch) noexcept {
        SequenceHandle out;
        out.owner_ = owner;
        out.lane_  = lane;
        out.epoch_ = epoch;
        return out;
    }
    static const void* owner(const SequenceHandle& h) noexcept { return h.owner_; }
    static std::uint32_t lane(const SequenceHandle& h) noexcept { return h.lane_; }
    static std::uint64_t epoch(const SequenceHandle& h) noexcept { return h.epoch_; }

    static PendingBatch make_pending(const void* owner, std::uint64_t transaction,
                                     std::span<const SequenceHandle> rows, std::span<const TokenId> tokens,
                                     runtime::ExecutionTiming timing, std::span<const std::int32_t> counts = {},
                                     std::uint32_t stride = 1) {
        PendingBatch out;
        out.owner_       = owner;
        out.transaction_ = transaction;
        out.row_count_   = rows.size();
        for (std::size_t i = 0; i < rows.size(); ++i) { out.rows_[i] = rows[i]; }
        out.tokens_     = tokens;
        out.row_counts_ = counts;
        out.row_stride_ = stride;
        out.timing_     = timing;
        return out;
    }
    static void set_constraint_failed(PendingBatch& p, std::size_t row, bool failed) noexcept {
        p.constraint_failed_[row] = failed;
    }
    static const void* owner(const PendingBatch& p) noexcept { return p.owner_; }
    static std::uint64_t transaction(const PendingBatch& p) noexcept { return p.transaction_; }
    static std::span<const SequenceHandle> rows(const PendingBatch& p) noexcept {
        return {p.rows_.data(), p.row_count_};
    }
    static void consume(PendingBatch& p) noexcept {
        p.owner_       = nullptr;
        p.transaction_ = 0;
        p.row_count_   = 0;
        p.tokens_      = {};
        p.row_counts_  = {};
        p.row_stride_  = 0;
    }
};

using Clock = std::chrono::steady_clock;

inline std::size_t align(std::size_t v) { return (v + 255) / 256 * 256; }

// A request's verification rounds whose rows were read behind the n-gram gate, and those reads.
struct GateTraffic {
    std::uint64_t rounds = 0, reads = 0, read_ns = 0;
};
inline constexpr std::size_t kGateStatsBytes     = 16;          // per lane: U64 wait ns, U64 waits
inline constexpr std::size_t kHandoffStatsBytes  = 16;          // a receiving segment: U64 wait ns, U64 waits
inline constexpr std::uint64_t kSlowGatedReadNs  = 100000000;   // a gated round's reads warn past 100 ms

inline std::uint64_t elapsed_ns(Clock::time_point start) {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count());
}

inline std::int32_t dim(std::uint64_t v) { return static_cast<std::int32_t>(v); }

inline ops::SamplingConfig translate(const ResolvedSamplingParameters& s) {
    if (!(s.temperature >= 0.0F) || !(s.top_p > 0.0F && s.top_p <= 1.0F) || s.min_p < 0.0F || s.min_p > 1.0F) {
        throw std::invalid_argument("Qwen4Exp: sampling parameters are out of range");
    }
    ops::SamplingConfig out;
    out.temperature       = s.temperature;
    out.top_k             = s.top_k;
    out.top_p             = s.top_p;
    out.min_p             = s.min_p;
    out.presence_penalty  = s.presence_penalty;
    out.frequency_penalty = s.frequency_penalty;
    out.seed              = s.seed;
    return out;
}

// A paused request (design §19.3.13): what its lane held besides model state. The pause published
// the state at `target` to the prefix cache (a request the cache publishes); a resume binds the
// ledger from the deepest cached state up to `target` and replays the rest without sampling.
struct ResumeStateImpl {
    std::uint32_t target = 0;   // the paused frontier: the positions the lane's state held
    bool generated       = false; // paused after its first token (the ledger runs past the prompt)
    std::vector<std::int32_t> history; // prompt, then committed output
    std::uint32_t prompt_tokens = 0;
    // Prefix-cache lookup keys over the ledger (the prompt's, extended over published output
    // blocks), and the capture the pause published.
    std::vector<std::uint64_t> hashes;
    std::uint64_t capture = 0;
    runtime::BeginSummary begin;
    std::uint64_t prefill_ns = 0, decode_ns = 0, decode_share_ns = 0;
    double vision_seconds = 0;
    SpeculativeStats speculative;
    std::array<double, 8> mtp_accept{};
    std::uint64_t mtp_policy_rounds = 0;
    std::vector<ConstrainedDraw> constraint_trace;
    ExpertResidency::Stats cache_at_admission;
    std::uint64_t cpu_served_at_admission = 0;
    NgramVolume::Counters ngram;
    GateTraffic gate;
};

class ProgramImpl {
public:
    enum class Phase : std::uint8_t { Free, Prefill, Decode, Finishable };

    struct Lane {
        Phase phase          = Phase::Free;
        std::uint64_t epoch  = 0;
        runtime::BeginSummary begin; // the Begin of the binding
        std::vector<std::int32_t> history; // prompt, then committed output
        std::uint32_t prompt_tokens = 0;
        std::uint32_t state_tokens  = 0; // positions already in the model state
        // Where the prefill phase ends: the prompt, or a resumed lane's paused frontier, which its
        // replay reaches without sampling (design §19.3.13).
        std::uint32_t prefill_end = 0;
        bool replay               = false;
        std::uint32_t extent      = 0; // the positions the request may ever hold (prompt + output)
        // The prompt's planned prefill calls (design §19.3.1): exclusive ends, the last prompt_tokens.
        std::vector<std::uint32_t> calls;
        std::size_t next_call = 0;
        LaneRope rope;                    // RoPE positions of the lane's tokens (design §19.3.2)
        std::unique_ptr<LaneVision> vision; // the encode window and handoff of a media prompt
        double vision_seconds = 0;          // the tower's GPU time
        ops::SamplingConfig sampling;
        // /v1/decide: the tokens the prompt's readout names, the constraint, the draws of the pending
        // round's tokens (one per licensed token) and the committed trace they move into.
        std::vector<TokenId> readout;
        std::shared_ptr<const ConstraintPlan> constraint;
        std::shared_ptr<const PreparedPrompt> prompt; // a waiting sibling compares its media
        std::vector<ConstrainedDraw> constraint_round, constraint_trace;
        std::optional<DeviceKVPageReservation> reservation;
        std::vector<DeviceKVPageLease> pages; // the head's pool hands them out
        KVExecutionRowLease row;              // the lane's block-table row in the head's tables
        // A layer split: the same row in each other segment's tables (by segment index; invalid for
        // the head and for a segment without KV), published with the head's page indices.
        std::vector<KVExecutionRowLease> mirror_rows;
        std::uint64_t prefill_ns      = 0;
        std::uint64_t decode_ns       = 0;
        std::uint64_t decode_share_ns = 0;
        ExpertResidency::Stats cache_at_admission;
        std::uint64_t cpu_served_at_admission = 0;
        HostExpertTier::Stats tier_at_admission{}; // SSD tier mode
        NgramVolume::Counters ngram; // this request's n-gram row traffic
        GateTraffic gate;            // its verification rounds' rows read behind the n-gram gate
        std::unique_ptr<qwen3_5::detail::NgramProposer> proposer; // copy proposals over history
        SpeculativeStats speculative;
        // MTP drafter (design 11.2; rule MR1 of 19.3.1): its KV cells [0, mtp_cells) are final. At
        // every settled point the lane's saved column holds the residual of position
        // state_tokens - 1, whose cell is written (mtp_cells == state_tokens) or pending
        // (state_tokens - 1: the next call or round writes it).
        std::uint32_t mtp_cells = 0;
        bool mtp_live           = false;
        // Draft-length policy (design 11.3): conditional acceptance of each draft position.
        std::array<double, 8> mtp_accept{};
        std::uint64_t mtp_policy_rounds = 0;
        LanePrefix prefix;
    };

    // Everything a segment allocates on its device besides the expert frames; validates the options.
    // A segment's recurrent state, verification records, QSA tails and KV planes cover its own
    // layers (its compact GDN and attention ranges, and the MTP block's on the head); a segment
    // without attention layers (a split at K <= 3) has no KV. Every segment holds the whole per-call
    // io layout, its workspace and staging; the logits, sampling, constraint and MTP buffers are the
    // head's, the n-gram gate's statistics the PLE segment's. The whole model's segment
    // (SegmentMap::single) is the unsplit plan.
    static DeviceLayout plan_segment(const ProgramOptions& o, const Config& config, std::int32_t public_tokens,
                                     const SegmentSpec& segment) {
        const TextConfig& c = config.text;
        if (segment.layers.begin >= segment.layers.end || segment.layers.end > c.num_hidden_layers ||
            segment.embed != (segment.layers.begin == 0) || segment.head != (segment.layers.end == c.num_hidden_layers) ||
            segment.mtp != (segment.head && config.mtp) ||
            segment.kv_layers != segment.attention_count + (segment.mtp ? 1U : 0U)) {
            throw std::logic_error("Qwen4Exp: the segment does not match the model's layers and roles");
        }
        if (o.max_concurrency == 0 || o.max_concurrency > kMaximumConcurrency) {
            throw std::invalid_argument("Qwen4Exp: max_concurrency must be in [1,8]");
        }
        if (o.max_context == 0 || o.prefill_chunk == 0) {
            throw std::invalid_argument("Qwen4Exp: max_context and prefill_chunk must be nonzero");
        }
        if (o.max_context > c.context_ceiling()) {
            throw std::invalid_argument("Qwen4Exp: max_context " + std::to_string(o.max_context) + " exceeds the " +
                                        std::to_string(c.context_ceiling()) +
                                        " positions of the model (native positions x the YaRN factor)");
        }
        if (o.ngram_draft_tokens > 15 ||
            (o.ngram_draft_tokens > 0 && (o.ngram_min_match < 4 || o.ngram_min_match > 64))) {
            throw std::invalid_argument("Qwen4Exp: n-gram drafts must be 0..15 with a minimum match of 4..64");
        }
        if (config.mtp && (o.mtp_draft_tokens == 0 || o.mtp_draft_tokens > 7)) {
            throw std::invalid_argument("Qwen4Exp: MTP needs 1..7 draft tokens");
        }
        DeviceLayout d;
        d.lanes         = static_cast<std::int32_t>(o.max_concurrency);
        d.token_domain  = public_tokens;
        d.owns_head     = segment.head;
        d.owns_ple      = segment.ple;
        d.mtp           = segment.mtp;
        d.head_rows     = config.proposal_rows != 0 ? dim(config.proposal_rows) : dim(c.vocab_size);
        d.chunk         = static_cast<std::int32_t>(std::min(o.prefill_chunk, o.max_context));
        // The round's drafts, the same on every segment (its width); the drafter runs on the head.
        d.mtp_k         = config.mtp ? static_cast<std::int32_t>(o.mtp_draft_tokens) : 0;
        d.max_width     = 1 + std::max(static_cast<std::int32_t>(o.ngram_draft_tokens), d.mtp_k);
        d.columns       = std::max(d.chunk, d.lanes * d.max_width);
        d.pages_per_row = (dim(o.max_context) + kPagedKVPageSize - 1) / kPagedKVPageSize;
        const auto kv_tokens = o.kv_capacity_tokens != 0 ? o.kv_capacity_tokens : o.max_context * o.max_concurrency;
        d.kv_pages      = (kv_tokens + kPagedKVPageSize - 1) / kPagedKVPageSize;
        const std::int32_t lanes = d.lanes, W = d.max_width;
        // One copy-on-write page per lane, so a whole-extent resume always fits (design §19.3.1).
        if (o.prefix_cache) { d.kv_pages += static_cast<std::uint32_t>(lanes); }
        const std::size_t width = c.residual_width(), di = c.qsa.index_head_dim, r = c.qsa.compress_ratio;

        // Recurrent state: one slot per lane of each of the segment's GDN layers; the PLE history (its
        // segment's); one QSA tail slab per attention layer, and one more for the MTP block.
        if (segment.gdn_count > 0) {
            LayoutBuilder state_builder;
            d.gdn       = plan_linear_attention_state_pool(state_builder, {
                                                                        .layers         = segment.gdn_count,
                                                                        .conv_channels  = dim(c.gdn.conv_channels()),
                                                                        .conv_width     = dim(c.gdn.conv_kernel - 1),
                                                                        .value_heads    = dim(c.gdn.value_heads),
                                                                        .value_head_dim = dim(c.gdn.value_head_dim),
                                                                        .key_head_dim   = dim(c.gdn.key_head_dim),
                                                                        .slot_count     = lanes,
                                                                    });
            d.gdn_bytes = state_builder.finish(256);
        }
        d.ple       = segment.ple ? width * c.ple.conv_span() * lanes * 2 : 0;
        d.kv_layers = segment.kv_layers;
        d.tails     = d.kv_layers * di * (r - 1) * lanes * 2;
        d.window    = prefix::kv_window_geometry(c, o.kv_cache, segment.attention_count, d.mtp,
                                                 static_cast<std::uint32_t>(lanes))
                       .bytes();

        // Paged KV: one page group holds 64 positions of every attention layer of the segment, over
        // the same page indices on every segment (the head's pool hands them out).
        // The page group's planes, in the order the prefix cache's records also use.
        if (d.kv_layers > 0) {
            const prefix::KvPageGeometry kv_geometry =
                prefix::kv_page_geometry(c, o.kv_cache, segment.attention_count, d.mtp);
            LayoutBuilder kv_builder;
            d.pool   = plan_device_kv_page_pool(kv_builder,
                                                {.page_group_count = d.kv_pages, .geometry = kv_geometry.geometry});
            d.tables = plan_kv_execution_tables(
                kv_builder, {.logical_page_capacity = static_cast<std::uint32_t>(d.pages_per_row), .table_rows = lanes});
            d.kv = kv_builder.finish(256);
            // Pages [0, n) are a prefix of each page-major plane, so growth maps whole chunks at each plane's end.
            d.kv_elastic    = d.pool.spec.geometry.device_plane_order == PagedKVPlaneOrder::PageMajor;
            d.kv_base_pages = std::min<std::uint32_t>(d.kv_pages, kKvBaseTokens / kPagedKVPageSize);
        }

        // Per-call inputs, logits and sampling counts.
        const std::size_t heads     = NgramHash(c.ple.ngram, 0).heads();
        const std::size_t row_bytes = c.ple.table.row_bytes;
        const std::size_t columns   = static_cast<std::size_t>(d.columns);
        const std::size_t mtp_subchunks = (columns + 1 + execution::kMtpChunkColumns - 1) / execution::kMtpChunkColumns;
        d.io.ids            = 0;
        d.io.positions      = align(d.io.ids + 4ULL * columns);
        d.io.rope           = align(d.io.positions + 4ULL * columns);
        d.io.block_rope     = align(d.io.rope + 12ULL * columns);
        d.io.slots          = align(d.io.block_rope + 12ULL * lanes);
        d.io.rows           = align(d.io.slots + 4ULL * lanes);
        d.io.columns        = align(d.io.rows + 4ULL * lanes);
        d.io.gate           = align(d.io.columns + 4ULL * columns);
        d.io.handoff        = d.io.gate + 16; // the gate's line: the unsplit layout stays as it was
        d.io.ngram          = align(d.io.handoff + 4);
        d.io.mtp_ids        = align(d.io.ngram + row_bytes * heads * columns);
        d.io.mtp_cells      = align(d.io.mtp_ids + 4ULL * (columns + 1));
        d.io.mtp_rope       = align(d.io.mtp_cells + 4ULL * (columns + 1));
        d.io.mtp_block_rope = align(d.io.mtp_rope + 12ULL * (columns + 1));
        d.io.vision_columns     = align(d.io.mtp_block_rope + 12ULL * mtp_subchunks);
        d.io.mtp_vision_columns = align(d.io.vision_columns + 4ULL * columns);
        d.io.bytes              = align(d.io.mtp_vision_columns + 4ULL * (columns + 1));
        if (d.io.block_rope + 12ULL * lanes > d.io.ngram || d.io.columns + 4ULL * columns > d.io.ngram ||
            d.io.handoff + 4 > d.io.ngram) {
            throw std::logic_error("Qwen4Exp: decode io must precede the n-gram rows");
        }
        if (d.owns_head) {
            d.logits32     = sizeof(float) * c.vocab_size * lanes * W;
            d.logits16     = 2ULL * c.vocab_size * lanes * W;
            d.token_counts = 4ULL * static_cast<std::size_t>(d.token_domain) * lanes;
        }

        // The workspace arena (the forward pass, sampling, draft acceptance and the drafter) covers
        // decode, verification and calls of up to kStaticWorkColumns columns; a wider prefill call
        // swaps in an arena of the chunk's size lent from expert frames (design §19.3.7, VRAM item 1).
        d.static_columns = std::max(lanes * W, std::min(d.chunk, kStaticWorkColumns));
        d.work           = workspace_bytes(c, o, d, lanes, d.static_columns);
        d.wide_work      = d.chunk > d.static_columns ? workspace_bytes(c, o, d, lanes, d.chunk) : 0;

        // Verification (W > 1): GDN replay records of the segment's GDN layers, the PLE's records
        // (its segment's), QSA records of its attention layers, the round's I32 arrays.
        if (W > 1) {
            if (segment.gdn_count > 0) {
                LayoutBuilder records_builder;
                d.records       = plan_gdn_replay_records(records_builder, {.layers          = dim(segment.gdn_count),
                                                                            .record_capacity = lanes,
                                                                            .width           = W,
                                                                            .conv_channels   = dim(c.gdn.conv_channels()),
                                                                            .qk_heads        = dim(c.gdn.key_heads),
                                                                            .value_heads     = dim(c.gdn.value_heads),
                                                                            .key_dim         = dim(c.gdn.key_head_dim),
                                                                            .value_dim       = dim(c.gdn.value_head_dim)});
                d.records_bytes = records_builder.finish(256);
            }
            d.ple_records   = segment.ple ? 2ULL * width * W * lanes : 0;
            d.qsa_records   = 2ULL * di * W * lanes * segment.attention_count;
            const std::size_t cells = static_cast<std::size_t>(lanes) * W;
            std::size_t at          = 0;
            const auto take         = [&](std::size_t words) {
                const std::size_t begin = at;
                at += (words + 63) / 64 * 64;
                return begin;
            };
            d.spec.target   = take(cells);
            d.spec.drafts   = take(cells);
            d.spec.extents  = take(lanes);
            d.spec.lengths  = take(lanes);
            d.spec.anchors  = take(lanes);
            d.spec.licensed = take(cells);
            d.spec.counts   = take(lanes);
            d.spec.accepted = take(lanes);
            d.spec.commit   = take(lanes);
            d.spec.words    = at;
        }

        // The MTP drafter (design 11.2): residual columns, the saved and chain columns, its QSA
        // records, the HC ones and its io.
        if (d.mtp) {
            d.mtp_columns      = std::max(lanes * W, std::min(d.chunk, 512));
            d.mtp_column_bytes = 2ULL * width;
            d.mtp_ones         = sizeof(float) * c.hc.streams * static_cast<std::size_t>(d.mtp_columns);
            const auto k       = static_cast<std::size_t>(d.mtp_k);
            d.mtp_io.ids              = 0;
            d.mtp_io.cells            = d.mtp_io.ids + lanes;
            d.mtp_io.rope_cells       = d.mtp_io.cells + lanes * k;
            d.mtp_io.block_rope_cells = d.mtp_io.rope_cells + 3 * lanes * k;
            d.mtp_io.drafts           = d.mtp_io.block_rope_cells + 3 * lanes * k;
            d.mtp_io.up_ids    = d.mtp_io.drafts + lanes * k;
            d.mtp_io.gather    = d.mtp_io.up_ids + static_cast<std::size_t>(lanes) * W;
            d.mtp_io.up_cells  = d.mtp_io.gather + lanes;
            d.mtp_io.logprobs  = d.mtp_io.up_cells + static_cast<std::size_t>(lanes) * W;
            d.mtp_io.words     = d.mtp_io.logprobs + lanes * k;
        }

        // Staging slots for each layer call's misses (design 8.6), one expert record each.
        const std::uint64_t shape[] = {c.moe.experts, c.hidden_size, c.moe.intermediate};
        d.record_stride = weight_geometry(QType::NVFP4_MUL, QuantLayout::ExpertRg16, shape).record_stride;
        d.staging       = static_cast<std::size_t>(kStagingSlots) * d.record_stride;

        auto& b     = d.bytes;
        b.kv_pages  = d.kv_pages;
        b.kv        = d.kv_elastic ? kv_mapped_bytes(d, d.kv_base_pages) : d.kv;
        b.kv_max    = d.kv;
        b.workspace = d.work;
        b.staging   = d.staging;
        b.state     = d.gdn_bytes + d.ple + d.tails + d.window + d.records_bytes + d.ple_records + d.qsa_records +
                  (d.mtp ? d.mtp_column_bytes * lanes * (W + 2) + 2ULL * di * W * lanes : 0);
        // The head's logits, sampled tokens and positions, sampling configs, penalty counts,
        // constraint controls and grammar masks; the PLE segment's gate statistics.
        const std::size_t head_io = d.owns_head ? d.logits32 + d.logits16 + 4ULL * lanes + 4ULL * lanes +
                                                 sizeof(ops::SamplingConfig) * lanes + d.token_counts
                                           : 0;
        b.io = d.io.bytes + head_io + 4ULL * d.spec.words + d.mtp_ones + 4ULL * d.mtp_io.words +
               (d.owns_ple ? kGateStatsBytes * lanes : 0) + (segment.embed ? 0 : kHandoffStatsBytes) +
               (d.owns_head ? constraint_bytes(lanes, W) + grammar_mask_bytes(d.token_domain, lanes, W) : 0);
        b.residency            = ExpertResidency::table_bytes(c, d.columns, segment.layers.size());
        b.expert_record_stride = d.record_stride;
        b.max_frames           = ExpertResidency::max_frames(c, segment.layers.size());
        // Decode and verification graphs on every segment, the drafter's on the head.
        b.graph_bound = graph_bound(o.max_concurrency, static_cast<std::uint32_t>(W),
                                    d.mtp ? static_cast<std::uint32_t>(d.mtp_k) : 0U);
        return d;
    }

    // The device plan of an unsplit Program: its one segment's.
    static DeviceLayout plan_device(const ProgramOptions& o, const Config& config, std::int32_t public_tokens) {
        return plan_segment(o, config, public_tokens, SegmentMap::single(config.text, config.mtp, 0)[0]);
    }

    // The chunks of the KV backing that pages [0, pages) of every plane and the block tables need.
    static std::vector<std::uint8_t> kv_chunks(const DeviceLayout& d, std::uint32_t pages) {
        std::vector<std::uint8_t> need((d.kv + kKvChunkBytes - 1) / kKvChunkBytes, 0);
        const auto mark = [&](std::size_t offset, std::size_t bytes) {
            if (bytes == 0) { return; }
            for (std::size_t c = offset / kKvChunkBytes; c <= (offset + bytes - 1) / kKvChunkBytes; ++c) { need[c] = 1; }
        };
        for (const auto& plane : d.pool.planes) { // page-major: the page is the outermost dimension
            const auto total = static_cast<std::size_t>(d.pool.spec.page_group_count);
            mark(plane.storage.region.offset, plane.storage.region.bytes / total * pages);
        }
        mark(d.tables.block_tables.region.offset, d.tables.block_tables.region.bytes);
        return need;
    }
    static std::size_t kv_mapped_bytes(const DeviceLayout& d, std::uint32_t pages) {
        const auto need = kv_chunks(d, pages);
        return static_cast<std::size_t>(std::count(need.begin(), need.end(), std::uint8_t{1})) * kKvChunkBytes;
    }

    // The workspace arena of a Program with `lanes` lanes and the plan's widths: the forward at its
    // widest call (a prefill chunk or every lane's widest round), sampling, acceptance and the
    // drafter. The plan sizes the arena with the Program's lane count; the per-lane report compares
    // lane counts.
    static std::size_t workspace_bytes(const TextConfig& c, const ProgramOptions& o, const DeviceLayout& d,
                                       std::int32_t lanes, std::int32_t columns) {
        const std::int32_t W = d.max_width;
        std::size_t bytes =
            execution::Forward::workspace_bytes(c, std::max(columns, lanes * W), dim(o.max_context), o.kv_cache);
        if (!d.owns_head) { return bytes; } // sampling, acceptance and the drafter run on the head
        bytes += ops::sampling_workspace_capacity_bytes(d.token_domain, 1, lanes);
        if (W > 1) {
            bytes += ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(d.token_domain, 1, W - 1, 1, lanes);
        }
        if (d.mtp) {
            bytes += execution::Forward::mtp_workspace_bytes(c, std::max(lanes * W, std::min(d.chunk, 512)), lanes,
                                                             d.head_rows, dim(o.max_context), o.kv_cache);
        }
        return bytes;
    }

    static void allocate(DeviceBuffer& buffer, std::size_t bytes, std::uint64_t& tally) {
        buffer = DeviceBuffer(bytes);
        tally += bytes;
    }

    // The startup failure for device allocations the driver placed in system memory.
    static std::string spill_message(std::uint64_t bytes) {
        return std::to_string(bytes >> 20) +
               " MiB of Infernix's VRAM allocations were placed in shared system memory by the driver's sysmem "
               "fallback: free VRAM, or lower --max-context, --max-concurrency or --prefill-chunk. Setting the NVIDIA "
               "Control Panel 'CUDA - Sysmem Fallback Policy' to 'Prefer No Sysmem Fallback' for this program makes "
               "this fail at allocation instead.";
    }

    // A Program over the device segments of `map` (segment_map.h): segment s runs on devices[s] with
    // parameters[s], the Parameters of its layers and role prepared on its device. One segment is the
    // unsplit Program, the whole model on one device. A layer split builds every segment's state, KV
    // planes, io, workspace, staging and expert cache on its own device (under its DeviceGuard), one
    // CPU expert service with a port per segment, and joins consecutive segments' Forwards through a
    // HandoffChannel (decode and verification rounds, in the options' handoff mode) and a BulkHandoff
    // (eager calls); the prefix cache copies each segment's planes and state on that segment's
    // streams. It refuses what it cannot run yet (split_capabilities.h): the SSD expert tier; and the
    // gate handoff on a split whose segments share a GPU, where a waiting gate can keep the sender
    // from running (design §16.4, layer split).
    ProgramImpl(std::span<const execution::Parameters* const> parameters, std::span<DeviceContext* const> devices,
                const SegmentMap& map, ProgramOptions options)
        : parameters_(first_parameters(parameters)), c_(parameters_.model.config().text),
          options_(std::move(options)), volume_(options_.ngram_volume, c_.ple.table), hash_(c_.ple.ngram, 0) {
        const Config& config = parameters_.model.config();
        check_segments(parameters, devices, map);
        const bool split = map.size() > 1;
        if (split) {
            static_assert(kSplitPrefixCache, "a layer split's prefix cache copies every segment's planes and state");
            if (parameters_.model.expert_store().has_value() && !kSplitExpertTier) {
                throw std::invalid_argument(
                    "Qwen4Exp: a layer split needs every expert in Host RAM (it has no SSD expert tier yet)");
            }
            if (options_.handoff == SplitHandoff::Gate && map.same_device()) {
                throw std::invalid_argument("Qwen4Exp: the gate handoff needs the split's segments on different GPUs "
                                            "(a split on one GPU uses the event handoff)");
            }
            handoff_ = options_.handoff == SplitHandoff::Gate ? HandoffMode::Gate : HandoffMode::Event;
            split_sync_ = split_sync_from_environment();
            split_walk_ = split_walk_from_environment();
            moe_offload_ = map.size() == 2 && split_walk_ != SplitWalk::ChunkMajor && moe_offload_from_environment();
        }
        timing_.on = split_timing_from_environment();
        ple_segment_ = map.ple().index;
        const auto public_tokens = dim(parameters_.model.resources().public_token_count);
        for (std::size_t s = 0; s < map.size(); ++s) {
            DeviceSegment& seg = *segments_.emplace_back(std::make_unique<DeviceSegment>());
            seg.spec           = map[s];
            seg.device         = devices[s];
            seg.parameters     = parameters[s];
            seg.plan           = plan_segment(options_, config, public_tokens, seg.spec);
        }
        // The drafter and its head belong to the head segment.
        const execution::Parameters& head_parameters = *head().parameters;
        const auto& draft_head                       = head_parameters.draft_head;
        mtp_                                         = head_parameters.mtp.has_value();
        if (mtp_ != head().plan.mtp ||
            (mtp_ && (draft_head.rows ? dim(draft_head.rows->weight.n) : dim(c_.vocab_size)) != head().plan.head_rows)) {
            throw std::logic_error("Qwen4Exp: the bound drafter differs from its configuration");
        }
        // The call shapes, the same in every segment's plan.
        const DeviceLayout& plan = head().plan;
        const std::int32_t lanes = plan.lanes;
        vocab_         = dim(c_.vocab_size);
        token_domain_  = plan.token_domain;
        chunk_         = plan.chunk;
        mtp_k_         = plan.mtp_k;
        max_width_     = plan.max_width;
        columns_       = plan.columns;
        pages_per_row_ = plan.pages_per_row;
        kv_pages_      = plan.kv_pages;
        width_         = dim(c_.residual_width());
        span_          = dim(c_.ple.conv_span());
        di_            = dim(c_.qsa.index_head_dim);
        r_             = dim(c_.qsa.compress_ratio);
        io_layout_     = plan.io;
        if (!options_.logits_hash.empty()) {
#ifdef _WIN32
            logits_hash_.reset(_wfopen(options_.logits_hash.c_str(), L"w"));
#else
            logits_hash_.reset(std::fopen(options_.logits_hash.c_str(), "w"));
#endif
            if (!logits_hash_) {
                throw std::runtime_error("Qwen4Exp: cannot create the logits hash file " + options_.logits_hash.string());
            }
        }

        if (timing_.on) {
            timing_.graph_ms.assign(segments_.size(), 0.0);
            for (const auto& segment : segments_) {
                const DeviceGuard on(segment->device->device);
                CUDA_CHECK(cudaEventCreate(&segment->timing_begin.event));
                CUDA_CHECK(cudaEventCreate(&segment->timing_end.event));
            }
            const DeviceGuard on(head().device->device);
            for (SegmentEvent* e : {&head().catch_begin, &head().catch_end, &head().draft_begin, &head().draft_end}) {
                CUDA_CHECK(cudaEventCreate(&e->event));
            }
        }
        // Every segment's fixed allocations, checked against its plan and its device's free memory.
        std::vector<SegmentBuild> builds(segments_.size());
        for (std::size_t s = 0; s < segments_.size(); ++s) { build_segment(*segments_[s], builds[s]); }
        const std::uint64_t record_stride = plan.record_stride;
        // The SSD expert tier (design §19.3.7): the banks were not pinned; RAM slots hold the
        // records the tier keeps and the rest are read from the artifact when a round needs them.
        // One segment only (refused above for a split).
        if (const auto& store = parameters_.model.expert_store(); store.has_value()) {
            if (store->record_bytes() != record_stride) {
                throw std::logic_error("Qwen4Exp: the expert store's records differ from the banks' stride");
            }
            // A call's CPU jobs hold their ring slots until the call's answer, and a pass's fetched
            // records until the pass has copied them: the ring holds both with a pass to spare.
            const std::uint32_t cpu_cap =
                options_.cpu_expert_workers > 0 && options_.cpu_expert_jobs > 0
                    ? std::min<std::uint32_t>(std::max(options_.cpu_expert_jobs, options_.cpu_assist_jobs),
                                              static_cast<std::uint32_t>(ops::offloaded_moe::kMaxCpuJobs))
                    : 0U;
            HostExpertTier::Options tier;
            tier.ring  = std::max<std::uint32_t>(128, cpu_cap + 2U * kStagingSlots);
            tier.slots = static_cast<std::uint32_t>(std::min<std::uint64_t>(
                options_.expert_ram_bytes / record_stride, static_cast<std::uint64_t>(store->layers()) * store->experts()));
            const std::uint32_t lists = tier.ring + tier.prefetch + tier.demotion;
            if (tier.slots < std::max<std::uint32_t>(kMinTierSlots, lists + kMinTierSlots / 2)) {
                throw std::runtime_error("Qwen3.8-Flash-Next: the SSD expert tier needs RAM for at least " +
                                         std::to_string(std::max<std::uint32_t>(kMinTierSlots, lists + kMinTierSlots / 2)) +
                                         " expert slots (" + std::to_string(tier.slots) + " fit in the RAM ledger's expert share)");
            }
            fetch_channel_ = std::make_unique<ops::offloaded_moe::FetchChannel>();
            tier.fetch     = fetch_channel_.get();
            // A layer walk's step calls ceil(layers / n) layers once per chunk of its n-chunk span (one
            // tier round per step): at most layers + n - 1 layer calls.
            tier.layer_calls = store->layers() + kWalkMaxTokens / static_cast<std::uint32_t>(std::max(chunk_, 1)) + 1U;
            tier_          = std::make_unique<HostExpertTier>(*store, std::move(tier));
            head().residency->attach_tier(tier_.get());
        }
        // After the tier (whose pinned slots are the probe's source when the banks stay in the
        // artifact), before the CPU service and the split take the measured rates and the prefix
        // cache its costs. Each segment measures its own device's link.
        for (const auto& segment : segments_) {
            const DeviceGuard on(segment->device->device);
            measure_link(*segment);
            if (segment->residency) {
                // A segment that promotes little or nothing during requests loads back what lending
                // evicts; a trickle's promotions run in a window of layers, issued after its share of
                // the next round (set_trickle).
                segment->residency->set_refill_lent(!promotes_in_rounds(*segment) || trickles(*segment));
                if (trickles(*segment)) { segment->residency->set_trickle(segment->trickle, true); }
            }
        }
        apply_link_costs();
        // After the links are measured: the tower runs where its weights stream fastest.
        place_vision();
        if (decode_promotions_ != DecodePromotions::On) {
            diagnostic(std::string("expert cache: decode promotions ") +
                       (decode_promotions_ == DecodePromotions::Off ? "off" : "only until the frames fill") +
                       " (INFERNIX_DECODE_PROMOTIONS)");
        }
        expert_error_ = PinnedHostBuffer(64);
        std::memset(expert_error_.data(), 0, 64);
        for (SegmentBuild& build : builds) { build.experts.error = static_cast<std::uint32_t*>(expert_error_.data()); }
        if (options_.prefill_stream && options_.expert_cache) {
            for (std::size_t s = 0; s < segments_.size(); ++s) {
                DeviceSegment& seg = *segments_[s];
                const DeviceGuard on(seg.device->device);
                seg.expert_stream      = make_expert_stream(seg);
                builds[s].experts.stream = seg.expert_stream.get();
            }
        }
        // CPU-served misses for decode and verify calls, and for every call of at most
        // kMaxCpuColumns columns (short prompts, chunk remainders, forced tokens), whose experts are
        // all thin and so give the same bits on either device (design 16.2); wider prefill chunks
        // stay on the GPU. A split serves every segment through one service, one port per segment
        // (on its device, with its link's PCIe divisor).
        const std::uint32_t cpu_workers = options_.cpu_expert_workers;
        const std::uint32_t cpu_jobs    = options_.cpu_expert_jobs;
        const std::uint32_t assist_jobs = options_.cpu_assist_jobs;
        if (cpu_workers > 0 && cpu_jobs > 0) {
            // Decode and verification calls are CPU-served up to decode_columns; prefill calls up to
            // kAssistMaxColumns take the assist cap (wider chunks stream or stage on the GPU).
            const int decode_columns = std::max(ops::offloaded_moe::kMaxCpuColumns, lanes * max_width_);
            const bool assist        = assist_jobs > 0 && decode_columns < kAssistMaxColumns;
            const int max_jobs       = static_cast<int>(std::min<std::uint32_t>(cpu_jobs, ops::offloaded_moe::kMaxCpuJobs));
            const int max_columns    = assist ? kAssistMaxColumns : decode_columns;
            const int wide_from      = assist ? decode_columns + 1 : 0;
            const int wide_jobs =
                assist ? static_cast<int>(std::min<std::uint32_t>(assist_jobs, ops::offloaded_moe::kMaxCpuJobs)) : 0;
            // Every layer's bank (the service's layers are global), from the model: a segment's
            // Parameters hold only its own.
            std::vector<ops::offloaded_moe::CpuMissService::Layer> service_layers;
            for (const auto& bank : parameters_.model.expert_banks()) {
                service_layers.push_back({.records       = reinterpret_cast<const std::uint8_t*>(bank.planes.records),
                                          .record_stride = bank.planes.record_stride,
                                          .scales        = bank.scales.data()});
            }
            if (!split) {
                cpu_service_ = std::make_unique<ops::offloaded_moe::CpuMissService>(
                    std::move(service_layers),
                    ops::offloaded_moe::CpuMissService::Options{
                        .workers      = static_cast<int>(cpu_workers),
                        .max_jobs     = max_jobs,
                        .max_columns  = max_columns,
                        .pcie_divisor = pcie_divisor(head()),
                        .wide_from    = wide_from,
                        .wide_jobs    = wide_jobs,
                        .cpus         = {},
                        .records      = tier_.get()}); // tier mode: SSD-only jobs read through the tier
            } else {
                // Each port with its link's PCIe divisor and CPU jobs: a slow link's segment hands the
                // CPU more of its decode misses (derive_link_policy).
                std::vector<ops::offloaded_moe::CpuMissService::PortOptions> ports;
                for (std::size_t s = 0; s < segments_.size(); ++s) {
                    DeviceSegment& seg = *segments_[s];
                    seg.cpu_port       = static_cast<int>(s);
                    ports.push_back({.device       = seg.device->device,
                                     .max_jobs     = static_cast<int>(std::min<std::uint32_t>(
                                         std::max<std::uint32_t>(seg.policy.cpu_jobs, cpu_jobs),
                                         static_cast<std::uint32_t>(ops::offloaded_moe::kMaxCpuJobs))),
                                     .max_columns  = max_columns,
                                     .pcie_divisor = pcie_divisor(seg),
                                     .wide_from    = wide_from,
                                     .wide_jobs    = wide_jobs});
                }
                cpu_service_ = std::make_unique<ops::offloaded_moe::CpuMissService>(
                    std::move(service_layers),
                    ops::offloaded_moe::CpuMissService::Options{
                        .workers = static_cast<int>(cpu_workers), .cpus = {}, .records = tier_.get()},
                    std::move(ports));
            }
            for (std::size_t s = 0; s < segments_.size(); ++s) {
                DeviceSegment& seg = *segments_[s];
                for (std::uint32_t l = seg.spec.layers.begin; l < seg.spec.layers.end; ++l) {
                    builds[s].experts.cpu.push_back(cpu_service_->channel(static_cast<int>(l), seg.cpu_port));
                }
                if (assist && seg.expert_stream && options_.prefill_cpu_split) {
                    seg.split_policy        = std::make_unique<SplitPolicy>(*this, seg);
                    builds[s].experts.split = seg.split_policy.get();
                }
            }
        }
        create_prefix_cache();

        // A split's segment whose cache could never lend its wide arena (one of a few layers: its
        // frame limit is every routed expert of its layers but one) holds the arena as its own
        // allocation, made before the frames take the remaining memory.
        if (split && options_.expert_cache) {
            for (const auto& segment : segments_) {
                DeviceSegment& seg         = *segment;
                const std::uint32_t limit  = seg.residency->frame_limit();
                const auto wide_frames     = static_cast<std::uint32_t>((seg.wide_work + record_stride - 1) / record_stride);
                if (seg.wide_work == 0 || wide_frames < limit - limit / 4) { continue; }
                const DeviceGuard on(seg.device->device);
                SpillGuard guard(*seg.vram);
                guard.begin();
                seg.wide_fallback = DeviceBuffer(seg.wide_work);
                if (const std::uint64_t spilled = guard.end(seg.wide_work); spilled != 0) {
                    throw std::runtime_error(segment_name(seg) + spill_message(spilled));
                }
                seg.wide_dedicated = true;
            }
        }
        // The VRAM expert cache takes what the fixed allocations, the reserve for graph
        // executables and the display headroom leave (design §19.3.7). Its frames are the one
        // elastic allocation: a chunk the driver places in system memory (another program took
        // memory since the reading) is given back and the cache stops growing there. Segments that
        // share a device (the same-device test mode) divide one sizing of it in proportion to their
        // layers (SegmentSpec::frame_share), each still taking what lends its wide arena, without
        // the runtime VRAM monitor.
        std::vector<std::pair<int, VramSizing>> shared_sizing;
        std::vector<std::uint32_t> shared_frames(segments_.size(), 0);
        for (std::size_t s = 0; s < segments_.size(); ++s) {
            DeviceSegment& seg = *segments_[s];
            const DeviceGuard on(seg.device->device);
            if (options_.expert_cache) {
                VramDemand demand;
                demand.headroom    = options_.vram_headroom;
                demand.elastic     = seg.residency->elastic() && options_.vram_monitor && !seg.spec.shares_device;
                demand.graphs      = seg.plan.bytes.graph_bound;
                demand.frame_bytes = record_stride;
                demand.max_frames  = seg.residency->frame_limit();
                std::uint32_t wanted = 0;
                if (!seg.spec.shares_device) {
                    seg.sizing = size_expert_frames(seg.vram->query(), demand);
                    wanted     = seg.sizing.frames;
                } else {
                    auto it = std::find_if(shared_sizing.begin(), shared_sizing.end(),
                                           [&](const auto& entry) { return entry.first == seg.spec.device; });
                    if (it == shared_sizing.end()) {
                        // The device's one sizing, before any of its segments takes frames.
                        VramDemand all = demand;
                        all.graphs     = 0;
                        all.max_frames = 0;
                        for (const auto& other : segments_) {
                            if (other->spec.device != seg.spec.device) { continue; }
                            all.graphs += other->plan.bytes.graph_bound;
                            all.max_frames += other->residency->frame_limit();
                        }
                        shared_sizing.emplace_back(seg.spec.device, size_expert_frames(seg.vram->query(), all));
                        it = std::prev(shared_sizing.end());
                        share_device_frames(seg.spec.device, it->second.frames, shared_frames);
                    }
                    seg.sizing = it->second;
                    wanted     = shared_frames[seg.spec.index];
                }
                const auto grown = seg.residency->resize(wanted, seg.device->stream, *seg.vram);
                if (grown.spilled != 0) {
                    diagnostic(segment_name(seg) + std::to_string(grown.spilled >> 20) +
                                   " MiB of the expert frames were placed in system memory and given back; the expert "
                                   "cache starts with " + std::to_string(grown.frames) + " of " + std::to_string(wanted) +
                                   " frames",
                               DiagnosticLevel::Warning);
                } else if (grown.refused) {
                    diagnostic(segment_name(seg) + "the GPU refused expert-cache memory; the cache starts with " +
                                   std::to_string(grown.frames) + " of " + std::to_string(wanted) + " frames",
                               DiagnosticLevel::Warning);
                }
                seg.sizing.frames = grown.frames;
                if (!seg.spec.shares_device) { start_vram_monitor(seg, demand); }
            }
            segment_sizing_.push_back(seg.sizing);
            execution::ForwardExperts& experts = builds[s].experts;
            experts.frame_base                 = seg.residency->frame_base();
            experts.frame_stride               = seg.residency->frame_stride();
            experts.route_log                  = seg.residency->route_log();
            experts.route_stride               = seg.residency->route_stride();
            for (std::uint32_t l = 0; l < seg.spec.layers.size(); ++l) { experts.frames[l] = seg.residency->table(l); }
            experts.landing       = seg.residency->landing_table();
            experts.landed        = seg.residency->landed_log();
            experts.landing_slots = static_cast<std::int32_t>(ExpertResidency::kLandingSlots);
            if (tier_) {
                for (std::uint32_t l = 0; l < seg.spec.layers.size(); ++l) {
                    experts.host_tables.push_back(seg.residency->record_table(l));
                    experts.fetch.push_back(fetch_channel_->channel(static_cast<int>(seg.spec.layers.begin + l)));
                }
            }
        }
        warm_start_experts();
        if (!options_.route_trace.empty()) {
            // A split's frames are its segments' (each caches its own layers).
            std::uint32_t frames = 0;
            for (const auto& segment : segments_) { frames += segment->residency->frames(); }
            trace_ = std::make_unique<RouteTrace>(
                options_.route_trace,
                RouteTraceSetup{.layers             = c_.num_hidden_layers,
                                .experts            = c_.moe.experts,
                                .top_k              = c_.moe.top_k,
                                .frames             = frames,
                                .max_columns        = static_cast<std::uint32_t>(columns_),
                                .lanes              = options_.max_concurrency,
                                .max_width          = static_cast<std::uint32_t>(max_width_),
                                .mtp_draft_tokens   = static_cast<std::uint32_t>(mtp_k_),
                                .ngram_draft_tokens = options_.ngram_draft_tokens,
                                .prefill_chunk      = static_cast<std::uint32_t>(chunk_)});
        }
        // A split's boundaries: per pair of consecutive segments, the channel of a decode or
        // verification round's residual (every lane at the widest round) and the bulk crossing of
        // an eager call's.
        if (split) {
            const std::size_t round_bytes = 2ULL * static_cast<std::size_t>(width_) * static_cast<std::size_t>(lanes) *
                                            static_cast<std::size_t>(max_width_);
            for (std::size_t s = 0; s + 1 < segments_.size(); ++s) {
                DeviceSegment& sender   = *segments_[s];
                DeviceSegment& receiver = *segments_[s + 1];
                channels_.push_back(
                    std::make_unique<HandoffChannel>(sender.device->device, receiver.device->device, round_bytes));
                bulk_.push_back(std::make_unique<BulkHandoff>(sender.device->device, receiver.device->device));
                sender.out      = channels_.back().get();
                receiver.in     = channels_.back().get();
                sender.bulk_out = bulk_.back().get();
            }
            for (const auto& segment : segments_) {
                const DeviceGuard on(segment->device->device);
                CUDA_CHECK(cudaEventCreateWithFlags(&segment->compute_done.event, cudaEventDisableTiming));
                CUDA_CHECK(cudaEventCreate(&segment->walk_first.event));
                CUDA_CHECK(cudaEventCreate(&segment->walk_last.event));
                for (auto& step : segment->walk_steps) {
                    CUDA_CHECK(cudaEventCreateWithFlags(&step.event, cudaEventDisableTiming));
                }
            }
        }
        for (std::size_t s = 0; s < segments_.size(); ++s) {
            DeviceSegment& seg = *segments_[s];
            const DeviceGuard on(seg.device->device);
            const execution::ForwardSegment segment{
                .layers         = seg.spec.layers,
                .gdn_base       = seg.spec.gdn_base,
                .attention_base = seg.spec.attention_base,
                .embed          = seg.spec.embed,
                .finish         = seg.spec.head,
                .in             = seg.in,
                .out            = seg.out,
                // A gated round's receiving wait (decode and verification rounds) reports a timeout
                // through the expert error word, with the receiving segment's first layer, and adds
                // its waits to the segment's statistics.
                .in_bound      = PinnedGateBound{.timeout_ns  = 1'000'000ULL * options_.handoff_timeout_ms,
                                                 .error_word  = static_cast<std::uint32_t*>(expert_error_.data()),
                                                 .error_value = (seg.spec.layers.begin << 16U) | kErrorHandoffTimeout},
                .in_wait_stats = static_cast<std::uint64_t*>(seg.handoff_stats.p)};
            seg.forward = std::make_unique<execution::Forward>(*seg.parameters, *seg.device, *seg.work,
                                                               std::move(builds[s].state), std::move(builds[s].kv),
                                                               std::move(builds[s].experts), dim(options_.max_context),
                                                               segment);
        }
        if (moe_offload_) { prepare_moe_offload(); }
        synchronize();
        report_lane_cost(lanes, record_stride);
    }

    // A split's teardown: each segment is destroyed with its device current, after the prefix cache
    // (whose blocks hold pages of the head's pool and whose streams run on every segment's device) and
    // every lane have returned their pages and rows to the segments' pools and tables (an unsplit
    // Program's members are destroyed in declaration order with its one device current).
    ~ProgramImpl() {
        if (segments_.size() < 2) { return; }
        try {
            synchronize();
        } catch (...) {}
        prefix_.cache.reset();
        prefix_.lane_image.reset();
        try {
            const DeviceGuard on(head().device->device);
            prefix_.flush_device = DeviceBuffer{};
        } catch (...) {}
        for (Lane& lane : lanes_) {
            lane.pages.clear();
            lane.reservation.reset();
            lane.row = KVExecutionRowLease{};
            lane.mirror_rows.clear();
        }
        while (!segments_.empty()) {
            try {
                const DeviceGuard on(segments_.back()->device->device);
                segments_.pop_back();
            } catch (...) {
                segments_.pop_back();
            }
        }
    }

    ProgramImpl(const ProgramImpl&)            = delete;
    ProgramImpl& operator=(const ProgramImpl&) = delete;

private:
    // A segment's Forward inputs while the constructor builds them.
    struct SegmentBuild {
        execution::ForwardState state;
        execution::ForwardKV kv;
        execution::ForwardExperts experts;
    };

    static const execution::Parameters& first_parameters(std::span<const execution::Parameters* const> parameters) {
        if (parameters.empty() || parameters.front() == nullptr) {
            throw std::invalid_argument("Qwen4Exp: a Program needs the Parameters of its segments");
        }
        return *parameters.front();
    }

    // One Parameters and one device context per segment, in order: each over the model's layers and
    // role of its segment (Parameters ties its role to its range) on the segment's device.
    void check_segments(std::span<const execution::Parameters* const> parameters,
                        std::span<DeviceContext* const> devices, const SegmentMap& map) const {
        if (parameters.size() != map.size() || devices.size() != map.size()) {
            throw std::invalid_argument("Qwen4Exp: a Program needs one Parameters and one device per segment");
        }
        for (std::size_t s = 0; s < map.size(); ++s) {
            const SegmentSpec& spec              = map[s];
            const execution::Parameters* segment = parameters[s];
            if (segment == nullptr || devices[s] == nullptr || &segment->model != &parameters_.model ||
                segment->layer_range().begin != spec.layers.begin || segment->layer_range().end != spec.layers.end ||
                segment->role().embed != spec.embed || segment->role().head != spec.head ||
                devices[s]->device != spec.device) {
                throw std::invalid_argument("Qwen4Exp: the Parameters or device of segment " + std::to_string(s) +
                                            " do not match its layers, role or device");
            }
        }
    }

    // The frames of each segment on `device` (the same-device test mode) out of the device's `total`:
    // in proportion to their layers (frame_share), within each one's limit, but at least what lets a
    // segment lend its wide arena (a quarter of the frames is never lent) when its limit allows,
    // taken from the segments with more than they need.
    void share_device_frames(int device, std::uint32_t total, std::vector<std::uint32_t>& frames) const {
        std::vector<std::uint32_t> need(segments_.size(), 0);
        for (const auto& segment : segments_) {
            const DeviceSegment& seg = *segment;
            if (seg.spec.device != device) { continue; }
            const std::uint32_t limit = seg.residency->frame_limit();
            const std::uint64_t stride = seg.plan.record_stride;
            const auto wide_frames     = static_cast<std::uint32_t>((seg.wide_work + stride - 1) / stride);
            frames[seg.spec.index] =
                std::min(limit, static_cast<std::uint32_t>(static_cast<double>(total) * seg.spec.frame_share));
            if (seg.wide_work > 0 && !seg.wide_dedicated) {
                need[seg.spec.index] = std::min(limit, wide_frames + wide_frames / 3 + 2);
            }
        }
        for (std::size_t s = 0; s < segments_.size(); ++s) {
            if (frames[s] >= need[s]) { continue; }
            std::uint32_t missing = need[s] - frames[s];
            frames[s]             = need[s];
            for (std::size_t o = 0; o < segments_.size() && missing > 0; ++o) {
                if (o == s || segments_[o]->spec.device != device || frames[o] <= need[o]) { continue; }
                const std::uint32_t taken = std::min(missing, frames[o] - need[o]);
                frames[o] -= taken;
                missing -= taken;
            }
        }
    }

    // A diagnostic's prefix naming a split's segment (empty without a split).
    std::string segment_name(const DeviceSegment& seg) const {
        if (segments_.size() < 2) { return {}; }
        return "segment " + std::to_string(seg.spec.index) + " (layers " + std::to_string(seg.spec.layers.begin) + "-" +
               std::to_string(seg.spec.layers.end - 1) + ", GPU " + std::to_string(seg.spec.device) + "): ";
    }

    // A segment's fixed device allocations, on its device: exactly its plan, which the driver must
    // place in device memory (design §19.3.7); with the Forward's state and KV views over them, its
    // expert cache over its own layers' banks (local order), its staging and its overlap stream. The
    // Program's host buffers come with the segment whose role reads them (the io with the first).
    void build_segment(DeviceSegment& seg, SegmentBuild& build) {
        const DeviceGuard on(seg.device->device);
        const SegmentSpec& spec  = seg.spec;
        const std::int32_t lanes = seg.plan.lanes;
        seg.vram                 = open_vram_budget_source(seg.device->device, options_.vram_past_budget);
        SpillGuard guard(*seg.vram);
        guard.begin();
        std::uint64_t allocated = 0;

        allocate(seg.state_backing, seg.plan.gdn_bytes, allocated);
        seg.state_backing.fill(0);
        if (spec.gdn_count > 0) {
            seg.gdn = std::make_unique<LinearAttentionStatePool>(
                DeviceSpan{seg.state_backing.p, seg.state_backing.bytes}, seg.plan.gdn);
        }
        allocate(seg.ple_backing, seg.plan.ple, allocated);
        seg.ple_backing.fill(0);
        allocate(seg.tails_backing, seg.plan.tails, allocated);
        seg.tails_backing.fill(0);
        seg.window_geometry = prefix::kv_window_geometry(c_, options_.kv_cache, spec.attention_count, spec.mtp,
                                                         static_cast<std::uint32_t>(options_.max_concurrency));
        if (seg.window_geometry.present()) {
            allocate(seg.window_backing, seg.plan.window, allocated);
            seg.window_backing.fill(0); // zero tags never match a row's (tags are odd)
        }

        // The segment's KV planes over the page indices the head's pool hands out (none without
        // attention layers).
        if (seg.plan.kv_layers > 0) {
            seg.kv_geometry = prefix::kv_page_geometry(c_, options_.kv_cache, spec.attention_count, spec.mtp);
            if (seg.plan.kv_elastic) {
                seg.kv_vmm  = std::make_unique<VmmRange>(seg.device->device, seg.plan.kv, kKvChunkBytes);
                seg.kv_span = DeviceSpan{seg.kv_vmm->base(), seg.plan.kv};
                if (!map_kv(seg, seg.plan.kv_base_pages)) {
                    throw std::runtime_error("Qwen4Exp: the KV pool's base pages do not fit");
                }
                allocated += seg.kv_vmm->mapped_bytes();
            } else {
                allocate(seg.kv_backing, seg.plan.kv, allocated);
                seg.kv_backing.fill(0);
                seg.kv_span = DeviceSpan{seg.kv_backing.p, seg.kv_backing.bytes};
            }
            seg.pool   = std::make_unique<DeviceKVPagePool>(seg.kv_span, seg.plan.pool);
            seg.tables = std::make_unique<KVExecutionTablePool>(seg.kv_span, seg.plan.tables, *seg.pool);
            if (seg.plan.kv_elastic) { seg.pool->set_backed_pages(seg.plan.kv_base_pages); }
        }

        allocate(seg.io_device, io_layout_.bytes, allocated);
        if (spec.index == 0) { io_host_ = PinnedHostBuffer(io_layout_.bytes); }
        if (spec.head) {
            allocate(seg.logits32, seg.plan.logits32, allocated);
            allocate(seg.logits16, seg.plan.logits16, allocated);
            allocate(seg.sampled, 4ULL * lanes, allocated);
            allocate(seg.sample_pos, 4ULL * lanes, allocated);
            allocate(seg.configs, sizeof(ops::SamplingConfig) * lanes, allocated);
            allocate(seg.token_counts, seg.plan.token_counts, allocated);
            seg.token_counts.fill(0);
            host_sampled_    = PinnedHostBuffer(4ULL * lanes);
            host_sample_pos_ = PinnedHostBuffer(4ULL * lanes);
            host_configs_    = PinnedHostBuffer(sizeof(ops::SamplingConfig) * lanes);
        }
        if (spec.ple) {
            allocate(seg.gate_stats, kGateStatsBytes * lanes, allocated);
            seg.gate_stats.fill(0);
        }
        if (spec.head) {
            allocate(seg.constraint, constraint_bytes(lanes, max_width_), allocated);
            allocate(seg.grammar_masks, grammar_mask_bytes(token_domain_, lanes, max_width_), allocated);
            grammar_host_ = PinnedHostBuffer(grammar_mask_bytes(token_domain_, lanes, max_width_));
            constraint_host_.assign(static_cast<std::size_t>(lanes * max_width_), -1);
        }
        if (spec.ple) { publish_pinned_word(gate_ready(), 0); } // pinned memory starts undefined; 0 is never a round's word
        if (!spec.embed) { // a split's receiving segment: its gated handoff's wait statistics
            allocate(seg.handoff_stats, kHandoffStatsBytes, allocated);
            seg.handoff_stats.fill(0);
        }

        execution::ForwardState& state = build.state;
        state.gdn                      = seg.gdn.get();
        if (spec.ple) { state.ple_conv = Tensor(seg.ple_backing.p, DType::BF16, {width_, span_, lanes}); }
        for (std::uint32_t l = 0; l < spec.attention_count; ++l) {
            state.qsa_tails.push_back(Tensor(static_cast<std::byte*>(seg.tails_backing.p) +
                                                 static_cast<std::size_t>(l) * di_ * (r_ - 1) * lanes * 2,
                                             DType::BF16, {di_, r_ - 1, lanes}));
        }
        const auto layout   = paged_kv_storage_layout(options_.kv_cache, dim(c_.attention.head_dim));
        const auto kv_heads = dim(c_.attention.kv_heads);
        execution::ForwardKV& kv = build.kv;
        if (seg.tables) { kv.block_tables = seg.tables->matrix(); }
        for (std::uint32_t l = 0; l < seg.plan.kv_layers; ++l) {
            const prefix::PlaneRange planes = prefix::kv_layer_planes(seg.kv_geometry, l);
            std::size_t plane               = planes.begin;
            ops::QsaKVLayer layer;
            layer.kv.storage      = options_.kv_cache;
            layer.kv.head_dim     = dim(c_.attention.head_dim);
            layer.kv.num_kv_heads = kv_heads;
            layer.kv.k_pages      = seg.pool->plane(plane++);
            if (layout.key.has_scale()) { layer.kv.k_scale_pages = seg.pool->plane(plane++); }
            layer.kv.v_pages = seg.pool->plane(plane++);
            if (layout.value.has_scale()) { layer.kv.v_scale_pages = seg.pool->plane(plane++); }
            layer.pooled_pages = seg.pool->plane(plane++);
            if (seg.window_geometry.present()) {
                layer.kv.window = prefix::kv_window_view(seg.window_geometry, seg.window_backing.p, l);
            }
            if (plane != planes.end) { throw std::logic_error("Qwen4Exp: KV planes do not match the page geometry"); }
            if (l < spec.attention_count) {
                kv.layers.push_back(layer);
            } else {
                kv.mtp = layer;
            }
        }
        if (spec.mtp) {
            state.mtp_tails = Tensor(static_cast<std::byte*>(seg.tails_backing.p) +
                                         static_cast<std::size_t>(spec.attention_count) * di_ * (r_ - 1) * lanes * 2,
                                     DType::BF16, {di_, r_ - 1, lanes});
            allocate_mtp(seg, lanes, allocated);
            state.mtp_ones = Tensor(seg.mtp_ones.p, DType::FP32, {dim(c_.hc.streams), mtp_columns_});
        }
        execution::ForwardExperts& experts = build.experts;
        experts.frames.assign(spec.layers.size(), nullptr);
        if (max_width_ > 1) { allocate_verification(seg, lanes, allocated); }
        seg.work_capacity = seg.plan.work;
        seg.wide_work     = seg.plan.wide_work;
        seg.work          = std::make_unique<WorkspaceArena>(seg.work_capacity);
        allocated += seg.work_capacity;

        // The segment's own layers, in local order (layer - range.begin): the others have no bank.
        std::vector<const std::uint8_t*> banks;
        std::uint64_t record_stride = 0;
        for (std::uint32_t l = spec.layers.begin; l < spec.layers.end; ++l) {
            const auto& bank = *seg.parameters->layers[l].moe.bank;
            banks.push_back(reinterpret_cast<const std::uint8_t*>(bank.planes.records));
            record_stride = bank.planes.record_stride;
        }
        if (record_stride != seg.plan.record_stride) {
            throw std::logic_error("Qwen4Exp: the expert banks' record stride differs from their format's");
        }
        // Staging slots for each layer call's misses (copied with a compact read window, design
        // 8.6), allocated before the frames take the remaining memory.
        allocate(seg.staging, seg.plan.staging, allocated);
        experts.staging_base  = static_cast<std::uint8_t*>(seg.staging.p);
        experts.staging_slots = kStagingSlots;
        // The expert cache's tables and route log; its frames come last (the constructor sizes them
        // once every segment's fixed allocations are made).
        seg.residency =
            std::make_unique<ExpertResidency>(c_, std::move(banks), record_stride, columns_, seg.device->device);
        allocated += seg.plan.bytes.residency;
        if (allocated != seg.plan.bytes.fixed_bytes()) {
            throw std::logic_error("Qwen4Exp: the Program's device allocations differ from its device plan");
        }
        if (const std::uint64_t spilled = guard.end(allocated); spilled != 0) {
            throw std::runtime_error(segment_name(seg) + spill_message(spilled));
        }
        CUDA_CHECK(cudaStreamCreateWithFlags(&seg.overlap.stream, cudaStreamNonBlocking));
        for (auto& event : seg.overlap.events) {
            CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
        }
        experts.overlap_stream = seg.overlap.stream;
        experts.overlap_events = seg.overlap.events;
    }

    // A segment's prefill expert stream over its own layers (local order, as its Forward's): records
    // from its layers' banks, or from the SSD tier by global key.
    std::unique_ptr<execution::ExpertStream> make_expert_stream(DeviceSegment& seg) {
        const LayerRange owned            = seg.spec.layers;
        const std::uint64_t record_stride = seg.plan.record_stride;
        execution::ExpertStream::RecordOf record_of;
        if (tier_) {
            record_of = [tier = tier_.get(), first = owned.begin](std::uint32_t layer, std::uint32_t expert) {
                return tier->record(tier->key(first + layer, expert));
            };
        } else {
            std::vector<const std::uint8_t*> banks; // local order, as the Forward's layers
            for (std::uint32_t l = owned.begin; l < owned.end; ++l) {
                banks.push_back(
                    reinterpret_cast<const std::uint8_t*>(seg.parameters->layers[l].moe.bank->planes.records));
            }
            record_of = [banks = std::move(banks), record_stride](std::uint32_t layer, std::uint32_t expert) {
                return banks[layer] + static_cast<std::size_t>(expert) * record_stride;
            };
        }
        execution::ExpertStream::Hold hold;
        if (tier_) {
            hold = [tier = tier_.get(), first = owned.begin](std::uint32_t layer, std::uint32_t expert) {
                tier->stream_pin(tier->key(first + layer, expert));
            };
        }
        return std::make_unique<execution::ExpertStream>(owned.size(), c_.moe.experts, record_stride,
                                                         std::move(record_of), !tier_, std::move(hold));
    }

public:
    // The first segment's device plan and frame sizing (the whole model's without a split).
    const DeviceLayout& device_layout() const noexcept { return first().plan; }
    const VramSizing& vram_sizing() const noexcept { return first().sizing; }
    std::span<const VramSizing> segment_vram_sizing() const noexcept { return segment_sizing_; }

    // The segments by role (SegmentMap): the first embeds; the PLE segment computes the PLE and its
    // n-gram gate; the head (the last) runs the final mixer, the head, sampling, acceptance and the
    // MTP drafter and allocates the KV pages. whole() is the one segment of an unsplit Program, which
    // runs every layer: the paths a split does not run yet (the SSD tier) reach the layers' state, KV,
    // expert cache and Forward through it; on a split Program it throws std::logic_error, so such a
    // path fails instead of running on segment 0 alone.
    DeviceSegment& first() noexcept { return *segments_.front(); }
    const DeviceSegment& first() const noexcept { return *segments_.front(); }
    DeviceSegment& head() noexcept { return *segments_.back(); }
    const DeviceSegment& head() const noexcept { return *segments_.back(); }
    DeviceSegment& ple() noexcept { return *segments_[ple_segment_]; }
    const DeviceSegment& ple() const noexcept { return *segments_[ple_segment_]; }
    DeviceSegment& whole() { return *segments_[whole_index()]; }
    const DeviceSegment& whole() const { return *segments_[whole_index()]; }
    std::size_t whole_index() const {
        if (segments_.size() != 1) {
            throw std::logic_error("Qwen4Exp: a path that drives one segment ran on a split Program");
        }
        return 0;
    }

    // Waits for every segment's compute stream: the head first (its sampling ends a round), then the
    // others.
    void synchronize() const {
        {
            const DeviceGuard on(head().device->device);
            head().device->synchronize();
        }
        for (const auto& segment : segments_) {
            if (segment.get() == &head()) { continue; }
            const DeviceGuard on(segment->device->device);
            segment->device->synchronize();
        }
    }
    // Submits every segment's queued work without waiting for it.
    void flush() const {
        for (const auto& segment : segments_) {
            const DeviceGuard on(segment->device->device);
            segment->device->flush();
        }
    }
    // Whether `a` and `b` run on one device (the same-device test mode of a layer split).
    static bool share_device(const DeviceSegment& a, const DeviceSegment& b) noexcept {
        return &a != &b && a.spec.device == b.spec.device;
    }

    // Work enqueued on one segment's stream, with its device current. On a GPU that runs two
    // segments (the same-device test mode) their kernels never overlap, as the MoE's bits depend on
    // it (design §19.3.8, "F1 nondeterminism"): the work first waits for what the device's other
    // segments enqueued last and, once enqueued, records the segment's compute_done for them. On
    // its own GPU a segment's work only selects the device.
    class SegmentWork {
    public:
        SegmentWork(const ProgramImpl& program, DeviceSegment& segment)
            : guard_(segment.device->device), segment_(segment) {
            if (!segment.spec.shares_device) { return; }
            for (const auto& other : program.segments_) {
                if (share_device(segment, *other)) {
                    CUDA_CHECK(cudaStreamWaitEvent(segment.device->stream, other->compute_done.event, 0));
                }
            }
        }
        ~SegmentWork() {
            if (segment_.spec.shares_device) {
                (void)cudaEventRecord(segment_.compute_done.event, segment_.device->stream);
            }
        }
        SegmentWork(const SegmentWork&)            = delete;
        SegmentWork& operator=(const SegmentWork&) = delete;

    private:
        DeviceGuard guard_;
        DeviceSegment& segment_;
    };

    // Debugging aid INFERNIX_SPLIT_SYNC=1 (a split only): every segment is synchronized before the
    // host restages the pinned io, round arrays or drafter io, so a restage can never race a later
    // segment's upload still queued (each round already synchronizes before its successor stages).
    static bool split_sync_from_environment() {
        const char* value = std::getenv("INFERNIX_SPLIT_SYNC");
        if (value == nullptr || *value == '\0' || std::strcmp(value, "0") == 0) { return false; }
        if (std::strcmp(value, "1") == 0) { return true; }
        throw std::invalid_argument(std::string("INFERNIX_SPLIT_SYNC must be 0 or 1, not '") + value + "'");
    }
    void before_restage() const {
        if (split_sync_) { synchronize(); }
    }

    // How a layer split's long prompts run (prefill_walk.cpp), INFERNIX_SPLIT_WALK (outputs are the
    // same bits in every mode):
    //   1 (the default) walks them layer-major on every segment (design §19.3.8 F4, multi-GPU WP16),
    //     every prefix tap ending a span as on an unsplit Program, so the snapshots are the same;
    //   2 ends a span at exact taps only: a flexible tap is realized where the span ends (fewer
    //     spans, a long prompt's history ladder snapshots later or not at all);
    //   0 keeps them chunk-major, every call crossing the segments in turn.
    enum class SplitWalk : std::uint8_t { ChunkMajor = 0, EveryTap = 1, ExactTaps = 2 };
    // INFERNIX_SPLIT_MOE_OFFLOAD (a split of two segments that walks): 1 (the default) lets the
    // second segment compute the first one's MoE (routed and shared experts) in the layer walk
    // (multi-GPU WP23, walk_offload_layers), 0 keeps it on the first segment, which streams its
    // routed experts over its own link. The same bits either way.
    static bool moe_offload_from_environment() {
        const char* value = std::getenv("INFERNIX_SPLIT_MOE_OFFLOAD");
        if (value == nullptr || *value == '\0' || std::strcmp(value, "1") == 0) { return true; }
        if (std::strcmp(value, "0") == 0) { return false; }
        throw std::invalid_argument(std::string("INFERNIX_SPLIT_MOE_OFFLOAD must be 0 or 1, not '") + value + "'");
    }
    static SplitWalk split_walk_from_environment() {
        const char* value = std::getenv("INFERNIX_SPLIT_WALK");
        if (value == nullptr || *value == '\0' || std::strcmp(value, "1") == 0) { return SplitWalk::EveryTap; }
        if (std::strcmp(value, "2") == 0) { return SplitWalk::ExactTaps; }
        if (std::strcmp(value, "0") == 0) { return SplitWalk::ChunkMajor; }
        throw std::invalid_argument(std::string("INFERNIX_SPLIT_WALK must be 0, 1 or 2, not '") + value + "'");
    }

    // Measurement knob INFERNIX_SPLIT_TIMING=1 (a split, or one GPU as its reference): where a
    // decode or verification round's time goes, reported at each request's end (debug level) as
    // means per round: the round period (launch to launch), the launch-to-synchronization wall time
    // and within it each segment's graph on its GPU (timing events around its replay, after its
    // handoff wait), and the host's drafting and commit (with the cache update) between rounds;
    // then the same per verified width (1 for a plain round), with the draft steps run before the
    // round, from which a verified column's marginal cost follows (choose_draft_length). Costs two
    // event records per segment and round.
    static bool split_timing_from_environment() {
        const char* value = std::getenv("INFERNIX_SPLIT_TIMING");
        if (value == nullptr || *value == '\0' || std::strcmp(value, "0") == 0) { return false; }
        if (std::strcmp(value, "1") == 0) { return true; }
        throw std::invalid_argument(std::string("INFERNIX_SPLIT_TIMING must be 0 or 1, not '") + value + "'");
    }
    // INFERNIX_SPLIT_TIMING's host marks of a round's phases, in their usual order.
    enum class Mark : std::uint8_t {
        Synced,        // the round's synchronization returned (timing_synchronized)
        CommitBegin,   // commit() entered
        SettleBegin,   // settle_round() entered (the cache update)
        Settled,       // settle_round() returned
        DecodeBegin,   // decode() entered
        DraftLaunched, // the draft graph and its downloads are enqueued
        DraftSynced,   // the draft's synchronization returned
        LaunchBegin,   // launch_round() entered
        FirstLaunched, // the first segment's graph launch returned
        Launched,      // launch_round() returned
        Count
    };
    static constexpr std::size_t kMarkCount = static_cast<std::size_t>(Mark::Count);
    static constexpr std::array<const char*, kMarkCount> kMarkNames = {
        "synced", "commit", "settle", "settled", "decode", "draft-launched", "draft-synced", "launch",
        "first-launched", "launched"};
    struct SplitTiming {
        bool on    = false;
        bool armed = false; // the round in flight recorded its events
        std::uint64_t rounds = 0, periods = 0;
        double period_ms = 0, wall_ms = 0, draft_ms = 0, commit_ms = 0;
        std::vector<double> graph_ms; // per segment
        Clock::time_point launched{}, previous{};
        // The drafting before the round in flight (its host span and draft steps).
        double round_draft_ms    = 0;
        std::int32_t round_steps = 0;
        // Per verified width W (index W - 1): sums over its rounds.
        struct Width {
            std::uint64_t rounds = 0, steps = 0;
            double wall_ms = 0, draft_ms = 0;
            std::vector<double> graph_ms; // per segment
        };
        std::vector<Width> widths;
        // Host marks between rounds (Mark): the time from each mark to the next one taken, summed
        // per transition, so the round's critical path between the head's synchronization and the
        // first segment's launch is broken down.
        Mark last_mark = Mark::Count;
        Clock::time_point last_at{};
        std::array<std::array<double, kMarkCount>, kMarkCount> mark_ms{};
        std::array<std::array<std::uint64_t, kMarkCount>, kMarkCount> mark_n{};
        // The head's drafter on its GPU: the catch-up (commit) and the draft graph, and the head's
        // idle gap between them; sums and counts over the rounds that ran them.
        bool catch_recorded = false, draft_recorded = false;
        double catch_ms = 0, gap_ms = 0, steps_ms = 0;
        std::uint64_t catch_n = 0, gap_n = 0, steps_n = 0;
    };
    SplitTiming timing_;
    // Adds the host time of its scope to `total` and stores it in `last` (none when `total` is
    // null).
    struct HostSpan {
        explicit HostSpan(double* total, double* last = nullptr) : total_(total), last_(last) {}
        ~HostSpan() {
            if (total_ != nullptr) {
                const double ms = std::chrono::duration<double, std::milli>(Clock::now() - start_).count();
                *total_ += ms;
                if (last_ != nullptr) { *last_ = ms; }
            }
        }
        HostSpan(const HostSpan&)            = delete;
        HostSpan& operator=(const HostSpan&) = delete;

    private:
        double* total_;
        double* last_;
        Clock::time_point start_ = Clock::now();
    };
    // Takes host mark `m`: adds the time since the last mark to their transition.
    void mark(Mark m) noexcept {
        if (!timing_.on) { return; }
        const auto now = Clock::now();
        if (timing_.last_mark != Mark::Count) {
            const auto from = static_cast<std::size_t>(timing_.last_mark), to = static_cast<std::size_t>(m);
            timing_.mark_ms[from][to] += std::chrono::duration<double, std::milli>(now - timing_.last_at).count();
            ++timing_.mark_n[from][to];
        }
        timing_.last_mark = m;
        timing_.last_at   = now;
    }
    // Records the head's drafter event `e` (catch_begin .. draft_end) on its stream.
    void drafter_event(SegmentEvent& e, bool& recorded) {
        if (!timing_.on) { return; }
        CUDA_CHECK(cudaEventRecord(e.event, head().device->stream));
        recorded = true;
    }
    // After the draft's synchronization: the catch-up, the head's gap before the draft graph, and
    // the draft graph, each when recorded since the last.
    void drafter_synchronized() {
        if (!timing_.on) { return; }
        const DeviceSegment& h = head();
        float ms = 0.0F;
        if (timing_.catch_recorded) {
            CUDA_CHECK(cudaEventElapsedTime(&ms, h.catch_begin.event, h.catch_end.event));
            timing_.catch_ms += ms;
            ++timing_.catch_n;
            if (timing_.draft_recorded) {
                CUDA_CHECK(cudaEventElapsedTime(&ms, h.catch_end.event, h.draft_begin.event));
                timing_.gap_ms += ms;
                ++timing_.gap_n;
            }
        }
        if (timing_.draft_recorded) {
            CUDA_CHECK(cudaEventElapsedTime(&ms, h.draft_begin.event, h.draft_end.event));
            timing_.steps_ms += ms;
            ++timing_.steps_n;
        }
        timing_.catch_recorded = timing_.draft_recorded = false;
    }
    void timing_launch() {
        if (!timing_.on) { return; }
        const auto now = Clock::now();
        if (timing_.previous != Clock::time_point{}) {
            timing_.period_ms += std::chrono::duration<double, std::milli>(now - timing_.previous).count();
            ++timing_.periods;
        }
        timing_.previous = timing_.launched = now;
        timing_.armed    = true;
        mark(Mark::LaunchBegin);
    }
    // After the synchronization of a round of verified width `width` (1: plain).
    void timing_synchronized(std::int32_t width) {
        if (!timing_.on || !timing_.armed) { return; }
        timing_.armed = false;
        mark(Mark::Synced);
        if (timing_.widths.size() < static_cast<std::size_t>(width)) {
            timing_.widths.resize(static_cast<std::size_t>(width));
        }
        SplitTiming::Width& w = timing_.widths[static_cast<std::size_t>(width - 1)];
        w.graph_ms.resize(segments_.size(), 0.0);
        const double wall = std::chrono::duration<double, std::milli>(Clock::now() - timing_.launched).count();
        timing_.wall_ms += wall;
        w.wall_ms += wall;
        w.draft_ms += timing_.round_draft_ms;
        w.steps += static_cast<std::uint64_t>(timing_.round_steps);
        ++w.rounds;
        timing_.round_draft_ms = 0;
        timing_.round_steps    = 0;
        for (std::size_t s = 0; s < segments_.size(); ++s) {
            const DeviceSegment& seg = *segments_[s];
            const DeviceGuard on(seg.device->device);
            float ms = 0.0F;
            CUDA_CHECK(cudaEventElapsedTime(&ms, seg.timing_begin.event, seg.timing_end.event));
            timing_.graph_ms[s] += ms;
            w.graph_ms[s] += ms;
        }
        ++timing_.rounds;
    }
    void report_split_timing() noexcept {
        if (!timing_.on) { return; }
        try {
            if (timing_.rounds > 0) {
                const auto n    = static_cast<double>(timing_.rounds);
                const double p  = timing_.periods ? timing_.period_ms / static_cast<double>(timing_.periods) : 0.0;
                const double w  = timing_.wall_ms / n, d = timing_.draft_ms / n, c = timing_.commit_ms / n;
                std::string line = "split round timing (INFERNIX_SPLIT_TIMING), " + std::to_string(timing_.rounds) +
                                   " rounds, mean ms: period " + std::to_string(p) + " = launch to sync " +
                                   std::to_string(w) + " (";
                for (std::size_t s = 0; s < segments_.size(); ++s) {
                    line += (s ? ", " : "") + std::string("segment ") + std::to_string(s) + " graph " +
                            std::to_string(timing_.graph_ms[s] / n);
                }
                line += ") + draft " + std::to_string(d) + " + commit " + std::to_string(c) + " + other " +
                        std::to_string(p - w - d - c);
                // Per verified width: rounds, mean draft steps and drafting before the round, mean
                // launch-to-sync wall time and graphs.
                line += "; by width:";
                for (std::size_t i = 0; i < timing_.widths.size(); ++i) {
                    const SplitTiming::Width& x = timing_.widths[i];
                    if (x.rounds == 0) { continue; }
                    const auto m = static_cast<double>(x.rounds);
                    line += " W" + std::to_string(i + 1) + " n " + std::to_string(x.rounds) + " steps " +
                            std::to_string(static_cast<double>(x.steps) / m) + " draft " +
                            std::to_string(x.draft_ms / m) + " wall " + std::to_string(x.wall_ms / m) + " (graphs";
                    for (const double g : x.graph_ms) { line += " " + std::to_string(g / m); }
                    line += ");";
                }
                // Between rounds: each host transition's mean and count, then the head's drafter.
                line += " between rounds (host, mean ms):";
                for (std::size_t from = 0; from < kMarkCount; ++from) {
                    for (std::size_t to = 0; to < kMarkCount; ++to) {
                        const std::uint64_t k = timing_.mark_n[from][to];
                        if (k == 0) { continue; }
                        line += " " + std::string(kMarkNames[from]) + "->" + kMarkNames[to] + " " +
                                std::to_string(timing_.mark_ms[from][to] / static_cast<double>(k)) + " (" +
                                std::to_string(k) + ")";
                    }
                }
                const auto mean = [](double sum, std::uint64_t k) { return k ? sum / static_cast<double>(k) : 0.0; };
                line += "; head drafter (GPU, mean ms): catch-up " + std::to_string(mean(timing_.catch_ms, timing_.catch_n)) +
                        " (" + std::to_string(timing_.catch_n) + "), idle before the draft " +
                        std::to_string(mean(timing_.gap_ms, timing_.gap_n)) + ", draft graph " +
                        std::to_string(mean(timing_.steps_ms, timing_.steps_n)) + " (" + std::to_string(timing_.steps_n) +
                        ")";
                diagnostic(line, DiagnosticLevel::Debug);
            }
        } catch (...) {}
        const bool on = timing_.on;
        std::vector<double> graphs(segments_.size(), 0.0);
        timing_          = SplitTiming{};
        timing_.on       = on;
        timing_.graph_ms = std::move(graphs);
    }

    // ---------------------------------------------------------------- admission
    RequestBasePlan plan_request(PreparedPrompt&& prompt, const runtime::ResolvedExecutionOptions& options) {
        const auto& data = qwen3_5::PreparedPromptAccess::view(prompt);
        std::shared_ptr<const VisionAdmission> vision;
        if (data.has_media()) { vision = plan_vision(data); }
        const auto n = static_cast<std::uint32_t>(data.token_ids.size());
        if (n == 0 || n > options_.max_context) {
            throw std::invalid_argument("Qwen4Exp: the prompt is empty or exceeds max_context");
        }
        for (const TokenId id : options.readout_tokens) {
            if (id < 0 || id >= token_domain_) {
                throw std::invalid_argument("readout token is outside the public token domain");
            }
        }
        auto base                       = std::make_unique<BasePlanImpl>();
        base->readout_tokens            = options.readout_tokens;
        base->constraint                = compile_constraint(options.constraint);
        base->summary.prompt_tokens     = n;
        base->summary.requested_output_tokens = options.requested_output_tokens;
        const std::uint32_t capacity_output   = options_.max_context - n + 1U;
        base->summary.effective_output_tokens = std::min(options.requested_output_tokens, capacity_output);
        base->summary.effective_limit_reason  = options.requested_output_tokens <= capacity_output
                                                    ? FinishReason::OutputLimit
                                                    : FinishReason::ContextCapacity;
        base->reuse = prefix_.cache != nullptr && options.allow_prefix_reuse && data.identity.reusable;
        base->summary.publish_continuation    = base->reuse;
        base->vision   = std::move(vision);
        base->sampling = translate(options.sampling);
        const std::uint32_t positions = std::min(options_.max_context, n + base->summary.effective_output_tokens);
        base->positions = positions;
        base->pages     = (positions + kPagedKVPageSize - 1) / kPagedKVPageSize;
        base->prompt = std::make_shared<const PreparedPrompt>(std::move(prompt));
        return RequestBasePlan(std::move(base));
    }

    bool feasible(const RequestBasePlan& base) const noexcept {
        const bool frames = !base.impl_ || !base.impl_->vision || vision_fits(base.impl_->vision->window);
        return base.impl_ != nullptr && frames && base.impl_->pages <= head().pool->capacity_pages() &&
               base.impl_->pages <= static_cast<std::uint32_t>(pages_per_row_);
    }

    // The positions one round may write past the lane's frontier: a verification's columns and the
    // MTP drafter's cells (design §19.3.13).
    std::uint32_t round_positions() const noexcept {
        return static_cast<std::uint32_t>(std::max(max_width_, mtp_ ? mtp_k_ + 1 : 1)) + 1U;
    }

    // The KV pages a binding of `base` whose prefill ends at `end` reserves: through one round past
    // it, within the request's extent.
    std::uint32_t binding_pages(const BasePlanImpl& base, std::uint32_t end) const noexcept {
        const std::uint32_t positions = std::min(base.positions, end + round_positions());
        return (positions + kPagedKVPageSize - 1) / kPagedKVPageSize;
    }

    static PrefixReusePath reuse_path_for(runtime::prefix_cache::SnapshotKind kind) noexcept {
        return kind == runtime::prefix_cache::SnapshotKind::Endpoint ? PrefixReusePath::HybridEndpoint
                                                                     : PrefixReusePath::HybridSnapshot;
    }

    // A source over `selection` (absent: a root start without a prefix cache).
    SourceCandidate make_source(const BasePlanImpl& base, const qwen3_5::PreparedPromptData& data,
                                std::optional<PrefixSelection> selection) const {
        auto impl      = std::make_shared<QuoteImpl>();
        impl->summary  = base.summary;
        impl->sampling = base.sampling;
        impl->pages    = binding_pages(base, base.summary.prompt_tokens);
        impl->vision   = base.vision;
        SourceCandidate out;
        std::size_t calls = plan_prefill(data, 0, {}, base.reuse).ends.size();
        if (selection) {
            if (selection->snapshot) {
                // Items inside the reused prefix are not encoded again (design §19.3.2).
                if (impl->vision && selection->frontier > 0) { impl->vision = plan_vision(data, selection->frontier); }
                out.reused_tokens = selection->frontier;
                out.reuse_path    = reuse_path_for(prefix_.cache->index().snapshot(*selection->snapshot).kind);
            }
            calls        = selection->plan.ends.size();
            impl->prefix = std::move(selection);
        }
        out.remaining_work.chunks = calls;
        out.remaining_work.tokens = base.summary.prompt_tokens - out.reused_tokens;
        out.hybrid                = std::move(impl);
        return out;
    }

    // The prefix cache's choice for the request (without a lane: a lane-resident snapshot is credited
    // at start_binding), then a root start.
    std::vector<SourceCandidate> hybrid_sources(const RequestBasePlan& base, std::uint32_t maximum_frontier) {
        const BasePlanImpl& b = *base.impl_;
        const auto& data      = qwen3_5::PreparedPromptAccess::view(*b.prompt);
        std::vector<SourceCandidate> out;
        if (prefix_.cache) {
            std::optional<PrefixSelection> chosen = prefix_select(data, b, kNoLane);
            const std::uint32_t reuse = chosen && chosen->snapshot ? chosen->frontier : 0U;
            // Coalescing only defers a fresh request (an empty list: the Engine asks again).
            if (b.reuse && maximum_frontier == UINT32_MAX && prefix_await_sibling(data, reuse)) { return {}; }
            if (chosen && chosen->snapshot && chosen->frontier > 0 && chosen->frontier <= maximum_frontier) {
                out.push_back(make_source(b, data, std::move(chosen)));
            }
            out.push_back(make_source(b, data, prefix_root(data, b)));
        } else {
            out.push_back(make_source(b, data, std::nullopt));
        }
        return out;
    }

    // Stages the binding of `source` on `lane`: re-selects against the cache as it is now (crediting
    // a lane-resident snapshot), reserves the whole KV extent and maps the cached blocks. The Begin
    // the Engine published names the source's frontier, so a changed choice is refused and the Engine
    // falls back to the next source (the root).
    runtime::ResourceReservation start_binding(const RequestBasePlan& base, runtime::LaneId destination,
                                               const SourceCandidate& source, ResumeState* resume) {
        if (source.hybrid == nullptr) { throw std::logic_error("Qwen4Exp: a binding source carries no quote"); }
        if (resume != nullptr && resume->impl_ == nullptr) { throw std::logic_error("Qwen4Exp: a resume carries no state"); }
        const QuoteImpl& q = *source.hybrid;
        const std::uint32_t index = destination.value;
        if (index >= options_.max_concurrency || lanes_[index].phase != Phase::Free || in_transaction()) {
            throw std::logic_error("Qwen4Exp: the binding destination is not free");
        }
        // A request paused after its first token binds its ledger; one paused inside its prompt binds
        // the prompt again, from the deepest cached state (its pause published the frontier).
        if (resume != nullptr && resume->impl_->generated) { return bind_replay(*base.impl_, index, *resume->impl_); }
        Lane& lane = lanes_[index];
        kv_busy();
        const BasePlanImpl& b = *base.impl_;
        const auto& data      = qwen3_5::PreparedPromptAccess::view(*b.prompt);
        std::optional<PrefixSelection> selection;
        if (prefix_.cache) {
            if (q.prefix && q.prefix->snapshot) {
                selection = prefix_select(data, b, index);
                if (!selection || !selection->snapshot || selection->frontier != source.reused_tokens) {
                    return {}; // the quoted snapshot changed: no shortage, the Engine moves to the root source
                }
            } else {
                selection = prefix_root(data, b);
            }
            prefix_pin(*selection);
        }
        lane.history.assign(data.token_ids.begin(), data.token_ids.end());
        lane.history.reserve(lane.history.size() + q.summary.effective_output_tokens + 1);
        lane.prompt_tokens  = static_cast<std::uint32_t>(lane.history.size());
        // A prompt with media keeps its M-RoPE positions; every other token is its index + delta.
        lane.rope.prompt_tokens = lane.prompt_tokens;
        lane.rope.delta         = data.rope_delta;
        if (data.has_media()) {
            lane.rope.prompt.assign(data.positions.begin(), data.positions.end());
        } else {
            lane.rope.prompt.clear();
        }
        // Reset before activation: a resume's activation sets the lane's frontier and state.
        lane.state_tokens   = 0;
        lane.vision_seconds = 0;
        lane.speculative    = {};
        lane.mtp_cells      = 0;
        lane.mtp_live       = mtp_;
        acquire_rows(lane, index);
        const auto shortage_of = [&](std::uint32_t need) {
            runtime::ResourceReservation out;
            const std::uint32_t available = head().pool->available_pages();
            out.shortage.main_kv_pages    = need > available ? need - available : 0;
            return out;
        };
        if (selection) {
            lane.prefix.reuse  = q.summary.publish_continuation;
            lane.prefix.hashes = data.block_hashes;
            lane.prefix.extras = data.block_extras;
            lane.prefix.trailing_extra = prefix_trailing_extra(data);
            lane.prefix.hints  = data.tap_hints.hints;
            lane.prefix.exclusions = prefix_exclusions(data);
            // Make room before reserving: an elastic pool grows only for what eviction cannot free, so
            // idle cache blocks never hold memory the expert frames could use (design §19.3.11).
            pc_make_room(selection->need);
            if (head().pool->available_pages() < selection->need) { (void)grow_kv(selection->need); }
            auto reservation = head().pool->reserve(selection->need);
            if (!reservation) {
                const auto out = shortage_of(selection->need);
                prefix_unpin(*selection);
                release_rows(lane);
                return out;
            }
            if (q.vision) { vision_reserve(lane, q.vision, data); }
            const std::uint64_t restored = prefix_.cache->counters().host_restore_bytes;
            prefix_activate(lane, index, *selection, *reservation, q.pages);
            lane.prefix.cached_tokens  = selection->cached_tokens;
            lane.prefix.restored_bytes = prefix_.cache->counters().host_restore_bytes - restored;
            lane.reservation    = std::move(*reservation);
            lane.prefix.reused  = selection->frontier;
            lane.calls          = selection->plan.ends;
        } else {
            if (head().pool->available_pages() < q.pages) { (void)grow_kv(q.pages); }
            auto reservation = head().pool->reserve(q.pages);
            if (!reservation) {
                const auto out = shortage_of(q.pages);
                release_rows(lane);
                return out;
            }
            if (q.vision) { vision_reserve(lane, q.vision, data); }
            lane.pages.clear();
            lane.pages.reserve(q.pages);
            head().pool->materialize(*reservation, q.pages, lane.pages);
            lane.reservation = std::move(*reservation);
            publish_rows(lane, 0, lane.pages);
            reset_slot(index);
            lane.calls = plan_prefill(data, 0, {}, false).ends;
        }
        lane.next_call = 0;
        lane.mtp_accept.fill(kAcceptancePrior);
        lane.mtp_policy_rounds = 0;
        if (max_width_ > 1) {
            if (options_.ngram_draft_tokens > 0) {
                lane.proposer =
                    std::make_unique<qwen3_5::detail::NgramProposer>(proposer_tokens_, proposer_tokens_ / 2);
                for (const auto token : lane.history) { lane.proposer->append(token); }
            }
            lane.speculative.enabled      = true;
            lane.speculative.backend      = mtp_ ? SpeculativeBackend::Mtp : SpeculativeBackend::None;
            lane.speculative.draft_window = static_cast<std::uint32_t>(max_width_ - 1);
            lane.speculative.accepted_per_position.assign(static_cast<std::size_t>(max_width_ - 1), 0);
        }
        lane.sampling       = q.sampling;
        lane.readout        = b.readout_tokens;
        lane.prompt         = b.prompt;
        lane.constraint     = b.constraint;
        lane.constraint_round.clear();
        lane.constraint_trace.clear();
        lane.sampling.token_counts =
            static_cast<std::int32_t*>(head().token_counts.p) + static_cast<std::size_t>(index) * token_domain_;
        lane.epoch           = ++next_epoch_;
        lane.phase           = Phase::Prefill;
        lane.prefill_ns      = 0;
        lane.decode_ns       = 0;
        lane.decode_share_ns = 0;
        lane.cache_at_admission = cache_stats();
        lane.cpu_served_at_admission = cpu_service_ ? cpu_service_->served_experts() : 0;
        if (tier_) { lane.tier_at_admission = tier_->stats(); }
        lane.ngram                   = {};
        lane.gate                    = {};
        lane.begin = runtime::BeginSummary{.prompt_tokens        = lane.prompt_tokens,
                                           .reused_prompt_tokens = source.reused_tokens,
                                           .prefix_reuse_path =
                                               source.reused_tokens ? source.reuse_path : PrefixReusePath::Root};
        {
            const SegmentWork on(*this, ple());
            CUDA_CHECK(cudaMemsetAsync(static_cast<std::byte*>(ple().gate_stats.p) + kGateStatsBytes * index, 0,
                                       kGateStatsBytes, ple().device->stream));
        }
        lane.prefill_end = lane.prompt_tokens;
        lane.replay      = false;
        lane.extent      = b.positions;
        // A request paused inside its prompt keeps what it had measured and decided.
        if (resume != nullptr) { restore_request(lane, *resume->impl_); }
        transaction_lane_ = index;
        replaying_        = false;
        ++revision_;
        runtime::ResourceReservation out;
        out.reserved = true;
        return out;
    }

    // ---------------------------------------------------------------- pause and replay (design §19.3.13)

    // The lane's request-level state a pause keeps (everything but the model state and the pages).
    void save_request(const Lane& lane, ResumeStateImpl& r) const {
        r.begin                   = lane.begin;
        r.prefill_ns              = lane.prefill_ns;
        r.decode_ns               = lane.decode_ns;
        r.decode_share_ns         = lane.decode_share_ns;
        r.vision_seconds          = lane.vision_seconds;
        r.speculative             = lane.speculative;
        r.mtp_accept              = lane.mtp_accept;
        r.mtp_policy_rounds       = lane.mtp_policy_rounds;
        r.constraint_trace        = lane.constraint_trace;
        r.cache_at_admission      = lane.cache_at_admission;
        r.cpu_served_at_admission = lane.cpu_served_at_admission;
        r.ngram                   = lane.ngram;
        r.gate                    = lane.gate;
    }

    void restore_request(Lane& lane, const ResumeStateImpl& r) {
        lane.begin                   = r.begin;
        lane.prefill_ns              = r.prefill_ns;
        lane.decode_ns               = r.decode_ns;
        lane.decode_share_ns         = r.decode_share_ns;
        lane.vision_seconds          = r.vision_seconds;
        lane.speculative             = r.speculative;
        lane.mtp_accept              = r.mtp_accept;
        lane.mtp_policy_rounds       = r.mtp_policy_rounds;
        lane.constraint_trace        = r.constraint_trace;
        lane.cache_at_admission      = r.cache_at_admission;
        lane.cpu_served_at_admission = r.cpu_served_at_admission;
        lane.ngram                   = r.ngram;
        lane.gate                    = r.gate;
    }

    // The resumed request's ledger as a prompt: its tokens through the next input, their RoPE
    // positions, and its lookup keys (the prompt's, chained over the output blocks as
    // prefix_publish chains them); no tap hints (a replay publishes blocks, not taps).
    qwen3_5::PreparedPromptData ledger_prompt(const qwen3_5::PreparedPromptData& prompt,
                                              const ResumeStateImpl& r) const;

    // Binds a request paused after its first token: the deepest cached state of its ledger up to
    // the paused frontier (the pause's own capture when it is still cached), then a replay of the
    // ledger to that frontier, without sampling. The pages cover the replay and one round.
    runtime::ResourceReservation bind_replay(const BasePlanImpl& b, std::uint32_t index, ResumeStateImpl& r) {
        Lane& lane = lanes_[index];
        kv_busy();
        const auto& prompt = qwen3_5::PreparedPromptAccess::view(*b.prompt);
        const qwen3_5::PreparedPromptData ledger = ledger_prompt(prompt, r);
        const std::uint32_t pages = binding_pages(b, r.target + 1U);
        std::optional<PrefixSelection> selection;
        if (prefix_.cache) {
            // The pause's capture lands first, so the selection can see it.
            prefix_.cache->drain();
            selection = b.reuse ? prefix_select(ledger, b, index) : prefix_root(ledger, b);
            if (!selection) {
                runtime::ResourceReservation out;
                const std::uint32_t available = head().pool->available_pages();
                out.shortage.main_kv_pages    = pages > available ? pages - available : 1U;
                return out;
            }
            // A state cached at the paused frontier itself leaves nothing to replay.
            selection->plan = selection->frontier < r.target
                                  ? prefix::plan_calls(selection->frontier, r.target, static_cast<std::uint32_t>(chunk_), {},
                                                       prefix::CallCost{.model = options_.prefix_cost,
                                                                        .span_seconds = options_.prefix_span_seconds})
                                  : prefix::CallPlan{};
            prefix_pin(*selection);
        }
        lane.history       = r.history;
        lane.prompt_tokens = r.prompt_tokens;
        lane.rope.prompt_tokens = r.prompt_tokens;
        lane.rope.delta         = prompt.rope_delta;
        if (prompt.has_media()) {
            lane.rope.prompt.assign(prompt.positions.begin(), prompt.positions.end());
        } else {
            lane.rope.prompt.clear();
        }
        lane.state_tokens   = 0;
        lane.vision_seconds = 0;
        lane.speculative    = {};
        lane.mtp_cells      = 0;
        lane.mtp_live       = mtp_;
        acquire_rows(lane, index);
        if (selection) {
            lane.prefix.reuse          = b.reuse;
            lane.prefix.hashes         = ledger.block_hashes;
            lane.prefix.extras         = prompt.block_extras;
            lane.prefix.trailing_extra = prefix_trailing_extra(prompt);
            lane.prefix.hints.clear();
            lane.prefix.exclusions = prefix_exclusions(prompt);
            pc_make_room(selection->need);
            if (head().pool->available_pages() < selection->need) { (void)grow_kv(selection->need); }
            auto reservation = head().pool->reserve(selection->need);
            if (!reservation) {
                runtime::ResourceReservation out;
                out.shortage.main_kv_pages = selection->need - std::min(selection->need, head().pool->available_pages());
                prefix_unpin(*selection);
                release_rows(lane);
                return out;
            }
            if (b.vision) { vision_reserve(lane, plan_vision(ledger, selection->frontier), ledger); }
            const std::uint64_t restored = prefix_.cache->counters().host_restore_bytes;
            prefix_activate(lane, index, *selection, *reservation, pages);
            lane.prefix.cached_tokens  = selection->cached_tokens;
            lane.prefix.restored_bytes = prefix_.cache->counters().host_restore_bytes - restored;
            lane.prefix.taps.clear();
            lane.reservation   = std::move(*reservation);
            lane.prefix.reused = selection->frontier;
            lane.calls         = selection->plan.ends;
        } else {
            if (head().pool->available_pages() < pages) { (void)grow_kv(pages); }
            auto reservation = head().pool->reserve(pages);
            if (!reservation) {
                runtime::ResourceReservation out;
                out.shortage.main_kv_pages = pages - std::min(pages, head().pool->available_pages());
                release_rows(lane);
                return out;
            }
            if (b.vision) { vision_reserve(lane, plan_vision(ledger, 0), ledger); }
            lane.pages.clear();
            lane.pages.reserve(pages);
            head().pool->materialize(*reservation, pages, lane.pages);
            lane.reservation = std::move(*reservation);
            publish_rows(lane, 0, lane.pages);
            reset_slot(index);
            lane.calls = prefix::plan_calls(0, r.target, static_cast<std::uint32_t>(chunk_), {},
                                            prefix::CallCost{.model = options_.prefix_cost,
                                                             .span_seconds = options_.prefix_span_seconds})
                             .ends;
        }
        lane.next_call = 0;
        if (max_width_ > 1 && options_.ngram_draft_tokens > 0) {
            lane.proposer = std::make_unique<qwen3_5::detail::NgramProposer>(proposer_tokens_, proposer_tokens_ / 2);
            for (const auto token : lane.history) { lane.proposer->append(token); }
        }
        lane.sampling = b.sampling;
        lane.sampling.token_counts =
            static_cast<std::int32_t*>(head().token_counts.p) + static_cast<std::size_t>(index) * token_domain_;
        rebuild_token_counts(lane);
        lane.readout.clear(); // the prompt's readout was delivered before the pause
        lane.prompt     = b.prompt;
        lane.constraint = b.constraint;
        lane.constraint_round.clear();
        lane.epoch = ++next_epoch_;
        restore_request(lane, r);
        {
            const SegmentWork on(*this, ple());
            CUDA_CHECK(cudaMemsetAsync(static_cast<std::byte*>(ple().gate_stats.p) + kGateStatsBytes * index, 0,
                                       kGateStatsBytes, ple().device->stream));
        }
        lane.extent      = b.positions;
        lane.prefill_end = r.target;
        // The cached state may reach the paused frontier itself: then nothing replays.
        lane.replay = lane.state_tokens < r.target;
        lane.phase  = lane.replay ? Phase::Prefill : Phase::Decode;
        if (!lane.replay && lane.vision) { vision_release(lane); }
        transaction_lane_ = index;
        replaying_        = lane.replay;
        ++revision_;
        runtime::ResourceReservation out;
        out.reserved = true;
        return out;
    }

    // Penalty counts over the request's output so far (the sampler counts output tokens only),
    // counted on the host: a binding may overlap a round that still uses the workspace.
    void rebuild_token_counts(const Lane& lane) {
        const SegmentWork on(*this, head());
        const cudaStream_t s = head().device->stream;
        auto* counts         = lane.sampling.token_counts;
        if (lane.sampling.presence_penalty == 0.0F && lane.sampling.frequency_penalty == 0.0F) {
            CUDA_CHECK(cudaMemsetAsync(counts, 0, 4ULL * token_domain_, s));
            return;
        }
        std::vector<std::int32_t> host(static_cast<std::size_t>(token_domain_), 0);
        for (std::size_t i = lane.prompt_tokens; i < lane.history.size(); ++i) {
            const std::int32_t token = lane.history[i];
            if (token >= 0 && token < token_domain_) { ++host[static_cast<std::size_t>(token)]; }
        }
        CUDA_CHECK(cudaMemcpyAsync(counts, host.data(), 4ULL * token_domain_, cudaMemcpyHostToDevice, s));
        CUDA_CHECK(cudaStreamSynchronize(s)); // the host table is read before it goes out of scope
    }

    // Pauses a resident at its committed boundary (design §19.3.13): the prefix cache receives the
    // lane's blocks and its state at the frontier, exactly as a consistent abort leaves them, the
    // lane is released and a Pause transaction hands the Engine the ResumeState.
    bool start_pause(SequenceHandle sequence) {
        if (in_transaction() || pending_transaction_ != 0) {
            diagnostic(std::string("Qwen4Exp: pause refused: ") +
                           (in_transaction() ? "a context transaction is open" : "a round awaits its commit"),
                       DiagnosticLevel::Warning);
            return false;
        }
        const std::uint32_t index = lane_of(sequence);
        Lane& lane                = lanes_[index];
        if (lane.phase != Phase::Prefill && lane.phase != Phase::Decode) {
            diagnostic("Qwen4Exp: pause refused: lane " + std::to_string(index) + " is in phase " +
                           std::to_string(static_cast<int>(lane.phase)),
                       DiagnosticLevel::Warning);
            return false;
        }
        // Inside a layer walk's span the layers are at different positions: the pause publishes
        // nothing past the span's start (its taps did), and the walk is abandoned with the lane.
        const bool walking = walk_.span.active && walk_.span.lane == index;
        synchronize();
        auto saved           = std::make_unique<ResumeStateImpl>();
        saved->generated     = lane.replay || lane.phase == Phase::Decode;
        saved->target        = lane.replay ? lane.prefill_end : lane.state_tokens;
        saved->history       = lane.history;
        saved->prompt_tokens = lane.prompt_tokens;
        save_request(lane, *saved);
        // A replaying lane short of its frontier publishes what it has rebuilt so far.
        if (!walking) { prefix_finish(lane, index); }
        saved->capture = lane.prefix.capture;
        release(index);
        pause_.emplace(ResumeState(std::move(saved)));
        ++revision_;
        return true;
    }

    // One prefill call of a resumed lane's ledger toward its paused frontier, without sampling.
    ReplayProgress advance_replay(SequenceHandle sequence) {
        const std::uint32_t index = lane_of(sequence);
        if (!lanes_[index].replay || lanes_[index].phase != Phase::Prefill) {
            throw std::logic_error("Qwen4Exp: replay on a lane that is not replaying");
        }
        const PrefillProgress progress = advance_prefill(sequence, nullptr); // replay samples nothing
        ReplayProgress out;
        out.processed_tokens = progress.processed_prompt_tokens;
        out.complete         = progress.complete;
        out.timing           = progress.timing;
        return out;
    }

    // Grows each unit's lane to its next round (design §19.3.13): a decode or control unit needs the
    // positions it writes, a verification's columns and the drafter's cells, within the request's
    // extent. A shortage cached blocks and the elastic pool cannot cover is reported, and the Engine
    // reclaims or pauses the youngest resident.
    runtime::ResourceReservation reserve_units(std::span<const ExecutionUnit> units) {
        runtime::ResourceReservation out;
        for (const ExecutionUnit& unit : units) {
            const std::uint32_t index = lane_of(unit.sequence);
            Lane& lane                = lanes_[index];
            std::uint32_t end         = lane.state_tokens + round_positions();
            if (unit.kind == ExecutionUnitKind::Control) { end += unit.tokens; }
            if (unit.kind == ExecutionUnitKind::Prefill || unit.kind == ExecutionUnitKind::Replay) {
                end = lane.prefill_end + round_positions();
            }
            const std::uint32_t needed = (std::min(end, lane.extent) + kPagedKVPageSize - 1) / kPagedKVPageSize;
            const std::uint32_t have   = lane.prefix.page_base + static_cast<std::uint32_t>(lane.pages.size());
            if (needed <= have) { continue; }
            const std::uint32_t missing = needed - have;
            kv_busy();
            if (prefix_.cache) { pc_make_room(missing); }
            if (head().pool->available_pages() < missing) { (void)grow_kv(missing); }
            if (head().pool->available_pages() < missing && prefix_.cache && prefix_.cache->transfers_pending()) {
                // Blocks pinned only by their in-flight Host writes (a lane just paused or finished)
                // become evictable once those land: wait for them before reporting a shortage.
                prefix_.cache->drain();
                pc_make_room(missing);
            }
            auto reservation = head().pool->reserve(missing);
            if (!reservation) {
                out.shortage.main_kv_pages += std::max(1U, missing - std::min(missing, head().pool->available_pages()));
                return out;
            }
            std::vector<DeviceKVPageLease> grown;
            grown.reserve(missing);
            head().pool->materialize(*reservation, missing, grown);
            publish_rows(lane, have, grown);
            for (DeviceKVPageLease& page : grown) { lane.pages.push_back(std::move(page)); }
            ++revision_;
        }
        out.reserved = true;
        return out;
    }

    // Advances the binding: a media prompt's encode window one step per call (design §19.3.2; other
    // lanes run between steps), then the lane is published. The first step makes no progress while
    // its leases would split a pool that another prefilling lane's calls free (vision_lend).
    ContextProgress poll_context(runtime::CancellationFlagView cancellation) {
        if (pause_) {
            ContextProgress out{.kind = ContextOperationKind::Pause};
            out.advanced  = true;
            out.complete  = true;
            out.published = true;
            out.paused.emplace(std::move(*pause_));
            pause_.reset();
            return out;
        }
        if (!transaction_lane_) { throw std::logic_error("Qwen4Exp: there is no context operation"); }
        ContextProgress out{.kind = ContextOperationKind::Bind};
        Lane& admitted = lanes_[*transaction_lane_];
        if (cancellation.requested()) {
            release(*transaction_lane_);
            transaction_lane_.reset();
            out.complete = true;
            return out;
        }
        if (admitted.vision && !vision_encoded(admitted)) {
            try {
                out.advanced = vision_step(admitted, *transaction_lane_);
            } catch (...) {
                release(*transaction_lane_);
                transaction_lane_.reset();
                throw;
            }
            if (!vision_encoded(admitted)) { return out; }
        }
        out.advanced  = true;
        out.complete  = true;
        out.published = true;
        out.replaying = replaying_;
        out.sequence  = handle(*transaction_lane_);
        transaction_lane_.reset();
        return out;
    }

    [[nodiscard]] bool in_transaction() const noexcept { return transaction_lane_.has_value() || pause_.has_value(); }

    // Whether the open binding holds `sequence` back (its lane is still being published).
    // A lane's context is blocked while it is being bound, and a prefilling lane while another
    // lane's layer walk holds the expert stream and the lent workspace (the walk's layers are at
    // different positions until its span ends, so no other prefill call may run in between).
    [[nodiscard]] bool context_blocks(SequenceHandle sequence) const noexcept {
        const std::uint32_t index = ContractAccess::lane(sequence);
        if (transaction_lane_ && *transaction_lane_ == index) { return true; }
        return walk_.span.active && walk_.span.lane != index && index < lanes_.size() &&
               lanes_[index].phase == Phase::Prefill;
    }

    // The Engine's reclaim for a binding shortage: evict cached Device blocks until it is covered.
    bool hybrid_reclaim(runtime::ContextResourceUsage shortage) {
        if (!prefix_.cache || shortage.main_kv_pages == 0) { return false; }
        const std::uint32_t before = head().pool->available_pages();
        pc_make_room(before + shortage.main_kv_pages);
        if (head().pool->available_pages() == before && prefix_.cache->transfers_pending()) {
            // Blocks pinned only by in-flight Host writes become evictable once those land.
            prefix_.cache->drain();
            pc_make_room(before + shortage.main_kv_pages);
        }
        return head().pool->available_pages() > before;
    }

    // ---------------------------------------------------------------- execution
    PrefillProgress advance_prefill(SequenceHandle sequence, runtime::TokenMaskProvider* masks) {
        const GrammarScope grammar(*this, masks);
        const auto start  = Clock::now();
        const auto index  = lane_of(sequence);
        Lane& lane        = lanes_[index];
        if (lane.phase != Phase::Prefill) { throw std::logic_error("Qwen4Exp: prefill on a lane that is not prefilling"); }
        ++prefill_steps_;
        if (walk_.span.active) {
            if (walk_.span.lane != index) {
                throw std::logic_error("Qwen4Exp: another lane's prefill holds the layer walk");
            }
            return walk_step(lane, index, start);
        }
        const std::int32_t begin = static_cast<std::int32_t>(lane.state_tokens);
        if (lane.next_call >= lane.calls.size() || lane.calls[lane.next_call] <= lane.state_tokens) {
            throw std::logic_error(
                "Qwen4Exp: the lane's prefill plan does not continue its state (lane " + std::to_string(index) +
                ", call " + std::to_string(lane.next_call) + " of " + std::to_string(lane.calls.size()) +
                (lane.next_call < lane.calls.size() ? ", ending at " + std::to_string(lane.calls[lane.next_call]) : "") +
                ", state " + std::to_string(lane.state_tokens) + ", prompt end " + std::to_string(lane.prefill_end) + ")");
        }
        if (const std::size_t end = walk_span_end(lane, index); end > lane.next_call) {
            walk_begin(lane, index, end);
            return walk_step(lane, index, start);
        }
        const std::int32_t width = static_cast<std::int32_t>(lane.calls[lane.next_call++] - lane.state_tokens);
        const bool last          = begin + width == static_cast<std::int32_t>(lane.prefill_end);
        // A call the CPU serves (an opener tail, a tiny suffix) promotes at the decode rate (design
        // §19.3.1). Wider prefill calls promote nothing: the answer's staged misses land in free
        // frames and pick its experts, while the prompt's would displace them (16 per layer per
        // call measured slower than none in every regime: serve C = 4 fill -4.3 %, warm -5.6 %;
        // design §19.3.17). Their routes still count toward the cache's scores.
        const std::vector<std::size_t> promotions = prefill_budgets(width);
        // Host phases of a chunk for nsys attribution (the GPU side is in the CUDA trace).
        const nvtx::ScopedRange chunk_range(nvtx::Name::PrefillChunk, nvtx::Category::Prefill,
                                            static_cast<std::uint64_t>(width));
        {
            const nvtx::ScopedRange rows_range(nvtx::Name::PrefillPleRows, nvtx::Category::Prefill,
                                               static_cast<std::uint64_t>(width));
            stage_sequence(index, begin, width, lane.history);
        }
        const auto chunk = stage_mtp_chunk(lane, index, begin, width);
        // A resumed lane's first call waits per layer for its restore; any call waits per layer for
        // a copy-out of this lane's state still in flight (design §19.3.1).
        prefix_.next_waits = prefix_waits(lane);
        run(1, width, 1, chunk ? &*chunk : nullptr, stage_vision(lane, static_cast<std::uint32_t>(begin), width),
            /*stream=*/!walk_.span.active);
        // The handoff is read only by the prompt's calls: it returns once the last is enqueued, ahead of
        // the stream leases (vision_release).
        if (last && lane.vision) { vision_release(lane); }
        if (last && !prefilling_besides(index)) { return_leases(); }
        lane.state_tokens += static_cast<std::uint32_t>(width);
        prefix_after_prefill_call(lane, index, last);
        if (!last) {
            {
                const nvtx::ScopedRange wait_range(nvtx::Name::DeviceWait, nvtx::Category::Prefill);
                synchronize();
            }
            check_expert_error();
            hash_logits("prefill", index, begin, width, 1);
            const nvtx::ScopedRange residency_range(nvtx::Name::PrefillResidency, nvtx::Category::Moe,
                                                    static_cast<std::uint64_t>(width));
            trace_round(RouteTraceKind::PrefillChunk, 1, width, static_cast<std::uint32_t>(width), promotions.front(),
                        static_cast<std::uint32_t>(begin));
            after_call(width, promotions);
            apply_vram_target(false);
        }

        return prefill_progress(lane, index, begin, width, static_cast<std::uint32_t>(width), last, promotions, start);
    }

    // Each segment's promotions for a prefill call of `width` columns: a call the CPU serves (an
    // opener tail, a tiny suffix) promotes at the decode rate (design §19.3.1), wider calls promote
    // nothing.
    std::vector<std::size_t> prefill_budgets(std::int32_t width) {
        std::vector<std::size_t> out(segments_.size(), 0);
        if (width > kServedCallColumns) { return out; }
        for (std::size_t s = 0; s < segments_.size(); ++s) {
            out[s] = decode_budget(*segments_[s], static_cast<std::uint32_t>(width));
        }
        return out;
    }

    // Once an eager call has completed (every segment synchronized): each segment's expert cache
    // applies the call's routes, with its promotions.
    void after_call(std::int32_t columns, std::span<const std::size_t> budgets) {
        for (std::size_t s = 0; s < segments_.size(); ++s) {
            DeviceSegment& seg = *segments_[s];
            const SegmentWork on(*this, seg);
            seg.residency->after_round(seg.device->stream, columns, budgets[s]);
        }
    }

    // After a prompt's last call is enqueued, with no other lane prefilling: each segment's stream
    // lease and wide arena go back.
    void return_leases() {
        for (const auto& segment : segments_) {
            const DeviceGuard on(segment->device->device);
            return_stream_lease(*segment);
            return_wide_work(*segment);
        }
    }

    // The progress of a prefill unit whose calls were enqueued (the last ends at `begin + width`,
    // `tokens` prompt tokens in all): for the prompt's last call the first token is sampled and the
    // pending batch returned. Non-last units have already synchronized and applied their routes.
    PrefillProgress prefill_progress(Lane& lane, std::uint32_t index, std::int32_t begin, std::int32_t width,
                                     std::uint32_t tokens, bool last, std::span<const std::size_t> promotions,
                                     Clock::time_point start) {
        PrefillProgress out;
        out.summary                 = prefill_summary(lane);
        out.processed_prompt_tokens = tokens;
        out.complete                = last;
        if (last && lane.replay) {
            // A replay ends at its paused frontier, whose next input the ledger already holds: the
            // lane decodes from there (design §19.3.13).
            {
                const nvtx::ScopedRange wait_range(nvtx::Name::DeviceWait, nvtx::Category::Prefill);
                synchronize();
            }
            hash_logits("prefill", index, begin, width, 1);
            trace_round(RouteTraceKind::PrefillChunk, 1, width, static_cast<std::uint32_t>(width), promotions.front(),
                        static_cast<std::uint32_t>(begin));
            after_call(width, promotions);
            apply_vram_target(false);
            lane.replay = false;
            lane.phase  = Phase::Decode;
            out.timing.submit_host_ns = elapsed_ns(start);
            lane.prefill_ns += out.timing.submit_host_ns;
            return out;
        }
        if (last) {
            const std::uint32_t lanes[] = {index};
            const std::int32_t positions[] = {begin + width - 1};
            sample(lanes, positions);
            if (!lane.readout.empty()) { out.readout = read_prompt_frontier(lane); }
            hash_logits("prefill", index, begin, width, 1);
            {
                const nvtx::ScopedRange residency_range(nvtx::Name::PrefillResidency, nvtx::Category::Moe,
                                                        static_cast<std::uint64_t>(width));
                trace_round(RouteTraceKind::PrefillChunk, 1, width, static_cast<std::uint32_t>(width),
                            promotions.front(), static_cast<std::uint32_t>(begin));
                after_call(width, promotions);
                apply_vram_target(false);
            }
            const SequenceHandle rows[] = {handle(index)};
            out.timing.submit_host_ns = elapsed_ns(start);
            lane.prefill_ns += out.timing.submit_host_ns;
            out.pending.emplace(ContractAccess::make_pending(this, ++next_transaction_, rows,
                                                             {pending_tokens_.data(), 1}, out.timing));
            mark_grammar_failures(*out.pending, {});
            plain_round_ = false;
            pending_transaction_ = next_transaction_;
        } else {
            out.timing.submit_host_ns = elapsed_ns(start);
            lane.prefill_ns += out.timing.submit_host_ns;
        }
        return out;
    }

    // The Begin the Engine admitted the request with, fixed at binding (a snapshot evicted since
    // must not change it).
    const runtime::BeginSummary& prefill_summary(const Lane& lane) const noexcept { return lane.begin; }

    PendingBatch decode(std::span<const SequenceHandle> sequences, std::span<const runtime::RoundBudget> budgets,
                        runtime::TokenMaskProvider* masks) {
        static_assert(kSplitDecode, "a layer split's rounds run across its segments");
        const GrammarScope grammar(*this, masks);
        const auto start = Clock::now();
        mark(Mark::DecodeBegin);
        const auto batch = static_cast<std::int32_t>(sequences.size());
        if (batch <= 0 || batch > static_cast<std::int32_t>(options_.max_concurrency)) {
            throw std::logic_error("Qwen4Exp: decode batch is empty or too large");
        }
        std::array<std::uint32_t, kMaximumConcurrency> lanes{};
        std::array<std::int32_t, kMaximumConcurrency> positions{};
        for (std::int32_t b = 0; b < batch; ++b) {
            lanes[b]   = lane_of(sequences[b]);
            Lane& lane = lanes_[lanes[b]];
            if (lane.phase != Phase::Decode || lane.history.size() != lane.state_tokens + 1ULL) {
                throw std::logic_error("Qwen4Exp: decode row is not at a committed frontier");
            }
            positions[b] = static_cast<std::int32_t>(lane.state_tokens);
        }
        // Proposals: a round verifies 1 + the longest row's drafts; a row may draft at most one token
        // fewer than it may still emit, so every verified position lies in its reservation. A longer
        // n-gram copy proposal replaces a row's MTP drafts. By round size (design 11.3, 19.3.5):
        // - one row: its own draft length, maximizing its expected tokens per round time;
        // - two rows: one draft length for both (choose_pair_draft_length), so no row is padded to the
        //   other's; its at most 2 x 3 = 6 columns stay on the column-invariant dense routes, so greedy
        //   output equals C = 1, and below the 7-column fork and CPU-job limits;
        // - three or more rows decode plain (one draft each measured slower at four rows).
        // A plain round's W = 1 catch-up keeps each drafter current.
        const std::int32_t pair_k =
            batch == 2 && mtp_ ? choose_pair_draft_length(lanes_[lanes[0]], lanes_[lanes[1]]) : 0;
        const bool speculate = batch == 1 || pair_k > 0;
        std::array<std::int32_t, kMaximumConcurrency> wanted{};
        std::int32_t steps = 0;
        if (mtp_) {
            for (std::int32_t b = 0; b < batch; ++b) {
                Lane& lane = lanes_[lanes[b]];
                wanted[b]  = speculate && lane.mtp_live ? (batch == 1 ? choose_draft_length(lane) : pair_k) : 0;
                steps      = std::max(steps, wanted[b]);
            }
        }
        // Each row's draft count and source are fixed before the drafter runs (an MTP row's count
        // is its draft length; a longer n-gram copy proposal replaces its drafts), so the round's
        // width, and with it every row's n-gram landing columns, is known while it drafts.
        std::array<std::int32_t, kMaximumConcurrency> extents{};
        std::int32_t width = 1;
        for (std::int32_t b = 0; b < batch && max_width_ > 1; ++b) {
            Lane& lane = lanes_[lanes[b]];
            drafts_[b].clear();
            from_ngram_[b] = false;
            const std::uint32_t remaining =
                static_cast<std::size_t>(b) < budgets.size() ? budgets[b].generated_tokens_remaining : 0U;
            if (remaining < 2) { continue; }
            auto limit = std::min<std::uint32_t>(static_cast<std::uint32_t>(max_width_ - 1), remaining - 1U);
            if (batch > 1) { limit = std::min<std::uint32_t>(limit, static_cast<std::uint32_t>(pair_k)); }
            if (mtp_ && lane.mtp_live && wanted[b] > 0) {
                extents[b] = static_cast<std::int32_t>(std::min<std::uint32_t>(limit, static_cast<std::uint32_t>(wanted[b])));
            }
            if (speculate && lane.proposer) {
                auto copy = lane.proposer->propose(lane.history, limit, options_.ngram_min_match).tokens;
                if (copy.size() > static_cast<std::size_t>(extents[b])) {
                    drafts_[b]     = std::move(copy);
                    from_ngram_[b] = true;
                    extents[b]     = static_cast<std::int32_t>(drafts_[b].size());
                }
            }
            width = std::max(width, 1 + extents[b]);
        }
        // A round whose every drafting row verifies an n-gram copy proposal instead runs no MTP
        // draft steps, whose drafts it would discard: the drafter then only writes an unwritten
        // pending cell (as a round without drafts does), and the round stages its rows' n-gram
        // rows itself. The steps' K/V writes are speculative cells inside the verified window,
        // which the commit's catch-up rewrites, so the drafter's state and every token are
        // unchanged; such a round saves the steps' time (about 2 ms of a 25 ms split round).
        if (steps > 0) {
            bool copies_only = true;
            for (std::int32_t b = 0; b < batch; ++b) {
                if (wanted[b] > 0 && !from_ngram_[b]) { copies_only = false; }
            }
            if (copies_only) { steps = 0; }
        }
        // Columns of each verification row whose n-gram rows are already staged: the drafter's
        // wait reads every committed-token column (the anchor's) and all of a copy row's.
        std::array<std::int32_t, kMaximumConcurrency> staged{};
        if (mtp_) {
            const HostSpan drafting(timing_.on ? &timing_.draft_ms : nullptr, &timing_.round_draft_ms);
            timing_.round_steps = steps;
            mtp_draft(std::span<const std::uint32_t>(lanes.data(), batch), steps, [&] {
                if (width == 1) { return; }
                for (std::int32_t b = 0; b < batch; ++b) {
                    staged[b] = from_ngram_[b] ? width : 1;
                    stage_verify_ngram(lanes_[lanes[b]], static_cast<std::size_t>(b), positions[b], width, 0, staged[b]);
                }
            });
            for (std::int32_t b = 0; b < batch && max_width_ > 1; ++b) {
                if (!from_ngram_[b] && extents[b] > 0) {
                    drafts_[b].assign(mtp_drafts_[b].begin(), mtp_drafts_[b].begin() + extents[b]);
                }
            }
            // A one-row round verifies only its confident drafts: those before the first whose
            // draft-head probability is below 0.5 (kDraftMinLogprob). Its width may still shrink, as
            // the drafter staged only column 0's n-gram rows; a pair round's columns are staged at
            // their offsets, so it keeps its length.
            if (batch == 1 && !from_ngram_[0] && extents[0] > 0) {
                extents[0] = std::min(extents[0], mtp_confident_[0]);
                drafts_[0].resize(static_cast<std::size_t>(extents[0]));
                width = 1 + extents[0];
            }
        }
        if (width > 1) {
            return verify(sequences, std::span<const std::uint32_t>(lanes.data(), batch),
                          std::span<const std::int32_t>(positions.data(), batch), width,
                          std::span<const std::int32_t>(staged.data(), batch), start);
        }
        for (std::int32_t b = 0; b < batch; ++b) { ++lanes_[lanes[b]].speculative.fallback_steps; }
        round_width_ = 1;
        plain_round_ = true;
        const std::span<const std::uint32_t> round_lanes(lanes.data(), static_cast<std::size_t>(batch));
        const std::span<const std::int32_t> round_positions(positions.data(), static_cast<std::size_t>(batch));
        // A replayed round reads its rows after the launch, while the GPU embeds and runs layer 0,
        // and releases them to the gate before ple_embed (design §12.3, n-gram S4b); a round that
        // warms or captures its graph reads them first.
        // A round whose rows are all in the host cache reads nothing, so it stages them first.
        const bool gated = round_replayed(batch, 1, false) && !decode_rows_cached(round_lanes, round_positions);
        before_restage();
        stage_decode(round_lanes, round_positions);
        const std::uint32_t word = next_gate_word();
        if (!gated) {
            stage_decode_rows(round_lanes, round_positions);
            publish_pinned_word(gate_ready(), word);
        }
        GateRelease release_gate{nullptr, word, io_prefix(batch) - io_layout_.ngram};
        HandoffRelease release_handoff;
        const auto read_rows = [&] {
            read_behind_gate(lanes_[lanes[0]], word, [&] { stage_decode_rows(round_lanes, round_positions); });
        };
        const bool rows_read = run_decode(batch, gated, release_gate, release_handoff, read_rows);
        for (std::int32_t b = 0; b < batch; ++b) { ++lanes_[lanes[b]].state_tokens; }
        sample(round_lanes, round_positions, [&] {
            if (!gated || rows_read) { return; }
            read_rows();
            release_gate.owner = nullptr;
        });
        timing_synchronized(1);
        release_handoff.owner = nullptr; // every segment has synchronized
        hash_logits("decode", lanes[0], positions[0], 1, batch);
        deferred_ = {.columns = batch, .tokens = 1, .live = false, .pending = true};
        runtime::ExecutionTiming timing;
        timing.submit_host_ns = elapsed_ns(start);
        for (std::int32_t b = 0; b < batch; ++b) {
            lanes_[lanes[b]].decode_ns += timing.submit_host_ns;
            lanes_[lanes[b]].decode_share_ns += timing.submit_host_ns / static_cast<std::uint64_t>(batch);
        }
        pending_transaction_  = ++next_transaction_;
        auto pending = ContractAccess::make_pending(this, pending_transaction_, sequences,
                                                    {pending_tokens_.data(), static_cast<std::size_t>(batch)}, timing);
        mark_grammar_failures(pending, {});
        return pending;
    }

    runtime::ExecutionTiming append_forced(std::span<const SequenceHandle> sequences, std::span<const TokenId> tokens,
                                           std::uint32_t stride) {
        const auto start = Clock::now();
        if (stride == 0 || tokens.size() != static_cast<std::size_t>(stride) * sequences.size()) {
            throw std::invalid_argument("Qwen4Exp: forced tokens do not match their rows");
        }
        for (std::size_t row = 0; row < sequences.size(); ++row) {
            const auto index = lane_of(sequences[row]);
            Lane& lane       = lanes_[index];
            if (lane.phase != Phase::Decode || lane.history.size() != lane.state_tokens + 1ULL) {
                throw std::logic_error("Qwen4Exp: forced tokens on a lane that is not at a committed frontier");
            }
            const auto forced = tokens.subspan(row * stride, stride);
            // The model reads the pending input token and every forced token but the last, which
            // becomes the next decode input.
            lane.history.insert(lane.history.end(), forced.begin(), forced.end());
            if (lane.proposer) {
                for (const auto token : forced) { lane.proposer->append(token); }
            }
            const auto begin = static_cast<std::int32_t>(lane.state_tokens);
            stage_sequence(index, begin, static_cast<std::int32_t>(stride), lane.history);
            const auto chunk = stage_mtp_chunk(lane, index, begin, static_cast<std::int32_t>(stride));
            run(1, static_cast<std::int32_t>(stride), 1, chunk ? &*chunk : nullptr);
            synchronize();
            check_expert_error();
            hash_logits("forced", index, begin, static_cast<std::int32_t>(stride), 1);
            std::vector<std::size_t> promotions(segments_.size());
            for (std::size_t s = 0; s < segments_.size(); ++s) { promotions[s] = forced_budget(*segments_[s]); }
            trace_round(RouteTraceKind::ForcedTokens, 1, static_cast<std::int32_t>(stride), stride, promotions.front(),
                        static_cast<std::uint32_t>(begin));
            after_call(static_cast<std::int32_t>(stride), promotions);
            apply_vram_target(false);
            lane.state_tokens += stride;
        }
        runtime::ExecutionTiming timing;
        timing.submit_host_ns = elapsed_ns(start);
        return timing;
    }

    CommitResult commit(PendingBatch&& pending, std::span<const runtime::CommitDecision> decisions) {
        if (ContractAccess::owner(pending) != this || ContractAccess::transaction(pending) != pending_transaction_ ||
            pending_transaction_ == 0) {
            throw std::logic_error("Qwen4Exp: commit of a foreign or stale pending batch");
        }
        const auto rows = ContractAccess::rows(pending);
        if (decisions.size() != rows.size()) { throw std::logic_error("Qwen4Exp: commit decisions are not row aligned"); }
        const HostSpan committing(timing_.on ? &timing_.commit_ms : nullptr);
        mark(Mark::CommitBegin);
        // A verification round's commit runs on every segment; a plain round's (the drafter's
        // catch-up, a cancelled row's saved column) on the head.
        if (round_width_ > 1) { return commit_verified(std::move(pending), decisions); }
        std::optional<SegmentWork> on_head;
        on_head.emplace(*this, head());
        before_restage();
        CommitResult out;
        out.row_count = rows.size();
        // After a plain decode round the drafter writes the round's cell (its pending cell moves
        // one position); after the prefill's first token the pending cell stays the prompt's last.
        const bool catch_up = mtp_ && plain_round_;
        std::int32_t* commit = catch_up ? spec_host(spec_layout_.commit) : nullptr;
        bool any_cell        = false;
        for (std::size_t row = 0; row < rows.size(); ++row) {
            const auto index = lane_of(rows[row]);
            Lane& lane       = lanes_[index];
            const auto& d    = decisions[row];
            if (commit != nullptr) { commit[row] = 0; }
            if (d.cancelled) {
                if (catch_up && lane.mtp_live) {
                    // MR2: decode() advanced the state past the round's position at submission and
                    // a cancelled row skips the catch-up, so that position's cell becomes the
                    // pending one: its residual, which the round exported, becomes the saved one.
                    CUDA_CHECK(cudaMemcpyAsync(saved_column(index).data,
                                               static_cast<const std::byte*>(head().mtp_residuals.p) + 2ULL * width_ * row,
                                               2ULL * width_, cudaMemcpyDeviceToDevice, head().device->stream));
                    lane.mtp_cells = lane.state_tokens - 1;
                }
                lane.constraint_round.clear();
                out.rows[row].timings     = timings(lane);
                out.rows[row].speculative = lane.speculative;
                out.rows[row].disposition = runtime::CommitDisposition::CancelledReleased;
                continue;
            }
            if (d.accepted_tokens != 1) { throw std::logic_error("Qwen4Exp: a row commits exactly one token"); }
            commit_constraint_draws(lane, 1);
            lane.history.push_back(pending_tokens_[row]);
            if (commit != nullptr && lane.mtp_live) {
                commit[row] = 1;
                any_cell    = true;
            }
            if (lane.proposer) { lane.proposer->append(pending_tokens_[row]); }
            if (lane.phase == Phase::Prefill) { lane.phase = Phase::Decode; }
            if (d.terminal) {
                lane.phase                = Phase::Finishable;
                out.rows[row].disposition = runtime::CommitDisposition::Finishable;
            } else {
                out.rows[row].disposition = runtime::CommitDisposition::Active;
            }
        }
        if (any_cell) {
            upload_pinned(spec_device(spec_layout_.commit), commit, 4ULL * rows.size(), head().device->stream);
            mtp_catch_up(rows, 1, commit);
        }
        on_head.reset(); // the head's work is enqueued; each segment settles on its own
        for (const auto& row : rows) { require_mtp_settled(lanes_[ContractAccess::lane(row)]); }
        for (std::size_t row = 0; row < rows.size(); ++row) {
            if (!decisions[row].cancelled) {
                Lane& lane = lanes_[ContractAccess::lane(rows[row])];
                prefix_publish(lane, prefix_frontier(lane, false));
            }
        }
        settle_round();
        for (std::size_t row = 0; row < rows.size(); ++row) {
            if (decisions[row].cancelled) { release(ContractAccess::lane(rows[row])); }
        }
        plain_round_ = false;
        ContractAccess::consume(pending);
        pending_transaction_ = 0;
        return out;
    }

    DiscardResult discard(PendingBatch&& pending) noexcept {
        DiscardResult out;
        if (ContractAccess::owner(pending) != this || ContractAccess::transaction(pending) != pending_transaction_) {
            return out;
        }
        const auto rows = ContractAccess::rows(pending);
        for (const auto& row : rows) {
            const auto index = ContractAccess::lane(row);
            if (index < options_.max_concurrency && lanes_[index].phase != Phase::Free) { release(index); }
        }
        out.status    = runtime::ConsumeStatus::Consumed;
        out.row_count = rows.size();
        ContractAccess::consume(pending);
        pending_transaction_ = 0;
        deferred_.pending    = false;
        return out;
    }

    FinishResult finish(SequenceHandle sequence) noexcept {
        FinishResult out;
        try {
            const auto index = lane_of(sequence);
            if (lanes_[index].phase != Phase::Finishable) {
                diagnostic("Qwen4Exp: finish of a lane that is not finishable", DiagnosticLevel::Error);
                return out;
            }
            if (!mtp_settled(lanes_[index])) {
                diagnostic("Qwen4Exp: a finished request's MTP cells are out of step with its state", DiagnosticLevel::Warning);
            }
            out.timings           = timings(lanes_[index]);
            out.speculative       = lanes_[index].speculative;
            out.constrained_draws = std::move(lanes_[index].constraint_trace);
            if (first().residency) {
                const ExpertResidency::Stats s = cache_stats();
                const auto& a                  = lanes_[index].cache_at_admission;
                out.expert_cache = ExpertCacheStats{.routed = s.routed - a.routed, .hits = s.hits - a.hits};
            }
            report_cache(lanes_[index]);
            report_ngram(lanes_[index]);
            report_handoff();
            report_split_timing();
            prefix_finish(lanes_[index], index);
            release(index);
            out.status = runtime::ConsumeStatus::Consumed;
        } catch (const std::exception& error) {
            // The Engine fails a terminal sequence that does not finish; say why.
            diagnostic(std::string("Qwen4Exp: finish failed: ") + error.what(), DiagnosticLevel::Error);
        } catch (...) {
            diagnostic("Qwen4Exp: finish failed", DiagnosticLevel::Error);
        }
        return out;
    }

    AbortResult abort(SequenceHandle sequence) noexcept {
        AbortResult out;
        try {
            const auto index = lane_of(sequence);
            out.timings      = timings(lanes_[index]);
            out.speculative  = lanes_[index].speculative;
            // A consistent abort (no unit in flight) leaves an endpoint a resend resumes from; inside
            // a layer walk the layers are at different positions, so it leaves none.
            if (pending_transaction_ == 0 && lanes_[index].phase != Phase::Free &&
                !(walk_.span.active && walk_.span.lane == index)) {
                prefix_finish(lanes_[index], index);
            }
            release(index);
            out.status = runtime::ConsumeStatus::Consumed;
        } catch (...) {}
        return out;
    }

    // After a RecoverableExecutionError: every lane is released without publishing anything (the
    // failed round's state is unknown), the cache's transfers in flight land, and a cache the
    // failure may have published into is emptied. False when that fails (the device itself failed;
    // the Engine then fails everything).
    bool recover_after_failure() noexcept {
        try {
            for (std::uint32_t i = 0; i < options_.max_concurrency; ++i) {
                if (lanes_[i].phase != Phase::Free) { release(i); }
            }
            transaction_lane_.reset();
            pending_transaction_ = 0;
            deferred_.pending    = false; // the failed round's cache update
            synchronize();
            if (prefix_.cache) {
                prefix_.cache->drain();
                if (clear_prefix_on_release_) { prefix_.cache->clear(); }
            }
            clear_prefix_on_release_ = false;
            return true;
        } catch (...) {
            return false;
        }
    }

    void release_all() noexcept {
        for (std::uint32_t i = 0; i < options_.max_concurrency; ++i) {
            if (lanes_[i].phase != Phase::Free) { release(i); }
        }
        transaction_lane_.reset();
        pending_transaction_ = 0;
        try { synchronize(); } catch (...) {}
        if (clear_prefix_on_release_) { // a failed round may have published state it never computed
            if (prefix_.cache) { prefix_.cache->clear(); }
            clear_prefix_on_release_ = false;
        }
    }

    PhysicalUsageSnapshot usage() const noexcept {
        PhysicalUsageSnapshot out;
        for (std::uint32_t i = 0; i < options_.max_concurrency; ++i) {
            if (lanes_[i].phase != Phase::Free) { ++out.occupied.state_slots; }
        }
        out.capacity.state_slots   = options_.max_concurrency;
        out.occupied.main_kv_pages = head().pool->allocated_pages();
        out.capacity.main_kv_pages = head().pool->capacity_pages();
        return out;
    }

    MemorySummary memory() const noexcept {
        MemorySummary out;
        out.max_context             = options_.max_context;
        out.kv_capacity             = kv_pages_ * static_cast<std::uint32_t>(kPagedKVPageSize);
        out.kv_capacity_page_groups = kv_pages_;
        out.kv_capacity_max_page_groups = kv_pages_;
        out.kv_cache                = options_.kv_cache;
        // Every segment's KV planes and workspace.
        for (const auto& segment : segments_) {
            out.kv_payload_bytes += segment->kv_vmm ? segment->kv_vmm->mapped_bytes() : segment->kv_backing.bytes;
            out.workspace_logical_peak_bytes += segment->work_capacity;
        }
        return out;
    }

    std::uint64_t revision() const noexcept { return revision_; }

private:
    // Promotions per layer call (design section 19.2): one per decode round measured fastest; a
    // promotion moves as many PCIe bytes as serving the expert once zero-copy.
    static constexpr std::size_t kDecodePromotionsPerLayer  = 1;
    // Once the frames are full, decode promotes only every kDecodePromotionInterval-th round on the
    // reference link (promotion_interval scales it to the measured one): a promotion moves as many
    // PCIe bytes as serving its expert once, and fewer of them measured faster at 512 tokens (design
    // section 19.2). Until then every round promotes, so a cold cache fills at the full rate.
    static constexpr std::uint64_t kDecodePromotionInterval = 4;
    // Misses staged per pass of a layer's experts: every decode and verify call's misses in one
    // pass; a prefill chunk's in several.
    static constexpr std::int32_t kStagingSlots = 64;
    // The SSD tier's smallest RAM (design §19.3.7 H_min: 1,024 slots, 2.6 GiB).
    static constexpr std::uint32_t kMinTierSlots = 1024;
    // Promotions follow the tokens a round advances (the longest row's), not rounds: a verification
    // round that advances four tokens promotes what four decode rounds would, so a speculative
    // round's cache warms per token as plain decode does.
    std::size_t decode_budget(DeviceSegment& seg, std::uint32_t tokens) {
        if (decode_promotions_ == DecodePromotions::Off || !promotes_in_rounds(seg)) { return 0; }
        if (trickles(seg)) { return trickle_budget(seg); }
        // The fill phase: until this session promoted or landed as many experts as there are frames.
        // Seeded frames (a warm start) do not end it: promotions only follow misses, so a workload
        // the saved state matches promotes little anyway, while one it does not match replaces the
        // seeded experts at the fill rate instead of the steady-state rate.
        const ExpertResidency::Stats& stats = seg.residency->stats();
        if (stats.promotions + stats.landed < seg.residency->frames()) {
            return kDecodePromotionsPerLayer * tokens;
        }
        if (decode_promotions_ == DecodePromotions::Fill) { return 0; }
        const std::uint64_t interval = promotion_interval(seg);
        seg.budget_tokens += tokens;
        const std::uint64_t due = seg.budget_tokens / interval;
        seg.budget_tokens %= interval;
        return static_cast<std::size_t>(due) * kDecodePromotionsPerLayer;
    }
    // A forced-token call's promotions: one per layer, as a decode round's while the frames fill,
    // within INFERNIX_DECODE_PROMOTIONS (none when off, none once the frames have filled with fill).
    std::size_t forced_budget(const DeviceSegment& seg) const {
        if (decode_promotions_ == DecodePromotions::Off || !promotes_in_rounds(seg)) { return 0; }
        if (trickles(seg)) { return trickle_budget(seg); }
        const ExpertResidency::Stats& stats = seg.residency->stats();
        if (decode_promotions_ == DecodePromotions::Fill &&
            stats.promotions + stats.landed >= seg.residency->frames()) {
            return 0;
        }
        return kDecodePromotionsPerLayer;
    }
    // The interval at the measured link (design §19.3.12): a promotion's bytes cost less wall time
    // on a faster link, whose rounds leave it more idle, so it promotes proportionally more often
    // (x8 4, x16 2; the x16 value is derived, not measured).
    static std::uint64_t promotion_interval(const DeviceSegment& seg) noexcept {
        static_assert(kDecodePromotionInterval == LinkPolicy{}.promotion_interval,
                      "derive_link_policy scales the reference link's promotion interval");
        return seg.policy.promotion_interval;
    }
    // Whether a segment's cache promotes during requests (its rounds and calls). A layer split's
    // segment follows its link's promotion policy (derive_link_policy): Dynamic as unsplit, or on a
    // slow link (PCIe 3.0 x4: a promoted expert costs more link time than a CPU-served miss) a
    // trickle (trickles), which promotes nothing with a count of 0; Frozen and IdleOnly promote
    // nothing while requests run. An unsplit Program keeps the dynamic policy on every link
    // (multi-GPU WP19 extends the policies to it).
    bool promotes_in_rounds(const DeviceSegment& seg) const noexcept {
        if (segments_.size() < 2) { return true; }
        return seg.policy.promotions == PromotionPolicy::Dynamic || (trickles(seg) && seg.trickle > 0);
    }
    // A layer split's segment on a slow link (PromotionPolicy::Trickle): its frames hold every
    // expert they can from the warm start, most misses go to the CPU, and each round promotes at
    // most one expert per layer in a window of seg.trickle layers (the policy's trickle_bytes in
    // records, or INFERNIX_TRICKLE_PROMOTIONS) that rotates over its layers. The copies wait until
    // the segment's share of the next round is enqueued (launch_round: issue_deferred), so they cross
    // its link while the later segments compute, and no new ones start while the last are still
    // deferred or copying: at most one round's worth is on the link. A trickle replaces the
    // misses of a topic its seeded set does not cover (measured on the 20-turn conversation:
    // without it the RTX PRO 4000's hit rate at K = 20-24 fell to 85-92 % over the second half).
    bool trickles(const DeviceSegment& seg) const noexcept {
        return segments_.size() > 1 && seg.policy.promotions == PromotionPolicy::Trickle;
    }
    // A trickle's per-layer budget (its window is the residency's, set_trickle): one, none while its
    // previous promotions are still deferred or copying, none once its frames have filled under
    // INFERNIX_DECODE_PROMOTIONS=fill.
    std::size_t trickle_budget(const DeviceSegment& seg) const {
        const ExpertResidency::Stats& stats = seg.residency->stats();
        if (decode_promotions_ == DecodePromotions::Fill &&
            stats.promotions + stats.landed >= seg.residency->frames()) {
            return 0;
        }
        return seg.residency->loads_pending() ? 0 : kDecodePromotionsPerLayer;
    }
    // A trickle's promotions per round: trickle_bytes in whole records (at least one), or the
    // measurement knob INFERNIX_TRICKLE_PROMOTIONS=<n> (0: none during requests, as before trickles).
    static std::uint32_t trickle_promotions(const LinkPolicy& policy, std::uint64_t record_stride) {
        if (const char* value = std::getenv("INFERNIX_TRICKLE_PROMOTIONS"); value != nullptr && *value != '\0') {
            char* end             = nullptr;
            const unsigned long n = std::strtoul(value, &end, 10);
            if (end == value || *end != '\0' || n > 4096) {
                throw std::invalid_argument(
                    std::string("INFERNIX_TRICKLE_PROMOTIONS must be a count in [0, 4096], not '") + value + "'");
            }
            return static_cast<std::uint32_t>(n);
        }
        if (record_stride == 0) { return 1; }
        return static_cast<std::uint32_t>(std::max<std::uint64_t>(1, policy.trickle_bytes / record_stride));
    }

    // Measurement knob INFERNIX_DECODE_PROMOTIONS: `on` (unset) promotes as above; `fill` only until
    // the frames have filled; `off` never, so a run separates the link's promotion traffic from
    // compute (decode_budget is 0). Prefill calls the CPU serves take the same budget; forced-token
    // calls follow the knob too (forced_budget).
    enum class DecodePromotions : std::uint8_t { On, Fill, Off };
    static DecodePromotions decode_promotions_from_environment() {
        const char* value = std::getenv("INFERNIX_DECODE_PROMOTIONS");
        if (value == nullptr || *value == '\0' || std::strcmp(value, "on") == 0) { return DecodePromotions::On; }
        if (std::strcmp(value, "fill") == 0) { return DecodePromotions::Fill; }
        if (std::strcmp(value, "off") == 0) { return DecodePromotions::Off; }
        throw std::invalid_argument(std::string("INFERNIX_DECODE_PROMOTIONS must be off, fill or on, not '") + value + "'");
    }
    const DecodePromotions decode_promotions_ = decode_promotions_from_environment();

    // A decode or verification round's cache update (routes, LFRU, promotions) runs on the host
    // after its commit has enqueued its GPU work (the GDN fold, tail commits, the drafter's
    // catch-up), so the policy overlaps that work instead of idling the GPU after the round.
    struct DeferredRound {
        std::int32_t columns  = 0;
        std::uint32_t tokens  = 0;
        bool live             = false; // only the verification's accepted columns count
        bool pending          = false;
    };
    DeferredRound deferred_;

    void settle_round() {
        if (!deferred_.pending) { return; }
        deferred_.pending = false;
        mark(Mark::SettleBegin);
        // Each segment's cache applies the round's routes with its own budget.
        std::vector<std::size_t> budgets(segments_.size());
        for (std::size_t s = 0; s < segments_.size(); ++s) {
            budgets[s] = decode_budget(*segments_[s], deferred_.tokens);
        }
        if (trace_) {
            const std::int32_t width = deferred_.live ? round_width_ : 1;
            trace_round(deferred_.live ? RouteTraceKind::Verify : RouteTraceKind::Decode, deferred_.columns / width,
                        width, deferred_.tokens, budgets.front(), RouteTrace::kUnknownPosition,
                        deferred_.live ? std::span<const std::uint8_t>(live_.data(),
                                                                       static_cast<std::size_t>(deferred_.columns))
                                       : std::span<const std::uint8_t>{});
        }
        for (std::size_t s = 0; s < segments_.size(); ++s) {
            DeviceSegment& seg = *segments_[s];
            const SegmentWork on(*this, seg);
            seg.residency->after_round(seg.device->stream, deferred_.columns, budgets[s],
                                       deferred_.live ? std::span<const std::uint8_t>(
                                                            live_.data(), static_cast<std::size_t>(deferred_.columns))
                                                      : std::span<const std::uint8_t>{});
            // A split submits the table upload and promotions now, so they run during the head's
            // drafting rather than just before the segment's next round (WDDM batches launches).
            if (segments_.size() > 1) { seg.device->flush(); }
        }
        apply_vram_target(false);
        mark(Mark::Settled);
    }

    // The expert caches' counters, summed over the segments.
    ExpertResidency::Stats cache_stats() const {
        ExpertResidency::Stats out;
        for (const auto& segment : segments_) {
            const ExpertResidency::Stats& s = segment->residency->stats();
            out.routed += s.routed;
            out.hits += s.hits;
            out.promotions += s.promotions;
            out.lent_frames += s.lent_frames;
            out.lend_evictions += s.lend_evictions;
            out.seeded += s.seeded;
            out.landed += s.landed;
            out.demotions += s.demotions;
        }
        return out;
    }

    // ---------------------------------------------------------------- VRAM monitor (design §19.3.7)
    // With `seg`'s device current.
    void start_vram_monitor(DeviceSegment& seg, const VramDemand& demand) {
        if (!options_.vram_monitor) { return; }
        const VramSnapshot now = seg.vram->query();
        if (seg.residency->elastic()) {
            seg.control.emplace(demand, ExpertResidency::kChunkBytes, now.local_usage, seg.residency->pool_bytes(),
                                options_.vram_grow_delay_seconds);
        }
        seg.frames_now     = seg.residency->frames();
        seg.pool_now       = seg.residency->pool_bytes();
        seg.startup_frames = seg.frames_now;
        seg.vram_epoch     = Clock::now();
        const auto on_snapshot = [this, &seg](const VramSnapshot& snapshot) { on_vram_snapshot(seg, snapshot); };
        seg.monitor            = std::make_unique<VramMonitor>(seg.device->device, *seg.vram, on_snapshot);
    }

    static double vram_seconds(const DeviceSegment& seg) {
        return std::chrono::duration<double>(Clock::now() - seg.vram_epoch).count();
    }

    // The monitor thread: wakes an idle engine when the cache should resize; without an elastic
    // pool, warns once per pressure episode that nothing absorbs it.
    void on_vram_snapshot(DeviceSegment& seg, const VramSnapshot& snapshot) {
        const std::lock_guard<std::mutex> lock(vram_mutex_);
        if (kv_idle_due() && waker_) { waker_(); }
        if (seg.control) {
            const auto d = seg.control->decide(snapshot, seg.pool_now, seg.frames_now, vram_seconds(seg), true);
            if ((d.shrink || d.grow) && waker_) { waker_(); }
            return;
        }
        const std::uint64_t headroom = display_headroom(snapshot.display, false, options_.vram_headroom);
        const bool pressure = snapshot.device_free < headroom / 2 ||
                              (snapshot.has_budget && snapshot.local_usage > snapshot.local_budget);
        if (pressure && !seg.pressure_warned) {
            const std::uint64_t short_by = headroom > snapshot.device_free ? headroom - snapshot.device_free : 0;
            diagnostic("free VRAM is " + std::to_string(snapshot.device_free >> 20) + " MiB, below the " +
                       std::to_string(headroom >> 20) + " MiB headroom by " + std::to_string(short_by >> 20) +
                       " MiB; the expert cache cannot shrink on this system, so Infernix memory may move to system "
                       "memory and slow down. Raise --vram-headroom-mib to " +
                       std::to_string((headroom + short_by + (256ULL << 20)) >> 20), DiagnosticLevel::Warning);
        }
        seg.pressure_warned = pressure;
    }

public:
    void set_maintenance_waker(std::function<void()> waker) {
        const std::lock_guard<std::mutex> lock(vram_mutex_);
        waker_ = std::move(waker);
    }

    // ---- warm start (design §19.3.5 S4b) ----
    // Fills the frames from the saved state, before the first request. With the SSD tier, RAM is
    // filled first (the frames load from it), then the seeds' RAM copies are released and their slots
    // refilled with the next keys of the fill order, so the two levels hold different experts.
    void warm_start_experts() {
        const auto keys = c_.num_hidden_layers * c_.moe.experts;
        expert_cache::ExpertStateLoad load;
        if (!options_.expert_state.empty() && options_.expert_cache) {
            load = expert_cache::load_expert_state(options_.expert_state, options_.expert_state_identity, keys);
        }
        const expert_cache::SavedState* state = load.state ? &*load.state : nullptr;
        TierFill fill;
        if (tier_) { fill = prefill_tier(state); }
        const std::uint32_t seeded = warm_start_frames(load);
        if (tier_) { finish_tier_fill(fill, seeded); }
    }

private:
    // The SSD tier's startup fill: its key order, how many keys the first pass read, and its time.
    struct TierFill {
        std::vector<std::uint32_t> order;
        std::uint32_t filled = 0;
        double seconds       = 0;
        bool saved           = false;
    };

    // The SSD tier's RAM fill order (design §19.3.7 §4.9): the saved state's ranking (the last
    // session's VRAM residents, best first), then every other key that session used, most used
    // first, then the keys it never used one expert of every layer in turn, so that no layer is left
    // wholly on the SSD (file order filled the first layers and left the last ones on disk).
    [[nodiscard]] std::vector<std::uint32_t> tier_fill_order(const expert_cache::SavedState* state) const {
        const std::uint32_t layers = c_.num_hidden_layers, experts = c_.moe.experts, keys = layers * experts;
        if (tier_->keys() != keys) { throw std::logic_error("Qwen3.8-Flash-Next: the SSD tier has another key count"); }
        // A key's place in the layer-interleaved order: expert e of every layer before expert e + 1.
        const auto spread = [&](std::uint32_t key) { return (key % experts) * layers + key / experts; };
        std::vector<std::uint32_t> order;
        order.reserve(keys);
        std::vector<std::uint8_t> listed(keys, 0);
        if (state != nullptr) {
            for (const std::uint32_t key : state->ranked) {
                if (key < keys && listed[key] == 0) {
                    order.push_back(key);
                    listed[key] = 1;
                }
            }
            if (state->counts.size() == keys) {
                std::vector<std::uint32_t> used;
                for (std::uint32_t key = 0; key < keys; ++key) {
                    if (listed[key] == 0 && state->counts[key] != 0) { used.push_back(key); }
                }
                std::sort(used.begin(), used.end(), [&](std::uint32_t a, std::uint32_t b) {
                    const std::uint32_t ca = state->counts[a], cb = state->counts[b];
                    return ca != cb ? ca > cb : spread(a) < spread(b);
                });
                for (const std::uint32_t key : used) {
                    order.push_back(key);
                    listed[key] = 1;
                }
            }
        }
        for (std::uint32_t i = 0; i < keys; ++i) {
            const std::uint32_t key = (i % layers) * experts + i / layers; // spread(key) == i
            if (listed[key] == 0) { order.push_back(key); }
        }
        return order;
    }

    // The SSD tier's first RAM pass, before the frames' warm start reads from it.
    TierFill prefill_tier(const expert_cache::SavedState* state) {
        const auto start = std::chrono::steady_clock::now();
        TierFill fill;
        fill.saved = state != nullptr;
        fill.order = tier_fill_order(state);
        if (state != nullptr && state->counts.size() == tier_->keys()) {
            tier_->controller().seed_uses(state->counts, kSeedCountCap);
        }
        fill.filled  = tier_->prefill(fill.order);
        fill.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        return fill;
    }

    // After the frames' warm start: the seeds are in VRAM, so their RAM copies (shadows) are
    // released and their slots take the next keys of the fill order. A seed VRAM later evicts is
    // demoted into RAM when it outranks the coldest resident (T4), as any key without a host copy.
    void finish_tier_fill(const TierFill& fill, std::uint32_t seeded) {
        const auto start        = std::chrono::steady_clock::now();
        std::uint32_t released  = 0, refilled = 0;
        if (seeded > 0) {
            released = tier_->controller().release_shadows();
            if (released > 0) {
                refilled = tier_->prefill(std::span<const std::uint32_t>(fill.order).subspan(fill.filled));
            }
        }
        const double seconds =
            fill.seconds + std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        const std::uint32_t keys = tier_->keys(), in_ram = fill.filled - released + refilled;
        const std::uint32_t slots = tier_->controller().config().slots;
        const double gib          = static_cast<double>(whole().residency->frame_stride()) / (1ULL << 30);
        char line[400];
        std::snprintf(line, sizeof(line),
                      "SSD expert tier: %u RAM slots (%.1f GiB) hold %u experts (%s)%s, read in %.1f s; %u of %u "
                      "experts are read from the artifact when used",
                      slots, gib * slots, in_ram,
                      fill.saved ? "the saved ranking, then the last session's use counts" : "spread across the layers",
                      released > 0 ? (", none of them among the " + std::to_string(released) + " in VRAM").c_str() : "",
                      seconds, keys - in_ram - released, keys);
        diagnostic(line);
    }

    // The frames' warm start from the saved state; returns how many experts it loaded.
    // The saved state keys the whole model; a split's segments each load their part
    // (expert_cache::split_saved_state). A split's segment that seeds every frame (a slow link's,
    // which promotes only a trickle during requests) is seeded even without a saved state (none
    // yet, or --expert-state off): every key of its layers, expert-major across them (extend_ranking
    // of an empty state), so it starts with every expert its frames hold rather than filling them
    // over thousands of rounds.
    std::uint32_t warm_start_frames(const expert_cache::ExpertStateLoad& load) {
        if (!options_.expert_cache) { return 0; }
        // A split's segment seeds its link policy's share (every frame on a slow link, whose cache
        // promotes only a trickle during requests); INFERNIX_SEED_SHARE overrides both.
        const auto share_of = [&](const DeviceSegment& seg) {
            return segments_.size() > 1 && !seed_share_set() ? seg.policy.seed_share : seed_share();
        };
        const bool saved = !options_.expert_state.empty() && load.state.has_value();
        const bool seeds = saved || (segments_.size() > 1 &&
                                     std::any_of(segments_.begin(), segments_.end(),
                                                 [&](const auto& segment) { return share_of(*segment) >= 1.0; }));
        if (!saved && !options_.expert_state.empty()) {
            diagnostic(seeds ? "expert cache: no saved state yet; the segments on a slow link start with every frame "
                               "filled across their layers, the others empty, filling as requests arrive"
                             : "expert cache: no saved state yet, so it starts empty and fills as requests arrive");
            diagnostic("expert cache starts empty: " + load.message, DiagnosticLevel::Debug);
            expert_state_saved_ = std::chrono::steady_clock::now();
        }
        if (!seeds) { return 0; }
        const auto start     = std::chrono::steady_clock::now();
        std::uint32_t loaded = 0, frames = 0;
        for (const auto& segment : segments_) {
            DeviceSegment& seg = *segment;
            const DeviceGuard on(seg.device->device);
            const double share  = share_of(seg);
            const auto max_keys = static_cast<std::uint32_t>(share * seg.residency->frames());
            frames += seg.residency->frames();
            if (!saved && share < 1.0) { continue; }
            expert_cache::SavedState part;
            const expert_cache::SavedState* state = saved ? &*load.state : &part;
            if (segments_.size() > 1) {
                if (saved) {
                    part = expert_cache::split_saved_state(*load.state, seg.spec.layers.begin, seg.spec.layers.end,
                                                           c_.moe.experts);
                }
                state = &part;
                // A segment that seeds every frame follows the ranking (the last session's
                // residents, which fill only as many frames as that cache had) with every other key
                // the session used, most used first, as the SSD tier's fill order does, then the keys
                // it never used, so its frames hold every expert of its layers when they can.
                if (share >= 1.0) {
                    extend_ranking(part, seg.spec.layers.end - seg.spec.layers.begin, c_.moe.experts);
                }
            }
            loaded += seg.residency->warm_start(*state, kSeedCountCap, max_keys, seg.device->stream);
        }
        synchronize();
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        const double loaded_gib =
            static_cast<double>(loaded) * static_cast<double>(head().residency->frame_stride()) / (1ULL << 30);
        char line[256];
        if (saved) {
            std::snprintf(line, sizeof(line),
                          "expert cache: restored the %u most-used experts from the last session (%.1f GiB in %.2f s)",
                          loaded, loaded_gib, seconds);
        } else {
            std::snprintf(line, sizeof(line),
                          "expert cache: seeded %u experts across the slow-link segments' layers (%.1f GiB in %.2f s)",
                          loaded, loaded_gib, seconds);
        }
        diagnostic(line);
        std::snprintf(line, sizeof(line), "expert cache warm start: %u of %u frames from %s", loaded, frames,
                      saved ? options_.expert_state.string().c_str() : "every key, expert-major");
        diagnostic(line, DiagnosticLevel::Debug);
        expert_state_saved_  = std::chrono::steady_clock::now();
        expert_state_routed_ = cache_stats().routed;
        return loaded;
    }

public:
    // After every synchronization of a round with MoE layers: a call that could not serve an expert
    // (design §19.3.7: an unreadable record, a host service that stopped answering) left undefined
    // outputs, so the round's requests fail and the engine keeps serving. Prefix-cache entries the
    // round may have published are dropped when the lanes are released.
    void check_expert_error() {
        auto* word           = static_cast<volatile std::uint32_t*>(expert_error_.data());
        std::uint32_t value  = *word;
        if (value == 0 && testing::take_expert_fault()) { value = EIO; } // layer 0
        if (value == 0) { return; }
        *word                    = 0;
        clear_prefix_on_release_ = true;
        // A stopped service or read agent fails every later call too: stop the Engine with its cause.
        if (cpu_service_) {
            if (const std::string why = cpu_service_->failure(); !why.empty()) {
                throw std::runtime_error("Qwen3.8-Flash-Next: the CPU expert service stopped (" + why + ")");
            }
        }
        if (tier_) {
            if (const std::string why = tier_->failure(); !why.empty()) {
                throw std::runtime_error("Qwen3.8-Flash-Next: the SSD expert tier's read agent stopped (" + why + ")");
            }
        }
        const std::uint32_t code = value & 0xFFFFU;
        if (code == kErrorHandoffTimeout) {
            // A layer split's gated handoff ran out of its bound (ForwardSegment::in_bound): the
            // receiving segment's first layer ran on a stale residual.
            throw runtime::RecoverableExecutionError(
                "Qwen3.8-Flash-Next layer " + std::to_string(value >> 16U) +
                ": the layer split's handoff from the previous GPU did not arrive within " +
                std::to_string(options_.handoff_timeout_ms) + " ms (--handoff-timeout-ms)");
        }
        std::string what         = code == ops::offloaded_moe::kErrorUnservedRecord ? "an SSD-only expert had no path to it"
                                   : code == ops::offloaded_moe::kErrorHostSilent   ? "the host expert service stopped answering"
                                                                                     : "an expert record could not be read (" +
                                                                                       std::generic_category().message(static_cast<int>(code)) + ")";
        throw runtime::RecoverableExecutionError("Qwen3.8-Flash-Next layer " + std::to_string(value >> 16U) + ": " + what);
    }

public:
    // Writes the saved state (at stop, and from maintain_expert_state). Never throws.
    // A split's segments' states merge into the whole model's (expert_cache::merge_saved_states).
    void save_expert_state() noexcept {
        if (options_.expert_state.empty() || segments_.empty() || !first().residency) { return; }
        std::uint32_t frames = 0;
        for (const auto& segment : segments_) { frames += segment->residency->frames(); }
        if (frames == 0) { return; }
        try {
            expert_cache::SavedState state;
            if (segments_.size() == 1) {
                state = first().residency->saved_state();
            } else {
                std::vector<expert_cache::SavedState> parts;
                std::vector<LayerRange> ranges;
                for (const auto& segment : segments_) {
                    parts.push_back(segment->residency->saved_state());
                    ranges.push_back(segment->spec.layers);
                }
                state = expert_cache::merge_saved_states(parts, ranges, c_.num_hidden_layers, c_.moe.experts);
            }
            const std::string error =
                expert_cache::save_expert_state(options_.expert_state, options_.expert_state_identity, state);
            if (!error.empty()) { diagnostic("expert state not saved: " + error, DiagnosticLevel::Warning); }
            expert_state_saved_  = std::chrono::steady_clock::now();
            expert_state_routed_ = cache_stats().routed;
        } catch (const std::exception& error) {
            diagnostic(std::string("expert state not saved: ") + error.what(), DiagnosticLevel::Warning);
        } catch (...) {}
    }
    // Between requests: saves when rounds ran since the last save and 10 minutes have passed.
    void maintain_expert_state() noexcept {
        if (options_.expert_state.empty() || segments_.empty() || !first().residency ||
            cache_stats().routed == expert_state_routed_) {
            return;
        }
        if (std::chrono::steady_clock::now() - expert_state_saved_ < kExpertStateInterval) { return; }
        save_expert_state();
    }

private:
    // A seeded count is capped, so an expert used heavily in an old session yields to new uses.
    static constexpr std::uint32_t kSeedCountCap = 16;
    // The share of the frames a warm start fills (the best-ranked experts); the rest stay free for
    // the fill phase's landing (S4), so a workload unlike the saved state still fills quickly.
    static constexpr double kSeedShare = 0.5;
    // kSeedShare, or the measurement knob INFERNIX_SEED_SHARE=<0..1> (1 fills every frame).
    static double seed_share() {
        const char* value = std::getenv("INFERNIX_SEED_SHARE");
        if (value == nullptr || *value == '\0') { return kSeedShare; }
        char* end          = nullptr;
        const double share = std::strtod(value, &end);
        if (end == value || *end != '\0' || !(share >= 0.0 && share <= 1.0)) {
            throw std::invalid_argument(std::string("INFERNIX_SEED_SHARE must be a number in [0, 1], not '") + value + "'");
        }
        return share;
    }
    // Appends to `state`'s ranking (keys of `layers` x `experts`) every key it does not list whose
    // count is nonzero, by count (descending; ties by key), then every other key (count zero, or no
    // counts at all), expert-major across the layers (keys are layer * experts + expert), so frames
    // too few for them all spread over the layers.
    static void extend_ranking(expert_cache::SavedState& state, std::uint32_t layers, std::uint32_t experts) {
        const std::size_t keys = static_cast<std::size_t>(layers) * experts;
        if (keys == 0 || (!state.counts.empty() && state.counts.size() != keys)) { return; }
        const auto count = [&](std::uint32_t key) { return state.counts.empty() ? 0U : state.counts[key]; };
        std::vector<std::uint8_t> listed(keys, 0);
        for (const std::uint32_t key : state.ranked) {
            if (key < listed.size()) { listed[key] = 1; }
        }
        std::vector<std::uint32_t> used;
        for (std::uint32_t key = 0; key < keys; ++key) {
            if (listed[key] == 0 && count(key) != 0) { used.push_back(key); }
        }
        std::stable_sort(used.begin(), used.end(),
                         [&](std::uint32_t a, std::uint32_t b) { return count(a) > count(b); });
        state.ranked.insert(state.ranked.end(), used.begin(), used.end());
        for (std::uint32_t e = 0; e < experts; ++e) {
            for (std::uint32_t l = 0; l < layers; ++l) {
                const std::uint32_t key = l * experts + e;
                if (listed[key] == 0 && count(key) == 0) { state.ranked.push_back(key); }
            }
        }
    }
    static bool seed_share_set() {
        const char* value = std::getenv("INFERNIX_SEED_SHARE");
        return value != nullptr && *value != '\0';
    }
    static constexpr std::chrono::minutes kExpertStateInterval{10};
    std::chrono::steady_clock::time_point expert_state_saved_{};
    std::uint64_t expert_state_routed_ = 0;

public:
    // Prefill calls up to this width are CPU-assisted (design §19.3.1 P7); wider chunks route nearly
    // every expert, and their misses stream or stage on the GPU.
    static constexpr int kAssistMaxColumns = 255;

    // At a boundary (no round's kernels in flight; `idle`: no request active): resizes each segment's
    // expert cache to its control law's decision.
    void apply_vram_target(bool idle) {
        if (idle && kv_idle_due() &&
            std::all_of(lanes_.begin(), lanes_.end(), [](const Lane& l) { return l.phase == Phase::Free; })) {
            kv_busy(); // one attempt per idle period: blocks without host copies may still hold the top
            shrink_kv(true);
        }
        for (const auto& segment : segments_) { apply_vram_target(*segment, idle); }
    }

private:
    void apply_vram_target(DeviceSegment& seg, bool idle) {
        if (!seg.control) { return; }
        const DeviceGuard on(seg.device->device);
        const std::lock_guard<std::mutex> lock(vram_mutex_);
        const auto d =
            seg.control->decide(seg.monitor->latest(), seg.pool_now, seg.frames_now, vram_seconds(seg), idle);
        if (d.frames == seg.frames_now) { return; }
        const std::uint32_t before = seg.frames_now;
        const auto resized         = seg.residency->resize(d.frames, seg.device->stream, *seg.vram);
        seg.frames_now             = resized.frames;
        seg.pool_now               = seg.residency->pool_bytes();
        const VramSnapshot after   = seg.monitor->refresh();
        diagnostic(segment_name(seg) + (d.shrink ? "free VRAM fell to " : "free VRAM rose to ") +
                   std::to_string(after.device_free >> 20) + " MiB: expert cache " + std::to_string(before) +
                   " -> " + std::to_string(seg.frames_now) + " frames" +
                   (resized.spilled != 0 ? " (a chunk placed in system memory was given back)" : ""), DiagnosticLevel::Info);
        if (d.shrink && seg.frames_now * 4 < seg.startup_frames && !seg.low_frames_warned) {
            diagnostic("another program holds VRAM: the expert cache is down to " + std::to_string(seg.frames_now) +
                           " of " + std::to_string(seg.startup_frames) +
                           " frames, so decode is slower until that memory is released",
                       DiagnosticLevel::Warning);
            seg.low_frames_warned = true;
        }
        if (seg.frames_now * 4 >= seg.startup_frames) { seg.low_frames_warned = false; }
    }

private:
    // Calls this narrow are CPU-served (design §16.2, prefix P0 step 3).
    static constexpr std::int32_t kServedCallColumns = 8;
    // Prefill calls up to this many columns run in the static workspace; wider ones in the wide arena.
    static constexpr std::int32_t kStaticWorkColumns = 512;

    // Appends the round whose routes after_round is about to apply to the internal route trace
    // (ProgramOptions::route_trace); rows' lanes are host_lanes_ as staged for the round.
    void trace_round(RouteTraceKind kind, std::int32_t rows, std::int32_t width, std::uint32_t tokens,
                     std::size_t budget, std::uint32_t position, std::span<const std::uint8_t> live = {}) {
        if (!trace_) { return; }
        const RouteTraceRound round{.kind     = kind,
                                    .rows     = static_cast<std::uint32_t>(rows),
                                    .width    = static_cast<std::uint32_t>(width),
                                    .tokens   = tokens,
                                    .budget   = static_cast<std::uint32_t>(std::min<std::size_t>(budget, 0xFFFFFFFFU)),
                                    .position = position};
        const std::span<const std::int32_t> lanes(host_lanes_.data(), static_cast<std::size_t>(rows));
        if (segments_.size() == 1) {
            trace_->append(round, lanes, live, first().residency->route_host(), first().residency->route_stride());
            return;
        }
        // A split's segments log their own layers; the record covers the whole model.
        std::vector<RouteTraceSegment> parts;
        for (const auto& segment : segments_) {
            parts.push_back({.routes       = segment->residency->route_host(),
                             .route_stride = segment->residency->route_stride(),
                             .layer_begin  = segment->spec.layers.begin,
                             .layer_end    = segment->spec.layers.end});
        }
        trace_->append_segments(round, lanes, live, parts);
    }

    // Reports the expert cache over the finished request (and since start) as an Engine diagnostic,
    // or on stderr without an observer: a split's segments summed, with each device's free memory.
    void report_cache(const Lane& lane) noexcept {
        try {
            const ExpertResidency& residency = *head().residency;
            const auto& at_admission         = lane.cache_at_admission;
            const ExpertResidency::Stats s   = cache_stats();
            const auto routed   = s.routed - at_admission.routed;
            const auto hits     = s.hits - at_admission.hits;
            const auto promoted = s.promotions - at_admission.promotions;
            const auto landed   = s.landed - at_admission.landed;
            const auto cpu      = cpu_service_ ? cpu_service_->served_experts() - lane.cpu_served_at_admission : 0;
            std::uint32_t frames = 0;
            std::string free_text;
            for (const auto& segment : segments_) {
                frames += segment->residency->frames();
                if (!free_text.empty() && share_device(*segment, *segments_[segment->spec.index - 1])) { continue; }
                std::size_t free_vram = 0, total_vram = 0;
                {
                    const DeviceGuard on(segment->device->device);
                    if (cudaMemGetInfo(&free_vram, &total_vram) != cudaSuccess) { free_vram = 0; }
                }
                free_text += (free_text.empty() ? "" : " / ") + std::to_string(free_vram >> 20);
            }
            char text[384];
            std::snprintf(text, sizeof(text),
                          "expert cache: %u frames; request %.1f%% of %llu routed experts hit, %llu promotions, "
                          "%llu landed, %llu misses CPU-served; since start %.1f%%; VRAM free %s MiB",
                          frames, routed ? 100.0 * static_cast<double>(hits) / static_cast<double>(routed) : 0.0,
                          static_cast<unsigned long long>(routed), static_cast<unsigned long long>(promoted),
                          static_cast<unsigned long long>(landed),
                          static_cast<unsigned long long>(cpu),
                          s.routed ? 100.0 * static_cast<double>(s.hits) / static_cast<double>(s.routed) : 0.0,
                          free_text.c_str());
            diagnostic(text, DiagnosticLevel::Debug);
            // A split's segments one by one, since start (a request's share is the difference of
            // two reports): how far each segment's frames cover its layers' experts, its hit rate
            // and the misses its port handed the CPU.
            for (std::size_t i = 0; segments_.size() > 1 && i < segments_.size(); ++i) {
                const DeviceSegment& seg         = *segments_[i];
                const ExpertResidency::Stats& ss = seg.residency->stats();
                const std::uint32_t layers       = seg.spec.layers.end - seg.spec.layers.begin;
                std::snprintf(text, sizeof(text),
                              "expert cache, segment %u (layers %u-%u, GPU %d): %u frames for %u experts; since "
                              "start %llu of %llu routed experts hit (%.1f%%), %llu promotions, %llu refilled after "
                              "lending, %llu misses CPU-served",
                              seg.spec.index, seg.spec.layers.begin, seg.spec.layers.end - 1, seg.device->device,
                              seg.residency->frames(), layers * c_.moe.experts,
                              static_cast<unsigned long long>(ss.hits), static_cast<unsigned long long>(ss.routed),
                              ss.routed ? 100.0 * static_cast<double>(ss.hits) / static_cast<double>(ss.routed) : 0.0,
                              static_cast<unsigned long long>(ss.promotions), static_cast<unsigned long long>(ss.refilled),
                              static_cast<unsigned long long>(cpu_service_ ? cpu_service_->served_experts(seg.cpu_port) : 0));
                diagnostic(text, DiagnosticLevel::Debug);
            }
            if (tier_) {
                // The tier's counters as of the last round boundary.
                const auto& t = tier_->stats();
                const auto& a = lane.tier_at_admission;
                const auto reads = t.demand_reads - a.demand_reads;
                std::snprintf(text, sizeof(text),
                              "SSD expert tier: request %llu expert reads (%.2f GiB, mean %.2f ms), %llu failed; %llu "
                              "fetched (%llu from RAM); %llu admitted to RAM, %llu demoted from VRAM",
                              static_cast<unsigned long long>(reads),
                              static_cast<double>(reads) * static_cast<double>(residency.frame_stride()) / (1ULL << 30),
                              reads ? 1e-6 * static_cast<double>(t.read_ns - a.read_ns) / static_cast<double>(reads) : 0.0,
                              static_cast<unsigned long long>(t.demand_failures - a.demand_failures),
                              static_cast<unsigned long long>(t.fetch_records - a.fetch_records),
                              static_cast<unsigned long long>(t.fetch_from_ram - a.fetch_from_ram),
                              static_cast<unsigned long long>(t.admitted - a.admitted),
                              static_cast<unsigned long long>(s.demotions - at_admission.demotions));
                diagnostic(text, DiagnosticLevel::Debug);
            }
        } catch (...) {}
    }

    // The workspace arena of `seg` in a Program with `lanes` lanes: the forward at its widest call (a
    // prefill chunk or every lane's widest round), and on the head sampling, acceptance and the drafter.
    std::size_t workspace_capacity(const DeviceSegment& seg, std::int32_t lanes) const {
        return workspace_bytes(c_, options_, seg.plan, lanes, seg.plan.static_columns);
    }

    // Reports at startup what each lane of --max-concurrency takes from the expert cache, in frames
    // (design 19.3.5): its KV extent unless --kv-capacity fixes the pool; its recurrent, convolution
    // and QSA-tail state; its verification and drafter records; its logit columns and penalty
    // counts; and its share of the workspace. Every other allocation is independent of the lane
    // count, so one more lane costs these frames.
    void report_lane_cost(std::int32_t lanes, std::uint64_t record_stride) noexcept {
        for (const auto& segment : segments_) { report_lane_cost(*segment, lanes, record_stride); }
    }
    void report_lane_cost(const DeviceSegment& seg, std::int32_t lanes, std::uint64_t record_stride) noexcept {
        try {
            const auto n             = static_cast<std::size_t>(lanes);
            const bool fixed_kv      = options_.kv_capacity_tokens != 0;
            const std::size_t kv     = fixed_kv ? 0 : seg.kv_span.bytes / n;
            const std::size_t state =
                (seg.state_backing.bytes + seg.ple_backing.bytes + seg.tails_backing.bytes + seg.logits32.bytes +
                 seg.logits16.bytes + seg.token_counts.bytes + seg.records_backing.bytes + seg.ple_records.bytes +
                 seg.qsa_records.bytes + seg.mtp_residuals.bytes + seg.mtp_saved.bytes + seg.mtp_chain.bytes +
                 seg.mtp_records.bytes) /
                n;
            // The workspace grows with the lane count by its sampling, acceptance and drafter rows:
            // the mean step from one lane to this count, or the second lane's at one lane.
            const std::size_t wide   = workspace_capacity(seg, lanes > 1 ? lanes : 2);
            const std::size_t narrow = workspace_capacity(seg, 1);
            const std::size_t workspace = wide > narrow ? (wide - narrow) / (lanes > 1 ? n - 1 : 1) : 0;
            const double lane_frames = static_cast<double>(kv + state + workspace) /
                                       static_cast<double>(std::max<std::uint64_t>(record_stride, 1));
            char kv_text[64];
            if (fixed_kv) {
                std::snprintf(kv_text, sizeof(kv_text), "KV in the fixed --kv-capacity pool");
            } else {
                std::snprintf(kv_text, sizeof(kv_text), "KV %zu MiB", kv >> 20);
            }
            char text[320];
            std::snprintf(text, sizeof(text),
                          "expert cache: %u frames of %.2f MiB at %d lane%s; each lane holds %.1f frames (%s; "
                          "recurrent state, records and workspace %zu MiB)",
                          seg.residency->frames(), static_cast<double>(record_stride) / 1048576.0, lanes,
                          lanes == 1 ? "" : "s", lane_frames, kv_text, (state + workspace) >> 20);
            diagnostic(segment_name(seg) + text, DiagnosticLevel::Debug);
        } catch (...) {}
    }

    // Reports the finished request's n-gram row traffic (design 12.4). Every row is read on the host
    // before the call that consumes it is launched.
    void report_ngram(const Lane& lane) noexcept {
        try {
            const auto& n = lane.ngram;
            char text[192];
            std::snprintf(text, sizeof(text),
                          "n-gram rows: %llu requested, %.1f%% host-cache hits, %llu NVMe reads, %.1f ms of reads",
                          static_cast<unsigned long long>(n.rows),
                          n.rows ? 100.0 * static_cast<double>(n.hits) / static_cast<double>(n.rows) : 0.0,
                          static_cast<unsigned long long>(n.reads), static_cast<double>(n.read_ns) * 1e-6);
            std::string line = text;
            if (const auto& g = lane.gate; g.rounds > 0) {
                const DeviceGuard on(ple().device->device);
                std::uint64_t waits[2] = {};
                CUDA_CHECK(cudaMemcpyAsync(waits,
                                           static_cast<const std::byte*>(ple().gate_stats.p) +
                                               kGateStatsBytes * lane_index(lane),
                                           sizeof(waits), cudaMemcpyDeviceToHost, ple().device->stream));
                ple().device->synchronize();
                std::snprintf(text, sizeof(text),
                              "; %llu gated rounds: %llu NVMe reads (%.1f ms) behind the gate, the GPU waited in %llu "
                              "(mean %.1f us)",
                              static_cast<unsigned long long>(g.rounds), static_cast<unsigned long long>(g.reads),
                              static_cast<double>(g.read_ns) * 1e-6, static_cast<unsigned long long>(waits[1]),
                              waits[1] ? static_cast<double>(waits[0]) * 1e-3 / static_cast<double>(waits[1]) : 0.0);
                line += text;
            }
            diagnostic(line, DiagnosticLevel::Debug);
        } catch (...) {}
    }

    // A split's gated handoffs since start (ForwardSegment::in_wait_stats): how often each receiving
    // segment's gate found the residual not yet published, and its mean wait.
    void report_handoff() noexcept {
        if (handoff_ != HandoffMode::Gate) { return; }
        try {
            for (const auto& segment : segments_) {
                const DeviceSegment& seg = *segment;
                if (seg.handoff_stats.p == nullptr) { continue; }
                const DeviceGuard on(seg.device->device);
                std::uint64_t waits[2] = {};
                CUDA_CHECK(cudaMemcpyAsync(waits, seg.handoff_stats.p, sizeof(waits), cudaMemcpyDeviceToHost,
                                           seg.device->stream));
                seg.device->synchronize();
                char text[192];
                std::snprintf(text, sizeof(text),
                              "gated handoff into layer %u: waited in %llu rounds since start (mean %.1f us)",
                              seg.spec.layers.begin, static_cast<unsigned long long>(waits[1]),
                              waits[1] ? static_cast<double>(waits[0]) * 1e-3 / static_cast<double>(waits[1]) : 0.0);
                diagnostic(segment_name(seg) + text, DiagnosticLevel::Debug);
            }
        } catch (...) {}
    }

    std::uint32_t* gate_ready() const { return static_cast<std::uint32_t*>(gate_host_.data()); }

    // The n-gram gate of a decode or verification graph: the pinned rows behind the round's word.
    execution::NgramRowGate row_gate() const {
        auto* io_device = static_cast<std::byte*>(ple().io_device.p);
        return {.pinned_rows = host_io() + io_layout_.ngram,
                .ready       = gate_ready(),
                .expected    = reinterpret_cast<const std::uint32_t*>(io_device + io_layout_.gate),
                .wait_stats  = static_cast<std::uint64_t*>(ple().gate_stats.p),
                .wait_row    = reinterpret_cast<const std::int32_t*>(io_device + io_layout_.slots)};
    }

    std::uint32_t lane_index(const Lane& lane) const noexcept {
        return static_cast<std::uint32_t>(&lane - lanes_.data());
    }

    // Publishes a gated round's word if the round ends before its rows are read: the rows are
    // zeroed (the round's results are discarded with the error), so the enqueued gate passes.
    struct GateRelease {
        ProgramImpl* owner = nullptr;
        std::uint32_t word = 0;
        std::size_t bytes  = 0;
        GateRelease(ProgramImpl* o, std::uint32_t w, std::size_t b) : owner(o), word(w), bytes(b) {}
        GateRelease(const GateRelease&)            = delete;
        GateRelease& operator=(const GateRelease&) = delete;
        ~GateRelease() {
            if (owner == nullptr) { return; }
            std::memset(owner->host_io() + owner->io_layout_.ngram, 0, bytes);
            publish_pinned_word(owner->gate_ready(), word);
        }
    };

    // ProgramOptions::logits_hash: after a call or round that computed `columns` columns of logits
    // (its device work finished), one line with the 64-bit FNV-1a of the head's FP32 logits32, bit
    // for bit. `begin` and `width` name the positions of the call's (or round's first row's) lane.
    void hash_logits(const char* kind, std::uint32_t lane, std::int32_t begin, std::int32_t width,
                     std::int32_t columns) {
        if (!logits_hash_) { return; }
        DeviceSegment& seg = head();
        const DeviceGuard on(seg.device->device);
        std::vector<std::uint8_t> logits(sizeof(float) * static_cast<std::size_t>(vocab_) *
                                         static_cast<std::size_t>(columns));
        CUDA_CHECK(cudaMemcpyAsync(logits.data(), seg.logits32.p, logits.size(), cudaMemcpyDeviceToHost,
                                   seg.device->stream));
        seg.device->synchronize();
        std::uint64_t hash = 14695981039346656037ULL;
        for (const std::uint8_t byte : logits) {
            hash ^= byte;
            hash *= 1099511628211ULL;
        }
        std::fprintf(logits_hash_.get(), "%s lane=%u begin=%d width=%d columns=%d fnv1a64=%016llx\n", kind, lane,
                     begin, width, columns, static_cast<unsigned long long>(hash));
        std::fflush(logits_hash_.get());
    }
    struct FileCloser {
        void operator()(std::FILE* file) const noexcept { std::fclose(file); }
    };
    std::unique_ptr<std::FILE, FileCloser> logits_hash_;

    // A diagnostic (Info unless stated) for the Engine's observer, or stderr without one.
    void diagnostic(const std::string& text, DiagnosticLevel level = DiagnosticLevel::Info) const noexcept {
        try {
            if (options_.diagnostics.callback) {
                options_.diagnostics.callback(Diagnostic{.level = level, .message = text});
            } else {
                std::fprintf(stderr, "[engine] %s\n", text.c_str());
            }
        } catch (...) {}
    }


    // The io bytes a decode or verification round of `columns` columns reads: everything before
    // the n-gram rows (ids, positions, slots, table rows, logit columns) and its columns' rows.
    // Prefill and forced-token calls copy the whole layout (their MTP cells follow the rows).
    std::size_t io_prefix(std::int32_t columns) const {
        return io_layout_.ngram + static_cast<std::size_t>(columns) * hash_.heads() * c_.ple.table.row_bytes;
    }

    SequenceHandle handle(std::uint32_t lane) const noexcept {
        return ContractAccess::make_sequence(this, lane, lanes_[lane].epoch);
    }

    std::uint32_t lane_of(const SequenceHandle& sequence) const {
        const auto lane = ContractAccess::lane(sequence);
        if (ContractAccess::owner(sequence) != this || lane >= options_.max_concurrency ||
            lanes_[lane].phase == Phase::Free || lanes_[lane].epoch != ContractAccess::epoch(sequence)) {
            throw std::logic_error("Qwen4Exp: stale or foreign sequence handle");
        }
        return lane;
    }

    GenerationTimings timings(const Lane& lane) const {
        GenerationTimings out;
        out.prefill_seconds      = static_cast<double>(lane.prefill_ns) * 1e-9;
        out.decode_seconds       = static_cast<double>(lane.decode_ns) * 1e-9;
        out.decode_share_seconds = static_cast<double>(lane.decode_share_ns) * 1e-9;
        out.vision_seconds       = lane.vision_seconds;
        return out;
    }

    // ---- block tables: one per segment with KV, every one holding the head pool's page indices ----
    // The lane's row in the head's tables and, in a split, the same row in every other segment's.
    void acquire_rows(Lane& lane, std::uint32_t index) {
        lane.row = head().tables->acquire(static_cast<std::int32_t>(index));
        lane.mirror_rows.clear();
        if (segments_.size() < 2) { return; }
        lane.mirror_rows.resize(segments_.size());
        for (std::size_t s = 0; s < segments_.size(); ++s) {
            DeviceSegment& seg = *segments_[s];
            if (&seg == &head() || !seg.tables) { continue; }
            const DeviceGuard on(seg.device->device);
            lane.mirror_rows[s] = seg.tables->acquire(static_cast<std::int32_t>(index));
        }
    }
    static void release_rows(Lane& lane) noexcept {
        lane.row = KVExecutionRowLease{};
        lane.mirror_rows.clear();
    }
    // Publishes `pages` (leases of the head's pool) at logical pages [logical_begin, ...) of the
    // lane's row: into the head's tables, then by page index into every other segment's, each on its
    // own stream (a split grows every segment's backed pages before any index past them is published).
    void publish_rows(const Lane& lane, std::uint32_t logical_begin, std::span<const DeviceKVPageLease> pages) {
        {
            const SegmentWork on(*this, head());
            head().tables->publish(lane.row.handle(), logical_begin, pages, head().device->stream);
        }
        if (lane.mirror_rows.empty()) { return; }
        publish_mirrors(lane, logical_begin, KVExecutionTablePool::indices_of(pages));
    }
    // The same for pages the lane maps without leasing them (the prefix cache's blocks).
    void publish_rows(const Lane& lane, std::uint32_t logical_begin, std::span<const DeviceKVPageHandle> pages) {
        {
            const SegmentWork on(*this, head());
            head().tables->publish(lane.row.handle(), logical_begin, pages, head().device->stream);
        }
        if (lane.mirror_rows.empty()) { return; }
        publish_mirrors(lane, logical_begin, KVExecutionTablePool::indices_of(pages));
    }
    void publish_mirrors(const Lane& lane, std::uint32_t logical_begin, std::span<const std::int32_t> indices) {
        for (std::size_t s = 0; s < segments_.size(); ++s) {
            if (!lane.mirror_rows[s].valid()) { continue; }
            DeviceSegment& seg = *segments_[s];
            const SegmentWork on(*this, seg);
            seg.tables->publish_indices(lane.mirror_rows[s].handle(), logical_begin, indices, seg.device->stream);
        }
    }

    void release(std::uint32_t index) noexcept {
        Lane& lane = lanes_[index];
        if (walk_.span.active && walk_.span.lane == index) { walk_abandon(); }
        if (!prefilling_besides(index)) {
            for (const auto& segment : segments_) {
                try {
                    const DeviceGuard on(segment->device->device);
                    if (segment->stream_lease.valid()) {
                        try { return_stream_lease(*segment); } catch (...) {}
                    }
                    try { return_wide_work(*segment); } catch (...) {}
                } catch (...) {}
            }
        }
        prefix_release(lane);
        // A tap's copy-out of the lane's partial tail page may still be in flight (an abort right
        // after the call that captured it, before the next call waited for it): each segment's
        // compute stream waits for its part before the page returns to the pool, so a new lane's
        // call or a restore (which starts behind compute) cannot rewrite it under the copy, and
        // shrink_kv's synchronization covers the copy before it unmaps the page's chunk.
        if (prefix_.cache && lane.prefix.tail_capture != 0) {
            try {
                prefix_.cache->order_after_capture(lane.prefix.tail_capture, prefix_.compute);
            } catch (...) {}
        }
        lane.prefix.tail_capture = 0;
        lane.pages.clear();
        lane.reservation.reset();
        release_rows(lane);
        lane.history.clear();
        lane.history.shrink_to_fit();
        lane.rope = {};
        if (lane.vision) { vision_release(lane); }
        lane.proposer.reset();
        lane.readout.clear();
        lane.constraint.reset();
        lane.prompt.reset();
        lane.constraint_round.clear();
        lane.constraint_trace.clear();
        lane.phase        = Phase::Free;
        lane.state_tokens = 0;
        lane.prefill_end  = 0;
        lane.replay       = false;
        lane.extent       = 0;
        ++revision_;
        kv_busy();
        try { shrink_kv(); } catch (...) {}
    }

    // Whether the elastic pool could grow to make `need` pages available (the quote's check).
    bool kv_growable(std::uint32_t need) const noexcept {
        return static_cast<std::uint64_t>(head().pool->available_pages()) + kv_grow_pages() >= need;
    }

    // The pages one growth could add at most: grow_kv takes up to a quarter of the frames, and none
    // while frames are lent (an estimate; grow_kv decides).
    // A split grows every segment's planes in lockstep, so the fewest pages any segment could add.
    std::uint32_t kv_grow_pages() const noexcept {
        std::uint32_t pages = UINT32_MAX;
        for (const auto& segment : segments_) {
            const DeviceSegment& seg = *segment;
            if (!seg.pool) { continue; } // no attention layers: no KV to grow
            if (!seg.kv_vmm || !seg.residency || !seg.residency->elastic() || seg.residency->frames() == 0 ||
                seg.residency->stats().lent_frames != 0) {
                return 0;
            }
            const std::uint32_t room = seg.pool->capacity_pages() - seg.pool->backed_pages();
            const std::uint64_t page_bytes =
                std::max<std::uint64_t>(1, seg.kv_span.bytes / std::max<std::uint32_t>(1, seg.pool->capacity_pages()));
            const std::uint64_t frame_room =
                static_cast<std::uint64_t>(seg.residency->frames() / 4) * seg.residency->frame_stride();
            const std::uint64_t slack  = ExpertResidency::kChunkBytes + 2 * kKvChunkBytes * seg.plan.pool.planes.size();
            const std::uint64_t usable = frame_room > slack ? frame_room - slack : 0;
            pages = std::min(pages, static_cast<std::uint32_t>(std::min<std::uint64_t>(room, usable / page_bytes)));
        }
        return pages == UINT32_MAX ? 0 : pages;
    }

    // Zeroes a slot's recurrent state, PLE convolution history, QSA tails (and exact-window tags) and
    // penalty counts: each segment its own layers' on its stream, the PLE segment the PLE history,
    // the head the counts.
    void reset_slot(std::uint32_t slot) {
        const std::size_t ple_bytes  = static_cast<std::size_t>(width_) * span_ * 2;
        const auto lanes             = static_cast<std::size_t>(options_.max_concurrency);
        const std::size_t tail_bytes = static_cast<std::size_t>(di_) * (r_ - 1) * 2;
        for (const auto& segment : segments_) {
            DeviceSegment& seg   = *segment;
            const SegmentWork on(*this, seg);
            const cudaStream_t s = seg.device->stream;
            for (std::uint32_t l = 0; l < seg.spec.gdn_count; ++l) {
                Tensor conv      = seg.gdn->conv_slot(l, static_cast<std::int32_t>(slot));
                Tensor recurrent = seg.gdn->recurrent_slot(l, static_cast<std::int32_t>(slot));
                CUDA_CHECK(cudaMemsetAsync(conv.data, 0, conv.bytes(), s));
                CUDA_CHECK(cudaMemsetAsync(recurrent.data, 0, recurrent.bytes(), s));
            }
            if (seg.spec.ple) {
                CUDA_CHECK(cudaMemsetAsync(static_cast<std::byte*>(seg.ple_backing.p) + slot * ple_bytes, 0, ple_bytes, s));
            }
            for (std::uint32_t l = 0; l < seg.plan.kv_layers; ++l) {
                CUDA_CHECK(cudaMemsetAsync(static_cast<std::byte*>(seg.tails_backing.p) + (l * lanes + slot) * tail_bytes, 0,
                                           tail_bytes, s));
                if (seg.window_geometry.present()) { // the window's tags: no earlier history's row is read
                    const std::size_t tags = seg.window_geometry.lane_bytes(4);
                    CUDA_CHECK(cudaMemsetAsync(static_cast<std::byte*>(seg.window_backing.p) +
                                                   seg.window_geometry.plane_offset(l, 4) + slot * tags,
                                               0, tags, s));
                }
            }
            if (seg.spec.head) {
                CUDA_CHECK(cudaMemsetAsync(static_cast<std::int32_t*>(seg.token_counts.p) +
                                               slot * static_cast<std::size_t>(token_domain_),
                                           0, 4ULL * token_domain_, s));
            }
        }
    }

    std::byte* host_io() const { return static_cast<std::byte*>(io_host_.data()); }

    // `count` I32 words of the pinned io at byte offset `offset`.
    std::span<std::int32_t> io_words(std::size_t offset, std::size_t count) const {
        return {reinterpret_cast<std::int32_t*>(host_io() + offset), count};
    }

    // RoPE positions of `count` consecutive tokens of `lane` from `first`, as columns column ..
    // column + count - 1 of a call of `columns` columns over `batch` sequences, and the pooled-block
    // start of the lane's sequence b.
    void stage_call_rope(const Lane& lane, std::uint32_t first, std::int32_t count, std::int32_t columns,
                         std::int32_t column, std::int32_t batch, std::int32_t b) const {
        stage_rope(lane.rope, first, count, io_words(io_layout_.rope, 3ULL * columns), columns, column);
        stage_rope_value(block_start_rope(lane.rope, first, static_cast<std::uint32_t>(r_)),
                         io_words(io_layout_.block_rope, 3ULL * batch), batch, b);
    }

    // The n-gram rows of `count` positions starting at `begin`, from the tokens before them; the
    // traffic counts toward `lane`'s request.
    void stage_ngram(Lane& lane, const std::vector<std::int32_t>& history, std::int32_t begin, std::int32_t count,
                     std::size_t column) {
        ngram_window(history, begin, count);
        read_ngram(lane, count, column);
    }

    // window_ = the tokens of positions [begin - (n - 1), begin + count), EOS before the sequence.
    void ngram_window(const std::vector<std::int32_t>& history, std::int32_t begin, std::int32_t count) {
        const std::int32_t context = static_cast<std::int32_t>(c_.ple.ngram.ngram_size) - 1;
        window_.clear();
        for (std::int32_t p = begin - context; p < begin + count; ++p) {
            window_.push_back(p < 0 ? static_cast<std::int32_t>(c_.eos_token_id) : history[static_cast<std::size_t>(p)]);
        }
    }

    // Verification row b's token at position q: its history up to the anchor (position p), then its
    // drafts, padded past their extent with the last draft (the anchor without drafts).
    std::int32_t verify_token(const Lane& lane, std::size_t b, std::int32_t p, std::int32_t q) const {
        if (q < 0) { return static_cast<std::int32_t>(c_.eos_token_id); }
        if (q <= p) { return lane.history[static_cast<std::size_t>(q)]; }
        const auto& d = drafts_[b];
        const auto j  = static_cast<std::size_t>(q - p - 1);
        return j < d.size() ? d[j] : (d.empty() ? lane.history.back() : d.back());
    }

    // The n-gram rows of verification row b's columns [from, to) (W columns from position p).
    void stage_verify_ngram(Lane& lane, std::size_t b, std::int32_t p, std::int32_t W, std::int32_t from, std::int32_t to) {
        if (from >= to) { return; }
        const std::int32_t context = static_cast<std::int32_t>(c_.ple.ngram.ngram_size) - 1;
        window_.clear();
        for (std::int32_t q = p + from - context; q < p + to; ++q) { window_.push_back(verify_token(lane, b, p, q)); }
        read_ngram(lane, to - from, b * static_cast<std::size_t>(W) + static_cast<std::size_t>(from));
    }

    // Reads the rows of the `count` positions whose tokens (with their context) are in window_ into
    // io columns from `column`.
    void read_ngram(Lane& lane, std::int32_t count, std::size_t column) {
        read_ngram_to(lane, count, host_io() + io_layout_.ngram + column * hash_.heads() * c_.ple.table.row_bytes);
    }

    // As read_ngram, into `out` (count * heads rows).
    void read_ngram_to(Lane& lane, std::int32_t count, std::byte* out) {
        const std::size_t heads = hash_.heads();
        row_ids_.resize(static_cast<std::size_t>(count) * heads);
        hash_.row_ids(window_, static_cast<std::size_t>(count), row_ids_.data());
        const std::size_t row_bytes = c_.ple.table.row_bytes;
        const NgramVolume::Counters before = volume_.counters();
        volume_.read_rows(row_ids_, std::span<std::byte>(out, static_cast<std::size_t>(count) * heads * row_bytes));
        const NgramVolume::Counters& after = volume_.counters();
        lane.ngram.rows += after.rows - before.rows;
        lane.ngram.hits += after.hits - before.hits;
        lane.ngram.reads += after.reads - before.reads;
        lane.ngram.read_ns += after.read_ns - before.read_ns;
    }

    // `rows`: also read the n-gram rows (a walk chunk whose rows were prefetched skips them).
    void stage_sequence(std::uint32_t lane, std::int32_t begin, std::int32_t width,
                        const std::vector<std::int32_t>& history, bool rows = true) {
        before_restage();
        auto* ids       = reinterpret_cast<std::int32_t*>(host_io() + io_layout_.ids);
        auto* positions = reinterpret_cast<std::int32_t*>(host_io() + io_layout_.positions);
        auto* columns   = reinterpret_cast<std::int32_t*>(host_io() + io_layout_.columns);
        for (std::int32_t i = 0; i < width; ++i) {
            ids[i]       = history[static_cast<std::size_t>(begin + i)];
            positions[i] = begin + i;
        }
        columns[0] = width - 1;
        stage_call_rope(lanes_[lane], static_cast<std::uint32_t>(begin), width, width, 0, 1, 0);
        reinterpret_cast<std::int32_t*>(host_io() + io_layout_.slots)[0] = static_cast<std::int32_t>(lane);
        reinterpret_cast<std::int32_t*>(host_io() + io_layout_.rows)[0]  = static_cast<std::int32_t>(lane);
        if (rows) { stage_ngram(lanes_[lane], history, begin, width, 0); }
        host_lanes_[0] = static_cast<std::int32_t>(lane);
    }

    void stage_decode(std::span<const std::uint32_t> lanes, std::span<const std::int32_t> positions) {
        auto* ids     = reinterpret_cast<std::int32_t*>(host_io() + io_layout_.ids);
        auto* pos     = reinterpret_cast<std::int32_t*>(host_io() + io_layout_.positions);
        auto* columns = reinterpret_cast<std::int32_t*>(host_io() + io_layout_.columns);
        auto* slots   = reinterpret_cast<std::int32_t*>(host_io() + io_layout_.slots);
        auto* rows    = reinterpret_cast<std::int32_t*>(host_io() + io_layout_.rows);
        for (std::size_t b = 0; b < lanes.size(); ++b) {
            Lane& lane       = lanes_[lanes[b]];
            ids[b]           = lane.history[static_cast<std::size_t>(positions[b])];
            pos[b]           = positions[b];
            columns[b]       = static_cast<std::int32_t>(b);
            slots[b]         = static_cast<std::int32_t>(lanes[b]);
            rows[b]          = static_cast<std::int32_t>(lanes[b]);
            host_lanes_[b]   = static_cast<std::int32_t>(lanes[b]);
            stage_call_rope(lane, static_cast<std::uint32_t>(positions[b]), 1, static_cast<std::int32_t>(lanes.size()),
                            static_cast<std::int32_t>(b), static_cast<std::int32_t>(lanes.size()), static_cast<std::int32_t>(b));
        }
    }

    // The n-gram rows of a plain decode round (column b: lane b's input token).
    void stage_decode_rows(std::span<const std::uint32_t> lanes, std::span<const std::int32_t> positions) {
        for (std::size_t b = 0; b < lanes.size(); ++b) {
            Lane& lane = lanes_[lanes[b]];
            stage_ngram(lane, lane.history, positions[b], 1, b);
        }
    }

    // Whether every n-gram row of a plain decode round is in the host row cache.
    bool decode_rows_cached(std::span<const std::uint32_t> lanes, std::span<const std::int32_t> positions) {
        const std::int32_t context = static_cast<std::int32_t>(c_.ple.ngram.ngram_size) - 1;
        for (std::size_t b = 0; b < lanes.size(); ++b) {
            const Lane& lane = lanes_[lanes[b]];
            window_.clear();
            for (std::int32_t p = positions[b] - context; p <= positions[b]; ++p) {
                window_.push_back(p < 0 ? static_cast<std::int32_t>(c_.eos_token_id) : lane.history[static_cast<std::size_t>(p)]);
            }
            row_ids_.resize(hash_.heads());
            hash_.row_ids(window_, 1, row_ids_.data());
            if (!volume_.cached(row_ids_)) { return false; }
        }
        return true;
    }

    std::uint32_t next_gate_word() {
        if (++gate_sequence_ == 0) { ++gate_sequence_; }
        const std::uint32_t word = gate_sequence_;
        *reinterpret_cast<std::uint32_t*>(host_io() + io_layout_.gate) = word;
        return word;
    }

    // A layer split's round word (IoLayout::handoff), staged with the io prefix: a gated handoff's
    // sender publishes it after the residual and its receiver waits for it. Strictly increasing and
    // never 0, as the gate's.
    std::uint32_t next_handoff_word() {
        if (++handoff_sequence_ == 0) { ++handoff_sequence_; }
        const std::uint32_t word = handoff_sequence_;
        *reinterpret_cast<std::uint32_t*>(host_io() + io_layout_.handoff) = word;
        return word;
    }

    // Publishes a gated round's handoff word on every boundary if the host unwinds once a receiving
    // segment is launched and before the round's synchronization: the receivers' gates pass at
    // once (over a stale residual; the round's results are discarded with the error) instead of
    // waiting out their bound.
    struct HandoffRelease {
        const ProgramImpl* owner = nullptr;
        std::uint32_t word       = 0;
        HandoffRelease()                                 = default;
        HandoffRelease(const HandoffRelease&)            = delete;
        HandoffRelease& operator=(const HandoffRelease&) = delete;
        ~HandoffRelease() {
            if (owner == nullptr) { return; }
            for (const auto& channel : owner->channels_) { channel->release(word); }
        }
    };

    // After a gated launch: submits the queued work, reads the rows (`read`), releases the gate and
    // counts the reads toward `lane` (the round's first row).
    void read_behind_gate(Lane& lane, std::uint32_t word, const std::function<void()>& read) {
        flush();
        const NgramVolume::Counters before = volume_.counters();
        read();
        const NgramVolume::Counters& after = volume_.counters();
        publish_pinned_word(gate_ready(), word);
        ++lane.gate.rounds;
        lane.gate.reads += after.reads - before.reads;
        lane.gate.read_ns += after.read_ns - before.read_ns;
        if (after.read_ns - before.read_ns > kSlowGatedReadNs) {
            diagnostic("n-gram rows behind the gate took " + std::to_string((after.read_ns - before.read_ns) / 1000000) +
                           " ms (the GPU waited at layer 1)",
                       DiagnosticLevel::Warning);
        }
    }

    // Runs one eager Forward call over the staged inputs (prefill chunks, forced tokens); logits of
    // `logit_columns` columns land in the head's logits32. With `stream`, a segment whose call is wide
    // enough streams its experts (stream_chunk). A split runs it segment by segment (run_split).
    void run(std::int32_t batch, std::int32_t width, std::int32_t logit_columns,
             const execution::MtpChunk* chunk = nullptr, const execution::VisionInput* vision = nullptr,
             bool stream = false) {
        if (segments_.size() > 1) {
            run_split(batch, width, logit_columns, chunk, vision, stream);
            return;
        }
        DeviceSegment& seg   = whole();
        const bool streamed  = stream && stream_chunk(seg, batch * width);
        const cudaStream_t s = seg.device->stream;
        upload_io(seg);
        seg.residency->before_round(s);
        // The ring's copies queue on the copy engine behind the io upload above. A call the CPU can
        // share (design §19.3.12) gates the stream on each layer's routing.
        seg.split_call = streamed && seg.split_policy && batch * width <= split_columns(seg);
        if (streamed) { seg.expert_stream->begin(ring_span(seg), seg.residency->host_table(), s, seg.split_call); }
        const WideWork wide(*this, seg, batch * width);
        forward_call(seg, batch, width, logit_columns, nullptr, chunk, vision, nullptr, streamed);
        if (streamed) { seg.expert_stream->end(); }
        seg.split_call = false;
        seg.residency->enqueue_route_download(s, batch * width);
    }

    // Uploads the staged io image into `seg`'s copy on its stream: all of it to the PLE segment, which
    // alone reads the n-gram rows, and the rest around the rows to the others (IoLayout).
    void upload_io(DeviceSegment& seg) {
        const cudaStream_t s = seg.device->stream;
        auto* device         = static_cast<std::byte*>(seg.io_device.p);
        if (seg.spec.ple) {
            CUDA_CHECK(cudaMemcpyAsync(device, io_host_.data(), io_layout_.bytes, cudaMemcpyHostToDevice, s));
            return;
        }
        CUDA_CHECK(cudaMemcpyAsync(device, host_io(), io_layout_.ngram, cudaMemcpyHostToDevice, s));
        CUDA_CHECK(cudaMemcpyAsync(device + io_layout_.mtp_ids, host_io() + io_layout_.mtp_ids,
                                   io_layout_.bytes - io_layout_.mtp_ids, cudaMemcpyHostToDevice, s));
    }

    // ---- prefill expert streaming (design §19.3.8 F2, F3) ----
    // Chunks of at least kStreamMinColumns route nearly every expert of a layer, so all of a layer's
    // non-resident experts are copied ahead; narrower chunks keep the MoE's staging. The ring holds
    // two halves of as many records as the layer with the most non-resident experts has when it is
    // lent (+ one frame for the slot tables), so no streamed layer leaves misses to staging; it is
    // lent by the expert cache for the prompt and returned when no lane is prefilling. Halves of
    // 384 records (a cold cache misses all 512 of a layer) were 9 % slower at pp4096 (design
    // §19.3.8, F2 ring size).
    static constexpr std::int32_t kStreamMinColumns = 256;
    static constexpr std::uint32_t kStreamHalfMin   = 16;

    // Whether this chunk streams; lends the ring first when none is held. With `walk_bytes`, the
    // lease also holds a layer walk's area after the ring (stream_walk_bytes; zero when the
    // lendable frames do not cover it): the experts its frames evict widen each half by their
    // per-layer share. `min_half`: the halves hold at least that many records (another segment's
    // layers stream through the ring, MoE offload). Without `ring_wanted` (the MoE offload's first
    // segment, whose routed experts run elsewhere) the lease holds the wide arena and the walk area
    // only, and no chunk streams through it.
    bool stream_chunk(DeviceSegment& seg, std::int32_t width, std::size_t walk_bytes = 0, bool ring_wanted = true,
                      std::uint32_t min_half = 0) {
        if (!seg.expert_stream || width < kStreamMinColumns || seg.residency->frames() == 0) { return false; }
        if (!seg.stream_lease.valid()) {
            // The segment's layers (its cache's tables hold them from 0).
            const std::uint32_t layers = seg.residency->layers();
            const std::int32_t* table  = seg.residency->host_table();
            const std::uint32_t E      = c_.moe.experts;
            std::uint32_t most         = std::min(min_half, E);
            for (std::uint32_t l = 0; l < layers; ++l) {
                std::uint32_t absent = 0;
                for (std::uint32_t e = 0; e < E; ++e) { absent += table[static_cast<std::size_t>(l) * E + e] < 0 ? 1U : 0U; }
                most = std::max(most, absent);
            }
            const std::uint64_t stride  = seg.residency->frame_stride();
            const auto frames_of        = [&](std::size_t bytes) {
                return static_cast<std::uint32_t>((bytes + stride - 1) / stride);
            };
            // The wide arena rides in the lease whenever the plan has one: a prompt with a streamed
            // call runs wide calls.
            const std::uint32_t wide_frames = seg.wide_dedicated ? 0U : frames_of(seg.wide_work);
            const auto walk_frames          = frames_of(walk_bytes);
            // Another lane's Vision leases can split the frames so that no run holds the lease (design
            // §19.3.2 "On a layer split"): it is then sized again by the longest run (squeezed halves, no
            // walk area, or no lease at all) rather than failing the prompt.
            std::uint32_t lendable = seg.residency->lendable();
            for (bool retry = false;; retry = true) {
                if (wide_frames >= lendable) { return false; }
                if (!ring_wanted) {
                    if (walk_frames == 0 || wide_frames + walk_frames > lendable) { return false; }
                    const auto lease = seg.residency->try_lend(wide_frames + walk_frames, seg.device->stream, {});
                    if (!lease) {
                        if (retry) { return false; }
                        lendable = std::min(lendable, seg.residency->longest_run());
                        continue;
                    }
                    seg.stream_lease        = *lease;
                    seg.stream_ring_bytes   = 0;
                    seg.stream_wide_bytes   = static_cast<std::size_t>(wide_frames) * stride;
                    seg.stream_walk_bytes   = static_cast<std::size_t>(walk_frames) * stride;
                    seg.stream_at_lease     = seg.expert_stream->stats();
                    seg.split_at_lease      = seg.forward->split_stats();
                    seg.prefetched_at_lease = walk_.prefetched_chunks;
                    diagnostic(segment_name(seg) + "prefill stream lease: " +
                                   std::to_string(seg.stream_lease.count) + " of " +
                                   std::to_string(seg.residency->frames()) +
                                   " frames (no ring: its routed experts run on the next segment; wide arena " +
                                   std::to_string(wide_frames) + "; walk area " + std::to_string(walk_frames) + ")",
                               DiagnosticLevel::Debug);
                    return false; // nothing streams through it
                }
                const std::uint32_t evicted = wide_frames + walk_frames;
                const std::uint32_t widened = std::min(E, most + (evicted + layers - 1) / layers);
                const bool full = walk_frames > 0 && 2 * widened + 1 + wide_frames + walk_frames <= lendable;
                // A split's segment whose lendable frames do not hold the walk area beside full halves
                // (a 16 GB card at a long context, two segments sharing one GPU) squeezes the halves
                // instead of giving up the walk: a layer's experts beyond its half are staged per chunk,
                // as any miss, and the rest still cross the link once per span. Without the walk, every
                // call of the prompt would stream all of them on every segment.
                const std::uint32_t room =
                    lendable > wide_frames + walk_frames ? lendable - wide_frames - walk_frames : 0U;
                const bool squeezed =
                    walk_frames > 0 && !full && segments_.size() > 1 && room >= 2 * kStreamHalfMin + 1 &&
                    seg.expert_stream->slots_in(static_cast<std::size_t>(room) * stride) >= 2 * kStreamHalfMin;
                const bool walk          = full || squeezed;
                const std::uint32_t half = full       ? widened
                                           : squeezed ? std::min(widened, (room - 1) / 2)
                                                      : std::min(E, most + (wide_frames + layers - 1) / layers);
                const std::uint32_t ring = std::min(2 * half + 1, lendable - wide_frames);
                if (half == 0 ||
                    seg.expert_stream->slots_in(static_cast<std::size_t>(ring) * stride) < 2 * kStreamHalfMin) {
                    return false;
                }
                // The ring's writers: the segment's expert stream and, with the MoE offload, the stream of
                // the previous segment's layers.
                std::vector<cudaStream_t> writers{seg.expert_stream->stream()};
                if (seg.remote_stream) { writers.push_back(seg.remote_stream->stream()); }
                const std::uint32_t count = ring + wide_frames + (walk ? walk_frames : 0U);
                const auto lease          = seg.residency->try_lend(count, seg.device->stream, writers);
                if (!lease) {
                    if (retry) { return false; }
                    lendable = std::min(lendable, seg.residency->longest_run());
                    continue;
                }
                seg.stream_lease        = *lease;
                seg.stream_ring_bytes   = static_cast<std::size_t>(ring) * stride;
                seg.stream_wide_bytes   = static_cast<std::size_t>(wide_frames) * stride;
                seg.stream_walk_bytes   = walk ? static_cast<std::size_t>(walk_frames) * stride : 0;
                seg.stream_at_lease     = seg.expert_stream->stats();
                seg.split_at_lease      = seg.forward->split_stats();
                seg.prefetched_at_lease = walk_.prefetched_chunks;
                diagnostic(segment_name(seg) + "prefill stream lease: " + std::to_string(seg.stream_lease.count) +
                               " of " + std::to_string(seg.residency->frames()) + " frames (ring " +
                               std::to_string(ring) + " for halves of " + std::to_string(half) +
                               ", the most non-resident experts of a layer " + std::to_string(most) +
                               "; wide arena " + std::to_string(wide_frames) + "; walk area " +
                               std::to_string(walk ? walk_frames : 0U) +
                               (walk_frames > 0 && !walk ? " (not lendable)"
                                : squeezed               ? ", the halves squeezed"
                                                         : "") +
                               (retry ? "; sized by the longest run of frames, " + std::to_string(lendable) +
                                            ", other leases splitting the rest"
                                      : std::string()) +
                               ")",
                           DiagnosticLevel::Debug);
                break;
            }
        }
        return seg.stream_ring_bytes > 0; // a held lease without a ring streams nothing
    }

    // The ring's part of the stream lease.
    static DeviceSpan ring_span(const DeviceSegment& seg) noexcept {
        return DeviceSpan{seg.stream_lease.memory.data, seg.stream_ring_bytes};
    }
    // The walk's area: after the ring and the wide arena.
    static std::byte* walk_area(const DeviceSegment& seg) noexcept {
        return static_cast<std::byte*>(seg.stream_lease.memory.data) + seg.stream_ring_bytes + seg.stream_wide_bytes;
    }

    // ---- the wide workspace (design §19.3.7, VRAM item 1) ----
    // A call wider than the static arena's columns runs in the wide arena: the stream lease's part
    // when it has one, else frames lent for it alone, else (no lendable frames: the VRAM monitor
    // shrank the cache, or there are no frames) a temporary device allocation, with a warning. The
    // Forward keeps its arena reference; the arenas' contents are swapped for the call.
    class WideWork {
    public:
        WideWork(ProgramImpl& program, DeviceSegment& segment, std::int32_t columns)
            : program_(program), segment_(segment) {
            if (columns > segment.plan.static_columns && segment.wide_work > 0) {
                program.enter_wide(segment);
                active_ = true;
            }
        }
        ~WideWork() {
            if (active_) { program_.leave_wide(segment_); }
        }
        WideWork(const WideWork&)            = delete;
        WideWork& operator=(const WideWork&) = delete;

    private:
        ProgramImpl& program_;
        DeviceSegment& segment_;
        bool active_ = false;
    };

    void enter_wide(DeviceSegment& seg) {
        DeviceSpan span{};
        if (seg.wide_dedicated) {
            span = DeviceSpan{seg.wide_fallback.p, seg.wide_work};
        } else if (seg.stream_lease.valid() && seg.stream_wide_bytes >= seg.wide_work) {
            span = {static_cast<std::byte*>(seg.stream_lease.memory.data) + seg.stream_ring_bytes, seg.wide_work};
        } else {
            if (!seg.work_lease.valid() && seg.wide_fallback.p == nullptr) {
                const std::uint64_t stride = seg.residency->frame_stride();
                const auto frames          = static_cast<std::uint32_t>((seg.wide_work + stride - 1) / stride);
                if (seg.residency->frames() > 0 && frames < seg.residency->lendable()) {
                    // Nothing when other leases split the frames (another lane's Vision leases).
                    const cudaStream_t writers[] = {seg.device->stream};
                    if (auto lease = seg.residency->try_lend(frames, seg.device->stream, writers)) {
                        seg.work_lease = *lease;
                    }
                }
                if (!seg.work_lease.valid()) {
                    diagnostic("the prefill workspace could not be lent from the expert cache; " +
                                   std::to_string(seg.wide_work >> 20) + " MiB are allocated for the prompt",
                               DiagnosticLevel::Warning);
                    seg.wide_fallback = DeviceBuffer(seg.wide_work);
                }
            }
            span = seg.work_lease.valid() ? DeviceSpan{seg.work_lease.memory.data, seg.wide_work}
                                          : DeviceSpan{seg.wide_fallback.p, seg.wide_work};
        }
        seg.wide_arena = std::make_unique<WorkspaceArena>(span);
        std::swap(*seg.work, *seg.wide_arena);
        seg.wide_active = true;
    }

    static void leave_wide(DeviceSegment& seg) noexcept {
        std::swap(*seg.work, *seg.wide_arena);
        seg.wide_active = false;
    }

    // After a prompt's last wide call is enqueued: the wide arena's own lease or allocation goes back
    // (the stream lease's part goes back with it).
    void return_wide_work(DeviceSegment& seg) {
        if (seg.work_lease.valid()) { seg.residency->give_back(seg.work_lease, seg.device->stream); }
        seg.work_lease = {};
        if (seg.wide_fallback.p != nullptr && !seg.wide_dedicated) {
            seg.device->synchronize();
            seg.wide_fallback = DeviceBuffer{};
        }
    }

    // Returns the ring (after the last streamed chunk is enqueued: compute waited for every copy).
    void return_stream_lease(DeviceSegment& seg) {
        if (!seg.stream_lease.valid() || walk_.span.active) { return; }
        const auto frames = seg.stream_lease.count;
        seg.residency->give_back(seg.stream_lease, seg.device->stream);
        seg.stream_lease      = {};
        seg.stream_ring_bytes = 0;
        seg.stream_wide_bytes = 0;
        seg.stream_walk_bytes = 0;
        const auto& now       = seg.expert_stream->stats();
        const auto streamed   = now.streamed - seg.stream_at_lease.streamed;
        const auto copies     = now.copies - seg.stream_at_lease.copies;
        const auto split      = seg.forward->split_stats();
        char line[320];
        std::snprintf(line, sizeof(line),
                      "prefill expert stream: %llu experts (%.2f GiB) in %llu copies through %u lent frames; "
                      "CPU split: %llu experts on the CPU, %llu gated copies; n-gram rows of %llu walk chunks read "
                      "during the previous span",
                      static_cast<unsigned long long>(streamed),
                      static_cast<double>(streamed) * static_cast<double>(seg.residency->frame_stride()) / (1ULL << 30),
                      static_cast<unsigned long long>(copies), frames,
                      static_cast<unsigned long long>(split.cpu_experts - seg.split_at_lease.cpu_experts),
                      static_cast<unsigned long long>(split.streamed_experts - seg.split_at_lease.streamed_experts),
                      static_cast<unsigned long long>(walk_.prefetched_chunks - seg.prefetched_at_lease));
        diagnostic(segment_name(seg) + line, DiagnosticLevel::Debug);
    }

    // ---- a layer split's eager calls ----
    // One segment's share of an eager call while the later segments are enqueued: its batch over its
    // io, its residual in its workspace (alloc_residual's fixed address), the head's logits, and the
    // wide arena and expert stream it holds until it ends.
    struct EagerShare {
        execution::ForwardBatch batch;
        Tensor residual;
        Tensor logits;
        std::optional<WideWork> wide;
        bool streamed = false;
    };

    // An eager call of a split Program (prefill chunks, forced tokens): segment by segment in layer
    // order, each with its device current, composed from its Forward's steps (begin_call,
    // alloc_residual, embed or the crossing, its layers, finish on the head). The residual crosses
    // each boundary through the boundary's BulkHandoff, ordered on both streams without a host wait
    // (a decode round's one-slot HandoffChannel cannot hold a chunk); the caller synchronizes once
    // every segment is enqueued. A segment's share is prepared (io, expert cache, stream lease and
    // copies) before the crossing, so on two devices it overlaps the previous segment's layers; on
    // one device each share waits for the kernels of the previous one (SegmentEvent).
    void run_split(std::int32_t batch, std::int32_t width, std::int32_t logit_columns,
                   const execution::MtpChunk* chunk, const execution::VisionInput* vision, bool stream) {
        const std::int32_t columns = batch * width;
        std::vector<EagerShare> shares(segments_.size());
        for (std::size_t i = 0; i < segments_.size(); ++i) {
            DeviceSegment& seg = *segments_[i];
            EagerShare& share  = shares[i];
            {
                const DeviceGuard on(seg.device->device);
                begin_share(i, share, batch, width, logit_columns, chunk, vision, stream);
            }
            if (i > 0) {
                DeviceSegment& sender = *segments_[i - 1];
                sender.bulk_out->transfer(shares[i - 1].residual.data, share.residual.data, share.residual.bytes(),
                                          sender.device->stream, seg.device->stream);
                const DeviceGuard on(sender.device->device);
                sender.device->flush();
                end_share(sender, shares[i - 1], columns);
            }
            const DeviceGuard on(seg.device->device);
            if (seg.spec.embed) { seg.forward->embed(share.batch, share.residual); }
            for (std::uint32_t l = seg.spec.layers.begin; l < seg.spec.layers.end; ++l) {
                seg.forward->layer(l, share.batch, share.residual);
            }
            if (seg.spec.head) { seg.forward->finish(share.batch, share.residual, &share.logits); }
            CUDA_CHECK(cudaEventRecord(seg.compute_done.event, seg.device->stream));
            seg.device->flush();
        }
        DeviceSegment& last = head();
        const DeviceGuard on(last.device->device);
        end_share(last, shares.back(), columns);
    }

    // Segment i's share up to its residual, with its device current: the wait for a previous share on
    // its device, the stream lease, the io, the expert cache's round, the stream, the wide arena,
    // begin_call and alloc_residual.
    void begin_share(std::size_t i, EagerShare& share, std::int32_t batch, std::int32_t width,
                     std::int32_t logit_columns, const execution::MtpChunk* chunk, const execution::VisionInput* vision,
                     bool stream) {
        DeviceSegment& seg         = *segments_[i];
        const cudaStream_t s       = seg.device->stream;
        const std::int32_t columns = batch * width;
        // On one device two segments' kernels never overlap: this share follows the previous
        // segment's (this call's, or for the first the last segment's in the previous call).
        const DeviceSegment& before = i > 0 ? *segments_[i - 1] : head();
        if (share_device(seg, before)) { CUDA_CHECK(cudaStreamWaitEvent(s, before.compute_done.event, 0)); }
        share.streamed = stream && stream_chunk(seg, columns);
        upload_io(seg);
        seg.residency->before_round(s);
        seg.split_call = share.streamed && seg.split_policy && columns <= split_columns(seg);
        if (share.streamed) { seg.expert_stream->begin(ring_span(seg), seg.residency->host_table(), s, seg.split_call); }
        share.wide.emplace(*this, seg, columns);
        share.batch = eager_batch(seg, batch, width, logit_columns, chunk, vision, share.streamed);
        if (seg.spec.head) { share.logits = Tensor(seg.logits32.p, DType::FP32, {vocab_, logit_columns}); }
        seg.forward->begin_call(share.batch, seg.spec.head ? &share.logits : nullptr);
        share.residual = seg.forward->alloc_residual(share.batch);
    }

    // The end of a segment's share once its kernels and its crossing are enqueued.
    void end_share(DeviceSegment& seg, EagerShare& share, std::int32_t columns) {
        if (share.streamed) { seg.expert_stream->end(); }
        seg.split_call = false;
        share.wide.reset();
        seg.residency->enqueue_route_download(seg.device->stream, columns);
    }

    // A segment's batch of an eager call, over its io copy: the MTP cells (and a round's residual
    // export) are the head's, the image columns the first segment's; the prefix waits are every
    // segment's (each waits for its own layers' entries, recorded on its own streams), consumed by
    // the head's.
    execution::ForwardBatch eager_batch(const DeviceSegment& seg, std::int32_t batch, std::int32_t width,
                                        std::int32_t logit_columns, const execution::MtpChunk* chunk,
                                        const execution::VisionInput* vision, bool streamed) {
        execution::ForwardBatch fb = io_batch(static_cast<std::byte*>(seg.io_device.p), batch, width, logit_columns);
        fb.mtp_chunk   = seg.spec.head ? chunk : nullptr;
        fb.layer_waits = prefix_.waits_for(seg.spec.index);
        if (seg.spec.head) { prefix_.next_waits = {}; }
        fb.vision = seg.spec.embed ? vision : nullptr;
        fb.stream = streamed;
        fb.split  = streamed && seg.split_call;
        if (mtp_ && chunk == nullptr && seg.spec.head) {
            fb.residual_out = Tensor(seg.mtp_residuals.p, DType::BF16, {width_, batch * width});
        }
        return fb;
    }

    // Whether a lane other than `index` is prefilling.
    [[nodiscard]] bool prefilling_besides(std::uint32_t index) const noexcept {
        for (std::uint32_t i = 0; i < options_.max_concurrency; ++i) {
            if (i != index && lanes_[i].phase == Phase::Prefill) { return true; }
        }
        return false;
    }

    // One decode round of `batch` sequences: its inputs are staged at fixed device addresses and
    // every row is chosen on the device, so one CUDA graph per batch size serves every round. The
    // first round of a size runs eagerly (it also performs the Ops' one-time setup); the next is
    // captured and later rounds replay it.
    bool run_decode(std::int32_t batch, bool gated, GateRelease& release_gate, HandoffRelease& release_handoff,
                    const std::function<void()>& read_rows) {
        return launch_round(batch, 1, false, gated, release_gate, release_handoff, read_rows);
    }

    // Holds `stream` for `ms` milliseconds before its next work (a host function that sleeps): the
    // testing::delay_next_round hook, which makes a gated handoff time out.
    static void CUDART_CB sleep_host(void* data) {
        std::this_thread::sleep_for(std::chrono::milliseconds(reinterpret_cast<std::uintptr_t>(data)));
    }
    static void delay_stream(cudaStream_t stream, std::uint32_t ms) {
        CUDA_CHECK(cudaLaunchHostFunc(stream, &sleep_host, reinterpret_cast<void*>(static_cast<std::uintptr_t>(ms))));
    }

    // The graph of `seg` for a decode (width 1) or verification round of `batch` rows.
    DecodeGraph& round_graph(DeviceSegment& seg, std::int32_t batch, std::int32_t width, bool verify) {
        if (!verify) { return seg.graphs[static_cast<std::size_t>(batch - 1)]; }
        return seg.verify_graphs[static_cast<std::size_t>(batch - 1) * max_width_ + (width - 1)];
    }

    // Whether every segment's graph of the round is replayed (none warms or captures): a gated
    // round reads its n-gram rows after the launch only then (a warm run's one-time setup may wait
    // on the host, which would then never publish the rows).
    bool round_replayed(std::int32_t batch, std::int32_t width, bool verify) {
        return std::all_of(segments_.begin(), segments_.end(), [&](const auto& segment) {
            return round_graph(*segment, batch, width, verify).executable.ready();
        });
    }

    // Launches a decode (width 1) or verification round of `batch` rows over the staged io: segment
    // by segment in layer order, each with its device current (SegmentWork) and its own graph family
    // (replay: warmed, captured and replayed on its stream). Each segment uploads the io prefix (the
    // head also the round's acceptance arrays), opens its expert cache's round with landing frames,
    // runs its layers between its boundaries and enqueues its route download. The handoff of each
    // boundary (core/handoff.h): Event: the sender records `sent` after its graph and the receiver's
    // stream waits for it before its own; Gate: no host step between them, the receiver's graph
    // starts with its bounded gate (release_handoff publishes the round's word if the host unwinds
    // after that). A gated round's n-gram gate is armed (release_gate) once the PLE segment's graph
    // is launched; with a PLE segment before the head, its rows are read right after that launch
    // (read_rows), in either handoff mode: the later segments wait for that segment anyway, and
    // its layer 1 waits less for the rows (with the event handoff it waited in 393 of 445 rounds,
    // 255 us on average, when the rows were read after the head's work was enqueued), and a later
    // segment's bounded gate does not wait on the host's reads. Returns whether the rows were read
    // here; otherwise the caller reads them after the head's work is enqueued, as an unsplit round
    // does.
    bool launch_round(std::int32_t batch, std::int32_t width, bool verify, bool gated, GateRelease& release_gate,
                      HandoffRelease& release_handoff, const std::function<void()>& read_rows) {
        const bool split    = segments_.size() > 1;
        const std::int32_t columns = batch * width;
        const std::uint32_t word   = split ? next_handoff_word() : 0U;
        // Test hook (testing::delay_next_round): the first segment's graph starts late.
        const std::uint32_t delay_ms       = split ? testing::take_round_delay() : 0U;
        const execution::NgramRowGate gate = row_gate();
        bool read = false;
        timing_launch();
        for (const auto& segment : segments_) {
            DeviceSegment& seg = *segment;
            const SegmentWork on(*this, seg);
            const cudaStream_t s = seg.device->stream;
            upload_pinned(seg.io_device.p, io_host_.data(), io_layout_.ngram, s);
            if (verify && seg.spec.head) {
                upload_pinned(head().spec_device.p, spec_host_.data(), 4ULL * spec_layout_.licensed, s);
            }
            seg.residency->before_round(s, /*landing=*/true);
            if (seg.in != nullptr && handoff_ == HandoffMode::Event) { seg.in->wait_sent(s); }
            if (&seg == &first() && delay_ms != 0) { delay_stream(s, delay_ms); }
            if (timing_.on) { CUDA_CHECK(cudaEventRecord(seg.timing_begin.event, s)); }
            replay(seg, round_graph(seg, batch, width, verify), [&] {
                if (!verify) {
                    forward_call(seg, batch, 1, batch, nullptr, nullptr, nullptr, &gate);
                    return;
                }
                const execution::ForwardVerify view = verify_view(seg, batch, width);
                forward_call(seg, batch, width, columns, &view, nullptr, nullptr, &gate);
            });
            if (timing_.on) { CUDA_CHECK(cudaEventRecord(seg.timing_end.event, s)); }
            if (&seg == &first()) { mark(Mark::FirstLaunched); }
            // From here an enqueued gate waits for its word: an error before the word is published
            // still publishes it (over zeroed rows or a stale residual), so the streams drain and
            // the error keeps its class.
            if (gated && &seg == &ple()) { release_gate.owner = this; }
            if (seg.out != nullptr && handoff_ == HandoffMode::Event) { seg.out->record_sent(s); }
            if (seg.in != nullptr && handoff_ == HandoffMode::Gate) {
                release_handoff.owner = this;
                release_handoff.word  = word;
            }
            seg.residency->enqueue_route_download(s, columns);
            if (!split) { continue; }
            // A trickle's promotions of the last round cross the link from here, after its layers.
            if (trickles(seg)) { seg.residency->issue_deferred(s); }
            seg.device->flush();
            if (gated && &seg == &ple() && &seg != &head()) {
                read_rows();
                release_gate.owner = nullptr;
                read = true;
            }
        }
        mark(Mark::Launched);
        return read;
    }

    // Runs a fixed-address round: the first time eagerly (the Ops' one-time setup), the second
    // captured, then by replay.
    void replay(DeviceSegment& seg, DecodeGraph& graph, const std::function<void()>& body) {
        const cudaStream_t s = seg.device->stream;
        if (graph.executable.ready()) {
            graph.executable.launch(s);
        } else if (!graph.warmed) {
            body();
            graph.warmed = true;
        } else {
            // A captured body bakes in whatever it reads by address: the arena *seg.work points at and
            // the per-layer waits forward_call consumes. Neither may be a call's temporary.
            const auto waits = prefix_.waits_for(seg.spec.index);
            if (seg.wide_active || !waits[0].empty() || !waits[1].empty()) {
                throw std::logic_error("Qwen4Exp: a graph capture inside a wide call or with pending layer waits");
            }
            const std::size_t free_before = trace_ ? trace_free() : 0;
            graph.definition.capture(s, body);
            graph.executable.instantiate(graph.definition);
            graph.executable.launch(s);
            if (trace_) { trace_graph(seg, graph, free_before); }
        }
    }

    // Device free bytes, and the route trace's record of a graph executable just instantiated
    // (its family and shape follow from the slot it occupies).
    static std::size_t trace_free() {
        std::size_t free_bytes = 0, total_bytes = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
        return free_bytes;
    }
    void trace_graph(const DeviceSegment& seg, const DecodeGraph& graph, std::size_t free_before) {
        RouteTraceGraph family = RouteTraceGraph::Decode;
        std::ptrdiff_t at      = route_trace_index(seg.graphs, &graph);
        std::ptrdiff_t stride  = 1;
        if (at < 0 && (at = route_trace_index(seg.verify_graphs, &graph)) >= 0) {
            family = RouteTraceGraph::Verify;
            stride = max_width_;
        } else if (at < 0 && (at = route_trace_index(seg.mtp_draft_graphs, &graph)) >= 0) {
            family = RouteTraceGraph::MtpDraft;
            stride = mtp_k_;
        } else if (at < 0 && (at = route_trace_index(seg.mtp_catch_graphs, &graph)) >= 0) {
            family = RouteTraceGraph::MtpCatchUp;
            stride = max_width_;
        }
        if (at < 0) { return; }
        trace_->graph(family, static_cast<std::uint32_t>(at / stride + 1), static_cast<std::uint32_t>(at % stride + 1),
                      free_before, trace_free());
    }

    // One segment's share of a call through its Forward's run(): the whole call without a split; in a
    // split a decode or verification round's share, which receives the residual from the previous
    // segment and sends it to the next through their HandoffChannels in the round's handoff mode (its
    // word in the segment's io). The n-gram gate is the PLE segment's, the logits and the drafter's
    // residual export the head's.
    void forward_call(DeviceSegment& seg, std::int32_t batch, std::int32_t width, std::int32_t logit_columns,
                      const execution::ForwardVerify* verify = nullptr, const execution::MtpChunk* chunk = nullptr,
                      const execution::VisionInput* vision = nullptr, const execution::NgramRowGate* gate = nullptr,
                      bool streamed = false) {
        auto* io                   = static_cast<std::byte*>(seg.io_device.p);
        execution::ForwardBatch fb = io_batch(io, batch, width, logit_columns);
        fb.ngram_gate    = &seg == &ple() ? gate : nullptr;
        fb.verify        = verify;
        fb.mtp_chunk     = chunk;
        fb.layer_waits   = prefix_.waits_for(seg.spec.index);
        // Every segment's call reads the same restores and copy-outs, each waiting for its own
        // layers' entries; the head's, the last a call or round enqueues, consumes them.
        if (&seg == &head()) { prefix_.next_waits = {}; }
        fb.vision        = vision;
        fb.stream        = streamed;
        fb.split         = streamed && seg.split_call;
        fb.handoff       = handoff_;
        fb.handoff_word  = reinterpret_cast<const std::uint32_t*>(io + io_layout_.handoff);
        // Decode and verification rounds export their final residuals for the MTP catch-up.
        const std::int32_t cols = batch * width;
        if (mtp_ && chunk == nullptr && seg.spec.head) {
            fb.residual_out = Tensor(seg.mtp_residuals.p, DType::BF16, {width_, cols});
        }
        if (!seg.spec.head) {
            seg.forward->run(fb, nullptr);
            return;
        }
        Tensor logits(seg.logits32.p, DType::FP32, {vocab_, logit_columns});
        seg.forward->run(fb, logits);
    }

    // A call's inputs as the Forward reads them from a staged io image at `base` (the io, or one
    // chunk's copy in a layer walk's area).
    execution::ForwardBatch io_batch(std::byte* base, std::int32_t batch, std::int32_t width,
                                     std::int32_t logit_columns) const {
        const std::int32_t cols = batch * width;
        execution::ForwardBatch fb;
        fb.ids           = Tensor(base + io_layout_.ids, DType::I32, {cols});
        fb.positions     = Tensor(base + io_layout_.positions, DType::I32, {cols});
        fb.rope_positions   = Tensor(base + io_layout_.rope, DType::I32, {cols, 3});
        fb.block_start_rope = Tensor(base + io_layout_.block_rope, DType::I32, {batch, 3});
        fb.slots         = Tensor(base + io_layout_.slots, DType::I32, {batch});
        fb.table_rows    = Tensor(base + io_layout_.rows, DType::I32, {batch});
        fb.logit_columns = Tensor(base + io_layout_.columns, DType::I32, {logit_columns});
        fb.ngram_rows    = Tensor(base + io_layout_.ngram, DType::U8,
                                  {dim(c_.ple.table.row_bytes), dim(hash_.heads()), cols});
        fb.host_slots      = std::span<const std::int32_t>(host_lanes_.data(), static_cast<std::size_t>(batch));
        fb.host_table_rows = fb.host_slots;
        fb.batch           = batch;
        fb.width           = width;
        return fb;
    }

    // ---------------------------------------------------------------- speculative verification
    // A segment's verification records (its layers', the PLE's on its segment) and round arrays; the
    // host side of the round with the head's.
    void allocate_verification(DeviceSegment& seg, std::int32_t lanes, std::uint64_t& allocated) {
        allocate(seg.records_backing, seg.plan.records_bytes, allocated);
        if (seg.spec.gdn_count > 0) {
            seg.records = GdnReplayRecords(DeviceSpan{seg.records_backing.p, seg.records_backing.bytes}, seg.plan.records);
        }
        allocate(seg.ple_records, seg.plan.ple_records, allocated);
        allocate(seg.qsa_records, seg.plan.qsa_records, allocated);
        seg.folds.resize(static_cast<std::size_t>(max_width_) + 1);
        seg.verify_graphs.resize(static_cast<std::size_t>(lanes) * max_width_);
        const std::size_t cells = static_cast<std::size_t>(lanes) * max_width_;
        allocate(seg.spec_device, 4ULL * seg.plan.spec.words, allocated);
        if (!seg.spec.head) { return; }
        spec_layout_ = seg.plan.spec;
        spec_host_   = PinnedHostBuffer(4ULL * spec_layout_.words);
        pending_tokens_.assign(cells, 0);
        live_.assign(cells, 0);
        proposer_tokens_ = std::bit_ceil(std::max<std::size_t>(64, options_.max_context + 64ULL));
    }

    std::int32_t* spec_host(std::size_t offset) const { return static_cast<std::int32_t*>(spec_host_.data()) + offset; }
    std::int32_t* spec_device(std::size_t offset) const {
        return static_cast<std::int32_t*>(head().spec_device.p) + offset;
    }

    // The segment's records: its GDN layers', the PLE's (its segment's), its attention layers' keys.
    execution::ForwardVerify verify_view(const DeviceSegment& seg, std::int32_t batch, std::int32_t width) const {
        execution::ForwardVerify v;
        // A split's segment may have no GDN layers (K = 47), no attention layers (K <= 3) and no PLE.
        if (seg.spec.gdn_count > 0) { v.gdn = seg.records.narrowed(width); }
        if (seg.spec.ple) { v.ple_inputs = Tensor(seg.ple_records.p, DType::BF16, {width_, width, batch}); }
        if (seg.spec.attention_count > 0) {
            v.qsa_keys = Tensor(seg.qsa_records.p, DType::BF16, {di_, width, batch, dim(seg.spec.attention_count)});
        }
        return v;
    }

    static const ops::GdnReplayFoldPlan& fold(DeviceSegment& seg, std::int32_t width) {
        auto& plan = seg.folds.at(static_cast<std::size_t>(width));
        if (!plan) { plan.emplace(seg.records.narrowed(width), seg.gdn->all_layers_view()); }
        return *plan;
    }

    // One verification round: every row's anchor and drafts (rows with fewer drafts are padded with
    // filler drafts past their extent), one forward over batch x width columns that leaves all
    // recurrent state in place, then on-device acceptance. The pending batch licenses each row's
    // accepted drafts and its correction or bonus token; commit folds the accepted prefix in.
    //
    // The n-gram rows not yet staged (`staged[b]` columns of row b are) are read on the host after
    // the launch, while the GPU embeds and runs layer 0, and released to the round's gate before
    // ple_embed (design §12.3, n-gram S2). A round that captures or warms its graph reads them
    // first and releases the gate before the launch. Every call that writes the landing area
    // ends in a stream sync before the next round stages, so a gate never reads it across calls.
    PendingBatch verify(std::span<const SequenceHandle> sequences, std::span<const std::uint32_t> lanes,
                        std::span<const std::int32_t> positions, std::int32_t width,
                        std::span<const std::int32_t> staged, Clock::time_point start) {
        const auto batch     = static_cast<std::int32_t>(lanes.size());
        const std::int32_t W = width, K = width - 1;
        before_restage();
        auto* ids     = reinterpret_cast<std::int32_t*>(host_io() + io_layout_.ids);
        auto* pos     = reinterpret_cast<std::int32_t*>(host_io() + io_layout_.positions);
        auto* columns = reinterpret_cast<std::int32_t*>(host_io() + io_layout_.columns);
        auto* slots   = reinterpret_cast<std::int32_t*>(host_io() + io_layout_.slots);
        auto* rows    = reinterpret_cast<std::int32_t*>(host_io() + io_layout_.rows);
        for (std::int32_t b = 0; b < batch; ++b) {
            Lane& lane = lanes_[lanes[b]];
            const auto n = static_cast<std::int32_t>(drafts_[b].size());
            const std::int32_t p = positions[b];
            for (std::int32_t j = 0; j < W; ++j) {
                ids[b * W + j]     = verify_token(lane, static_cast<std::size_t>(b), p, p + j);
                pos[b * W + j]     = p + j;
                columns[b * W + j] = b * W + j;
            }
            for (std::int32_t j = 0; j < K; ++j) { spec_host(spec_layout_.drafts)[b * K + j] = ids[b * W + j + 1]; }
            slots[b]       = static_cast<std::int32_t>(lanes[b]);
            rows[b]        = static_cast<std::int32_t>(lanes[b]);
            host_lanes_[b] = static_cast<std::int32_t>(lanes[b]);
            stage_call_rope(lane, static_cast<std::uint32_t>(p), W, batch * W, b * W, batch, b);
            spec_host(spec_layout_.extents)[b] = n;
            spec_host(spec_layout_.lengths)[b] = p;
            spec_host(spec_layout_.anchors)[b] = lane.history.back();
        }
        const auto read_rows = [&] {
            for (std::int32_t b = 0; b < batch; ++b) {
                stage_verify_ngram(lanes_[lanes[b]], static_cast<std::size_t>(b), positions[b], W, staged[b], W);
            }
        };
        const std::uint32_t word = next_gate_word();
        const bool gated         = round_replayed(batch, W, true);
        if (!gated) {
            read_rows();
            publish_pinned_word(gate_ready(), word);
        }
        // An enqueued gate waits for `word`: an error before it is published still publishes it
        // (over zeroed rows), so the streams drain and the error keeps its class.
        GateRelease release_gate{nullptr, word, io_prefix(batch * W) - io_layout_.ngram};
        HandoffRelease release_handoff;
        const bool rows_read = launch_round(batch, W, true, gated, release_gate, release_handoff,
                                            [&] { read_behind_gate(lanes_[lanes[0]], word, read_rows); });

        // Acceptance on the head's device, from BF16-rounded logits as in plain decode (design 16.5).
        const SegmentWork on_head(*this, head());
        const cudaStream_t s = head().device->stream;
        Tensor wide(head().logits32.p, DType::FP32, {vocab_, batch * W});
        Tensor narrow(head().logits16.p, DType::BF16, {vocab_, batch * W});
        ops::cast_fp32_to_bf16(wide, narrow, s);
        Tensor target(spec_device(spec_layout_.target), DType::I32, {batch * W});
        ops::argmax(narrow, target, token_domain_, s);
        const bool constrained = stage_constraints(lanes, W);
        if (constrained) { constrain(narrow, &target); }
        auto* configs = static_cast<ops::SamplingConfig*>(host_configs_.data());
        for (std::int32_t b = 0; b < batch; ++b) {
            configs[b]      = lanes_[lanes[b]].sampling;
            configs[b].mask = grammar_mask(static_cast<std::size_t>(b), drafts_[b]);
        }
        upload_pinned(head().configs.p, configs, sizeof(ops::SamplingConfig) * batch, s);
        Tensor drafts(spec_device(spec_layout_.drafts), DType::I32, {K, batch});
        Tensor extents(spec_device(spec_layout_.extents), DType::I32, {batch});
        Tensor lengths(spec_device(spec_layout_.lengths), DType::I32, {batch});
        Tensor anchors(spec_device(spec_layout_.anchors), DType::I32, {batch});
        Tensor licensed(spec_device(spec_layout_.licensed), DType::I32, {W, batch});
        Tensor counts(spec_device(spec_layout_.counts), DType::I32, {batch});
        Tensor accepted(spec_device(spec_layout_.accepted), DType::I32, {batch});
        {
            auto scope = head().work->scope();
            ops::speculative_accept_greedy_drafts(target.view({W, batch}), narrow.view({vocab_, W, batch}), drafts,
                                                  extents, lengths, anchors, licensed, counts, accepted,
                                                  token_domain_,
                                                  static_cast<const ops::SamplingConfig*>(head().configs.p),
                                                  *head().work, s);
        }
        CUDA_CHECK(cudaMemcpyAsync(spec_host(spec_layout_.licensed), licensed.data,
                                   4ULL * (spec_layout_.commit - spec_layout_.licensed), cudaMemcpyDeviceToHost, s));
        if (gated && !rows_read) {
            read_behind_gate(lanes_[lanes[0]], word, read_rows);
            release_gate.owner = nullptr;
        }
        synchronize();
        timing_synchronized(W);
        release_handoff.owner = nullptr; // every segment has synchronized
        check_expert_error();
        hash_logits("verify", lanes[0], positions[0], W, batch * W);

        const std::int32_t* licensed_host = spec_host(spec_layout_.licensed);
        for (std::int32_t b = 0; b < batch; ++b) {
            Lane& lane     = lanes_[lanes[b]];
            const auto n   = static_cast<std::uint64_t>(drafts_[b].size());
            const auto L   = spec_host(spec_layout_.counts)[b];
            const auto A   = spec_host(spec_layout_.accepted)[b];
            if (L < 1 || L > W || A != L - 1) { throw std::logic_error("Qwen4Exp: acceptance returned an invalid extent"); }
            for (std::int32_t j = 0; j < W; ++j) {
                pending_tokens_[static_cast<std::size_t>(b * W + j)] = j < L ? licensed_host[b * W + j] : 0;
                live_[static_cast<std::size_t>(b * W + j)]           = j <= A ? 1 : 0;
            }
            pending_counts_[b] = L;
            if (constrained) {
                take_constraint_draws(lane, static_cast<std::size_t>(b * W),
                                      std::span<const std::int32_t>(&pending_tokens_[static_cast<std::size_t>(b * W)],
                                                                    static_cast<std::size_t>(L)));
            }
            auto& stats        = lane.speculative;
            if (n == 0) {
                ++stats.fallback_steps;
                continue;
            }
            ++stats.rounds;
            stats.drafted_tokens += n;
            stats.accepted_tokens += static_cast<std::uint64_t>(A);
            if (!from_ngram_[b]) {
                // Positions 1..A were accepted; position A + 1 (when drafted) was the first rejected.
                for (std::uint64_t j = 0; j < n && j <= static_cast<std::uint64_t>(A); ++j) {
                    double& a = lane.mtp_accept[static_cast<std::size_t>(j)];
                    a = (1.0 - kAcceptanceWeight) * a + kAcceptanceWeight * (j < static_cast<std::uint64_t>(A) ? 1.0 : 0.0);
                }
            }
            if (from_ngram_[b]) {
                ++stats.ngram_rounds;
                stats.ngram_drafted_tokens += n;
                stats.ngram_accepted_tokens += static_cast<std::uint64_t>(A);
            }
            for (std::int32_t j = 0; j < A; ++j) { ++stats.accepted_per_position[static_cast<std::size_t>(j)]; }
        }
        std::uint32_t advanced = 1;
        for (std::int32_t b = 0; b < batch; ++b) {
            advanced = std::max<std::uint32_t>(advanced, static_cast<std::uint32_t>(pending_counts_[b]));
        }
        deferred_ = {.columns = batch * W, .tokens = advanced, .live = true, .pending = true};
        round_width_ = W;
        runtime::ExecutionTiming timing;
        timing.submit_host_ns = elapsed_ns(start);
        for (std::int32_t b = 0; b < batch; ++b) {
            lanes_[lanes[b]].decode_ns += timing.submit_host_ns;
            lanes_[lanes[b]].decode_share_ns += timing.submit_host_ns / static_cast<std::uint64_t>(batch);
        }
        pending_transaction_ = ++next_transaction_;
        auto pending = ContractAccess::make_pending(this, pending_transaction_, sequences,
                                                    {pending_tokens_.data(), static_cast<std::size_t>(batch * W)},
                                                    timing, {pending_counts_.data(), static_cast<std::size_t>(batch)},
                                                    static_cast<std::uint32_t>(W));
        mark_grammar_failures(pending, {pending_counts_.data(), static_cast<std::size_t>(batch)});
        return pending;
    }

    // Commits a verification round: each row keeps its first accepted_tokens licensed tokens, so
    // the model state advances by that many columns (the anchor and the accepted drafts before the
    // last kept token). The GDN fold, the QSA tails and the PLE history replay those columns' records.
    // Each segment commits its own layers on its stream, from its copy of the round's io and the commit
    // counts uploaded to it: its GDN layers' fold, its attention layers' QSA tails, the PLE history on
    // the PLE segment; then the drafter's catch-up on the head.
    CommitResult commit_verified(PendingBatch&& pending, std::span<const runtime::CommitDecision> decisions) {
        const auto rows      = ContractAccess::rows(pending);
        const auto batch     = static_cast<std::int32_t>(rows.size());
        const std::int32_t W = round_width_;
        CommitResult out;
        out.row_count = rows.size();
        std::array<ops::GdnReplayFoldRow, kMaximumConcurrency> fold_rows{};
        before_restage();
        std::int32_t* commit = spec_host(spec_layout_.commit);
        for (std::int32_t row = 0; row < batch; ++row) {
            const auto index = ContractAccess::lane(rows[row]);
            Lane& lane       = lanes_.at(index);
            const auto& d    = decisions[static_cast<std::size_t>(row)];
            fold_rows[row]   = {static_cast<std::int32_t>(index), static_cast<std::int32_t>(index), 0};
            commit[row]      = 0;
            if (d.cancelled) {
                lane.constraint_round.clear();
                out.rows[row].timings     = timings(lane);
                out.rows[row].speculative = lane.speculative;
                out.rows[row].disposition = runtime::CommitDisposition::CancelledReleased;
                continue;
            }
            const auto k = static_cast<std::int32_t>(d.accepted_tokens);
            if (k < 1 || k > pending_counts_[row]) { throw std::logic_error("Qwen4Exp: commit exceeds the licensed tokens"); }
            commit_constraint_draws(lane, static_cast<std::size_t>(k));
            for (std::int32_t j = 0; j < k; ++j) {
                const std::int32_t token = pending_tokens_[static_cast<std::size_t>(row * W + j)];
                lane.history.push_back(token);
                if (lane.proposer) { lane.proposer->append(token); }
            }
            fold_rows[row].commit_columns = k;
            commit[row]                   = k;
            lane.state_tokens += static_cast<std::uint32_t>(k);
            if (d.terminal) {
                lane.phase                = Phase::Finishable;
                out.rows[row].disposition = runtime::CommitDisposition::Finishable;
            } else {
                out.rows[row].disposition = runtime::CommitDisposition::Active;
            }
        }
        for (const auto& segment : segments_) {
            DeviceSegment& seg = *segment;
            const SegmentWork on(*this, seg);
            const cudaStream_t s = seg.device->stream;
            auto* commit_device  = static_cast<std::int32_t*>(seg.spec_device.p) + spec_layout_.commit;
            upload_pinned(commit_device, commit, 4ULL * batch, s);
            if (seg.spec.gdn_count > 0) {
                fold(seg, W).execute(
                    std::span<const ops::GdnReplayFoldRow>(fold_rows.data(), static_cast<std::size_t>(batch)), s);
            }
            auto* io = static_cast<std::byte*>(seg.io_device.p);
            const Tensor commit_columns(commit_device, DType::I32, {batch});
            const Tensor slots(io + io_layout_.slots, DType::I32, {batch});
            const Tensor positions(io + io_layout_.positions, DType::I32, {W, batch});
            if (const auto layers = dim(seg.spec.attention_count); layers > 0) {
                Tensor tails(seg.tails_backing.p, DType::BF16, {di_, r_ - 1, dim(options_.max_concurrency), layers});
                ops::qsa_commit_tails(Tensor(seg.qsa_records.p, DType::BF16, {di_, W, batch, layers}), positions,
                                      commit_columns, tails, slots, execution::qsa_geometry(c_), s);
            }
            if (seg.spec.ple) {
                Tensor ple_states(seg.ple_backing.p, DType::BF16, {width_, span_, dim(options_.max_concurrency)});
                ops::ple_conv_commit(Tensor(seg.ple_records.p, DType::BF16, {width_, W, batch}), commit_columns,
                                     ple_states, slots, s);
            }
            if (seg.spec.head && mtp_) { mtp_catch_up(rows, W, commit); }
            // A split submits an earlier segment's commit now, so its fold and tails run while the
            // later segments commit and the head catches up and drafts, not just before its next
            // round (a batched driver model, WDDM, holds launches until a flush or a blocking call).
            if (segments_.size() > 1 && !seg.spec.head) { seg.device->flush(); }
        }
        for (const auto& row : rows) { require_mtp_settled(lanes_[ContractAccess::lane(row)]); }
        for (std::int32_t row = 0; row < batch; ++row) {
            if (!decisions[static_cast<std::size_t>(row)].cancelled) {
                Lane& lane = lanes_[ContractAccess::lane(rows[row])];
                prefix_publish(lane, prefix_frontier(lane, false));
            }
        }
        settle_round();
        for (std::int32_t row = 0; row < batch; ++row) {
            if (decisions[static_cast<std::size_t>(row)].cancelled) { release(ContractAccess::lane(rows[row])); }
        }
        ContractAccess::consume(pending);
        pending_transaction_ = 0;
        round_width_         = 1;
        return out;
    }

    // ---------------------------------------------------------------- MTP drafter (design 11.2)
    void allocate_mtp(DeviceSegment& seg, std::int32_t lanes, std::uint64_t& allocated) {
        const std::size_t column = seg.plan.mtp_column_bytes;
        mtp_columns_ = seg.plan.mtp_columns;
        allocate(seg.mtp_residuals, column * lanes * max_width_, allocated);
        allocate(seg.mtp_saved, column * lanes, allocated);
        allocate(seg.mtp_chain, column * lanes, allocated);
        allocate(seg.mtp_records, 2ULL * di_ * max_width_ * lanes, allocated);
        seg.mtp_saved.fill(0);
        std::vector<float> ones(static_cast<std::size_t>(c_.hc.streams) * mtp_columns_, 1.0F);
        allocate(seg.mtp_ones, seg.plan.mtp_ones, allocated);
        seg.mtp_ones.copy_from_host(ones.data(), ones.size() * sizeof(float));
        mtp_io_ = seg.plan.mtp_io;
        allocate(seg.mtp_device, 4ULL * mtp_io_.words, allocated);
        mtp_host_ = PinnedHostBuffer(4ULL * mtp_io_.words);
        for (auto& d : mtp_drafts_) { d.assign(static_cast<std::size_t>(mtp_k_), 0); }
        seg.mtp_catch_graphs.resize(static_cast<std::size_t>(lanes) * max_width_);
        seg.mtp_draft_graphs.resize(static_cast<std::size_t>(lanes) * mtp_k_);
    }

    std::int32_t* mtp_host(std::size_t offset) const { return static_cast<std::int32_t*>(mtp_host_.data()) + offset; }
    std::int32_t* mtp_device(std::size_t offset) const {
        return static_cast<std::int32_t*>(head().mtp_device.p) + offset;
    }
    Tensor saved_column(std::uint32_t lane) const {
        return Tensor(static_cast<std::byte*>(head().mtp_saved.p) + 2ULL * width_ * lane, DType::BF16, {width_});
    }

    // MR1's settled invariant (see Lane): the pending cell is the last processed position's.
    bool mtp_settled(const Lane& lane) const noexcept {
        return !mtp_ || !lane.mtp_live || lane.mtp_cells == lane.state_tokens ||
               lane.mtp_cells + 1U == lane.state_tokens;
    }
    // A drafter whose cells fall out of step with the model state would only propose worse drafts,
    // which verification rejects: output is unaffected. The request keeps decoding without drafts
    // and the defect is reported, rather than failing the request.
    void require_mtp_settled(Lane& lane) const noexcept {
        if (mtp_settled(lane)) { return; }
        lane.mtp_live = false;
        diagnostic("Qwen4Exp: the MTP drafter's cells are out of step with the model state; drafting is "
                 "off for the rest of this request", DiagnosticLevel::Warning);
    }

    // The MTP cells of a chunk (prefill or forced tokens) of one lane. The pending cell (the
    // position before the chunk) is prepended unless written; the chunk's last cell is included
    // only when the token after it is known (forced tokens). Stages ids and cells into the io.
    std::optional<execution::MtpChunk> stage_mtp_chunk(Lane& lane, std::uint32_t index, std::int32_t begin,
                                                       std::int32_t width) {
        if (!mtp_) { return std::nullopt; }
        const bool prepend = begin > 0 && lane.mtp_cells < static_cast<std::uint32_t>(begin);
        const bool known   = static_cast<std::int32_t>(lane.history.size()) > begin + width;
        const std::int32_t columns = known ? width : width - 1;
        const std::int32_t first   = prepend ? begin - 1 : begin;
        const std::int32_t cells   = (prepend ? 1 : 0) + columns;
        auto* ids   = reinterpret_cast<std::int32_t*>(host_io() + io_layout_.mtp_ids);
        auto* cells_host = reinterpret_cast<std::int32_t*>(host_io() + io_layout_.mtp_cells);
        for (std::int32_t i = 0; i < cells; ++i) {
            cells_host[i] = first + i;
            ids[i]        = lane.history[static_cast<std::size_t>(first + i + 1)];
        }
        // Cell c takes the target's RoPE position of c (design §19.3.2), per sub-chunk.
        const std::int32_t subchunks = (cells + execution::kMtpChunkColumns - 1) / execution::kMtpChunkColumns;
        stage_mtp_chunk_rope(lane.rope, static_cast<std::uint32_t>(first), cells, execution::kMtpChunkColumns,
                             static_cast<std::uint32_t>(r_), io_words(io_layout_.mtp_rope, 3ULL * cells),
                             io_words(io_layout_.mtp_block_rope, 3ULL * subchunks));
        lane.mtp_cells = static_cast<std::uint32_t>(begin + columns);
        lane.mtp_live  = true;
        auto* base = static_cast<std::byte*>(head().io_device.p);
        execution::MtpChunk chunk;
        chunk.saved     = saved_column(index);
        chunk.prepend   = prepend;
        chunk.columns   = columns;
        // A one-token prompt has no cell yet (its only position is the pending cell).
        if (cells > 0) {
            chunk.ids              = Tensor(base + io_layout_.mtp_ids, DType::I32, {cells});
            chunk.positions        = Tensor(base + io_layout_.mtp_cells, DType::I32, {cells});
            chunk.rope_positions   = reinterpret_cast<std::int32_t*>(base + io_layout_.mtp_rope);
            chunk.block_start_rope = reinterpret_cast<std::int32_t*>(base + io_layout_.mtp_block_rope);
            chunk.vision           = stage_mtp_vision(lane, static_cast<std::uint32_t>(first), cells, subchunks);
        }
        return chunk;
    }

    // MTP drafts for every live lane of a decode round: the pending cell's K/V when unwritten,
    // then mtp_k_ chained full steps from the saved residual and the anchor token.
    // Draft-length policy (design 11.3): the K in [0, mtp_k_] that maximizes expected tokens per
    // round time. Expected tokens are 1 + the sum of the products of the lane's conditional
    // position acceptances (learned online); a round of K drafts costs 1 + kWidthCost * K plain
    // rounds, the slope measured on the 5090 (design 19.2), so the plain round time cancels. Every
    // kProbeInterval rounds a row drafts one token more than its choice, so the estimate of the
    // next position stays current (K = 0 then probes one draft).
    std::int32_t choose_draft_length(Lane& lane) const {
        std::int32_t best = 0;
        double best_score = 1.0;
        double reach = 1.0, tokens = 1.0;
        for (std::int32_t k = 1; k <= mtp_k_; ++k) {
            reach *= lane.mtp_accept[static_cast<std::size_t>(k - 1)];
            tokens += reach;
            const double score = tokens / (1.0 + width_cost_ * k);
            if (score > best_score) {
                best_score = score;
                best       = k;
            }
        }
        if (++lane.mtp_policy_rounds % kProbeInterval == 0) { best = std::min(best + 1, mtp_k_); }
        return best;
    }

    // Two-row rounds (design 19.3.5): the K in [0, kPairMaxDrafts] that maximizes both rows' expected
    // tokens per round time, where a round of K drafts per row costs 1 + kPairWidthCost * K plain
    // two-row rounds. Every kProbeInterval pair rounds it drafts one token more than its choice, so
    // the rows' estimates of the next position stay current.
    std::int32_t choose_pair_draft_length(const Lane& a, const Lane& b) {
        if (!a.mtp_live || !b.mtp_live) { return 0; }
        const std::int32_t most = std::min(kPairMaxDrafts, mtp_k_);
        std::int32_t best = 0;
        double best_score = 2.0;
        double reach_a = 1.0, reach_b = 1.0, tokens = 2.0;
        for (std::int32_t k = 1; k <= most; ++k) {
            reach_a *= a.mtp_accept[static_cast<std::size_t>(k - 1)];
            reach_b *= b.mtp_accept[static_cast<std::size_t>(k - 1)];
            tokens += reach_a + reach_b;
            const double score = tokens / (1.0 + kPairWidthCost * k);
            if (score > best_score) {
                best_score = score;
                best       = k;
            }
        }
        if (++pair_policy_rounds_ % kProbeInterval == 0) { best = std::min(best + 1, most); }
        return best;
    }

    // `while_drafting` runs on the host after the draft steps are submitted, before their wait
    // (only when draft steps run).
    void mtp_draft(std::span<const std::uint32_t> lanes, std::int32_t steps,
                   const std::function<void()>& while_drafting = {}) {
        const cudaStream_t s = head().device->stream;
        const auto batch     = static_cast<std::int32_t>(lanes.size());
        const std::size_t column = 2ULL * width_;
        bool any = false, unwritten = false;
        for (std::int32_t b = 0; b < batch; ++b) {
            any |= lanes_[lanes[b]].mtp_live;
            unwritten |= lanes_[lanes[b]].mtp_live && lanes_[lanes[b]].mtp_cells < lanes_[lanes[b]].state_tokens;
        }
        // Without drafts (a gated or zero-length round) only unwritten pending cells need the drafter.
        if (!any || (steps == 0 && !unwritten)) { return; }
        // The drafter runs on the head, before the round's segments.
        const SegmentWork on_head(*this, head());
        before_restage();
        auto* slots = reinterpret_cast<std::int32_t*>(host_io() + io_layout_.slots);
        auto* rows  = reinterpret_cast<std::int32_t*>(host_io() + io_layout_.rows);
        for (std::int32_t b = 0; b < batch; ++b) {
            const Lane& lane = lanes_[lanes[b]];
            const std::int32_t cell  = static_cast<std::int32_t>(lane.state_tokens) - 1;
            // The lane's last mapped position: its shared prefix pages, then its private ones.
            const std::int32_t limit =
                static_cast<std::int32_t>(lane.prefix.page_base + lane.pages.size()) * kPagedKVPageSize - 1;
            mtp_host(mtp_io_.ids)[b] = lane.history.back();
            for (std::int32_t j = 0; j < mtp_k_; ++j) {
                const std::int32_t at = std::min(cell + j, limit);
                mtp_host(mtp_io_.cells)[j * batch + b] = at;
                const std::size_t step = 3ULL * static_cast<std::size_t>(batch) * j;
                stage_rope_value(rope_of(lane.rope, static_cast<std::uint32_t>(at)),
                                 std::span<std::int32_t>(mtp_host(mtp_io_.rope_cells + step), 3ULL * batch), batch, b);
                stage_rope_value(block_start_rope(lane.rope, static_cast<std::uint32_t>(at), static_cast<std::uint32_t>(r_)),
                                 std::span<std::int32_t>(mtp_host(mtp_io_.block_rope_cells + step), 3ULL * batch), batch, b);
            }
            slots[b] = static_cast<std::int32_t>(lanes[b]);
            rows[b]  = static_cast<std::int32_t>(lanes[b]);
            CUDA_CHECK(cudaMemcpyAsync(static_cast<std::byte*>(head().mtp_chain.p) + column * b,
                                       saved_column(lanes[b]).data, column, cudaMemcpyDeviceToDevice, s));
        }
        auto* io = static_cast<std::byte*>(head().io_device.p);
        upload_pinned(io + io_layout_.slots, slots, 4ULL * batch, s);
        upload_pinned(io + io_layout_.rows, rows, 4ULL * batch, s);
        upload_pinned(head().mtp_device.p, mtp_host_.data(), 4ULL * mtp_io_.drafts, s);
        const Tensor slot_tensor(io + io_layout_.slots, DType::I32, {batch});
        const Tensor row_tensor(io + io_layout_.rows, DType::I32, {batch});
        Tensor chain(head().mtp_chain.p, DType::BF16, {width_, batch});
        Tensor anchors(mtp_device(mtp_io_.ids), DType::I32, {batch});
        if (unwritten) {
            // The pending cells' K/V (rewriting a written cell is idempotent).
            head().forward->run_mtp({.residuals        = chain,
                                     .ids              = anchors,
                                     .positions        = Tensor(mtp_device(mtp_io_.cells), DType::I32, {batch}),
                                     .rope_positions   = Tensor(mtp_device(mtp_io_.rope_cells), DType::I32, {batch, 3}),
                                     .block_start_rope = Tensor(mtp_device(mtp_io_.block_rope_cells), DType::I32, {batch, 3}),
                                     .slots            = slot_tensor,
                                     .table_rows       = row_tensor,
                                     .batch            = batch,
                                     .width            = 1,
                                     .kv_only          = true});
        }
        if (steps == 0) {
            // No synchronize here: the uploads above may still be queued when the caller stages the
            // verification round into the same pinned slots and rows, which is safe only because it
            // stages the same lanes in the same order (and mtp_host_ is next written after that
            // round's synchronize). A caller that stages other lanes there must synchronize first.
            for (std::int32_t b = 0; b < batch; ++b) {
                Lane& lane = lanes_[lanes[b]];
                if (lane.mtp_live) { lane.mtp_cells = lane.state_tokens; }
            }
            return;
        }
        drafter_event(head().draft_begin, timing_.draft_recorded);
        replay(head(), head().mtp_draft_graphs[static_cast<std::size_t>(batch - 1) * mtp_k_ + (steps - 1)], [&] {
        for (std::int32_t j = 0; j < steps; ++j) {
            Tensor drafts(mtp_device(mtp_io_.drafts + static_cast<std::size_t>(j) * batch), DType::I32, {batch});
            const std::size_t step = 3ULL * static_cast<std::size_t>(batch) * j;
            head().forward->run_mtp({.residuals    = chain,
                                     .ids          = j == 0 ? anchors
                                                            : Tensor(mtp_device(mtp_io_.drafts + static_cast<std::size_t>(j - 1) * batch),
                                                                     DType::I32, {batch}),
                                     .positions    = Tensor(mtp_device(mtp_io_.cells + static_cast<std::size_t>(j) * batch),
                                                            DType::I32, {batch}),
                                     .rope_positions   = Tensor(mtp_device(mtp_io_.rope_cells + step), DType::I32, {batch, 3}),
                                     .block_start_rope = Tensor(mtp_device(mtp_io_.block_rope_cells + step), DType::I32, {batch, 3}),
                                     .slots        = slot_tensor,
                                     .table_rows   = row_tensor,
                                     .batch        = batch,
                                     .width        = 1,
                                     .kv_only      = false,
                                     .residual_out = chain,
                                     .drafts       = drafts,
                                     .draft_logprobs =
                                         Tensor(mtp_device(mtp_io_.logprobs + static_cast<std::size_t>(j) * batch), DType::FP32,
                                                {1, batch})});
        }
        });
        drafter_event(head().draft_end, timing_.draft_recorded);
        CUDA_CHECK(cudaMemcpyAsync(mtp_host(mtp_io_.drafts), mtp_device(mtp_io_.drafts), 4ULL * steps * batch,
                                   cudaMemcpyDeviceToHost, s));
        CUDA_CHECK(cudaMemcpyAsync(mtp_host(mtp_io_.logprobs), mtp_device(mtp_io_.logprobs), 4ULL * steps * batch,
                                   cudaMemcpyDeviceToHost, s));
        mark(Mark::DraftLaunched);
        if (while_drafting) {
            head().device->flush();
            while_drafting();
        }
        head().device->synchronize();
        mark(Mark::DraftSynced);
        drafter_synchronized();
        for (std::int32_t b = 0; b < batch; ++b) {
            Lane& lane = lanes_[lanes[b]];
            if (unwritten && lane.mtp_live) { lane.mtp_cells = lane.state_tokens; }
            for (std::int32_t j = 0; j < steps; ++j) {
                mtp_drafts_[b][static_cast<std::size_t>(j)] = mtp_host(mtp_io_.drafts)[j * batch + b];
            }
            // The row's confident drafts: those before the first below kDraftMinLogprob.
            mtp_confident_[b] = steps;
            const auto* logprobs = reinterpret_cast<const float*>(mtp_host(mtp_io_.logprobs));
            for (std::int32_t j = 0; j < steps; ++j) {
                if (logprobs[j * batch + b] < kDraftMinLogprob) {
                    mtp_confident_[b] = j;
                    break;
                }
            }
        }
    }

    // After a verification round's commit: the drafter's cells of the verified window from its
    // residuals and committed tokens (index-key tails advance by the committed count), and each
    // row's last committed residual becomes its pending cell, already written.
    // With the head's device current (its SegmentWork).
    void mtp_catch_up(std::span<const SequenceHandle> rows, std::int32_t W, const std::int32_t* commit) {
        const cudaStream_t s = head().device->stream;
        before_restage();
        const auto batch     = static_cast<std::int32_t>(rows.size());
        const std::size_t column = 2ULL * width_;
        auto* io = static_cast<std::byte*>(head().io_device.p);
        for (std::int32_t b = 0; b < batch; ++b) {
            const std::int32_t k = commit[b];
            for (std::int32_t j = 0; j < W; ++j) {
                mtp_host(mtp_io_.up_ids)[b * W + j] =
                    j < k ? pending_tokens_[static_cast<std::size_t>(b * W + j)] : pending_tokens_[static_cast<std::size_t>(b * W)];
            }
        }
        upload_pinned(mtp_device(mtp_io_.up_ids), mtp_host(mtp_io_.up_ids), 4ULL * batch * W, s);
        const Tensor positions(io + io_layout_.positions, DType::I32, {W * batch});
        const Tensor rope(io + io_layout_.rope, DType::I32, {W * batch, 3});
        const Tensor block_rope(io + io_layout_.block_rope, DType::I32, {batch, 3});
        const Tensor slots(io + io_layout_.slots, DType::I32, {batch});
        drafter_event(head().catch_begin, timing_.catch_recorded);
        replay(head(), head().mtp_catch_graphs[static_cast<std::size_t>(batch - 1) * max_width_ + (W - 1)], [&] {
        head().forward->run_mtp({.residuals   = Tensor(head().mtp_residuals.p, DType::BF16, {width_, W * batch}),
                                 .ids         = Tensor(mtp_device(mtp_io_.up_ids), DType::I32, {W * batch}),
                                 .positions   = positions,
                                 .rope_positions   = rope,
                                 .block_start_rope = block_rope,
                                 .slots       = slots,
                                 .table_rows  = Tensor(io + io_layout_.rows, DType::I32, {batch}),
                                 .batch       = batch,
                                 .width       = W,
                                 .kv_only     = true,
                                 .key_records = Tensor(head().mtp_records.p, DType::BF16, {di_, W, batch})});
        // The MTP block's tail slab follows the head's attention layers' (the whole model's unsplit).
        Tensor tails(static_cast<std::byte*>(head().tails_backing.p) + static_cast<std::size_t>(head().spec.attention_count) *
                                                                         di_ * (r_ - 1) * options_.max_concurrency * 2,
                     DType::BF16, {di_, r_ - 1, dim(options_.max_concurrency), 1});
        ops::qsa_commit_tails(Tensor(head().mtp_records.p, DType::BF16, {di_, W, batch, 1}),
                              positions.view({W, batch}), Tensor(spec_device(spec_layout_.commit), DType::I32, {batch}),
                              tails, slots, execution::qsa_geometry(c_), s);
        });
        drafter_event(head().catch_end, timing_.catch_recorded);
        for (std::int32_t b = 0; b < batch; ++b) {
            const std::int32_t k = commit[b];
            const auto lane      = ContractAccess::lane(rows[static_cast<std::size_t>(b)]);
            if (k < 1) { continue; }
            CUDA_CHECK(cudaMemcpyAsync(saved_column(lane).data,
                                       static_cast<const std::byte*>(head().mtp_residuals.p) + column * (b * W + k - 1),
                                       column, cudaMemcpyDeviceToDevice, s));
            lanes_[lane].mtp_cells = lanes_[lane].state_tokens;
        }
    }

    // Samples the next token of each row from the head's logits32 (row b = column b) into pending_tokens_.
    // `before_wait` runs on the host after the sampling work is queued, before its wait.
    void sample(std::span<const std::uint32_t> lanes, std::span<const std::int32_t> positions,
                const std::function<void()>& before_wait = {}) {
        const SegmentWork on(*this, head());
        const cudaStream_t s = head().device->stream;
        const auto batch     = static_cast<std::int32_t>(lanes.size());
        Tensor wide(head().logits32.p, DType::FP32, {vocab_, batch});
        Tensor narrow(head().logits16.p, DType::BF16, {vocab_, batch});
        // The sampler reads BF16 logits; BF16 and FP32 logits measured equal in perplexity
        // (design §16.5).
        ops::cast_fp32_to_bf16(wide, narrow, s);
        auto* configs = static_cast<ops::SamplingConfig*>(host_configs_.data());
        // Its own pinned staging, not the io image's positions: a split's later segments upload the io
        // only once the earlier ones have run, so the call's io copies may still be queued here.
        auto* sample_positions = static_cast<std::int32_t*>(host_sample_pos_.data());
        for (std::int32_t b = 0; b < batch; ++b) {
            configs[b]          = lanes_[lanes[b]].sampling;
            configs[b].mask     = grammar_mask(static_cast<std::size_t>(b), {});
            sample_positions[b] = positions[b];
        }
        upload_pinned(head().configs.p, configs, sizeof(ops::SamplingConfig) * batch, s);
        upload_pinned(head().sample_pos.p, sample_positions, 4ULL * batch, s);
        const bool constrained = stage_constraints(lanes, 1);
        if (constrained) { constrain(narrow, nullptr); }
        Tensor out(head().sampled.p, DType::I32, {batch});
        Tensor logical(head().sample_pos.p, DType::I32, {batch});
        {
            auto scope = head().work->scope();
            ops::sample(narrow, out, token_domain_, static_cast<const ops::SamplingConfig*>(head().configs.p), logical,
                        ops::kSamplePurposeDecode, *head().work, s);
        }
        CUDA_CHECK(cudaMemcpyAsync(host_sampled_.data(), head().sampled.p, 4ULL * batch, cudaMemcpyDeviceToHost, s));
        if (before_wait) { before_wait(); }
        synchronize();
        check_expert_error();
        const auto* tokens = static_cast<const std::int32_t*>(host_sampled_.data());
        for (std::int32_t b = 0; b < batch; ++b) { pending_tokens_[b] = tokens[b]; }
        if (constrained) {
            for (std::int32_t b = 0; b < batch; ++b) {
                take_constraint_draws(lanes_[lanes[b]], static_cast<std::size_t>(b),
                                      std::span<const std::int32_t>(&pending_tokens_[b], 1));
            }
        }
    }

    const execution::Parameters& parameters_;
    const TextConfig& c_;
    ProgramOptions options_;
    NgramVolume volume_;
    NgramHash hash_;
    // Every segment's control law and the cache size its monitor reads, and the waker.
    std::mutex vram_mutex_;
    std::function<void()> waker_;

    std::int32_t vocab_ = 0, token_domain_ = 0, chunk_ = 0, columns_ = 0, pages_per_row_ = 0;
    std::int32_t width_ = 0, span_ = 0, di_ = 0, r_ = 0;
    std::uint32_t kv_pages_ = 0;

    // ---------------------------------------------------------------- prefix cache (prefix_program.cpp)
    void create_prefix_cache();
    // A lane index meaning "no lane yet" (a quote before the Engine picks the destination).
    static constexpr std::uint32_t kNoLane = 0xFFFFFFFFU;
    // The root start of a request with a prefix cache: every page reserved, the cold call grid.
    PrefixSelection prefix_root(const qwen3_5::PreparedPromptData& prompt, const BasePlanImpl& base) {
        PrefixSelection root;
        root.need = binding_pages(base, static_cast<std::uint32_t>(prompt.token_ids.size()));
        root.plan = plan_prefill(prompt, 0, {}, base.reuse);
        return root;
    }
    std::optional<PrefixSelection> prefix_select(const qwen3_5::PreparedPromptData& prompt, const BasePlanImpl& base,
                                                 std::uint32_t lane);
    void pc_make_room(std::uint32_t pages);
    // Whether a fresh prompt reusing `reuse` tokens should wait for a prefilling sibling's snapshot
    // inside their shared prefix; plans that snapshot as a boundary tap of the sibling when so.
    bool prefix_await_sibling(const qwen3_5::PreparedPromptData& prompt, std::uint32_t reuse);
    // A restore batch prefetching a blocked head's blocks (none once it has landed).
    std::optional<prefix::RestoreTicket> prefetch_ticket_;
    [[nodiscard]] bool prefetch_landing() const noexcept {
        return prefetch_ticket_ && !prefix_.cache->restore_events(*prefetch_ticket_).empty();
    }

public:
    std::optional<std::uint32_t> hybrid_prefetch(const RequestBasePlan& base);
    [[nodiscard]] std::uint32_t hybrid_prefetch_room() const noexcept;
    void hybrid_hold_queue(std::span<const RequestBasePlan* const> queue);
    [[nodiscard]] std::uint64_t hybrid_cache_epoch() const noexcept;

private:
    void prefix_pin(const PrefixSelection& selection);
    void prefix_unpin(const PrefixSelection& selection) noexcept;
    void prefix_activate(Lane& lane, std::uint32_t index, const PrefixSelection& selection,
                         DeviceKVPageReservation& reservation, std::uint32_t pages);
    void copy_cache_page(DeviceKVPageHandle source, DeviceKVPageHandle destination);
    [[nodiscard]] std::uint32_t prefix_frontier(const Lane& lane, bool finishing) const noexcept;
    void prefix_publish(Lane& lane, std::uint32_t frontier);
    void prefix_capture(Lane& lane, std::uint32_t index, runtime::prefix_cache::SnapshotKind kind, bool hand_over_tail);
    void prefix_after_prefill_call(Lane& lane, std::uint32_t index, bool last);
    void prefix_finish(Lane& lane, std::uint32_t index);
    void prefix_release(Lane& lane) noexcept;
    // The cumulative Vision key of every item of the prompt: the key of each block after it.
    [[nodiscard]] static std::uint64_t prefix_trailing_extra(const qwen3_5::PreparedPromptData& prompt);
    // Every Vision item's token span: no tap and no resume lies strictly inside one.
    [[nodiscard]] static std::vector<runtime::prefix_cache::TapExclusion>
    prefix_exclusions(const qwen3_5::PreparedPromptData& prompt);
    [[nodiscard]] std::optional<runtime::prefix_cache::SnapshotRef> prefix_resident(const Lane& lane) const;
    [[nodiscard]] std::array<std::span<const cudaEvent_t>, 2> prefix_waits(const Lane& lane) const noexcept;
    void mtp_flush_cell(Lane& lane, std::uint32_t index);

public:
    // Persistence of the Host tier (--prefix-cache-file).
    PrefixCachePersistence attach_prefix_cache_file(const std::filesystem::path& path, std::string fingerprint,
                                                    const StartupObserver& observer);
    void save_prefix_cache_for_shutdown() noexcept;
    [[nodiscard]] std::optional<PrefixCachePersistence> prefix_shutdown_save() const { return prefix_shutdown_save_; }
    [[nodiscard]] std::optional<PrefixCachePersistence> save_prefix_cache_now(const CancellationView& abandoned);

private:
    std::filesystem::path prefix_file_;
    std::string prefix_fingerprint_;
    std::optional<PrefixCachePersistence> prefix_shutdown_save_;
    std::uint64_t prefix_saved_writes_ = 0; // host_write_bytes at the last save or load

public:
    [[nodiscard]] HybridPrefixCacheStats prefix_stats() const noexcept;

private:
    // The prompt's prefill calls from `frontier`, and with `taps` the prefix taps they realize.
    [[nodiscard]] prefix::CallPlan plan_prefill(const qwen3_5::PreparedPromptData& prompt, std::uint32_t frontier,
                                                std::span<const std::uint32_t> existing, bool taps) const;

    // ---- the host-to-device link (design §19.4: bandwidth-dependent choices follow the machine) ----
    // Measures `seg`'s link once, by copying expert records from the pinned banks (the SSD tier's
    // pinned slots when the banks stay in the artifact) into its staging slots before any call uses
    // them (or takes ProgramOptions::link_bytes_per_second or INFERNIX_LINK_GBPS), and derives the
    // segment's link policy from the rate. With nothing to probe the segment keeps the reference
    // link's policy. The Program-wide costs follow once every segment has measured (apply_link_costs).
    void measure_link(DeviceSegment& seg) {
        double rate = options_.link_bytes_per_second;
        if (rate <= 0.0) {
            // INFERNIX_LINK_GBPS fixes the rate (GB/s) for machines where the probe cannot be trusted.
            if (const char* fixed = std::getenv("INFERNIX_LINK_GBPS"); fixed != nullptr && std::atof(fixed) > 0.0) {
                rate = std::atof(fixed) * 1e9;
            }
        }
        const bool fixed   = rate > 0.0;
        const void* source = seg.parameters->layers[seg.parameters->layer_range().begin].moe.bank->planes.records;
        if (source == nullptr && tier_) { source = tier_->slot_bytes(0); }
        if (rate <= 0.0 && source != nullptr && seg.staging.p != nullptr && seg.plan.staging > 0) {
            // A GPU left idle while the banks were pinned runs its PCIe link at a lower generation
            // until its compute engine wakes it (an RTX 5080 behind PCIe 4.0 x16 copies 3.6 GB/s at
            // Gen1 and 7.1 at Gen2 instead of 28.4; its copy engines alone may not wake it): keep the
            // device busy for a moment first. Then probe in spaced rounds and keep the best, stopping
            // once three rounds agree.
            wake_device_link(seg.staging.p, seg.plan.staging, std::chrono::milliseconds(250), seg.device->device);
            constexpr int kRounds = 16;
            constexpr auto kGap   = std::chrono::milliseconds(250);
            double previous       = 0.0;
            int agreeing          = 0;
            for (int round = 0; round < kRounds; ++round) {
                const double r =
                    measure_h2d_bytes_per_second(source, seg.staging.p, seg.plan.staging, 3, seg.device->device);
                rate           = std::max(rate, r);
                agreeing       = previous > 0.0 && std::abs(r - previous) <= 0.03 * previous ? agreeing + 1 : 0;
                previous       = r;
                if (agreeing >= 2 && r >= 0.97 * rate) { break; }
                std::this_thread::sleep_for(kGap);
            }
        }
        const LinkPolicyInputs inputs{.cpu_split_columns         = static_cast<double>(options_.cpu_split_columns),
                                      .cpu_expert_jobs           = options_.cpu_expert_jobs,
                                      .cpu_pcie_divisor_override = options_.cpu_pcie_divisor};
        seg.link_measured = rate > 0.0;
        seg.policy        = derive_link_policy(seg.link_measured ? rate : kReferenceLinkBytesPerSecond, inputs);
        seg.trickle       = seg.policy.promotions == PromotionPolicy::Trickle
                                ? trickle_promotions(seg.policy, seg.plan.record_stride)
                                : 0U;
        if (!seg.link_measured) { return; } // no frames to probe with: keep the reference rate
        char text[256];
        std::snprintf(text, sizeof(text),
                      "host-to-device link: %.1f GB/s%s; decode misses kept on the link: 1/%d; decode promotes every "
                      "%llu tokens; prefill CPU split up to %d columns",
                      rate / 1e9, fixed ? " (set)" : " measured", pcie_divisor(seg),
                      static_cast<unsigned long long>(promotion_interval(seg)),
                      options_.prefill_cpu_split ? split_columns(seg) : 0);
        diagnostic(text, DiagnosticLevel::Debug);
        std::snprintf(text, sizeof(text), "PCIe link to the GPU: %.1f GB/s%s", rate / 1e9,
                      fixed ? " (fixed)" : " measured");
        std::string line = segment_name(seg) + text;
        if (segments_.size() > 1) {
            // A split's segment follows its link's policy (promotes_in_rounds, the CPU port's jobs,
            // the warm start's seed share).
            char promotes[96];
            if (seg.policy.promotions == PromotionPolicy::Dynamic) {
                std::snprintf(promotes, sizeof(promotes), "the expert cache promotes during requests");
            } else if (trickles(seg) && seg.trickle > 0) {
                std::snprintf(promotes, sizeof(promotes),
                              "at most %u promotions per round, after its layers (slow link)", seg.trickle);
            } else {
                std::snprintf(promotes, sizeof(promotes), "no promotions during requests (slow link)");
            }
            std::snprintf(text, sizeof(text),
                          "; %s; decode misses kept on the link: 1/%d, at most %u CPU jobs per layer call; warm "
                          "start seeds %.0f%% of the frames",
                          promotes, pcie_divisor(seg), seg.policy.cpu_jobs, 100.0 * seg.policy.seed_share);
            line += text;
        }
        diagnostic(line);
    }

    // Rescales the link-bound costs once every segment has measured its link: the prefix cost's
    // restore rate and the walk-span cost (one pass of the experts over the link), both fitted at the
    // reference rate. Each segment restores and streams its own layers over its own link, so the
    // slowest measured link bounds both; with none measured the reference costs stay.
    void apply_link_costs() {
        if (link_costs_applied_) { throw std::logic_error("Qwen4Exp: the link costs are rescaled once"); }
        link_costs_applied_ = true;
        double rate         = 0.0;
        for (const auto& segment : segments_) {
            if (!segment->link_measured) { continue; }
            const double r = segment->policy.bytes_per_second;
            rate           = rate > 0.0 ? std::min(rate, r) : r;
        }
        if (rate <= 0.0) { return; }
        options_.prefix_cost.h2d_bytes_per_second = rate;
        options_.prefix_span_seconds *= kReferenceLinkBytesPerSecond / rate;
        if (prefix_.cache) { prefix_.cache->index().set_cost(options_.prefix_cost); }
    }
    bool link_costs_applied_ = false;

    // The share of a decode call's misses the PCIe stage keeps (misses / divisor): 3 measured
    // fastest on the reference link; a faster link carries more of them (x16: 2), a slower one fewer
    // (derive_link_policy).
    [[nodiscard]] static int pcie_divisor(const DeviceSegment& seg) noexcept { return seg.policy.pcie_divisor; }

    // The widest call the prefill CPU split takes (ProgramOptions::cpu_split_columns on the reference
    // link, inversely to the measured one; at most the CPU channel's call width; derive_link_policy).
    [[nodiscard]] static std::int32_t split_columns(const DeviceSegment& seg) noexcept {
        return seg.policy.split_columns;
    }

public:
    // The slowest segment's link (the whole model's without a split).
    [[nodiscard]] double link_bytes_per_second() const {
        double rate = head().policy.bytes_per_second;
        for (const auto& segment : segments_) { rate = std::min(rate, segment->policy.bytes_per_second); }
        return rate;
    }

private:
    // ---- elastic KV (design §19.3.11) ----
    static constexpr std::size_t kKvChunkBytes   = 2ULL << 20;
    static constexpr std::uint32_t kKvBaseTokens = 32768; // backed at startup
    static constexpr std::uint32_t kKvGrowPages  = 64;    // growth unit (4,096 tokens)
    static constexpr std::chrono::seconds kKvIdleShrink{60};
    // The last admission, release or idle shrink (steady-clock ticks), and whether the pool backs more
    // than its base: the monitor thread reads both to wake an idle engine for the shrink.
    std::atomic<std::int64_t> kv_busy_at_{0};
    std::atomic<bool> kv_grown_{false};
    void kv_busy() noexcept { kv_busy_at_.store(std::chrono::steady_clock::now().time_since_epoch().count()); }
    bool kv_idle_due() const noexcept {
        return kv_grown_.load() && std::chrono::steady_clock::now().time_since_epoch().count() - kv_busy_at_.load() >=
                                       std::chrono::duration_cast<std::chrono::steady_clock::duration>(kKvIdleShrink).count();
    }

    // Maps (and zeroes) the chunks pages [0, pages) need; false when the device has no memory.
    static bool map_kv(DeviceSegment& seg, std::uint32_t pages) {
        const auto need = kv_chunks(seg.plan, pages);
        for (std::size_t c = 0; c < need.size(); ++c) {
            if (!need[c] || seg.kv_vmm->mapped(c)) { continue; }
            if (!seg.kv_vmm->map(c)) { return false; }
            CUDA_CHECK(cudaMemsetAsync(static_cast<std::byte*>(seg.kv_vmm->base()) + c * kKvChunkBytes, 0,
                                       kKvChunkBytes, seg.device->stream));
        }
        return true;
    }

    // Raises the pool's backed pages so `need` pages are available, taking the memory from the expert
    // frames (never below three quarters of them). False when it cannot: a lease holds the frames, or
    // the pool is at its capacity. A split grows every segment's planes in lockstep (the head's pool
    // hands out the pages, every segment's tables index them), each segment from its own device's
    // frames; the growth fails, and every segment keeps its frames, when one segment cannot grow.
    bool grow_kv(std::uint32_t need) {
        DeviceSegment& top = head();
        if (!top.kv_vmm || top.pool->available_pages() >= need) { return top.pool->available_pages() >= need; }
        const std::uint32_t backed = top.pool->backed_pages();
        std::uint32_t target       = backed + (need - top.pool->available_pages());
        target = std::min(top.pool->capacity_pages(), (target + kKvGrowPages - 1) / kKvGrowPages * kKvGrowPages);
        if (target - backed < need - top.pool->available_pages()) { return false; }
        const std::lock_guard<std::mutex> lock(vram_mutex_);
        // Per segment with KV: the bytes its planes map, its frames before, and the frames it gives.
        struct Growth {
            DeviceSegment* seg  = nullptr;
            std::size_t added   = 0;
            std::uint32_t frames = 0, give = 0;
        };
        std::vector<Growth> growth;
        for (const auto& segment : segments_) {
            DeviceSegment& seg = *segment;
            if (!seg.pool) { continue; } // no attention layers
            const DeviceGuard on(seg.device->device);
            const std::size_t added    = kv_mapped_bytes(seg.plan, target) - seg.kv_vmm->mapped_bytes();
            const std::uint64_t stride = seg.residency->frame_stride();
            const std::uint32_t frames = seg.residency->frames();
            // Memory the sizing function would give the frames anyway comes first; the rest is taken
            // from the frames, one frame chunk over so the pool unmaps at least that much.
            const std::uint32_t spare =
                seg.control
                    ? std::max<std::uint32_t>(seg.control->target(seg.vram->query(), seg.pool_now), frames) - frames
                    : 0;
            const std::uint64_t spare_bytes = static_cast<std::uint64_t>(spare) * stride;
            std::uint32_t give              = 0;
            if (added > spare_bytes) {
                give = static_cast<std::uint32_t>((added - spare_bytes + ExpertResidency::kChunkBytes + stride - 1) /
                                                  stride);
            }
            if (give > frames / 4) { return false; } // grow by at most a quarter of the cache at once
            growth.push_back({.seg = &seg, .added = added, .frames = frames, .give = give});
        }
        synchronize();
        // The frames of the segments before `n` that gave some come back.
        const auto undo = [&](std::size_t n) {
            for (std::size_t i = 0; i < n; ++i) {
                const Growth& g = growth[i];
                if (g.give == 0) { continue; }
                const DeviceGuard on(g.seg->device->device);
                (void)g.seg->residency->resize(g.frames, g.seg->device->stream, *g.seg->vram);
            }
        };
        for (std::size_t i = 0; i < growth.size(); ++i) {
            const Growth& g = growth[i];
            if (g.give == 0) { continue; }
            const DeviceGuard on(g.seg->device->device);
            const auto resized = g.seg->residency->resize(g.frames - g.give, g.seg->device->stream, *g.seg->vram);
            if (resized.frames > g.frames - g.give) { // a lease holds the frames
                undo(i);
                return false;
            }
        }
        for (const Growth& g : growth) {
            const DeviceGuard on(g.seg->device->device);
            if (!map_kv(*g.seg, target)) {
                undo(growth.size());
                return false;
            }
        }
        for (const Growth& g : growth) {
            DeviceSegment& seg = *g.seg;
            seg.pool->set_backed_pages(target);
            if (seg.control) { seg.control->account_fixed(static_cast<std::int64_t>(g.added)); }
            seg.frames_now = seg.residency->frames();
            seg.pool_now   = seg.residency->pool_bytes();
            diagnostic(segment_name(seg) + "KV pool grew to " + std::to_string(target) + " pages (" +
                       std::to_string(seg.kv_vmm->mapped_bytes() >> 20) + " MiB); expert cache " +
                       std::to_string(g.frames) + " -> " + std::to_string(seg.residency->frames()) + " frames");
        }
        kv_grown_.store(true);
        return true;
    }

    // Lowers the pool's backed pages to the free top (not below the base), and returns the memory to the
    // expert frames. Between rounds: nothing reads free pages. `evict` (every lane free for
    // kKvIdleShrink): the cache's idle device blocks that have host copies are released first, in
    // the index's order (nothing is lost: a later match restores them from the Host tier), so a long
    // conversation that has ended gives its memory back to the frames. The delay keeps a follow-up
    // turn from paying a restore and an expert reload for memory the idle gap never used.
    void shrink_kv(bool evict = false) {
        DeviceSegment& top = head();
        if (!top.kv_vmm || top.pool->backed_pages() <= top.plan.kv_base_pages) { return; }
        if (evict && prefix_.cache) {
            runtime::prefix_cache::PrefixCacheIndex& index = prefix_.cache->index();
            for (std::uint32_t budget = top.pool->backed_pages() - top.plan.kv_base_pages;
                 budget != 0 && !top.pool->can_back(top.plan.kv_base_pages) && index.evict_backed_device_blocks(1) != 0;
                 --budget) {}
        }
        std::uint32_t target = top.pool->backed_pages();
        while (target > top.plan.kv_base_pages && top.pool->can_back(target - std::min(target, kKvGrowPages))) {
            target -= std::min(target, kKvGrowPages);
        }
        target = std::max(target, top.plan.kv_base_pages);
        if (target == top.pool->backed_pages()) { return; }
        const std::lock_guard<std::mutex> lock(vram_mutex_);
        synchronize();
        // Every segment's planes in lockstep (another segment's pool never hands out pages, so it can
        // always lower its limit with the head's).
        for (const auto& segment : segments_) {
            DeviceSegment& seg = *segment;
            if (!seg.pool) { continue; }
            const DeviceGuard on(seg.device->device);
            const std::size_t before = seg.kv_vmm->mapped_bytes();
            seg.pool->set_backed_pages(target);
            const auto need = kv_chunks(seg.plan, target);
            for (std::size_t c = 0; c < need.size(); ++c) {
                if (!need[c] && seg.kv_vmm->mapped(c)) { seg.kv_vmm->unmap(c); }
            }
            const std::size_t freed    = before - seg.kv_vmm->mapped_bytes();
            const std::uint64_t stride = seg.residency->frame_stride();
            const std::uint32_t frames = seg.residency->frames();
            (void)seg.residency->resize(frames + static_cast<std::uint32_t>(freed / stride), seg.device->stream,
                                        *seg.vram);
            if (seg.control) { seg.control->account_fixed(-static_cast<std::int64_t>(freed)); }
            seg.frames_now = seg.residency->frames();
            seg.pool_now   = seg.residency->pool_bytes();
            diagnostic(segment_name(seg) + "KV pool shrank to " + std::to_string(target) + " pages; expert cache " +
                       std::to_string(frames) + " -> " + std::to_string(seg.residency->frames()) + " frames");
        }
        kv_grown_.store(target > top.plan.kv_base_pages);
    }

    // ---- /v1/decide (constraint.cpp) ----
    // The constraint controls: each lane's kMaximumConstraintSets sets of up to
    // ops::kTokenConstraintChoices tokens and their counts, then one descriptor and one probability
    // record per round column (lanes x max width).
    static std::size_t constraint_bytes(std::int32_t lanes, std::int32_t width) noexcept {
        const auto sets    = static_cast<std::size_t>(lanes) * kMaximumConstraintSets;
        const auto columns = static_cast<std::size_t>(lanes) * static_cast<std::size_t>(width);
        return 4ULL * (sets * ops::kTokenConstraintChoices + sets + columns + columns * ops::kTokenConstraintRecord);
    }
    [[nodiscard]] std::shared_ptr<const ConstraintPlan> compile_constraint(const TokenConstraint& constraint);
    // Writes the round's descriptors (`width` columns per row, rows in `lanes` order) and the
    // constrained lanes' sets; false (and nothing written) when no row is constrained.
    bool stage_constraints(std::span<const std::uint32_t> lanes, std::int32_t width);
    // Masks the staged round's columns of `logits` [vocab, columns] and rewrites `argmax` for them.
    void constrain(Tensor& logits, Tensor* argmax);
    // After the round synchronized: the draws of `tokens`, chosen by consecutive columns from
    // `first_column` for the lane's next output steps (none for an unconstrained lane).
    void take_constraint_draws(Lane& lane, std::size_t first_column, std::span<const std::int32_t> tokens);
    // Moves the first `accepted` draws of the pending round into the lane's trace.
    static void commit_constraint_draws(Lane& lane, std::size_t accepted);
    // The readout of the lane's prompt from its last prefill call's logits (column 0).
    [[nodiscard]] PromptReadout read_prompt_frontier(const Lane& lane);
    // ---- Constrained decoding (GBNF, JSON, JSON Schema, choice, regex, tool calls) ----
    // The Engine's grammar matchers fill packed vocabulary masks for a row's positions 0..drafts
    // (TokenMaskProvider); the sampler and draft acceptance read them through the row's sampling
    // config. Sampling and acceptance run on the host-launched path after each forward, where every
    // row's drafts are known, so no graph split is needed.
    static std::size_t grammar_mask_bytes(std::int32_t token_domain, std::int32_t lanes, std::int32_t width) noexcept {
        return 4ULL * ((static_cast<std::size_t>(token_domain) + 31) / 32) * static_cast<std::size_t>(width) *
               static_cast<std::size_t>(lanes);
    }
    // Row `row`'s masks for positions 0..drafts.size(), uploaded; an empty mask when the row is
    // unconstrained. Records which positions are dead ends (no legal token).
    ops::SamplingMask grammar_mask(std::size_t row, std::span<const TokenId> drafts) {
        grammar_dead_[row] = 0;
        if (grammar_ == nullptr || !grammar_->constrained(row)) { return {}; }
        if (drafts.size() + 1 > static_cast<std::size_t>(max_width_)) {
            throw std::logic_error("Qwen4Exp: a constrained row verifies more positions than its mask holds");
        }
        const std::size_t words  = (static_cast<std::size_t>(token_domain_) + 31) / 32;
        const std::size_t offset = row * static_cast<std::size_t>(max_width_) * words;
        std::span<std::uint32_t> host(static_cast<std::uint32_t*>(grammar_host_.data()) + offset,
                                      (drafts.size() + 1) * words);
        grammar_dead_[row] = grammar_->fill(row, drafts, host);
        auto* device_words = static_cast<std::uint32_t*>(head().grammar_masks.p) + offset;
        CUDA_CHECK(cudaMemcpyAsync(device_words, host.data(), host.size_bytes(), cudaMemcpyHostToDevice,
                                   head().device->stream));
        grammar_->uploaded(row, host.size_bytes());
        return {device_words, static_cast<std::int32_t>(words)};
    }
    // A row fails when the tokens it emits reach a dead-end position (one mask position per token).
    void mark_grammar_failures(PendingBatch& pending, std::span<const std::int32_t> counts) const noexcept {
        for (std::size_t row = 0; row < pending.row_count(); ++row) {
            const auto count = counts.empty() ? 1 : counts[row];
            const std::uint64_t reached = count >= 64 ? ~std::uint64_t{0} : (std::uint64_t{1} << count) - 1U;
            ContractAccess::set_constraint_failed(pending, row, (grammar_dead_[row] & reached) != 0);
        }
    }
    // Set for the duration of one decode or prefill call.
    struct GrammarScope {
        ProgramImpl& impl;
        GrammarScope(ProgramImpl& owner, runtime::TokenMaskProvider* masks) : impl(owner) { impl.grammar_ = masks; }
        ~GrammarScope() { impl.grammar_ = nullptr; }
    };
    PinnedHostBuffer grammar_host_{1};
    runtime::TokenMaskProvider* grammar_ = nullptr;
    std::array<std::uint64_t, kMaximumConcurrency> grammar_dead_{};

    std::vector<std::int32_t> constraint_host_;
    std::int32_t constraint_columns_ = 0; // the staged round's columns
    std::array<std::uint64_t, kMaximumConcurrency> constraint_serials_{};
    std::uint64_t next_constraint_serial_ = 1;
    PinnedHostBuffer io_host_{1}, host_sampled_{1}, host_sample_pos_{1}, host_configs_{1};
    IoLayout io_layout_;
    // SSD tier mode (design §19.3.7): the fetch channel and the tier outlive everything that reads
    // their memory (the residency's promotion copies, the stream, the CPU service), declared later.
    std::unique_ptr<ops::offloaded_moe::FetchChannel> fetch_channel_;
    std::unique_ptr<HostExpertTier> tier_;
    PinnedHostBuffer expert_error_{1}; // mapped u32: ops::MoeExpertSource::error of every call
    bool clear_prefix_on_release_ = false;

    // ---------------------------------------------------------------- layer walk (prefill_walk.cpp)
    // Design §19.3.8 F4: a span of consecutive streamed calls of one prompt runs layer-major
    // (prefill_walk.h holds its state).
    WalkState walk_;
    void walk_prefetch_rows(Lane& lane, std::uint32_t index, std::size_t chunks);
    [[nodiscard]] bool walk_prefetched(std::uint32_t index, std::size_t first_call, std::size_t c, std::int32_t begin,
                                       std::int32_t width);
    [[nodiscard]] std::uint64_t window_key() const noexcept;
    static constexpr std::uint32_t kWalkMaxTokens = 65536;

    [[nodiscard]] std::size_t walk_bytes(std::uint32_t tokens, std::size_t chunks) const noexcept;
    std::size_t walk_span_end(const Lane& lane, std::uint32_t index);
    void walk_begin(Lane& lane, std::uint32_t index, std::size_t end);
    PrefillProgress walk_step(Lane& lane, std::uint32_t index, Clock::time_point start);
    // The segment that runs decoder layer `layer` (the whole model's without a split).
    [[nodiscard]] DeviceSegment& layer_segment(std::uint32_t layer);
    // Chunk c's staged io image in `seg`'s walk area, after the span's residual.
    [[nodiscard]] std::byte* walk_slot(const DeviceSegment& seg, std::size_t chunk) const noexcept;
    void walk_upload(DeviceSegment& seg, std::size_t chunk, const std::byte* host_slot);
    void walk_pass(std::size_t chunk, std::uint32_t layer);
    // The MoE offload (moe_offload_; multi-GPU WP23): its resources at construction, the walk
    // area's part for the results on the first segment, and layers of the first segment over the
    // span's chunks with their routed experts on the second.
    void prepare_moe_offload();
    [[nodiscard]] std::size_t offload_bytes(std::size_t chunks) const noexcept;
    [[nodiscard]] std::size_t offload_input_bytes(std::size_t columns) const noexcept;
    void walk_offload_layers(std::uint32_t from, std::uint32_t to, std::int32_t widest);
    // A split's crossing after its first segment's part of the span with the MoE offload: every
    // chunk's residual into the next segment's walk residual, whose stream starts there.
    void walk_offload_cross();
    // An eager walk batch over chunk c's io image in `seg`'s walk area.
    [[nodiscard]] execution::ForwardBatch walk_batch(const DeviceSegment& seg, std::size_t chunk);
    void walk_enqueue(std::uint32_t to_layer);
    void walk_report(Clock::time_point began) noexcept;
    void walk_abandon() noexcept;
    // `exact_only`: only an exact tap (a planned call end) is due; flexible taps wait for a later boundary.
    [[nodiscard]] bool prefix_tap_due(const Lane& lane, std::size_t next_call, bool exact_only = false) const;
    // Whether a layer walk's span ends at calls[next_call - 1] for a tap (prefix_tap_due; a split in
    // SplitWalk::ExactTaps ends its spans at exact taps only).
    [[nodiscard]] bool walk_tap_due(const Lane& lane, std::size_t next_call) const;
    std::unique_ptr<RouteTrace> trace_;
    std::unique_ptr<ops::offloaded_moe::CpuMissService> cpu_service_;

    // ---- prefill CPU split (design §19.3.12) ----
    // Gives the CPU the thinnest non-resident experts of a layer while the link carries the rest:
    // narrow ones (n <= 8) in order of n, then expert id, as many as minimise the later of the
    // CPU's predicted time and the link's time for the experts left to it.
    class SplitPolicy final : public execution::CpuSplitPolicy {
    public:
        SplitPolicy(const ProgramImpl& program, const DeviceSegment& segment)
            : program_(program), segment_(segment) {}
        std::uint32_t choose(std::uint32_t layer, std::span<const std::int32_t> columns,
                             std::span<std::uint8_t> cpu) const override {
            const auto& p              = program_;
            const std::uint32_t E      = p.c_.moe.experts;
            const std::int32_t* frames = segment_.residency->host_table() + static_cast<std::size_t>(layer) * E;
            const double link =
                static_cast<double>(segment_.residency->frame_stride()) / segment_.policy.bytes_per_second;
            const double per_expert    = 1.0 / p.options_.cpu_split_expert_rate;
            const double per_column    = 1.0 / p.options_.cpu_split_column_rate;
            constexpr int kNarrow      = ops::offloaded_moe::kMaxCpuColumns;
            std::array<std::uint32_t, kNarrow + 1> width_count{};
            std::uint32_t absent = 0;
            for (std::uint32_t e = 0; e < E; ++e) {
                if (columns[e] <= 0 || frames[e] >= 0) { continue; }
                ++absent;
                if (columns[e] <= kNarrow) { ++width_count[static_cast<std::size_t>(columns[e])]; }
            }
            // The best count: the CPU's time grows and the link's shrinks with each expert moved.
            double best_time = absent * link, cpu_time = 0.0;
            std::uint32_t best = 0, k = 0;
            for (int n = 1; n <= kNarrow; ++n) {
                for (std::uint32_t i = 0; i < width_count[static_cast<std::size_t>(n)]; ++i) {
                    if (k == static_cast<std::uint32_t>(ops::offloaded_moe::kMaxCpuJobs)) { break; }
                    cpu_time += std::max(per_expert, n * per_column);
                    ++k;
                    const double time = std::max(cpu_time, (absent - k) * link);
                    if (time < best_time) {
                        best_time = time;
                        best      = k;
                    }
                }
            }
            // Mark the first `best` in that order.
            std::uint32_t marked = 0;
            for (int n = 1; n <= kNarrow && marked < best; ++n) {
                for (std::uint32_t e = 0; e < E && marked < best; ++e) {
                    if (columns[e] == n && frames[e] < 0) {
                        cpu[e] = 1;
                        ++marked;
                    }
                }
            }
            return marked;
        }

    private:
        const ProgramImpl& program_;
        const DeviceSegment& segment_;
    };

    // A layer split's boundaries, one per pair of consecutive segments (DeviceSegment::in, out and
    // bulk_out point at them): declared before the segments, whose Forwards and streams use them.
    std::vector<std::unique_ptr<HandoffChannel>> channels_;
    std::vector<std::unique_ptr<BulkHandoff>> bulk_;
    // The MoE offload's crossings (DeviceSegment::moe_out, moe_back), both on the first segment.
    std::vector<std::unique_ptr<BulkHandoff>> moe_bulk_;
    // The device segments in layer order (device_segment.h): one covering every layer, or a layer
    // split's. Declared after the host objects their expert caches, Forwards and VRAM monitors read
    // (the SSD tier and its fetch channel, the expert error word, the CPU expert service, the route
    // trace, the boundaries, the VRAM mutex, the waker and the KV idle clock), and before the lanes
    // and the prefix cache, whose page leases return to a segment's pool.
    std::vector<std::unique_ptr<DeviceSegment>> segments_;
    // Each segment's frame sizing at construction (DeviceSegment::sizing), contiguous in segment order.
    std::vector<VramSizing> segment_sizing_;
    std::size_t ple_segment_ = 0; // the segment holding the PLE layer (SegmentMap::ple)

    std::array<Lane, kMaximumConcurrency> lanes_{};
    std::array<std::int32_t, kMaximumConcurrency> host_lanes_{};
    std::vector<TokenId> pending_tokens_ = std::vector<TokenId>(kMaximumConcurrency, 0);
    std::array<std::int32_t, kMaximumConcurrency> pending_counts_{};

    // MTP drafter.
    bool mtp_            = false;
    std::int32_t mtp_k_  = 0;
    std::int32_t mtp_columns_ = 0;
    PinnedHostBuffer mtp_host_{1};
    MtpIo mtp_io_;
    std::array<std::vector<std::int32_t>, kMaximumConcurrency> mtp_drafts_;
    std::array<std::int32_t, kMaximumConcurrency> mtp_confident_{}; // per row: drafts above kDraftMinLogprob
    std::array<bool, kMaximumConcurrency> from_ngram_{};
    // Draft-length policy: acceptance prior and EWMA weight, and the cost of each draft column in
    // plain rounds (warm tg512 on the 5090: W = 3, 4, 5 rounds took 1.84, 2.13 and 2.54 plain
    // rounds, design 19.2).
    static constexpr double kAcceptancePrior      = 0.75;
    static constexpr double kAcceptanceWeight     = 0.1;
    static constexpr double kWidthCost            = 0.38;
    static constexpr std::uint64_t kProbeInterval = 8;
    // kWidthCost, or the measurement knob INFERNIX_WIDTH_COST=<x> (a draft column's cost in plain
    // rounds, in [0, 4]): the draft lengths it picks change which rounds verify how many columns,
    // never a token (one row verifies at most 8 columns, on the column-invariant routes).
    static double width_cost_from_environment() {
        const char* value = std::getenv("INFERNIX_WIDTH_COST");
        if (value == nullptr || *value == '\0') { return kWidthCost; }
        char* end         = nullptr;
        const double cost = std::strtod(value, &end);
        if (end == value || *end != '\0' || !(cost >= 0.0 && cost <= 4.0)) {
            throw std::invalid_argument(std::string("INFERNIX_WIDTH_COST must be a number in [0, 4], not '") + value +
                                        "'");
        }
        return cost;
    }
    const double width_cost_ = width_cost_from_environment();
    // Two-row rounds: at most 2 drafts per row, each costing kPairWidthCost plain two-row rounds
    // (C = 2 decode on the 5090, AIME prompts sampled: K = 1 and 2 rounds took 1.44 and 1.95 plain
    // two-row rounds; K = 3, 8 columns, took 3.1 and lost to plain decode; design 19.3.5).
    static constexpr std::int32_t kPairMaxDrafts = 2;
    static constexpr double kPairWidthCost       = 0.47;
    // A one-row round verifies its drafts up to the first whose draft-head probability is below 0.5
    // (Strata's --spec-min-p rule; C = 1 MTP, AIME prompts sampled on the 5090: 138.8 -> 147.2 tok/s,
    // acceptance 75.6 -> 81.8 %, repeatable to the decimal; 0.7 gave 142.6; design 19.3.5).
    static constexpr double kDraftMinLogprob = -0.6931471805599453; // log(0.5)
    std::uint64_t pair_policy_rounds_            = 0;
    bool plain_round_ = false;

    // Speculative verification (max_width_ > 1).
    std::int32_t max_width_   = 1;
    std::int32_t round_width_ = 1;
    std::size_t proposer_tokens_ = 0;
    PinnedHostBuffer spec_host_{1};
    SpecLayout spec_layout_;
    std::array<std::vector<std::int32_t>, kMaximumConcurrency> drafts_;
    // The n-gram gate of verification rounds (design §12.3): the pinned ready word and its last
    // published value (strictly increasing, never 0); the PLE segment keeps the per-lane device
    // wait statistics.
    PinnedHostBuffer gate_host_{64};
    std::uint32_t gate_sequence_ = 0;
    // A layer split's rounds: how the residual crosses each boundary (ProgramOptions::handoff), the
    // last round word staged (next_handoff_word) and INFERNIX_SPLIT_SYNC (before_restage).
    HandoffMode handoff_            = HandoffMode::Event;
    std::uint32_t handoff_sequence_ = 0;
    bool split_sync_                = false;
    SplitWalk split_walk_           = SplitWalk::EveryTap; // INFERNIX_SPLIT_WALK (a split only)
    bool moe_offload_               = false; // INFERNIX_SPLIT_MOE_OFFLOAD (prepare_moe_offload)
    std::vector<std::uint8_t> live_;
    std::vector<std::int32_t> window_;
    std::vector<std::uint32_t> row_ids_;
    std::optional<std::uint32_t> transaction_lane_;
    std::optional<ResumeState> pause_; // the open Pause transaction's result
    bool replaying_ = false;           // the open binding replays toward a paused frontier
    std::uint64_t next_epoch_          = 0;
    std::uint64_t next_transaction_    = 0;
    std::uint64_t pending_transaction_ = 0;
    std::uint64_t revision_            = 1;
    std::uint64_t prefill_steps_       = 0; // prefill steps of every lane (a waiting Vision lend's progress)
    // ---------------------------------------------------------------- vision (vision_program.cpp)
    // The segment that runs the Vision tower (vision_tower_segment; INFERNIX_SPLIT_VISION), and on a
    // layer split, per segment index, the pinned crossing of the handoff from the tower's GPU into a
    // reading segment on another GPU (null elsewhere). Set by place_vision at construction.
    std::size_t vision_segment_ = 0;
    std::vector<std::unique_ptr<BulkHandoff>> vision_bulk_;
    DeviceSegment& vision_seg() noexcept { return *segments_[vision_segment_]; }
    const DeviceSegment& vision_seg() const noexcept { return *segments_[vision_segment_]; }
    void place_vision();
    // Whether the caches can ever lend a window's frames: the tower's its handoff and window, each
    // reading segment's on another GPU its copy of the handoff (a quarter of each cache is never lent).
    [[nodiscard]] bool vision_fits(const VisionWindowPlan& window) const noexcept;
    // The encode window of the items ending past the reused prefix [0, reused).
    [[nodiscard]] std::shared_ptr<const VisionAdmission> plan_vision(const qwen3_5::PreparedPromptData& prompt,
                                                                     std::uint32_t reused = 0) const;
    void vision_reserve(Lane& lane, std::shared_ptr<const VisionAdmission> admission,
                        const qwen3_5::PreparedPromptData& prompt);
    [[nodiscard]] static bool vision_encoded(const Lane& lane) noexcept { return !lane.vision || lane.vision->encoded; }
    // One step of the lane's encode; false when it made none (vision_lend waits).
    [[nodiscard]] bool vision_step(Lane& lane, std::uint32_t index);
    // Before the encode: lends the tower's handoff (and window) and each reading segment's copy on
    // another GPU, each at the bottom of its pool; false, lending nothing, while that would split a
    // pool that another prefilling lane's leases will free.
    [[nodiscard]] bool vision_lend(LaneVision& v, std::uint32_t index);
    // After the encode: the handoff's copy into each reading segment on another GPU.
    void vision_mirror(LaneVision& v);
    // The handoff memory `seg` reads: its copy, or the tower's handoff on the tower's GPU.
    [[nodiscard]] std::byte* vision_memory(const LaneVision& v, const DeviceSegment& seg) const;
    void vision_release(Lane& lane) noexcept;
    // The image columns of a prefill call [begin, begin + width), staged into the io; null without any.
    const execution::VisionInput* stage_vision(Lane& lane, std::uint32_t begin, std::int32_t width);
    // Per MTP sub-chunk: the cells whose next token is an image token; null without media.
    const execution::VisionInput* stage_mtp_vision(Lane& lane, std::uint32_t first, std::int32_t cells,
                                                   std::int32_t subchunks);
    // The call inputs that point into the io.
    execution::VisionInput vision_input_;
    std::vector<execution::VisionInput> mtp_vision_inputs_;

    // The prefix cache, declared last so it is destroyed first: it returns its page leases to the
    // head's pool and its lane image reads every segment's state and the lane buffers.
    prefix::PrefixState prefix_;
};

} // namespace detail
} // namespace infernix::models::qwen4_exp
