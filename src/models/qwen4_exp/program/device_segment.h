#pragma once

// One device's part of a Qwen4Exp Program: the decoder layers of a SegmentSpec and the single
// components the spec assigns it (the embedding, the PLE and its n-gram gate, the head with sampling,
// acceptance and the MTP drafter, the KV page allocator), with everything the Program allocates and
// runs on that device: the plan of its fixed allocations, its VRAM budget and control law, recurrent
// state and KV pages, the per-call io, the head's buffers, the workspace, the expert cache with its
// stream and staging, the link's rate and policy, the verification records, the drafter's buffers, the
// Forward and its CUDA graphs. Host state (the pinned io image, the lanes, sampling and draft policy,
// the n-gram volume, the CPU expert service, the SSD tier, the route trace and the prefix cache)
// stays with the Program.
//
// An unsplit Program has one segment covering every layer (SegmentMap::single).

#include "models/qwen4_exp/program/program.h"

#include "core/arena.h"
#include "core/decode_graph.h"
#include "core/device.h"
#include "core/gdn_replay_records.h"
#include "core/handoff.h"
#include "core/linear_attention_state.h"
#include "core/paged_kv_cache.h"
#include "core/vmm_arena.h"
#include "core/vram_budget.h"
#include "models/qwen4_exp/execution/expert_stream.h"
#include "models/qwen4_exp/execution/forward.h"
#include "models/qwen4_exp/execution/parameters.h"
#include "models/qwen4_exp/memory_plan.h"
#include "models/qwen4_exp/program/expert_residency.h"
#include "models/qwen4_exp/program/link_policy.h"
#include "models/qwen4_exp/program/prefix/state_image.h"
#include "models/qwen4_exp/program/vram_monitor.h"
#include "models/qwen4_exp/segment_map.h"
#include "infernix/ops/gdn_replay.h"

#include <cuda_runtime.h>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace infernix::models::qwen4_exp::detail {

// Per-call inputs staged through pinned memory, one device copy. Decode and verification rounds
// upload only the prefix before `ngram` plus their rows (io_prefix), so every array they read,
// the RoPE positions and pooled-block starts included, precedes `ngram`. `rope` holds a call's
// [columns, 3] axis-major RoPE positions, `block_rope` its [batch, 3] block starts; `mtp_rope` and
// `mtp_block_rope` the MTP chunk's per sub-chunk (execution::MtpChunk). `gate` is the U32 a gated
// verification round's rows wait for (design §12.3): it precedes `ngram`, so the round uploads it
// with the prefix and its rows go through the gate instead. `handoff` is the U32 a layer split's
// gated handoff publishes and waits for (ForwardBatch::handoff_word), in the gate's 256-byte line,
// so the decode prefix carries it and the layout after it is the unsplit one.
//
// Every segment's io has this one layout. Only the PLE reads the n-gram rows [ngram, mtp_ids), so
// they go to the PLE segment (SegmentSpec::ple) alone, which is segment 1, not 0, when the split is
// at layer 1: the upload rule follows the role, not the index. The other segments upload
// [0, ngram) and [mtp_ids, bytes). Decode and verification rounds upload [0, ngram) to every
// segment; their rows reach the PLE segment through the n-gram gate.
struct IoLayout {
    std::size_t ids = 0, positions = 0, rope = 0, block_rope = 0, slots = 0, rows = 0, columns = 0, gate = 0,
                handoff = 0, ngram = 0, mtp_ids = 0, mtp_cells = 0, mtp_rope = 0, mtp_block_rope = 0,
                vision_columns = 0, mtp_vision_columns = 0, bytes = 0;
};

// Device I32 arrays of a verification round, each with room for every lane and width.
struct SpecLayout {
    std::size_t target = 0, drafts = 0, extents = 0, lengths = 0, anchors = 0, licensed = 0, counts = 0,
                accepted = 0, commit = 0, words = 0;
};

// I32 MTP drafter io: chain ids [B], cells [K, B], per draft step the cells' RoPE positions and
// pooled-block starts [K][3][B] (axis-major per step), drafts [K, B], catch-up ids [W, B], gather
// columns [B], catch-up cells [W, B]. Everything before `drafts` is uploaded per draft round.
struct MtpIo {
    std::size_t ids = 0, cells = 0, rope_cells = 0, block_rope_cells = 0, drafts = 0, up_ids = 0, gather = 0,
                up_cells = 0, logprobs = 0, words = 0;
};

// Every fixed device allocation of a Program's segment and the layouts in it (design §19.3.7). One
// function (ProgramImpl::plan_segment) computes it from the options, the configuration and the
// segment's spec alone: before the weights are read, for the startup check, and in the constructor,
// which allocates exactly it.
struct DeviceLayout {
    std::int32_t lanes = 0, token_domain = 0, head_rows = 0, chunk = 0, mtp_k = 0, max_width = 1, columns = 0,
                 pages_per_row = 0, mtp_columns = 0;
    // The segment's roles (SegmentSpec): the head's buffers (logits, sampling, constraints, the
    // drafter when `mtp`) and the PLE segment's gate statistics are planned only where they run.
    bool owns_head = true, owns_ple = true;
    bool mtp = false;
    std::uint32_t kv_pages = 0, kv_layers = 0;
    std::uint64_t record_stride = 0;
    LinearAttentionStatePoolLayout gdn;
    DeviceKVPagePoolLayout pool;
    KVExecutionTableLayout tables;
    GdnReplayRecordLayout records;
    IoLayout io;
    SpecLayout spec;
    MtpIo mtp_io;
    std::size_t gdn_bytes = 0, ple = 0, tails = 0, kv = 0, records_bytes = 0, ple_records = 0, qsa_records = 0;
    // The vq2/k4v2 exact window of every KV layer and lane (prefix::KvWindowGeometry; zero otherwise).
    std::size_t window = 0;
    std::size_t logits32 = 0, logits16 = 0, token_counts = 0, work = 0, staging = 0;
    // The arena wide prefill calls swap in (columns above static_columns), lent from expert frames
    // while a prompt runs such calls; zero when the static arena covers every call.
    std::size_t wide_work = 0;
    std::int32_t static_columns = 0;
    // Elastic KV (design §19.3.11): the backing reserves `kv` bytes of address space and maps the
    // chunks pages [0, n) of every plane (and the block tables) need; `kv_base_pages` are mapped at
    // startup and counted as fixed, the rest grows from the expert frames when admissions need it.
    bool kv_elastic             = false;
    std::uint32_t kv_base_pages = 0;
    std::size_t mtp_column_bytes = 0, mtp_ones = 0;
    ProgramDevicePlan bytes;
};

// A fixed-address round's CUDA graph: run eagerly the first time (the Ops' one-time setup), captured
// the second, replayed after.
struct DecodeGraph {
    DecodeGraphDefinition definition;
    DecodeGraphExecutable executable;
    bool warmed = false;
};

// A layer split's boundary event of a segment: recorded on its compute stream once its share of an
// eager call or of a round is enqueued, so a segment on the same device waits for it (two
// segments' kernels never overlap on one GPU: the MoE's results depend on that, design §19.3.8
// "F1 nondeterminism"; ProgramImpl::SegmentWork).
struct SegmentEvent {
    SegmentEvent() = default;
    SegmentEvent(const SegmentEvent&)            = delete;
    SegmentEvent& operator=(const SegmentEvent&) = delete;
    ~SegmentEvent() {
        if (event != nullptr) { cudaEventDestroy(event); }
    }
    cudaEvent_t event = nullptr;
};

// Prefill staging overlap (see ForwardExperts); destroyed after every call has completed.
struct OverlapResources {
    ~OverlapResources() {
        for (auto event : events) {
            if (event != nullptr) { cudaEventDestroy(event); }
        }
        if (stream != nullptr) { cudaStreamDestroy(stream); }
    }
    cudaStream_t stream = nullptr;
    std::array<cudaEvent_t, 5> events{};
};

// Members are declared in the order their destruction needs: what an object's destructor reaches
// (a pool its leases return to, the memory a stream still copies into) is declared before it, and
// the VRAM monitor, whose thread reads the source, the control law and the residency, comes last.
// The Program's members the monitor's callback reads (its mutex, the engine waker, the KV idle
// clock) and the host objects the expert cache and the Forward read (the SSD tier, the CPU expert
// service, the expert error word) are declared before ProgramImpl::segments_, so they outlive it.
struct DeviceSegment {
    SegmentSpec spec;
    DeviceContext* device                   = nullptr;
    const execution::Parameters* parameters = nullptr;
    DeviceLayout plan;

    // ---- VRAM (design §19.3.7): the device's budget source, the frames' startup sizing, and the
    // control law of an elastic pool with the frame count and pool bytes it last applied (the monitor
    // thread reads them under ProgramImpl::vram_mutex_).
    std::unique_ptr<VramBudgetSource> vram;
    VramSizing sizing;
    std::optional<VramControl> control;
    std::uint32_t frames_now = 0, startup_frames = 0;
    std::uint64_t pool_now  = 0;
    bool low_frames_warned  = false;
    bool pressure_warned    = false;
    std::chrono::steady_clock::time_point vram_epoch{};

    // ---- recurrent state and KV (design §19.3.11 for the elastic backing) ----
    DeviceBuffer state_backing, ple_backing, tails_backing, kv_backing, staging;
    // The vq2/k4v2 exact window (design §19.3.15), every KV layer and lane.
    DeviceBuffer window_backing;
    prefix::KvWindowGeometry window_geometry;
    std::unique_ptr<VmmRange> kv_vmm;
    DeviceSpan kv_span{};

    // ---- the host-to-device link (design §19.4) ----
    // The link's rate and the choices that follow it (derive_link_policy), set by
    // ProgramImpl::measure_link. The reference link's policy until then, and when nothing could be
    // probed (link_measured false). The Program applies the PCIe divisor, the promotion interval and
    // the CPU split's widest call; a layer split's segments also their promotion policy (a trickle
    // on a slow link), seed share and CPU jobs (ProgramImpl::promotes_in_rounds, trickles).
    LinkPolicy policy;
    bool link_measured = false;
    // A trickle's promotions per round (ProgramImpl::trickle_promotions); 0 for other policies.
    std::uint32_t trickle = 0;
    // The CPU expert service's port that serves this segment's layers: 0 for an unsplit Program.
    int cpu_port = 0;
    // Tokens toward the next steady-state decode promotion (ProgramImpl::decode_budget).
    std::uint64_t budget_tokens = 0;

    // ---- per-call inputs, and the head's logits, sampling, penalty counts, n-gram gate statistics,
    // constraint controls and grammar masks ----
    DeviceBuffer io_device, logits32, logits16, sampled, sample_pos, configs, token_counts;
    DeviceBuffer grammar_masks;
    DeviceBuffer constraint;

    std::unique_ptr<LinearAttentionStatePool> gdn;
    prefix::KvPageGeometry kv_geometry;
    std::unique_ptr<DeviceKVPagePool> pool;
    std::unique_ptr<KVExecutionTablePool> tables;
    std::size_t work_capacity = 0;
    std::unique_ptr<WorkspaceArena> work;

    // ---- the expert cache, its prefill stream and lease (design §19.3.8 F2, F3) and the wide
    // workspace (design §19.3.7, VRAM item 1) ----
    std::unique_ptr<ExpertResidency> residency;
    std::unique_ptr<execution::ExpertStream> expert_stream;
    ExpertResidency::FrameLease stream_lease;
    // The stream lease: the ring, the wide arena, then a walk area.
    std::size_t stream_ring_bytes = 0, stream_wide_bytes = 0, stream_walk_bytes = 0;
    std::size_t wide_work = 0;               // the plan's wide arena bytes (0: none)
    ExpertResidency::FrameLease work_lease;  // the wide arena lent alone (no stream lease)
    DeviceBuffer wide_fallback;              // the wide arena when nothing could be lent
    // A split's segment of a few layers, whose frames could never lend the wide arena: wide_fallback
    // is its arena for good, allocated at construction.
    bool wide_dedicated = false;
    std::unique_ptr<WorkspaceArena> wide_arena; // the arena swapped with *work around a wide call
    bool wide_active = false;                // *work is the wide arena (enter_wide .. leave_wide)
    // The stream's, the CPU split's and the walk's prefetch counters when the lease was taken.
    execution::ExpertStream::Stats stream_at_lease;
    execution::Forward::SplitStats split_at_lease;
    std::uint64_t prefetched_at_lease = 0;

    // ---- prefill CPU split (design §19.3.12) ----
    std::unique_ptr<execution::CpuSplitPolicy> split_policy;
    bool split_call = false;
    OverlapResources overlap;

    // ---- a layer split's boundaries (core/handoff.h; the Program owns them, none without a split):
    // the channels into and out of this segment (decode and verification rounds), the bulk crossing
    // out of it (eager calls), the event its share of a call or round records (SegmentEvent), and
    // for a receiving segment the device U64 [2] its gated handoff waits add to (wait ns, waits) ----
    HandoffChannel* in    = nullptr;
    HandoffChannel* out   = nullptr;
    BulkHandoff* bulk_out = nullptr;
    SegmentEvent compute_done;
    DeviceBuffer handoff_stats;
    // INFERNIX_SPLIT_TIMING: timing events around its share of a round (ProgramImpl::SplitTiming).
    SegmentEvent timing_begin, timing_end;

    // ---- the Forward and its CUDA graph families ----
    std::unique_ptr<execution::Forward> forward;
    std::array<DecodeGraph, kMaximumConcurrency> graphs;
    std::vector<DecodeGraph> mtp_draft_graphs; // (batch - 1) * mtp_k + steps - 1
    std::vector<DecodeGraph> mtp_catch_graphs; // (batch - 1) * max_width + width - 1
    std::vector<DecodeGraph> verify_graphs;    // (batch - 1) * max_width + width - 1

    // ---- the MTP drafter (design 11.2): residual columns, saved and chain columns, its QSA records,
    // the HC ones and its io ----
    DeviceBuffer mtp_residuals, mtp_saved, mtp_chain, mtp_records, mtp_ones, mtp_device;

    // ---- speculative verification: GDN replay, PLE and QSA records, the round's I32 arrays ----
    DeviceBuffer records_backing, ple_records, qsa_records, spec_device;
    GdnReplayRecords records;
    std::vector<std::optional<ops::GdnReplayFoldPlan>> folds;
    // Per-lane device wait statistics of the n-gram gate (design §12.3).
    DeviceBuffer gate_stats;

    // Last: its thread reads the source, the control law and the residency, so it stops first.
    std::unique_ptr<VramMonitor> monitor;
};

} // namespace infernix::models::qwen4_exp::detail
