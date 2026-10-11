#pragma once

// The state of Qwen4Exp's prefill layer walk (prefill_walk.cpp; design §19.3.8 F4 in
// docs/maintainer/qwen3_8-flash-next-design.md): a span of consecutive streamed calls of one prompt
// runs layer-major (every decoder layer over all of the span's chunks, then the next), so each
// layer's non-resident experts are streamed once per span instead of once per call. The chunk grid,
// and with it every output bit, is the chunk-major one. The span's residual stream [S*H, tokens] and
// each chunk's staged io image live in the stream lease after the ring: on a layer split in every
// segment's lease, at the same offsets (multi-GPU WP16).

#include "core/arena.h"
#include "models/qwen4_exp/execution/forward.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>
#include <vector>

namespace infernix::models::qwen4_exp::detail {

struct WalkState {
    // The running span.
    struct LayerWalk {
        bool active = false;
        std::uint32_t lane = 0;
        std::size_t first_call = 0, end_call = 0;
        std::vector<std::int32_t> begins, widths, offsets; // per chunk: position, width, residual column
        std::uint32_t next_layer      = 0;                // the next decoder layer to enqueue
        std::uint32_t layers_per_step = 1;
        std::size_t io_stride = 0, residual_bytes = 0;
        std::int32_t host_slot = 0; // the lane's state slot as the Forward reads it on the host
        std::vector<std::optional<execution::MtpChunk>> mtp;
        std::vector<std::optional<execution::VisionInput>> vision;
        std::vector<std::vector<execution::VisionInput>> mtp_vision;
        std::uint64_t steps = 0;
        std::chrono::steady_clock::time_point began{}; // walk_begin (the span's debug report)
        // The MoE offload (walk_offload_layers): fronts whose backs are not enqueued yet, oldest
        // first (layer, chunk), carried from one step to the next, and the fronts enqueued so far
        // (the staging slots' rotation).
        std::deque<std::pair<std::uint32_t, std::size_t>> pending;
        std::uint64_t fronts = 0;
    } span;
    PinnedHostBuffer host{1}; // every chunk's io image, uploaded once per span
    // The next span's n-gram rows, read into host's chunk slots while the current span's steps run
    // on the device: chunk k is call first_call + k, its rows stand at slot k's n-gram columns, and
    // key is a hash of the tokens its rows hash.
    struct WalkPrefetch {
        bool valid = false;
        std::uint32_t lane = 0;
        std::size_t first_call = 0;
        std::vector<std::int32_t> begins, widths;
        std::vector<std::uint64_t> keys;
        bool done = false; // no further chunk can join the next span
    } prefetch;
    std::uint64_t spans_run = 0, calls_run = 0;
    std::uint64_t prefetched_chunks = 0; // chunks whose rows were prefetched
};

} // namespace infernix::models::qwen4_exp::detail
