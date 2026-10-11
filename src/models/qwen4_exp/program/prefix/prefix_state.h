#pragma once

// The Program's prefix-cache state (prefix_program.cpp; docs/maintainer/qwen3_8-flash-next-design.md
// §19.3.1): the lanes' state image and its layout, the cache binding, each segment's compute stream
// (its copies' producer and consumer), the capture ids, the per-layer waits the next forward call
// enqueues, and the staging of the MR3 flush cell. The Program declares it after its device
// segments, so it is destroyed first: the cache returns its page leases to the head's pool and the
// lane image reads every segment's state.

#include "core/arena.h"
#include "models/qwen4_exp/program/prefix/prefix_cache.h"
#include "models/qwen4_exp/program/prefix/state_image.h"

#include <cuda_runtime_api.h>

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace infernix::models::qwen4_exp::prefix {

struct PrefixState {
    StateImageLayout image_layout;
    std::unique_ptr<LaneStateImage> lane_image;
    std::unique_ptr<PrefixCache> cache; // null without --prefix-cache
    // Each segment's compute stream, in segment order: a capture, a write-through and a restore
    // start after its queued work, and a slot's reuse waits on it for the copy-outs of the slot.
    std::vector<cudaStream_t> compute;
    std::uint64_t next_capture = 0;
    // The next forward call's per-layer waits: a resumed lane's restore and a copy-out of its state
    // still in flight (design §19.3.1). Every segment's part of that call reads them; the head's
    // part, the last one enqueued, consumes them.
    std::array<std::span<const cudaEvent_t>, 2> next_waits{};
    // The MR3 flush cell's inputs (token, cell, slot, table row, RoPE position, pooled-block start),
    // on the head's device.
    PinnedHostBuffer flush_host{1};
    DeviceBuffer flush_device;
    cudaEvent_t flush_uploaded = nullptr;

    // The waits the next forward call of segment `segment` enqueues on its stream, indexed by the
    // model's decoder layer (then the MTP block): entry l is recorded on the restore or transfer
    // stream of the segment that runs layer l, and a segment's Forward waits only on its own
    // layers' entries (Forward::wait_layer), so no wait crosses devices.
    [[nodiscard]] std::array<std::span<const cudaEvent_t>, 2> waits_for(std::uint32_t segment) const noexcept {
        static_cast<void>(segment);
        return next_waits;
    }
};

} // namespace infernix::models::qwen4_exp::prefix
