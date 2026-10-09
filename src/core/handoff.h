#pragma once

// Device-to-device handoffs without peer access, through pinned host memory: the transfer
// primitives of a layer split, whose segments run on different GPUs of one process and hand the
// hyper-connection residual from one segment to the next.
//
// HandoffChannel carries one bounded payload per round (decode and verify crossings) through one
// pinned slot, in one of two modes:
//   - Event: the sender's stream copies the payload into the slot with its copy engine
//     (cudaMemcpyAsync D2H), the host records `sent` on the sender's stream after the round's
//     launch and makes the receiver's stream wait for it (cross-device cudaStreamWaitEvent); the
//     receiver's stream then reads the slot with its SMs (upload_pinned). The host never blocks.
//   - Gate: the sender's stream stores the payload and publishes the round's word
//     (download_pinned_publish); the receiver's stream polls the word and copies once it matches
//     (upload_pinned_when_bounded). No host step separates the two streams, so either may be
//     launched first; a gate that is not released within its bound reports through the error word
//     instead of trapping.
// Both modes' stream operations are capturable, so each side may live in a CUDA graph. The words
// of the Gate mode are device words staged per round in stream order and must strictly increase,
// as for every pinned gate (core/device.h). The slot holds one round: the owner orders the
// sender's next round after the receiver's read of the current one (the round protocol
// synchronizes both devices at the end of every round).
//
// BulkHandoff moves large payloads (prefill chunks, layer-walk slices) through a ring of pinned
// pieces with copy engines on both sides and cross-device events, so the sender's download of one
// piece overlaps the receiver's upload of the previous one.
//
// Sender and receiver may be one device (the same-device test mode of a layer split). Every
// operation selects its device itself (DeviceGuard), so it may be called with any device current.

#include "core/arena.h"
#include "core/device.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace infernix {

// The code of a handoff gate's timeout in an error word ((boundary_layer << 16) | code), beside
// the MoE fetch channel's 0xFF01 and 0xFF02 (ops/offloaded_sparse_moe/cpu/fetch_request.h).
inline constexpr std::uint32_t kErrorHandoffTimeout = 0xFF03;

enum class HandoffMode : std::uint8_t { Event, Gate };

// One boundary's per-round channel, sized for its largest crossing (lanes * max width * 20 KiB of
// residual per column). The slot and its ready word are pinned host memory, addressed by both
// devices at their host addresses (unified addressing). Destroy it only once the streams and
// graphs that use it are idle.
class HandoffChannel {
public:
    HandoffChannel(int sender_device, int receiver_device, std::size_t capacity_bytes);
    ~HandoffChannel();

    HandoffChannel(const HandoffChannel&)            = delete;
    HandoffChannel& operator=(const HandoffChannel&) = delete;

    // Sender stream, capturable: `bytes` (at most capacity()) of `device_src` into the slot.
    //   Event: cudaMemcpyAsync D2H; `device_word` is not read.
    //   Gate: download_pinned_publish, which publishes `*device_word` after the data (16-byte
    //         multiples and alignment).
    void send(const void* device_src, std::size_t bytes, HandoffMode mode, const std::uint32_t* device_word,
              cudaStream_t sender_stream);

    // Event mode only, never inside a stream capture (both throw std::logic_error on a capturing
    // stream): after the sender's launch, records `sent` on its stream; before the receiver's
    // launch, makes the receiver's stream wait for the latest record.
    void record_sent(cudaStream_t sender_stream);
    void wait_sent(cudaStream_t receiver_stream) const;

    // Receiver stream, capturable: `bytes` of the slot into `device_dst`.
    //   Event: upload_pinned (after wait_sent); `device_expected`, `bound` and `wait_stats` are
    //          not read.
    //   Gate: upload_pinned_when_bounded with `bound`, waiting for `*device_expected`; waits that
    //         found the word unpublished add to `wait_stats` row 0 when it is set.
    void receive(void* device_dst, std::size_t bytes, HandoffMode mode, const std::uint32_t* device_expected,
                 const PinnedGateBound& bound, std::uint64_t* wait_stats, cudaStream_t receiver_stream) const;

    // Publishes `word` from the host (publish_pinned_word): error unwinding releases a receiver's
    // gate whose sender will not run, rather than leaving it to its timeout.
    void release(std::uint32_t word) noexcept;

    [[nodiscard]] std::size_t capacity() const noexcept;
    [[nodiscard]] int sender() const noexcept;
    [[nodiscard]] int receiver() const noexcept;

private:
    void check_payload(const char* op, const void* device_buffer, std::size_t bytes) const;

    int sender_           = 0;
    int receiver_         = 0;
    std::size_t capacity_ = 0;
    PinnedHostBuffer slot_; // capacity_ bytes of payload, then the ready word's own line
    std::uint32_t* ready_ = nullptr;
    cudaEvent_t sent_     = nullptr; // sender's device; Event mode
};

// A ring of `pieces` pinned pieces of `piece_bytes` between two devices. transfer() splits a copy
// into pieces: for each, the sender's stream downloads it into a free piece and records `filled`;
// the receiver's stream waits for that event, uploads the piece and records `consumed`; a piece is
// refilled only after its `consumed` event. The ring position and the events carry over between
// calls, so consecutive transfers pipeline as well. Calls come from one thread at a time. The
// destructor waits for the ring's last uploads before it frees the ring.
class BulkHandoff {
public:
    BulkHandoff(int sender_device, int receiver_device, std::size_t piece_bytes = 16ULL << 20,
                std::uint32_t pieces = 4);
    ~BulkHandoff();

    BulkHandoff(const BulkHandoff&)            = delete;
    BulkHandoff& operator=(const BulkHandoff&) = delete;

    // Copies `bytes` from `device_src` (sender's memory, ordered after earlier work of
    // `sender_stream`) to `device_dst` (receiver's memory; later work of `receiver_stream` sees it).
    // Enqueues only, unless host waits are on: then the host waits for each piece's download
    // before enqueuing its upload, and for a piece's last upload before refilling it. Like any
    // enqueue under a batched driver model (WDDM), the work is submitted by the caller's next
    // flush or blocking call. Not capturable.
    void transfer(const void* device_src, void* device_dst, std::size_t bytes, cudaStream_t sender_stream,
                  cudaStream_t receiver_stream);

    // Host waits (cudaEventSynchronize) instead of cross-device stream waits: the fallback should
    // cross-device waits prove slow under WDDM.
    void set_host_wait(bool host_wait) noexcept;

private:
    struct Piece {
        std::uint32_t slot = 0;
        std::size_t offset = 0;
        std::size_t bytes  = 0;
    };

    [[nodiscard]] std::byte* piece(std::uint32_t slot) const noexcept;
    void fill(const Piece& p, const std::byte* src, cudaStream_t sender_stream);
    void deliver(const Piece& p, std::byte* dst, cudaStream_t receiver_stream);
    void destroy_events() noexcept;

    int sender_              = 0;
    int receiver_            = 0;
    std::size_t piece_bytes_ = 0;
    std::uint32_t pieces_    = 0;
    PinnedHostBuffer ring_;
    std::vector<cudaEvent_t> filled_;   // sender's device
    std::vector<cudaEvent_t> consumed_; // receiver's device
    std::vector<std::uint8_t> used_;    // consumed_[slot] has been recorded
    std::uint32_t next_ = 0;
    bool host_wait_     = false;
};

} // namespace infernix
