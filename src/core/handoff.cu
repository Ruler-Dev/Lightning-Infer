#include "core/handoff.h"

#include "core/device_guard.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>

namespace infernix {
namespace {

// The ready word sits on its own line after the payload, so the sender's payload stores and the
// receiver's polling never share one.
constexpr std::size_t kWordLine = 64;

std::string cuda_error_message(const char* prefix, cudaError_t err) {
    return std::string(prefix) + ": " + cudaGetErrorName(err) + ": " + cudaGetErrorString(err);
}

void log_cuda_error(const char* op, cudaError_t err) noexcept {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA cleanup failed during %s: %s: %s\n", op, cudaGetErrorName(err),
                     cudaGetErrorString(err));
    }
}

// Both sides address pinned memory at its host address (SM stores and loads through the mapping).
void require_unified_device(int device, const char* owner) {
    int count       = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) { throw std::runtime_error(cuda_error_message("cudaGetDeviceCount failed", err)); }
    if (device < 0 || device >= count) {
        throw std::invalid_argument(std::string(owner) + ": CUDA device " + std::to_string(device) +
                                    " does not exist");
    }
    int unified = 0;
    err         = cudaDeviceGetAttribute(&unified, cudaDevAttrUnifiedAddressing, device);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaDeviceGetAttribute(UnifiedAddressing) failed", err));
    }
    if (unified == 0) {
        throw std::runtime_error(std::string(owner) + ": CUDA device " + std::to_string(device) +
                                 " has no unified addressing");
    }
}

// `bytes` of pinned memory, allocated with `device` current once it is checked.
PinnedHostBuffer pinned_on(int device, std::size_t bytes, const char* owner) {
    require_unified_device(device, owner);
    const DeviceGuard guard(device);
    return PinnedHostBuffer(bytes);
}

std::size_t slot_bytes(std::size_t capacity) {
    if (capacity == 0 || capacity > std::numeric_limits<std::size_t>::max() / 2) {
        throw std::invalid_argument("HandoffChannel: the capacity must be nonzero and addressable");
    }
    return (capacity + kWordLine - 1) / kWordLine * kWordLine + kWordLine;
}

std::size_t ring_bytes(std::size_t piece_bytes, std::uint32_t pieces) {
    if (piece_bytes == 0 || pieces == 0 || piece_bytes > std::numeric_limits<std::size_t>::max() / pieces) {
        throw std::invalid_argument("BulkHandoff: the ring needs at least one piece of at least one byte");
    }
    return piece_bytes * pieces;
}

cudaEvent_t create_event(int device) {
    const DeviceGuard guard(device);
    cudaEvent_t event     = nullptr;
    const cudaError_t err = cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
    if (err != cudaSuccess) { throw std::runtime_error(cuda_error_message("cudaEventCreateWithFlags failed", err)); }
    return event;
}

void destroy_event(cudaEvent_t& event) noexcept {
    if (event != nullptr) {
        log_cuda_error("cudaEventDestroy", cudaEventDestroy(event));
        event = nullptr;
    }
}

bool capturing(cudaStream_t stream) {
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &status));
    return status != cudaStreamCaptureStatusNone;
}

} // namespace

HandoffChannel::HandoffChannel(int sender_device, int receiver_device, std::size_t capacity_bytes)
    : sender_(sender_device), receiver_(receiver_device), capacity_(capacity_bytes),
      slot_(pinned_on(sender_device, slot_bytes(capacity_bytes), "HandoffChannel")) {
    require_unified_device(receiver_device, "HandoffChannel");
    ready_ = reinterpret_cast<std::uint32_t*>(static_cast<std::byte*>(slot_.data()) + slot_.size() - kWordLine);
    publish_pinned_word(ready_, 0); // pinned memory starts undefined; 0 is never a round's word
    sent_ = create_event(sender_device);
}

HandoffChannel::~HandoffChannel() { destroy_event(sent_); }

void HandoffChannel::send(const void* device_src, std::size_t bytes, HandoffMode mode,
                          const std::uint32_t* device_word, cudaStream_t sender_stream) {
    check_payload("HandoffChannel::send", device_src, bytes);
    const DeviceGuard guard(sender_);
    if (mode == HandoffMode::Event) {
        if (bytes != 0) {
            CUDA_CHECK(cudaMemcpyAsync(slot_.data(), device_src, bytes, cudaMemcpyDeviceToHost, sender_stream));
        }
        return;
    }
    download_pinned_publish(slot_.data(), device_src, bytes, ready_, device_word, sender_stream);
}

void HandoffChannel::record_sent(cudaStream_t sender_stream) {
    const DeviceGuard guard(sender_);
    if (capturing(sender_stream)) {
        throw std::logic_error("HandoffChannel::record_sent: the sender's stream is capturing");
    }
    CUDA_CHECK(cudaEventRecord(sent_, sender_stream));
}

void HandoffChannel::wait_sent(cudaStream_t receiver_stream) const {
    const DeviceGuard guard(receiver_);
    if (capturing(receiver_stream)) {
        throw std::logic_error("HandoffChannel::wait_sent: the receiver's stream is capturing");
    }
    CUDA_CHECK(cudaStreamWaitEvent(receiver_stream, sent_, 0));
}

void HandoffChannel::receive(void* device_dst, std::size_t bytes, HandoffMode mode,
                             const std::uint32_t* device_expected, const PinnedGateBound& bound,
                             std::uint64_t* wait_stats, cudaStream_t receiver_stream) const {
    check_payload("HandoffChannel::receive", device_dst, bytes);
    const DeviceGuard guard(receiver_);
    if (mode == HandoffMode::Event) {
        upload_pinned(device_dst, slot_.data(), bytes, receiver_stream);
        return;
    }
    upload_pinned_when_bounded(device_dst, slot_.data(), bytes, ready_, device_expected, wait_stats, nullptr, bound,
                               receiver_stream);
}

void HandoffChannel::release(std::uint32_t word) noexcept { publish_pinned_word(ready_, word); }

void HandoffChannel::check_payload(const char* op, const void* device_buffer, std::size_t bytes) const {
    if (bytes > capacity_) {
        throw std::invalid_argument(std::string(op) + ": " + std::to_string(bytes) + " bytes exceed the capacity of " +
                                    std::to_string(capacity_));
    }
    if (bytes != 0 && device_buffer == nullptr) { throw std::invalid_argument(std::string(op) + ": no device buffer"); }
}

std::size_t HandoffChannel::capacity() const noexcept { return capacity_; }

int HandoffChannel::sender() const noexcept { return sender_; }

int HandoffChannel::receiver() const noexcept { return receiver_; }

BulkHandoff::BulkHandoff(int sender_device, int receiver_device, std::size_t piece_bytes, std::uint32_t pieces)
    : sender_(sender_device), receiver_(receiver_device), piece_bytes_(piece_bytes), pieces_(pieces),
      ring_(pinned_on(sender_device, ring_bytes(piece_bytes, pieces), "BulkHandoff")) {
    require_unified_device(receiver_device, "BulkHandoff");
    filled_.assign(pieces, nullptr);
    consumed_.assign(pieces, nullptr);
    used_.assign(pieces, 0);
    try {
        for (std::uint32_t i = 0; i < pieces; ++i) {
            filled_[i]   = create_event(sender_device);   // recorded on the sender's stream
            consumed_[i] = create_event(receiver_device); // recorded on the receiver's stream
        }
    } catch (...) {
        destroy_events();
        throw;
    }
}

BulkHandoff::~BulkHandoff() {
    // A piece's last upload may still be reading the ring; its download completed before it.
    for (std::uint32_t i = 0; i < pieces_; ++i) {
        if (used_[i] != 0) { log_cuda_error("cudaEventSynchronize", cudaEventSynchronize(consumed_[i])); }
    }
    destroy_events();
}

void BulkHandoff::transfer(const void* device_src, void* device_dst, std::size_t bytes, cudaStream_t sender_stream,
                           cudaStream_t receiver_stream) {
    if (bytes == 0) { return; }
    if (device_src == nullptr || device_dst == nullptr) {
        throw std::invalid_argument("BulkHandoff::transfer: the source and the destination must be set");
    }
    // The legacy default stream would be whichever device's is current at each call.
    if (sender_stream == nullptr || receiver_stream == nullptr) {
        throw std::invalid_argument("BulkHandoff::transfer needs the devices' own streams");
    }
    const auto* src = static_cast<const std::byte*>(device_src);
    auto* dst       = static_cast<std::byte*>(device_dst);
    // Each piece's upload is enqueued after the next piece's download, so that with host waits the
    // host waits for one download while the next one runs.
    std::optional<Piece> pending;
    for (std::size_t offset = 0; offset < bytes; offset += piece_bytes_) {
        const Piece p{next_, offset, std::min(piece_bytes_, bytes - offset)};
        next_ = (next_ + 1) % pieces_;
        if (pending && pending->slot == p.slot) { // a one-piece ring: deliver before refilling
            deliver(*pending, dst, receiver_stream);
            pending.reset();
        }
        fill(p, src, sender_stream);
        if (pending) { deliver(*pending, dst, receiver_stream); }
        pending = p;
    }
    deliver(*pending, dst, receiver_stream);
}

void BulkHandoff::set_host_wait(bool host_wait) noexcept { host_wait_ = host_wait; }

std::byte* BulkHandoff::piece(std::uint32_t slot) const noexcept {
    return static_cast<std::byte*>(ring_.data()) + static_cast<std::size_t>(slot) * piece_bytes_;
}

void BulkHandoff::fill(const Piece& p, const std::byte* src, cudaStream_t sender_stream) {
    const bool reused = used_[p.slot] != 0;
    if (reused && host_wait_) { CUDA_CHECK(cudaEventSynchronize(consumed_[p.slot])); }
    const DeviceGuard guard(sender_);
    if (reused && !host_wait_) { CUDA_CHECK(cudaStreamWaitEvent(sender_stream, consumed_[p.slot], 0)); }
    CUDA_CHECK(cudaMemcpyAsync(piece(p.slot), src + p.offset, p.bytes, cudaMemcpyDeviceToHost, sender_stream));
    CUDA_CHECK(cudaEventRecord(filled_[p.slot], sender_stream));
}

void BulkHandoff::deliver(const Piece& p, std::byte* dst, cudaStream_t receiver_stream) {
    if (host_wait_) { CUDA_CHECK(cudaEventSynchronize(filled_[p.slot])); }
    const DeviceGuard guard(receiver_);
    if (!host_wait_) { CUDA_CHECK(cudaStreamWaitEvent(receiver_stream, filled_[p.slot], 0)); }
    CUDA_CHECK(cudaMemcpyAsync(dst + p.offset, piece(p.slot), p.bytes, cudaMemcpyHostToDevice, receiver_stream));
    CUDA_CHECK(cudaEventRecord(consumed_[p.slot], receiver_stream));
    used_[p.slot] = 1;
}

void BulkHandoff::destroy_events() noexcept {
    for (cudaEvent_t& event : filled_) { destroy_event(event); }
    for (cudaEvent_t& event : consumed_) { destroy_event(event); }
}

} // namespace infernix
