#include "core/device.h"
#include "core/device_guard.h"
#include "core/per_device.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#if defined(_M_X64) || defined(__x86_64__)
#include <immintrin.h>
#endif

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace infernix {
namespace {

struct SyncSchedule {
    std::string_view name;
    unsigned int flags;
};

constexpr SyncSchedule kSyncSchedules[] = {
    {"spin", cudaDeviceScheduleSpin},
    {"blocking", cudaDeviceScheduleBlockingSync},
    {"yield", cudaDeviceScheduleYield},
    {"auto", cudaDeviceScheduleAuto},
};

unsigned int sync_schedule_from_environment() {
    const char* value = std::getenv("INFERNIX_CUDA_SYNC");
    if (value == nullptr) { return cudaDeviceScheduleSpin; }
    for (const auto& schedule : kSyncSchedules) {
        if (schedule.name == value) { return schedule.flags; }
    }
    throw std::invalid_argument("INFERNIX_CUDA_SYNC must be spin, blocking, yield, or auto");
}

// INFERNIX_NUMERICS_SMS: the pinned multiprocessor count, 0 (no pin) when unset or empty.
int numerics_sms_from_environment() {
    const char* value = std::getenv("INFERNIX_NUMERICS_SMS");
    if (value == nullptr || *value == '\0') { return 0; }
    char* end        = nullptr;
    errno            = 0;
    const long count = std::strtol(value, &end, 10);
    if (end == value || *end != '\0' || errno == ERANGE || count < 0 ||
        count > std::numeric_limits<int>::max()) {
        throw std::invalid_argument(
            "INFERNIX_NUMERICS_SMS must be a multiprocessor count (0 or unset: each device's own)");
    }
    return static_cast<int>(count);
}

// Read once, so every device and Op of the process sees one pin.
int pinned_numerics_sms() {
    static const int count = numerics_sms_from_environment();
    return count;
}

std::string cuda_error_message(const char* prefix, cudaError_t err) {
    return std::string(prefix) + ": " + cudaGetErrorName(err) + ": " + cudaGetErrorString(err);
}

void log_cuda_error(const char* op, cudaError_t err) noexcept {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA cleanup failed during %s: %s: %s\n", op, cudaGetErrorName(err),
                     cudaGetErrorString(err));
    }
}

void destroy_stream(cudaStream_t& stream) noexcept {
    if (stream != nullptr) {
        log_cuda_error("cudaStreamDestroy", cudaStreamDestroy(stream));
        stream = nullptr;
    }
}

void destroy_event(cudaEvent_t& event) noexcept {
    if (event != nullptr) {
        log_cuda_error("cudaEventDestroy", cudaEventDestroy(event));
        event = nullptr;
    }
}

} // namespace

void cuda_check(cudaError_t err, const char* expr, const char* file, int line) {
    if (err == cudaSuccess) { return; }
    std::fprintf(stderr, "%s:%d: CUDA_CHECK(%s) failed: %s: %s\n", file, line, expr,
                 cudaGetErrorName(err), cudaGetErrorString(err));
    std::abort();
}

int numerics_multiprocessor_count(int device) {
    if (device < 0 || device >= kMaxDevices) {
        throw std::out_of_range("CUDA device " + std::to_string(device) + " is not below " +
                                std::to_string(kMaxDevices));
    }
    // 0 until the device's first query; racing first queries store the same count.
    static std::array<std::atomic<int>, kMaxDevices> physical{};
    std::atomic<int>& cached = physical[static_cast<std::size_t>(device)];
    int count                = cached.load(std::memory_order_relaxed);
    if (count == 0) {
        const cudaError_t err =
            cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                cuda_error_message("cudaDeviceGetAttribute(MultiProcessorCount) failed", err));
        }
        if (count <= 0) {
            throw std::runtime_error("CUDA device " + std::to_string(device) +
                                     " reports no multiprocessors");
        }
        cached.store(count, std::memory_order_relaxed);
    }
    const int pinned = pinned_numerics_sms();
    if (pinned == 0) { return count; }
    if (pinned > count) {
        throw std::invalid_argument("INFERNIX_NUMERICS_SMS=" + std::to_string(pinned) +
                                    " exceeds the " + std::to_string(count) +
                                    " multiprocessors of CUDA device " + std::to_string(device));
    }
    return pinned;
}

DeviceContext::DeviceContext(int device_id) : device(device_id) {
    const unsigned int sync_flags = sync_schedule_from_environment();
    int count       = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaGetDeviceCount failed", err));
    }
    if (count <= 0) { throw std::runtime_error("no CUDA devices available"); }
    if (device_id < 0 || device_id >= count) { throw std::runtime_error("invalid CUDA device id"); }

    bind_to_current_thread();

    err = cudaSetDeviceFlags(sync_flags);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaSetDeviceFlags failed", err));
    }

    err = cudaGetDeviceProperties(&props, device_id);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaGetDeviceProperties failed", err));
    }

    // Validates INFERNIX_NUMERICS_SMS against this device and caches its count here, so that
    // multiprocessor_count() never queries or throws.
    (void)numerics_multiprocessor_count(device_id);

    cudaStream_t compute = nullptr;
    cudaStream_t load    = nullptr;
    err                  = cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            cuda_error_message("cudaStreamCreateWithFlags(stream) failed", err));
    }

    err = cudaStreamCreateWithFlags(&load, cudaStreamNonBlocking);
    if (err != cudaSuccess) {
        destroy_stream(compute);
        throw std::runtime_error(
            cuda_error_message("cudaStreamCreateWithFlags(transfer_stream) failed", err));
    }

    stream          = compute;
    transfer_stream = load;
}

DeviceContext::~DeviceContext() {
    if (stream != nullptr || transfer_stream != nullptr) { bind_to_current_thread_noexcept(); }
    destroy_stream(transfer_stream);
    destroy_stream(stream);
}

DeviceContext::DeviceContext(DeviceContext&& other) noexcept
    : device(other.device), stream(other.stream), transfer_stream(other.transfer_stream),
      props(other.props) {
    other.stream          = nullptr;
    other.transfer_stream = nullptr;
}

DeviceContext& DeviceContext::operator=(DeviceContext&& other) noexcept {
    if (this == &other) { return *this; }

    if (stream != nullptr || transfer_stream != nullptr) { bind_to_current_thread_noexcept(); }
    destroy_stream(transfer_stream);
    destroy_stream(stream);

    device          = other.device;
    props           = other.props;
    stream          = other.stream;
    transfer_stream = other.transfer_stream;

    other.stream          = nullptr;
    other.transfer_stream = nullptr;
    return *this;
}

void DeviceContext::bind_to_current_thread() const {
    const cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaSetDevice failed", err));
    }
}

void DeviceContext::bind_to_current_thread_noexcept() const noexcept {
    log_cuda_error("cudaSetDevice", cudaSetDevice(device));
}

int DeviceContext::compute_capability() const noexcept { return props.major * 10 + props.minor; }

// The count launch decompositions are sized with (numerics_multiprocessor_count, validated and
// cached by the constructor): INFERNIX_NUMERICS_SMS pins it, as for the Ops that query it
// themselves. props.multiProcessorCount stays the physical count.
int DeviceContext::multiprocessor_count() const noexcept {
    return numerics_multiprocessor_count(device);
}

DeviceExecutionView DeviceContext::execution_view() const noexcept {
    return {.stream = stream, .multiprocessor_count = multiprocessor_count()};
}

std::size_t DeviceContext::total_vram() const noexcept { return props.totalGlobalMem; }

const char* DeviceContext::sync_mode() const {
    bind_to_current_thread();
    unsigned int flags    = 0;
    const cudaError_t err = cudaGetDeviceFlags(&flags);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaGetDeviceFlags failed", err));
    }
    for (const auto& schedule : kSyncSchedules) {
        if ((flags & cudaDeviceScheduleMask) == schedule.flags) { return schedule.name.data(); }
    }
    throw std::runtime_error("unknown CUDA synchronization schedule");
}

void DeviceContext::synchronize() const { CUDA_CHECK(cudaStreamSynchronize(stream)); }

std::size_t DeviceContext::free_bytes() const {
    const DeviceGuard guard(device);
    std::size_t free  = 0;
    std::size_t total = 0;
    CUDA_CHECK(cudaMemGetInfo(&free, &total));
    return free;
}

void DeviceContext::flush() const {
    const cudaError_t status = cudaStreamQuery(stream);
    if (status == cudaErrorNotReady) {
        // Pending work is the expected answer; do not leave it as the thread's last error.
        (void)cudaGetLastError();
        return;
    }
    CUDA_CHECK(status);
}

CudaEventTimer::CudaEventTimer(const DeviceContext& ctx) : CudaEventTimer(ctx, ctx.stream) {}

CudaEventTimer::CudaEventTimer(const DeviceContext& ctx, cudaStream_t stream) : stream_(stream) {
    if (stream == nullptr) { throw std::invalid_argument("CUDA timer stream is null"); }
    ctx.bind_to_current_thread();

    cudaEvent_t start = nullptr;
    cudaEvent_t stop  = nullptr;
    cudaError_t err   = cudaEventCreate(&start);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaEventCreate(start) failed", err));
    }

    err = cudaEventCreate(&stop);
    if (err != cudaSuccess) {
        destroy_event(start);
        throw std::runtime_error(cuda_error_message("cudaEventCreate(stop) failed", err));
    }

    start_ = start;
    stop_  = stop;
}

CudaEventTimer::~CudaEventTimer() {
    destroy_event(stop_);
    destroy_event(start_);
}

CudaEventTimer::CudaEventTimer(CudaEventTimer&& other) noexcept
    : stream_(other.stream_), start_(other.start_), stop_(other.stop_) {
    other.stream_ = nullptr;
    other.start_  = nullptr;
    other.stop_   = nullptr;
}

CudaEventTimer& CudaEventTimer::operator=(CudaEventTimer&& other) noexcept {
    if (this == &other) { return *this; }

    destroy_event(stop_);
    destroy_event(start_);

    stream_ = other.stream_;
    start_  = other.start_;
    stop_   = other.stop_;

    other.stream_ = nullptr;
    other.start_  = nullptr;
    other.stop_   = nullptr;
    return *this;
}

void CudaEventTimer::start() { CUDA_CHECK(cudaEventRecord(start_, stream_)); }

void CudaEventTimer::record_stop() { CUDA_CHECK(cudaEventRecord(stop_, stream_)); }

float CudaEventTimer::elapsed_ms() const {
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
    return ms;
}

float CudaEventTimer::stop_ms() {
    record_stop();
    CUDA_CHECK(cudaEventSynchronize(stop_));
    return elapsed_ms();
}

CudaCompletionEvent::CudaCompletionEvent(const DeviceContext& ctx) : device_(ctx.device) {
    ctx.bind_to_current_thread();
    const cudaError_t err = cudaEventCreateWithFlags(&event_, cudaEventDisableTiming);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaEventCreateWithFlags failed", err));
    }
}

CudaCompletionEvent::~CudaCompletionEvent() { destroy_event(event_); }

CudaCompletionEvent::CudaCompletionEvent(CudaCompletionEvent&& other) noexcept
    : device_(other.device_), event_(std::exchange(other.event_, nullptr)) {}

CudaCompletionEvent& CudaCompletionEvent::operator=(CudaCompletionEvent&& other) noexcept {
    if (this == &other) { return *this; }
    destroy_event(event_);
    device_ = other.device_;
    event_  = std::exchange(other.event_, nullptr);
    return *this;
}

void CudaCompletionEvent::record(cudaStream_t stream) {
    if (event_ == nullptr || stream == nullptr) {
        throw std::logic_error("CUDA completion event is not recordable");
    }
    CUDA_CHECK(cudaEventRecord(event_, stream));
}

void CudaCompletionEvent::wait(cudaStream_t stream) const {
    if (event_ == nullptr || stream == nullptr) {
        throw std::logic_error("CUDA completion event is not waitable");
    }
    CUDA_CHECK(cudaStreamWaitEvent(stream, event_, 0));
}

void CudaCompletionEvent::record_external(cudaStream_t stream) {
    if (event_ == nullptr || stream == nullptr) {
        throw std::logic_error("CUDA completion event is not recordable");
    }
    CUDA_CHECK(cudaEventRecordWithFlags(event_, stream, cudaEventRecordExternal));
}

bool CudaCompletionEvent::ready() const {
    if (event_ == nullptr) { throw std::logic_error("CUDA completion event is empty"); }
    const cudaError_t status = cudaEventQuery(event_);
    if (status == cudaSuccess) { return true; }
    if (status == cudaErrorNotReady) { return false; }
    CUDA_CHECK(status);
    return false;
}

void CudaCompletionEvent::synchronize() const {
    if (event_ == nullptr) { throw std::logic_error("CUDA completion event is empty"); }
    CUDA_CHECK(cudaEventSynchronize(event_));
}


namespace {

__global__ void upload_pinned_kernel(void* __restrict__ dst, const void* __restrict__ src, std::size_t bytes) {
    const std::size_t thread = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    const bool vectors = (reinterpret_cast<std::uintptr_t>(dst) | reinterpret_cast<std::uintptr_t>(src)) % 16 == 0;
    const std::size_t head = vectors ? bytes / 16 : 0;
    for (std::size_t i = thread; i < head; i += stride) {
        static_cast<uint4*>(dst)[i] = static_cast<const uint4*>(src)[i];
    }
    for (std::size_t i = head * 16 + thread; i < bytes; i += stride) {
        static_cast<unsigned char*>(dst)[i] = static_cast<const unsigned char*>(src)[i];
    }
}

constexpr std::size_t kGateCtaBytes = 32 * 1024;
constexpr int kGateMaxCtas         = 16;
constexpr int kGateThreads         = 256;

// The CTA-strided copy of the gate kernels, kBatch loads in flight per thread before its stores.
// kUncachedSource: .cv loads, for a pinned source rewritten every round, so that no cache line of
// an earlier round is used.
template <bool kUncachedSource>
__device__ __forceinline__ void copy_vectors(uint4* __restrict__ dst, const uint4* __restrict__ src,
                                             std::size_t vectors) {
    constexpr int kBatch     = 8;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t base = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; base < vectors;
         base += kBatch * stride) {
        uint4 loaded[kBatch];
#pragma unroll
        for (int b = 0; b < kBatch; ++b) {
            const std::size_t i = base + b * stride;
            if (i < vectors) {
                if constexpr (kUncachedSource) {
                    loaded[b] = __ldcv(src + i);
                } else {
                    loaded[b] = src[i];
                }
            }
        }
#pragma unroll
        for (int b = 0; b < kBatch; ++b) {
            const std::size_t i = base + b * stride;
            if (i < vectors) { dst[i] = loaded[b]; }
        }
    }
}

__device__ __forceinline__ std::uint64_t global_ns() {
    std::uint64_t now;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
    return now;
}

// Adds one wait of `ns` to the gate's statistics row (wait ns, count).
__device__ __forceinline__ void record_gate_wait(std::uint64_t* stats, const std::int32_t* row, std::uint64_t ns) {
    std::uint64_t* at = stats + 2 * (row != nullptr ? *row : 0);
    atomicAdd(reinterpret_cast<unsigned long long*>(at), static_cast<unsigned long long>(ns));
    atomicAdd(reinterpret_cast<unsigned long long*>(at + 1), 1ULL);
}

// Each CTA's thread 0 polls the host word, so no CTA depends on another being resident.
__global__ void __launch_bounds__(kGateThreads)
    upload_pinned_when_kernel(uint4* __restrict__ dst, const uint4* __restrict__ src, std::size_t vectors,
                              const std::uint32_t* ready, const std::uint32_t* expected, std::uint64_t* stats,
                              const std::int32_t* row) {
    if (threadIdx.x == 0) {
        const std::uint32_t want = *expected;
        const auto* word         = reinterpret_cast<const volatile std::uint32_t*>(ready);
        if (*word != want) {
            const std::uint64_t start = global_ns();
            std::uint64_t now;
            do {
                __nanosleep(200);
                now = global_ns();
                if (now - start > kPinnedGateTimeoutNs) { asm volatile("trap;"); }
            } while (*word != want);
            if (stats != nullptr && blockIdx.x == 0) { record_gate_wait(stats, row, now - start); }
        }
        __threadfence_system();
    }
    __syncthreads();
    copy_vectors<true>(dst, src, vectors);
}

// upload_pinned_when_kernel with a soft bound: a CTA whose word has not matched for `timeout_ns`
// reports `error_value` and skips its copy. The word is read once more after the last sleep, so a
// word published during it still passes.
__global__ void __launch_bounds__(kGateThreads)
    upload_pinned_when_bounded_kernel(uint4* __restrict__ dst, const uint4* __restrict__ src, std::size_t vectors,
                                      const std::uint32_t* ready, const std::uint32_t* expected,
                                      std::uint64_t* stats, const std::int32_t* row, std::uint64_t timeout_ns,
                                      std::uint32_t* error_word, std::uint32_t error_value) {
    __shared__ bool expired;
    if (threadIdx.x == 0) {
        const std::uint32_t want = *expected;
        const auto* word         = reinterpret_cast<const volatile std::uint32_t*>(ready);
        bool arrived             = *word == want;
        if (!arrived) {
            const std::uint64_t start = global_ns();
            std::uint64_t now;
            do {
                __nanosleep(200);
                arrived = *word == want;
                now     = global_ns();
            } while (!arrived && now - start <= timeout_ns);
            if (stats != nullptr && blockIdx.x == 0) { record_gate_wait(stats, row, now - start); }
        }
        if (!arrived) { *reinterpret_cast<volatile std::uint32_t*>(error_word) = error_value; }
        // Arrived: orders the word's load before the copy's. Expired: makes the error visible.
        __threadfence_system();
        expired = !arrived;
    }
    __syncthreads();
    if (expired) { return; }
    copy_vectors<true>(dst, src, vectors);
}

// The first launch of download_pinned_publish. Each thread fences its own stores at system scope
// before the grid ends, so the publishing launch that follows in stream order releases them.
__global__ void __launch_bounds__(kGateThreads)
    download_pinned_kernel(uint4* __restrict__ dst, const uint4* __restrict__ src, std::size_t vectors) {
    copy_vectors<false>(dst, src, vectors);
    __threadfence_system();
}

// The second launch: one thread loads the round's word and releases it at system scope.
__global__ void publish_pinned_word_kernel(std::uint32_t* ready, const std::uint32_t* value) {
    const std::uint32_t word = *value;
    __threadfence_system();
    asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(ready), "r"(word) : "memory");
}

unsigned int gate_ctas(std::size_t bytes) {
    return static_cast<unsigned int>(std::min<std::size_t>((bytes + kGateCtaBytes - 1) / kGateCtaBytes, kGateMaxCtas));
}

void check_gate_buffers(const char* op, const void* device_buffer, const void* pinned_buffer, std::size_t bytes) {
    if (bytes % 16 != 0 ||
        (reinterpret_cast<std::uintptr_t>(device_buffer) | reinterpret_cast<std::uintptr_t>(pinned_buffer)) % 16 != 0) {
        throw std::invalid_argument(std::string(op) + ": bytes and buffers must be 16-byte multiples");
    }
}

} // namespace

void upload_pinned_when(void* device_dst, const void* pinned_src, std::size_t bytes, const std::uint32_t* pinned_ready,
                        const std::uint32_t* device_expected, std::uint64_t* wait_stats, const std::int32_t* wait_row,
                        cudaStream_t stream) {
    if (bytes == 0) { return; }
    check_gate_buffers("upload_pinned_when", device_dst, pinned_src, bytes);
    upload_pinned_when_kernel<<<gate_ctas(bytes), kGateThreads, 0, stream>>>(
        static_cast<uint4*>(device_dst), static_cast<const uint4*>(pinned_src), bytes / 16, pinned_ready,
        device_expected, wait_stats, wait_row);
    CUDA_CHECK(cudaGetLastError());
}

void upload_pinned_when_bounded(void* device_dst, const void* pinned_src, std::size_t bytes,
                                const std::uint32_t* pinned_ready, const std::uint32_t* device_expected,
                                std::uint64_t* wait_stats, const std::int32_t* wait_row, const PinnedGateBound& bound,
                                cudaStream_t stream) {
    if (bytes == 0) { return; }
    check_gate_buffers("upload_pinned_when_bounded", device_dst, pinned_src, bytes);
    if (pinned_ready == nullptr || device_expected == nullptr) {
        throw std::invalid_argument("upload_pinned_when_bounded: the ready word and the expected value must be set");
    }
    if (bound.error_word == nullptr) {
        throw std::invalid_argument("upload_pinned_when_bounded: a bounded gate needs an error word");
    }
    upload_pinned_when_bounded_kernel<<<gate_ctas(bytes), kGateThreads, 0, stream>>>(
        static_cast<uint4*>(device_dst), static_cast<const uint4*>(pinned_src), bytes / 16, pinned_ready,
        device_expected, wait_stats, wait_row, bound.timeout_ns, bound.error_word, bound.error_value);
    CUDA_CHECK(cudaGetLastError());
}

void download_pinned_publish(void* pinned_dst, const void* device_src, std::size_t bytes, std::uint32_t* pinned_ready,
                             const std::uint32_t* device_value, cudaStream_t stream) {
    if (pinned_ready == nullptr || device_value == nullptr) {
        throw std::invalid_argument("download_pinned_publish: the ready word and the device value must be set");
    }
    check_gate_buffers("download_pinned_publish", device_src, pinned_dst, bytes);
    if (bytes != 0) {
        download_pinned_kernel<<<gate_ctas(bytes), kGateThreads, 0, stream>>>(
            static_cast<uint4*>(pinned_dst), static_cast<const uint4*>(device_src), bytes / 16);
        CUDA_CHECK(cudaGetLastError());
    }
    publish_pinned_word_kernel<<<1, 1, 0, stream>>>(pinned_ready, device_value);
    CUDA_CHECK(cudaGetLastError());
}

void publish_pinned_word(std::uint32_t* pinned_ready, std::uint32_t value) noexcept {
#if defined(_M_X64) || defined(__x86_64__)
    _mm_sfence();
#endif
    std::atomic_thread_fence(std::memory_order_release);
    *static_cast<volatile std::uint32_t*>(pinned_ready) = value;
}

void upload_pinned(void* device_dst, const void* pinned_src, std::size_t bytes, cudaStream_t stream) {
    if (bytes == 0) { return; }
    static PerDevice<bool> unified_addressing;
    const bool unified = unified_addressing.get([](int device) {
        int value = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&value, cudaDevAttrUnifiedAddressing, device));
        return value != 0;
    });
    if (!unified) { throw std::runtime_error("upload_pinned needs unified addressing"); }
    constexpr int kThreads = 256;
    // One 16-byte load per thread: every load of a call is in flight at once.
    const std::size_t blocks = std::min<std::size_t>((bytes / 16 + kThreads) / kThreads + 1, 64);
    upload_pinned_kernel<<<static_cast<unsigned int>(blocks), kThreads, 0, stream>>>(device_dst, pinned_src, bytes);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace infernix
