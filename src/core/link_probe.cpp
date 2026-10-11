#include "core/link_probe.h"

#include "core/device.h"
#include "core/device_guard.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>

namespace infernix {

void wake_device_link(void* device_scratch, std::size_t bytes, std::chrono::milliseconds duration, int device) {
    if (device_scratch == nullptr || bytes == 0) {
        throw std::invalid_argument("link wake-up needs device scratch memory");
    }
    if (device < -1) { throw std::invalid_argument("link wake-up device must be an ordinal, or -1 for the current one"); }
    int target = device;
    if (target == -1) { CUDA_CHECK(cudaGetDevice(&target)); }
    const DeviceGuard guard(target);
    const std::size_t span = std::min<std::size_t>(bytes, std::size_t{64} << 20);
    cudaStream_t stream    = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    try {
        const auto until = std::chrono::steady_clock::now() + duration;
        do {
            for (int i = 0; i < 16; ++i) { CUDA_CHECK(cudaMemsetAsync(device_scratch, i, span, stream)); }
            CUDA_CHECK(cudaStreamSynchronize(stream));
        } while (std::chrono::steady_clock::now() < until);
    } catch (...) {
        cudaStreamDestroy(stream);
        throw;
    }
    cudaStreamDestroy(stream);
}

double measure_h2d_bytes_per_second(const void* pinned_src, void* device_dst, std::size_t bytes, int repetitions,
                                    int device) {
    if (pinned_src == nullptr || device_dst == nullptr || bytes == 0 || repetitions <= 0) {
        throw std::invalid_argument("link probe needs a source, a destination, bytes and repetitions");
    }
    if (device < -1) { throw std::invalid_argument("link probe device must be an ordinal, or -1 for the current one"); }
    int target = device;
    if (target == -1) { CUDA_CHECK(cudaGetDevice(&target)); }
    const DeviceGuard guard(target);
    cudaStream_t stream = nullptr;
    cudaEvent_t start = nullptr, stop = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    double best = 0.0;
    try {
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaMemcpyAsync(device_dst, pinned_src, bytes, cudaMemcpyHostToDevice, stream)); // warm-up
        for (int r = 0; r < repetitions; ++r) {
            CUDA_CHECK(cudaEventRecord(start, stream));
            CUDA_CHECK(cudaMemcpyAsync(device_dst, pinned_src, bytes, cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaEventRecord(stop, stream));
            CUDA_CHECK(cudaEventSynchronize(stop));
            float ms = 0.0F;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            if (ms > 0.0F) { best = std::max(best, static_cast<double>(bytes) / (static_cast<double>(ms) * 1e-3)); }
        }
    } catch (...) {
        if (start != nullptr) { cudaEventDestroy(start); }
        if (stop != nullptr) { cudaEventDestroy(stop); }
        cudaStreamDestroy(stream);
        throw;
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
    return best;
}

} // namespace infernix
