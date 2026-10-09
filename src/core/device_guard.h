#pragma once

// Scoped selection of the calling thread's current CUDA device.
//
// The CUDA runtime keeps one current device per host thread, and a stream-ordered call takes the
// context it runs in from it: a kernel launched into a stream of another device fails, an event
// recorded into one does too, and cudaMemGetInfo or cudaMalloc act on whichever device is current.
// With one device per process a thread binds once (DeviceContext::bind_to_current_thread). A
// layer split drives several devices from one thread, so every per-segment CUDA call (launch,
// copy, event, allocation, cudaMemGetInfo) runs under a DeviceGuard for that segment's device.
//
// A guard selects its device only when the thread is on another one, so a guard for the current
// device costs one cudaGetDevice. Guards nest: each restores the device that was current when it
// was constructed. A guard switches no device inside stream capture when its device is already
// current, which is how capture regions use it.
//
// A thread that never selected a device (no current context) reports device 0. Restoring that
// would create device 0's primary context (cudaSetDevice initializes it since CUDA 12) on a GPU the
// thread never used, so a guard that switched such a thread leaves it on the guard's device.

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <stdexcept>
#include <string>

namespace infernix {

class DeviceGuard {
public:
    // Remembers the current device and makes `device` current if it is not. An invalid ordinal
    // throws std::runtime_error and leaves the thread on its device and its CUDA error state
    // clean.
    explicit DeviceGuard(int device) {
        cudaError_t err = cudaGetDevice(&previous_);
        if (err == cudaSuccess && device != previous_) {
            // Before the switch, which makes the new device's context current.
            CUcontext current = nullptr;
            const bool bound  = cuCtxGetCurrent(&current) == CUDA_SUCCESS && current != nullptr;
            err               = cudaSetDevice(device);
            switched_         = err == cudaSuccess && bound;
        }
        if (err != cudaSuccess) {
            (void)cudaGetLastError(); // the exception reports it; later launch checks must not
            throw std::runtime_error("DeviceGuard: cannot select CUDA device " +
                                     std::to_string(device) + ": " + cudaGetErrorName(err) + ": " +
                                     cudaGetErrorString(err));
        }
    }

    // Restores the device that was current at construction (a thread that had selected none stays
    // on the guard's device); a failure is logged, not thrown.
    ~DeviceGuard() {
        if (!switched_) { return; }
        const cudaError_t err = cudaSetDevice(previous_);
        if (err != cudaSuccess) {
            (void)cudaGetLastError();
            std::fprintf(stderr, "DeviceGuard: cannot restore CUDA device %d: %s: %s\n", previous_,
                         cudaGetErrorName(err), cudaGetErrorString(err));
        }
    }

    DeviceGuard(const DeviceGuard&)            = delete;
    DeviceGuard& operator=(const DeviceGuard&) = delete;

    // The device that was current when the guard was constructed (and is current again after it).
    // On a thread that had selected none it is the 0 cudaGetDevice reports.
    [[nodiscard]] int previous() const noexcept { return previous_; }

private:
    int previous_  = 0;
    bool switched_ = false; // previous_ is restored on destruction
};

} // namespace infernix
