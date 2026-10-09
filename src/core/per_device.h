#pragma once

#include "core/device.h"

#include <cuda_runtime.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>

namespace infernix {

// Device ordinals a PerDevice can key.
inline constexpr int kMaxDevices = 16;

// The calling thread's current CUDA device (cudaGetDevice), below kMaxDevices.
inline int current_device() {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    if (device < 0 || device >= kMaxDevices) {
        throw std::out_of_range("CUDA device " + std::to_string(device) + " is not below " +
                                std::to_string(kMaxDevices));
    }
    return device;
}

// One lazily built T per CUDA device, for host state that belongs to one device context: kernel
// attributes set with cudaFuncSetAttribute, device attributes such as the SM count, and device
// buffers. A process-wide static would hold the state of the first device that reached it and
// apply it to every other device.
//
// get() returns the instance of the calling thread's current device and builds it on that
// device's first call, with that device current. Concurrent first calls build it once; a build
// that throws is retried by the next call. Every later call costs one cudaGetDevice and one
// acquire load. Instances live as long as the PerDevice, normally a function-local static.
template <class T>
class PerDevice {
public:
    // The current device's T, value-initialized on its first call.
    T& get() {
        return build_once([](std::optional<T>& value, int) { value.emplace(); });
    }

    // The current device's T, initialized from make(device) on its first call.
    template <class Make>
    T& get(Make&& make) {
        return build_once(
            [&](std::optional<T>& value, int device) { value.emplace(make(device)); });
    }

private:
    struct Slot {
        std::once_flag once;
        std::atomic<bool> ready{false};
        std::optional<T> value;
    };

    template <class Build>
    T& build_once(Build&& build) {
        const int device = current_device();
        Slot& slot       = slots_[static_cast<std::size_t>(device)];
        if (!slot.ready.load(std::memory_order_acquire)) {
            std::call_once(slot.once, [&] {
                build(slot.value, device);
                slot.ready.store(true, std::memory_order_release);
            });
        }
        return *slot.value;
    }

    std::array<Slot, kMaxDevices> slots_;
};

// Runs `action` on the first call with each device current; `done` records the devices where it
// completed. For settings that live in a device context, such as cudaFuncSetAttribute.
template <class Action>
void once_per_device(PerDevice<bool>& done, Action&& action) {
    (void)done.get([&](int) {
        action();
        return true;
    });
}

} // namespace infernix
