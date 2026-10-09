// Per-device CUDA host state (src/core/per_device.h), host side only: no kernel launch and no
// device allocation. Every visible device (up to kMaxDevices) is checked, so a machine with two
// GPUs also covers the keying by device.
//
//   - PerDevice<T>::get(): one value-initialized instance per device, built with that device
//     current and handed its ordinal; the same instance on every later call, also after another
//     device's; a build that throws leaves the slot buildable by the next call; concurrent first
//     calls build once.
//   - once_per_device(): the action runs once on each device.
//   - numerics_multiprocessor_count(), DeviceContext::multiprocessor_count() and execution_view(),
//     in the mode the arguments name (each registration sets INFERNIX_NUMERICS_SMS to match):
//       --unpinned (default)  unset or 0: each device's cudaDevAttrMultiProcessorCount;
//       --pinned N            N on every device; props.multiProcessorCount stays physical;
//       --invalid             malformed or above a device's count: std::invalid_argument from the
//                             count and from DeviceContext construction.
//     A device outside [0, kMaxDevices) throws std::out_of_range in every mode.

#include "core/device.h"
#include "core/per_device.h"

#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <latch>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {

using infernix::kMaxDevices;
using infernix::PerDevice;

int g_failures = 0;

void expect(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << '\n';
        ++g_failures;
    }
}

void use_device(int device) { CUDA_CHECK(cudaSetDevice(device)); }

int physical_multiprocessors(int device) {
    int count = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device));
    return count;
}

void test_value_initialized(int devices) {
    PerDevice<int> values;
    std::vector<int*> instances;
    for (int device = 0; device < devices; ++device) {
        use_device(device);
        int& value = values.get();
        expect(value == 0, "get() value-initializes device " + std::to_string(device) + "'s T");
        value = 100 + device;
        expect(&values.get() == &value,
               "get() returns device " + std::to_string(device) + "'s instance again");
        for (int* other : instances) {
            expect(other != &value, "device " + std::to_string(device) + " has its own instance");
        }
        instances.push_back(&value);
    }
    for (int device = 0; device < devices; ++device) {
        use_device(device);
        expect(values.get() == 100 + device,
               "device " + std::to_string(device) + " keeps its value after the other devices'");
    }
}

void test_make(int devices) {
    PerDevice<int> values;
    int builds = 0;
    for (int round = 0; round < 2; ++round) {
        for (int device = 0; device < devices; ++device) {
            use_device(device);
            const int value = values.get([&](int ordinal) {
                ++builds;
                expect(ordinal == device, "make() receives the current device's ordinal");
                expect(infernix::current_device() == device, "make() runs with its device current");
                return 7 * ordinal + 3;
            });
            expect(value == 7 * device + 3,
                   "get(make) returns device " + std::to_string(device) + "'s built value");
        }
    }
    expect(builds == devices, "get(make) builds once per device (" + std::to_string(builds) +
                                  " builds on " + std::to_string(devices) + " devices)");
}

void test_retry_after_throw(int device) {
    use_device(device);
    PerDevice<int> values;
    bool threw = false;
    try {
        (void)values.get([](int) -> int { throw std::runtime_error("build failed"); });
    } catch (const std::runtime_error&) { threw = true; }
    expect(threw, "a throwing build propagates its exception");
    int builds      = 0;
    const int value = values.get([&](int) {
        ++builds;
        return 42;
    });
    expect(builds == 1 && value == 42, "the call after a throwing build builds the instance");
    expect(values.get([](int) { return -1; }) == 42, "a built instance is not built again");
}

void test_concurrent_first_calls(int device) {
    constexpr int kThreads = 8;
    PerDevice<int> values;
    std::atomic<int> builds{0};
    std::latch start(kThreads);
    std::vector<const int*> seen(kThreads, nullptr);
    std::vector<std::thread> threads;
    for (int i = 0; i < kThreads; ++i) {
        threads.emplace_back([&, i] {
            use_device(device); // the current device is per thread
            start.arrive_and_wait();
            seen[i] = &values.get([&](int) {
                builds.fetch_add(1);
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
                return 5;
            });
        });
    }
    for (auto& thread : threads) { thread.join(); }
    expect(builds.load() == 1,
           "concurrent first calls build once (" + std::to_string(builds.load()) + " builds)");
    for (const int* instance : seen) {
        expect(instance == seen[0] && *instance == 5,
               "concurrent first calls return the one instance");
    }
}

void test_once_per_device(int devices) {
    PerDevice<bool> done;
    std::vector<int> runs(static_cast<std::size_t>(devices), 0);
    for (int round = 0; round < 3; ++round) {
        for (int device = 0; device < devices; ++device) {
            use_device(device);
            infernix::once_per_device(done, [&] { ++runs[static_cast<std::size_t>(device)]; });
        }
    }
    for (int device = 0; device < devices; ++device) {
        expect(runs[static_cast<std::size_t>(device)] == 1,
               "once_per_device runs its action once on device " + std::to_string(device));
    }
}

// Whether call() throws an Exception (and not some other exception).
template <class Exception, class Call>
bool throws(Call&& call) {
    try {
        call();
    } catch (const Exception&) { return true; } catch (...) {
        // Another exception type fails the check as well.
    }
    return false;
}

void test_numerics(int devices, std::string_view mode, int pinned) {
    expect(throws<std::out_of_range>([] { (void)infernix::numerics_multiprocessor_count(-1); }),
           "a negative device ordinal is out of range");
    expect(throws<std::out_of_range>(
               [] { (void)infernix::numerics_multiprocessor_count(kMaxDevices); }),
           "kMaxDevices is out of range");
    for (int device = 0; device < devices; ++device) {
        const std::string label = "device " + std::to_string(device);
        const int physical      = physical_multiprocessors(device);
        if (mode == "--invalid") {
            expect(throws<std::invalid_argument>(
                       [&] { (void)infernix::numerics_multiprocessor_count(device); }),
                   label + ": an invalid INFERNIX_NUMERICS_SMS throws std::invalid_argument");
            expect(throws<std::invalid_argument>([&] {
                       const infernix::DeviceContext context(device);
                       (void)context;
                   }),
                   label + ": DeviceContext rejects an invalid INFERNIX_NUMERICS_SMS");
            continue;
        }
        const int expected = mode == "--pinned" ? pinned : physical;
        const int count    = infernix::numerics_multiprocessor_count(device);
        expect(count == expected, label + ": numerics count " + std::to_string(count) +
                                      ", expected " + std::to_string(expected));
        const infernix::DeviceContext context(device);
        expect(context.multiprocessor_count() == expected,
               label + ": DeviceContext::multiprocessor_count() reports the numerics count");
        expect(context.execution_view().multiprocessor_count == expected,
               label + ": execution_view() reports the numerics count");
        expect(context.props.multiProcessorCount == physical,
               label + ": props.multiProcessorCount stays the physical count");
    }
}

} // namespace

int main(int argc, char** argv) {
    const std::string_view mode = argc >= 2 ? std::string_view(argv[1]) : "--unpinned";
    int pinned                  = 0;
    if (mode == "--pinned") {
        if (argc != 3) {
            std::cerr << "usage: infernix_per_device_test --pinned N\n";
            return 2;
        }
        pinned = std::atoi(argv[2]);
    } else if ((mode != "--unpinned" && mode != "--invalid") || argc > 2) {
        std::cerr << "usage: infernix_per_device_test [--unpinned | --pinned N | --invalid]\n";
        return 2;
    }
    int count               = 0;
    const cudaError_t error = cudaGetDeviceCount(&count);
    if (error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver ||
        (error == cudaSuccess && count == 0)) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    if (error != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(error) << '\n';
        return 1;
    }
    const int devices = count < kMaxDevices ? count : kMaxDevices;
    try {
        test_value_initialized(devices);
        test_make(devices);
        test_once_per_device(devices);
        for (int device = 0; device < devices; ++device) {
            use_device(device);
            expect(infernix::current_device() == device,
                   "current_device() is the thread's device " + std::to_string(device));
            test_retry_after_throw(device);
            test_concurrent_first_calls(device);
        }
        test_numerics(devices, mode, pinned);
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
    if (g_failures != 0) {
        std::cerr << g_failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "per-device state: all checks passed on " << devices << " device(s), " << mode
              << '\n';
    return 0;
}
