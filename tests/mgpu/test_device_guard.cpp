// Device selection for a layer split's per-segment CUDA calls (src/core/device_guard.h) and the core
// queries that select their own device: DeviceContext::free_bytes, the VRAM budget source and the
// link probe. Every visible device is used, so a machine with two GPUs also checks that each query
// reaches the device it names and not the thread's current one.
//
//   - DeviceGuard: a guard for the current device switches nothing; nested guards each restore the
//     device that was current at their construction; an invalid ordinal throws std::runtime_error
//     and leaves the thread's device and CUDA error state as they were. On a thread that never
//     selected a device (two GPUs), a guard for device 1 leaves the thread on device 1 and creates
//     no primary context on device 0; a guard nested in it still restores device 1.
//   - Two DeviceContexts of one device: the second constructor's cudaSetDeviceFlags, on a device
//     the first initialized, overwrites the flags rather than failing, and both contexts report the
//     spin schedule (the registration unsets INFERNIX_CUDA_SYNC). Their streams are distinct.
//   - With another device current: DeviceContext::free_bytes() and open_vram_budget_source(d)'s
//     query() read device d (an allocation on d lowers d's free memory and not the other device's;
//     the snapshot's total is d's) and measure_h2d_bytes_per_second(..., d) measures a copy to d's
//     memory; each leaves the caller's device current. Device -1 is the current device.

#include "core/device.h"
#include "core/device_guard.h"
#include "core/link_probe.h"
#include "core/vram_budget.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>

namespace {

using infernix::DeviceGuard;

int g_failures = 0;

void expect(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << '\n';
        ++g_failures;
    }
}

void use_device(int device) { CUDA_CHECK(cudaSetDevice(device)); }

int current() {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    return device;
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

// cudaMemGetInfo of `device`, read with that device current.
void memory_of(int device, std::size_t& free_bytes, std::size_t& total_bytes) {
    const DeviceGuard guard(device);
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
}

void cu_check(CUresult result, const char* what) {
    if (result != CUDA_SUCCESS) {
        const char* name = nullptr;
        (void)cuGetErrorName(result, &name);
        throw std::runtime_error(std::string(what) + " failed: " + (name != nullptr ? name : "unknown CUresult"));
    }
}

bool primary_context_active(int device) {
    CUdevice handle = 0;
    cu_check(cuDeviceGet(&handle, device), "cuDeviceGet");
    unsigned int flags = 0;
    int active         = 0;
    cu_check(cuDevicePrimaryCtxGetState(handle, &flags, &active), "cuDevicePrimaryCtxGetState");
    return active != 0;
}

bool thread_has_context() {
    CUcontext context = nullptr;
    return cuCtxGetCurrent(&context) == CUDA_SUCCESS && context != nullptr;
}

// A thread that never selected a device: cudaGetDevice reports 0 on it, and restoring device 0
// would create that device's primary context. Runs before any other check, while nothing in this
// process has made device 0 current.
void test_unbound_thread(int devices) {
    if (devices < 2) { return; }
    cu_check(cuInit(0), "cuInit");
    const bool device0_before = primary_context_active(0);
    // Runs `body` on a new thread, which has no current context.
    const auto on_new_thread = [](auto body) {
        std::string error;
        std::thread worker([&] {
            try {
                expect(!thread_has_context(), "a new thread has no current context");
                body();
            } catch (const std::exception& e) { error = e.what(); }
        });
        worker.join();
        if (!error.empty()) { throw std::runtime_error(error); }
    };
    on_new_thread([&] {
        {
            const DeviceGuard guard(1);
            expect(guard.previous() == 0, "an unbound thread's guard records device 0");
            expect(current() == 1 && thread_has_context(), "the guard selects device 1");
            CUDA_CHECK(cudaFree(nullptr));
        }
        expect(current() == 1 && thread_has_context(), "the guard leaves an unbound thread on device 1");
        if (!device0_before) {
            expect(!primary_context_active(0), "a guard on an unbound thread creates no context on device 0");
        }
    });
    on_new_thread([&] {
        {
            const DeviceGuard outer(1);
            {
                const DeviceGuard inner(0);
                expect(inner.previous() == 1 && current() == 0, "a nested guard selects device 0");
            }
            expect(current() == 1, "the nested guard restores device 1");
        }
        expect(current() == 1, "the outer guard leaves the thread on device 1");
    });
    if (device0_before) {
        std::cout << "note: device 0's primary context was active before the unbound-thread check" << std::endl;
    }
}

void test_current_device_guard(int devices) {
    for (int device = 0; device < devices; ++device) {
        const std::string label = "device " + std::to_string(device);
        use_device(device);
        {
            const DeviceGuard guard(device);
            expect(guard.previous() == device, label + ": a guard records the current device");
            expect(current() == device, label + ": a guard for the current device keeps it");
        }
        expect(current() == device, label + ": the current device stays after the guard");
    }
}

void test_nested_guards(int devices) {
    const int other = devices > 1 ? 1 : 0;
    use_device(0);
    {
        const DeviceGuard outer(other);
        expect(outer.previous() == 0, "the outer guard records device 0");
        expect(current() == other, "the outer guard selects device " + std::to_string(other));
        {
            const DeviceGuard inner(0);
            expect(inner.previous() == other, "the inner guard records the outer guard's device");
            expect(current() == 0, "the inner guard selects device 0");
            {
                const DeviceGuard innermost(other);
                expect(innermost.previous() == 0, "the innermost guard records the inner guard's device");
                expect(current() == other, "the innermost guard selects its device");
            }
            expect(current() == 0, "the innermost guard restores the inner guard's device");
        }
        expect(current() == other, "the inner guard restores the outer guard's device");
    }
    expect(current() == 0, "the outer guard restores device 0");
}

void test_invalid_guard(int count) {
    const int last = count - 1;
    use_device(last);
    for (const int invalid : {-1, count}) {
        const std::string label = "a guard for device " + std::to_string(invalid);
        expect(throws<std::runtime_error>([&] { const DeviceGuard guard(invalid); }), label + " throws");
        expect(current() == last, label + " leaves the thread on its device");
        expect(cudaGetLastError() == cudaSuccess, label + " leaves no CUDA error behind");
    }
}

void test_two_contexts(int devices) {
    use_device(0);
    const infernix::DeviceContext first(0);
    const infernix::DeviceContext second(0);
    expect(std::string_view(first.sync_mode()) == "spin", "the first context of device 0 spins");
    expect(std::string_view(second.sync_mode()) == "spin", "the second context of device 0 spins");
    expect(first.stream != second.stream && first.transfer_stream != second.transfer_stream,
           "two contexts of one device have their own streams");
    // Both stream sets work.
    int* value = nullptr;
    CUDA_CHECK(cudaMalloc(&value, 2 * sizeof(int)));
    CUDA_CHECK(cudaMemsetAsync(value, 0x11, sizeof(int), first.stream));
    CUDA_CHECK(cudaMemsetAsync(value + 1, 0x22, sizeof(int), second.stream));
    first.synchronize();
    second.synchronize();
    int host[2] = {};
    CUDA_CHECK(cudaMemcpy(host, value, sizeof(host), cudaMemcpyDeviceToHost));
    expect(host[0] == 0x11111111 && host[1] == 0x22222222, "both contexts' streams run their work");
    CUDA_CHECK(cudaFree(value));
    if (devices > 1) {
        const infernix::DeviceContext other(1);
        const infernix::DeviceContext again(0);
        expect(std::string_view(other.sync_mode()) == "spin" && std::string_view(again.sync_mode()) == "spin",
               "contexts of devices 1 and 0 created alternately spin");
    }
}

void test_free_bytes(int devices) {
    constexpr std::size_t kStep = 512ULL << 20;
    for (int device = 0; device < devices; ++device) {
        const std::string label = "DeviceContext(" + std::to_string(device) + ")::free_bytes()";
        const infernix::DeviceContext context(device);
        const int other = (device + 1) % devices;
        use_device(other);
        const std::size_t before = context.free_bytes();
        expect(current() == other, label + " restores the caller's device");
        expect(before > 0 && before <= context.total_vram(), label + " lies in (0, total VRAM]");
        if (devices < 2) { continue; }
        std::size_t other_before = 0, other_after = 0, total = 0;
        memory_of(other, other_before, total);
        void* step = nullptr;
        {
            const DeviceGuard guard(device);
            if (cudaMalloc(&step, kStep) != cudaSuccess) {
                (void)cudaGetLastError();
                std::cout << "NOTE: " << label << ": no room for a " << (kStep >> 20)
                          << " MiB allocation; the binding check is skipped\n";
                continue;
            }
            CUDA_CHECK(cudaMemset(step, 0, kStep)); // resident, also under WDDM
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        const std::size_t after = context.free_bytes();
        memory_of(other, other_after, total);
        const auto drop = [](std::size_t from, std::size_t to) {
            return static_cast<long long>(from) - static_cast<long long>(to);
        };
        expect(drop(before, after) >= static_cast<long long>(kStep - (64ULL << 20)),
               label + " sees an allocation on its device while device " + std::to_string(other) +
                   " is current (dropped " + std::to_string(drop(before, after) / (1LL << 20)) + " MiB)");
        expect(drop(other_before, other_after) < static_cast<long long>(kStep / 2),
               "an allocation on device " + std::to_string(device) + " leaves device " + std::to_string(other) +
                   "'s free memory (dropped " + std::to_string(drop(other_before, other_after) / (1LL << 20)) + " MiB)");
        expect(current() == other, label + " restores the caller's device after the allocation");
        const DeviceGuard guard(device);
        CUDA_CHECK(cudaFree(step));
    }
}

void test_vram_budget_source(int devices) {
    for (int device = 0; device < devices; ++device) {
        const std::string label = "the VRAM budget source of device " + std::to_string(device);
        const int other = (device + 1) % devices;
        std::size_t free_bytes = 0, total = 0;
        memory_of(device, free_bytes, total);
        use_device(other);
        const auto source = infernix::open_vram_budget_source(device);
        const infernix::VramSnapshot snapshot = source->query();
        expect(current() == other, label + " restores the caller's device");
        expect(snapshot.device_total == total, label + " reports its device's total (" +
                                                   std::to_string(snapshot.device_total >> 20) + " MiB, expected " +
                                                   std::to_string(total >> 20) + " MiB)");
        expect(snapshot.device_free > 0 && snapshot.device_free <= total,
               label + " reports free memory within its device's total");
    }
}

void test_link_probe(int devices) {
    constexpr std::size_t kBytes = 8ULL << 20;
    void* source = nullptr;
    CUDA_CHECK(cudaMallocHost(&source, kBytes));
    for (int device = 0; device < devices; ++device) {
        const std::string label = "the link probe of device " + std::to_string(device);
        void* destination = nullptr;
        {
            const DeviceGuard guard(device);
            CUDA_CHECK(cudaMalloc(&destination, kBytes));
        }
        const int other = (device + 1) % devices;
        use_device(other);
        const double named = infernix::measure_h2d_bytes_per_second(source, destination, kBytes, 2, device);
        expect(named > 0.0, label + " measures a rate");
        expect(current() == other, label + " restores the caller's device");
        use_device(device);
        const double current_rate = infernix::measure_h2d_bytes_per_second(source, destination, kBytes, 2);
        expect(current_rate > 0.0 && current() == device, label + " measures the current device for -1");
        std::cout << "device " << device << ": host-to-device " << named / 1e9 << " GB/s (named), "
                  << current_rate / 1e9 << " GB/s (current)\n";
        expect(throws<std::invalid_argument>(
                   [&] { (void)infernix::measure_h2d_bytes_per_second(source, destination, kBytes, 1, -2); }),
               label + " rejects device -2");
        CUDA_CHECK(cudaFree(destination));
    }
    CUDA_CHECK(cudaFreeHost(source));
}

} // namespace

int main() {
    int count               = 0;
    const cudaError_t error = cudaGetDeviceCount(&count);
    if (error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver || (error == cudaSuccess && count == 0)) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    if (error != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(error) << '\n';
        return 1;
    }
    try {
        test_unbound_thread(count);
        test_current_device_guard(count);
        test_nested_guards(count);
        test_invalid_guard(count);
        test_two_contexts(count);
        test_free_bytes(count);
        test_vram_budget_source(count);
        test_link_probe(count);
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
    if (g_failures != 0) {
        std::cerr << g_failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "device guard: all checks passed on " << count << " device(s)"
              << (count < 2 ? " (one device: the cross-device binding checks need two)" : "") << '\n';
    return 0;
}
