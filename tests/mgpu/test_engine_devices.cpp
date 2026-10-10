// The Engine's device set (src/runtime/engine/device_set.h), which the Engine constructor runs
// before it creates anything else on a device.
//
//   - select_pci_bus_device_order, before the process's first CUDA call (the registration unsets
//     CUDA_DEVICE_ORDER): a value already set is kept; an unset variable becomes PCI_BUS_ID, and
//     CUDA then numbers the devices in ascending PCI (domain, bus, device) order.
//   - initialize_devices: one entry equal to the primary creates one context and logs nothing, as
//     `device` alone does. {0,0} creates two contexts of device 0 with distinct streams; {1,0}
//     (two GPUs) creates device 1's context and then device 0's. With more than one device each
//     context is logged in creation order, every context reports the spin schedule (the
//     registration unsets INFERNIX_CUDA_SYNC), the primary is the thread's device on return, and
//     one CudaInitialize phase covers the whole set. A device list that does not begin with the
//     primary is a logic error.
//   - layer_split_ends (no device): none on one device; {K, L} for --layer-split K on two;
//     K = 0, K >= L, a missing K, auto and @FILE (no split planner yet) are refused.
//
// Skips (77) without a CUDA device once the checks without one have passed; the {1,0} case needs
// two.

#include "core/device.h"
#include "runtime/engine/device_set.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

namespace {

int g_failures = 0;

void check(bool condition, const std::string& message) {
    if (condition) { return; }
    std::cerr << "FAIL: " << message << '\n';
    ++g_failures;
}

void set_environment(const char* name, const char* value) {
#if defined(_WIN32)
    // An empty value removes the variable.
    (void)::_putenv_s(name, value == nullptr ? "" : value);
#else
    if (value == nullptr) {
        (void)::unsetenv(name);
    } else {
        (void)::setenv(name, value, 1);
    }
#endif
}

std::string environment(const char* name) {
    const char* value = std::getenv(name);
    return value == nullptr ? std::string() : std::string(value);
}

int current_device() {
    int device = -1;
    (void)cudaGetDevice(&device);
    return device;
}

struct Observed {
    std::vector<std::string> lines;
    int begins    = 0;
    int completes = 0;
};

// initialize_devices over `devices` (device = devices.front()), recording its diagnostics and
// startup events.
infernix::DeviceContext create(const std::vector<int>& devices, Observed& observed,
                               std::vector<std::unique_ptr<infernix::DeviceContext>>& extra) {
    infernix::EngineOptions options;
    options.device                       = devices.front();
    options.devices                      = devices;
    options.diagnostic_observer.callback = [&](const infernix::Diagnostic& diagnostic) {
        observed.lines.push_back(diagnostic.message);
    };
    options.startup_observer.callback = [&](const infernix::StartupEvent& event) {
        if (event.phase != infernix::StartupPhase::CudaInitialize) { return; }
        observed.begins += event.status == infernix::StartupStatus::Begin ? 1 : 0;
        observed.completes += event.status == infernix::StartupStatus::Complete ? 1 : 0;
    };
    return infernix::runtime::initialize_devices(options, extra);
}

void check_set(const std::vector<int>& devices) {
    std::string name = "{";
    for (std::size_t i = 0; i < devices.size(); ++i) {
        name += (i == 0 ? "" : ",") + std::to_string(devices[i]);
    }
    name += "}";
    Observed observed;
    std::vector<std::unique_ptr<infernix::DeviceContext>> extra;
    infernix::DeviceContext primary = create(devices, observed, extra);
    check(observed.begins == 1 && observed.completes == 1, name + ": one CudaInitialize phase");
    check(primary.device == devices.front(), name + ": the primary is the first entry");
    check(extra.size() == devices.size() - 1, name + ": one further context per further entry");
    check(current_device() == devices.front(), name + ": the primary is current on return");
    std::vector<const infernix::DeviceContext*> contexts{&primary};
    for (const auto& context : extra) { contexts.push_back(context.get()); }
    for (std::size_t i = 0; i < contexts.size() && i < devices.size(); ++i) {
        const std::string label = name + ": context " + std::to_string(i);
        check(contexts[i]->device == devices[i], label + " is on its entry's device");
        check(std::string_view(contexts[i]->sync_mode()) == "spin", label + " spins");
        for (std::size_t j = 0; j < i; ++j) {
            check(contexts[i]->stream != contexts[j]->stream &&
                      contexts[i]->transfer_stream != contexts[j]->transfer_stream,
                  name + ": every context has its own streams");
        }
    }
    if (devices.size() == 1) {
        check(observed.lines.empty(), name + ": one device logs nothing");
    } else {
        check(observed.lines.size() == devices.size(), name + ": every context is logged");
        for (std::size_t i = 0; i < observed.lines.size() && i < devices.size(); ++i) {
            const std::string& line    = observed.lines[i];
            const std::string expected = "device " + std::to_string(i) + " of " +
                                         std::to_string(devices.size()) + ": CUDA " +
                                         std::to_string(devices[i]) + ",";
            check(line.starts_with(expected) && line.ends_with("spin synchronization"),
                  name + ": log line " + std::to_string(i) + " is \"" + line + "\"");
            std::cout << name << ' ' << line << '\n';
        }
    }
    // Each context runs work on its own stream.
    for (const infernix::DeviceContext* context : contexts) {
        context->bind_to_current_thread();
        void* scratch  = nullptr;
        const bool ran = cudaMalloc(&scratch, 256) == cudaSuccess &&
                         cudaMemsetAsync(scratch, 0, 256, context->stream) == cudaSuccess &&
                         cudaStreamSynchronize(context->stream) == cudaSuccess;
        const bool freed = scratch == nullptr || cudaFree(scratch) == cudaSuccess;
        check(ran && freed, name + ": a context's stream runs work");
    }
    primary.bind_to_current_thread();
}

// The layer split's device segments from the options and the layer count.
void check_layer_split_ends() {
    using infernix::runtime::layer_split_ends;
    const auto refused = [](const infernix::EngineOptions& options, std::uint32_t layers) {
        try {
            (void)layer_split_ends(options, layers);
        } catch (const std::invalid_argument&) { return true; }
        return false;
    };
    infernix::EngineOptions options;
    options.layer_split = 24;
    check(layer_split_ends(options, 48).empty(), "one device has no layer split");
    options.devices = {0};
    check(layer_split_ends(options, 48).empty(), "one listed device has no layer split");
    options.devices = {0, 1};
    check(layer_split_ends(options, 48) == std::vector<std::uint32_t>{24, 48},
          "--layer-split 24 of 48 layers gives segments ending at 24 and 48");
    options.layer_split = 1;
    check(layer_split_ends(options, 48) == std::vector<std::uint32_t>{1, 48}, "K = 1 is allowed");
    options.layer_split = 47;
    check(layer_split_ends(options, 48) == std::vector<std::uint32_t>{47, 48},
          "K = L - 1 is allowed");
    for (const std::uint32_t k : {0U, 48U, 49U}) {
        options.layer_split = k;
        check(refused(options, 48), "--layer-split " + std::to_string(k) + " of 48 is refused");
    }
    options.layer_split.reset();
    check(refused(options, 48), "two devices without K are refused");
    options.layer_split_auto = true;
    check(refused(options, 48), "--layer-split auto is refused until the planner exists");
    options.layer_split_auto = false;
    options.layer_split_plan = "plan.json";
    check(refused(options, 48), "--layer-split @FILE is refused until the planner exists");
}

} // namespace

int main() {
    check_layer_split_ends();
    // Before any CUDA call.
    if (!environment("CUDA_DEVICE_ORDER").empty()) {
        std::cerr << "FAIL: the registration must unset CUDA_DEVICE_ORDER\n";
        return 1;
    }
    set_environment("CUDA_DEVICE_ORDER", "FASTEST_FIRST");
    infernix::runtime::select_pci_bus_device_order();
    check(environment("CUDA_DEVICE_ORDER") == "FASTEST_FIRST",
          "a CUDA_DEVICE_ORDER the user set is kept");
    set_environment("CUDA_DEVICE_ORDER", nullptr);
    infernix::runtime::select_pci_bus_device_order();
    check(environment("CUDA_DEVICE_ORDER") == "PCI_BUS_ID",
          "an unset CUDA_DEVICE_ORDER becomes PCI_BUS_ID");
    infernix::runtime::select_pci_bus_device_order();
    check(environment("CUDA_DEVICE_ORDER") == "PCI_BUS_ID",
          "selecting the order twice keeps PCI_BUS_ID");

    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        std::cout << "SKIP: no usable CUDA device\n";
        return g_failures == 0 ? 77 : 1;
    }
    std::tuple<int, int, int> previous{-1, -1, -1};
    for (int device = 0; device < count; ++device) {
        cudaDeviceProp props{};
        if (cudaGetDeviceProperties(&props, device) != cudaSuccess) {
            check(false, "device properties of ordinal " + std::to_string(device));
            continue;
        }
        const std::tuple<int, int, int> pci{props.pciDomainID, props.pciBusID, props.pciDeviceID};
        check(pci > previous,
              "CUDA ordinal " + std::to_string(device) + " follows its predecessor in PCI order");
        previous = pci;
        std::cout << "CUDA " << device << ": " << props.name << ", PCI " << props.pciDomainID << ':'
                  << props.pciBusID << ':' << props.pciDeviceID << '\n';
    }

    check_set({0});
    check_set({0, 0});
    if (count >= 2) {
        check_set({1, 0});
    } else {
        std::cout << "one CUDA device: the {1,0} case is skipped\n";
    }
    {
        infernix::EngineOptions options;
        options.device  = 0;
        options.devices = {1, 0};
        std::vector<std::unique_ptr<infernix::DeviceContext>> extra;
        bool refused = false;
        try {
            (void)infernix::runtime::initialize_devices(options, extra);
        } catch (const std::logic_error&) { refused = true; }
        check(refused && extra.empty(), "a device list not beginning with the primary is refused");
    }

    std::cout << (g_failures == 0 ? "PASS" : "FAIL") << " Engine device set\n";
    return g_failures == 0 ? 0 : 1;
}
