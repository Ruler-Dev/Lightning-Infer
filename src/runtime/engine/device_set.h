#pragma once

// The CUDA devices an Engine runs on (EngineOptions::device and ::devices): the numbering their
// ordinals use, the creation of their contexts and the layer split's segments across them.

#include "core/device.h"
#include "core/startup.h"
#include "infernix/types.h"
#include "runtime/engine/diagnostics.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace infernix::runtime {

// CUDA numbers devices fastest first unless CUDA_DEVICE_ORDER says otherwise, so on a machine with
// two different GPUs its ordinals need not match nvidia-smi's or the PCI slots. This selects
// PCI_BUS_ID unless the variable is already set (a user's choice is kept). CUDA reads the variable
// once, when the process first uses it, so this takes effect only before the process's first CUDA
// call: the Engine constructor calls it first.
inline void select_pci_bus_device_order() {
#if defined(_WIN32)
    std::size_t length = 0;
    if (::getenv_s(&length, nullptr, 0, "CUDA_DEVICE_ORDER") == 0 && length != 0) { return; }
    if (::_putenv_s("CUDA_DEVICE_ORDER", "PCI_BUS_ID") != 0) {
        throw std::runtime_error("cannot set CUDA_DEVICE_ORDER=PCI_BUS_ID");
    }
#else
    if (::setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID", 0) != 0) {
        throw std::runtime_error("cannot set CUDA_DEVICE_ORDER=PCI_BUS_ID");
    }
#endif
}

// Creates the context of the primary device (`options.device`), then one for each further
// `options.devices` entry, in that order and before anything allocates: each constructor applies
// the CUDA synchronization schedule to its device (cudaSetDeviceFlags). A repeated ordinal gets a
// context of its own, with its own streams on the same GPU. With more than one device every
// context is logged in creation order. Returns the primary, which is then the calling thread's
// device; `extra` (empty on entry) receives the others.
inline DeviceContext initialize_devices(const EngineOptions& options,
                                        std::vector<std::unique_ptr<DeviceContext>>& extra) {
    if ((!options.devices.empty() && options.devices.front() != options.device) || !extra.empty()) {
        throw std::logic_error("initialize_devices needs devices to begin with the primary device");
    }
    StartupPhaseScope phase(options.startup_observer, StartupPhase::CudaInitialize);
    DeviceContext primary(options.device);
    if (options.devices.size() > 1) {
        extra.reserve(options.devices.size() - 1);
        for (std::size_t i = 1; i < options.devices.size(); ++i) {
            extra.push_back(std::make_unique<DeviceContext>(options.devices[i]));
        }
        const auto describe = [&](std::size_t index, const DeviceContext& context) {
            const double gib = static_cast<double>(context.props.totalGlobalMem) /
                               static_cast<double>(1ULL << 30);
            publish_diagnostic(options.diagnostic_observer, DiagnosticLevel::Info,
                               "device %zu of %zu: CUDA %d, %s, PCI %04x:%02x:%02x.0, %.1f GiB, %s "
                               "synchronization",
                               index, options.devices.size(), context.device, context.props.name,
                               static_cast<unsigned>(context.props.pciDomainID),
                               static_cast<unsigned>(context.props.pciBusID),
                               static_cast<unsigned>(context.props.pciDeviceID), gib,
                               context.sync_mode());
        };
        describe(0, primary);
        for (std::size_t i = 0; i < extra.size(); ++i) { describe(i + 1, *extra[i]); }
        // Every constructor (and sync_mode) selected its own device.
        primary.bind_to_current_thread();
    }
    phase.complete();
    return primary;
}

// The device segments of a layer split (models::LoadOptions::segment_ends) of a model of `layers`
// decoder layers: none on one device; on two, [0, K) on the first and [K, L) on the second for
// --layer-split K. --layer-split auto (the default with two devices) and @FILE need the split
// planner, which does not exist yet. Throws std::invalid_argument, in the CLI's terms, for those
// and for a K that leaves a device without a layer (validate_engine_options cannot check K < L:
// the layer count is the artifact's).
[[nodiscard]] inline std::vector<std::uint32_t> layer_split_ends(const EngineOptions& options,
                                                                std::uint32_t layers) {
    if (options.devices.size() < 2) { return {}; }
    if (options.layer_split_auto || !options.layer_split_plan.empty()) {
        throw std::invalid_argument(
            std::string("--layer-split ") + (options.layer_split_auto ? "auto" : "@FILE") +
            " needs the split planner, which is not available yet: pass --layer-split K");
    }
    if (!options.layer_split) {
        throw std::invalid_argument("a layer split across devices needs --layer-split K");
    }
    const std::uint32_t k = *options.layer_split;
    if (k == 0 || k >= layers) {
        throw std::invalid_argument("--layer-split K must leave each device at least one of the "
                                    "model's " + std::to_string(layers) + " layers (1 <= K < " +
                                    std::to_string(layers) + ")");
    }
    return {k, layers};
}

} // namespace infernix::runtime
