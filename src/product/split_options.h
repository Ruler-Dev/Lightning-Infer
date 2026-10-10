#pragma once

// The device and layer-split flags infernix and infernix-serve share: --devices, --layer-split,
// --split-handoff and --handoff-timeout-ms, parsed into the EngineOptions fields of the same names.
// The Engine checks how they combine (validate_engine_options).

#include "infernix/types.h"

#include <charconv>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace infernix::product {

// --devices A[,B]: one or two CUDA device ordinals, the primary first.
[[nodiscard]] inline std::vector<int> parse_devices(std::string_view text) {
    std::vector<int> devices;
    while (true) {
        const std::size_t comma     = text.find(',');
        const std::string_view item = text.substr(0, comma);
        const char* const last      = item.data() + item.size();
        int ordinal                 = 0;
        const auto [end, error]     = std::from_chars(item.data(), last, ordinal);
        if (item.empty() || error != std::errc{} || end != last || ordinal < 0 ||
            devices.size() == 2) {
            throw std::invalid_argument(
                "--devices takes one or two comma-separated CUDA device ordinals");
        }
        devices.push_back(ordinal);
        if (comma == std::string_view::npos) { break; }
        text.remove_prefix(comma + 1);
    }
    return devices;
}

// --layer-split K|auto|@FILE: exactly one of the three is set.
struct LayerSplit {
    std::optional<std::uint32_t> layer;
    bool automatic = false;
    std::filesystem::path plan;
};

[[nodiscard]] inline LayerSplit parse_layer_split(std::string_view text) {
    LayerSplit split;
    if (text == "auto") {
        split.automatic = true;
    } else if (text.starts_with('@')) {
        if (text.size() == 1) {
            throw std::invalid_argument("--layer-split @FILE needs a plan file");
        }
        split.plan = std::filesystem::path(std::string(text.substr(1)));
    } else {
        const char* const last  = text.data() + text.size();
        std::uint32_t layer     = 0;
        const auto [end, error] = std::from_chars(text.data(), last, layer);
        if (text.empty() || error != std::errc{} || end != last || layer == 0) {
            throw std::invalid_argument(
                "--layer-split takes a positive layer count K, auto or @FILE");
        }
        split.layer = layer;
    }
    return split;
}

[[nodiscard]] inline SplitHandoff parse_split_handoff(std::string_view text) {
    if (text == "auto") { return SplitHandoff::Auto; }
    if (text == "event") { return SplitHandoff::Event; }
    if (text == "gate") { return SplitHandoff::Gate; }
    throw std::invalid_argument("--split-handoff accepts auto, event or gate");
}

[[nodiscard]] inline std::uint32_t parse_handoff_timeout_ms(std::string_view text) {
    const char* const last     = text.data() + text.size();
    std::uint32_t milliseconds = 0;
    const auto [end, error]    = std::from_chars(text.data(), last, milliseconds);
    if (text.empty() || error != std::errc{} || end != last || milliseconds == 0 ||
        milliseconds > kMaximumHandoffTimeoutMs) {
        throw std::invalid_argument("--handoff-timeout-ms must be in [1," +
                                    std::to_string(kMaximumHandoffTimeoutMs) + "]");
    }
    return milliseconds;
}

} // namespace infernix::product
