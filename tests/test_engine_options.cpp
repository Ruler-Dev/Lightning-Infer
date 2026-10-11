// Engine option combinations the Engine rejects before it reads any artifact.
#include "infernix/engine.h"
#include "models/qwen4_exp/program/split_capabilities.h"
#include "models/registry.h"
#include "runtime/engine/model_instance.h"

#include <cuda_runtime.h>

#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

// The exception an Engine over a nonexistent artifact raises for these options.
std::string construction_error(infernix::EngineOptions options, bool& invalid_argument,
                               const char* artifact = "absent-artifact-for-option-validation.ninfer") {
    options.artifact_path = artifact;
    try {
        infernix::Engine engine(std::move(options));
    } catch (const std::invalid_argument& error) {
        invalid_argument = true;
        return error.what();
    } catch (const std::exception& error) {
        invalid_argument = false;
        return error.what();
    }
    invalid_argument = false;
    return {};
}

// What validate_engine_options refuses for this architecture (empty: accepted).
std::string validation_error(const infernix::EngineOptions& options,
                             infernix::models::Architecture architecture) {
    try {
        infernix::runtime::validate_engine_options(options, architecture);
    } catch (const std::invalid_argument& error) { return error.what(); }
    return {};
}

bool contains(const std::string& text, const char* part) {
    return text.find(part) != std::string::npos;
}

// The device set and the layer-split options. Every refusal comes before any device context
// exists; a Qwen4Exp split is refused while split_capabilities.h has not enabled it.
int check_device_options() {
    using infernix::models::Architecture;
    namespace split = infernix::models::qwen4_exp;
    int failures       = 0;
    const auto refusal = [](infernix::EngineOptions options) {
        bool invalid_argument  = false;
        const std::string what = construction_error(std::move(options), invalid_argument);
        return invalid_argument ? what : std::string("(accepted) ") + what;
    };
    {
        // One --devices entry is the device alone: validation passes and the artifact fails.
        infernix::EngineOptions single;
        single.devices         = {0};
        const std::string what = refusal(single);
        failures += check(!contains(what, "device") && !contains(what, "split"),
                          "devices {device} was refused");
    }
    infernix::EngineOptions options;
    options.devices = {1};
    failures += check(contains(refusal(options), "begin with device"),
                      "devices not beginning with the primary device were accepted");
    options         = infernix::EngineOptions{};
    options.device  = -1;
    options.devices = {-1, 0};
    failures +=
        check(contains(refusal(options), "nonnegative"), "a negative device ordinal was accepted");
    options.device  = 0;
    options.devices = {0, 0, 0};
    failures +=
        check(contains(refusal(options), "at most two devices"), "three devices were accepted");
    options.devices = {0, 0};
    failures += check(contains(refusal(options), "needs a readable Qwen3.8-Flash-Next artifact"),
                      "a Qwen3.5 layer split was accepted");
    // The split's own options need two devices.
    for (int option = 0; option < 6; ++option) {
        infernix::EngineOptions one;
        switch (option) {
        case 0: one.layer_split = 24U; break;
        case 1: one.layer_split_auto = true; break;
        case 2: one.layer_split_plan = "plan.json"; break;
        case 3: one.layer_split_report = true; break;
        case 4: one.split_handoff = infernix::SplitHandoff::Event; break;
        default: one.handoff_timeout_ms = 500; break;
        }
        failures += check(contains(refusal(one), "needs two devices"),
                          "a layer-split option was accepted on one device");
        one.devices = {0};
        failures += check(contains(refusal(one), "needs two devices"),
                          "a layer-split option was accepted with one --devices entry");
    }
    infernix::EngineOptions two;
    two.devices          = {0, 0};
    two.layer_split      = 24U;
    two.layer_split_auto = true;
    failures += check(contains(refusal(two), "select one layer split"),
                      "two layer-split forms were accepted together");
    two.layer_split_auto = false;
    two.layer_split      = 0U;
    failures += check(contains(refusal(two), "K >= 1"), "--layer-split 0 was accepted");
    two.layer_split = 24U;
    for (const std::uint32_t milliseconds : {0U, infernix::kMaximumHandoffTimeoutMs + 1U}) {
        two.handoff_timeout_ms = milliseconds;
        failures += check(contains(refusal(two), "--handoff-timeout-ms must be in"),
                          "an out-of-range handoff timeout was accepted");
    }

    // Qwen4Exp, without its artifact: the split's capabilities.
    infernix::EngineOptions qwen4;
    qwen4.artifact_path         = "absent-artifact.infernix";
    qwen4.devices               = {1, 0};
    qwen4.device                = 1;
    qwen4.context_cache.enabled = false;
    if (!split::kSplitExecution) {
        const std::string refused = validation_error(qwen4, Architecture::Qwen4Exp);
        failures += check(contains(refused, "not available yet"),
                          "a Qwen4Exp layer split was accepted before its execution exists");
    } else {
        failures += check(validation_error(qwen4, Architecture::Qwen4Exp).empty(),
                          "a Qwen4Exp layer split was refused");
        if (!split::kSplitVision) {
            infernix::EngineOptions vision = qwen4;
            vision.enable_vision           = true;
            failures +=
                check(contains(validation_error(vision, Architecture::Qwen4Exp), "--vision"),
                      "a layer split accepted vision before it supports it");
        }
        // The gate handoff needs two GPUs; on one the event handoff runs.
        infernix::EngineOptions gated = qwen4;
        gated.split_handoff           = infernix::SplitHandoff::Gate;
        failures += check(validation_error(gated, Architecture::Qwen4Exp).empty(),
                          "a gated layer split across two GPUs was refused");
        gated.devices = {1, 1};
        failures += check(contains(validation_error(gated, Architecture::Qwen4Exp), "two different GPUs"),
                          "a gated layer split on one GPU was accepted");
        // The hybrid prefix cache runs across the split (each segment copies its own planes).
        static_assert(split::kSplitPrefixCache);
        infernix::EngineOptions cached = qwen4;
        cached.context_cache.enabled   = true;
        cached.context_cache.mode      = infernix::ContextCacheMode::Hybrid;
        failures += check(validation_error(cached, Architecture::Qwen4Exp).empty(),
                          "a layer split refused the hybrid prefix cache");
    }
    // Two devices without K or a plan file resolve to auto; one device never does.
    const auto normalized = [](infernix::EngineOptions engine_options) {
        return infernix::runtime::normalize_engine_options(std::move(engine_options),
                                                           Architecture::Qwen4Exp);
    };
    failures += check(normalized(qwen4).layer_split_auto,
                      "two devices without a layer split did not select auto");
    infernix::EngineOptions explicit_k = qwen4;
    explicit_k.layer_split             = 24U;
    infernix::EngineOptions planned    = qwen4;
    planned.layer_split_plan           = "plan.json";
    infernix::EngineOptions alone      = qwen4;
    alone.devices                      = {};
    failures += check(!normalized(explicit_k).layer_split_auto &&
                          normalized(explicit_k).layer_split == 24U &&
                          !normalized(planned).layer_split_auto &&
                          !normalized(alone).layer_split_auto,
                      "an explicit layer split or one device resolved to auto");
    return failures;
}

} // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = 0;
    infernix::EngineOptions options;
    // Both artifact extensions name the same container; any other is rejected before reading.
    for (const char* artifact : {"absent-artifact.infernix", "absent-artifact.ninfer"}) {
        bool invalid_argument  = false;
        const std::string what = construction_error(options, invalid_argument, artifact);
        failures += check(what.find("accepts only") == std::string::npos,
                          "an .infernix or .ninfer artifact path was rejected by its extension");
    }
    {
        bool invalid_argument  = false;
        const std::string what = construction_error(options, invalid_argument, "absent-artifact.gguf");
        failures += check(invalid_argument &&
                              what.find("accepts only .infernix and .ninfer") != std::string::npos,
                          "an artifact path with another extension was accepted");
    }
    // Without the opt-outs every KV format passes this check; INT8 and NVFP4 then take the fast
    // kernels.
    for (const auto storage :
         {infernix::KvCacheStorage::BFloat16, infernix::KvCacheStorage::Int8Group64,
          infernix::KvCacheStorage::Nvfp4Group16}) {
        options.kv_cache       = storage;
        bool invalid_argument  = false;
        const std::string what = construction_error(options, invalid_argument);
        failures += check(what.find("INT8 KV") == std::string::npos &&
                              what.find("NVFP4 KV") == std::string::npos,
                          "the default prefill kernel choice was rejected");
    }
    options.original_int8_prefill_kernel = true;
    for (const auto storage : {infernix::KvCacheStorage::BFloat16, infernix::KvCacheStorage::Fp8E4M3Row256,
                               infernix::KvCacheStorage::Nvfp4Group16}) {
        options.kv_cache       = storage;
        bool invalid_argument  = false;
        const std::string what = construction_error(options, invalid_argument);
        failures +=
            check(invalid_argument && what.find("INT8 KV") != std::string::npos,
                  "the original INT8 prefill kernel was accepted without the INT8 KV cache");
    }
    // With INT8 KV the options pass, so construction proceeds to (and fails on) the artifact.
    options.kv_cache       = infernix::KvCacheStorage::Int8Group64;
    bool invalid_argument  = false;
    const std::string what = construction_error(options, invalid_argument);
    failures += check(what.find("INT8 KV") == std::string::npos,
                      "the original INT8 prefill kernel was rejected with the INT8 KV cache");
    options.original_int8_prefill_kernel  = false;
    options.original_nvfp4_prefill_kernel = true;
    for (const auto storage : {infernix::KvCacheStorage::BFloat16, infernix::KvCacheStorage::Int8Group64,
                               infernix::KvCacheStorage::Fp8E4M3Row256}) {
        options.kv_cache       = storage;
        bool nvfp4_invalid     = false;
        const std::string text = construction_error(options, nvfp4_invalid);
        failures +=
            check(nvfp4_invalid && text.find("NVFP4 KV") != std::string::npos,
                  "the original NVFP4 prefill kernel was accepted without the NVFP4 KV cache");
    }
    options.kv_cache             = infernix::KvCacheStorage::Nvfp4Group16;
    bool nvfp4_invalid           = false;
    const std::string nvfp4_what = construction_error(options, nvfp4_invalid);
    failures += check(nvfp4_what.find("NVFP4 KV") == std::string::npos,
                      "the original NVFP4 prefill kernel was rejected with the NVFP4 KV cache");
    // 8-bit prefill P*V is chosen per KV format by default and every KV cache and prompt kernel
    // accepts either form.
    options.original_nvfp4_prefill_kernel = false;
    failures += check(infernix::EngineOptions{}.prefill_8bit_pv == infernix::PrefillPv8::Auto,
                      "8-bit prefill P*V must default to auto");
    for (const auto pv8 : {infernix::PrefillPv8::Auto, infernix::PrefillPv8::On, infernix::PrefillPv8::Off}) {
        options.prefill_8bit_pv = pv8;
        for (const auto storage :
             {infernix::KvCacheStorage::BFloat16, infernix::KvCacheStorage::Int8Group64,
              infernix::KvCacheStorage::Fp8E4M3Row256, infernix::KvCacheStorage::Nvfp4Group16,
              infernix::KvCacheStorage::Fp8KeyNvfp4Value, infernix::KvCacheStorage::Vq2,
              infernix::KvCacheStorage::Q4KeyVq2Value}) {
            for (const bool original : {false, true}) {
                options.kv_cache = storage;
                options.original_int8_prefill_kernel =
                    original && storage == infernix::KvCacheStorage::Int8Group64;
                options.original_nvfp4_prefill_kernel =
                    original && storage == infernix::KvCacheStorage::Nvfp4Group16;
                bool pv8_invalid       = false;
                const std::string text = construction_error(options, pv8_invalid);
                failures += check(text.find("P*V") == std::string::npos,
                                  "the 8-bit prefill P*V setting was rejected");
            }
        }
    }
    // The prompt split workspace is bounded and applies to every KV cache.
    options.original_int8_prefill_kernel  = false;
    options.original_nvfp4_prefill_kernel = false;
    options.prefill_8bit_pv               = infernix::PrefillPv8::Off;
    options.kv_cache                      = infernix::KvCacheStorage::BFloat16;
    for (const std::uint32_t mib : {0U, infernix::kMaximumPrefillSplitWorkspaceMiB}) {
        options.prefill_split_workspace_mib = mib;
        bool split_invalid                  = false;
        const std::string text              = construction_error(options, split_invalid);
        failures += check(text.find("prefill_split_workspace_mib") == std::string::npos,
                          "a prompt split workspace within [0,16384] MiB was rejected");
    }
    options.prefill_split_workspace_mib = infernix::kMaximumPrefillSplitWorkspaceMiB + 1;
    bool split_invalid                  = false;
    const std::string split_text        = construction_error(options, split_invalid);
    failures += check(split_invalid && split_text.find("prefill_split_workspace_mib") !=
                                           std::string::npos,
                      "a prompt split workspace above 16384 MiB was accepted");
    failures += check_device_options();
    std::cout << (failures == 0 ? "PASS" : "FAIL") << " Engine option validation\n";
    return failures == 0 ? 0 : 1;
}
