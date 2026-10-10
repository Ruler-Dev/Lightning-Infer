#include "runtime/engine/qwen4_exp_instance.h"

#include "artifact/reader.h"
#include "core/device.h"
#include "core/device_guard.h"
#include "core/host_memory.h"
#include "core/startup.h"
#include "core/vram_budget.h"
#include "models/load_options.h"
#include "models/qwen4_exp/load.h"
#include "models/qwen4_exp/memory_plan.h"
#include "models/qwen4_exp/program/ngram_volume.h"
#include "models/qwen4_exp/program/route_trace.h"
#include "models/qwen4_exp/program/split_capabilities.h"
#include "models/qwen4_exp/program/vram_monitor.h"
#include "models/qwen4_exp/segment_map.h"
#include "models/registry.h"
#include "runtime/engine/device_set.h"
#include "runtime/engine/model_instance.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <iterator>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace infernix::runtime {
namespace {

// The pinned Host tier of Qwen3.8-Flash-Next's prefix cache when --host-context-mib is not given.
// Planning allowance for the Program's pinned and mapped buffers (I/O staging, sampling, the CPU
// expert service's mapped records); the measured total is well below it.
constexpr std::uint64_t kProgramPinnedAllowance = 256ULL << 20;
// The materializer's upload slots and direct-read bounce buffer during the load.
constexpr std::uint64_t kLoadStagingBytes = 4ULL * (64ULL << 20) + (8ULL << 20);
// Below this many expert frames most routed experts miss the VRAM cache and decode slows down
// markedly (design §19.2).
constexpr std::uint32_t kFewFrames = 2048;

unsigned long long mib(std::uint64_t bytes) { return static_cast<unsigned long long>(bytes >> 20); }

// The fixed device allocations, largest first in the order a user can act on them.
std::string device_contributors(std::uint64_t dense, const models::qwen4_exp::ProgramDevicePlan& plan) {
    char kv[64];
    if (plan.kv_max > plan.kv) {
        std::snprintf(kv, sizeof(kv), "%llu (grows to %llu from the expert cache)", mib(plan.kv), mib(plan.kv_max));
    } else {
        std::snprintf(kv, sizeof(kv), "%llu", mib(plan.kv));
    }
    char text[320];
    std::snprintf(text, sizeof(text),
                  "dense weights %llu MiB, KV %s, workspace %llu, expert staging %llu, state and io %llu, expert "
                  "tables %llu",
                  mib(dense), kv, mib(plan.workspace), mib(plan.staging), mib(plan.state + plan.io),
                  mib(plan.residency));
    return text;
}

// A device's part of the engine: the dense arenas of its segments, the Program's fixed allocations
// on it (its segments' plans summed: two segments of the same-device test mode share a device), its
// VRAM source with its readings, and the frames its segments' expert caches got.
struct DeviceShare {
    DeviceContext* context = nullptr;
    std::uint64_t dense    = 0;
    std::uint32_t layers   = 0;
    models::qwen4_exp::ProgramDevicePlan plan;
    std::unique_ptr<VramBudgetSource> vram;
    VramSnapshot before;
    std::optional<SpillGuard> guard;
    std::uint64_t after_weights = 0, after_startup = 0;
    std::uint32_t frames = 0;
    std::optional<models::qwen4_exp::VramSizing> sizing; // its first segment's
};

void add_plan(models::qwen4_exp::ProgramDevicePlan& sum, const models::qwen4_exp::ProgramDevicePlan& part) {
    sum.kv_pages = part.kv_pages; // every segment indexes the same pages
    sum.kv += part.kv;
    sum.kv_max += part.kv_max;
    sum.workspace += part.workspace;
    sum.staging += part.staging;
    sum.state += part.state;
    sum.io += part.io;
    sum.residency += part.residency;
    sum.expert_record_stride = part.expert_record_stride;
    sum.max_frames += part.max_frames;
    sum.graph_bound += part.graph_bound;
}

void report(const EngineOptions& options, DiagnosticLevel level, const std::string& text) {
    if (options.diagnostic_observer.callback) {
        try {
            options.diagnostic_observer.callback(Diagnostic{.level = level, .message = text});
        } catch (...) {}
    } else if (level != DiagnosticLevel::Debug) {
        std::fprintf(stderr, "[engine] %s\n", text.c_str());
    }
}

void validate(const EngineOptions& options) {
    if (options.purpose != EnginePurpose::Generation) {
        throw std::invalid_argument("Qwen3.8-Flash-Next serves generation only (causal scoring is not wired yet)");
    }
    if ((options.speculative.backend != SpeculativeBackend::None &&
         options.speculative.backend != SpeculativeBackend::Mtp) ||
        options.speculative.ngram_archive_bytes != 0) {
        throw std::invalid_argument("Qwen3.8-Flash-Next speculates with its MTP drafter and n-gram copy proposals");
    }
    // Its loader binds the stored activation permissions as recorded (canonical W4A4 experts); a
    // 16-bit-activation reference route is not built, so the option is refused, not ignored.
    if (options.a16_activations) {
        throw std::invalid_argument("Qwen3.8-Flash-Next has no 16-bit-activation reference route (a16_activations)");
    }
    if (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("Engine max_concurrency must be in [1,8]");
    }
}

// One Parameters per device segment of the model, each built under its segment's device (where its
// expert scales go).
std::vector<std::unique_ptr<const models::qwen4_exp::execution::Parameters>>
segment_parameters(const models::qwen4_exp::Model& model) {
    namespace execution = models::qwen4_exp::execution;
    std::vector<std::unique_ptr<const execution::Parameters>> out;
    for (const models::qwen4_exp::SegmentSpec& segment : model.segments().segments()) {
        const DeviceGuard guard(segment.device);
        out.push_back(std::make_unique<const execution::Parameters>(
            model, segment.layers, execution::segment_role(segment)));
    }
    return out;
}

} // namespace

bool artifact_is_qwen4_exp(const std::filesystem::path& artifact) {
    const artifact::Reader reader(artifact);
    return models::qwen4_exp::is_qwen4_exp(reader);
}

Qwen4ExpInstance::Qwen4ExpInstance(std::unique_ptr<models::qwen4_exp::Model> source, const EngineOptions& options)
    : model(std::move(source)), parameters(segment_parameters(*model)),
      frontend(models::qwen3_5::make_frontend(model->resources(),
                                              {.chat_template_path      = options.chat_template_path,
                                               .architecture            = models::Architecture::Qwen4Exp,
                                               .vision_enabled          = options.enable_vision,
                                               .max_context             = options.max_context,
                                               .media_cache_bytes       = options.media_cache_bytes,
                                               .media_live_bytes        = options.media_live_bytes,
                                               .media_preprocess_threads = options.media_preprocess_threads,
                                               .vision_max_merged_tokens = options.vision_max_merged_tokens,
                                               .thinking_budget_message = options.thinking_budget_message})),
      capacity(options.max_context) {}

Qwen4ExpInstance::~Qwen4ExpInstance() = default;

const models::qwen4_exp::execution::Parameters& Qwen4ExpInstance::primary() const {
    return *parameters.front();
}

ConstructedQwen4Exp construct_qwen4_exp(const EngineOptions& options,
                                        std::span<DeviceContext* const> devices) {
    validate(options);
    bool matches = devices.size() == std::max<std::size_t>(options.devices.size(), 1);
    for (std::size_t i = 0; matches && i < devices.size(); ++i) {
        const int ordinal = options.devices.empty() ? options.device : options.devices[i];
        matches           = devices[i] != nullptr && devices[i]->device == ordinal;
    }
    if (!matches) {
        throw std::logic_error(
            "construct_qwen4_exp needs one device context per device of its options, in order");
    }
    // validate_engine_options refuses a second device before its context exists.
    if (devices.size() > 1 && !models::qwen4_exp::kSplitExecution) {
        throw std::logic_error("a Qwen3.8-Flash-Next layer split cannot be constructed yet");
    }
    DeviceContext& device = *devices.front();
    const auto start = std::chrono::steady_clock::now();
    StartupPhaseScope inspect(options.startup_observer, StartupPhase::ArtifactInspect);
    const artifact::Reader reader(options.artifact_path);
    if (!models::qwen4_exp::is_qwen4_exp(reader)) {
        throw std::invalid_argument("artifact is not Qwen4ExpForCausalLM");
    }
    inspect.complete();
    auto load = models::load_options(options, models::Architecture::Qwen4Exp);
    auto plan = models::qwen4_exp::plan_load(reader, load);
    // A layer split's device segments, once the artifact gives the layer count.
    load.segment_ends = layer_split_ends(options, plan.config().text.num_hidden_layers);
    if (!load.segment_ends.empty()) { plan = models::qwen4_exp::plan_load(reader, load); }
    // The RAM ledger (design §19.3.7): the experts take what the reserve and every other planned
    // allocation leave, before anything is pinned or read.
    models::qwen4_exp::HostMemoryDemand demand;
    demand.reserve      = options.ram_headroom_bytes;
    demand.expert_banks = plan.pinned_expert_bytes();
    demand.other_pinned = plan.pinned_other_bytes();
    demand.later_pinned = kProgramPinnedAllowance;
    // The prefix cache pins its whole Host tier at startup, beside the experts.
    if (options.context_cache.enabled && options.context_cache.mode == ContextCacheMode::Hybrid) {
        demand.prefix_cache = options.context_cache.host_capacity_bytes.value();
    }
    demand.pageable     = models::qwen4_exp::NgramVolume::cache_bytes(plan.config().text.ple.table);
    demand.load_staging = kLoadStagingBytes;
    demand.expert_cap   = options.expert_ram_bytes;
    const auto ledger   = models::qwen4_exp::plan_host_memory(host_memory_snapshot(), demand);
    report(options, DiagnosticLevel::Info, ledger.summary());
    report(options, DiagnosticLevel::Debug, ledger.describe());
    if (ledger.cap_too_large) {
        throw std::runtime_error("--expert-ram-mib " + std::to_string(*options.expert_ram_bytes >> 20) +
                                 " asks for more Host RAM than is free now (" + ledger.describe() +
                                 "); lower it, use auto, or free Host memory");
    }
    // The SSD tier (design §19.3.7): the banks stay in the artifact; the Program pins RAM slots for
    // the experts the ledger leaves room for (less their lock overhead) and reads the rest on use.
    const bool tier = ledger.placement == models::qwen4_exp::ExpertPlacement::Tier;
    if (tier && devices.size() > 1 && !models::qwen4_exp::kSplitExpertTier) {
        throw std::invalid_argument(
            "a layer split across devices needs every expert in Host RAM (it has no SSD expert "
            "tier yet), and the RAM ledger leaves " +
            std::to_string(mib(ledger.expert_ram)) + " of the " +
            std::to_string(mib(ledger.full_need)) +
            " MiB they need: free Host RAM, lower --ram-headroom-mib or raise --expert-ram-mib");
    }
    if (tier) {
        auto streamed           = load;
        streamed.stream_experts = true;
        plan                    = models::qwen4_exp::plan_load(reader, streamed);
    }
    // A small reserve trims other programs' working sets; keep Infernix's own pageable memory
    // resident so its heap and driver pages never hard-fault from the page file.
    if (const auto refused = reserve_process_working_set(demand.process_growth); !refused.empty()) {
        report(options, DiagnosticLevel::Warning,
               "Windows would not guarantee Infernix " + std::to_string(demand.process_growth >> 20) +
                   " MiB of resident working memory (" + refused +
                   "); everything still works, but if RAM runs short Windows may move some of it to the "
                   "page file and responses slow down");
    }
    plan.set_host_reserve(options.ram_headroom_bytes, demand.later_pinned);

    models::qwen4_exp::ProgramOptions program_options;
    program_options.max_context     = options.max_context;
    program_options.max_concurrency = options.max_concurrency;
    program_options.prefill_chunk   = options.prefill_chunk;
    program_options.kv_cache        = options.kv_cache;
    program_options.diagnostics     = options.diagnostic_observer;
    program_options.kv_capacity_tokens =
        options.kv_capacity.mode == KvCapacityMode::Explicit ? options.kv_capacity.explicit_tokens : 0U;
    program_options.ngram_draft_tokens = options.speculative.ngram_draft_tokens;
    program_options.ngram_min_match    = options.speculative.ngram_min_match;
    program_options.mtp_draft_tokens   = options.speculative.backend == SpeculativeBackend::Mtp
                                             ? options.speculative.draft_tokens
                                             : 0U;
    if (!options.expert_state_path.empty()) {
        // The state belongs to the artifact: its experts and their records, not the build or KV format.
        std::error_code error;
        const auto size = std::filesystem::file_size(options.artifact_path, error);
        const auto time = std::filesystem::last_write_time(options.artifact_path, error);
        program_options.expert_state          = options.expert_state_path;
        program_options.expert_state_identity = "artifact=" + std::filesystem::absolute(options.artifact_path).string() +
                                                ";size=" + std::to_string(error ? 0U : size) +
                                                ";mtime=" + std::to_string(error ? 0 : time.time_since_epoch().count());
    }
    program_options.ngram_volume = options.ngram_volume_path.empty()
                                       ? models::qwen4_exp::default_ngram_volume(options.artifact_path)
                                       : options.ngram_volume_path;
    // The hybrid prefix cache (design §19.3.1). Host-born snapshots live in the pinned slab pool;
    // the cost model ranks resume sources by the measured prefill coefficients.
    const ContextCacheOptions& cache = options.context_cache;
    if (cache.enabled && cache.mode == ContextCacheMode::Hybrid) {
        const std::uint32_t chunk         = options.prefill_chunk;
        program_options.prefix_cache      = true;
        // normalize_engine_options resolved every value for this architecture.
        program_options.prefix_host_bytes          = cache.host_capacity_bytes.value();
        program_options.prefix_taps.max_new_taps   = cache.hybrid.max_new_taps.value();
        program_options.prefix_taps.ladder_tokens  = cache.hybrid.tap_ladder_tokens.value();
        program_options.prefix_taps.min_gap_tokens = cache.hybrid.tap_min_gap_tokens.value();
        // Fitted 2026-10-05 (RTX 5090 at PCIe 5.0 x8, INT8 KV; design §19.4): 1.67 s per layer-walk
        // span, 0.16 s per further call, 22 us per token and 1.06 ns per attention pair, within 5 %
        // from 4K to 128K. The span is one pass of the experts over the link: the Program rescales
        // it, and the restore rate, to the link it measures at startup.
        program_options.prefix_cost.chunk_seconds          = 0.1645;
        program_options.prefix_cost.chunk_tokens           = chunk;
        program_options.prefix_cost.token_seconds          = 2.23e-5;
        program_options.prefix_cost.attention_pair_seconds = 1.06e-9;
        program_options.prefix_cost.call_route_fraction    = 10.0 / 512.0;
        program_options.prefix_span_seconds                = 1.67;
        program_options.prefix_cost.h2d_bytes_per_second   = 27.5e9; // the reference link, until measured
        program_options.prefix_cost.transfer_batch_seconds = 20.0e-6;
        program_options.prefix_save                        = cache.hybrid.persistent_save;
        // A request waiting for a sibling's snapshot stays in the FIFO, so the predicted wait is
        // kept well inside its queue timeout.
        program_options.prefix_coalesce_wait_seconds = static_cast<double>(options.pending_timeout_ms) / 1000.0 / 2.0;
    }
    program_options.route_trace   = models::qwen4_exp::testing::route_trace();
    program_options.vram_grow_delay_seconds = models::qwen4_exp::testing::vram_grow_delay();
    program_options.vram_headroom = options.vram_headroom_bytes;
    program_options.vram_past_budget = options.vram_past_budget;
    if (tier) {
        program_options.expert_ram_bytes =
            static_cast<std::uint64_t>(static_cast<double>(ledger.expert_ram) / (1.0 + demand.lock_overhead));
    }

    // A layer split's handoffs (decode and verification rounds): Auto is the event handoff until the
    // gate's soak passes, so the gate stays opt-in.
    program_options.handoff            = options.split_handoff == SplitHandoff::Gate ? SplitHandoff::Gate
                                                                                    : SplitHandoff::Event;
    program_options.handoff_timeout_ms = options.handoff_timeout_ms;
    program_options.logits_hash        = models::qwen4_exp::testing::logits_hash();

    // The model's device segments (one without a split) and, per device, its segments' dense arenas
    // and the Program's fixed allocations on it (two segments of the same-device test mode share
    // one device).
    const models::qwen4_exp::Config& config = plan.config();
    std::vector<int> ordinals;
    for (const DeviceContext* context : devices) { ordinals.push_back(context->device); }
    const models::qwen4_exp::SegmentMap segments =
        load.segment_ends.empty()
            ? models::qwen4_exp::SegmentMap::single(config.text, config.mtp, device.device)
            : models::qwen4_exp::SegmentMap::contiguous(config.text, config.mtp, ordinals, load.segment_ends);
    std::vector<DeviceShare> shares;
    for (const models::qwen4_exp::SegmentSpec& segment : segments.segments()) {
        auto share = std::find_if(shares.begin(), shares.end(),
                                  [&](const DeviceShare& s) { return s.context->device == segment.device; });
        if (share == shares.end()) {
            shares.push_back(DeviceShare{.context = devices[segment.index]});
            share = std::prev(shares.end());
        }
        const auto segment_plan = models::qwen4_exp::Program::plan_segment(
            program_options, config, static_cast<std::int32_t>(plan.public_token_count()), segment);
        share->dense += plan.device_bytes(segment.index);
        share->layers += segment.layers.size();
        add_plan(share->plan, segment_plan);
    }
    const bool split = segments.size() > 1;
    // The VRAM check (design §19.3.7) of each device, before the weights are read: the dense weights,
    // the Program's fixed allocations, the reserve for graph executables and the display headroom
    // must leave room for the expert cache.
    for (DeviceShare& share : shares) {
        share.vram   = open_vram_budget_source(share.context->device, options.vram_past_budget);
        share.before = share.vram->query();
        models::qwen4_exp::VramDemand vram_demand;
        vram_demand.headroom    = options.vram_headroom_bytes;
        vram_demand.graphs      = share.plan.graph_bound;
        vram_demand.fixed       = share.dense + share.plan.fixed_bytes();
        vram_demand.frame_bytes = share.plan.expert_record_stride;
        vram_demand.max_frames  = share.plan.max_frames;
        const auto check        = models::qwen4_exp::size_expert_frames(share.before, vram_demand);
        const std::string where = split ? " of GPU " + std::to_string(share.context->device) : std::string();
        if (check.frame_bytes < 0) {
            throw std::runtime_error(
                "Qwen3.8-Flash-Next does not fit in device memory" + where + ": " +
                std::to_string(mib(share.before.device_free)) + " MiB free; " +
                device_contributors(share.dense, share.plan) + "; reserve " + std::to_string(mib(check.reserve)) +
                "; headroom " + std::to_string(mib(check.headroom)) + " (" + check.describe() +
                "). Lower --max-context, --max-concurrency or --prefill-chunk, use --kv-dtype int8, lower "
                "--vram-headroom-mib, " + (split ? "move the layer split, " : "") + "or free VRAM");
        }
        // A device's frames serve its layers: the recommended count shrinks with them.
        const std::uint32_t few =
            static_cast<std::uint32_t>(static_cast<std::uint64_t>(kFewFrames) * share.layers / config.text.num_hidden_layers);
        if (check.frames < few) {
            report(options, DiagnosticLevel::Warning,
                   "decode will be slow: only " + std::to_string(check.frames) + " expert frames fit in VRAM" + where +
                       " (" + std::to_string(few) + " or more recommended); lower --max-context or "
                       "--max-concurrency, " + (split ? "move the layer split, " : "") + "or free VRAM");
        }
    }

    // The dense arenas must land in device memory (WDDM can place them in system memory silently).
    for (DeviceShare& share : shares) {
        share.guard.emplace(*share.vram);
        share.guard->begin();
    }
    auto model =
        models::qwen4_exp::materialize_model(std::move(plan), devices, &options.startup_observer);
    for (DeviceShare& share : shares) {
        const DeviceGuard on(share.context->device);
        share.context->synchronize();
        if (const std::uint64_t spilled = share.guard->end(share.dense); spilled != 0) {
            throw std::runtime_error(std::to_string(mib(spilled)) +
                                     " MiB of the dense weights were placed in shared system memory" +
                                     (split ? " of GPU " + std::to_string(share.context->device) : std::string()) +
                                     " by the driver's sysmem fallback: free VRAM and start again");
        }
        share.after_weights = share.vram->query().device_free;
    }

    StartupPhaseScope frontend(options.startup_observer, StartupPhase::FrontendInitialize);
    auto instance = std::make_unique<Qwen4ExpInstance>(std::move(model), options);
    frontend.complete();

    StartupPhaseScope program(options.startup_observer, StartupPhase::ProgramInitialize);
    std::vector<const models::qwen4_exp::execution::Parameters*> parameters;
    for (const auto& segment : instance->parameters) { parameters.push_back(segment.get()); }
    instance->program = std::make_unique<models::qwen4_exp::Program>(
        parameters, devices, instance->model->segments(), std::move(program_options));
    const std::span<const models::qwen4_exp::VramSizing> sizing = instance->program->segment_vram_sizing();
    for (const models::qwen4_exp::SegmentSpec& segment : segments.segments()) {
        DeviceShare& share = *std::find_if(shares.begin(), shares.end(), [&](const DeviceShare& s) {
            return s.context->device == segment.device;
        });
        share.frames += sizing[segment.index].frames;
        if (!share.sizing) { share.sizing = sizing[segment.index]; }
    }
    const MemorySummary memory = instance->program->memory_summary();
    auto& resolution     = instance->kv_capacity_resolution;
    resolution.mode                     = options.kv_capacity.mode;
    resolution.main_page_groups         = memory.kv_capacity_page_groups;
    resolution.maximum_main_page_groups = memory.kv_capacity_max_page_groups;
    resolution.resolved_tokens          = memory.kv_capacity;
    // A split's devices are summed.
    for (DeviceShare& share : shares) {
        const std::uint64_t frames_bytes = static_cast<std::uint64_t>(share.frames) * share.plan.expert_record_stride;
        share.after_startup              = share.vram->query().device_free;
        const VramSnapshot& before       = share.before;
        const char* display = before.display == DisplayState::Headless   ? "no display"
                              : before.display == DisplayState::Attached ? "display attached"
                                                                         : "display unknown";
        // --vram-past-budget: the allowance already counted as free.
        const std::string past = before.budget_allowance == 0
                                     ? std::string()
                                     : " (" + std::to_string(mib(before.budget_allowance)) + " MiB past the OS budget)";
        const std::string gpu = split ? " (GPU " + std::to_string(share.context->device) + ")" : std::string();
        char line[704];
        std::snprintf(line, sizeof(line),
                      "VRAM ledger%s: %llu MiB card, %s, %llu in use before loading%s; %s; reserve %llu (%u graphs); "
                      "headroom %llu (%s); expert frames %u (%.2f GiB); %llu free after startup",
                      gpu.c_str(), mib(before.device_total), display, mib(before.device_total - before.device_free),
                      past.c_str(), device_contributors(share.dense, share.plan).c_str(), mib(share.sizing->reserve),
                      share.plan.graph_bound, mib(share.sizing->headroom), options.vram_headroom_bytes ? "set" : "auto",
                      share.frames, static_cast<double>(frames_bytes) / static_cast<double>(1ULL << 30),
                      mib(share.after_startup));
        report(options, DiagnosticLevel::Debug, line);
        const double gib = static_cast<double>(1ULL << 30);
        char summary[320];
        if (share.plan.kv_max > share.plan.kv) {
            std::snprintf(summary, sizeof(summary),
                          "VRAM%s: model %.2f GiB, KV cache %.2f GiB (grows to %.2f GiB as contexts lengthen, taking room "
                          "from the expert cache), expert cache %u experts (%.1f GiB)",
                          gpu.c_str(), static_cast<double>(share.dense) / gib, static_cast<double>(share.plan.kv) / gib,
                          static_cast<double>(share.plan.kv_max) / gib, share.frames,
                          static_cast<double>(frames_bytes) / gib);
        } else {
            std::snprintf(summary, sizeof(summary),
                          "VRAM%s: model %.2f GiB, KV cache %.2f GiB, expert cache %u experts (%.1f GiB)", gpu.c_str(),
                          static_cast<double>(share.dense) / gib, static_cast<double>(share.plan.kv) / gib,
                          share.frames, static_cast<double>(frames_bytes) / gib);
        }
        report(options, DiagnosticLevel::Info, summary);
        resolution.runtime_reservation_bytes += share.plan.fixed_bytes() + frames_bytes;
        resolution.available_after_weights_bytes += share.after_weights;
        resolution.available_after_startup_bytes += share.after_startup;
        resolution.automatic_headroom_bytes += share.sizing->headroom;
    }
    resolution.planned_slack_bytes =
        resolution.available_after_weights_bytes > resolution.runtime_reservation_bytes
            ? resolution.available_after_weights_bytes - resolution.runtime_reservation_bytes
            : 0;
    LoadSummary::PrefixCacheRestore restore;
    if (const std::filesystem::path& file = options.context_cache.hybrid.persistent_file;
        options.context_cache.enabled && options.context_cache.mode == ContextCacheMode::Hybrid && !file.empty()) {
        const models::qwen4_exp::PrefixCachePersistence loaded = instance->program->attach_prefix_cache_file(
            file, hybrid_cache_fingerprint(options, "qwen4_exp"), options.startup_observer);
        restore = LoadSummary::PrefixCacheRestore{
            .attempted           = true,
            .restored            = loaded.ok,
            .message             = loaded.message,
            .blocks              = loaded.blocks,
            .snapshots           = loaded.snapshots,
            .bytes               = loaded.bytes,
            .seconds             = loaded.seconds,
            .saved_blocks        = loaded.saved_blocks,
            .saved_snapshots     = loaded.saved_snapshots,
            .required_host_bytes = loaded.required_host_bytes,
            .host_bytes          = loaded.host_bytes,
        };
    }
    program.complete();

    const auto& stats = instance->model->storage_stats();
    const auto& text  = instance->model->config().text;
    LoadSummary summary;
    summary.architecture         = std::string(models::architecture_name(models::Architecture::Qwen4Exp));
    summary.model_name           = instance->model->info().name;
    summary.load_seconds         = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    summary.upload_seconds       = stats.upload_seconds;
    summary.artifact_bytes_read  = stats.read_bytes;
    summary.host_to_device_bytes = stats.h2d_bytes;
    summary.peak_staging_bytes   = stats.peak_staging_bytes;
    summary.device_object_count  = stats.device_object_count;
    summary.host_object_count    = stats.host_object_count;
    summary.weight_formats       = {"bf16", "fp32", "nvfp4_mul"};
    summary.prefix_cache         = std::move(restore);

    ModelMetadata metadata;
    metadata.model_id       = instance->model->info().name;
    metadata.vocab_size     = text.vocab_size;
    metadata.embedding_size = text.hidden_size;
    metadata.native_context = text.max_position_embeddings;
    metadata.weights_id     = "nvfp4_mul";
    metadata.weight_bytes   = stats.device_capacity_bytes + stats.pinned_bytes;

    EngineOptions resolved = options;
    return {std::move(instance), std::move(summary), std::move(metadata), std::move(resolved)};
}

} // namespace infernix::runtime
