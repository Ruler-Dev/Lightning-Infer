// A layer split on the real artifact (multi-GPU WP13, WP14): an Engine whose layers are split at K
// across --devices A,B (by default two contexts of device 0, the same-device test mode) prefills
// prompts, samples their first token and decodes bit for bit like the unsplit Engine on the
// reference device. Every call and round that computes logits writes the FNV-1a of the head's FP32
// logits (ProgramOptions::logits_hash, through testing::set_logits_hash), so the comparison covers
// each prefill call's logits, the first token, forced tokens' logits and every decode or
// verification round's logits, besides the generated ids.
//
// Cases, per K: raw-token prompts of 1, 255, 256 and 4113 tokens (two prefill calls at the default
// --prefill-chunk 4096: 4096 + 17), each generating one token; the 255-token prompt generating
// --decode tokens (decode rounds, or with MTP the drafter, verification rounds and their commits);
// a chat prompt with thinking whose budget of one token makes the Engine append the thinking-close
// control as forced tokens, then decode the answer; and a prompt cancelled while it prefills, after
// which a request still gives the reference's bits. With --long, a 40,000-token prompt also grows
// the KV pool past its 32K-token base, in lockstep on every segment with KV (the same page count
// each). With MTP (the default) the head's prefill calls also write the drafter's cells.
// --pair runs every Engine with two lanes and adds two requests decoding together (C = 2 pair
// rounds, with MTP two drafts at most): each one's ids equal the unsplit Engine's pair, and the
// 255-token one's the decoded request's at C = 1 (the pair's columns stay on batch-invariant routes).
// --handoff gate runs the split's rounds with the gate handoff (two GPUs only); --timeout then
// delays one round's first segment past the gate's bound (testing::delay_next_round): that request
// fails with a recoverable handoff timeout and the next one gives the reference's bits.
//
// --walk-modes runs each split twice, walking its long prompts layer-major on both segments (multi-GPU
// WP16, the default) and chunk-major (INFERNIX_SPLIT_WALK=0), and compares each with the unsplit run
// and with the default one: equal ids, and each run's logits lines either equal (the same calls
// computed logits) or, where a layer walk computed fewer (no logits before a span's last chunk),
// every line of the walking run found in order among the other's, the last ones equal. --prompt-decode N lets every prompt
// generate N tokens (default 1), so its decode rounds read every layer's state after the prompt.
// E.g. --prompts 12388,73828 --context 81920 --walk-modes --prompt-decode 8: 3 x 4096 + 100 tokens
// (one span, then a narrow call) and 18 x 4096 + 100 (two spans: 16 calls at kWalkMaxTokens, 2).
//
// Two GPUs: --devices 0,1 --reference 0 with INFERNIX_NUMERICS_SMS set to the smaller GPU's SM
// count (the same launch decompositions on both). Skips (77) unless INFERNIX_QWEN4_ARTIFACT names
// a Qwen4Exp artifact. Each Engine pins the expert banks (about 64 GiB of Host RAM); they run one
// after another.
//
//   INFERNIX_QWEN4_ARTIFACT=a.infernix [INFERNIX_QWEN4_NGRAM=a.ngram]
//   infernix_qwen4_exp_split_prefill_real_test [--devices A,B] [--reference D] [--splits K,...]
//       [--prompts N,...] [--decode N] [--chunk N] [--context N] [--plain] [--no-forced] [--no-cancel]
//       [--long] [--pair] [--handoff event|gate] [--timeout] [--walk-modes] [--prompt-decode N] [--work DIR]

#include "models/qwen4_exp/program/program.h"
#include "infernix/engine.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

namespace fs = std::filesystem;

struct Options {
    std::vector<int> devices{0, 0};
    std::optional<int> reference;
    std::vector<std::uint32_t> splits{1, 2, 24, 47};
    std::vector<std::uint32_t> prompts{1, 255, 256, 4113};
    std::uint32_t decode  = 48;
    std::uint32_t chunk   = 4096;
    std::uint32_t context = 8192;
    bool mtp              = true;
    bool forced           = true;
    bool cancel           = true;
    bool long_prompt      = false;
    bool gate             = false;
    bool timeout          = false;
    bool pair             = false;
    bool walk_modes       = false;
    std::uint32_t prompt_decode = 1;
    fs::path work         = fs::temp_directory_path() / "infernix_split_prefill";
};

std::vector<std::uint32_t> parse_list(const std::string& text) {
    std::vector<std::uint32_t> out;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ',')) { out.push_back(static_cast<std::uint32_t>(std::stoul(item))); }
    if (out.empty()) { throw std::invalid_argument("empty list: " + text); }
    return out;
}

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const auto value      = [&]() -> std::string {
            if (i + 1 >= argc) { throw std::invalid_argument(arg + " needs a value"); }
            return argv[++i];
        };
        if (arg == "--devices") {
            o.devices.clear();
            for (const std::uint32_t d : parse_list(value())) { o.devices.push_back(static_cast<int>(d)); }
            if (o.devices.size() != 2) { throw std::invalid_argument("--devices takes two ordinals"); }
        } else if (arg == "--reference") {
            o.reference = std::stoi(value());
        } else if (arg == "--splits") {
            o.splits = parse_list(value());
        } else if (arg == "--prompts") {
            o.prompts = parse_list(value());
        } else if (arg == "--decode") {
            o.decode = static_cast<std::uint32_t>(std::stoul(value()));
        } else if (arg == "--handoff") {
            const std::string mode = value();
            if (mode != "event" && mode != "gate") { throw std::invalid_argument("--handoff takes event or gate"); }
            o.gate = mode == "gate";
        } else if (arg == "--timeout") {
            o.timeout = true;
        } else if (arg == "--pair") {
            o.pair = true;
        } else if (arg == "--chunk") {
            o.chunk = static_cast<std::uint32_t>(std::stoul(value()));
        } else if (arg == "--context") {
            o.context = static_cast<std::uint32_t>(std::stoul(value()));
        } else if (arg == "--plain") {
            o.mtp = false;
        } else if (arg == "--no-forced") {
            o.forced = false;
        } else if (arg == "--no-cancel") {
            o.cancel = false;
        } else if (arg == "--long") {
            o.long_prompt = true;
        } else if (arg == "--walk-modes") {
            o.walk_modes = true;
        } else if (arg == "--prompt-decode") {
            o.prompt_decode = static_cast<std::uint32_t>(std::stoul(value()));
            if (o.prompt_decode == 0) { throw std::invalid_argument("--prompt-decode needs N > 0"); }
        } else if (arg == "--work") {
            o.work = value();
        } else {
            throw std::invalid_argument("unknown option " + arg);
        }
    }
    if (o.long_prompt) { o.context = std::max<std::uint32_t>(o.context, 49152); }
    if (o.timeout && (!o.gate || o.decode == 0)) {
        throw std::invalid_argument("--timeout needs --handoff gate and --decode N > 0");
    }
    if (o.pair && o.decode == 0) { throw std::invalid_argument("--pair needs --decode N > 0"); }
    return o;
}

int failures = 0;

void check(bool ok, const std::string& what) {
    std::printf("%s: %s\n", ok ? "ok" : "FAIL", what.c_str());
    if (!ok) { ++failures; }
}

// One request's outcome: its generated ids (empty when it failed), its error, and the logits-hash
// lines its calls wrote.
struct Outcome {
    std::vector<infernix::TokenId> ids;
    std::string error;
    std::vector<std::string> lines;
};

// One Engine's run over the cases.
struct Run {
    std::map<std::uint32_t, Outcome> prompts; // by prompt length
    std::optional<Outcome> decoded;           // the 255-token prompt generating --decode tokens
    std::optional<Outcome> timed_out, after_timeout;
    std::optional<Outcome> pair_a, pair_b; // two requests decoding together (--pair)
    std::optional<Outcome> forced;
    std::optional<Outcome> cancelled, after_cancel;
    std::optional<Outcome> long_prompt;
    std::vector<std::string> diagnostics;
};

// Reads the hash file's lines past the first `seen`.
std::vector<std::string> new_lines(const fs::path& file, std::size_t& seen) {
    std::ifstream in(file);
    std::vector<std::string> all;
    for (std::string line; std::getline(in, line);) {
        if (!line.empty()) { all.push_back(line); }
    }
    std::vector<std::string> out(all.begin() + static_cast<std::ptrdiff_t>(std::min(seen, all.size())), all.end());
    seen = std::max(seen, all.size());
    return out;
}

// A deterministic text long enough for every prompt length (raw tokens, no chat template).
std::string corpus() {
    std::string text;
    for (int i = 0; text.size() < 400000; ++i) {
        text += "Section " + std::to_string(i) +
                ". The lighthouse keeper counted the ships that passed the northern reef, wrote their names in a "
                "ledger, and compared the tides with the almanac his grandfather had annotated in red ink. When "
                "the fog came in from the sea, he lit the lamp, wound the clockwork that turned the lens, and "
                "listened to the bell buoy marking the channel. ";
        text += "def checksum_" + std::to_string(i) + "(values):\n    total = 0\n    for index, value in "
                "enumerate(values):\n        total = (total * 31 + value + index) % 1000003\n    return total\n\n";
    }
    return text;
}

infernix::RequestOptions greedy(std::uint32_t tokens) {
    infernix::RequestOptions request;
    request.execution.requested_output_tokens = tokens;
    request.execution.sampling.temperature    = 0.0F;
    request.execution.allow_prefix_reuse      = false;
    return request;
}

// `tokens` (empty on the first run) receives the corpus through the Engine's tokenizer.
Run run_engine(const Options& o, const char* artifact, const char* ngram, const std::vector<int>& devices,
               std::optional<std::uint32_t> k, const fs::path& hash_file, std::vector<infernix::TokenId>& tokens) {
    Run run;
    std::mutex diagnostics_mutex;
    infernix::EngineOptions options;
    options.artifact_path = artifact;
    if (ngram != nullptr) { options.ngram_volume_path = ngram; }
    options.device = devices.front();
    if (devices.size() > 1) {
        options.devices       = devices;
        options.layer_split   = k;
        options.split_handoff = o.gate ? infernix::SplitHandoff::Gate : infernix::SplitHandoff::Event;
    }
    options.max_context           = o.context;
    options.prefill_chunk         = o.chunk;
    options.max_concurrency       = o.pair ? 2 : 1;
    options.kv_cache              = infernix::KvCacheStorage::Int8Group64;
    options.kv_capacity           = infernix::KvCapacityPolicy::explicit_capacity(o.context);
    options.context_cache.enabled = false;
    if (o.mtp) {
        options.speculative.backend      = infernix::SpeculativeBackend::Mtp;
        options.speculative.draft_tokens = 4;
    }
    options.diagnostic_observer.callback = [&](const infernix::Diagnostic& d) {
        const std::lock_guard<std::mutex> lock(diagnostics_mutex);
        run.diagnostics.push_back(d.message);
        if (d.level != infernix::DiagnosticLevel::Debug) { std::printf("  [engine] %s\n", d.message.c_str()); }
    };
    infernix::models::qwen4_exp::testing::set_logits_hash(hash_file);
    std::size_t seen = 0;
    {
        infernix::Engine engine(options);
        infernix::models::qwen4_exp::testing::set_logits_hash({});
        if (tokens.empty()) { tokens = engine.tokenize_text(corpus()); }
        const std::uint32_t longest =
            std::max(o.long_prompt ? 40000U : 0U, *std::max_element(o.prompts.begin(), o.prompts.end()));
        if (tokens.size() < longest) { throw std::logic_error("the corpus is shorter than the longest prompt"); }
        const auto request = [&](std::vector<infernix::TokenId> ids, std::uint32_t outputs) {
            Outcome out;
            try {
                out.ids = engine.generate(engine.prepare_tokens(std::move(ids), false), greedy(outputs)).generated_token_ids;
            } catch (const std::exception& error) { out.error = error.what(); }
            out.lines = new_lines(hash_file, seen);
            return out;
        };
        for (const std::uint32_t n : o.prompts) {
            const auto started = std::chrono::steady_clock::now();
            run.prompts[n] =
                request(std::vector<infernix::TokenId>(tokens.begin(), tokens.begin() + n), o.prompt_decode);
            std::printf("  %u-token prompt: %zu call(s), %.1f s%s\n", n, run.prompts[n].lines.size(),
                        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count(),
                        run.prompts[n].error.empty() ? "" : (", failed: " + run.prompts[n].error).c_str());
        }
        const std::uint32_t decoded = std::min<std::uint32_t>(255, static_cast<std::uint32_t>(tokens.size()));
        if (o.decode > 0) {
            const auto started = std::chrono::steady_clock::now();
            run.decoded = request(std::vector<infernix::TokenId>(tokens.begin(), tokens.begin() + decoded), o.decode);
            const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
            std::printf("  %u-token prompt, %u tokens: %zu calls and rounds, %.1f s%s\n", decoded, o.decode,
                        run.decoded->lines.size(), seconds,
                        run.decoded->error.empty() ? "" : (", failed: " + run.decoded->error).c_str());
        }
        if (o.pair && tokens.size() >= decoded + 300) {
            // Two requests decoding together: the 255-token prompt and another 300-token one.
            Outcome a, b;
            try {
                auto first = engine.submit(
                    engine.prepare_tokens(std::vector<infernix::TokenId>(tokens.begin(), tokens.begin() + decoded), false),
                    greedy(o.decode));
                auto second = engine.submit(
                    engine.prepare_tokens(
                        std::vector<infernix::TokenId>(tokens.begin() + decoded, tokens.begin() + decoded + 300), false),
                    greedy(o.decode));
                try { a.ids = first.wait().generated_token_ids; } catch (const std::exception& error) { a.error = error.what(); }
                try { b.ids = second.wait().generated_token_ids; } catch (const std::exception& error) { b.error = error.what(); }
            } catch (const std::exception& error) { a.error = b.error = error.what(); }
            a.lines = new_lines(hash_file, seen); // both requests' rounds, interleaved (not compared)
            std::printf("  pair: %zu and %zu tokens, %zu calls and rounds%s%s\n", a.ids.size(), b.ids.size(),
                        a.lines.size(), a.error.empty() ? "" : (", first failed: " + a.error).c_str(),
                        b.error.empty() ? "" : (", second failed: " + b.error).c_str());
            run.pair_a = std::move(a);
            run.pair_b = std::move(b);
        }
        if (o.timeout && devices.size() > 1) {
            // One round's first segment starts 1.5 s late: the gated handoff after it outlasts its
            // 1 s bound and the request fails; the next request runs as if nothing had happened.
            infernix::models::qwen4_exp::testing::delay_next_round(1500);
            run.timed_out = request(std::vector<infernix::TokenId>(tokens.begin(), tokens.begin() + decoded), o.decode);
            infernix::models::qwen4_exp::testing::delay_next_round(0);
            run.after_timeout =
                request(std::vector<infernix::TokenId>(tokens.begin(), tokens.begin() + decoded), o.decode);
            std::printf("  delayed round: %s; the next request %s\n",
                        run.timed_out->error.empty() ? "no error" : run.timed_out->error.c_str(),
                        run.after_timeout->error.empty() ? "succeeded" : run.after_timeout->error.c_str());
        }
        if (o.forced) {
            // A thinking budget of one token: after the first (reasoning) token the Engine appends the
            // thinking-close control as forced tokens; the answer after them needs decode rounds.
            infernix::PromptInput input;
            infernix::ChatMessage message;
            message.role = infernix::ChatRole::User;
            message.parts.push_back({.kind = infernix::MessagePartKind::Text,
                                     .text = "Explain in one sentence why the sky is blue."});
            input.messages.push_back(std::move(message));
            input.options.enable_thinking = true;
            infernix::RequestOptions forced = greedy(64);
            forced.execution.thinking.budget = 1;
            Outcome out;
            try {
                out.ids = engine.generate(engine.prepare(std::move(input)), forced).generated_token_ids;
            } catch (const std::exception& error) { out.error = error.what(); }
            out.lines = new_lines(hash_file, seen);
            std::printf("  forced tokens: %zu call(s)%s\n", out.lines.size(),
                        out.error.empty() ? "" : (", failed: " + out.error).c_str());
            run.forced = std::move(out);
        }
        if (o.cancel) {
            // Cancelled while it prefills (its handle is dropped), then the 255-token prompt again.
            const std::uint32_t n = std::min<std::uint32_t>(4113, static_cast<std::uint32_t>(tokens.size()));
            {
                auto handle = engine.submit(
                    engine.prepare_tokens(std::vector<infernix::TokenId>(tokens.begin(), tokens.begin() + n), false),
                    greedy(1));
                std::this_thread::sleep_for(std::chrono::milliseconds(1500));
            }
            Outcome cancelled;
            run.after_cancel = request(std::vector<infernix::TokenId>(tokens.begin(), tokens.begin() + 255), 1);
            // The cancelled request's calls precede the next request's in the file.
            const std::size_t own = run.after_cancel->lines.size() >= 1 ? 1 : 0;
            cancelled.lines.assign(run.after_cancel->lines.begin(), run.after_cancel->lines.end() - own);
            run.after_cancel->lines.erase(run.after_cancel->lines.begin(), run.after_cancel->lines.end() - own);
            run.cancelled = std::move(cancelled);
            std::printf("  cancelled prompt: %zu of its calls ran before the cancel\n", run.cancelled->lines.size());
        }
        if (o.long_prompt) {
            const auto started = std::chrono::steady_clock::now();
            run.long_prompt =
                request(std::vector<infernix::TokenId>(tokens.begin(), tokens.begin() + 40000), 1);
            std::printf("  40000-token prompt: %zu call(s), %.1f s%s\n", run.long_prompt->lines.size(),
                        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count(),
                        run.long_prompt->error.empty() ? "" : (", failed: " + run.long_prompt->error).c_str());
        }
    }
    return run;
}

std::string first_line(const Outcome& outcome) { return outcome.lines.empty() ? "(none)" : outcome.lines.back(); }

// Every reference line is the split's same call's, bit for bit (a layer walk of the unsplit Engine
// computes no logits before a span's last chunk, so the reference may have fewer lines).
bool lines_cover(const std::vector<std::string>& reference, const std::vector<std::string>& split) {
    if (reference.empty() || split.empty() || reference.back() != split.back()) { return false; }
    std::size_t at = 0;
    for (const std::string& line : reference) {
        while (at < split.size() && split[at] != line) { ++at; }
        if (at == split.size()) { return false; }
        ++at;
    }
    return true;
}

// The logits of two runs of one request agree: their lines are equal, or the run with fewer lines (a
// layer walk computes no logits before a span's last chunk) is covered by the other's.
bool same_logits(const std::vector<std::string>& a, const std::vector<std::string>& b) {
    if (a.size() == b.size()) { return !a.empty() && a == b; }
    return a.size() < b.size() ? lines_cover(a, b) : lines_cover(b, a);
}

void set_variable(const char* name, const char* value) {
#if defined(_WIN32)
    ::_putenv_s(name, value);
#else
    ::setenv(name, value, 1);
#endif
}

// The layer walk's diagnostics of a run (debug level): span reports and each lease's streamed experts.
void print_walks(const std::vector<std::string>& diagnostics) {
    for (const std::string& line : diagnostics) {
        if (line.find("layer walk span") != std::string::npos || line.find("prefill expert stream") != std::string::npos ||
            line.find("prefill stream lease") != std::string::npos) {
            std::printf("  [walk] %s\n", line.c_str());
        }
    }
}

// The page counts the segments' "KV pool grew to N pages" reports name.
std::vector<std::string> growth_reports(const std::vector<std::string>& diagnostics) {
    std::vector<std::string> out;
    for (const std::string& line : diagnostics) {
        if (line.find("KV pool grew to") != std::string::npos) { out.push_back(line); }
    }
    return out;
}

} // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const char* artifact = std::getenv("INFERNIX_QWEN4_ARTIFACT");
    if (artifact == nullptr) {
        std::printf("SKIP: set INFERNIX_QWEN4_ARTIFACT\n");
        return 77;
    }
    const char* ngram = std::getenv("INFERNIX_QWEN4_NGRAM");
    try {
        const Options o = parse(argc, argv);
        // The Engine numbers devices in PCI bus order; so must this count, which is CUDA's first use.
#if defined(_WIN32)
        std::size_t length = 0;
        if (::getenv_s(&length, nullptr, 0, "CUDA_DEVICE_ORDER") != 0 || length == 0) {
            ::_putenv_s("CUDA_DEVICE_ORDER", "PCI_BUS_ID");
        }
#else
        ::setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID", 0);
#endif
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess) { count = 0; }
        for (const int device : o.devices) {
            if (device >= count) {
                std::printf("SKIP: device %d is not present (%d CUDA devices)\n", device, count);
                return 77;
            }
        }
        fs::create_directories(o.work);
        const int reference_device = o.reference.value_or(o.devices.front());
        const std::uint32_t longest =
            std::max(o.long_prompt ? 40000U : 0U, *std::max_element(o.prompts.begin(), o.prompts.end()));
        if (longest >= o.context) { throw std::invalid_argument("a prompt does not fit in --context"); }

        std::printf("reference: device %d, unsplit\n", reference_device);
        // The prompts' tokens: the corpus through the reference Engine's tokenizer (run_engine).
        std::vector<infernix::TokenId> tokens;
        const Run reference =
            run_engine(o, artifact, ngram, {reference_device}, std::nullopt, o.work / "reference.hash", tokens);
        if (tokens.size() < longest) { throw std::logic_error("the corpus is shorter than the longest prompt"); }
        for (const auto& [n, outcome] : reference.prompts) {
            check(outcome.error.empty() && outcome.ids.size() == o.prompt_decode && !outcome.lines.empty(),
                  "reference: the " + std::to_string(n) + "-token prompt gives " + std::to_string(o.prompt_decode) +
                      " token(s) and its calls' logits");
        }

        if (o.walk_modes) { print_walks(reference.diagnostics); }
        // (K, mode): 0 the default (layer walk), 1 chunk-major.
        std::vector<std::pair<std::uint32_t, int>> runs;
        for (const std::uint32_t k : o.splits) {
            runs.emplace_back(k, 0);
            if (o.walk_modes) { runs.emplace_back(k, 1); }
        }
        const char* const mode_names[] = {" (layer walk)", " (chunk-major)"};
        const char* const mode_files[] = {"", "-chunk"};
        std::map<std::uint32_t, Run> walked; // per K, the default run (--walk-modes)
        for (const auto& [k, mode] : runs) {
            const bool walking     = mode == 0;
            const std::string name = "--devices " + std::to_string(o.devices[0]) + "," + std::to_string(o.devices[1]) +
                                     " --layer-split " + std::to_string(k) + (o.walk_modes ? mode_names[mode] : "");
            std::printf("%s\n", name.c_str());
            set_variable("INFERNIX_SPLIT_WALK", mode == 1 ? "0" : "2");
            const Run split = run_engine(o, artifact, ngram, o.devices, k,
                                         o.work / ("split-" + std::to_string(k) + mode_files[mode] + ".hash"), tokens);
            set_variable("INFERNIX_SPLIT_WALK", "2");
            if (o.walk_modes) { print_walks(split.diagnostics); }
            for (const std::uint32_t n : o.prompts) {
                const Outcome& want = reference.prompts.at(n);
                const Outcome& got  = split.prompts.at(n);
                check(got.error.empty() && got.ids == want.ids && !want.ids.empty(),
                      name + ": the " + std::to_string(n) + "-token prompt's " + std::to_string(want.ids.size()) +
                          " generated token(s) equal the unsplit run's" +
                          (got.error.empty() ? "" : " (failed: " + got.error + ")"));
                if (o.walk_modes) {
                    check(same_logits(want.lines, got.lines),
                          name + ": the " + std::to_string(n) + "-token prompt's calls and rounds have the unsplit " +
                              "run's logits (" + std::to_string(got.lines.size()) + " lines vs " +
                              std::to_string(want.lines.size()) + ", last: " + first_line(got) + " vs " +
                              first_line(want) + ")");
                    if (!walking && walked.count(k) != 0) {
                        const Outcome& walk = walked.at(k).prompts.at(n);
                        check(got.ids == walk.ids && same_logits(walk.lines, got.lines),
                              name + ": the " + std::to_string(n) + "-token prompt gives the default run's ids and " +
                                  "logits (" + std::to_string(walk.lines.size()) + " lines vs " +
                                  std::to_string(got.lines.size()) + ")");
                    }
                } else {
                    check(got.lines == want.lines,
                          name + ": the " + std::to_string(n) + "-token prompt's " + std::to_string(want.lines.size()) +
                              " call(s) have the unsplit run's logits (last: " + first_line(got) + " vs " +
                              first_line(want) + ")");
                }
            }
            if (o.forced) {
                const Outcome& want = *reference.forced;
                const Outcome& got  = *split.forced;
                const bool has_forced = std::any_of(want.lines.begin(), want.lines.end(), [](const std::string& line) {
                    return line.rfind("forced ", 0) == 0;
                });
                check(want.error.empty() && has_forced, "reference: the thinking budget appended forced tokens");
                check(got.error.empty() && got.ids == want.ids,
                      name + ": the forced tokens' request generates the unsplit run's ids (" +
                          std::to_string(got.ids.size()) + " tokens" +
                          (got.error.empty() ? "" : ", failed: " + got.error) + ")");
                check(got.lines == want.lines,
                      name + ": the prefill and forced-token calls and the answer's rounds have the unsplit run's "
                             "logits (" +
                          std::to_string(got.lines.size()) + " vs " + std::to_string(want.lines.size()) + " calls)");
            }
            if (o.decode > 0) {
                const Outcome& want = *reference.decoded;
                const Outcome& got  = *split.decoded;
                check(got.error.empty() && got.ids == want.ids && !want.ids.empty(),
                      name + ": decoding " + std::to_string(o.decode) + " tokens gives the unsplit run's ids (" +
                          std::to_string(got.ids.size()) + " vs " + std::to_string(want.ids.size()) + " tokens" +
                          (got.error.empty() ? "" : ", failed: " + got.error) + ")");
                check(got.lines == want.lines,
                      name + ": every call and round of the decoded request has the unsplit run's logits (" +
                          std::to_string(got.lines.size()) + " vs " + std::to_string(want.lines.size()) +
                          " lines, last: " + first_line(got) + " vs " + first_line(want) + ")");
            }
            if (o.pair && reference.pair_a && split.pair_a) {
                check(reference.pair_a->error.empty() && reference.decoded &&
                          reference.pair_a->ids == reference.decoded->ids,
                      "reference: the pair's 255-token request gives the C = 1 request's ids");
                check(split.pair_a->error.empty() && split.pair_a->ids == reference.pair_a->ids &&
                          split.pair_b->error.empty() && split.pair_b->ids == reference.pair_b->ids &&
                          !reference.pair_b->ids.empty(),
                      name + ": two requests decoding together give the unsplit run's ids (" +
                          std::to_string(split.pair_a->ids.size()) + " and " + std::to_string(split.pair_b->ids.size()) +
                          " tokens" + (split.pair_a->error.empty() ? "" : ", failed: " + split.pair_a->error) +
                          (split.pair_b->error.empty() ? "" : ", failed: " + split.pair_b->error) + ")");
            }
            if (o.timeout && split.timed_out) {
                check(split.timed_out->error.find("did not arrive within") != std::string::npos,
                      name + ": a round whose handoff outlasts its bound fails with the timeout (" +
                          (split.timed_out->error.empty() ? std::string("no error") : split.timed_out->error) + ")");
                check(split.after_timeout->error.empty() && split.after_timeout->ids == reference.decoded->ids,
                      name + ": the request after the timeout gives the unsplit run's ids");
                check(split.after_timeout->lines == reference.decoded->lines,
                      name + ": the request after the timeout has the unsplit run's logits");
            }
            if (o.cancel) {
                check(split.after_cancel->error.empty() && split.after_cancel->ids == reference.after_cancel->ids,
                      name + ": after a cancelled prompt a request gives the unsplit run's token");
                check(split.after_cancel->lines == reference.after_cancel->lines,
                      name + ": after a cancelled prompt a request has the unsplit run's logits (" +
                          std::to_string(split.cancelled->lines.size()) + " of the cancelled prompt's calls ran)");
            }
            if (o.long_prompt) {
                const Outcome& want = *reference.long_prompt;
                const Outcome& got  = *split.long_prompt;
                check(got.error.empty() && got.ids == want.ids,
                      name + ": the 40000-token prompt's first token equals the unsplit run's");
                check(lines_cover(want.lines, got.lines),
                      name + ": the 40000-token prompt's calls have the unsplit run's logits (" +
                          std::to_string(got.lines.size()) + " calls, the reference computed " +
                          std::to_string(want.lines.size()) + ")");
                const auto grown = growth_reports(split.diagnostics);
                // One report per segment with KV per growth, all to the same page count.
                std::string pages;
                bool same = !grown.empty();
                for (const std::string& line : grown) {
                    const std::size_t at = line.find("grew to ");
                    const std::string count =
                        line.substr(at + 8, line.find(' ', at + 8) - (at + 8));
                    if (pages.empty()) { pages = count; }
                    same = same && count == pages;
                    std::printf("  %s\n", line.c_str());
                }
                const std::size_t kv_segments = k <= 3 ? 1 : 2; // segment 0 has attention layers from K = 4
                check(same && grown.size() % kv_segments == 0,
                      name + ": the KV pool grew in lockstep on every segment with KV (" +
                          std::to_string(grown.size()) + " reports)");
            }
            if (walking && o.walk_modes) { walked.emplace(k, split); }
        }
    } catch (const std::exception& error) {
        std::printf("FAIL: %s\n", error.what());
        return 1;
    }
    if (failures != 0) {
        std::printf("%d checks failed\n", failures);
        return 1;
    }
    std::printf("split prefill checks passed\n");
    return 0;
}
