// Qwen4Exp prefix cache on the real artifact (design §19.3.1). Skips unless INFERNIX_QWEN4_ARTIFACT
// names a Qwen4Exp artifact (INFERNIX_QWEN4_NGRAM: its n-gram volume, default <artifact>.ngram).
// INFERNIX_QWEN4_KV selects the KV storage (bf16, int8, fp8, nvfp4, k8v4, vq2, k4v2; default int8).
// One Engine per drafter mode serves every scenario, so the 64.5 GiB model loads once per mode:
//
//   tap resume   (E3) A request resumes from another's flexible tap at a prefill-chunk boundary; its
//                greedy output equals a cold run of the same prompt (allow_prefix_reuse = false),
//                which has the same call boundaries.
//   host blocks  (E3 through the Host tier) A pressure prompt evicts the cached blocks; the next
//                resume restores them from the slab pool and still equals the cold run.
//   endpoint     (E1) A second turn resumes from the first turn's endpoint left in its lane, with no
//                image restore; after pressure evicts everything Device-side, the same second turn
//                resumes through a Host image restore and must generate the same tokens.
//   X5           Chat turn 2 without turn 1's reasoning block resumes at least at turn 1's generation
//                opener tap.
//   X6           Sessions sharing only a system block resume at its structural tap (an exact split).
//   X3           A repeat of a chat prompt resumes at its generation opener (an exact off-grid tap) and
//                generates what the capturing run generated.
//   readout      (two lanes, plain) A prompt resumed from the cache reads its readout within rounding
//                of its cold run.
//   coalesce     (two lanes, plain) Two requests submitted together share a 1,024-token prefix: the
//                second waits for the first's snapshot at the divergence and equals an uncached run.
//   prefetch     (two lanes, plain) While a long request decodes, the waiting head's Host-only blocks
//                are copied to the Device; it resumes at its tap and equals an uncached run.
//   X12          (plain mode) Engine 1 serves two chat turns and stops, saving its Host tier to a file;
//                Engine 2 loads the file and resumes turn 2 from it, generating what Engine 1 did.
//
// A layer split (multi-GPU WP15): --devices A,B --layer-split K runs every scenario on the unsplit
// Engine of device --reference (default A), then on the split Engine, and compares them: every
// sequential request's ids, reused tokens, readout bits and per-call logits hashes
// (INFERNIX_LOGITS_HASH lines; an unsplit layer walk computes no logits before its span's last chunk,
// so the unsplit lines need only appear, in order, among the split's), and the concurrent scenarios'
// ids. The Host tier files are crossed: the split Engine 2 loads the file the unsplit Engine 1 saved,
// and an unsplit Engine loads the split Engine 1's; each resumes turn 2 with the unsplit turn 2's ids.
// Two GPUs need INFERNIX_NUMERICS_SMS at the smaller GPU's SM count.
//
//   infernix_qwen4_exp_prefix_cache_real_test [--mtp] [--plain] [--no-concurrent] [--no-persistence]
//       [--devices A,B --layer-split K [--reference D]] [--work DIR]   (default: every mode)

#include "kv_cache_storage.h"
#include "models/qwen4_exp/program/program.h"
#include "infernix/engine.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

namespace {

namespace fs = std::filesystem;

constexpr std::uint32_t kChunk = 512;

std::vector<infernix::TokenId> synthetic_tokens(std::size_t count, std::uint32_t seed) {
    std::vector<infernix::TokenId> tokens;
    tokens.reserve(count);
    std::uint32_t state = seed * 2654435761U + 1U;
    for (std::size_t i = 0; i < count; ++i) {
        state = state * 1664525U + 1013904223U;
        tokens.push_back(static_cast<infernix::TokenId>(1000U + (state >> 8U) % 30000U));
    }
    return tokens;
}

std::vector<infernix::TokenId> concat(std::vector<infernix::TokenId> a, const std::vector<infernix::TokenId>& b) {
    a.insert(a.end(), b.begin(), b.end());
    return a;
}

infernix::RequestOptions greedy(std::uint32_t outputs, bool reuse) {
    infernix::RequestOptions options;
    options.execution.requested_output_tokens = outputs;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.allow_prefix_reuse      = reuse;
    options.stop.include_model_defaults       = false;
    return options;
}

int check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "  ok  " : "  FAIL", what.c_str());
    return ok ? 0 : 1;
}

std::string first_difference(const std::vector<infernix::TokenId>& a, const std::vector<infernix::TokenId>& b) {
    for (std::size_t i = 0; i < a.size() && i < b.size(); ++i) {
        if (a[i] != b[i]) { return "first difference at " + std::to_string(i); }
    }
    return a.size() == b.size() ? "identical" : "lengths differ";
}

// Where an Engine runs: one device, or a layer split at K across two.
struct Placement {
    std::vector<int> devices{0};
    std::optional<std::uint32_t> layer_split;

    [[nodiscard]] bool split() const noexcept { return devices.size() > 1; }
    [[nodiscard]] std::string name() const {
        if (!split()) { return "device " + std::to_string(devices.front()); }
        return "--devices " + std::to_string(devices[0]) + "," + std::to_string(devices[1]) + " --layer-split " +
               std::to_string(*layer_split);
    }
};

// One request's outcome, in scenario order: its ids, reused prompt tokens, readout bits and the
// logits-hash lines its calls and rounds wrote (none for concurrent scenarios, whose rounds' grouping
// depends on timing).
struct Outcome {
    std::string name;
    std::vector<infernix::TokenId> ids;
    std::uint32_t reused = 0;
    std::vector<std::uint32_t> readout;
    std::vector<std::string> lines;
    bool concurrent = false;
};

// An Engine of the run with its logits-hash file, recording each request's outcome.
class Session {
public:
    Session(infernix::EngineOptions options, const fs::path& hash_file, std::vector<Outcome>& outcomes)
        : hash_file_(hash_file), outcomes_(outcomes) {
        std::error_code ignored;
        fs::remove(hash_file_, ignored);
        infernix::models::qwen4_exp::testing::set_logits_hash(hash_file_);
        try {
            engine_.emplace(std::move(options));
        } catch (...) {
            infernix::models::qwen4_exp::testing::set_logits_hash({});
            throw;
        }
        infernix::models::qwen4_exp::testing::set_logits_hash({});
    }

    infernix::Engine& engine() { return *engine_; }

    // Records `result` under `name` with the hash lines written since the last record; returns it.
    infernix::GenerationResult record(const std::string& name, infernix::GenerationResult result,
                                      bool concurrent = false) {
        Outcome out;
        out.name       = name;
        out.ids        = result.generated_token_ids;
        out.reused     = result.reused_prompt_tokens;
        out.concurrent = concurrent;
        if (result.readout) {
            for (const float lp : result.readout->logprobs) {
                std::uint32_t bits = 0;
                std::memcpy(&bits, &lp, sizeof(bits));
                out.readout.push_back(bits);
            }
        }
        out.lines = new_lines();
        if (concurrent) { out.lines.clear(); }
        outcomes_.push_back(std::move(out));
        return result;
    }
    // Skips the lines of requests that are not recorded.
    void skip() { (void)new_lines(); }

private:
    std::vector<std::string> new_lines() {
        std::ifstream in(hash_file_);
        std::vector<std::string> all;
        for (std::string line; std::getline(in, line);) {
            if (!line.empty()) { all.push_back(line); }
        }
        std::vector<std::string> out(all.begin() + static_cast<std::ptrdiff_t>(std::min(seen_, all.size())), all.end());
        seen_ = std::max(seen_, all.size());
        return out;
    }

    fs::path hash_file_;
    std::vector<Outcome>& outcomes_;
    std::optional<infernix::Engine> engine_;
    std::size_t seen_ = 0;
};

infernix::EngineOptions base_options(const char* artifact, const char* ngram, const Placement& where,
                                     std::uint32_t lanes = 1) {
    infernix::EngineOptions options;
    options.artifact_path = artifact;
    if (ngram != nullptr) { options.ngram_volume_path = ngram; }
    options.device = where.devices.front();
    if (where.split()) {
        options.devices     = where.devices;
        options.layer_split = where.layer_split;
    }
    options.max_context          = 4096;
    // 72 pages (+1 copy-on-write page per lane): a 3,900-token pressure prompt evicts nearly every block.
    options.kv_capacity          = infernix::KvCapacityPolicy::explicit_capacity(4608);
    options.prefill_chunk        = kChunk;
    const char* kv               = std::getenv("INFERNIX_QWEN4_KV");
    options.kv_cache             = kv != nullptr ? infernix::test::parse_kv_cache_storage(kv)
                                                 : infernix::KvCacheStorage::Int8Group64;
    options.max_concurrency      = lanes;
    options.max_pending_requests = lanes == 1 ? 1 : 4;
    // The prefetch scenario's head waits while a 3,300-token request prefills and decodes 300
    // tokens: on a slow link with a cold expert cache that outlasts the default 30 s.
    if (lanes > 1) { options.pending_timeout_ms = 600000; }
    options.context_cache.mode                    = infernix::ContextCacheMode::Hybrid;
    options.context_cache.host_capacity_bytes = 2ULL << 30;
    return options;
}

std::string rules_text() {
    std::string rules;
    for (int r = 1; r <= 110; ++r) {
        rules += "Rule " + std::to_string(r) + ": items of colour " + std::to_string(r % 9) + " go to shelf " +
                 std::to_string((r * 7) % 13) + ".\n";
    }
    return rules;
}

// User turns longer than kMinimumTapSeparation (64 tokens): an opener closer to the structural tap
// joins its cluster, and only the earlier tap is kept.
std::string user_context() {
    std::string context;
    for (int i = 0; i < 4; ++i) {
        context += "Today a delivery of mixed items arrived at the warehouse and I need to plan where each "
                   "box goes before the afternoon shift starts. ";
    }
    return context;
}

infernix::ChatMessage message(infernix::ChatRole role, std::string text) {
    infernix::ChatMessage m;
    m.role = role;
    m.parts.push_back(infernix::MessagePart{.kind = infernix::MessagePartKind::Text, .text = std::move(text)});
    return m;
}

infernix::GenerationResult chat_on(infernix::Engine& engine, std::vector<infernix::ChatMessage> messages,
                                   std::uint32_t outputs) {
    infernix::PromptInput input;
    input.messages                = std::move(messages);
    input.options.enable_thinking = false;
    infernix::RequestOptions request    = greedy(outputs, true);
    request.stop.include_model_defaults = true;
    return engine.generate(engine.prepare(std::move(input)), request);
}

struct Run {
    std::vector<infernix::TokenId> tokens;
    std::uint32_t reused = 0;
};

int run_mode(const char* artifact, const char* ngram, const Placement& where, bool mtp, const fs::path& work,
             std::vector<Outcome>& outcomes) {
    std::printf("== %s, %s\n", where.name().c_str(), mtp ? "MTP drafter (--spec mtp, 3 drafts)" : "plain decode");
    infernix::EngineOptions options = base_options(artifact, ngram, where);
    if (mtp) {
        options.speculative.backend      = infernix::SpeculativeBackend::Mtp;
        options.speculative.draft_tokens = 3;
    }
    Session session(std::move(options), work / (std::string(mtp ? "mtp" : "plain") + ".hash"), outcomes);
    infernix::Engine& engine = session.engine();
    const auto generate = [&](const std::string& name, const std::vector<infernix::TokenId>& prompt,
                              std::uint32_t outputs, bool reuse) {
        const infernix::GenerationResult r =
            session.record(name, engine.generate(engine.prepare_tokens(prompt), greedy(outputs, reuse)));
        return Run{r.generated_token_ids, r.reused_prompt_tokens};
    };
    const auto image_restores = [&] { return engine.runtime_stats().hybrid_host_image_restores; };
    int failures = 0;

    // tap resume: A leaves a flexible tap at the start of its final call (1024); B shares 1,400
    // tokens, so it resumes at 1024 and prefills 1024..1536..1700 as a cold run does.
    const std::vector<infernix::TokenId> a = synthetic_tokens(1500, 11);
    std::vector<infernix::TokenId> b(a.begin(), a.begin() + 1400);
    const auto suffix = synthetic_tokens(300, 12);
    b.insert(b.end(), suffix.begin(), suffix.end());
    const Run cold_b = generate("cold b", b, 48, false);
    (void)generate("a", a, 16, true);
    const Run warm_b = generate("tap resume b", b, 48, true);
    failures += check(warm_b.reused == 1024, "tap resume reuses 1024 tokens (got " + std::to_string(warm_b.reused) + ")");
    failures += check(warm_b.tokens == cold_b.tokens, "tap resume equals the cold run (" +
                                                          first_difference(warm_b.tokens, cold_b.tokens) + ")");

    // host blocks: pressure evicts the Device copies; B now resumes from its own prompt-tail tap
    // (1536) through Host block restores.
    const std::vector<infernix::TokenId> pressure = synthetic_tokens(3900, 13);
    (void)generate("pressure", pressure, 4, true);
    const Run host_b = generate("host resume b", b, 48, true);
    failures += check(host_b.reused >= 1024, "Host resume reuses " + std::to_string(host_b.reused) + " tokens");
    failures += check(host_b.tokens == cold_b.tokens, "Host block restore equals the cold run (" +
                                                          first_difference(host_b.tokens, cold_b.tokens) + ")");

    // endpoint: turn 2 echoes turn 1 and its output; the lane still holds turn 1's endpoint, so the
    // resume restores no image.
    const std::vector<infernix::TokenId> t1 = synthetic_tokens(700, 14);
    const Run turn1 = generate("turn 1", t1, 96, true);
    std::vector<infernix::TokenId> t2 = t1;
    t2.insert(t2.end(), turn1.tokens.begin(), turn1.tokens.end());
    const auto more = synthetic_tokens(64, 15);
    t2.insert(t2.end(), more.begin(), more.end());
    const std::uint64_t before_resident = image_restores();
    const Run resident = generate("turn 2 resident", t2, 48, true);
    const std::uint64_t resident_restores = image_restores() - before_resident;
    const std::uint32_t endpoint = static_cast<std::uint32_t>(t1.size() + turn1.tokens.size() - 1U);
    failures += check(resident.reused == endpoint, "turn 2 resumes at the endpoint " + std::to_string(endpoint) +
                                                       " (got " + std::to_string(resident.reused) + ")");
    failures += check(resident_restores == 0, "the same-slot endpoint resume restores no image (" +
                                                  std::to_string(resident_restores) + " image restores)");
    // Re-create turn 1's endpoint, evict, then resume turn 2 through the Host image.
    (void)generate("turn 1 again", t1, 96, true);
    (void)generate("pressure again", pressure, 4, true);
    const std::uint64_t before_restored = image_restores();
    const Run restored = generate("turn 2 restored", t2, 48, true);
    const std::uint64_t restored_restores = image_restores() - before_restored;
    failures += check(restored.reused == endpoint, "turn 2 resumes at the endpoint after eviction (got " +
                                                       std::to_string(restored.reused) + ")");
    failures += check(restored_restores == 1, "the endpoint resume after eviction restores the Host image (" +
                                                  std::to_string(restored_restores) + " image restores)");
    failures += check(restored.tokens == resident.tokens, "Host-restored endpoint equals the lane-resident resume (" +
                                                              first_difference(restored.tokens, resident.tokens) + ")");

    // Chat prompts carry the structural (end of the system block) and generation-opener taps.
    const auto chat = [&](const std::string& name, std::vector<infernix::ChatMessage> messages, std::uint32_t outputs) {
        return session.record(name, chat_on(engine, std::move(messages), outputs));
    };
    const infernix::ChatMessage system = message(infernix::ChatRole::System, rules_text());
    const std::string context          = user_context();
    const auto user = [&](const std::string& question) {
        return message(infernix::ChatRole::User, context + question);
    };

    // X5 chat no-echo: turn 2 re-renders turn 1's reply without its reasoning block; it resumes at
    // least at turn 1's generation opener (a few tokens before turn 1's prompt end).
    const infernix::GenerationResult s1 = chat("X5 turn 1", {system, user("Which shelf takes colour 4?")}, 32);
    const std::uint32_t n1 = s1.prompt.prompt_tokens;
    const infernix::GenerationResult x5 =
        chat("X5 turn 2",
             {system, user("Which shelf takes colour 4?"), message(infernix::ChatRole::Assistant, s1.content),
              message(infernix::ChatRole::User, "And colour 5?")},
             32);
    std::printf("        X5: turn 1 prompt %u, turn 2 prompt %u reused %u\n", n1, x5.prompt.prompt_tokens,
                x5.reused_prompt_tokens);
    failures += check(x5.reused_prompt_tokens + 16U >= n1, "X5 turn 2 reuses at least turn 1's opener (reused " +
                                                               std::to_string(x5.reused_prompt_tokens) + ", turn 1 prompt " +
                                                               std::to_string(n1) + ")");

    // X6 shared preamble: sessions 2 and 3 share only the system block; they resume at its
    // structural tap, an exact split the first session made.
    const infernix::GenerationResult s2 = chat("X6 session 2", {system, user("Summarise rule 12 in one sentence.")}, 32);
    const infernix::GenerationResult s3 =
        chat("X6 session 3", {system, user("How many rules send items to shelf 0?")}, 32);
    std::printf("        X6: sessions 2 and 3 reused %u and %u of %u and %u\n", s2.reused_prompt_tokens,
                s3.reused_prompt_tokens, s2.prompt.prompt_tokens, s3.prompt.prompt_tokens);
    failures += check(s2.reused_prompt_tokens >= 1024 && s2.reused_prompt_tokens + 8U < s2.prompt.prompt_tokens &&
                          s3.reused_prompt_tokens == s2.reused_prompt_tokens,
                      "X6 sessions 2 and 3 resume at the shared structural frontier");

    // X3 split equivalence at an exact off-grid tap: session 2 again resumes at its own generation
    // opener; the run that captured it split its prefill there, so the resume repeats its computation.
    const infernix::GenerationResult again = chat("X3 repeat", {system, user("Summarise rule 12 in one sentence.")}, 32);
    failures += check(again.reused_prompt_tokens + 16U >= s2.prompt.prompt_tokens &&
                          again.reused_prompt_tokens < s2.prompt.prompt_tokens,
                      "X3 the repeat resumes at the opener (reused " + std::to_string(again.reused_prompt_tokens) + ")");
    failures += check(again.generated_token_ids == s2.generated_token_ids,
                      "X3 the opener resume equals the capturing run (" +
                          first_difference(again.generated_token_ids, s2.generated_token_ids) + ")");
    return failures;
}

// Two lanes, plain decode: a resumed readout, coalescing and prefetch (as the decide test's).
int run_concurrent(const char* artifact, const char* ngram, const Placement& where, const fs::path& work,
                   std::vector<Outcome>& outcomes) {
    std::printf("== %s, two lanes (readout, coalesce, prefetch)\n", where.name().c_str());
    Session session(base_options(artifact, ngram, where, 2), work / "concurrent.hash", outcomes);
    infernix::Engine& engine = session.engine();
    const auto run = [&](const std::vector<infernix::TokenId>& prompt, const infernix::RequestOptions& options) {
        return engine.generate(engine.prepare_tokens(prompt), options);
    };
    int failures = 0;

    // readout: a prompt resumed from the cache reads what its cold run reads, within rounding.
    const std::vector<infernix::TokenId> prompt = synthetic_tokens(700, 71);
    const auto cold = session.record("readout cold first token", run(prompt, greedy(1, false)));
    std::vector<infernix::TokenId> named = {cold.generated_token_ids.at(0)};
    for (const infernix::TokenId t : synthetic_tokens(15, 72)) {
        if (std::find(named.begin(), named.end(), t) == named.end()) { named.push_back(t); }
    }
    (void)session.record("readout publish", run(prompt, greedy(2, true))); // its blocks, taps and endpoint
    const std::vector<infernix::TokenId> longer = concat(prompt, synthetic_tokens(150, 73));
    infernix::RequestOptions cold_longer       = greedy(1, false);
    cold_longer.execution.readout_tokens       = named;
    infernix::RequestOptions warm_longer       = cold_longer;
    warm_longer.execution.allow_prefix_reuse   = true;
    const auto a = session.record("readout cold", run(longer, cold_longer));
    const auto b = session.record("readout resumed", run(longer, warm_longer));
    double worst = 0.0;
    if (a.readout && b.readout) {
        for (std::size_t i = 0; i < named.size(); ++i) {
            worst = std::max(worst, std::fabs(static_cast<double>(a.readout->logprobs[i] - b.readout->logprobs[i])));
        }
    }
    failures += check(b.reused_prompt_tokens > 0, "readout: the longer prompt resumes (reused " +
                                                      std::to_string(b.reused_prompt_tokens) + ")");
    failures += check(a.readout && b.readout && a.readout->top_token == b.readout->top_token && worst <= 0.05,
                      "readout: resumed equals cold within 0.05 nats (worst " + std::to_string(worst) + ")");

    // coalesce: the follower waits for the leader's snapshot at the divergence and resumes from it.
    const std::vector<infernix::TokenId> shared = synthetic_tokens(1024, 51);
    const auto first                            = concat(shared, synthetic_tokens(40, 52));
    const auto second                           = concat(shared, synthetic_tokens(40, 53));
    const auto reference = session.record("coalesce reference", run(second, greedy(16, false))).generated_token_ids;
    infernix::GenerationHandle leader   = engine.submit(engine.prepare_tokens(first), greedy(1, true));
    infernix::GenerationHandle follower = engine.submit(engine.prepare_tokens(second), greedy(16, true));
    const auto led      = session.record("coalesce leader", leader.wait(), true);
    const auto followed = session.record("coalesce follower", follower.wait(), true);
    failures += check(led.reused_prompt_tokens == 0 && followed.reused_prompt_tokens == 1024,
                      "coalesce: the follower resumes at the divergence (leader " +
                          std::to_string(led.reused_prompt_tokens) + ", follower " +
                          std::to_string(followed.reused_prompt_tokens) + ")");
    failures += check(followed.generated_token_ids == reference, "coalesce: the follower equals an uncached run");

    // prefetch: a 1,600-token prompt publishes taps at its chunk boundaries; after pressure evicts
    // them, a long request holds most pages while the head waits, and the head's Host-only blocks
    // are copied into what is free meanwhile.
    const std::vector<infernix::TokenId> base = synthetic_tokens(1600, 61);
    (void)session.record("prefetch base", run(base, greedy(1, true)));
    const auto head = concat(std::vector<infernix::TokenId>(base.begin(), base.begin() + 1536), synthetic_tokens(40, 62));
    const auto uncached = session.record("prefetch reference", run(head, greedy(8, false))).generated_token_ids;
    (void)session.record("prefetch pressure", run(synthetic_tokens(3900, 63), greedy(1, false)));
    const infernix::RuntimeStats before = engine.runtime_stats();
    infernix::GenerationHandle holder =
        engine.submit(engine.prepare_tokens(synthetic_tokens(3300, 64)), greedy(300, false));
    infernix::GenerationHandle waiting = engine.submit(engine.prepare_tokens(head), greedy(8, true));
    (void)session.record("prefetch holder", holder.wait(), true);
    const auto waited                  = session.record("prefetch head", waiting.wait(), true);
    const infernix::RuntimeStats after = engine.runtime_stats();
    const auto prefetched              = after.hybrid_prefetched_blocks - before.hybrid_prefetched_blocks;
    failures += check(prefetched > 0, "prefetch: the waiting head's blocks were prefetched (" +
                                          std::to_string(prefetched) + " blocks)");
    failures += check(waited.reused_prompt_tokens >= 1024 && waited.reused_prompt_tokens % kChunk == 0,
                      "prefetch: the head resumes at a chunk-boundary tap (" +
                          std::to_string(waited.reused_prompt_tokens) + ")");
    failures += check(waited.generated_token_ids == uncached, "prefetch: the head equals an uncached run");
    return failures;
}

// X12's second turn (its prompt carries Engine 1's first reply) and what Engine 1 generated for it.
struct SecondTurn {
    std::vector<infernix::ChatMessage> messages;
    std::vector<infernix::TokenId> ids;
    std::uint32_t prompt_tokens = 0;
};

// X12: the Host tier saved at an Engine's stop and loaded by the next Engine. `load` (when set)
// replaces the file Engine 2 loads (the other placement's file: a crossed load); `keep` receives a
// copy of the file Engine 1 saved and `second` Engine 1's second turn.
int run_persistence(const char* artifact, const char* ngram, const Placement& where, const fs::path& work,
                    std::vector<Outcome>& outcomes, const std::optional<fs::path>& load, const fs::path& keep,
                    SecondTurn& second) {
    std::printf("== %s, persistence (X12)%s\n", where.name().c_str(),
                load ? (", Engine 2 loads " + load->filename().string()).c_str() : "");
    const fs::path file = work / ("x12-" + std::to_string(where.devices.size()) + ".cache");
    std::error_code ignored;
    fs::remove(file, ignored);
    const auto options = [&](const fs::path& path) {
        infernix::EngineOptions o                  = base_options(artifact, ngram, where);
        o.context_cache.hybrid.persistent_file     = path;
        o.context_cache.hybrid.persistent_identity = "x12";
        return o;
    };
    const infernix::ChatMessage system = message(infernix::ChatRole::System, rules_text());
    const std::string context          = user_context();
    const std::vector<infernix::ChatMessage> turn1{system,
                                                   message(infernix::ChatRole::User, context + "Which shelf takes colour 2?")};
    int failures = 0;
    std::vector<infernix::ChatMessage> turn2 = turn1;
    infernix::GenerationResult before;
    {
        Session session(options(file), work / "x12-1.hash", outcomes);
        failures += check(!session.engine().load_summary().prefix_cache.restored, "Engine 1 starts without a saved file");
        const infernix::GenerationResult r1 = session.record("X12 turn 1", chat_on(session.engine(), turn1, 32));
        turn2.push_back(message(infernix::ChatRole::Assistant, r1.content));
        turn2.push_back(message(infernix::ChatRole::User, context + "And colour 3?"));
        before = session.record("X12 turn 2", chat_on(session.engine(), turn2, 32));
        second = SecondTurn{turn2, before.generated_token_ids, before.prompt.prompt_tokens};
    } // the stop saves the Host tier
    std::error_code size_error;
    const std::uintmax_t saved_bytes = fs::file_size(file, size_error);
    failures += check(!size_error && saved_bytes > 0, "the stop saved the Host tier (" +
                                                          std::to_string(size_error ? 0U : saved_bytes) + " bytes)");
    fs::copy_file(file, keep, fs::copy_options::overwrite_existing, ignored);
    const fs::path loaded = load ? work / ("x12-load-" + std::to_string(where.devices.size()) + ".cache") : file;
    if (load) { fs::copy_file(*load, loaded, fs::copy_options::overwrite_existing); }
    {
        Session session(options(loaded), work / "x12-2.hash", outcomes);
        const infernix::LoadSummary::PrefixCacheRestore restore = session.engine().load_summary().prefix_cache;
        std::printf("        restored %llu blocks and %llu snapshots (%s)\n",
                    static_cast<unsigned long long>(restore.blocks), static_cast<unsigned long long>(restore.snapshots),
                    restore.message.c_str());
        failures += check(restore.restored && restore.snapshots > 0, "Engine 2 restores the saved snapshots");
        // Turn 2 again: Engine 1 left a snapshot at its own generation opener, so the restarted
        // Engine resumes there and repeats Engine 1's computation of turn 2.
        const infernix::GenerationResult after = session.record("X12 turn 2 after the restart", chat_on(session.engine(), turn2, 32));
        failures += check(after.reused_prompt_tokens + 16U >= before.prompt.prompt_tokens,
                          "after the restart turn 2 resumes at its opener (reused " +
                              std::to_string(after.reused_prompt_tokens) + " of " +
                              std::to_string(after.prompt.prompt_tokens) + ")");
        failures += check(after.generated_token_ids == before.generated_token_ids,
                          "the restored resume equals Engine 1's turn 2 (" +
                              first_difference(after.generated_token_ids, before.generated_token_ids) + ")");
    }
    fs::remove(file, ignored);
    fs::remove(loaded, ignored);

    // The periodic save (persistent_save_interval): the file appears while the Engine runs, through
    // its .tmp, once the interval has passed after a change. A save during the request may find the
    // Host tier still filling and the next one rewrite it; once the Host tier stops changing, the
    // file holds the endpoint and is not rewritten.
    {
        infernix::EngineOptions o                       = options(file);
        o.context_cache.hybrid.persistent_save_interval = std::chrono::seconds(1);
        infernix::Engine engine(o);
        failures += check(!fs::exists(file), "the periodic test starts without a file");
        (void)chat_on(engine, turn1, 16);
        fs::path tmp = file;
        tmp += ".tmp";
        bool saved = false;
        for (int i = 0; i < 100 && !saved; ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            saved = fs::exists(file) && !fs::exists(tmp);
        }
        failures += check(saved, "the periodic save wrote the file while the Engine was idle");
        bool settled = false;
        for (int i = 0; i < 4 && saved && !settled; ++i) {
            const auto written = fs::last_write_time(file, ignored);
            std::this_thread::sleep_for(std::chrono::milliseconds(2500));
            settled = fs::last_write_time(file, ignored) == written;
        }
        failures += check(settled, "an unchanged Host tier is not saved again");
        const std::uintmax_t size = saved ? fs::file_size(file, ignored) : 0;
        failures += check(size > (std::uintmax_t{64} << 20),
                          "the settled periodic file holds turn 1's snapshot (" + std::to_string(size) + " bytes)");
    }
    fs::remove(file, ignored);
    return failures;
}

// An Engine loads the file another placement's Engine 1 saved and resumes X12's second turn there,
// generating what that Engine 1 generated (the reverse of the split Engine 2's crossed load).
int load_crossed(const char* artifact, const char* ngram, const Placement& where, const fs::path& work,
                 const fs::path& saved, const SecondTurn& second, std::vector<Outcome>& outcomes) {
    std::printf("== %s loads %s\n", where.name().c_str(), saved.filename().string().c_str());
    const fs::path file = work / "x12-crossed.cache";
    fs::copy_file(saved, file, fs::copy_options::overwrite_existing);
    infernix::EngineOptions o                  = base_options(artifact, ngram, where);
    o.context_cache.hybrid.persistent_file     = file;
    o.context_cache.hybrid.persistent_identity = "x12";
    int failures                               = 0;
    {
        Session session(std::move(o), work / "x12-crossed.hash", outcomes);
        const infernix::LoadSummary::PrefixCacheRestore restore = session.engine().load_summary().prefix_cache;
        std::printf("        restored %llu blocks and %llu snapshots (%s)\n",
                    static_cast<unsigned long long>(restore.blocks), static_cast<unsigned long long>(restore.snapshots),
                    restore.message.c_str());
        failures += check(restore.restored && restore.snapshots > 0, "the other placement's file restores its snapshots");
        const infernix::GenerationResult after =
            session.record("X12 turn 2 from the other placement's file", chat_on(session.engine(), second.messages, 32));
        failures += check(after.reused_prompt_tokens + 16U >= second.prompt_tokens,
                          "turn 2 resumes at its opener from the other placement's file (reused " +
                              std::to_string(after.reused_prompt_tokens) + " of " +
                              std::to_string(after.prompt.prompt_tokens) + ")");
        failures += check(after.generated_token_ids == second.ids,
                          "the resume from the other placement's file equals its Engine 1's turn 2 (" +
                              first_difference(after.generated_token_ids, second.ids) + ")");
    }
    std::error_code ignored;
    fs::remove(file, ignored);
    return failures;
}

// Every reference line is the other run's same call's, bit for bit, in order, and the last lines
// agree (an unsplit layer walk computes no logits before a span's last chunk).
bool lines_cover(const std::vector<std::string>& reference, const std::vector<std::string>& other) {
    if (reference.size() == other.size()) { return reference == other; }
    if (reference.empty() || other.empty() || reference.back() != other.back()) { return false; }
    std::size_t at = 0;
    for (const std::string& line : reference) {
        while (at < other.size() && other[at] != line) { ++at; }
        if (at == other.size()) { return false; }
        ++at;
    }
    return true;
}

int compare(const std::vector<Outcome>& reference, const std::vector<Outcome>& split, const std::string& what) {
    int failures = check(reference.size() == split.size(), what + ": both runs made the same requests (" +
                                                               std::to_string(reference.size()) + " and " +
                                                               std::to_string(split.size()) + ")");
    std::size_t equal = 0, lines = 0;
    for (std::size_t i = 0; i < reference.size() && i < split.size(); ++i) {
        const Outcome& r = reference[i];
        const Outcome& s = split[i];
        const bool same  = r.name == s.name && r.ids == s.ids && r.reused == s.reused && r.readout == s.readout &&
                          (r.concurrent || lines_cover(r.lines, s.lines));
        if (same) {
            ++equal;
            lines += r.lines.size();
            continue;
        }
        std::string why;
        if (r.ids != s.ids) { why += " ids: " + first_difference(r.ids, s.ids) + ";"; }
        if (r.reused != s.reused) { why += " reused " + std::to_string(r.reused) + " vs " + std::to_string(s.reused) + ";"; }
        if (r.readout != s.readout) { why += " readout bits differ;"; }
        if (!r.concurrent && !lines_cover(r.lines, s.lines)) {
            why += " logits hashes: " + std::to_string(r.lines.size()) + " vs " + std::to_string(s.lines.size()) +
                   " lines (last " + (r.lines.empty() ? "none" : r.lines.back()) + " vs " +
                   (s.lines.empty() ? "none" : s.lines.back()) + ");";
        }
        failures += check(false, what + ": '" + r.name + "' differs:" + why);
    }
    failures += check(equal == reference.size() && reference.size() == split.size(),
                      what + ": " + std::to_string(equal) + " of " + std::to_string(reference.size()) +
                          " requests equal the unsplit run's (ids, reused tokens, readout bits; " + std::to_string(lines) +
                          " unsplit logits hashes found among the split's)");
    return failures;
}

std::vector<int> parse_devices(const std::string& text) {
    std::vector<int> out;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ',')) { out.push_back(std::stoi(item)); }
    if (out.size() != 2) { throw std::invalid_argument("--devices takes two ordinals"); }
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
    bool plain = true, mtp = true, concurrent = true, persistence = true;
    Placement split;
    std::optional<int> reference_device;
    fs::path work = fs::temp_directory_path() / "infernix_qwen4_exp_prefix";
    try {
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            const auto value      = [&]() -> std::string {
                if (i + 1 >= argc) { throw std::invalid_argument(arg + " needs a value"); }
                return argv[++i];
            };
            if (arg == "--mtp") {
                plain = false;
            } else if (arg == "--plain") {
                mtp = false;
            } else if (arg == "--no-concurrent") {
                concurrent = false;
            } else if (arg == "--no-persistence") {
                persistence = false;
            } else if (arg == "--devices") {
                split.devices = parse_devices(value());
            } else if (arg == "--layer-split") {
                split.layer_split = static_cast<std::uint32_t>(std::stoul(value()));
            } else if (arg == "--reference") {
                reference_device = std::stoi(value());
            } else if (arg == "--work") {
                work = value();
            } else {
                throw std::invalid_argument("unknown option " + arg);
            }
        }
        if (split.split() != split.layer_split.has_value()) {
            throw std::invalid_argument("--devices A,B and --layer-split K go together");
        }
        if (split.split()) {
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
            for (const int device : split.devices) {
                if (device >= count) {
                    std::printf("SKIP: device %d is not present (%d CUDA devices)\n", device, count);
                    return 77;
                }
            }
        }
        fs::create_directories(work);
        int failures = 0;
        if (!split.split()) {
            // One placement: every scenario's own checks.
            const Placement one;
            std::vector<Outcome> outcomes;
            if (plain) { failures += run_mode(artifact, ngram, one, false, work, outcomes); }
            if (mtp) { failures += run_mode(artifact, ngram, one, true, work, outcomes); }
            if (concurrent && plain) { failures += run_concurrent(artifact, ngram, one, work, outcomes); }
            if (persistence && plain) {
                SecondTurn second;
                failures +=
                    run_persistence(artifact, ngram, one, work, outcomes, std::nullopt, work / "x12-saved.cache", second);
            }
        } else {
            const Placement unsplit{.devices = {reference_device.value_or(split.devices.front())}};
            const fs::path unsplit_file = work / "x12-unsplit.cache", split_file = work / "x12-split.cache";
            std::vector<Outcome> reference, other;
            SecondTurn unsplit_turn, split_turn;
            const Placement* const order[] = {&unsplit, &split};
            for (const Placement* where : order) {
                std::vector<Outcome>& outcomes = where == &unsplit ? reference : other;
                if (plain) { failures += run_mode(artifact, ngram, *where, false, work, outcomes); }
                if (mtp) { failures += run_mode(artifact, ngram, *where, true, work, outcomes); }
                if (concurrent && plain) { failures += run_concurrent(artifact, ngram, *where, work, outcomes); }
                if (persistence && plain) {
                    // The split Engine 2 loads the unsplit Engine 1's file.
                    failures += run_persistence(artifact, ngram, *where, work, outcomes,
                                                where == &split ? std::optional<fs::path>(unsplit_file) : std::nullopt,
                                                where == &unsplit ? unsplit_file : split_file,
                                                where == &unsplit ? unsplit_turn : split_turn);
                }
            }
            failures += compare(reference, other, split.name());
            if (persistence && plain) {
                // The unsplit Engine loads the split Engine 1's file and resumes turn 2 as both did.
                std::vector<Outcome> crossed;
                failures += load_crossed(artifact, ngram, unsplit, work, split_file, split_turn, crossed);
                std::error_code a, b;
                const bool same_size = fs::file_size(unsplit_file, a) == fs::file_size(split_file, b) && !a && !b;
                bool same_bytes      = same_size;
                if (same_size) {
                    std::ifstream x(unsplit_file, std::ios::binary), y(split_file, std::ios::binary);
                    std::vector<char> p(1 << 20), q(1 << 20);
                    while (same_bytes && x && y) {
                        x.read(p.data(), static_cast<std::streamsize>(p.size()));
                        y.read(q.data(), static_cast<std::streamsize>(q.size()));
                        same_bytes = x.gcount() == y.gcount() &&
                                     std::memcmp(p.data(), q.data(), static_cast<std::size_t>(x.gcount())) == 0;
                    }
                }
                std::printf("        the unsplit and split Engines' saved files are %s\n",
                            same_bytes ? "byte-identical" : "not byte-identical");
            }
        }
        std::printf(failures == 0 ? "qwen4_exp prefix cache checks passed\n" : "FAIL: %d checks failed\n", failures);
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
