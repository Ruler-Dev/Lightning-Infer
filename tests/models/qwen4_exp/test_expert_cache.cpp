// Host expert cache of Qwen3.8-Flash-Next (docs/maintainer/qwen3_8-flash-next-design.md §9):
// LFRU decisions equal tools/expert_cache_replay step for step, residency words round-trip,
// frames are never reused while a round that may read them is in flight, a resized pool (design
// §19.3.7, RT8) evicts the lowest-score experts wherever they sit, moves the survivors below the new
// top and never names a frame above it, and a layer split's saved state splits into per-segment
// parts and merges back (multi-GPU WP11).

#include "models/qwen4_exp/program/expert_cache/expert_cache.h"
#include "models/qwen4_exp/program/expert_cache/expert_state.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <cstdio>
#include <fstream>
#include <optional>
#include <random>
#include <set>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef INFERNIX_SOURCE_DIR
#    define INFERNIX_SOURCE_DIR "."
#endif

using namespace infernix::models::qwen4_exp::expert_cache;
using infernix::models::qwen4_exp::LayerRange;

namespace {

int g_failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_failures;
    }
}

void test_lfru_conformance() {
    const std::string path = std::string(INFERNIX_SOURCE_DIR) + "/tests/fixtures/expert_cache/lfru_conformance.txt";
    std::ifstream in(path);
    if (!in) {
        check(false, "conformance fixture readable");
        return;
    }
    std::string line;
    std::getline(in, line);
    unsigned keys = 0, capacity = 0;
    long hits = 0, misses = 0;
    std::sscanf(line.c_str(), "# keys %u capacity %u hits %ld misses %ld", &keys, &capacity, &hits, &misses);
    LfruPolicy policy(keys, capacity);
    LfruPolicy::Step step;
    long got_hits = 0, got_misses = 0, groups = 0, mismatches = 0;
    while (std::getline(in, line)) {
        std::istringstream s(line);
        std::string token;
        s >> token; // "G"
        std::vector<std::uint32_t> group, victims;
        bool after_bar = false;
        while (s >> token) {
            if (token == "|") {
                after_bar = true;
                continue;
            }
            (after_bar ? victims : group).push_back(static_cast<std::uint32_t>(std::stoul(token)));
        }
        policy.step(group, step);
        got_hits += static_cast<long>(step.hits.size());
        got_misses += static_cast<long>(step.misses.size());
        if (step.victims != victims) { ++mismatches; }
        ++groups;
    }
    std::printf("LFRU conformance: %ld groups, %ld mismatching steps\n", groups, mismatches);
    check(groups > 1000 && mismatches == 0, "LFRU victims equal the replay tool's, step for step");
    check(got_hits == hits && got_misses == misses, "LFRU hit and miss totals equal the replay tool's");
}

void test_residency_entries() {
    ResidencyEntry e{ResidencyState::kReady, 63, ResidencyEntry::kMaxFrame};
    check(ResidencyEntry::decode(e.encode()).frame == ResidencyEntry::kMaxFrame, "frame round trip");
    check(e.encode() == 0xBFFFFFFFU, "entry bit layout");
    const auto absent = e.next(ResidencyState::kAbsent, 0);
    check(absent.generation == 0 && absent.frame == 0 && absent.state == ResidencyState::kAbsent, "generation wraps");
    bool threw = false;
    try {
        (void)ResidencyEntry::decode(0xC0000000U);
    } catch (const std::exception&) { threw = true; }
    check(threw, "state 3 is invalid");
}

void test_budgeted_admission() {
    std::mt19937 rng(4);
    LfruPolicy policy(1024, 100);
    LfruPolicy::Step step;
    for (int i = 0; i < 5000; ++i) {
        std::set<std::uint32_t> g;
        while (g.size() < 10) { g.insert(static_cast<std::uint32_t>(rng() % (i % 500 < 250 ? 300 : 1024))); }
        const std::vector<std::uint32_t> group(g.begin(), g.end());
        const std::size_t budget = i % 3;
        policy.step(group, step, budget);
        check(step.admitted.size() <= std::max<std::size_t>(budget, 0) || budget >= step.misses.size(), "budget respected");
        check(policy.resident_count() <= policy.capacity(), "capacity respected");
        for (std::uint32_t v : step.victims) { check(!g.count(v), "current group never evicted"); }
        // With one admission, the victim is the lowest (score, key) resident outside the group, as
        // a full sort of the residents would order them (scores of other keys are unchanged by
        // the eviction).
        if (budget == 1 && step.victims.size() == 1) {
            const std::uint32_t v = step.victims[0];
            for (std::uint32_t r : policy.residents()) {
                if (g.count(r) || std::find(step.admitted.begin(), step.admitted.end(), r) != step.admitted.end()) {
                    continue;
                }
                const double sr = policy.score(r), sv = policy.score(v);
                check(sr > sv || (sr == sv && r > v), "the single victim is the lowest (score, key) resident");
            }
        }
        if (g_failures) { return; }
    }
    // The cost of one budgeted step at production size (9,000 residents of 24,576 keys), reported
    // for comparison; no bound is asserted.
    LfruPolicy big(24576, 9000);
    std::mt19937 rng2(5);
    for (int i = 0; i < 9000 / 8 + 200; ++i) {
        std::set<std::uint32_t> g;
        while (g.size() < 8) { g.insert(static_cast<std::uint32_t>(rng2() % 24576)); }
        big.step(std::vector<std::uint32_t>(g.begin(), g.end()), step);
    }
    const auto start = std::chrono::steady_clock::now();
    constexpr int kSteps = 2000;
    for (int i = 0; i < kSteps; ++i) {
        std::set<std::uint32_t> g;
        while (g.size() < 8) { g.insert(static_cast<std::uint32_t>(rng2() % 24576)); }
        big.step(std::vector<std::uint32_t>(g.begin(), g.end()), step, 1);
    }
    const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
    // The previous selection (a (score, key) pair per resident, partially sorted) on the same
    // residents, for the comparison; it must name the same key as a scan.
    std::vector<std::pair<double, std::uint32_t>> pairs;
    std::uint32_t named = 0;
    const auto start_sort = std::chrono::steady_clock::now();
    for (int i = 0; i < kSteps; ++i) {
        pairs.clear();
        for (std::uint32_t k : big.residents()) { pairs.emplace_back(big.score(k), k); }
        std::partial_sort(pairs.begin(), pairs.begin() + 1, pairs.end());
        named = pairs[0].second;
    }
    const double us_sort = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start_sort).count();
    std::uint32_t scanned = big.residents()[0];
    for (std::uint32_t k : big.residents()) {
        const double s = big.score(k), b = big.score(scanned);
        if (s < b || (s == b && k < scanned)) { scanned = k; }
    }
    check(named == scanned, "a scan names the partial sort's first resident");
    std::printf("budgeted LFRU step at 9,000 residents: %.2f us; one sort-based victim selection: %.2f us\n",
                us / kSteps, us_sort / kSteps);
}

void test_frame_epochs() {
    FramePool pool(3, 1);
    auto a = pool.acquire(), b = pool.acquire(), c = pool.acquire();
    check(a && b && c && !pool.acquire(), "three frames, then none");
    pool.retire(*a, 10); // round_started sampled as 10 after the ABSENT write became visible
    pool.on_round_done(10);
    check(!pool.is_free(*a), "round 11 may still read the frame");
    pool.on_round_done(11);
    check(pool.is_free(*a) && pool.acquire() == a, "free once round_done >= r_s + D");
    pool.retire(*b, 50);
    pool.on_quiescent();
    check(pool.is_free(*b), "quiescence frees every pending frame");
}

void test_pool_resize() {
    FramePool pool(4, 1, 8);
    check(pool.backed() == 4 && pool.max_frames() == 8, "4 of 8 frames backed");
    std::vector<std::uint32_t> held;
    while (auto f = pool.acquire()) { held.push_back(*f); }
    check(held.size() == 4, "only backed frames are handed out");
    pool.resize(6);
    auto e = pool.acquire(), f = pool.acquire();
    check(e && f && *e >= 4 && *f >= 4 && !pool.acquire(), "growing frees exactly the new frames");
    pool.release_now(*e);
    pool.release_now(*f);
    pool.release_now(held[0]);
    const auto low = pool.acquire_below(4);
    check(low && *low < 4, "a relocation target lies below the limit");
    pool.release_now(*low);
    pool.resize(4);
    check(pool.backed() == 4 && pool.free_count() == 1, "shrinking drops the top frames from the free list");
    bool threw = false;
    try {
        pool.resize(2); // frames 2 and 3 hold experts
    } catch (const std::logic_error&) { threw = true; }
    check(threw, "a shrink over held frames is refused");
}

// Fills a 12-frame pool, gives the experts distinct scores, shrinks to 5 and grows back to 9:
// victims are the 7 lowest-score experts wherever their frames are; every survivor ends below 5
// (relocated by kRelocate commands from its old frame); growing loads queued experts only into the
// new frames.
void test_controller_resize() {
    constexpr std::uint32_t kKeys = 64;
    CacheController cache(kKeys, 12, 0, 1, 16);
    std::vector<CacheController::Command> cmds;
    std::vector<std::uint32_t> frame_of(kKeys, ~0U);
    const auto apply = [&] {
        for (const auto& c : cmds) {
            if (c.kind == CacheController::Command::Kind::kCopy) { frame_of[c.key] = c.frame; }
            if (c.kind == CacheController::Command::Kind::kRelocate) {
                check(frame_of[c.key] == c.source, "a relocation starts from the expert's frame");
                frame_of[c.key] = c.frame;
            }
            if (c.kind == CacheController::Command::Kind::kWriteEntry &&
                ResidencyEntry::decode(c.word).state == ResidencyState::kAbsent) {
                frame_of[c.key] = ~0U;
            }
        }
        cmds.clear();
    };
    std::uint64_t round = 0;
    // Key k (0..11) is routed k + 1 times: key 11 scores highest.
    for (std::uint32_t k = 0; k < 12; ++k) {
        for (std::uint32_t use = 0; use <= k; ++use) {
            const std::uint32_t group[] = {k};
            cache.on_route(group, ++round, cmds);
            cache.on_quiescent(cmds);
            apply();
        }
    }
    check(cache.policy().resident_count() == 12 && cache.queued_loads() == 0, "12 experts resident");
    cache.resize(5, cmds);
    const std::vector<CacheController::Command> shrink = cmds;
    apply();
    check(cache.frames() == 5 && cache.policy().resident_count() == 5, "the pool and the policy shrink to 5");
    for (std::uint32_t k = 0; k < 12; ++k) {
        const bool kept = k >= 7;
        check(cache.policy().resident(k) == kept, "the 7 lowest-score experts are evicted");
        check(kept ? (frame_of[k] < 5 && cache.entry(k).state == ResidencyState::kReady &&
                      cache.entry(k).frame == frame_of[k])
                   : (frame_of[k] == ~0U && cache.entry(k).state == ResidencyState::kAbsent),
              "survivors sit below the new top, victims are absent");
    }
    const bool relocations_last = std::is_partitioned(shrink.begin(), shrink.end(), [](const auto& c) {
        return c.kind == CacheController::Command::Kind::kWriteEntry;
    });
    check(relocations_last, "evictions precede relocations");
    // Growing makes room: the next routes' experts load into the new frames.
    cache.resize(9, cmds);
    apply();
    check(cache.frames() == 9, "the pool grows to 9");
    for (std::uint32_t k = 20; k < 24; ++k) {
        const std::uint32_t group[] = {k};
        cache.on_route(group, ++round, cmds);
        cache.on_quiescent(cmds);
        apply();
    }
    std::set<std::uint32_t> frames;
    for (std::uint32_t k = 0; k < kKeys; ++k) {
        if (frame_of[k] != ~0U) {
            check(frame_of[k] < 9, "no expert above the grown top");
            check(frames.insert(frame_of[k]).second, "one expert per frame");
        }
    }
    check(cache.policy().resident_count() == 9, "the grown pool fills");
    // Shrinking to zero evicts everything and leaves a working, empty cache.
    cache.resize(0, cmds);
    apply();
    check(cache.frames() == 0 && cache.policy().resident_count() == 0, "a pool can shrink to no frames");
    const std::uint32_t group[] = {30};
    cache.on_route(group, ++round, cmds);
    check(cmds.empty() && cache.queued_loads() == 0, "an empty pool admits nothing");
}

// An agent-and-device simulation. Each round reads the frames named by READY entries; the agent
// applies the route log while that round runs and its commands become visible to later rounds.
// Invariants: a READY entry's frame holds its expert, and a frame is overwritten only after every
// round that read its previous expert has finished (design §9.5).
void test_agent_simulation() {
    const std::uint32_t keys = 6 * 64, frames = 120;
    // Slack covers peak admissions per round times (D + 1): ~27 per round here, held one extra round.
    CacheController cache(keys, frames + 48, 48, 1);
    std::vector<std::uint32_t> device_entry(keys, 0);
    std::vector<std::int64_t> frame_content(frames + 48, -1);
    std::vector<std::uint64_t> frame_last_read(frames + 48, 0);
    std::vector<CacheController::Command> cmds;
    std::mt19937 rng(77);
    std::uint64_t round_done = 0;
    long cpu_served = 0, gpu_served = 0, policy_hits = 0, violations = 0;
    auto apply = [&](std::uint64_t done) {
        for (const auto& c : cmds) {
            if (c.kind == CacheController::Command::Kind::kCopy) {
                if (frame_content[c.frame] >= 0 && frame_last_read[c.frame] > done) { ++violations; }
                frame_content[c.frame] = c.key;
            } else {
                device_entry[c.key] = c.word;
            }
        }
        cmds.clear();
    };
    for (std::uint64_t round = 1; round <= 3000; ++round) {
        for (int layer = 0; layer < 6; ++layer) {
            std::set<std::uint32_t> g;
            while (g.size() < 8) { g.insert(layer * 64 + static_cast<std::uint32_t>(rng() % (round % 400 < 200 ? 24 : 64))); }
            for (std::uint32_t k : g) {
                policy_hits += cache.policy().resident(k) ? 1 : 0;
                const auto e = ResidencyEntry::decode(device_entry[k]);
                if (e.state == ResidencyState::kReady) {
                    if (frame_content[e.frame] != static_cast<std::int64_t>(k)) { ++violations; }
                    frame_last_read[e.frame] = round;
                    ++gpu_served;
                } else {
                    ++cpu_served;
                }
            }
            const std::vector<std::uint32_t> group(g.begin(), g.end());
            cache.on_route(group, round, cmds);
            apply(round_done);
        }
        round_done = (rng() % 2 == 0 || round == 1) ? round : round - 1; // completion lags by 0 or 1
        cache.on_round_done(round_done, cmds);
        apply(round_done);
    }
    std::printf("agent simulation: %ld GPU-served, %ld CPU-served, %ld policy hits, %zu loads still queued\n",
                gpu_served, cpu_served, policy_hits, cache.queued_loads());
    check(violations == 0, "no frame reused while a round may read it; READY frames hold their expert");
    // Queued loads lag the policy only while frames retire: the device serves almost every policy hit.
    check(gpu_served >= policy_hits * 95 / 100, "device residency tracks the policy");
}

} // namespace

// The ABA case of load publication (design §19.3.2, "Frame lending and the two expert-cache fixes"):
// K loads into the only frame (serial s1); K2 evicts it and loads into the same frame; K re-enters it
// (serial s2) while s1's copy could still be in flight. Only the newest copy of a key in its frame
// may be published: complete_load(K, f, s1) is false, (K, f, s2) true, and K2's copy is stale too.
void test_load_serials() {
    constexpr std::uint32_t kKeys = 8, K = 3, K2 = 5;
    CacheController cache(kKeys, 1, 0, 1);
    std::vector<CacheController::Command> cmds;
    struct Copy {
        std::uint32_t key, frame;
        std::uint64_t serial;
    };
    std::vector<Copy> copies;
    std::uint64_t round = 0;
    // Routes `key` until its copy is issued; returns that copy.
    const auto admit = [&](std::uint32_t key) -> std::optional<Copy> {
        for (int i = 0; i < 64; ++i) {
            const std::uint32_t group[] = {key};
            cache.on_route(group, ++round, cmds);
            cache.on_quiescent(cmds);
            std::optional<Copy> issued;
            for (const auto& c : cmds) {
                if (c.kind == CacheController::Command::Kind::kCopy && c.key == key) { issued = Copy{c.key, c.frame, c.serial}; }
            }
            cmds.clear();
            if (issued) { return issued; }
        }
        return std::nullopt;
    };
    const auto first = admit(K);
    check(first.has_value(), "K loads into the empty cache");
    if (!first) { return; }
    check(cache.complete_load(K, first->frame, first->serial), "K's only copy publishes");
    const auto other = admit(K2);
    check(other.has_value() && other->frame == first->frame, "K2 evicts K and loads into the same frame");
    if (!other) { return; }
    check(!cache.complete_load(K, first->frame, first->serial), "an evicted key's copy does not publish");
    const auto again = admit(K);
    check(again.has_value() && again->frame == first->frame, "K re-enters the same frame");
    if (!again) { return; }
    check(again->serial != first->serial, "each copy has its own serial");
    check(!cache.complete_load(K, first->frame, first->serial), "K's first copy (ABA) does not publish");
    check(cache.complete_load(K, again->frame, again->serial), "K's newest copy publishes");
    check(!cache.complete_load(K2, other->frame, other->serial), "the evicted K2's copy does not publish");
}

// Frame lending (design §19.3.2, VT4): choose_run picks the run of held experts with the minimum
// summed LFRU score, never a lent frame, and avoids busy frames while a run without them exists;
// lend evicts every expert in the run (ABSENT, no frame maps to it) and shrinks the capacity by
// the run; give_back restores the capacity and loads queued experts into it. With lending, the ABA
// case: K in frame f (serial s1), f lent and given back, K re-admitted into f (s2): s1 does not
// publish, s2 does. Every frame is back after the last give_back.
void test_lending() {
    constexpr std::uint32_t kKeys = 64, kFrames = 8;
    CacheController cache(kKeys, kFrames, 0, 1);
    std::vector<CacheController::Command> cmds;
    std::vector<std::uint32_t> frame_of(kKeys, ~0U);
    std::vector<std::uint64_t> serial_of(kKeys, 0);
    const auto apply = [&] {
        for (const auto& c : cmds) {
            if (c.kind == CacheController::Command::Kind::kCopy) {
                frame_of[c.key]  = c.frame;
                serial_of[c.key] = c.serial;
            }
            if (c.kind == CacheController::Command::Kind::kWriteEntry &&
                ResidencyEntry::decode(c.word).state == ResidencyState::kAbsent) {
                frame_of[c.key] = ~0U;
            }
        }
        cmds.clear();
    };
    std::uint64_t round = 0;
    // Keys 0..7 fill the frames; key k is routed k + 1 times (key 7 scores highest).
    for (std::uint32_t k = 0; k < kFrames; ++k) {
        for (std::uint32_t use = 0; use <= k; ++use) {
            const std::uint32_t group[] = {k};
            cache.on_route(group, ++round, cmds);
            cache.on_quiescent(cmds);
            apply();
        }
    }
    bool all = true;
    for (std::uint32_t k = 0; k < kFrames; ++k) { all &= frame_of[k] != ~0U; }
    check(all, "eight keys fill the eight frames");

    // The lowest summed score over 3 contiguous frames, by brute force.
    const auto brute = [&](std::uint32_t count, const std::vector<std::uint8_t>& busy, bool avoid) {
        std::optional<std::uint32_t> best;
        double best_score = 0.0;
        for (std::uint32_t first = 0; first + count <= kFrames; ++first) {
            double sum = 0.0;
            bool ok    = true;
            for (std::uint32_t f = first; f < first + count; ++f) {
                ok &= !(avoid && busy[f]);
                const auto key = cache.frame_key(f);
                sum += key ? cache.policy().score(*key) : 0.0;
            }
            if (ok && (!best || sum < best_score)) {
                best       = first;
                best_score = sum;
            }
        }
        return best;
    };
    const std::vector<std::uint8_t> none(kFrames, 0);
    check(cache.choose_run(3, none) == brute(3, none, false), "choose_run picks the minimum-score run");
    std::vector<std::uint8_t> busy(kFrames, 0);
    if (const auto run = cache.choose_run(3, none)) { busy[*run] = 1; }
    const auto avoided = cache.choose_run(3, busy);
    check(avoided.has_value() && avoided == brute(3, busy, true), "choose_run avoids busy frames while it can");
    check(cache.choose_run(kFrames, std::vector<std::uint8_t>(kFrames, 1)) == std::optional<std::uint32_t>(0),
          "choose_run takes busy frames when no run avoids them");

    // K = the key in the first frame of the run; lend the run.
    const std::uint32_t first = *cache.choose_run(3, none);
    const std::uint32_t K     = *cache.frame_key(first);
    const std::uint64_t s1    = serial_of[K];
    const std::uint32_t capacity = cache.policy().capacity();
    const std::uint32_t evicted  = cache.lend(first, 3, cmds);
    apply();
    check(evicted == 3, "lend evicts the three experts in the run");
    check(cache.loaned_frames() == 3 && cache.policy().capacity() == capacity - 3, "lend shrinks the capacity by the run");
    bool clear = true;
    for (std::uint32_t f = first; f < first + 3; ++f) { clear &= !cache.frame_key(f).has_value(); }
    for (std::uint32_t k = 0; k < kKeys; ++k) { clear &= frame_of[k] == ~0U || frame_of[k] < first || frame_of[k] >= first + 3; }
    check(clear, "no expert maps to a lent frame");
    check(cache.entry(K).state == ResidencyState::kAbsent, "the run's experts are ABSENT");
    check(!cache.choose_run(kFrames, none).has_value(), "lent frames are never chosen");

    // Routes K until it is resident again (queued while the frames are lent, then loaded).
    const auto route_k = [&] {
        for (int i = 0; i < 64 && frame_of[K] == ~0U; ++i) {
            const std::uint32_t group[] = {K};
            cache.on_route(group, ++round, cmds);
            cache.on_quiescent(cmds);
            apply();
        }
    };
    cache.give_back(first, 3, cmds);
    apply();
    check(cache.loaned_frames() == 0 && cache.policy().capacity() == capacity, "give_back restores the capacity");
    route_k();
    check(frame_of[K] != ~0U, "K is resident again after the give-back");
    check(!cache.complete_load(K, first, s1), "K's copy from before the loan (ABA) does not publish");
    check(cache.complete_load(K, frame_of[K], serial_of[K]) && serial_of[K] != s1, "K's newest copy publishes");
    bool threw = false;
    try {
        cache.give_back(first, 1, cmds);
    } catch (const std::logic_error&) { threw = true; }
    check(threw, "a frame is given back only while lent");
}

// Refilling lending (a layer split's slow-link segment, which promotes nothing during requests):
// lend with refill remembers the experts it evicts, and give_back with refill loads them back into
// the returned frames (one copy each, into distinct frames, READY once landed); without refill the
// returned frames stay empty. A key that came back meanwhile is not loaded twice.
void test_lending_refill() {
    constexpr std::uint32_t kKeys = 64, kFrames = 8;
    for (const bool refill : {false, true}) {
        CacheController cache(kKeys, kFrames, 0, 1);
        std::vector<CacheController::Command> cmds;
        std::vector<std::uint32_t> ranked;
        for (std::uint32_t k = 0; k < kFrames; ++k) { ranked.push_back(k); }
        check(cache.seed(ranked, {}, cmds) == kFrames, "the warm start fills every frame");
        cmds.clear();
        const std::vector<std::uint8_t> none(kFrames, 0);
        const std::uint32_t first = *cache.choose_run(3, none);
        std::vector<std::uint32_t> lent;
        for (std::uint32_t f = first; f < first + 3; ++f) { lent.push_back(*cache.frame_key(f)); }
        check(cache.lend(first, 3, cmds, refill) == 3, "lend evicts the run's three experts");
        cmds.clear();
        const std::uint32_t loaded = cache.give_back(first, 3, cmds, refill);
        std::set<std::uint32_t> frames, keys;
        for (const auto& c : cmds) {
            if (c.kind == CacheController::Command::Kind::kCopy) {
                frames.insert(c.frame);
                keys.insert(c.key);
            }
        }
        if (!refill) {
            check(loaded == 0 && keys.empty(), "without refill the returned frames stay empty");
            continue;
        }
        check(loaded == 3 && keys == std::set<std::uint32_t>(lent.begin(), lent.end()) && frames.size() == 3,
              "give_back with refill loads the three evicted experts into distinct frames");
        bool loading = true;
        for (const std::uint32_t key : lent) { loading &= cache.entry(key).state != ResidencyState::kAbsent; }
        check(loading, "the refilled experts are no longer ABSENT");
        // Lend again: each evicted key is loaded back exactly once.
        cmds.clear();
        const std::uint32_t again = *cache.choose_run(2, none);
        const std::uint32_t K     = *cache.frame_key(again);
        cache.lend(again, 2, cmds, true);
        cmds.clear();
        cache.give_back(again, 2, cmds, true);
        std::size_t copies_of_k = 0;
        for (const auto& c : cmds) {
            copies_of_k += c.kind == CacheController::Command::Kind::kCopy && c.key == K ? 1U : 0U;
        }
        check(copies_of_k == 1, "an evicted key is loaded back once");
    }
}

// A lease held across a prompt's calls starts the pool (choose_run at_start, the Vision handoff):
// a one-frame lease where the cheapest frame sits can split the pool so that no run of most of its
// frames remains (the prompt's stream lease on a layer split's 16 GB head needs over half of them);
// at the start, every other frame stays one run. With the start lent (another media prompt's lease,
// several prompts in flight), at_start takes the first fit past it, never the cheapest run: the
// leases stack at the bottom, and one that fits a hole a returned lease left takes it. longest_run
// tells the run a later lease can take, with leases about to come back counted free and one about
// to be made counted taken (ExpertResidency::lend_fit).
void test_lending_at_start() {
    constexpr std::uint32_t kKeys = 64, kFrames = 8;
    CacheController cache(kKeys, kFrames, 0, 1);
    std::vector<CacheController::Command> cmds;
    std::vector<std::uint32_t> ranked;
    for (std::uint32_t k = 0; k < kFrames; ++k) { ranked.push_back(k); }
    check(cache.seed(ranked, {}, cmds) == kFrames, "the warm start fills every frame");
    cmds.clear();
    // Frame 4's expert is routed least, so the cheapest single frame lies inside the pool.
    std::uint64_t round = 0;
    for (std::uint32_t k = 0; k < kFrames; ++k) {
        const auto key = cache.frame_key(k);
        if (!key || k == 4) { continue; }
        for (int use = 0; use < 4; ++use) {
            const std::uint32_t group[] = {*key};
            cache.on_route(group, ++round, cmds);
            cache.on_quiescent(cmds);
            cmds.clear();
        }
    }
    const std::vector<std::uint8_t> none(kFrames, 0);
    check(cache.choose_run(1, none) == std::optional<std::uint32_t>(4), "the cheapest frame is inside the pool");
    check(cache.choose_run(1, none, true) == std::optional<std::uint32_t>(0), "at_start takes the pool's first frame");
    check(cache.choose_run(2, std::vector<std::uint8_t>(kFrames, 1), true) == std::optional<std::uint32_t>(0),
          "at_start takes busy frames (the lease waits for their loads)");
    cache.lend(0, 1, cmds);
    cmds.clear();
    check(cache.choose_run(kFrames - 1, none) == std::optional<std::uint32_t>(1),
          "after a lease at the start every other frame is one run");
    check(cache.choose_run(1, none, true) == std::optional<std::uint32_t>(1),
          "with the start lent, at_start takes the first fit past it (not the cheapest frame, 4)");
    check(cache.longest_run() == kFrames - 1, "longest_run: every frame past the lease");
    cache.lend(1, 2, cmds); // a second media prompt's lease stacks on the first
    cmds.clear();
    check(cache.longest_run() == kFrames - 3, "two leases at the bottom leave the rest in one run");
    cache.give_back(0, 1, cmds); // the first prompt ends
    cmds.clear();
    check(cache.choose_run(1, none, true) == std::optional<std::uint32_t>(0),
          "a lease that fits the hole a returned one left takes it");
    check(cache.choose_run(2, none, true) == std::optional<std::uint32_t>(3),
          "a larger one takes the first fit past the leases");
    check(cache.longest_run({}, {3, 2}) == kFrames - 5, "longest_run with a lease about to be made");
    // A lease inside the pool (a prompt's stream lease) splits it; counted as coming back, it joins
    // its neighbours again.
    cache.lend(5, 1, cmds);
    cmds.clear();
    check(cache.longest_run() == 2, "a lease inside the pool splits the rest");
    const CacheController::Run back[] = {{5, 1}};
    check(cache.longest_run(back) == kFrames - 3, "a lease counted as coming back joins its neighbours");
    check(!cache.choose_run(3, none, true).has_value(), "no run of three frames while the pool is split");
    cache.give_back(5, 1, cmds);
    cache.give_back(1, 2, cmds);
    cmds.clear();
    check(cache.longest_run() == kFrames, "every frame back: one run");
    cache.lend(4, 1, cmds);
    cmds.clear();
    check(!cache.choose_run(kFrames - 1, none).has_value(), "a lease inside the pool splits it");
    cache.give_back(4, 1, cmds);
    cmds.clear();
    check(cache.loaned_frames() == 0, "every frame is back");
}

// Warm start (design §19.3.5 S4b): seed sets the counts, then loads the ranked keys best first into
// free frames while the policy has capacity (frames - slack), skipping keys already resident or out
// of range; each loaded key issues one copy into a distinct frame and publishes when that copy lands.
void test_seed() {
    constexpr std::uint32_t kKeys = 64, kFrames = 10, kSlack = 2;
    CacheController cache(kKeys, kFrames, kSlack, 1);
    std::vector<std::uint32_t> counts(kKeys, 0);
    for (std::uint32_t k = 0; k < kKeys; ++k) { counts[k] = k % 7; }
    const std::vector<std::uint32_t> ranked{40, 3, 40, 99, 7, 12, 5, 6, 8, 9, 10, 11, 13};
    std::vector<CacheController::Command> cmds;
    const std::uint32_t loaded = cache.seed(ranked, counts, cmds);
    check(loaded == kFrames - kSlack, "seed fills the policy's capacity (frames - slack)");
    std::set<std::uint32_t> frames, keys;
    for (const auto& c : cmds) {
        if (c.kind == CacheController::Command::Kind::kCopy) {
            frames.insert(c.frame);
            keys.insert(c.key);
            check(cache.complete_load(c.key, c.frame, c.serial), "a seeded copy publishes");
        }
    }
    check(frames.size() == loaded && keys.size() == loaded, "each seeded key copies into its own frame");
    check(keys.count(40) == 1 && keys.count(3) == 1 && keys.count(99) == 0, "duplicates and out-of-range keys are skipped");
    check(keys.count(11) == 0 && keys.count(13) == 0, "keys past the capacity are not loaded");
    check(cache.policy().resident(40) && cache.policy().count(40) == 40 % 7, "seeded keys are resident with their counts");
    check(cache.policy().count(50) == 50 % 7, "counts are seeded for every key");
    cmds.clear();
    check(cache.seed(ranked, {}, cmds) == 0 && cmds.empty(), "a full cache seeds nothing more");
}

// The saved state file: a round trip keeps counts and ranking; another identity, key count, version
// or a truncated file loads nothing (with a reason).
void test_state_file() {
    const auto dir  = std::filesystem::temp_directory_path();
    const auto path = dir / "infernix_expert_state_test.bin";
    SavedState state;
    state.counts = {5, 0, 9, 1};
    state.ranked = {2, 0, 3};
    check(save_expert_state(path, "artifact A", state).empty(), "the state saves");
    const ExpertStateLoad same = load_expert_state(path, "artifact A", 4);
    check(same.state && same.state->counts == state.counts && same.state->ranked == state.ranked, "a saved state round-trips");
    check(!load_expert_state(path, "artifact B", 4).state, "another artifact's state is ignored");
    check(!load_expert_state(path, "artifact A", 5).state, "a state with another key count is ignored");
    check(!load_expert_state(dir / "infernix_expert_state_missing.bin", "artifact A", 4).state, "a missing file loads nothing");
    {
        const auto size = std::filesystem::file_size(path);
        std::filesystem::resize_file(path, size - 3);
    }
    const ExpertStateLoad cut = load_expert_state(path, "artifact A", 4);
    check(!cut.state && !cut.message.empty(), "a truncated file is refused with a reason");
    std::filesystem::remove(path);
    check(!std::filesystem::exists(path.string() + ".tmp"), "no temporary file is left behind");
}

// A layer split's saved state (multi-GPU WP11): split_saved_state keeps a range's counts and its
// ranked keys, in the global order, as local keys; merge_saved_states puts the counts back at their
// layers and interleaves the rankings in proportion to their lengths, each in its own order. A
// split after a merge returns every part exactly, a merged state round-trips through split and
// merge, and one range over the whole model changes nothing.
void test_saved_state_split() {
    constexpr std::uint32_t kLayers = 48, kExperts = 8, kKeys = kLayers * kExperts;
    SavedState global;
    global.counts.resize(kKeys);
    for (std::uint32_t k = 0; k < kKeys; ++k) { global.counts[k] = (k * 7U) % 13U; }
    std::vector<std::uint32_t> keys(kKeys);
    for (std::uint32_t k = 0; k < kKeys; ++k) { keys[k] = k; }
    std::mt19937 rng(11);
    std::shuffle(keys.begin(), keys.end(), rng);
    global.ranked.assign(keys.begin(), keys.begin() + 150);

    const LayerRange ranges[] = {{0, 24}, {24, kLayers}};
    const auto split_all      = [&](const SavedState& state) {
        std::vector<SavedState> out;
        for (const LayerRange& range : ranges) {
            out.push_back(split_saved_state(state, range.begin, range.end, kExperts));
        }
        return out;
    };
    const std::vector<SavedState> parts = split_all(global);
    bool sliced                         = true;
    for (std::size_t i = 0; i < parts.size(); ++i) {
        const std::uint32_t base = ranges[i].begin * kExperts, local = ranges[i].size() * kExperts;
        sliced = sliced && parts[i].counts.size() == local &&
                 std::equal(parts[i].counts.begin(), parts[i].counts.end(), global.counts.begin() + base);
        std::vector<std::uint32_t> expected;
        for (const std::uint32_t key : global.ranked) {
            if (key >= base && key < base + local) { expected.push_back(key - base); }
        }
        sliced = sliced && parts[i].ranked == expected;
    }
    check(sliced, "a split keeps its layers' counts and ranked keys, in the global order, as local keys");
    check(parts[0].ranked.size() + parts[1].ranked.size() == global.ranked.size(),
          "every ranked key lands in one part");

    const SavedState merged = merge_saved_states(parts, ranges, kLayers, kExperts);
    check(merged.counts == global.counts, "the merged counts equal the global counts");
    std::vector<std::uint32_t> merged_keys = merged.ranked, global_keys = global.ranked;
    std::sort(merged_keys.begin(), merged_keys.end());
    std::sort(global_keys.begin(), global_keys.end());
    check(merged_keys == global_keys, "the merged ranking holds the global ranking's keys");
    const std::vector<SavedState> again = split_all(merged);
    bool exact                          = true;
    for (std::size_t i = 0; i < parts.size(); ++i) {
        exact = exact && again[i].counts == parts[i].counts && again[i].ranked == parts[i].ranked;
    }
    check(exact, "a split after a merge returns every part exactly");
    const SavedState remerged = merge_saved_states(again, ranges, kLayers, kExperts);
    check(remerged.counts == merged.counts && remerged.ranked == merged.ranked,
          "a merged state round-trips through split and merge");
    const SavedState whole        = split_saved_state(global, 0, kLayers, kExperts);
    const LayerRange everything[] = {{0, kLayers}};
    const SavedState rewhole =
        merge_saved_states(std::span<const SavedState>(&whole, 1), everything, kLayers, kExperts);
    check(whole.counts == global.counts && whole.ranked == global.ranked && rewhole.counts == global.counts &&
              rewhole.ranked == global.ranked,
          "a whole-model split and merge change nothing");

    // Fractions (n + 1/2) / size: 1/4, 3/4 for the first part, 1/12 ... 11/12 for the second; the
    // equal fractions 1/4 and 3/4 go to the first part.
    SavedState first, second;
    first.ranked                  = {5, 1};
    second.ranked                 = {0, 1, 2, 3, 4, 5};
    const SavedState two_parts[]  = {first, second};
    const LayerRange two_layers[] = {{0, 1}, {1, 2}};
    const SavedState mixed        = merge_saved_states(two_parts, two_layers, 2, kExperts);
    const std::vector<std::uint32_t> interleaved = {8, 5, 9, 10, 11, 1, 12, 13};
    check(mixed.ranked == interleaved, "rankings interleave in proportion to their lengths, ties to the earlier part");
    check(mixed.counts.empty(), "parts without counts merge to no counts");

    SavedState counted;
    counted.counts.assign(kExperts, 3);
    counted.ranked            = {2, 200}; // 200 is past the part's one layer
    const LayerRange middle[] = {{1, 2}};
    const SavedState partial  = merge_saved_states(std::span<const SavedState>(&counted, 1), middle, 3, kExperts);
    bool zeros                = partial.counts.size() == 3 * kExperts;
    for (std::uint32_t k = 0; zeros && k < 3 * kExperts; ++k) {
        zeros = partial.counts[k] == (k / kExperts == 1 ? 3U : 0U);
    }
    check(zeros, "layers no part covers merge with zero counts");
    check(partial.ranked == std::vector<std::uint32_t>{kExperts + 2}, "a part's keys past its range are dropped");

    const auto refused = [](const auto& call) {
        try {
            call();
        } catch (const std::invalid_argument&) { return true; }
        return false;
    };
    const SavedState none[2]        = {};
    const LayerRange overlapping[]  = {{0, 2}, {1, 3}};
    const LayerRange past_the_end[] = {{0, 2}, {2, 4}};
    const LayerRange shifted[]      = {{0, 23}, {23, kLayers}};
    check(refused([&] { (void)merge_saved_states(none, overlapping, 3, kExperts); }),
          "overlapping ranges are refused");
    check(refused([&] { (void)merge_saved_states(none, past_the_end, 3, kExperts); }),
          "a range past the last layer is refused");
    check(refused([&] { (void)merge_saved_states(none, everything, kLayers, kExperts); }),
          "a merge needs one range per part");
    check(refused([&] { (void)merge_saved_states(parts, shifted, kLayers, kExperts); }),
          "a part whose counts differ from its range's keys is refused");
    check(refused([&] { (void)split_saved_state(global, 24, kLayers + 1, kExperts); }),
          "a split past the state's layers is refused");
    check(refused([&] { (void)split_saved_state(global, 5, 5, kExperts); }), "an empty split is refused");
}

// Fill-phase landing (design §19.3.5 S4): reserve_free pops free frames only while the policy has
// room; adopt makes a landed key READY and resident in its frame, drops its queued load (no second
// copy), refuses a key loading elsewhere, and release returns unused frames.
void test_landing() {
    constexpr std::uint32_t kKeys = 32, kFrames = 6;
    CacheController cache(kKeys, kFrames, 0, 1);
    std::vector<CacheController::Command> cmds;
    std::vector<std::uint32_t> reserved;
    check(cache.reserve_free(4, reserved) == 4 && cache.free_frames() == kFrames - 4, "reserve_free pops free frames");
    // Key 3 lands in reserved[0]; key 5 is loading elsewhere when it lands: refused.
    const std::uint32_t group[] = {3, 5};
    check(cache.adopt(3, reserved[0], group, 1, cmds), "a landed key is adopted");
    check(cache.entry(3).state == ResidencyState::kReady && cache.entry(3).frame == reserved[0] &&
              cache.policy().resident(3) && cache.frame_key(reserved[0]) == 3U,
          "an adopted key is READY and resident in its frame");
    check(!cache.adopt(3, reserved[1], group, 1, cmds), "a resident key is not adopted twice");
    cache.release(reserved[1]);
    cmds.clear();
    const std::uint32_t five[] = {5};
    cache.on_route(five, 1, cmds); // admitted and loaded into a free frame
    check(cache.entry(5).state != ResidencyState::kAbsent, "key 5 loads through on_route");
    check(!cache.adopt(5, reserved[2], group, 1, cmds), "a key loading elsewhere is refused");
    cache.release(reserved[2]);
    cache.release(reserved[3]);
    check(cache.free_frames() == kFrames - 2, "released frames return to the pool");
    // Room: the policy holds 2 of 6; reserve_free never reserves past capacity - residents.
    reserved.clear();
    check(cache.reserve_free(10, reserved) == 4, "reserve_free stops at the policy's room");
    for (const auto f : reserved) { cache.release(f); }
}

// The SSD tier's pieces (design §19.3.7): the admission filter and the eviction gate of the LFRU step,
// and a demoted expert's frame held past quiescence until its copy completes.
void test_tier_gates() {
    LfruPolicy policy(16, 2);
    LfruPolicy::Step step;
    const LfruPolicy::Admissible only_even = [](std::uint32_t key) { return key % 2 == 0; };
    const std::uint32_t first[] = {0, 1, 2};
    policy.step(first, step, static_cast<std::size_t>(-1), &only_even);
    check(policy.resident(0) && policy.resident(2) && !policy.resident(1) && step.misses.size() == 3,
          "a key without a host copy counts as used but is not admitted");
    // A hotter key would replace the coldest resident, but the gate refuses that victim.
    for (int i = 0; i < 5; ++i) {
        const std::uint32_t hot[] = {4};
        const LfruPolicy::Admissible refuse = [](std::uint32_t) { return false; };
        policy.step(hot, step, static_cast<std::size_t>(-1), nullptr, &refuse);
        check(step.victims.empty() && !policy.resident(4), "a refused victim stops the admission");
    }
    const std::uint32_t hot[] = {4};
    const LfruPolicy::Admissible allow = [](std::uint32_t) { return true; };
    policy.step(hot, step, static_cast<std::size_t>(-1), nullptr, &allow);
    check(policy.resident(4) && step.victims.size() == 1, "an allowed victim makes room");

    CacheController cache(8, 2, 0, 1);
    std::vector<CacheController::Command> cmds;
    const std::uint32_t a[] = {1, 2};
    cache.on_route(a, 1, cmds);
    cache.on_quiescent(cmds);
    std::uint32_t frame_of_1 = cache.entry(1).frame;
    // Key 3 replaces key 1 (key 2 is in the group); key 1's frame is held for its demotion.
    for (int i = 0; i < 3; ++i) {
        const std::uint32_t b[] = {2, 3};
        cmds.clear();
        cache.on_route(b, 2, cmds);
    }
    if (cache.entry(1).state == ResidencyState::kAbsent) {
        cache.hold(frame_of_1);
        cmds.clear();
        cache.on_quiescent(cmds);
        check(cache.free_frames() == 0 && cache.entry(3).state == ResidencyState::kAbsent,
              "a held frame is not freed by quiescence and its newcomer waits");
        cmds.clear();
        cache.release_held(frame_of_1, cmds);
        check(cache.entry(3).state != ResidencyState::kAbsent && cache.entry(3).frame == frame_of_1,
              "releasing the held frame loads the queued newcomer into it");
    } else {
        check(false, "key 1 should have been evicted");
    }
}

int main() {
    test_residency_entries();
    test_tier_gates();
    test_lfru_conformance();
    test_budgeted_admission();
    test_frame_epochs();
    test_agent_simulation();
    test_pool_resize();
    test_controller_resize();
    test_load_serials();
    test_lending();
    test_lending_refill();
    test_lending_at_start();
    test_seed();
    test_landing();
    test_state_file();
    test_saved_state_split();
    if (g_failures) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("all expert cache checks passed\n");
    return 0;
}
