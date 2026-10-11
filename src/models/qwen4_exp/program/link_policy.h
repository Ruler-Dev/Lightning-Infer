#pragma once

// A segment's choices that follow its host-to-device link
// (docs/maintainer/qwen3_8-flash-next-design.md §19.3.12: bandwidth-dependent choices follow the
// machine), derived from the link's measured rate (or ProgramOptions::link_bytes_per_second,
// INFERNIX_LINK_GBPS). Each split segment measures its own device's link, so one process can run an
// x16 card with today's dynamic cache beside an x4 card that serves most misses from the CPU and
// trickles a few promotions per round over its link.
//
// The Program takes its link-scaled constants from derive_link_policy (the PCIe share of decode
// misses, the steady-state promotion interval and the prefill CPU split's widest call), whose
// expressions are the ones it computed itself before. A layer split's segments also apply the
// promotion policy, warm-start seed share and CPU jobs it picks (an unsplit Program keeps the
// dynamic cache on every link).
//
// Pure host data, no CUDA.

#include "ops/offloaded_sparse_moe/cpu/miss_request.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace infernix::models::qwen4_exp {

// The rate at which the link-dependent constants were fitted (RTX 5090 at PCIe 5.0 x8).
inline constexpr double kReferenceLinkBytesPerSecond = 27.5e9;
// Below this rate (PCIe 3.0 x4, or 4.0 x4 behind a chipset) a promoted 2.64 MiB expert costs about
// 0.9 ms of link time, more than a CPU-served miss (80-125 us on DDR4): such a segment fills every
// frame at a warm start, gives the CPU more jobs and promotes only a trickle per round, at a point
// where its link is idle (after its layers of the next round, while the next segment runs).
inline constexpr double kSlowLinkBytesPerSecond = 8e9;
inline constexpr std::uint32_t kSlowLinkCpuJobs = 32;
// A slow link's promotion bytes per round: the link time of kSlowLinkTrickleSeconds (6 ms: about
// six 2.64 MiB experts at 3.1 GB/s), well inside the time a decode round's later segments keep it
// idle (about 14-16 ms on an RTX PRO 4000 + RTX 5080 split at K = 20-24).
inline constexpr double kSlowLinkTrickleSeconds = 6e-3;

// When a segment's expert cache promotes misses into frames.
//   Dynamic:  today's LFRU: every miss-driven promotion the decode budget allows, scaled by
//             fill_scale while the frames fill;
//   Frozen:   never (the warm start's set stays);
//   IdleOnly: never while requests run (between them at most idle_refill_promotions per tick: not
//             implemented, so the Program treats it as Frozen);
//   Trickle:  at most trickle_bytes per round (the slow link's policy).
enum class PromotionPolicy : std::uint8_t { Dynamic, Frozen, IdleOnly, Trickle };

// The defaults are the reference link's policy.
struct LinkPolicy {
    double bytes_per_second              = 27.5e9;
    int pcie_divisor                     = 3;    // max(2, round(1 + 2*27.5e9/rate))
    std::uint64_t promotion_interval     = 4;    // clamp(round(4*27.5e9/rate), 1, 8)
    std::int32_t split_columns           = 1536; // min(cpu_split_columns*27.5e9/rate, 4096)
    double fill_scale                    = 1.0;  // min(1, rate/27.5e9)
    PromotionPolicy promotions           = PromotionPolicy::Dynamic;
    double seed_share                    = 0.5; // the share of the frames a warm start fills
    std::uint32_t cpu_jobs               = 16;  // the most misses one layer call hands to the CPU
    std::uint64_t trickle_bytes          = 0;   // Trickle: promotion bytes per round
    std::uint32_t idle_refill_promotions = 32;  // IdleOnly: promotions per idle maintenance tick
};

// The ProgramOptions the link-scaled constants start from: cpu_split_columns (on the reference
// link), cpu_expert_jobs and cpu_pcie_divisor (> 0 fixes the divisor at every rate).
struct LinkPolicyInputs {
    double cpu_split_columns      = 1536;
    std::uint32_t cpu_expert_jobs = 16;
    int cpu_pcie_divisor_override = 0;
};

// The policy of a link of `rate` bytes per second; a rate that is not positive and finite (nothing
// measured) yields the reference link's, as the single-GPU Program keeps the reference rate then.
// rate < 8e9: Trickle (rate * 6 ms per round), seed 1.0 and at least 32 CPU jobs (0, CPU-served
// misses disabled, stays 0).
[[nodiscard]] inline LinkPolicy derive_link_policy(double rate,
                                                   const LinkPolicyInputs& in) noexcept {
    if (!(rate > 0.0) || !std::isfinite(rate)) { rate = kReferenceLinkBytesPerSecond; }
    constexpr double kReference = kReferenceLinkBytesPerSecond;
    // The expressions below are the single-GPU Program's, operation for operation, so the results
    // are bit-identical. Rounding after a clamp to integral bounds equals clamping after rounding,
    // and cannot overflow `long` (32 bits on Windows) on an absurdly slow link.
    const auto rounded = [](double x, double lo, double hi) {
        return std::lround(std::clamp(x, lo, hi));
    };
    constexpr double kMaxDivisor = std::numeric_limits<std::int32_t>::max();

    LinkPolicy p;
    p.bytes_per_second = rate;
    // The share of a decode call's misses the PCIe stage keeps (misses / divisor): 3 measured
    // fastest on the reference link; a faster link carries more of them (x16: 2), a slower one
    // fewer.
    p.pcie_divisor =
        in.cpu_pcie_divisor_override > 0
            ? in.cpu_pcie_divisor_override
            : static_cast<int>(rounded(1.0 + 2.0 * kReference / rate, 2.0, kMaxDivisor));
    // Once the frames are full, decode promotes every 4th token on the reference link; a faster
    // link leaves more idle time per round, so it promotes proportionally more often.
    p.promotion_interval = static_cast<std::uint64_t>(rounded(4.0 * kReference / rate, 1.0, 8.0));
    // The prefill CPU split's widest call scales inversely with the link (the bytes it saves cost
    // less on a faster one), at most a CPU call's columns.
    p.split_columns = static_cast<std::int32_t>(std::min<double>(
        in.cpu_split_columns * kReference / rate, ops::offloaded_moe::kMaxCpuCallColumns));
    // Fill-phase promotions per layer and token: one on the reference link or faster.
    p.fill_scale = std::min(1.0, rate / kReference);
    if (rate < kSlowLinkBytesPerSecond) {
        p.promotions    = PromotionPolicy::Trickle;
        p.trickle_bytes = static_cast<std::uint64_t>(rate * kSlowLinkTrickleSeconds);
        p.seed_share    = 1.0;
        p.cpu_jobs   = in.cpu_expert_jobs > 0 ? std::max(in.cpu_expert_jobs, kSlowLinkCpuJobs) : 0;
    } else {
        p.promotions = PromotionPolicy::Dynamic;
        p.seed_share = 0.5;
        p.cpu_jobs   = in.cpu_expert_jobs;
    }
    return p;
}

} // namespace infernix::models::qwen4_exp
