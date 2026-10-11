// The link policies (host only). At the measured rates of this machine's links (RTX PRO 4000 on
// PCIe 3.0 x4: 3.1 GB/s; the RTX 5080 right after a large pin: 7.4 GB/s, settled on 4.0 x16:
// 28.4 GB/s) derive_link_policy returns exactly the single-GPU Program's PCIe divisor, promotion
// interval and CPU split width, both against the Program's own expressions and against values
// computed by hand; a divisor override wins at every rate; links below 8 GB/s stop promoting during
// requests, seed every frame and hand the CPU 32 jobs; an unmeasured rate gives the reference
// policy.

#include "models/qwen4_exp/program/link_policy.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>

using namespace infernix::models::qwen4_exp;

namespace {

void require(bool value, const std::string& message) {
    if (!value) { throw std::runtime_error(message); }
}

// The single-GPU Program's link-scaled constants as ProgramImpl computed them itself before it took
// them from derive_link_policy (pcie_divisor, promotion_interval, split_columns), with its
// constants kReferenceLinkBytesPerSecond = 27.5e9, kDecodePromotionInterval = 4 and
// kMaxCpuCallColumns = 4096 spelled out.
struct Today {
    int pcie_divisor;
    std::uint64_t promotion_interval;
    std::int32_t split_columns;
};

Today today(double link, std::int32_t cpu_pcie_divisor, std::uint32_t cpu_split_columns) {
    constexpr std::uint64_t kInterval = 4;
    Today t{};
    t.pcie_divisor       = cpu_pcie_divisor > 0
                               ? cpu_pcie_divisor
                               : std::max(2, static_cast<int>(std::lround(1.0 + 2.0 * 27.5e9 / link)));
    const double scaled  = static_cast<double>(kInterval) * 27.5e9 / link;
    t.promotion_interval = static_cast<std::uint64_t>(std::clamp(std::lround(scaled), 1L, 8L));
    const double columns = cpu_split_columns * 27.5e9 / link;
    t.split_columns      = static_cast<std::int32_t>(std::min<double>(columns, 4096));
    return t;
}

std::string gbps(double rate) { return " at " + std::to_string(rate / 1e9) + " GB/s"; }

void require_today(double rate, const LinkPolicyInputs& in) {
    const LinkPolicy p = derive_link_policy(rate, in);
    const Today t =
        today(rate, in.cpu_pcie_divisor_override, static_cast<std::uint32_t>(in.cpu_split_columns));
    require(p.bytes_per_second == rate, "the policy keeps its rate" + gbps(rate));
    require(p.pcie_divisor == t.pcie_divisor, "today's PCIe divisor" + gbps(rate));
    require(p.promotion_interval == t.promotion_interval,
            "today's promotion interval" + gbps(rate));
    require(p.split_columns == t.split_columns, "today's CPU split columns" + gbps(rate));
}

void test_today() {
    // Hand-computed: 27.5/3.1 = 8.87 (1 + 17.74 = 18.74, 35.5, 13,626); 27.5/7.4 = 3.72
    // (8.43, 14.9, 5,708); 27.5/28.4 = 0.968 (2.94, 3.87, 1,487.3).
    struct Expected {
        double rate;
        int divisor;
        std::uint64_t interval;
        std::int32_t columns;
    };

    for (const Expected& e : {Expected{3.1e9, 19, 8, 4096}, Expected{7.4e9, 8, 8, 4096},
                              Expected{28.4e9, 3, 4, 1487}, Expected{27.5e9, 3, 4, 1536}}) {
        const LinkPolicy p = derive_link_policy(e.rate, {});
        require(p.pcie_divisor == e.divisor && p.promotion_interval == e.interval &&
                    p.split_columns == e.columns,
                "hand-computed divisor, interval and split columns" + gbps(e.rate));
        require_today(e.rate, {});
    }
    // Today's expressions across the whole range of plausible links, with other split widths and a
    // fixed divisor; rounding and truncation boundaries included.
    for (const std::uint32_t columns : {1536U, 1024U, 64U, 0U}) {
        for (const int divisor : {0, 5}) {
            const LinkPolicyInputs in{.cpu_split_columns         = double(columns),
                                      .cpu_pcie_divisor_override = divisor};
            for (double rate = 0.25e9; rate <= 128e9; rate *= 1.01) { require_today(rate, in); }
            // Rates at which a rounded or truncated value steps.
            for (const double edge :
                 {2.0 * 27.5e9 / 1.5, 2.0 * 27.5e9 / 2.5, 4.0 * 27.5e9 / 1.5, 4.0 * 27.5e9 / 8.5,
                  1536.0 * 27.5e9 / 4096.0, 1536.0 * 27.5e9 / 1487.0}) {
                for (const double rate :
                     {std::nextafter(edge, 0.0), edge, std::nextafter(edge, 1e12)}) {
                    require_today(rate, in);
                }
            }
        }
    }
    // A fixed divisor wins at every rate.
    LinkPolicyInputs fixed;
    fixed.cpu_pcie_divisor_override = 5;
    require(derive_link_policy(3.1e9, fixed).pcie_divisor == 5 &&
                derive_link_policy(28.4e9, fixed).pcie_divisor == 5,
            "cpu_pcie_divisor fixes the divisor");
}

void test_slow_link() {
    for (const double rate : {3.1e9, 7.4e9, std::nextafter(8e9, 0.0)}) {
        const LinkPolicy p = derive_link_policy(rate, {});
        require(p.promotions == PromotionPolicy::Trickle && p.seed_share == 1.0 &&
                    p.cpu_jobs == 32,
                "a slow link trickles promotions, seeds every frame and gives the CPU 32 jobs" +
                    gbps(rate));
        require(p.trickle_bytes == static_cast<std::uint64_t>(rate * kSlowLinkTrickleSeconds) &&
                    p.idle_refill_promotions == 32,
                "the trickle is 13 ms of the link per round" + gbps(rate));
    }
    for (const double rate : {8e9, 27.5e9, 28.4e9, 64e9}) {
        const LinkPolicy p = derive_link_policy(rate, {});
        require(p.promotions == PromotionPolicy::Dynamic && p.seed_share == 0.5 && p.cpu_jobs == 16 &&
                    p.trickle_bytes == 0,
                "from 8 GB/s the dynamic cache as today" + gbps(rate));
    }
    // The CPU job count: the given one on a fast link; at least 32 on a slow one; none stays none.
    LinkPolicyInputs in;
    in.cpu_expert_jobs = 8;
    require(derive_link_policy(28.4e9, in).cpu_jobs == 8 &&
                derive_link_policy(3.1e9, in).cpu_jobs == 32,
            "8 jobs: 8 fast, 32 slow");
    in.cpu_expert_jobs = 64;
    require(derive_link_policy(3.1e9, in).cpu_jobs == 64, "64 jobs stay 64 on a slow link");
    in.cpu_expert_jobs = 0;
    require(derive_link_policy(3.1e9, in).cpu_jobs == 0 &&
                derive_link_policy(28.4e9, in).cpu_jobs == 0,
            "disabled CPU-served misses stay disabled");
}

void test_fill_scale() {
    require(derive_link_policy(3.1e9, {}).fill_scale == 3.1e9 / 27.5e9,
            "the x4 link fills at its share");
    require(derive_link_policy(13.75e9, {}).fill_scale == 0.5,
            "half the reference link fills at half the rate");
    require(derive_link_policy(27.5e9, {}).fill_scale == 1.0 &&
                derive_link_policy(56e9, {}).fill_scale == 1.0,
            "the reference link or faster fills at the full rate");
}

void test_reference() {
    // The defaults are the reference link's policy, and a rate the probe could not measure gives
    // it.
    const LinkPolicy defaults;
    constexpr double kInfinity = std::numeric_limits<double>::infinity();
    for (const double rate : {27.5e9, 0.0, -1.0, std::nan(""), kInfinity}) {
        const LinkPolicy p = derive_link_policy(rate, {});
        require(p.bytes_per_second == defaults.bytes_per_second &&
                    p.pcie_divisor == defaults.pcie_divisor &&
                    p.promotion_interval == defaults.promotion_interval &&
                    p.split_columns == defaults.split_columns &&
                    p.fill_scale == defaults.fill_scale && p.promotions == defaults.promotions &&
                    p.seed_share == defaults.seed_share && p.cpu_jobs == defaults.cpu_jobs &&
                    p.trickle_bytes == defaults.trickle_bytes &&
                    p.idle_refill_promotions == defaults.idle_refill_promotions,
                "the reference policy for rate " + std::to_string(rate));
    }
    // An absurdly slow link saturates instead of overflowing.
    const LinkPolicy crawl = derive_link_policy(1.0, {});
    require(crawl.pcie_divisor == std::numeric_limits<std::int32_t>::max() &&
                crawl.promotion_interval == 8 && crawl.split_columns == 4096 &&
                crawl.promotions == PromotionPolicy::Trickle && crawl.trickle_bytes == 0,
            "a 1 B/s link saturates every constant");
}

} // namespace

int main() {
    try {
        test_today();
        test_slow_link();
        test_fill_scale();
        test_reference();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
    std::printf("qwen4_exp link policy checks passed\n");
    return 0;
}
