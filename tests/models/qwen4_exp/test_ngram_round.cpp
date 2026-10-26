// Qwen4Exp's n-gram copy rounds (program/ngram_round.h) and the proposer hygiene it uses
// (qwen3_5::detail::NgramProposer with a match window, special-token boundaries and recency order):
//   - the selection rule: DFlash2 rows take a whole copy even when it is as long as the drafter's
//     block (the production case: --ngram-draft-tokens 7 beside a 7-token block, which master's rule
//     never verified), MTP rows only a longer one; the DFlash2 rule equals qwen3_5's finish_round on
//     every input, the legacy rule master's `copy.size() > extent`;
//   - the lookahead: a copy whose source ends within two tokens of the round's drafts is partial;
//   - boundaries: a special token ends a copy source and no match spans one;
//   - recency: with a cap, a newer match beats an older, longer one; without, the longest wins.

#include "models/qwen3_5/program/ngram_proposer.h"
#include "models/qwen4_exp/program/ngram_round.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

using infernix::models::qwen3_5::detail::NgramProposer;
using infernix::models::qwen4_exp::detail::copy_drafts;
using infernix::models::qwen4_exp::detail::copy_proposal_tokens;
using infernix::models::qwen4_exp::detail::CopyRule;
using infernix::models::qwen4_exp::detail::kNgramMatchCap;
using infernix::models::qwen4_exp::detail::kNgramMatchWindow;
using Tokens = std::vector<std::int32_t>;

namespace {

void require(bool ok, const std::string& what) {
    if (!ok) { throw std::runtime_error(what); }
}

// `count` distinct tokens from `first`.
Tokens run(std::int32_t first, std::size_t count) {
    Tokens out(count);
    for (std::size_t i = 0; i < count; ++i) { out[i] = first + static_cast<std::int32_t>(i); }
    return out;
}

Tokens join(std::initializer_list<Tokens> parts) {
    Tokens out;
    for (const Tokens& part : parts) { out.insert(out.end(), part.begin(), part.end()); }
    return out;
}

// A lane's proposer as Qwen4Exp builds it (make_proposer): legacy, or with boundaries and recency.
NgramProposer proposer_over(const Tokens& history, const Tokens& boundaries, std::uint32_t cap) {
    NgramProposer proposer(1U << 12, 1U << 11);
    proposer.set_boundaries(boundaries);
    proposer.set_recency(cap);
    for (const auto token : history) { proposer.append(token); }
    return proposer;
}

// The decode round's choice for one row (ProgramImpl::decode): the copy's drafts, empty for none.
Tokens choose(const NgramProposer& proposer, const Tokens& history, std::uint32_t maximum, std::uint32_t neural,
              CopyRule rule, std::uint32_t min_match = 12) {
    const std::span<const std::int32_t> all(history);
    const bool legacy = rule.lookahead == 0;
    auto copy = proposer.propose(legacy ? all : all.last(std::min(kNgramMatchWindow, all.size())),
                                 copy_proposal_tokens(maximum, rule), min_match)
                    .tokens;
    copy.resize(copy_drafts(copy.size(), maximum, neural, rule));
    return copy;
}

void selection_rule() {
    const CopyRule dflash2 = CopyRule::dflash2(), mtp = CopyRule::mtp(), legacy = CopyRule::legacy();
    // Production: limit 7, DFlash2 block 7 (the row's extent before its proposal), a copy of 7 drafts
    // whose source shows the 2-token lookahead (9 tokens proposed).
    require(copy_proposal_tokens(7, dflash2) == 9, "a round of 7 drafts asks for 9 source tokens");
    require(copy_drafts(9, 7, 7, dflash2) == 7, "DFlash2: a whole copy of 7 replaces the 7-token block");
    require(copy_drafts(7, 7, 7, legacy) == 0, "legacy: a copy of 7 never beats a 7-token block (the bug)");
    require(copy_drafts(9, 7, 7, mtp) == 0, "MTP: a whole copy as long as the drafts does not replace them");
    require(copy_drafts(9, 7, 3, mtp) == 7, "MTP: a longer copy replaces 3 drafts");
    // Partial copies (the source ends inside the lookahead) must be longer than the neural drafts.
    require(copy_drafts(8, 7, 7, dflash2) == 0, "DFlash2: a partial copy of 6 loses to the block");
    require(copy_drafts(8, 7, 5, dflash2) == 6, "DFlash2: a partial copy of 6 beats 5 neural drafts");
    require(copy_drafts(8, 7, 3, mtp) == 6 && copy_drafts(8, 7, 6, mtp) == 0, "MTP: partial copies");
    require(copy_drafts(2, 7, 0, dflash2) == 0 && copy_drafts(3, 7, 0, dflash2) == 1,
            "the lookahead's tokens are never drafted");
    require(copy_drafts(9, 0, 0, dflash2) == 0 && copy_proposal_tokens(0, dflash2) == 0, "no room, no copy");
    // A budget-limited round (2 drafts fit): a copy showing 4 source tokens is whole.
    require(copy_drafts(4, 2, 2, dflash2) == 2 && copy_drafts(3, 2, 2, dflash2) == 0, "budget-limited rounds");
    // Every input: DFlash2's rule is qwen3_5's finish_round; legacy's is master's; MTP's is
    // "longer than neural" after the lookahead.
    for (std::uint32_t maximum = 0; maximum <= 15; ++maximum) {
        for (std::uint32_t neural = 0; neural <= 15; ++neural) {
            for (std::size_t available = 0; available <= copy_proposal_tokens(maximum, dflash2); ++available) {
                NgramProposer::Match match;
                match.tokens = run(1000, available);
                const auto finished = NgramProposer::finish_round(match, maximum, neural);
                const std::string at = " (available " + std::to_string(available) + ", maximum " +
                                       std::to_string(maximum) + ", neural " + std::to_string(neural) + ")";
                require(copy_drafts(available, maximum, neural, dflash2) == finished.tokens.size(),
                        "DFlash2's rule equals qwen3_5's finish_round" + at);
                const auto source = static_cast<std::uint32_t>(std::min<std::size_t>(
                    maximum, available > 2 ? available - 2 : 0));
                require(copy_drafts(available, maximum, neural, mtp) == (source > neural ? source : 0U),
                        "MTP's rule is longer-than-neural" + at);
                if (available <= maximum) {
                    require(copy_drafts(available, maximum, neural, legacy) ==
                                (available > neural ? static_cast<std::uint32_t>(available) : 0U),
                            "legacy's rule is master's copy.size() > extent" + at);
                }
            }
        }
    }
}

void production_copy() {
    // A 40-token file in the prompt, other text, then the model re-emits the file: its first 20
    // tokens are committed. The copy continues with the file's next tokens.
    const Tokens file = run(5000, 40);
    const Tokens history = join({run(100, 30), file, run(300, 50), Tokens(file.begin(), file.begin() + 20)});
    const Tokens next(file.begin() + 20, file.begin() + 27);
    const NgramProposer fresh = proposer_over(history, {}, kNgramMatchCap);
    require(choose(fresh, history, 7, 7, CopyRule::dflash2()) == next,
            "DFlash2 limit 7, block 7: the file's next 7 tokens are verified as a copy");
    const NgramProposer master = proposer_over(history, {}, 0);
    require(master.propose(history, 7, 12).tokens == next, "master's proposal is the same 7 tokens");
    require(choose(master, history, 7, 7, CopyRule::legacy()).empty(), "master's rule drops it beside the block");
    require(choose(fresh, history, 7, 3, CopyRule::mtp()) == next, "MTP with 3 drafts takes it");
    require(choose(fresh, history, 7, 7, CopyRule::mtp()).empty(), "MTP with 7 drafts keeps them");
    // The file's last 8 tokens remain (32..39): without a boundary its source runs on into the next
    // text, so the copy is whole; ending at a special token, it is partial (6 drafts).
    const Tokens late = join({run(100, 30), file, run(300, 50), Tokens(file.begin(), file.begin() + 32)});
    const NgramProposer at_end = proposer_over(late, {}, kNgramMatchCap);
    require(choose(at_end, late, 7, 7, CopyRule::dflash2()) == Tokens(file.begin() + 32, file.begin() + 39),
            "the source continues past the file (no boundary): a whole copy");
    const Tokens bounded = join({run(100, 30), file, Tokens{7}, run(300, 50), Tokens(file.begin(), file.begin() + 32)});
    const NgramProposer at_boundary = proposer_over(bounded, {7}, kNgramMatchCap);
    require(choose(at_boundary, bounded, 7, 7, CopyRule::dflash2()).empty(),
            "a source ending at a special token 8 tokens on is partial beside the block");
    require(choose(at_boundary, bounded, 7, 3, CopyRule::mtp()) == Tokens(file.begin() + 32, file.begin() + 38),
            "MTP with 3 drafts takes its 6 tokens, leaving the last source token");
}

void boundaries() {
    // Special token 7 ends the copy source; without boundaries the copy runs through it.
    const Tokens source = run(5000, 24);
    const Tokens history = join({source, Tokens{7}, run(800, 4), run(300, 40), Tokens(source.begin(), source.begin() + 20)});
    const NgramProposer plain = proposer_over(history, {}, kNgramMatchCap);
    const auto through = plain.propose(history, 9, 12).tokens;
    require(through.size() == 9 && through[4] == 7, "without boundaries the copy crosses the special token");
    const NgramProposer bounded = proposer_over(history, {7}, kNgramMatchCap);
    require(bounded.propose(history, 9, 12).tokens == Tokens(source.begin() + 20, source.end()),
            "with boundaries the copy stops before the special token");
    // No match spans a special token: X 7 Y occurs before Z and is the committed tail. Without
    // boundaries it matches (16 tokens, the cap, of 19) and copies Z; with them only Y (8 tokens)
    // can match, below the minimum of 12.
    const Tokens X = run(5000, 10), Y = run(5010, 8), Z = run(6000, 12);
    const Tokens crossing = join({X, Tokens{7}, Y, Z, run(300, 40), X, Tokens{7}, Y});
    const NgramProposer through_special = proposer_over(crossing, {}, kNgramMatchCap);
    require(through_special.propose(crossing, 9, 12).tokens == Tokens(Z.begin(), Z.begin() + 9),
            "without boundaries the match spans the special token");
    const NgramProposer across = proposer_over(crossing, {7}, kNgramMatchCap);
    require(across.propose(crossing, 9, 12).tokens.empty(), "a match never extends across a special token");
}

void recency() {
    // P (20 tokens) occurs twice before the committed tail P: whole in an older passage followed by A,
    // and its last 16 tokens in a newer one followed by B. The older match is 20 tokens, the newer 16.
    const Tokens P = run(5000, 20), A = run(6000, 12), B = run(7000, 12);
    const Tokens history = join({run(100, 10), P, A, run(300, 30), Tokens{900, 901, 902, 903},
                                 Tokens(P.begin() + 4, P.end()), B, run(400, 30), P});
    const NgramProposer capped = proposer_over(history, {}, 16);
    const auto newer = capped.propose(history, 9, 12);
    require(newer.tokens == Tokens(B.begin(), B.begin() + 9), "cap 16: the newer 16-token match wins");
    require(newer.matched == 16, "cap 16: backward extension stops at the cap");
    const NgramProposer longest = proposer_over(history, {}, 0);
    require(longest.propose(history, 9, 12).tokens == Tokens(A.begin(), A.begin() + 9),
            "no cap: the older, longer match wins (master's order)");
    const NgramProposer wide = proposer_over(history, {}, 32);
    require(wide.propose(history, 9, 12).tokens == Tokens(A.begin(), A.begin() + 9),
            "cap 32: the 20-token match is longer than the cap's tie");
    // Equal capped lengths: a whole continuation beats a newer short one, then the newer wins.
    const Tokens short_tail = join({run(100, 10), P, A, run(300, 30), P, Tokens{950, 951}, Tokens{7}, run(400, 30), P});
    const NgramProposer shorter = proposer_over(short_tail, {7}, 16);
    require(shorter.propose(short_tail, 9, 12).tokens == Tokens(A.begin(), A.begin() + 9),
            "a newer match with a 2-token continuation loses to an older whole one");
    // Deterministic: the same history gives the same proposal.
    for (int i = 0; i < 4; ++i) {
        require(proposer_over(history, {}, 16).propose(history, 9, 12).tokens == newer.tokens, "deterministic");
    }
    // The window bounds the backward extension without a cap: a 100-token repeat matches 64.
    const Tokens long_run = run(5000, 100);
    const Tokens repeat = join({long_run, run(300, 20), Tokens(long_run.begin(), long_run.begin() + 90)});
    const NgramProposer unbounded = proposer_over(repeat, {}, 0);
    const std::span<const std::int32_t> all(repeat);
    require(unbounded.propose(all, 9, 12).matched == 90, "the whole history extends the match to 90");
    require(unbounded.propose(all.last(kNgramMatchWindow), 9, 12).matched == kNgramMatchWindow,
            "the match window bounds it at 64");
}

} // namespace

int main() {
    try {
        selection_rule();
        production_copy();
        boundaries();
        recency();
        std::printf("ngram round tests passed\n");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
