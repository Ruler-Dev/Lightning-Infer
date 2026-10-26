#pragma once

// Qwen4Exp's n-gram copy rounds (design 11.3): whether a decode row verifies a copy of its own
// history (prompt and committed output, qwen3_5::detail::NgramProposer) instead of its neural
// drafter's tokens, and how many of the copy's tokens. Pure, so it is unit-tested
// (tests/models/qwen4_exp/test_ngram_round.cpp).
//
// The rule follows qwen3_5's (NgramProposer::propose_for_round / finish_round): a copy is matched
// over the last kNgramMatchWindow committed tokens and proposed with `lookahead` tokens past the
// round's `maximum` drafts. The target emits a bonus token after the drafts, so the lookahead
// leaves the copy source's last token to the neural drafter: a source ending (a closing quote, a
// changed line) may need different text. A copy whose source shows the whole lookahead is whole.
//
// Under DFlash2 a whole copy wins even when it is exactly as long as the drafter's block. Before
// this rule (master 69ef2495) a copy had to be longer than the row's neural drafts, and DFlash2's
// draft count before its proposal is the full block (--draft-tokens auto chooses its length after
// the proposal), so with --ngram-draft-tokens 7 beside a 7-token block no copy was ever verified.
// Under MTP a copy still has to be longer than the row's chosen draft length. A partial copy (its
// source ends inside the lookahead) is verified only when it is longer than the neural drafts.

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace infernix::models::qwen4_exp::detail {

// Committed tokens a copy is matched over (qwen3_5's kNgramMatchHistoryTokens): bounds the match's
// backward extension, and with it the host's time per round on repetitive long contexts.
inline constexpr std::size_t kNgramMatchWindow = 64;
// Recency order of copy candidates (NgramProposer::set_recency): matches of at least this many
// tokens tie on length and the newest wins. The idea is HyperQwen's lookup (syv-ai/HyperQwen,
// Apache-2.0, patches/dflash2-lookup-drafting.patch: score = min(len, NMAX) * 2^32 + end, NMAX 12,
// measured better than 32 on quote-and-explain); 16 is our unmeasured choice beside a minimum
// match of 12. INFERNIX_NGRAM_MATCH_CAP overrides it (0: the longest match, as on master).
inline constexpr std::uint32_t kNgramMatchCap = 16;

struct CopyRule {
    // Source tokens a copy must show past `maximum` drafts to be whole; its drafts stop this many
    // tokens before its source ends.
    std::uint32_t lookahead = 2;
    // Whether a whole copy wins whatever the row's neural drafter would verify.
    bool whole_copy_wins = true;

    // DFlash2 rows (and rows without a neural drafter).
    [[nodiscard]] static constexpr CopyRule dflash2() noexcept { return {2, true}; }
    // MTP rows: a copy wins only when it is longer than the row's chosen draft length.
    [[nodiscard]] static constexpr CopyRule mtp() noexcept { return {2, false}; }
    // Master 69ef2495 (INFERNIX_NGRAM_DFLASH2=0): no lookahead, a copy wins only when it is longer.
    [[nodiscard]] static constexpr CopyRule legacy() noexcept { return {0, false}; }
};

// The tokens a row asks its proposer for: the round's drafts plus the lookahead.
[[nodiscard]] constexpr std::uint32_t copy_proposal_tokens(std::uint32_t maximum, CopyRule rule) noexcept {
    return maximum == 0 ? 0U : maximum + rule.lookahead;
}

// The drafts a row's copy verifies, 0 for none. `available` source tokens follow the match (it was
// proposed with copy_proposal_tokens(maximum, rule)), the round fits `maximum` drafts, and the row's
// neural drafter would verify `neural`. The copy's d = min(maximum, available - lookahead) drafts
// are taken when d > 0 and either the copy is whole (available >= maximum + lookahead) and
// rule.whole_copy_wins, or d > neural.
[[nodiscard]] constexpr std::uint32_t copy_drafts(std::size_t available, std::uint32_t maximum,
                                                  std::uint32_t neural, CopyRule rule) noexcept {
    if (maximum == 0 || available <= rule.lookahead) { return 0; }
    const std::size_t source = available - rule.lookahead;
    const auto drafts        = static_cast<std::uint32_t>(std::min<std::size_t>(maximum, source));
    const bool whole         = source >= maximum;
    return (whole && rule.whole_copy_wins) || drafts > neural ? drafts : 0U;
}

} // namespace infernix::models::qwen4_exp::detail
