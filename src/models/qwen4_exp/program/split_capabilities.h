#pragma once

// What a layer split across two devices (EngineOptions::devices: one DeviceSegment per device) can
// run so far. Each flag turns true with the work that implements it; until then the Engine refuses
// the configuration before it creates a device context (validate_engine_options), or, for the SSD
// expert tier that the RAM ledger selects, before it reads the weights (construct_qwen4_exp).
//
// Pure constants, no CUDA.

namespace infernix::models::qwen4_exp {

// Construction over two segments and the eager calls (prefill, forced tokens, the first token).
inline constexpr bool kSplitExecution = false;
// Decode, MTP draft and verify rounds across the segments.
inline constexpr bool kSplitDecode = false;
// The hybrid prefix cache across the segments: restores, captures and its persistent file.
inline constexpr bool kSplitPrefixCache = false;
// Vision input.
inline constexpr bool kSplitVision = false;
// The SSD expert tier (the experts not in Host RAM are read from the artifact on use).
inline constexpr bool kSplitExpertTier = false;

} // namespace infernix::models::qwen4_exp
