#pragma once

// What a layer split across two devices (EngineOptions::devices: one DeviceSegment per device) can
// run so far. Each flag turns true with the work that implements it; until then the Engine refuses
// the configuration before it creates a device context (validate_engine_options), or, for the SSD
// expert tier that the RAM ledger selects, before it reads the weights (construct_qwen4_exp).
//
// Pure constants, no CUDA.

namespace infernix::models::qwen4_exp {

// Construction over two segments and the eager calls (prefill, forced tokens, the first token).
inline constexpr bool kSplitExecution = true;
// Decode, MTP draft and verify rounds across the segments (one graph family per segment, the
// residual through each boundary's HandoffChannel in the event or the gate mode).
inline constexpr bool kSplitDecode = true;
// The hybrid prefix cache across the segments: restores, captures and its persistent file (each
// segment copies its own KV planes and state on its own streams; records and files keep the whole
// model's layout, so they are interchangeable with an unsplit run's).
inline constexpr bool kSplitPrefixCache = true;
// Vision input: the tower runs on one segment (an offloaded tower on the fastest link's, a resident
// one on the first), and the segments that read its embeddings on another GPU get a copy through
// pinned memory (design §19.3.2, "On a layer split").
inline constexpr bool kSplitVision = true;
// The SSD expert tier (the experts not in Host RAM are read from the artifact on use).
inline constexpr bool kSplitExpertTier = false;

} // namespace infernix::models::qwen4_exp
