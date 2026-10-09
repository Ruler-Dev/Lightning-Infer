#pragma once

// The host-to-device copy rate of pinned memory (the PCIe link as copy engines see it), measured
// once at startup so bandwidth-dependent choices follow the machine instead of constants: a GPU at
// PCIe 5.0 x8 copies about 27.5 GB/s, at x16 about twice that.

#include <cstddef>

namespace infernix {

// Copies `bytes` from `pinned_src` (pinned host memory) to `device_dst` (memory of `device`) on a
// private stream of `device`, once to warm up and then `repetitions` times, and returns the best
// rate in bytes per second. `device_dst` is overwritten; the caller's streams are not involved.
// `device` -1 is the calling thread's current device; another ordinal is made current for the
// measurement only (DeviceGuard), so each device of a layer split measures its own link.
[[nodiscard]] double measure_h2d_bytes_per_second(const void* pinned_src, void* device_dst, std::size_t bytes,
                                                  int repetitions, int device = -1);

} // namespace infernix
