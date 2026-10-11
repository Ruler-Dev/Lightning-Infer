#pragma once

// The current device's own multiprocessor count, for occupancy-only launch choices whose result
// bits do not depend on it: grids sized to be resident at once, register caps, a route among
// bit-identical ones. Unlike numerics_multiprocessor_count (src/core/per_device.h), it ignores
// INFERNIX_NUMERICS_SMS, which pins the count only for decompositions that change results (op
// development §4.2).

#include "core/device.h"
#include "core/per_device.h"

#include <cuda_runtime.h>

namespace infernix::ops {

inline int device_multiprocessors() {
    static PerDevice<int> count;
    return count.get([](int device) {
        int value = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&value, cudaDevAttrMultiProcessorCount, device));
        return value;
    });
}

// Decode-width routes measured on GB203 (the 84-SM RTX 5080 and the 70-SM RTX PRO 4000,
// 2026-10-11) serve devices of at most this many SMs; larger parts keep the routes measured on the
// 170-SM RTX 5090.
inline constexpr int kGb203MaxMultiprocessors = 96;

inline bool gb203_class() { return device_multiprocessors() <= kGb203MaxMultiprocessors; }

} // namespace infernix::ops
