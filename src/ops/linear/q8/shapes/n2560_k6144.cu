#include "ops/common/device_sms.h"
#include "ops/linear/q8/q8_launch.h"
#include "ops/linear/q8/q8_shapes.h"
#include "ops/linear/q8/q8_simt_launch.cuh"
#include "ops/linear/q8/q8_stream_launch.cuh"

namespace infernix::ops::detail {
namespace {
// Decode widths on GB203 (temporary sweep of 2026-10-11 on the RTX 5080 and the RTX PRO 4000, cold
// L2): the SIMT route with 16-group stages up to 4 columns (0-22 % faster than 32-group stages),
// the register-streamed route with x read through L1 for 5-8 columns (18-26 % faster than the SIMT
// route's 8-column tile, whose 176 registers leave one CTA per SM). Larger parts keep the RTX
// 5090's SIMT routes (the register-streamed route measured 16-29 % slower there, M1 sweep). Every
// decode-width route has one warp per row, takes a lane's chunks k = 8 lane + 256 m in increasing m
// and reduces with the same shuffle tree, so they give the same bits (checked at every swept point
// and end to end, design §19.3.21).
using SimtG16  = Q8A16SimtSchedule<8, 4, 1, 16, 2, Cache::cg, 1>;
using StreamL1 = Q8A16StreamSchedule<6144, 1, 1, 8, 8, 8, Q8StreamX::Global, 2>;

void launch_simt_g16(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_q8_a16_simt<SimtG16>(q8_linear_operands(x, w),
                                LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), out.ne[0]},
                                LinearIdentityEpilogue{}, stream);
}
} // namespace

// Qwen3.8-Flash-Next GDN and QSA output projections.
Q8Launch select_q8_n2560_k6144(std::int32_t tokens) {
    if (tokens <= 8 && gb203_class()) {
        return tokens <= 4 ? launch_simt_g16 : launch_q8_a16_stream_linear<StreamL1>;
    }
    if (tokens <= 4) return launch_q8_a16_simt_r8_t4;
    if (tokens <= 8) return launch_q8_a16_simt_r8_t8;
    if (tokens <= 64) return launch_q8_a16_mma_r32_t64;
    if (tokens <= 96) return launch_q8_a16_mma_r32_t96;
    // Beyond 96 columns (prefill chunks): the MMA tile sweep of 2026-10-05 (fn/rigs/layer/q8tiles2.bat,
    // RTX 5090). Every MMA tile accumulates each output over K in the same order, so their bits are
    // identical (checked at every swept point); only speed differs.
    if (tokens <= 256) return launch_q8_a16_mma_r32_t64;
    return launch_q8_a16_mma_r64_t128;
}

} // namespace infernix::ops::detail
