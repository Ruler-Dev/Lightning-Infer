#include "ops/common/device_sms.h"
#include "ops/linear/q8/q8_shapes.h"
#include "ops/linear/q8/q8_simt_launch.cuh"
#include "ops/linear/q8/q8_stream_launch.cuh"

namespace infernix::ops::detail {
namespace {
// Qwen3.8-Flash-Next GDN q/k/v/z projection (packed): the register-streamed route (2 rows per warp,
// x staged in shared memory per pass) up to 8 columns (5-28 % faster than SIMT on the RTX 5090, M1
// sweep), the MMA tiles beyond. On GB203 (temporary sweep of 2026-10-11 on the RTX 5080 and the RTX
// PRO 4000, cold L2) the SIMT route with 16-group stages serves up to 4 columns (2-17 % faster than
// the register-streamed route) and a 6-column register-streamed tile 5-6 columns (6-10 % faster
// than the 8-column tile). Both routes have one warp per row, take a lane's chunks k = 8 lane + 256 m
// in increasing m and reduce with the same shuffle tree, so they give the same bits (checked at
// every swept point and end to end, design §19.3.21).
template <int MaxCols, int ColsPerPass>
using Stream  = Q8A16StreamSchedule<2560, 2, 1, 8, MaxCols, ColsPerPass, Q8StreamX::Shared, 2>;
using SimtG16 = Q8A16SimtSchedule<8, 4, 1, 16, 2, Cache::cg, 1>;

void launch_simt_g16(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_q8_a16_simt<SimtG16>(q8_linear_operands(x, w),
                                LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), out.ne[0]},
                                LinearIdentityEpilogue{}, stream);
}
} // namespace

Q8Launch select_q8_n16384_k2560(std::int32_t tokens) {
    if (tokens <= 6 && gb203_class()) {
        return tokens <= 4 ? launch_simt_g16 : launch_q8_a16_stream_linear<Stream<6, 6>>;
    }
    if (tokens <= 4) return launch_q8_a16_stream_linear<Stream<4, 4>>;
    if (tokens <= 8) return launch_q8_a16_stream_linear<Stream<8, 8>>;
    if (tokens <= 64) return launch_q8_a16_mma_r32_t64;
    if (tokens <= 96) return launch_q8_a16_mma_r32_t96;
    // Beyond 96 columns (prefill chunks): the MMA tile sweep of 2026-10-05 (fn/rigs/layer/q8tiles2.bat,
    // RTX 5090). Every MMA tile accumulates each output over K in the same order, so their bits are
    // identical (checked at every swept point); only speed differs.
    return launch_q8_a16_mma_r64_t128;
}

} // namespace infernix::ops::detail
