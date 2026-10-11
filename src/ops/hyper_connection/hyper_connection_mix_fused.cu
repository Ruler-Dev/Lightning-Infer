// The fused route of hyper_connection_mix: two kernels instead of norm, the down projection,
// gates, the up projection and collapse (docs/maintainer/qwen3_8-flash-next-design.md §8.3 K1a/K1b,
// §19.3.3 Phase 1b), for Q8_G32_FP16 RowSplit weights (Dense8) and BF16 contiguous weights (the
// bit-exact artifact). The weight codec only changes how a lane's 8 weights are read; the order of
// every product and sum is the codec-independent schedule below.
//
// K1a hc_down_partials: 8 or 16 rows of W_down per row group, one or two per warp. The CTA stages
// its stream's slice of R for up to 8 columns, reduces each column's sum of squares in a fixed order
// (per-thread sums over its elements in index order, a warp butterfly, warps in order), forms
// Rn = R * inv * (1 + w) in FP32 shared memory and computes the slice's dot products in the
// lane-chunk order (lane l owns weights k = 8l + 256m, fmaf in k order, a warp butterfly). Outputs:
// FP32 partial[s][row][t]; the first CTA of each stream writes inv[s][t].
//
// K1b hc_up_collapse: 16 features d per feature block, a warp per d. Every CTA sums the partials in
// the fixed order s = 0..S-1 into z, forms m = SiLU(z / S) in FP32 shared memory (CTA 0 also writes
// inject = 2 sigmoid(z[rank + s] / S)), computes the S rows (s*H + d) of W_up against m in the same
// lane-chunk order, and writes x[d] = bf16((1/S) * sum_s sigmoid(u_s) * Rn_s) with Rn_s formed in
// FP32 from R, inv and w.
//
// Both grids are sized to be resident at once on the current device (Grids: its own SM count times
// the CTAs per SM each kernel fits): a K1a CTA owns every gridDim.x-th row group of its stream and
// forms Rn once for all of them, a K1b CTA every gridDim.x-th feature block and m once for all of
// them, each loading its next block's weights while it computes the current one. The RTX 5090's
// 170 SMs keep a row per warp and give one row group and one feature block per CTA (164 and 160
// CTAs, the original decomposition); on GB203 (the 84-SM RTX 5080, the 70-SM RTX PRO 4000) a warp
// holds two rows, so it reads each Rn chunk once for both, and the CTAs run 1-3 row groups and
// feature blocks each in one wave instead of 2-3 waves that each repeat the norm and m (inject
// mixer at 5 columns, cold L2: 69.9 -> 40.9 us on the RTX PRO 4000, 43.5 -> 27.1 us on the RTX
// 5080, infernix_hyper_connection_mix_bench). Which CTA or warp computes an output does not change
// its arithmetic.
//
// No intermediate is rounded to BF16 (op development §6.1): Rn, z, m and u stay FP32. Each column is
// computed independently in a fixed order, so a column's result does not depend on how many
// columns share the call, nor on the device's SM count.

#include "ops/hyper_connection/hyper_connection_mix_fused.h"

#include "core/per_device.h"
#include "ops/common/device_sms.h"
#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <stdexcept>
#include <string>

namespace infernix::ops::detail {
namespace {

using bf16 = __nv_bfloat16;

constexpr int kStreams      = 4;
constexpr int kHidden       = 2560;
constexpr int kChunks       = kHidden / 256; // a lane's 8-weight chunks per stream slice
constexpr int kMaxRank      = 512;
constexpr int kDownWarps    = 8; // the norm's block: its sum-of-squares order is fixed by these 256 threads
constexpr int kDownThreads  = kDownWarps * 32;
constexpr int kPassColumns  = 8;
constexpr int kUpFeatures   = 16; // features d per CTA (a d per warp)
constexpr int kUpThreads    = kUpFeatures * 32;
constexpr int kUpBlocks     = kHidden / kUpFeatures; // 160 feature blocks

void check_launch(const char* what) {
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("hyper_connection_mix ") + what + ": " + cudaGetErrorString(error));
    }
}

__device__ __forceinline__ float sigmoidf(float x) { return 1.0F / (1.0F + expf(-x)); }

__device__ __forceinline__ float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) { value += __shfl_xor_sync(0xFFFFFFFFU, value, offset); }
    return value;
}

__device__ __forceinline__ float half_bits_to_float(std::uint16_t bits) {
    return __half2float(__ushort_as_half(static_cast<unsigned short>(bits)));
}

// q8_g32_fp16 in row_split_k128_v1: int8 codes [rows, padded_k], FP16 scales per 32 codes.
struct Q8Codec {
    struct Params {
        const std::uint8_t* codes;
        const std::uint8_t* scales;
        int padded_k;
    };
    struct Chunk {
        uint2 code;
        float scale;
    };
    __device__ static Chunk load(const Params& p, std::int64_t row, int k) {
        return {ld_nc_na<uint2>(p.codes + row * p.padded_k + k),
                half_bits_to_float(ld_nc_na<std::uint16_t>(p.scales + row * (p.padded_k / 32) * 2 + (k / 32) * 2))};
    }
    __device__ static void decode(const Chunk& c, float (&w)[8]) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            w[i] = static_cast<float>(static_cast<std::int8_t>((i < 4 ? c.code.x : c.code.y) >> ((i & 3) * 8))) *
                   c.scale;
        }
    }
};

// BF16 contiguous: rows of `stride` elements.
struct Bf16Codec {
    struct Params {
        const bf16* data;
        int stride;
    };
    struct Chunk {
        uint4 bits;
    };
    __device__ static Chunk load(const Params& p, std::int64_t row, int k) {
        return {ld_nc_na<uint4>(p.data + row * p.stride + k)};
    }
    __device__ static void decode(const Chunk& c, float (&w)[8]) {
        const float2 a = bf16x2_bits_to_float2(c.bits.x), b = bf16x2_bits_to_float2(c.bits.y);
        const float2 d = bf16x2_bits_to_float2(c.bits.z), e = bf16x2_bits_to_float2(c.bits.w);
        w[0] = a.x, w[1] = a.y, w[2] = b.x, w[3] = b.y, w[4] = d.x, w[5] = d.y, w[6] = e.x, w[7] = e.y;
    }
};

// Rows of W_down per K1a row group when a warp holds RowsPerWarp of them.
template <int RowsPerWarp>
constexpr int kDownRows = kDownWarps * RowsPerWarp;

// Warp `warp`'s rows of row group g: g * kDownRows + r * kDownWarps + warp for r < RowsPerWarp.
template <int RowsPerWarp>
__device__ __forceinline__ int down_row(int g, int r, int warp) {
    return g * kDownRows<RowsPerWarp> + r * kDownWarps + warp;
}

template <class Codec, int RowsPerWarp>
__device__ __forceinline__ void load_down_rows(const typename Codec::Params& down, int g, int rows, int s, int warp,
                                               int lane, typename Codec::Chunk (&chunk)[RowsPerWarp][kChunks]) {
#pragma unroll
    for (int r = 0; r < RowsPerWarp; ++r) {
        const int row = down_row<RowsPerWarp>(g, r, warp);
        const std::int64_t source = row < rows ? row : 0;
#pragma unroll
        for (int m = 0; m < kChunks; ++m) { chunk[r][m] = Codec::load(down, source, s * kHidden + m * 256 + lane * 8); }
    }
}

// K1a. CTA (p, s) owns the row groups p, p + gridDim.x, ... of stream s (kDownRows rows of W_down
// each, RowsPerWarp rows per warp); hc_mix_fused sizes gridDim.x from the device's SM count so the
// grid is resident at once. The CTA forms its stream's Rn once per column pass and reuses it for all
// its row groups, loading the next group's weights while it computes the current one; a warp reads
// each Rn chunk once for its rows (shared-memory bandwidth bounds this loop).
template <class Codec, int RowsPerWarp>
__global__ __launch_bounds__(kDownThreads, 1) void hc_down_partials_kernel(
    const bf16* __restrict__ residual, const bf16* __restrict__ norm_weight, typename Codec::Params down, int rows,
    int columns, float eps, float* __restrict__ partial, float* __restrict__ inv_out) {
    extern __shared__ __align__(16) float rn[]; // [min(columns, kPassColumns)][kHidden]
    __shared__ float scratch[kPassColumns][kDownThreads / 32];
    __shared__ float inv[kPassColumns];
    const int lane = static_cast<int>(threadIdx.x) & 31, warp = static_cast<int>(threadIdx.x) >> 5;
    const int s      = static_cast<int>(blockIdx.y);
    const int groups = (rows + kDownRows<RowsPerWarp> - 1) / kDownRows<RowsPerWarp>;
    constexpr int width = kStreams * kHidden;
    // The sum-of-squares partition: thread i owns the elements [8i, 8i + 8) and, when i < kSecond,
    // [8i + kStep, 8i + kStep + 8), summed in that order.
    constexpr int kStep   = kDownThreads * 8;
    constexpr int kSecond = (kHidden - kStep) / 8;
    static_assert(kHidden > kStep && kHidden <= 2 * kStep && kHidden % 8 == 0, "one or two staging steps per thread");
    const int i0      = static_cast<int>(threadIdx.x) * 8;
    const bool second = static_cast<int>(threadIdx.x) < kSecond;

    // 1. Weights first: this warp's rows of the stream's K slice in the first row group, before any
    //    activation.
    typename Codec::Chunk chunk[RowsPerWarp][kChunks];
    load_down_rows<Codec, RowsPerWarp>(down, static_cast<int>(blockIdx.x), rows, s, warp, lane, chunk);
    const bf16* w = norm_weight + static_cast<std::int64_t>(s) * kHidden;

#pragma unroll 1
    for (int c0 = 0; c0 < columns; c0 += kPassColumns) {
        const int pass = min(kPassColumns, columns - c0);
        // 2. Stage the slice in FP32 and reduce each column's sum of squares in a fixed order (the
        //    thread's elements in index order, a warp butterfly, the warps in order). The loads of
        //    all the pass's columns are in flight together.
        uint4 head[kPassColumns], tail[kPassColumns];
#pragma unroll
        for (int c = 0; c < kPassColumns; ++c) {
            if (c < pass) {
                const bf16* r =
                    residual + static_cast<std::int64_t>(c0 + c) * width + static_cast<std::int64_t>(s) * kHidden;
                head[c] = load_ldg<uint4>(r + i0);
                if (second) { tail[c] = load_ldg<uint4>(r + i0 + kStep); }
            }
        }
#pragma unroll
        for (int c = 0; c < kPassColumns; ++c) {
            if (c < pass) {
                float sum = 0.0F;
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    if (h == 0 || second) {
                        const uint4 values = h == 0 ? head[c] : tail[c];
                        const float2 v0 = bf16x2_bits_to_float2(values.x), v1 = bf16x2_bits_to_float2(values.y);
                        const float2 v2 = bf16x2_bits_to_float2(values.z), v3 = bf16x2_bits_to_float2(values.w);
                        const float v[8] = {v0.x, v0.y, v1.x, v1.y, v2.x, v2.y, v3.x, v3.y};
                        float* out = rn + c * kHidden + i0 + h * kStep;
#pragma unroll
                        for (int j = 0; j < 8; ++j) {
                            out[j] = v[j];
                            sum += v[j] * v[j];
                        }
                    }
                }
                sum = warp_sum(sum);
                if (lane == 0) { scratch[c][warp] = sum; }
            }
        }
        __syncthreads();
        if (static_cast<int>(threadIdx.x) < pass) {
            const int c = static_cast<int>(threadIdx.x);
            float total = 0.0F;
#pragma unroll
            for (int v = 0; v < kDownThreads / 32; ++v) { total += scratch[c][v]; }
            inv[c] = rsqrtf(total / static_cast<float>(kHidden) + eps);
        }
        __syncthreads();
        // Rn = R * inv * (1 + w), FP32.
        for (int i = static_cast<int>(threadIdx.x); i < pass * kHidden; i += kDownThreads) {
            const int c = i / kHidden, d = i - c * kHidden;
            rn[i]       = rn[i] * inv[c] * (1.0F + __bfloat162float(w[d]));
        }
        if (blockIdx.x == 0 && static_cast<int>(threadIdx.x) < pass) {
            inv_out[static_cast<std::int64_t>(s) * columns + c0 + static_cast<int>(threadIdx.x)] = inv[threadIdx.x];
        }
        __syncthreads();
        // 3. Each row group's dot products: the lane-chunk order, then a warp butterfly.
#pragma unroll 1
        for (int g = static_cast<int>(blockIdx.x); g < groups; g += static_cast<int>(gridDim.x)) {
            // The weights the next step reads: the next row group, or this CTA's first one again
            // for the next column pass.
            const int following = g + static_cast<int>(gridDim.x);
            const int next      = following < groups              ? following
                                  : c0 + kPassColumns < columns ? static_cast<int>(blockIdx.x)
                                                                : -1;
            typename Codec::Chunk ahead[RowsPerWarp][kChunks];
            if (next >= 0) { load_down_rows<Codec, RowsPerWarp>(down, next, rows, s, warp, lane, ahead); }
            float acc[RowsPerWarp][kPassColumns];
#pragma unroll
            for (int r = 0; r < RowsPerWarp; ++r) {
#pragma unroll
                for (int c = 0; c < kPassColumns; ++c) { acc[r][c] = 0.0F; }
            }
#pragma unroll
            for (int m = 0; m < kChunks; ++m) {
                const int kk = m * 256 + lane * 8;
                float wv[RowsPerWarp][8];
#pragma unroll
                for (int r = 0; r < RowsPerWarp; ++r) { Codec::decode(chunk[r][m], wv[r]); }
#pragma unroll
                for (int c = 0; c < kPassColumns; ++c) {
                    if (c < pass) {
                        const float4 a = *reinterpret_cast<const float4*>(rn + c * kHidden + kk);
                        const float4 b = *reinterpret_cast<const float4*>(rn + c * kHidden + kk + 4);
#pragma unroll
                        for (int r = 0; r < RowsPerWarp; ++r) {
                            acc[r][c] = fmaf(wv[r][0], a.x, acc[r][c]);
                            acc[r][c] = fmaf(wv[r][1], a.y, acc[r][c]);
                            acc[r][c] = fmaf(wv[r][2], a.z, acc[r][c]);
                            acc[r][c] = fmaf(wv[r][3], a.w, acc[r][c]);
                            acc[r][c] = fmaf(wv[r][4], b.x, acc[r][c]);
                            acc[r][c] = fmaf(wv[r][5], b.y, acc[r][c]);
                            acc[r][c] = fmaf(wv[r][6], b.z, acc[r][c]);
                            acc[r][c] = fmaf(wv[r][7], b.w, acc[r][c]);
                        }
                    }
                }
            }
#pragma unroll
            for (int r = 0; r < RowsPerWarp; ++r) {
                const int row = down_row<RowsPerWarp>(g, r, warp);
#pragma unroll
                for (int c = 0; c < kPassColumns; ++c) {
                    if (c < pass) {
                        const float total = warp_sum(acc[r][c]);
                        if (lane == 0 && row < rows) {
                            partial[(static_cast<std::int64_t>(s) * rows + row) * columns + c0 + c] = total;
                        }
                    }
                }
            }
            if (next >= 0) {
#pragma unroll
                for (int r = 0; r < RowsPerWarp; ++r) {
#pragma unroll
                    for (int m = 0; m < kChunks; ++m) { chunk[r][m] = ahead[r][m]; }
                }
            }
        }
        __syncthreads(); // the next pass overwrites rn
    }
}

template <class Codec>
__device__ __forceinline__ void load_up_rows(const typename Codec::Params& up, int d, int lane, int chunks,
                                             typename Codec::Chunk (&chunk)[kStreams][2]) {
#pragma unroll
    for (int s = 0; s < kStreams; ++s) {
        const std::int64_t row = static_cast<std::int64_t>(s) * kHidden + d;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int c = lane + 32 * h;
            chunk[s][h] = c < chunks ? Codec::load(up, row, c * 8) : typename Codec::Chunk{};
        }
    }
}

// K1b. CTA q owns the feature blocks q, q + gridDim.x, ... (kUpFeatures features each, a feature per
// warp); hc_mix_fused sizes gridDim.x from the device's SM count. The CTA forms m once, loads the
// next block's weights while it computes the current one, and lane t applies column t's collapse.
template <class Codec>
__global__ __launch_bounds__(kUpThreads, 1) void hc_up_collapse_kernel(
    const float* __restrict__ partial, const float* __restrict__ inv_in, const bf16* __restrict__ residual,
    const bf16* __restrict__ norm_weight, typename Codec::Params up, int rank, int down_rows, int columns,
    float* __restrict__ inject, bf16* __restrict__ x) {
    extern __shared__ __align__(16) float m[];              // [T][rank]
    __shared__ float inv[kStreams * kHcMixFusedMaxColumns]; // [S][T]
    const int lane = static_cast<int>(threadIdx.x) & 31, warp = static_cast<int>(threadIdx.x) >> 5;
    constexpr int width = kStreams * kHidden;
    const int chunks    = rank / 8;

    // 1. Weights first: the S rows (s*H + d) of W_up of the first block, chunks lane and lane + 32.
    typename Codec::Chunk chunk[kStreams][2];
    load_up_rows<Codec>(up, static_cast<int>(blockIdx.x) * kUpFeatures + warp, lane, chunks, chunk);

    // 2. z = sum_s partial[s] in order s = 0..S-1; m = SiLU(z / S); CTA 0 writes the injects.
    const float inv_streams = 1.0F / static_cast<float>(kStreams);
    for (int i = static_cast<int>(threadIdx.x); i < down_rows * columns; i += kUpThreads) {
        const int k = i / columns, t = i - k * columns;
        float z     = 0.0F;
#pragma unroll
        for (int s = 0; s < kStreams; ++s) { z += partial[(static_cast<std::int64_t>(s) * down_rows + k) * columns + t]; }
        const float v = z * inv_streams;
        if (k < rank) {
            m[t * rank + k] = v * sigmoidf(v);
        } else if (inject != nullptr && blockIdx.x == 0) {
            inject[static_cast<std::int64_t>(t) * kStreams + (k - rank)] = 2.0F * sigmoidf(v);
        }
    }
    for (int i = static_cast<int>(threadIdx.x); i < kStreams * columns; i += kUpThreads) { inv[i] = inv_in[i]; }
    __syncthreads();

    // 3. Per block: u_s = W_up[s*H + d] . m in the lane-chunk order for each column (the warp
    //    butterfly leaves the sum in every lane), then lane t writes
    //    x[d, t] = bf16((1/S) * sum_s sigmoid(u_s) * Rn_s) with Rn_s formed in FP32 from R, inv and w.
#pragma unroll 1
    for (int block = static_cast<int>(blockIdx.x); block < kUpBlocks; block += static_cast<int>(gridDim.x)) {
        const int d = block * kUpFeatures + warp;
        float r[kStreams] = {};
        if (lane < columns) {
#pragma unroll
            for (int s = 0; s < kStreams; ++s) {
                r[s] = __bfloat162float(
                    residual[static_cast<std::int64_t>(lane) * width + static_cast<std::int64_t>(s) * kHidden + d]);
            }
        }
        float w_bias[kStreams];
#pragma unroll
        for (int s = 0; s < kStreams; ++s) {
            w_bias[s] = 1.0F + __bfloat162float(norm_weight[static_cast<std::int64_t>(s) * kHidden + d]);
        }
        const int next = block + static_cast<int>(gridDim.x);
        typename Codec::Chunk ahead[kStreams][2];
        if (next < kUpBlocks) { load_up_rows<Codec>(up, next * kUpFeatures + warp, lane, chunks, ahead); }
        // The block's weights, decoded once for every column.
        float wv[kStreams][2][8];
#pragma unroll
        for (int s = 0; s < kStreams; ++s) {
#pragma unroll
            for (int h = 0; h < 2; ++h) { Codec::decode(chunk[s][h], wv[s][h]); }
        }
        float u[kStreams] = {}; // the sums of this lane's column (lane < columns)
#pragma unroll 1
        for (int t = 0; t < columns; ++t) {
            const float* mt = m + t * rank;
            float acc[kStreams];
#pragma unroll
            for (int s = 0; s < kStreams; ++s) { acc[s] = 0.0F; }
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int c = lane + 32 * h;
                if (c < chunks) {
                    const float4 a = *reinterpret_cast<const float4*>(mt + c * 8);
                    const float4 b = *reinterpret_cast<const float4*>(mt + c * 8 + 4);
#pragma unroll
                    for (int s = 0; s < kStreams; ++s) {
                        acc[s] = fmaf(wv[s][h][0], a.x, acc[s]);
                        acc[s] = fmaf(wv[s][h][1], a.y, acc[s]);
                        acc[s] = fmaf(wv[s][h][2], a.z, acc[s]);
                        acc[s] = fmaf(wv[s][h][3], a.w, acc[s]);
                        acc[s] = fmaf(wv[s][h][4], b.x, acc[s]);
                        acc[s] = fmaf(wv[s][h][5], b.y, acc[s]);
                        acc[s] = fmaf(wv[s][h][6], b.z, acc[s]);
                        acc[s] = fmaf(wv[s][h][7], b.w, acc[s]);
                    }
                }
            }
#pragma unroll
            for (int s = 0; s < kStreams; ++s) {
                acc[s] = warp_sum(acc[s]);
                if (lane == t) { u[s] = acc[s]; }
            }
        }
        if (lane < columns) {
            float sum = 0.0F;
#pragma unroll
            for (int s = 0; s < kStreams; ++s) {
                const float rn = r[s] * inv[s * columns + lane] * w_bias[s];
                sum += sigmoidf(u[s]) * rn;
            }
            x[static_cast<std::int64_t>(lane) * kHidden + d] = __float2bfloat16_rn(sum * inv_streams);
        }
        if (next < kUpBlocks) {
#pragma unroll
            for (int s = 0; s < kStreams; ++s) {
#pragma unroll
                for (int h = 0; h < 2; ++h) { chunk[s][h] = ahead[s][h]; }
            }
        }
    }
}

bool q8_rowsplit(const Weight& w) noexcept {
    return w.qtype == QType::Q8_G32_FP16 && w.layout == QuantLayout::RowSplit && w.qdata != nullptr &&
           w.scales != nullptr && w.padded_shape[1] % 128 == 0 && w.padded_shape[1] >= w.k;
}

int bf16_stride(const Weight& w) noexcept { return w.padded_shape[1] >= w.k ? w.padded_shape[1] : w.k; }

bool bf16_contiguous(const Weight& w) noexcept {
    return w.qtype == QType::BF16 && w.layout == QuantLayout::Contiguous && w.qdata != nullptr &&
           reinterpret_cast<std::uintptr_t>(w.qdata) % 16 == 0 && bf16_stride(w) % 8 == 0;
}

Q8Codec::Params q8_params(const Weight& w) {
    return {static_cast<const std::uint8_t*>(w.qdata), static_cast<const std::uint8_t*>(w.scales), w.padded_shape[1]};
}

Bf16Codec::Params bf16_params(const Weight& w) { return {static_cast<const bf16*>(w.qdata), bf16_stride(w)}; }

// The grids, in CTAs resident at once on the current device: the device's own SM count (not the
// numerics count of INFERNIX_NUMERICS_SMS, since no result depends on the grid) times the CTAs per
// SM each kernel fits at this call's shared memory. Built once per device, codec and K1a rows per
// warp.
struct Grids {
    int down[kPassColumns + 1]          = {}; // CTAs per stream, by the columns of a pass
    int up[kHcMixFusedMaxColumns + 1] = {};   // by the call's columns
};

std::size_t down_shared_bytes(int pass_columns) {
    return static_cast<std::size_t>(pass_columns) * kHidden * sizeof(float);
}

std::size_t up_shared_bytes(int rank, int columns) {
    return static_cast<std::size_t>(columns) * rank * sizeof(float);
}

template <class Codec, int RowsPerWarp>
const Grids& grids() {
    static PerDevice<Grids> cache;
    return cache.get([](int device) {
        const cudaError_t error = cudaFuncSetAttribute(hc_down_partials_kernel<Codec, RowsPerWarp>,
                                                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                       static_cast<int>(down_shared_bytes(kPassColumns)));
        if (error != cudaSuccess) {
            throw std::runtime_error(std::string("hyper_connection_mix: shared memory: ") + cudaGetErrorString(error));
        }
        (void)device;
        const int sms = device_multiprocessors();
        Grids g;
        for (int p = 1; p <= kPassColumns; ++p) {
            int per_sm = 0;
            CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, hc_down_partials_kernel<Codec, RowsPerWarp>,
                                                                     kDownThreads, down_shared_bytes(p)));
            g.down[p] = std::max(1, sms * std::max(per_sm, 1) / kStreams);
        }
        for (int t = 1; t <= kHcMixFusedMaxColumns; ++t) {
            int per_sm = 0;
            CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, hc_up_collapse_kernel<Codec>, kUpThreads,
                                                                     up_shared_bytes(kMaxRank, t)));
            g.up[t] = std::max(1, std::min(kUpBlocks, sms * std::max(per_sm, 1)));
        }
        return g;
    });
}

template <class Codec, int RowsPerWarp>
void launch(const Tensor& residual, const Tensor& norm_weight, typename Codec::Params down,
            typename Codec::Params up, int down_rows, int rank, int streams, float eps, Tensor& x, Tensor* inject,
            void* partials, cudaStream_t stream) {
    const Grids& g    = grids<Codec, RowsPerWarp>();
    const int columns = residual.ne[1];
    const int pass    = std::min(columns, kPassColumns);
    const int groups  = (down_rows + kDownRows<RowsPerWarp> - 1) / kDownRows<RowsPerWarp>;
    auto* partial     = static_cast<float*>(partials);
    const std::size_t partial_bytes =
        (static_cast<std::size_t>(streams) * down_rows * columns * sizeof(float) + 255) / 256 * 256;
    auto* inv = reinterpret_cast<float*>(static_cast<std::byte*>(partials) + partial_bytes);
    hc_down_partials_kernel<Codec, RowsPerWarp><<<dim3(std::min(groups, g.down[pass]), streams), kDownThreads,
                                                  down_shared_bytes(pass), stream>>>(
        static_cast<const bf16*>(residual.data), static_cast<const bf16*>(norm_weight.data), down, down_rows,
        columns, eps, partial, inv);
    check_launch("down partials");
    hc_up_collapse_kernel<Codec><<<g.up[columns], kUpThreads, up_shared_bytes(rank, columns), stream>>>(
        partial, inv, static_cast<const bf16*>(residual.data), static_cast<const bf16*>(norm_weight.data), up, rank,
        down_rows, columns, inject != nullptr ? static_cast<float*>(inject->data) : nullptr,
        static_cast<bf16*>(x.data));
    check_launch("up collapse");
}

} // namespace

bool hc_mix_fused_supported(const Weight& down, const Weight& up, std::int32_t hidden, std::int32_t streams,
                            std::int32_t rank, std::int32_t columns) noexcept {
    const bool codec = (q8_rowsplit(down) && q8_rowsplit(up)) || (bf16_contiguous(down) && bf16_contiguous(up));
    return codec && columns >= 1 && columns <= kHcMixFusedMaxColumns && hidden == kHidden && streams == kStreams &&
           rank > 0 && rank % 8 == 0 && rank <= kMaxRank && down.k == kStreams * kHidden &&
           (down.n == rank || down.n == rank + kStreams) && up.n == kStreams * kHidden && up.k == rank;
}

std::size_t hc_mix_fused_workspace_bytes(std::int32_t down_rows, std::int32_t streams, std::int32_t columns) noexcept {
    const auto align = [](std::size_t bytes) { return (bytes + 255) / 256 * 256; };
    return align(static_cast<std::size_t>(streams) * down_rows * columns * sizeof(float)) +
           align(static_cast<std::size_t>(streams) * columns * sizeof(float));
}

void hc_mix_fused(const Tensor& residual, const Tensor& norm_weight, const Weight& down, const Weight& up,
                  std::int32_t streams, std::int32_t rank, float eps, Tensor& x, Tensor* inject, void* partials,
                  cudaStream_t stream) {
    if (!hc_mix_fused_supported(down, up, x.ne[0], streams, rank, residual.ne[1])) {
        throw std::invalid_argument("hyper_connection_mix: the fused route does not serve this problem");
    }
    if (inject != nullptr && down.n != rank + streams) {
        throw std::invalid_argument("hyper_connection_mix: injects need the down projection's injection rows");
    }
    // Two W_down rows per K1a warp only where measured (Q8 on GB203); the RTX 5090 and the BF16 codec
    // (twice the weight registers per row) keep a row per warp. The choice does not change a bit.
    if (down.qtype == QType::Q8_G32_FP16 && gb203_class()) {
        launch<Q8Codec, 2>(residual, norm_weight, q8_params(down), q8_params(up), down.n, rank, streams, eps, x,
                           inject, partials, stream);
    } else if (down.qtype == QType::Q8_G32_FP16) {
        launch<Q8Codec, 1>(residual, norm_weight, q8_params(down), q8_params(up), down.n, rank, streams, eps, x,
                           inject, partials, stream);
    } else {
        launch<Bf16Codec, 1>(residual, norm_weight, bf16_params(down), bf16_params(up), down.n, rank, streams, eps, x,
                             inject, partials, stream);
    }
}

} // namespace infernix::ops::detail
