// Resident routed experts (include/infernix/ops/resident_moe.h). One CTA per (row tile, entry):
// each warp streams whole weight rows in 16-byte chunks against the entry's activation staged in
// shared memory as FP32, applies each chunk's FP16 group scale once, and reduces by a fixed
// butterfly, so an output's bits depend only on the shapes, never on the batch or placement.
//
// The staged activation is padded by four floats after every 32 (padded()): lane l reads chunk l's
// activations, one 32- or 16-float run per lane, and without the padding every lane of a warp would
// address the same shared-memory bank (a 32-way conflict on each read). With it the 16-byte reads of
// eight consecutive lanes fall in distinct banks.
#include "infernix/ops/resident_moe.h"

#include "core/device.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace infernix::ops {
namespace {

using bf16 = __nv_bfloat16;

constexpr int kThreads       = 256;
constexpr int kWarps         = kThreads / 32;
constexpr int kGateRowsPerCta = kWarps * 2; // intermediate rows (each a gate and an up row)
constexpr int kDownRowsPerCta = kWarps * 4;
constexpr int kMaxHidden      = 4096;
constexpr int kMaxIntermediate = 2048;

// Shared-memory index of activation element i: four padding floats after every 32.
__host__ __device__ constexpr int padded(int i) { return i + 4 * (i >> 5); }

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string("resident_moe_experts: ") + message); }
}

// Row-split codecs: one 16-byte chunk holds Q4's 32 or Q8's 16 consecutive codes of one group.
// x is the chunk's activations in shared memory, 16-byte aligned and contiguous.
struct Q4 {
    static constexpr int kCodes = 32, kGroup = 64;
    __device__ static float chunk_dot(uint4 w, const float* x) {
        const std::uint32_t words[4] = {w.x, w.y, w.z, w.w};
        const auto* x4               = reinterpret_cast<const float4*>(x);
        float dot = 0.0F;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float4 a = x4[2 * i], b = x4[2 * i + 1];
            const float xv[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int u = static_cast<int>((words[i] >> (4 * j)) & 0xFU);
                dot = fmaf(static_cast<float>(u >= 8 ? u - 16 : u), xv[j], dot);
            }
        }
        return dot;
    }
};

struct Q8 {
    static constexpr int kCodes = 16, kGroup = 32;
    __device__ static float chunk_dot(uint4 w, const float* x) {
        const auto* c  = reinterpret_cast<const std::int8_t*>(&w);
        const auto* x4 = reinterpret_cast<const float4*>(x);
        float dot      = 0.0F;
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const float4 v    = x4[q];
            const float xv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
            for (int i = 0; i < 4; ++i) { dot = fmaf(static_cast<float>(c[4 * q + i]), xv[i], dot); }
        }
        return dot;
    }
};
static_assert(Q4::kCodes <= 32 && Q8::kCodes <= 32 && 32 % Q4::kCodes == 0 && 32 % Q8::kCodes == 0,
              "a chunk's activations lie in one padded run of 32");

struct Bank {
    const std::uint8_t* codes;
    const __half* scales;
    int padded_k;
};

// Dot product of row `row` of a bank with the FP32 vector x (K = padded_k, staged at padded()
// indices), reduced over the warp.
template <class Codec>
__device__ float row_dot(const Bank& bank, std::int64_t row, const float* x, int lane) {
    const int chunks       = bank.padded_k / Codec::kCodes;
    const int groups       = bank.padded_k / Codec::kGroup;
    const auto* chunks_row = reinterpret_cast<const uint4*>(bank.codes) + row * chunks;
    const __half* scale_row = bank.scales + row * groups;
    float sum = 0.0F;
    for (int c = lane; c < chunks; c += 32) {
        const uint4 w     = __ldcs(chunks_row + c);
        const float scale = __half2float(__ldg(scale_row + c * Codec::kCodes / Codec::kGroup));
        sum               = fmaf(Codec::chunk_dot(w, x + padded(c * Codec::kCodes)), scale, sum);
    }
    for (int offset = 16; offset > 0; offset >>= 1) { sum += __shfl_xor_sync(0xFFFFFFFFU, sum, offset); }
    return sum;
}

template <class Codec>
__global__ void __launch_bounds__(kThreads)
    gate_up_kernel(const bf16* __restrict__ x, const std::int32_t* __restrict__ ids, int top_k, int hidden,
                   int intermediate, Bank bank, float* __restrict__ h) {
    __shared__ __align__(16) float xs[padded(kMaxHidden)];
    const int entry = static_cast<int>(blockIdx.y);
    const int t     = entry / top_k;
    const int e     = ids[entry];
    for (int i = static_cast<int>(threadIdx.x); i < bank.padded_k; i += kThreads) {
        xs[padded(i)] = i < hidden ? __bfloat162float(x[static_cast<std::int64_t>(t) * hidden + i]) : 0.0F;
    }
    __syncthreads();
    const int lane = static_cast<int>(threadIdx.x % 32), warp = static_cast<int>(threadIdx.x / 32);
    for (int i = 0; i < 2; ++i) {
        const int r = static_cast<int>(blockIdx.x) * kGateRowsPerCta + warp * 2 + i;
        if (r >= intermediate) { break; }
        const std::int64_t base = static_cast<std::int64_t>(e) * 2 * intermediate;
        const float g = row_dot<Codec>(bank, base + r, xs, lane);
        const float u = row_dot<Codec>(bank, base + intermediate + r, xs, lane);
        if (lane == 0) { h[static_cast<std::int64_t>(entry) * intermediate + r] = g / (1.0F + __expf(-g)) * u; }
    }
}

template <class Codec>
__global__ void __launch_bounds__(kThreads)
    down_kernel(const float* __restrict__ h, const std::int32_t* __restrict__ ids, int hidden, int intermediate,
                Bank bank, bf16* __restrict__ out) {
    __shared__ __align__(16) float hs[padded(kMaxIntermediate)];
    const int entry = static_cast<int>(blockIdx.y);
    const int e     = ids[entry];
    for (int i = static_cast<int>(threadIdx.x); i < bank.padded_k; i += kThreads) {
        hs[padded(i)] = i < intermediate ? h[static_cast<std::int64_t>(entry) * intermediate + i] : 0.0F;
    }
    __syncthreads();
    const int lane = static_cast<int>(threadIdx.x % 32), warp = static_cast<int>(threadIdx.x / 32);
    for (int i = 0; i < kDownRowsPerCta / kWarps; ++i) {
        const int r = static_cast<int>(blockIdx.x) * kDownRowsPerCta + warp * (kDownRowsPerCta / kWarps) + i;
        if (r >= hidden) { break; }
        const float y = row_dot<Codec>(bank, static_cast<std::int64_t>(e) * hidden + r, hs, lane);
        if (lane == 0) { out[static_cast<std::int64_t>(entry) * hidden + r] = __float2bfloat16_rn(y); }
    }
}

Bank bank_of(const Weight& w) {
    return {static_cast<const std::uint8_t*>(w.qdata), static_cast<const __half*>(w.scales), w.padded_shape[1]};
}

template <class Codec>
void launch(const Tensor& x, const Tensor& ids, const Weight& gate_up, const Weight& down, int top_k, int hidden,
            int intermediate, float* h, Tensor& outputs, cudaStream_t stream) {
    const int entries = ids.ne[0] * ids.ne[1];
    gate_up_kernel<Codec><<<dim3((intermediate + kGateRowsPerCta - 1) / kGateRowsPerCta, entries), kThreads, 0,
                            stream>>>(static_cast<const bf16*>(x.data), static_cast<const std::int32_t*>(ids.data),
                                      top_k, hidden, intermediate, bank_of(gate_up), h);
    CUDA_CHECK(cudaGetLastError());
    down_kernel<Codec><<<dim3((hidden + kDownRowsPerCta - 1) / kDownRowsPerCta, entries), kThreads, 0, stream>>>(
        h, static_cast<const std::int32_t*>(ids.data), hidden, intermediate, bank_of(down),
        static_cast<bf16*>(outputs.data));
    CUDA_CHECK(cudaGetLastError());
}

bool row_split(const Weight& w, QType qtype) {
    return w.qtype == qtype && w.layout == QuantLayout::RowSplit && w.qdata != nullptr && w.scales != nullptr &&
           w.qhigh == nullptr;
}

} // namespace

std::size_t resident_moe_workspace_bytes(std::int32_t pairs, std::int32_t intermediate) {
    return (static_cast<std::size_t>(pairs) * intermediate * sizeof(float) + 255) / 256 * 256;
}

void resident_moe_experts(const Tensor& x, const Tensor& ids, const Weight& gate_up, const Weight& down,
                          std::int32_t experts, std::int32_t intermediate, void* workspace,
                          std::size_t workspace_bytes, Tensor& outputs, cudaStream_t stream) {
    const int hidden = x.ne[0], columns = x.ne[1], top_k = ids.ne[0];
    require(x.dtype == DType::BF16 && x.is_contiguous() && ids.dtype == DType::I32 && ids.is_contiguous() &&
                outputs.dtype == DType::BF16 && outputs.is_contiguous(),
            "x, ids and outputs must be contiguous BF16/I32/BF16");
    require(ids.ne[1] == columns && top_k > 0 && outputs.ne[0] == hidden && outputs.ne[1] == top_k * columns,
            "ids must be [k, T] and outputs [H, k*T]");
    require(hidden > 0 && hidden <= kMaxHidden && intermediate > 0 && intermediate <= kMaxIntermediate,
            "hidden or intermediate size is out of range");
    require(gate_up.qtype == down.qtype, "both banks must share one codec");
    require(gate_up.n == experts * 2 * intermediate && gate_up.k == hidden && down.n == experts * hidden &&
                down.k == intermediate,
            "bank shapes differ from [E*2I, H] and [E*H, I]");
    require(workspace_bytes >= resident_moe_workspace_bytes(top_k * columns, intermediate) && workspace != nullptr,
            "workspace is too small");
    require(gate_up.padded_shape[1] <= kMaxHidden && down.padded_shape[1] <= kMaxIntermediate &&
                gate_up.padded_shape[1] >= hidden && down.padded_shape[1] >= intermediate,
            "padded rows exceed the staged activation");
    auto* h = static_cast<float*>(workspace);
    if (row_split(gate_up, QType::Q4_G64_FP16) && row_split(down, QType::Q4_G64_FP16)) {
        require(gate_up.padded_shape[1] % 64 == 0 && down.padded_shape[1] % 64 == 0, "Q4 rows must hold whole groups");
        launch<Q4>(x, ids, gate_up, down, top_k, hidden, intermediate, h, outputs, stream);
    } else if (row_split(gate_up, QType::Q8_G32_FP16) && row_split(down, QType::Q8_G32_FP16)) {
        require(gate_up.padded_shape[1] % 32 == 0 && down.padded_shape[1] % 32 == 0, "Q8 rows must hold whole groups");
        launch<Q8>(x, ids, gate_up, down, top_k, hidden, intermediate, h, outputs, stream);
    } else {
        throw std::invalid_argument("resident_moe_experts: banks must be row-split Q4_G64_FP16 or Q8_G32_FP16");
    }
}

} // namespace infernix::ops
