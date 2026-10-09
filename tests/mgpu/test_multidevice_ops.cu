// The Ops of a Qwen3.8-Flash-Next layer split across two GPUs, bit for bit on each of them.
//
// Every case builds its host inputs once, then runs on device a, on device b and on device a again
// (--devices a,b; default 0,1). Each run uploads the inputs, enqueues the Op on that device's
// DeviceContext stream and downloads every output, and the three runs must agree bit for bit. The
// run on b catches CUDA host state kept for one device only (kernel attributes, SM counts, TMA
// descriptor buffers: core/per_device.h); the second run on a, state that the run on b disturbed.
// INFERNIX_NUMERICS_SMS pins the decompositions sized in SMs (QSA splits and prompt stages, QSA
// selection grids, the wide experts' persistent grid, the chunked GDN value tile, every
// DeviceExecutionView); when it is unset the test pins it to the smaller SM count of the two
// devices, so GPUs with different SM counts must agree as well. Cases, at the model's shapes:
//   - linear [16384, 2560] (the GDN input projection): q8_g32_fp16 at T = 1, 8 and 512; BF16 at
//     T = 64 and 512 (TMA routes);
//   - projection_fp32: the router and shared-expert gate rows [513, 2560] at T = 1, 8, 64; a
//     32,768-row head in q8_g32_fp16 at T = 1 and 12 and in BF16 at T = 4 and 12;
//   - hyper_connection_mix (S = 4, H = 2560, rank 320, with injects): fused T = 8, composed T = 64;
//   - qsa_attention over BF16 and INT8-G64 KV: decode (W = 1, two rows), W = 20 and W = 40 (whose
//     split, route or stages depend on the SM count) and a prompt-route call of W = 256;
//   - offloaded_sparse_moe (moe_route, moe_dispatch, moe_experts; records in device frames and in
//     pinned host memory read through staging slots): a narrow call (T = 2, top-10 of 24 experts)
//     and a wide one (T = 64);
//   - gated_delta_net, chunked prefill (16 Q/K heads, 48 value heads, T = 512).
//
// --streams runs on device a alone (one GPU suffices): two streams of the device enqueue eager TMA
// launches (the wide experts' grouped GEMMs, the BF16 TMA linear) concurrently, alternating
// without synchronization, and every launch must equal its stream's isolated reference. With one
// eager descriptor buffer per device instead of one per stream (core/tma_descriptor_staging.cuh),
// a staging kernel of one stream overwrites descriptors that a TMA kernel of the other is reading.
//
// Run both forms once under `compute-sanitizer --tool memcheck` as well.

#include "core/arena.h"
#include "core/device.h"
#include "core/paged_kv_cache.h"
#include "core/paged_kv_storage.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "infernix/ops/gated_delta_net.h"
#include "infernix/ops/hyper_connection.h"
#include "infernix/ops/linear.h"
#include "infernix/ops/offloaded_sparse_moe.h"
#include "infernix/ops/projection_fp32.h"
#include "infernix/ops/qsa.h"
#include "ops/direct_bf16_weight.h"
#include "ops/offloaded_moe_fixtures.h"
#include "ops/offloaded_sparse_moe/cuda/wide_expert.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <initializer_list>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

namespace ops      = infernix::ops;
namespace t        = infernix::test;
namespace moe      = infernix::ops::offloaded_moe;
namespace fixtures = infernix::test::offloaded_moe;
namespace qw       = infernix::test::quantized_weight;
using infernix::DeviceArena;
using infernix::DeviceBuffer;
using infernix::DeviceContext;
using infernix::DType;
using infernix::KvCacheStorage;
using infernix::QType;
using infernix::Tensor;
using infernix::Weight;
using Bytes = std::vector<std::uint8_t>;

// Model shapes (namespace scope: lambdas read them without capturing).
constexpr std::int32_t kHidden        = 2560;
constexpr std::int32_t kGdnProjection = 16384; // the GDN q, k, v and z projections
constexpr std::int32_t kExperts       = 512;   // router rows (the shared-expert gate adds one)
constexpr std::int32_t kHeadRows      = 32768; // a head as tall as the tall/MMA routes' floor
constexpr std::int32_t kStreams       = 4;     // hyper-connection streams
constexpr std::int32_t kRank          = 320;   // hyper-connection rank
constexpr std::int32_t kResidual      = kStreams * kHidden;
constexpr std::int32_t kTopK          = 10;
constexpr std::int32_t kGdnDim        = 128;
constexpr std::int32_t kGdnQkHeads    = 16;
constexpr std::int32_t kGdnValueHeads = 48;
constexpr std::int32_t kGdnTokens     = 512;
constexpr std::int32_t kQsaDim        = 256;
constexpr std::int32_t kQsaHeads      = 24;
constexpr std::int32_t kQsaKvHeads    = 2;
constexpr std::int32_t kQsaIndexHeads = 4;
constexpr std::int32_t kQsaIndexDim   = 128;
constexpr std::int32_t kQsaRatio      = 4;
constexpr std::int32_t kQsaPage       = 64;
constexpr int kStagingSlots           = 8;   // fewer than the misses: the jobs run in passes
constexpr std::int32_t kRaceTokens    = 512; // the two-stream BF16 linear's columns

const ops::QsaGeometry kQsaGeometry{.heads          = kQsaHeads,
                                    .kv_heads       = kQsaKvHeads,
                                    .head_dim       = kQsaDim,
                                    .index_heads    = kQsaIndexHeads,
                                    .index_head_dim = kQsaIndexDim,
                                    .rotary_dim     = 64,
                                    .budget         = 2048,
                                    .ratio          = kQsaRatio,
                                    .theta          = 1.0e7F,
                                    .eps            = 1.0e-6F};

int g_failures = 0;

void fail(const std::string& what) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++g_failures;
}

// ------------------------------------------------------------------------------------- inputs

// splitmix64: one portable stream of host inputs per seed.
class Random {
public:
    explicit Random(std::uint64_t seed) : state_(seed) {}

    std::uint64_t next() {
        std::uint64_t z = (state_ += 0x9E3779B97F4A7C15ULL);
        z               = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z               = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }

    // Uniform in [lo, hi).
    float uniform(float lo, float hi) {
        return lo + (hi - lo) * static_cast<float>(next() >> 40) * 0x1p-24F;
    }

    // Uniform in [lo, hi].
    int integer(int lo, int hi) {
        return lo + static_cast<int>(next() % static_cast<std::uint64_t>(hi - lo + 1));
    }

private:
    std::uint64_t state_;
};

std::uint16_t half_bits(float value) {
    const __half half  = __float2half_rn(value);
    std::uint16_t bits = 0;
    std::memcpy(&bits, &half, sizeof(bits));
    return bits;
}

std::vector<std::uint16_t> random_bf16(std::size_t count, float magnitude, std::uint64_t seed) {
    Random random(seed);
    std::vector<std::uint16_t> values(count);
    for (auto& value : values) { value = t::f32_to_bf16(random.uniform(-magnitude, magnitude)); }
    return values;
}

std::vector<float> random_f32(std::size_t count, float lo, float hi, std::uint64_t seed) {
    Random random(seed);
    std::vector<float> values(count);
    for (auto& value : values) { value = random.uniform(lo, hi); }
    return values;
}

// A row-split q8_g32_fp16 weight [n, k] of the model's scale: codes uniform in [-127, 127] (0 in
// the K padding) and each group's FP16 scale g / (127 sqrt(k)) with g in [2, 6), so projections
// stay of order one and the hyper-connection gates do not saturate.
qw::PackedWeight random_q8(std::int32_t n, std::int32_t k, std::uint64_t seed) {
    // The fixture lays the planes out; its (hashed, cheap) codes and scales are then replaced.
    qw::PackedWeight weight = qw::make_patterned_weight(
        QType::Q8_G32_FP16, n, k, static_cast<std::uint32_t>(seed),
        {qw::RowSplitScalePattern::Small, qw::RowSplitCodePattern::Hashed});
    const std::int32_t padded = weight.weight.padded_shape[1];
    const std::int32_t groups = padded / 32;
    const float unit          = 1.0F / (127.0F * std::sqrt(static_cast<float>(k)));
    Random random(seed);
    for (std::int32_t row = 0; row < n; ++row) {
        std::uint8_t* codes = weight.payload.data() + static_cast<std::size_t>(row) * padded;
        for (std::int32_t column = 0; column < padded; ++column) {
            const int code = column < k ? random.integer(-127, 127) : 0;
            codes[column]  = static_cast<std::uint8_t>(static_cast<std::int8_t>(code));
        }
        for (std::int32_t group = 0; group < groups; ++group) {
            const std::uint16_t bits = half_bits(random.uniform(2.0F, 6.0F) * unit);
            std::memcpy(weight.payload.data() + weight.scale_plane_offset +
                            (static_cast<std::size_t>(row) * groups + group) * sizeof(bits),
                        &bits, sizeof(bits));
        }
    }
    return weight;
}

// ------------------------------------------------------------------------------------ harness

template <class T>
DeviceBuffer upload(const std::vector<T>& host) {
    return t::to_device(host);
}

// cudaMemcpy from pageable memory may return before its DMA lands; the Ops run on the context's
// non-blocking stream, so wait for every upload before enqueueing them.
void inputs_ready() { t::cuda_check(cudaDeviceSynchronize(), "inputs"); }

// Synchronizes the context's stream and downloads `buffers`, concatenated in order.
Bytes outputs(const DeviceContext& context, std::initializer_list<const DeviceBuffer*> buffers) {
    context.synchronize();
    t::cuda_check(cudaGetLastError(), "launch");
    Bytes bytes;
    for (const DeviceBuffer* buffer : buffers) {
        const std::size_t at = bytes.size();
        bytes.resize(at + buffer->bytes);
        buffer->copy_to_host(bytes.data() + at, buffer->bytes);
    }
    return bytes;
}

// Byte positions where `b` differs from `a`: the count and the first.
std::pair<std::size_t, std::size_t> difference(const Bytes& a, const Bytes& b) {
    std::size_t differing = 0, first = a.size();
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i] && differing++ == 0) { first = i; }
    }
    return {differing, first};
}

// Runs `run` once per context, with the context's device current, and requires every run's
// outputs to equal the first run's bit for bit.
void check_case(const std::string& name, const std::vector<DeviceContext*>& contexts,
                const std::function<Bytes(DeviceContext&)>& run) {
    std::vector<Bytes> results;
    for (DeviceContext* context : contexts) {
        context->bind_to_current_thread();
        results.push_back(run(*context));
    }
    bool same = true;
    for (std::size_t i = 1; i < results.size(); ++i) {
        const std::string runs = "run " + std::to_string(i) + " (device " +
                                 std::to_string(contexts[i]->device) + ") against run 0 (device " +
                                 std::to_string(contexts[0]->device) + ")";
        if (results[i].size() != results[0].size()) {
            fail(name + ": " + runs + ": the outputs differ in size");
            same = false;
            continue;
        }
        const auto [differing, first] = difference(results[0], results[i]);
        if (differing != 0) {
            fail(name + ": " + runs + ": " + std::to_string(differing) + " of " +
                 std::to_string(results[0].size()) + " output bytes differ, the first at byte " +
                 std::to_string(first));
            same = false;
        }
    }
    std::printf("%s %s (%zu output bytes)\n", same ? "ok  " : "FAIL", name.c_str(),
                results.empty() ? std::size_t{0} : results[0].size());
}

// ------------------------------------------------------------------------------ linear cases

void q8_linear(const std::vector<DeviceContext*>& contexts) {
    const qw::PackedWeight weight = random_q8(kGdnProjection, kHidden, 101);
    for (const std::int32_t tokens : {1, 8, 512}) {
        const auto x = random_bf16(static_cast<std::size_t>(kHidden) * tokens, 1.0F, 103 + tokens);
        check_case("linear q8_g32_fp16 [16384, 2560] T=" + std::to_string(tokens), contexts,
                   [&](DeviceContext& context) {
                       const DeviceBuffer w = upload(weight.payload), dx = upload(x);
                       const DeviceBuffer out(static_cast<std::size_t>(kGdnProjection) * tokens *
                                              2);
                       const std::size_t capacity = ops::linear_workspace_capacity_bytes(
                           QType::Q8_G32_FP16, kGdnProjection, kHidden, ops::LinearPolicy::A16Only,
                           tokens, tokens);
                       DeviceArena workspace(std::max<std::size_t>(capacity, 256));
                       inputs_ready();
                       const Tensor input(dx.p, DType::BF16, {kHidden, tokens});
                       Tensor output(out.p, DType::BF16, {kGdnProjection, tokens});
                       ops::linear(input, weight.device_weight(w.p), output,
                                   ops::LinearPolicy::A16Only, workspace, context.stream);
                       return outputs(context, {&out});
                   });
    }
}

void bf16_linear(const std::vector<DeviceContext*>& contexts) {
    const auto weight = t::direct_bf16_weight::make_patterned(kGdnProjection, kHidden, 107);
    for (const std::int32_t tokens : {64, 512}) {
        const auto x = random_bf16(static_cast<std::size_t>(kHidden) * tokens, 1.0F, 109 + tokens);
        check_case("linear BF16 TMA [16384, 2560] T=" + std::to_string(tokens), contexts,
                   [&](DeviceContext& context) {
                       const DeviceBuffer w = upload(weight.bits), dx = upload(x);
                       const DeviceBuffer out(static_cast<std::size_t>(kGdnProjection) * tokens *
                                              2);
                       inputs_ready();
                       const Tensor input(dx.p, DType::BF16, {kHidden, tokens});
                       Tensor output(out.p, DType::BF16, {kGdnProjection, tokens});
                       ops::linear(input, weight.device_weight(w.p), output, context.stream);
                       return outputs(context, {&out});
                   });
    }
}

void router_projection(const std::vector<DeviceContext*>& contexts) {
    const auto router = t::direct_bf16_weight::make_patterned(kExperts, kHidden, 113);
    const auto gate   = t::direct_bf16_weight::make_patterned(1, kHidden, 127);
    for (const std::int32_t tokens : {1, 8, 64}) {
        const auto x = random_bf16(static_cast<std::size_t>(kHidden) * tokens, 1.0F, 131 + tokens);
        check_case("projection_fp32 router [513, 2560] T=" + std::to_string(tokens), contexts,
                   [&](DeviceContext& context) {
                       const DeviceBuffer dr = upload(router.bits), dg = upload(gate.bits);
                       const DeviceBuffer dx = upload(x);
                       const DeviceBuffer out(static_cast<std::size_t>(kExperts + 1) * tokens * 4);
                       inputs_ready();
                       const Tensor rows(dr.p, DType::BF16, {kHidden, kExperts});
                       const Tensor shared(dg.p, DType::BF16, {kHidden, 1});
                       const Tensor* weights[] = {&rows, &shared};
                       Tensor logits(out.p, DType::FP32, {kExperts + 1, tokens});
                       ops::projection_fp32(Tensor(dx.p, DType::BF16, {kHidden, tokens}), weights,
                                            logits, context.stream);
                       return outputs(context, {&out});
                   });
    }
}

void head_projection(const std::vector<DeviceContext*>& contexts) {
    {
        const qw::PackedWeight head = random_q8(kHeadRows, kHidden, 137);
        for (const std::int32_t tokens : {1, 12}) {
            const auto x =
                random_bf16(static_cast<std::size_t>(kHidden) * tokens, 1.0F, 139 + tokens);
            check_case("projection_fp32 q8_g32_fp16 head [32768, 2560] T=" + std::to_string(tokens),
                       contexts, [&](DeviceContext& context) {
                           const DeviceBuffer w = upload(head.payload), dx = upload(x);
                           const DeviceBuffer out(static_cast<std::size_t>(kHeadRows) * tokens * 4);
                           inputs_ready();
                           Tensor logits(out.p, DType::FP32, {kHeadRows, tokens});
                           ops::projection_fp32(Tensor(dx.p, DType::BF16, {kHidden, tokens}),
                                                head.device_weight(w.p), logits, context.stream);
                           return outputs(context, {&out});
                       });
        }
    }
    const auto head = t::direct_bf16_weight::make_patterned(kHeadRows, kHidden, 149);
    for (const std::int32_t tokens : {4, 12}) {
        const auto x = random_bf16(static_cast<std::size_t>(kHidden) * tokens, 1.0F, 151 + tokens);
        check_case("projection_fp32 BF16 head [32768, 2560] T=" + std::to_string(tokens), contexts,
                   [&](DeviceContext& context) {
                       const DeviceBuffer w = upload(head.bits), dx = upload(x);
                       const DeviceBuffer out(static_cast<std::size_t>(kHeadRows) * tokens * 4);
                       inputs_ready();
                       const Tensor rows(w.p, DType::BF16, {kHidden, kHeadRows});
                       const Tensor* weights[] = {&rows};
                       Tensor logits(out.p, DType::FP32, {kHeadRows, tokens});
                       ops::projection_fp32(Tensor(dx.p, DType::BF16, {kHidden, tokens}), weights,
                                            logits, context.stream);
                       return outputs(context, {&out});
                   });
    }
}

// --------------------------------------------------------------------- hyper-connection mixer

void hyper_connection(const std::vector<DeviceContext*>& contexts) {
    const qw::PackedWeight down = random_q8(kRank + kStreams, kResidual, 157);
    const qw::PackedWeight up   = random_q8(kResidual, kRank, 163);
    const auto norm             = random_bf16(kResidual, 0.25F, 167);
    for (const std::int32_t tokens : {8, 64}) {
        // Streams at different scales, so that the per-stream norms matter.
        std::vector<std::uint16_t> residual(static_cast<std::size_t>(kResidual) * tokens);
        Random random(173 + tokens);
        for (std::size_t i = 0; i < residual.size(); ++i) {
            const int stream = static_cast<int>(i % kResidual) / kHidden;
            residual[i] =
                t::f32_to_bf16(random.uniform(-1.0F, 1.0F) * static_cast<float>(1 << stream));
        }
        check_case(
            std::string("hyper_connection_mix ") + (tokens <= 16 ? "fused" : "composed") +
                " T=" + std::to_string(tokens),
            contexts, [&](DeviceContext& context) {
                const DeviceBuffer dd = upload(down.payload), du = upload(up.payload);
                const DeviceBuffer dn = upload(norm), dr = upload(residual);
                const DeviceBuffer x(static_cast<std::size_t>(kHidden) * tokens * 2);
                const DeviceBuffer inject(static_cast<std::size_t>(kStreams) * tokens * 4);
                const Weight down_w        = down.device_weight(dd.p);
                const Weight up_w          = up.device_weight(du.p);
                const std::size_t capacity = ops::hyper_connection_mix_workspace_capacity_bytes(
                    down_w, up_w, ops::LinearPolicy::A16Only, kStreams, kRank, tokens);
                DeviceArena workspace(std::max<std::size_t>(capacity, 256));
                inputs_ready();
                Tensor mixed(x.p, DType::BF16, {kHidden, tokens});
                Tensor injects(inject.p, DType::FP32, {kStreams, tokens});
                ops::hyper_connection_mix(Tensor(dr.p, DType::BF16, {kResidual, tokens}),
                                          Tensor(dn.p, DType::BF16, {kResidual}), down_w, up_w,
                                          ops::LinearPolicy::A16Only, kStreams, kRank, 1.0e-6F,
                                          mixed, &injects, workspace, context.stream);
                return outputs(context, {&x, &inject});
            });
    }
}

// --------------------------------------------------------------------------------------- QSA

// Random contents of a paged plane of `count` elements of `dtype`: values of order one, or the
// small positive FP16 scales of an INT8-G64 plane.
Bytes kv_plane(DType dtype, std::size_t count, bool scales, Random& random) {
    Bytes bytes(count * infernix::dtype_size(dtype));
    for (std::size_t i = 0; i < count; ++i) {
        std::uint16_t bits = 0;
        switch (dtype) {
        case DType::BF16:
            bits = t::f32_to_bf16(random.uniform(-2.0F, 2.0F));
            std::memcpy(bytes.data() + 2 * i, &bits, 2);
            break;
        case DType::FP16:
            bits = half_bits(scales ? random.uniform(0.004F, 0.012F) : random.uniform(-2.0F, 2.0F));
            std::memcpy(bytes.data() + 2 * i, &bits, 2);
            break;
        case DType::I8:
            bytes[i] =
                static_cast<std::uint8_t>(static_cast<std::int8_t>(random.integer(-127, 127)));
            break;
        default:
            throw std::logic_error("kv_plane: no generator for this dtype");
        }
    }
    return bytes;
}

// One qsa_attention call: `starts.size()` rows of `width` columns, row r's first at starts[r],
// over shuffled pages.
void qsa_case(const std::vector<DeviceContext*>& contexts, KvCacheStorage storage,
              const std::string& name, std::int32_t width, const std::vector<std::int32_t>& starts,
              std::uint64_t seed) {
    const auto layout                = infernix::paged_kv_storage_layout(storage, kQsaDim);
    const std::int32_t rows          = static_cast<std::int32_t>(starts.size());
    const std::int32_t width_columns = rows * width;
    std::int32_t tokens              = 0;
    for (const std::int32_t start : starts) { tokens = std::max(tokens, start + width); }
    const std::int32_t pages_per_row = (tokens + kQsaPage - 1) / kQsaPage;
    const std::int32_t pages         = rows * pages_per_row;
    const std::int32_t max_context   = pages_per_row * kQsaPage;
    Random random(seed);
    std::vector<std::int32_t> tables(static_cast<std::size_t>(pages));
    std::iota(tables.begin(), tables.end(), 0);
    for (std::size_t i = tables.size(); i > 1; --i) {
        std::swap(tables[i - 1], tables[random.next() % i]);
    }
    const std::size_t token_slots = static_cast<std::size_t>(kQsaPage) * kQsaKvHeads * pages;

    const auto plane = [&](const infernix::PagedKVVectorLayout& vector, bool scale) {
        if (!scale) {
            return kv_plane(vector.data_dtype, token_slots * vector.data_leading_extent, false,
                            random);
        }
        if (!vector.has_scale()) { return Bytes{}; }
        return kv_plane(vector.scale_dtype, token_slots * vector.scale_leading_extent, true,
                        random);
    };
    const Bytes k_data = plane(layout.key, false), k_scales = plane(layout.key, true);
    const Bytes v_data = plane(layout.value, false), v_scales = plane(layout.value, true);
    const auto pooled = random_bf16(
        static_cast<std::size_t>(kQsaIndexDim / kQsaRatio) * kQsaPage * pages, 1.0F, random.next());
    const auto q = random_bf16(static_cast<std::size_t>(kQsaDim) * kQsaHeads * width_columns, 1.0F,
                               random.next());
    const auto iq =
        random_bf16(static_cast<std::size_t>(kQsaIndexDim) * kQsaIndexHeads * width_columns, 1.0F,
                    random.next());
    std::vector<std::int32_t> positions(static_cast<std::size_t>(width_columns)), table_rows(rows),
        tail_slots(rows, 0);
    for (std::int32_t r = 0; r < rows; ++r) {
        table_rows[static_cast<std::size_t>(r)] = r;
        for (std::int32_t j = 0; j < width; ++j) {
            positions[static_cast<std::size_t>(r) * width + j] =
                starts[static_cast<std::size_t>(r)] + j;
        }
    }
    check_case("qsa_attention " + name, contexts, [&](DeviceContext& context) {
        const DeviceBuffer dk = upload(k_data), dv = upload(v_data);
        DeviceBuffer dks, dvs;
        if (!k_scales.empty()) { dks = upload(k_scales); }
        if (!v_scales.empty()) { dvs = upload(v_scales); }
        const DeviceBuffer dpooled = upload(pooled), dq = upload(q), diq = upload(iq);
        const DeviceBuffer dtables = upload(tables), dpos = upload(positions);
        const DeviceBuffer drows = upload(table_rows), dslots = upload(tail_slots);
        ops::QsaKVLayer layer;
        layer.kv.storage      = storage;
        layer.kv.head_dim     = kQsaDim;
        layer.kv.num_kv_heads = kQsaKvHeads;

        const auto view = [&](const DeviceBuffer& buffer, DType dtype, std::int32_t leading) {
            return Tensor(buffer.p, dtype, {leading, kQsaPage, kQsaKvHeads, pages});
        };
        layer.kv.k_pages = view(dk, layout.key.data_dtype, layout.key.data_leading_extent);
        layer.kv.v_pages = view(dv, layout.value.data_dtype, layout.value.data_leading_extent);
        if (layout.key.has_scale()) {
            layer.kv.k_scale_pages =
                view(dks, layout.key.scale_dtype, layout.key.scale_leading_extent);
        }
        if (layout.value.has_scale()) {
            layer.kv.v_scale_pages =
                view(dvs, layout.value.scale_dtype, layout.value.scale_leading_extent);
        }
        layer.pooled_pages =
            Tensor(dpooled.p, DType::BF16, {kQsaIndexDim / kQsaRatio, kQsaPage, 1, pages});
        const ops::QsaBatch batch{.block_tables =
                                      Tensor(dtables.p, DType::I32, {pages_per_row, rows}),
                                  .table_rows = Tensor(drows.p, DType::I32, {rows}),
                                  .positions  = Tensor(dpos.p, DType::I32, {width_columns}),
                                  .tail_slots = Tensor(dslots.p, DType::I32, {rows}),
                                  .batch      = rows,
                                  .width      = width};
        const std::size_t workspace_bytes =
            ops::qsa_attention_workspace_bytes(kQsaGeometry, width_columns, max_context, storage);
        const DeviceBuffer workspace(workspace_bytes);
        const DeviceBuffer out(static_cast<std::size_t>(kQsaDim) * kQsaHeads * width_columns * 2);
        inputs_ready();
        Tensor attended(out.p, DType::BF16, {kQsaDim, kQsaHeads, width_columns});
        ops::qsa_attention(
            Tensor(dq.p, DType::BF16, {kQsaDim, kQsaHeads, width_columns}),
            Tensor(diq.p, DType::BF16, {kQsaIndexDim, kQsaIndexHeads, width_columns}), layer, batch,
            kQsaGeometry, 1.0F / 16.0F, max_context, workspace.p, workspace_bytes, attended,
            context.stream);
        return outputs(context, {&out});
    });
}

void qsa_cases(const std::vector<DeviceContext*>& contexts) {
    struct Shape {
        const char* name;
        std::int32_t width;
        std::vector<std::int32_t> starts;
    };

    // With 2 KV heads, W = 18-21 splits on 84 SMs but takes the prompt route on 70; W = 35-41 takes
    // the prompt route, with 1 stage on 70 SMs and 2 on 84 where two CTAs fit an SM.
    const Shape shapes[] = {{"decode W=1 B=2", 1, {2600, 3100}},
                            {"W=20 B=1 (SM-dependent split)", 20, {5000}},
                            {"W=40 B=1 (SM-dependent prompt stages)", 40, {5000}},
                            {"prompt W=256 B=1", 256, {6001}}};
    std::uint64_t seed   = 181;
    for (const auto& [storage, label] : {std::pair{KvCacheStorage::BFloat16, "bf16"},
                                         std::pair{KvCacheStorage::Int8Group64, "int8"}}) {
        for (const Shape& shape : shapes) {
            qsa_case(contexts, storage, std::string(label) + " " + shape.name, shape.width,
                     shape.starts, seed++);
        }
    }
}

// ----------------------------------------------------------------------------- routed experts

// A layer's experts: every third record in a device frame, every record in host memory.
struct MoeBank {
    std::int32_t experts = 0;
    std::vector<moe::ExpertScales> scales;
    std::vector<std::int32_t> frames; // expert -> frame, -1 when not resident
    Bytes frame_records;              // the resident records in frame order
    Bytes host_records;               // every record in expert order
};

MoeBank make_moe_bank(std::int32_t experts, std::uint32_t seed) {
    std::mt19937 rng(seed);
    MoeBank bank;
    bank.experts = experts;
    bank.frames.assign(static_cast<std::size_t>(experts), -1);
    bank.host_records.resize(moe::kRecordBytes * static_cast<std::size_t>(experts));
    std::int32_t resident = 0;
    for (std::int32_t e = 0; e < experts; ++e) {
        // One gate/up input scale per expert: the wide route's condition.
        const fixtures::Expert expert = fixtures::random_expert(rng, false);
        bank.scales.push_back(expert.scales);
        std::copy(expert.record.begin(), expert.record.end(),
                  bank.host_records.begin() + static_cast<std::ptrdiff_t>(moe::kRecordBytes * e));
        if (e % 3 == 0) {
            bank.frames[static_cast<std::size_t>(e)] = resident++;
            bank.frame_records.insert(bank.frame_records.end(), expert.record.begin(),
                                      expert.record.end());
        }
    }
    return bank;
}

// One call's inputs: router logits [E + 1, T] whose top-k in every column are k experts drawn
// without replacement by `popularity` (logits above 20; the rest below 1), and BF16 x [H, T].
struct MoeInputs {
    std::int32_t columns = 0;
    std::vector<float> logits;
    std::vector<std::uint16_t> x;
    std::int32_t wide_experts = 0; // experts that take the wide route
};

MoeInputs make_moe_inputs(const MoeBank& bank, std::int32_t columns,
                          const std::vector<double>& popularity, std::uint64_t seed) {
    Random random(seed);
    MoeInputs inputs;
    inputs.columns = columns;
    inputs.logits.resize(static_cast<std::size_t>(bank.experts + 1) * columns);
    std::vector<std::int32_t> routed(static_cast<std::size_t>(bank.experts), 0);
    for (std::int32_t column = 0; column < columns; ++column) {
        float* logits =
            inputs.logits.data() + static_cast<std::size_t>(column) * (bank.experts + 1);
        for (std::int32_t e = 0; e <= bank.experts; ++e) {
            logits[e] = random.uniform(-1.0F, 1.0F);
        }
        std::vector<double> weight = popularity;
        for (std::int32_t rank = 0; rank < kTopK; ++rank) {
            const double total = std::accumulate(weight.begin(), weight.end(), 0.0);
            double pick        = static_cast<double>(random.next() >> 11) * 0x1p-53 * total;
            std::int32_t e     = 0;
            while (e + 1 < bank.experts && (weight[static_cast<std::size_t>(e)] == 0.0 ||
                                            pick >= weight[static_cast<std::size_t>(e)])) {
                pick -= weight[static_cast<std::size_t>(e)];
                ++e;
            }
            while (weight[static_cast<std::size_t>(e)] == 0.0) { --e; } // rounding past the last
            weight[static_cast<std::size_t>(e)] = 0.0;
            logits[e]                           = 20.0F + static_cast<float>(kTopK - rank);
            ++routed[static_cast<std::size_t>(e)];
        }
    }
    for (std::int32_t e = 0; e < bank.experts; ++e) {
        inputs.wide_experts += moe::wide_route(routed[static_cast<std::size_t>(e)],
                                               bank.scales[static_cast<std::size_t>(e)])
                                   ? 1
                                   : 0;
    }
    std::mt19937 rng(static_cast<std::uint32_t>(seed));
    inputs.x = fixtures::random_activations(rng, columns);
    return inputs;
}

// Pinned, mapped host memory freed on scope exit.
class PinnedHost {
public:
    explicit PinnedHost(const Bytes& contents) {
        t::cuda_check(cudaHostAlloc(&host_, contents.size(), cudaHostAllocMapped), "cudaHostAlloc");
        std::memcpy(host_, contents.data(), contents.size());
        t::cuda_check(cudaHostGetDevicePointer(&device_, host_, 0), "cudaHostGetDevicePointer");
    }

    ~PinnedHost() {
        if (host_ != nullptr) { cudaFreeHost(host_); }
    }

    PinnedHost(const PinnedHost&)            = delete;
    PinnedHost& operator=(const PinnedHost&) = delete;

    [[nodiscard]] const std::uint8_t* device() const {
        return static_cast<const std::uint8_t*>(device_);
    }

private:
    void* host_   = nullptr;
    void* device_ = nullptr;
};

// One offloaded_sparse_moe call on the current device, with device state of its own: the bank's
// frames and pinned records (misses staged through kStagingSlots slots), the routing and dispatch
// of the inputs' logits, the workspace.
class MoeCall {
public:
    MoeCall(const MoeBank& bank, const MoeInputs& inputs)
        : experts_(bank.experts), columns_(inputs.columns), entries_(kTopK * inputs.columns),
          max_jobs_(std::min(bank.experts, entries_)), host_(bank.host_records),
          frame_records_(upload(bank.frame_records)), frames_(upload(bank.frames)),
          scales_(upload(bank.scales)), logits_(upload(inputs.logits)), x_(upload(inputs.x)),
          ids_(sizeof(std::int32_t) * entries_), weights_(sizeof(float) * entries_),
          shared_gate_(sizeof(float) * columns_),
          dispatch_(ops::moe_dispatch_bytes(experts_, entries_)),
          staging_(moe::kRecordBytes * kStagingSlots),
          workspace_(ops::moe_experts_workspace_bytes(max_jobs_, entries_)) {}

    [[nodiscard]] std::size_t output_bytes() const {
        return static_cast<std::size_t>(moe::kHidden) * entries_ * 2;
    }

    // moe_route and moe_dispatch.
    void route(cudaStream_t stream) {
        routing_ = ops::MoeRouting{Tensor(ids_.p, DType::I32, {kTopK, columns_}),
                                   Tensor(weights_.p, DType::FP32, {kTopK, columns_}),
                                   Tensor(shared_gate_.p, DType::FP32, {columns_})};
        ops::moe_route(Tensor(logits_.p, DType::FP32, {experts_ + 1, columns_}), kTopK, routing_,
                       stream);
        dispatch_view_ = ops::carve_moe_dispatch(dispatch_.p, experts_, entries_);
        ops::moe_dispatch(routing_, experts_, dispatch_view_, nullptr, stream);
    }

    // moe_experts into `out` (BF16 [H, k * T]), after route() in stream order.
    void experts(void* out, cudaStream_t stream) {
        const ops::MoeExpertSource source{
            .frame_base    = static_cast<const std::uint8_t*>(frame_records_.p),
            .frames        = static_cast<const std::int32_t*>(frames_.p),
            .host_records  = host_.device(),
            .record_stride = moe::kRecordBytes,
            .scales        = static_cast<const moe::ExpertScales*>(scales_.p),
            .staging_base  = static_cast<std::uint8_t*>(staging_.p),
            .staging_slots = kStagingSlots};
        Tensor outputs_view(out, DType::BF16, {moe::kHidden, entries_});
        ops::moe_experts(Tensor(x_.p, DType::BF16, {moe::kHidden, columns_}), dispatch_view_,
                         source, kTopK, max_jobs_, workspace_.p, outputs_view, stream);
    }

    [[nodiscard]] const DeviceBuffer& ids() const { return ids_; }

    [[nodiscard]] const DeviceBuffer& weights() const { return weights_; }

    [[nodiscard]] const DeviceBuffer& shared_gate() const { return shared_gate_; }

private:
    std::int32_t experts_, columns_, entries_, max_jobs_;
    PinnedHost host_;
    DeviceBuffer frame_records_, frames_, scales_, logits_, x_;
    DeviceBuffer ids_, weights_, shared_gate_, dispatch_, staging_, workspace_;
    ops::MoeRouting routing_;
    ops::MoeDispatch dispatch_view_;
};

void moe_case(const std::vector<DeviceContext*>& contexts, const std::string& name,
              const MoeBank& bank, const MoeInputs& inputs, bool wide) {
    if ((inputs.wide_experts > 0) != wide) {
        fail(name + ": the routing does not exercise the " + (wide ? "wide" : "narrow") + " route");
    }
    check_case(name + ", " + std::to_string(inputs.wide_experts) + " wide experts", contexts,
               [&](DeviceContext& context) {
                   MoeCall call(bank, inputs);
                   const DeviceBuffer out(call.output_bytes());
                   inputs_ready();
                   call.route(context.stream);
                   call.experts(out.p, context.stream);
                   return outputs(context,
                                  {&call.ids(), &call.weights(), &call.shared_gate(), &out});
               });
}

void routed_experts(const std::vector<DeviceContext*>& contexts) {
    const MoeBank bank = make_moe_bank(24, 191);
    // Narrow: 20 entries over 24 experts, at most 2 columns each.
    moe_case(contexts, "offloaded_sparse_moe narrow T=2 top-10 of 24", bank,
             make_moe_inputs(bank, 2, std::vector<double>(24, 1.0), 193), false);
    // Wide: about 30 columns per popular expert; four rare experts may stay narrow.
    std::vector<double> popularity(24, 1.0);
    for (std::size_t e = 20; e < 24; ++e) { popularity[e] = 0.02; }
    moe_case(contexts, "offloaded_sparse_moe wide T=64 top-10 of 24", bank,
             make_moe_inputs(bank, 64, popularity, 197), true);
}

// ---------------------------------------------------------------------------- Gated DeltaNet

void chunked_gdn(const std::vector<DeviceContext*>& contexts) {
    const std::size_t qk_count = static_cast<std::size_t>(kGdnDim) * kGdnQkHeads * kGdnTokens;
    const std::size_t v_count  = static_cast<std::size_t>(kGdnDim) * kGdnValueHeads * kGdnTokens;
    const std::size_t gates    = static_cast<std::size_t>(kGdnValueHeads) * kGdnTokens;
    const std::size_t state    = static_cast<std::size_t>(kGdnDim) * kGdnDim * kGdnValueHeads;
    const auto q               = random_bf16(qk_count, 1.0F, 211);
    const auto k               = random_bf16(qk_count, 1.0F, 223);
    const auto v               = random_bf16(v_count, 1.0F, 227);
    const auto g               = random_f32(gates, -1.0F, -0.01F, 229); // log decay
    const auto beta            = random_f32(gates, 0.05F, 0.95F, 233);
    const auto initial         = random_f32(state, -0.1F, 0.1F, 239);
    check_case("gated_delta_net chunked Hqk=16 Hv=48 T=512", contexts, [&](DeviceContext& context) {
        const DeviceBuffer dq = upload(q), dk = upload(k), dv = upload(v);
        const DeviceBuffer dg = upload(g), dbeta = upload(beta), dinitial = upload(initial);
        const DeviceBuffer final_state(state * sizeof(float));
        const DeviceBuffer out(v_count * 2);
        DeviceArena workspace(
            std::max<std::size_t>(ops::gated_delta_net_workspace_capacity_bytes(
                                      kGdnQkHeads, kGdnValueHeads, kGdnTokens, kGdnTokens),
                                  256));
        inputs_ready();
        Tensor state_out(final_state.p, DType::FP32, {kGdnDim, kGdnDim, kGdnValueHeads});
        Tensor o(out.p, DType::BF16, {kGdnDim, kGdnValueHeads, kGdnTokens});
        ops::gated_delta_net(Tensor(dq.p, DType::BF16, {kGdnDim, kGdnQkHeads, kGdnTokens}),
                             Tensor(dk.p, DType::BF16, {kGdnDim, kGdnQkHeads, kGdnTokens}),
                             Tensor(dv.p, DType::BF16, {kGdnDim, kGdnValueHeads, kGdnTokens}),
                             Tensor(dg.p, DType::FP32, {kGdnValueHeads, kGdnTokens}),
                             Tensor(dbeta.p, DType::FP32, {kGdnValueHeads, kGdnTokens}),
                             1.0F / std::sqrt(static_cast<float>(kGdnDim)), true, workspace,
                             Tensor(dinitial.p, DType::FP32, {kGdnDim, kGdnDim, kGdnValueHeads}),
                             state_out, o, context.execution_view());
        return outputs(context, {&out, &final_state});
    });
}

// ------------------------------------------------------------------------------- two streams

class Stream {
public:
    Stream() {
        t::cuda_check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
                      "cudaStreamCreate");
    }

    ~Stream() { cudaStreamDestroy(stream_); }

    Stream(const Stream&)            = delete;
    Stream& operator=(const Stream&) = delete;

    [[nodiscard]] cudaStream_t get() const { return stream_; }

private:
    cudaStream_t stream_ = nullptr;
};

// launch(s, output, stream) enqueues stream s's call (s = 0, 1; distinct inputs) writing `output`.
// Each stream's reference is its call alone, synchronized; then `rounds` times both streams enqueue
// their call, alternating, without synchronizing, and every raced output must equal its reference.
void check_race(const std::string& name, const Stream (&streams)[2], std::size_t output_bytes,
                int rounds, const std::function<void(int, void*, cudaStream_t)>& launch) {
    std::vector<DeviceBuffer> reference, raced;
    for (int s = 0; s < 2; ++s) { reference.emplace_back(output_bytes); }
    for (int i = 0; i < 2 * rounds; ++i) { raced.emplace_back(output_bytes); }
    for (int s = 0; s < 2; ++s) {
        launch(s, reference[static_cast<std::size_t>(s)].p, streams[s].get());
        t::cuda_synchronize(streams[s].get());
    }
    for (int round = 0; round < rounds; ++round) {
        for (int s = 0; s < 2; ++s) {
            launch(s, raced[static_cast<std::size_t>(2 * round + s)].p, streams[s].get());
        }
    }
    for (const Stream& stream : streams) { t::cuda_synchronize(stream.get()); }
    t::cuda_check(cudaGetLastError(), "launch");
    const auto download = [&](const DeviceBuffer& buffer) {
        Bytes bytes(output_bytes);
        buffer.copy_to_host(bytes.data(), output_bytes);
        return bytes;
    };
    const Bytes expected[2] = {download(reference[0]), download(reference[1])};
    bool same               = true;
    if (expected[0] == expected[1]) {
        fail(name + ": both streams' calls give the same outputs, so a crossed read cannot show");
        same = false;
    }
    for (int i = 0; i < 2 * rounds; ++i) {
        const auto [differing, first] =
            difference(expected[i % 2], download(raced[static_cast<std::size_t>(i)]));
        if (differing != 0) {
            fail(name + ": stream " + std::to_string(i % 2) + " round " + std::to_string(i / 2) +
                 ": " + std::to_string(differing) + " of " + std::to_string(output_bytes) +
                 " output bytes differ from its isolated launch, the first at byte " +
                 std::to_string(first));
            same = false;
        }
    }
    std::printf("%s %s, %d rounds on two streams\n", same ? "ok  " : "FAIL", name.c_str(), rounds);
}

void two_streams(DeviceContext& context) {
    context.bind_to_current_thread();
    std::printf("eager TMA launches of two streams on device %d\n", context.device);
    const Stream streams[2];
    constexpr int kRounds = 6;
    {
        const auto weight = t::direct_bf16_weight::make_patterned(kGdnProjection, kHidden, 251);
        const std::size_t columns = static_cast<std::size_t>(kHidden) * kRaceTokens;
        const DeviceBuffer w      = upload(weight.bits);
        const DeviceBuffer x[2]   = {upload(random_bf16(columns, 1.0F, 257)),
                                     upload(random_bf16(columns, 1.0F, 263))};
        inputs_ready();
        check_race("linear BF16 TMA [16384, 2560] T=512", streams,
                   static_cast<std::size_t>(kGdnProjection) * kRaceTokens * 2, kRounds,
                   [&](int s, void* out, cudaStream_t stream) {
                       Tensor output(out, DType::BF16, {kGdnProjection, kRaceTokens});
                       ops::linear(Tensor(x[s].p, DType::BF16, {kHidden, kRaceTokens}),
                                   weight.device_weight(w.p), output, stream);
                   });
    }
    {
        const MoeBank bank        = make_moe_bank(24, 269);
        const MoeInputs inputs[2] = {make_moe_inputs(bank, 64, std::vector<double>(24, 1.0), 271),
                                     make_moe_inputs(bank, 64, std::vector<double>(24, 1.0), 277)};
        if (inputs[0].wide_experts == 0 || inputs[1].wide_experts == 0) {
            fail("two streams: the routing does not exercise the wide route");
        }
        MoeCall first(bank, inputs[0]), second(bank, inputs[1]);
        MoeCall* calls[2] = {&first, &second};
        inputs_ready();
        for (int s = 0; s < 2; ++s) { calls[s]->route(streams[s].get()); }
        for (const Stream& stream : streams) { t::cuda_synchronize(stream.get()); }
        check_race("offloaded_sparse_moe wide T=64 top-10 of 24", streams, first.output_bytes(),
                   kRounds,
                   [&](int s, void* out, cudaStream_t stream) { calls[s]->experts(out, stream); });
    }
}

// ---------------------------------------------------------------------------------------- main

// INFERNIX_NUMERICS_SMS is read once per process, so it is set before the first DeviceContext or
// Op reads it: unless the environment sets it, to the smaller SM count of the two devices.
void pin_numerics_sms(int a, int b) {
    if (const char* value = std::getenv("INFERNIX_NUMERICS_SMS");
        value != nullptr && *value != '\0') {
        std::printf("INFERNIX_NUMERICS_SMS=%s (from the environment)\n", value);
        return;
    }
    int sms[2] = {};
    t::cuda_check(cudaDeviceGetAttribute(&sms[0], cudaDevAttrMultiProcessorCount, a), "SM count");
    t::cuda_check(cudaDeviceGetAttribute(&sms[1], cudaDevAttrMultiProcessorCount, b), "SM count");
    const std::string pinned = std::to_string(std::min(sms[0], sms[1]));
#ifdef _WIN32
    const int status = _putenv_s("INFERNIX_NUMERICS_SMS", pinned.c_str());
#else
    const int status = setenv("INFERNIX_NUMERICS_SMS", pinned.c_str(), 1);
#endif
    if (status != 0) { throw std::runtime_error("cannot set INFERNIX_NUMERICS_SMS"); }
    std::printf("INFERNIX_NUMERICS_SMS=%s (devices %d and %d have %d and %d SMs)\n", pinned.c_str(),
                a, b, sms[0], sms[1]);
}

int usage() {
    std::fprintf(stderr, "usage: infernix_multidevice_ops_test [--devices a,b] [--streams]\n");
    return 2;
}

} // namespace

int main(int argc, char** argv) {
    // Unbuffered, so that a crashing case's last progress line survives.
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    bool streams          = false;
    bool explicit_devices = false;
    int devices[2]        = {0, 1};
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        if (arg == "--streams") {
            streams = true;
        } else if (arg == "--devices" && i + 1 < argc) {
            const std::string list(argv[++i]);
            const std::size_t comma = list.find(',');
            if (comma == std::string::npos) { return usage(); }
            devices[0]       = std::atoi(list.substr(0, comma).c_str());
            devices[1]       = std::atoi(list.substr(comma + 1).c_str());
            explicit_devices = true;
        } else {
            return usage();
        }
    }
    if (t::cuda_unavailable()) {
        std::printf("SKIP: no usable CUDA device\n");
        return 77;
    }
    int count = 0;
    t::cuda_check(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
    if (!streams && !explicit_devices && count < 2) {
        std::printf(
            "SKIP: fewer than two CUDA devices (--devices 0,0 runs both contexts on one)\n");
        return 77;
    }
    for (int i = 0; i < (streams ? 1 : 2); ++i) {
        if (devices[i] < 0 || devices[i] >= count) {
            std::fprintf(stderr, "FAIL: no CUDA device %d (%d visible)\n", devices[i], count);
            return 1;
        }
    }
    try {
        if (streams) {
            DeviceContext context(devices[0]);
            two_streams(context);
        } else {
            pin_numerics_sms(devices[0], devices[1]);
            DeviceContext a(devices[0]);
            DeviceContext b(devices[1]);
            const std::vector<DeviceContext*> contexts = {&a, &b, &a};
            q8_linear(contexts);
            bf16_linear(contexts);
            router_projection(contexts);
            head_projection(contexts);
            hyper_connection(contexts);
            qsa_cases(contexts);
            routed_experts(contexts);
            chunked_gdn(contexts);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
    if (g_failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("all multi-device op checks passed\n");
    return 0;
}
