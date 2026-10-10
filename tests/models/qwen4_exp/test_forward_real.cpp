// Qwen4Exp forward on the real converted artifact, against the independent FP64 reference
// (tools/flash_next/reference.py). Skips unless INFERNIX_QWEN4_ARTIFACT names a Qwen4Exp artifact.
//
//   INFERNIX_QWEN4_ARTIFACT=out.ninfer [INFERNIX_QWEN4_NGRAM=out.ninfer.ngram]
//   infernix_qwen4_exp_forward_real_test TOKENS [--kv KV] [--logits OUT.bin] [--residuals OUT.bin]
//       [--routes OUT.bin] [--blocks OUT.bin]
//   infernix_qwen4_exp_forward_real_test TOKENS --dump-logits OUT.bin [--chunk N] [--kv KV]
//       [--dump-from P] [TOKENS --dump-logits OUT.bin [--chunk N] [--kv KV] [--dump-from P]]...
//   infernix_qwen4_exp_forward_real_test TOKENS --cpu-columns [--kv KV]
//   infernix_qwen4_exp_forward_real_test TOKENS --split K[,K...] [--split-device D] [--cpu-columns]
//       [--kv KV]
//
// KV is a KV-cache storage: bf16 (default), int8, fp8, nvfp4 or k8v4.
//
// With --cpu-columns the test checks that calls the Program serves with the CPU expert service
// (decode widths, and prefill calls up to 255 columns with the assist) give the GPU route's logits
// bit for bit (check_cpu_columns).
// With --split each K splits the decoder layers (models/qwen4_exp/segment_map.h) between Forwards
// over [0, K) on device 0 and [K, L) on device D (default 0: a second stream of device 0), each
// with its own state, KV, tails, workspace and expert sources, joined by a HandoffChannel once in
// Event and once in Gate mode. A prefill chunk of min(4096, n - 5) tokens, the next token as a
// decode step and the four after it as a verification call (teacher-forced drafts, replay records)
// must give the unsplit Forward's logits, every residual tap, the route and landing logs and the
// verification records bit for bit (check_split). The same prefill as four chunks enqueued back to
// back, crossing through a BulkHandoff (a Program's eager crossing) with no host wait between them,
// must give the unsplit chunked prefill's last chunk bit for bit (on one GPU a stream event keeps
// the segments' chunks from overlapping: overlapping, they are not bit-exact; SplitPair::run_bulk). Both use a Program's kinds of
// expert sources:
// four resident experts per layer in device frames, 64 staging slots, the overlap stream and four
// landing frames per layer, the split's segments reading slices of one table per device. With
// --cpu-columns as well, the CPU-served calls also run split, through a CPU service with a port per
// segment. Across two GPUs set INFERNIX_NUMERICS_SMS to the smaller SM count; segment 1 then reads a
// copy of the model's device weights made on device D (this test does not load per segment).
// TOKENS is a comma-separated id list, or @FILE holding ids separated by commas or whitespace; the
// options after a TOKENS apply to it. With --dump-logits the test scores the text teacher-forced
// instead: it prefills it in chunks of N (default 256) with FP32 logits at every position, writes
// them in Strata's --dump-logits layout (int32 vocabulary, int32 rows, then one FP32 row per
// position) and prints the perplexity. With --dump-from P only positions from P on are written and
// scored (long texts: the logits of every position would not fit on disk). Several such texts are
// scored in order with one model load. --rope-yarn-factor F, anywhere, loads the model with that text
// YaRN factor (contexts up to the native positions times F).
// Otherwise the test prefills all but the last token as one chunk,
// then decodes the last token twice in one batch (two sequences with identical histories), and
// checks that both decode rows equal each other bit for bit and agree with a single prefill of
// every token at the last position. It writes the prefill logits of the last position (FP32) for
// comparison with the reference.

#include "artifact/reader.h"
#include "kv_cache_storage.h"
#include "core/device.h"
#include "core/device_guard.h"
#include "core/gdn_replay_records.h"
#include "core/handoff.h"
#include "core/host_memory.h"
#include "core/layout.h"
#include "core/linear_attention_state.h"
#include "core/paged_kv_cache.h"
#include "core/paged_kv_storage.h"
#include "core/per_device.h"
#include "models/qwen4_exp/execution/forward.h"
#include "models/qwen4_exp/execution/parameters.h"
#include "models/qwen4_exp/frontend/ngram_hash.h"
#include "models/qwen4_exp/load.h"
#include "models/qwen4_exp/program/prefix/state_image.h"
#include "models/qwen4_exp/program/ngram_volume.h"
#include "models/qwen4_exp/program/rope_positions.h"
#include "models/qwen4_exp/segment_map.h"
#include "ops/offloaded_sparse_moe/cpu/miss_service.h"

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iterator>
#include <memory>
#include <numeric>
#include <span>
#include <sstream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#endif

using namespace infernix;
namespace q4 = infernix::models::qwen4_exp;

namespace {

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e)); }
}

std::vector<std::int32_t> parse_tokens(const std::string& argument) {
    std::string text = argument;
    if (!text.empty() && text.front() == '@') {
        std::ifstream file(text.substr(1));
        if (!file) { throw std::invalid_argument("cannot read " + text.substr(1)); }
        text.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    }
    for (char& ch : text) {
        if (ch == ',' || ch == '\n' || ch == '\r' || ch == '\t') { ch = ' '; }
    }
    std::vector<std::int32_t> out;
    std::stringstream stream(text);
    std::int32_t id = 0;
    while (stream >> id) { out.push_back(id); }
    return out;
}

// Host memory after each stage: the pinned weights are the bulk; the rest should stay small.
void print_memory(const char* stage) {
    constexpr double gib = 1073741824.0;
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS_EX counters{};
    if (K32GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters),
                                sizeof(counters))) {
        std::printf("[memory] %s: private %.2f GiB, working set %.2f GiB, system available %.2f GiB\n", stage,
                    counters.PrivateUsage / gib, counters.WorkingSetSize / gib, available_host_memory_bytes() / gib);
        return;
    }
#endif
    std::printf("[memory] %s: system available %.2f GiB\n", stage, available_host_memory_bytes() / gib);
}

double seconds_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
}

std::vector<std::uint32_t> parse_splits(const std::string& value) {
    std::vector<std::uint32_t> out;
    std::stringstream stream(value);
    std::string item;
    while (std::getline(stream, item, ',')) {
        out.push_back(static_cast<std::uint32_t>(std::stoul(item)));
    }
    if (out.empty()) { throw std::invalid_argument("--split needs K[,K...]"); }
    return out;
}

// An object holding memory, streams or events of one device, built and destroyed with that device
// current (a layer split drives two devices from this thread).
template <class T>
class OnDevice {
public:
    OnDevice() = default;

    template <class... Args>
    explicit OnDevice(int device, Args&&... args) : device_(device) {
        const DeviceGuard on(device);
        value_ = std::make_unique<T>(std::forward<Args>(args)...);
    }

    // Takes an object built while `device` was current.
    static OnDevice adopt(int device, std::unique_ptr<T> value) {
        OnDevice out;
        out.device_ = device;
        out.value_  = std::move(value);
        return out;
    }

    ~OnDevice() { reset(); }

    OnDevice(OnDevice&& other) noexcept : device_(other.device_), value_(std::move(other.value_)) {}

    OnDevice& operator=(OnDevice&& other) noexcept {
        if (this != &other) {
            reset();
            device_ = other.device_;
            value_  = std::move(other.value_);
        }
        return *this;
    }

    void reset() noexcept {
        if (!value_) { return; }
        try {
            const DeviceGuard on(device_);
            value_.reset();
        } catch (...) { value_.reset(); }
    }

    [[nodiscard]] T& operator*() const noexcept { return *value_; }

    [[nodiscard]] T* operator->() const noexcept { return value_.get(); }

    [[nodiscard]] T* get() const noexcept { return value_.get(); }

private:
    int device_ = 0;
    std::unique_ptr<T> value_;
};

// Every slot's recurrent state, KV pages for `rows` sequences of up to `context` tokens, and no
// resident experts (every expert is read zero-copy from the pinned bank), on the current device.
// `range` ({0, 0}: every layer) holds a layer split's segment instead: its state, KV and tails then
// cover its own layers only, from 0, and `forward` stays null (forward_with builds its Forwards).
struct Harness {
    Harness(const q4::execution::Parameters& parameters, DeviceContext& device, std::int32_t context, std::int32_t columns,
            std::int32_t rows, KvCacheStorage storage, q4::LayerRange range = {})
        : config(parameters.model.config().text) {
        const auto& c = config;
        layers        = range.end == 0 ? q4::LayerRange{0, c.num_hidden_layers} : range;
        // Each kind's compact indices count it in layer order: the range's slices start after the
        // layers of that kind before it.
        for (std::uint32_t l = 0; l < layers.end; ++l) {
            const bool gdn_layer  = c.layer_types[l] == q4::MixerKind::Gdn;
            std::uint32_t& before = gdn_layer ? gdn_base : attention_base;
            std::uint32_t& inside = gdn_layer ? gdn_count : attention_count;
            ++(l < layers.begin ? before : inside);
        }
        if (gdn_count > 0) {
            LayoutBuilder builder;
            const LinearAttentionStatePoolSpec gdn_spec{
                .layers         = gdn_count,
                .conv_channels  = static_cast<std::int32_t>(c.gdn.conv_channels()),
                .conv_width     = static_cast<std::int32_t>(c.gdn.conv_kernel - 1),
                .value_heads    = static_cast<std::int32_t>(c.gdn.value_heads),
                .value_head_dim = static_cast<std::int32_t>(c.gdn.value_head_dim),
                .key_head_dim   = static_cast<std::int32_t>(c.gdn.key_head_dim),
                .slot_count     = rows,
            };
            const auto gdn_layout = plan_linear_attention_state_pool(builder, gdn_spec);
            state_bytes           = builder.finish(256);
            state_backing         = DeviceBuffer(state_bytes);
            state_backing.fill(0);
            gdn = std::make_unique<LinearAttentionStatePool>(DeviceSpan{state_backing.p, state_backing.bytes},
                                                             gdn_layout);
        }

        const std::int32_t width = static_cast<std::int32_t>(c.residual_width());
        const std::int32_t span  = static_cast<std::int32_t>(c.ple.conv_span());
        ple_backing = DeviceBuffer(static_cast<std::size_t>(width) * span * rows * 2);
        ple_backing.fill(0);
        const std::int32_t di = static_cast<std::int32_t>(c.qsa.index_head_dim);
        const std::int32_t r  = static_cast<std::int32_t>(c.qsa.compress_ratio);
        if (attention_count > 0) {
            tails_backing = DeviceBuffer(static_cast<std::size_t>(attention_count) * di * (r - 1) * rows * 2);
            tails_backing.fill(0);
        }

        const auto layout = paged_kv_storage_layout(storage, static_cast<std::int32_t>(c.attention.head_dim));
        pages_per_row     = (context + kPagedKVPageSize - 1) / kPagedKVPageSize;
        if (attention_count > 0) {
            KVPageGeometry geometry;
            for (std::uint32_t l = 0; l < attention_count; ++l) {
                const auto kv_heads = static_cast<std::int32_t>(c.attention.kv_heads);
                geometry.planes.push_back({layout.key.data_dtype, layout.key.data_leading_extent, kv_heads});
                if (layout.key.has_scale()) {
                    geometry.planes.push_back({layout.key.scale_dtype, layout.key.scale_leading_extent, kv_heads});
                }
                geometry.planes.push_back({layout.value.data_dtype, layout.value.data_leading_extent, kv_heads});
                if (layout.value.has_scale()) {
                    geometry.planes.push_back({layout.value.scale_dtype, layout.value.scale_leading_extent, kv_heads});
                }
                geometry.planes.push_back({DType::BF16, di / r, 1});
            }
            LayoutBuilder kv_builder;
            const auto pool_layout = plan_device_kv_page_pool(
                kv_builder, {.page_group_count = static_cast<std::uint32_t>(pages_per_row * rows), .geometry = geometry});
            const auto table_layout = plan_kv_execution_tables(
                kv_builder, {.logical_page_capacity = static_cast<std::uint32_t>(pages_per_row), .table_rows = rows});
            kv_backing = DeviceBuffer(kv_builder.finish(256));
            kv_backing.fill(0);
            pool   = std::make_unique<DeviceKVPagePool>(DeviceSpan{kv_backing.p, kv_backing.bytes}, pool_layout);
            tables = std::make_unique<KVExecutionTablePool>(DeviceSpan{kv_backing.p, kv_backing.bytes}, table_layout,
                                                            *pool);
            for (std::int32_t row = 0; row < rows; ++row) {
                auto reservation = pool->reserve(static_cast<std::uint32_t>(pages_per_row));
                if (!reservation) { throw std::runtime_error("KV pool too small"); }
                std::vector<DeviceKVPageLease> pages;
                pages.reserve(static_cast<std::size_t>(pages_per_row));
                pool->materialize(*reservation, static_cast<std::uint32_t>(pages_per_row), pages);
                row_leases.push_back(tables->acquire(row));
                tables->publish(row_leases.back().handle(), 0, pages, device.stream);
                page_leases.push_back(std::move(pages));
                reservations.push_back(std::move(*reservation));
            }
            device.synchronize();
        }

        frames_backing = DeviceBuffer(sizeof(std::int32_t) * c.moe.experts * layers.size());
        frames_backing.fill(0xFF); // -1: every expert served from the pinned host bank

        q4::execution::ForwardState state;
        state.gdn      = gdn.get();
        state.ple_conv = Tensor(ple_backing.p, DType::BF16, {width, span, rows});
        for (std::uint32_t l = 0; l < attention_count; ++l) {
            state.qsa_tails.push_back(Tensor(static_cast<std::byte*>(tails_backing.p) +
                                                 static_cast<std::size_t>(l) * di * (r - 1) * rows * 2,
                                             DType::BF16, {di, r - 1, rows}));
        }
        // vq2/k4v2: the exact window of every attention layer and row, zeroed (no tag matches).
        window_geometry =
            q4::prefix::kv_window_geometry(c, storage, attention_count, false, static_cast<std::uint32_t>(rows));
        if (window_geometry.present()) {
            window_backing = DeviceBuffer(window_geometry.bytes());
            window_backing.fill(0);
        }
        q4::execution::ForwardKV kv;
        if (tables) { kv.block_tables = tables->matrix(); }
        std::size_t plane   = 0;
        const auto kv_heads = static_cast<std::int32_t>(c.attention.kv_heads);
        for (std::uint32_t l = 0; l < attention_count; ++l) {
            ops::QsaKVLayer layer;
            layer.kv.storage      = storage;
            layer.kv.head_dim     = static_cast<std::int32_t>(c.attention.head_dim);
            layer.kv.num_kv_heads = kv_heads;
            layer.kv.k_pages      = pool->plane(plane++);
            if (layout.key.has_scale()) { layer.kv.k_scale_pages = pool->plane(plane++); }
            layer.kv.v_pages = pool->plane(plane++);
            if (layout.value.has_scale()) { layer.kv.v_scale_pages = pool->plane(plane++); }
            layer.pooled_pages = pool->plane(plane++);
            if (window_geometry.present()) {
                layer.kv.window = q4::prefix::kv_window_view(window_geometry, window_backing.p, l);
            }
            kv.layers.push_back(layer);
        }
        q4::execution::ForwardExperts experts;
        for (std::uint32_t l = 0; l < layers.size(); ++l) {
            experts.frames.push_back(static_cast<const std::int32_t*>(frames_backing.p) + l * c.moe.experts);
        }
        work_capacity = q4::execution::Forward::workspace_bytes(c, columns, context, storage);
        work          = std::make_unique<WorkspaceArena>(work_capacity);
        state_view    = state;
        kv_view       = kv;
        experts_view  = experts;
        if (layers.begin == 0 && layers.end == c.num_hidden_layers) {
            forward = std::make_unique<q4::execution::Forward>(parameters, device, *work, std::move(state),
                                                               std::move(kv), std::move(experts), context);
        }
    }

    // Another Forward over the same state, KV and workspace, with other expert sources (a segment's
    // Forward with the harness's segment()).
    std::unique_ptr<q4::execution::Forward>
    forward_with(const q4::execution::Parameters& parameters, DeviceContext& device,
                 q4::execution::ForwardExperts sources, std::int32_t context,
                 const q4::execution::ForwardSegment& segment = {}) const {
        return std::make_unique<q4::execution::Forward>(parameters, device, *work, state_view,
                                                        kv_view, std::move(sources), context, segment);
    }

    // The harness's layers as a segment of a split, joined to its neighbours by `in` and `out`.
    [[nodiscard]] q4::execution::ForwardSegment segment(HandoffChannel* in, HandoffChannel* out,
                                                        const PinnedGateBound& bound) const {
        q4::execution::ForwardSegment s;
        s.layers         = layers;
        s.gdn_base       = gdn_base;
        s.attention_base = attention_base;
        s.embed          = layers.begin == 0;
        s.finish         = layers.end == config.num_hidden_layers;
        s.in             = in;
        s.out            = out;
        s.in_bound       = bound;
        return s;
    }

    // Verification records of `batch` sequences of `width` columns for the harness's layers (one
    // set: each call reuses it).
    q4::execution::ForwardVerify verify(std::int32_t width, std::int32_t batch) {
        const auto& c = config;
        q4::execution::ForwardVerify v;
        if (gdn_count > 0) {
            LayoutBuilder builder;
            const auto layout = plan_gdn_replay_records(
                builder, {.layers          = static_cast<std::int32_t>(gdn_count),
                          .record_capacity = batch,
                          .width           = width,
                          .conv_channels   = static_cast<std::int32_t>(c.gdn.conv_channels()),
                          .qk_heads        = static_cast<std::int32_t>(c.gdn.key_heads),
                          .value_heads     = static_cast<std::int32_t>(c.gdn.value_heads),
                          .key_dim         = static_cast<std::int32_t>(c.gdn.key_head_dim),
                          .value_dim       = static_cast<std::int32_t>(c.gdn.value_head_dim)});
            records_backing = DeviceBuffer(builder.finish(256));
            v.gdn           = GdnReplayRecords(DeviceSpan{records_backing.p, records_backing.bytes}, layout);
        }
        const auto W  = static_cast<std::int32_t>(c.residual_width());
        const auto di = static_cast<std::int32_t>(c.qsa.index_head_dim);
        ple_records   = DeviceBuffer(static_cast<std::size_t>(W) * width * batch * 2);
        v.ple_inputs  = Tensor(ple_records.p, DType::BF16, {W, width, batch});
        if (attention_count > 0) {
            qsa_records = DeviceBuffer(static_cast<std::size_t>(di) * width * batch * attention_count * 2);
            v.qsa_keys  = Tensor(qsa_records.p, DType::BF16,
                                 {di, width, batch, static_cast<std::int32_t>(attention_count)});
        }
        return v;
    }

    const q4::TextConfig& config;
    q4::LayerRange layers;
    // The layers of each kind before the range (its compact bases) and in it.
    std::uint32_t gdn_base = 0, attention_base = 0, gdn_count = 0, attention_count = 0;
    DeviceBuffer records_backing, ple_records, qsa_records;
    q4::execution::ForwardState state_view;
    q4::execution::ForwardKV kv_view;
    q4::execution::ForwardExperts experts_view;
    std::size_t state_bytes = 0;
    DeviceBuffer state_backing, ple_backing, tails_backing, kv_backing, frames_backing;
    DeviceBuffer window_backing;
    q4::prefix::KvWindowGeometry window_geometry;
    std::unique_ptr<LinearAttentionStatePool> gdn;
    std::unique_ptr<DeviceKVPagePool> pool;
    std::unique_ptr<KVExecutionTablePool> tables;
    std::vector<DeviceKVPageReservation> reservations;
    std::vector<std::vector<DeviceKVPageLease>> page_leases;
    std::vector<KVExecutionRowLease> row_leases;
    std::int32_t pages_per_row = 0;
    std::size_t work_capacity  = 0;
    std::unique_ptr<WorkspaceArena> work;
    std::unique_ptr<q4::execution::Forward> forward;
};

// Device copies of one call's inputs.
struct Call {
    DeviceBuffer buffer;
    q4::execution::ForwardBatch batch;
    std::vector<std::int32_t> host_slots, host_rows;
};

Call make_call(const q4::TextConfig& c, const q4::NgramVolume& volume, const std::vector<std::vector<std::int32_t>>& histories,
               std::int32_t first, std::int32_t width, std::vector<std::int32_t> slots, std::vector<std::int32_t> rows,
               bool every_column = false) {
    const auto batch = static_cast<std::int32_t>(histories.size());
    const std::int32_t columns = batch * width;
    const q4::NgramHash hash(c.ple.ngram, 0);
    const std::size_t row_bytes = c.ple.table.row_bytes;
    const std::size_t heads     = hash.heads();
    std::vector<std::int32_t> ids, positions, last;
    // Text prompts: RoPE positions equal the index on all three axes.
    std::vector<std::int32_t> rope(3 * static_cast<std::size_t>(columns)), block_rope(3 * static_cast<std::size_t>(batch));
    const q4::LaneRope text;
    std::vector<std::byte> ngram(columns * heads * row_bytes);
    for (std::int32_t b = 0; b < batch; ++b) {
        const auto& history = histories[b];
        std::vector<std::int32_t> padded(c.ple.ngram.ngram_size - 1, c.eos_token_id);
        padded.insert(padded.end(), history.begin(), history.begin() + first + width);
        std::vector<std::uint32_t> row_ids(static_cast<std::size_t>(width) * heads);
        hash.row_ids(padded, static_cast<std::size_t>(width), row_ids.data());
        volume.read_rows(row_ids, std::span<std::byte>(ngram).subspan(static_cast<std::size_t>(b) * width * heads * row_bytes,
                                                                    static_cast<std::size_t>(width) * heads * row_bytes));
        for (std::int32_t i = 0; i < width; ++i) {
            ids.push_back(history[first + i]);
            positions.push_back(first + i);
        }
        q4::stage_rope(text, static_cast<std::uint32_t>(first), width, rope, columns, b * width);
        q4::stage_rope_value(q4::block_start_rope(text, static_cast<std::uint32_t>(first), c.qsa.compress_ratio),
                             block_rope, batch, b);
        if (every_column) {
            for (std::int32_t i = 0; i < width; ++i) { last.push_back(b * width + i); }
        } else {
            last.push_back(b * width + width - 1);
        }
    }
    Call call;
    const std::size_t i32 = sizeof(std::int32_t);
    const auto padded_bytes = [](std::size_t n) { return (n + 255) / 256 * 256; };
    const std::size_t bytes = padded_bytes(ids.size() * i32) + padded_bytes(positions.size() * i32) +
                              padded_bytes(rope.size() * i32) + padded_bytes(block_rope.size() * i32) +
                              padded_bytes(slots.size() * i32) + padded_bytes(rows.size() * i32) +
                              padded_bytes(last.size() * i32) + padded_bytes(ngram.size());
    call.buffer = DeviceBuffer(bytes);
    auto* base = static_cast<std::byte*>(call.buffer.p);
    std::size_t offset = 0;
    auto put = [&](const void* data, std::size_t n) {
        call.buffer.copy_from_host(data, n, offset);
        void* p = base + offset;
        offset += (n + 255) / 256 * 256;
        return p;
    };
    call.batch.ids        = Tensor(put(ids.data(), ids.size() * i32), DType::I32, {columns});
    call.batch.positions  = Tensor(put(positions.data(), positions.size() * i32), DType::I32, {columns});
    call.batch.rope_positions   = Tensor(put(rope.data(), rope.size() * i32), DType::I32, {columns, 3});
    call.batch.block_start_rope = Tensor(put(block_rope.data(), block_rope.size() * i32), DType::I32, {batch, 3});
    call.batch.slots      = Tensor(put(slots.data(), slots.size() * i32), DType::I32, {batch});
    call.batch.table_rows = Tensor(put(rows.data(), rows.size() * i32), DType::I32, {batch});
    call.batch.logit_columns =
        Tensor(put(last.data(), last.size() * i32), DType::I32, {static_cast<std::int32_t>(last.size())});
    call.batch.ngram_rows = Tensor(put(ngram.data(), ngram.size()), DType::U8,
                                   {static_cast<std::int32_t>(row_bytes), static_cast<std::int32_t>(heads), columns});
    call.host_slots       = std::move(slots);
    call.host_rows        = std::move(rows);
    call.batch.host_slots = call.host_slots;
    call.batch.host_table_rows = call.host_rows;
    call.batch.batch = batch;
    call.batch.width = width;
    return call;
}

std::vector<float> to_float(const DeviceBuffer& logits, std::size_t count) {
    std::vector<float> out(count);
    logits.copy_to_host(out.data(), count * sizeof(float));
    return out;
}

std::vector<int> top(const std::vector<float>& v, std::size_t offset, std::size_t n, int k) {
    std::vector<int> index(n);
    std::iota(index.begin(), index.end(), 0);
    std::partial_sort(index.begin(), index.begin() + k, index.end(),
                      [&](int a, int b) { return v[offset + a] > v[offset + b]; });
    index.resize(k);
    return index;
}

// ---- layer split (--split) ----------------------------------------------------------------------

const char* mode_name(HandoffMode mode) { return mode == HandoffMode::Gate ? "Gate" : "Event"; }

// make_call's inputs of one sequence in slot and row 0, on `device`.
OnDevice<Call> call_on(int device, const q4::TextConfig& c, const q4::NgramVolume& volume,
                       const std::vector<std::int32_t>& tokens, std::int32_t first,
                       std::int32_t width, bool every_column) {
    const DeviceGuard on(device);
    return OnDevice<Call>::adopt(device,
                                 std::make_unique<Call>(make_call(c, volume, {tokens}, first, width,
                                                                  {0}, {0}, every_column)));
}

// The expert sources of the split check on one device, by decoder layer: kResident experts per
// layer in device frames (the others are read from the pinned banks), and kLanding free frames per
// layer that decode and verification calls land their staged misses in
// (ops::MoeExpertSource::landing). Built on the current device.
struct ExpertFrames {
    static constexpr std::int32_t kResident = 4, kLanding = 4;

    explicit ExpertFrames(const q4::execution::Parameters& parameters) {
        const auto& c         = parameters.model.config().text;
        const std::uint32_t L = c.num_hidden_layers;
        experts               = static_cast<std::int32_t>(c.moe.experts);
        stride                = parameters.layers.front().moe.bank->planes.record_stride;
        pool = DeviceBuffer(static_cast<std::size_t>(L) * (kResident + kLanding) * stride);
        std::vector<std::int32_t> resident(static_cast<std::size_t>(L) * experts, -1), free_frames;
        for (std::uint32_t l = 0; l < L; ++l) {
            const auto& bank = *parameters.layers[l].moe.bank;
            for (std::int32_t i = 0; i < kResident; ++i) {
                const auto e = (l * 37U + static_cast<std::uint32_t>(i) * 61U) %
                               static_cast<std::uint32_t>(experts);
                const auto frame =
                    static_cast<std::size_t>(l) * kResident + static_cast<std::size_t>(i);
                resident[static_cast<std::size_t>(l) * experts + e] =
                    static_cast<std::int32_t>(frame);
                check(cudaMemcpy(static_cast<std::byte*>(pool.p) + frame * stride,
                                 bank.planes.records + e * stride, stride, cudaMemcpyHostToDevice),
                      "cudaMemcpy expert record");
            }
            for (std::int32_t j = 0; j < kLanding; ++j) {
                free_frames.push_back(static_cast<std::int32_t>(L) * kResident +
                                      static_cast<std::int32_t>(l) * kLanding + j);
            }
        }
        table = DeviceBuffer(resident.size() * sizeof(std::int32_t));
        table.copy_from_host(resident.data(), table.bytes);
        landing = DeviceBuffer(free_frames.size() * sizeof(std::int32_t));
        landing.copy_from_host(free_frames.data(), landing.bytes);
    }

    std::int32_t experts = 0;
    std::uint64_t stride = 0;
    DeviceBuffer pool, table, landing; // frames; I32 [L][E] frame or -1; I32 [L][kLanding]
};

// The route log (I32 [L][k * columns]) and landed log (I32 [L][kLanding]) of the calls run on one
// device, by decoder layer: a segment's Forward writes them from its first layer's rows.
struct ExpertLogs {
    ExpertLogs(const q4::TextConfig& c, std::int32_t columns)
        : stride(static_cast<std::size_t>(c.moe.top_k) * static_cast<std::size_t>(columns)),
          route(stride * c.num_hidden_layers * sizeof(std::int32_t)),
          landed(static_cast<std::size_t>(c.num_hidden_layers) * ExpertFrames::kLanding *
                 sizeof(std::int32_t)) {}

    // With the device current: every entry -1, so entries a call does not write compare equal.
    void clear() {
        route.fill(0xFF);
        landed.fill(0xFF);
    }

    std::size_t stride = 0;
    DeviceBuffer route, landed;
};

// A Forward's staging slots and its overlap stream with the five events (ops::MoeExpertSource), on
// the current device.
struct SideStream {
    static constexpr std::int32_t kStagingSlots = 64;

    explicit SideStream(std::uint64_t record_stride) : staging(kStagingSlots * record_stride) {
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");
        for (auto& event : events) {
            check(cudaEventCreateWithFlags(&event, cudaEventDisableTiming), "cudaEventCreate");
        }
    }

    ~SideStream() {
        for (auto event : events) {
            if (event != nullptr) { cudaEventDestroy(event); }
        }
        if (stream != nullptr) { cudaStreamDestroy(stream); }
    }

    SideStream(const SideStream&)            = delete;
    SideStream& operator=(const SideStream&) = delete;

    DeviceBuffer staging;
    cudaStream_t stream = nullptr;
    std::array<cudaEvent_t, 5> events{};
};

// The expert sources of a Forward over `layers` from one device's frames, logs and side stream: its
// slices of the per-layer tables, as a Program hands a segment its residency's.
q4::execution::ForwardExperts expert_sources(const ExpertFrames& frames, ExpertLogs& logs,
                                             const SideStream& side, q4::LayerRange layers) {
    q4::execution::ForwardExperts x;
    x.frame_base   = static_cast<const std::uint8_t*>(frames.pool.p);
    x.frame_stride = frames.stride;
    for (std::uint32_t l = layers.begin; l < layers.end; ++l) {
        x.frames.push_back(static_cast<const std::int32_t*>(frames.table.p) +
                           static_cast<std::size_t>(l) * frames.experts);
    }
    const std::size_t landing_from =
        static_cast<std::size_t>(layers.begin) * ExpertFrames::kLanding;
    x.route_log      = static_cast<std::int32_t*>(logs.route.p) + layers.begin * logs.stride;
    x.route_stride   = logs.stride;
    x.staging_base   = static_cast<std::uint8_t*>(side.staging.p);
    x.staging_slots  = SideStream::kStagingSlots;
    x.overlap_stream = side.stream;
    x.overlap_events = side.events;
    x.landing        = static_cast<const std::int32_t*>(frames.landing.p) + landing_from;
    x.landed         = static_cast<std::int32_t*>(logs.landed.p) + landing_from;
    x.landing_slots  = ExpertFrames::kLanding;
    return x;
}

// What every split of a check shares: segment s reads parameters[s] and runs on devices[s]
// (devices[0] is the reference's context; devices[1] a second context, on the same device in the
// same-device mode).
struct SplitSetup {
    std::array<const q4::execution::Parameters*, 2> parameters{};
    std::array<DeviceContext*, 2> devices{};
    std::int32_t context = 0, columns = 0;
    KvCacheStorage kv = KvCacheStorage::BFloat16;
};

// A segment's expert sources, given its harness (built on its device, which is current).
using SourcesOf =
    std::function<q4::execution::ForwardExperts(std::size_t segment, const Harness& harness)>;

// Two Forwards over decoder layers [0, K) and [K, L), each over its own harness (state, KV, tails
// and workspace for its layers, one slot) on its device, joined by a HandoffChannel in `mode`.
class SplitPair {
public:
    SplitPair(const SplitSetup& setup, std::uint32_t k, HandoffMode mode, const SourcesOf& sources)
        : setup_(setup), mode_(mode), error_(64) {
        std::memset(error_.data(), 0, error_.size());
        const auto& c         = setup.parameters[0]->model.config().text;
        const std::uint32_t L = c.num_hidden_layers;
        channel_              = std::make_unique<HandoffChannel>(
            device(0), device(1), static_cast<std::size_t>(c.residual_width()) * setup.columns * 2);
        const PinnedGateBound bound{.timeout_ns  = 1'000'000'000ULL,
                                    .error_word  = static_cast<std::uint32_t*>(error_.data()),
                                    .error_value = (k << 16) | kErrorHandoffTimeout};
        const q4::LayerRange ranges[2] = {{0, k}, {k, L}};
        gate_waits_ = OnDevice<DeviceBuffer>(device(1), 2 * sizeof(std::uint64_t));
        {
            const DeviceGuard on(device(1));
            gate_waits_->fill(0);
        }
        for (std::size_t s = 0; s < 2; ++s) {
            harness_[s] = OnDevice<Harness>(device(s), *setup.parameters[s], *setup.devices[s],
                                            setup.context, setup.columns, 1, setup.kv, ranges[s]);
            words_[s]   = OnDevice<DeviceBuffer>(device(s), sizeof(std::uint32_t));
            const DeviceGuard on(device(s));
            auto segment = harness_[s]->segment(s == 1 ? channel_.get() : nullptr,
                                                s == 0 ? channel_.get() : nullptr, bound);
            if (s == 1) { segment.in_wait_stats = static_cast<std::uint64_t*>(gate_waits_->p); }
            forward_[s] = OnDevice<q4::execution::Forward>::adopt(
                device(s),
                harness_[s]->forward_with(*setup.parameters[s], *setup.devices[s],
                                          sources(s, *harness_[s]), setup.context, segment));
        }
    }

    [[nodiscard]] Harness& harness(std::size_t s) const { return *harness_[s]; }

    // One call: calls[s] holds segment s's inputs on its device; the logits are on segment 1's.
    // Event mode orders segment 1 after segment 0's send with the channel's event (the host never
    // waits). Gate mode either launches segment 1's gate first, so that it waits for segment 0's
    // word, or launches segment 1 once segment 0 has finished (long calls, which would outlast the
    // gate's bound). Ends with both devices synchronized; a gate timeout throws.
    void run(const std::array<Call*, 2>& calls, Tensor* logits,
             const q4::execution::ForwardTap* tap, bool receiver_first) {
        const std::uint32_t word = ++word_;
        for (std::size_t s = 0; s < 2; ++s) {
            const DeviceGuard on(device(s));
            check(cudaMemcpy(words_[s]->p, &word, sizeof(word), cudaMemcpyHostToDevice),
                  "cudaMemcpy handoff word");
            calls[s]->batch.handoff      = mode_;
            calls[s]->batch.handoff_word = static_cast<const std::uint32_t*>(words_[s]->p);
        }
        // Enqueues segment s and submits it (a batched driver model holds launches otherwise).
        const auto enqueue = [&](std::size_t s) {
            const DeviceGuard on(device(s));
            forward_[s]->run(calls[s]->batch, s == 1 ? logits : nullptr, tap);
            setup_.devices[s]->flush();
        };
        if (mode_ == HandoffMode::Event) {
            enqueue(0);
            channel_->record_sent(setup_.devices[0]->stream);
            {
                const DeviceGuard on(device(0));
                setup_.devices[0]->flush();
            }
            channel_->wait_sent(setup_.devices[1]->stream);
            enqueue(1);
        } else if (receiver_first) {
            // Only segment 1's gate goes ahead of segment 0; its layers follow segment 0's launches
            // (run()'s steps, so the same bits). Work enqueued behind a waiting gate held segment 0
            // back on one GPU (RTX PRO 4000, WDDM: segment 0's launches never ran and the gate
            // expired), and a long segment 1 can fill the launch queue before segment 0 is
            // enqueued at all.
            q4::execution::Forward& receiver         = *forward_[1];
            const q4::execution::ForwardBatch& batch = calls[1]->batch;
            Tensor residual;
            {
                const DeviceGuard on(device(1));
                receiver.begin_call(batch, logits);
                residual = receiver.alloc_residual(batch);
                receiver.receive(batch, residual);
                setup_.devices[1]->flush();
            }
            enqueue(0);
            const DeviceGuard on(device(1));
            const q4::LayerRange layers = receiver.segment().layers;
            for (std::uint32_t l = layers.begin; l < layers.end; ++l) {
                receiver.layer(l, batch, residual, tap);
            }
            receiver.finish(batch, residual, logits);
            setup_.devices[1]->flush();
        } else {
            enqueue(0);
            {
                const DeviceGuard on(device(0));
                setup_.devices[0]->synchronize();
            }
            enqueue(1);
        }
        for (const std::size_t s : {std::size_t{1}, std::size_t{0}}) {
            const DeviceGuard on(device(s));
            setup_.devices[s]->synchronize();
        }
        const std::uint32_t error = *static_cast<const volatile std::uint32_t*>(error_.data());
        if (error != 0) {
            throw std::runtime_error("the handoff gate at layer " + std::to_string(error >> 16) +
                                     " timed out (error word " + std::to_string(error) + ")");
        }
    }

    // Prefill chunks back to back, crossing through `bulk` as a Program's eager calls do (a
    // HandoffChannel's slot holds one decode or verification residual): segment 0's steps, then
    // BulkHandoff::transfer of its residual into segment 1's (alloc_residual's fixed address in its
    // workspace), then segment 1's steps; each device is flushed, and the host never waits until
    // the last chunk is enqueued. Ends with both devices synchronized.
    //
    // Across two GPUs segment 0's next chunk runs while segment 1 computes the last one. On one
    // GPU it waits for it (a stream event, still no host wait): with the two streams' kernels
    // overlapping, the MoE output of a 2,048-column call came out different with an identical
    // input (layer 0 of the second chunk, K = 47; RTX PRO 4000, 2026-10-10), the timing-dependent
    // results of design §19.3.8's "F1 nondeterminism".
    void run_bulk(std::span<const std::array<Call*, 2>> chunks, BulkHandoff& bulk, Tensor* logits,
                  const q4::execution::ForwardTap* tap) {
        q4::execution::Forward& sender   = *forward_[0];
        q4::execution::Forward& receiver = *forward_[1];
        const bool one_gpu = device(0) == device(1);
        cudaEvent_t done   = nullptr; // one GPU: segment 1's last chunk
        if (one_gpu) {
            const DeviceGuard on(device(1));
            check(cudaEventCreateWithFlags(&done, cudaEventDisableTiming), "cudaEventCreate");
        }
        for (const std::array<Call*, 2>& calls : chunks) {
            const q4::execution::ForwardBatch& a = calls[0]->batch;
            const q4::execution::ForwardBatch& b = calls[1]->batch;
            Tensor from, to;
            {
                const DeviceGuard on(device(0));
                if (done != nullptr) {
                    check(cudaStreamWaitEvent(setup_.devices[0]->stream, done, 0),
                          "cudaStreamWaitEvent");
                }
                sender.begin_call(a, nullptr);
                from = sender.alloc_residual(a);
                sender.embed(a, from);
                for (std::uint32_t l = sender.segment().layers.begin; l < sender.segment().layers.end;
                     ++l) {
                    sender.layer(l, a, from, tap);
                }
            }
            {
                const DeviceGuard on(device(1));
                receiver.begin_call(b, logits);
                to = receiver.alloc_residual(b);
            }
            bulk.transfer(from.data, to.data, from.bytes(), setup_.devices[0]->stream,
                          setup_.devices[1]->stream);
            {
                const DeviceGuard on(device(0));
                setup_.devices[0]->flush();
            }
            const DeviceGuard on(device(1));
            for (std::uint32_t l = receiver.segment().layers.begin;
                 l < receiver.segment().layers.end; ++l) {
                receiver.layer(l, b, to, tap);
            }
            receiver.finish(b, to, logits);
            if (done != nullptr) {
                check(cudaEventRecord(done, setup_.devices[1]->stream), "cudaEventRecord");
            }
            setup_.devices[1]->flush();
        }
        for (const std::size_t s : {std::size_t{1}, std::size_t{0}}) {
            const DeviceGuard on(device(s));
            setup_.devices[s]->synchronize();
        }
        if (done != nullptr) { check(cudaEventDestroy(done), "cudaEventDestroy"); }
    }

    // The Gate-mode waits of segment 1's receiving gate that found the word unpublished, so far:
    // their count and total microseconds.
    [[nodiscard]] std::pair<std::uint64_t, double> gate_waits() const {
        std::uint64_t stats[2] = {0, 0};
        const DeviceGuard on(device(1));
        gate_waits_->copy_to_host(stats, sizeof(stats));
        return {stats[1], static_cast<double>(stats[0]) / 1e3};
    }

private:
    [[nodiscard]] int device(std::size_t s) const { return setup_.devices[s]->device; }

    SplitSetup setup_;
    HandoffMode mode_;
    PinnedHostBuffer error_; // the receiving gate's error word
    std::unique_ptr<HandoffChannel> channel_;
    OnDevice<DeviceBuffer> gate_waits_; // segment 1's ForwardSegment::in_wait_stats
    // Destroyed in reverse: the Forwards before the harnesses whose state and workspace they use.
    std::array<OnDevice<Harness>, 2> harness_;
    std::array<OnDevice<DeviceBuffer>, 2> words_; // each device's copy of the call's word
    std::array<OnDevice<q4::execution::Forward>, 2> forward_;
    std::uint32_t word_ = 0;
};

// Copies of one device's allocations on another, for a split across two GPUs while this test loads
// the model on one: map() moves a device pointer into the copy of its whole allocation
// (cuMemGetAddressRange; the copy is made on first use) and keeps host memory and the target's own
// memory where they are.
class DeviceMirror {
public:
    DeviceMirror(int from, int to) : from_(from), to_(to) {}

    ~DeviceMirror() {
        try {
            const DeviceGuard on(to_);
            copies_.clear();
        } catch (...) { copies_.clear(); }
    }

    DeviceMirror(const DeviceMirror&)            = delete;
    DeviceMirror& operator=(const DeviceMirror&) = delete;

    const void* map(const void* p) {
        if (p == nullptr) { return p; }
        cudaPointerAttributes attributes{};
        check(cudaPointerGetAttributes(&attributes, p), "cudaPointerGetAttributes");
        if (attributes.type != cudaMemoryTypeDevice || attributes.device == to_) { return p; }
        if (attributes.device != from_) {
            throw std::runtime_error("parameters in memory of an unexpected device");
        }
        const auto address = reinterpret_cast<std::uintptr_t>(p);
        for (const auto& copy : copies_) {
            if (address >= copy.base && address - copy.base < copy.bytes) {
                return static_cast<const std::byte*>(copy.buffer.p) + (address - copy.base);
            }
        }
        CUdeviceptr base  = 0;
        std::size_t bytes = 0;
        {
            const DeviceGuard on(from_);
            if (cuMemGetAddressRange(&base, &bytes, static_cast<CUdeviceptr>(address)) !=
                CUDA_SUCCESS) {
                throw std::runtime_error("cuMemGetAddressRange failed");
            }
        }
        Copy copy;
        copy.base  = static_cast<std::uintptr_t>(base);
        copy.bytes = bytes;
        {
            const DeviceGuard on(to_);
            copy.buffer = DeviceBuffer(bytes);
        }
        check(cudaMemcpyPeer(copy.buffer.p, to_, reinterpret_cast<const void*>(copy.base), from_,
                             bytes),
              "cudaMemcpyPeer");
        copies_.push_back(std::move(copy));
        copied_ += bytes;
        return static_cast<const std::byte*>(copies_.back().buffer.p) +
               (address - copies_.back().base);
    }

    void tensor(Tensor& t) { t.data = const_cast<void*>(map(t.data)); }

    void weight(Weight& w) {
        w.payload = map(w.payload);
        w.qdata   = map(w.qdata);
        w.qhigh   = map(w.qhigh);
        w.scales  = map(w.scales);
    }

    void linear(q4::execution::LinearParameters& p) { weight(p.weight); }

    void hc(q4::execution::HyperConnectionParameters& p) {
        tensor(p.norm);
        linear(p.down);
        linear(p.up);
    }

    void attention(q4::execution::AttentionParameters& p) {
        linear(p.projection);
        for (Tensor* t : {&p.query_norm, &p.key_norm, &p.index_query_norm, &p.index_key_norm}) {
            tensor(*t);
        }
        linear(p.output);
    }

    [[nodiscard]] std::size_t copied_bytes() const noexcept { return copied_; }

private:
    struct Copy {
        std::uintptr_t base = 0;
        std::size_t bytes   = 0;
        DeviceBuffer buffer;
    };

    int from_ = 0, to_ = 0;
    std::vector<Copy> copies_;
    std::size_t copied_ = 0;
};

// Points every weight and tensor the forward pass and the drafter read (the vision tower excepted)
// at the mirror's copies. `p` was built with the mirror's target current, so its expert scales are
// already there; the pinned host banks and the embedding are read in place.
void rebase(DeviceMirror& m, q4::execution::Parameters& p) {
    namespace ex = q4::execution;
    m.weight(p.token_embedding);
    m.tensor(p.output_head);
    if (p.output_head_q8) { m.linear(*p.output_head_q8); }
    m.hc(p.final_mixer);
    for (ex::BlockParameters& block : p.layers) {
        m.hc(block.attn_hc);
        m.hc(block.mlp_hc);
        if (auto* a = std::get_if<ex::AttentionParameters>(&block.mixer)) {
            m.attention(*a);
        } else {
            auto& g = std::get<ex::GdnParameters>(block.mixer);
            m.linear(g.projection);
            m.linear(g.control);
            for (Tensor* t : {&g.a_log, &g.dt_bias, &g.convolution, &g.norm}) { m.tensor(*t); }
            m.linear(g.output);
        }
        m.tensor(block.moe.router);
        m.tensor(block.moe.shared_score);
        m.linear(block.moe.shared_gate_up);
        m.linear(block.moe.shared_down);
        block.moe.device_scales =
            static_cast<const ops::offloaded_moe::ExpertScales*>(m.map(block.moe.device_scales));
        if (block.ple) {
            auto& e = *block.ple;
            m.linear(e.projection);
            for (Tensor* t :
                 {&e.key_norm, &e.query_norm, &e.conv_norm, &e.convolution, &e.ngram_scale}) {
                m.tensor(*t);
            }
        }
    }
    if (p.mtp) {
        auto& d = *p.mtp;
        for (Tensor* t : {&d.embedding_norm, &d.hidden_norm, &d.router, &d.shared_score}) {
            m.tensor(*t);
        }
        m.linear(d.embedding_projection);
        m.linear(d.hidden_projection);
        m.hc(d.attn_hc);
        m.hc(d.mlp_hc);
        m.hc(d.final_mixer);
        m.attention(d.attention);
        m.linear(d.shared_gate_up);
        m.linear(d.shared_down);
        m.weight(d.experts_gate_up);
        m.weight(d.experts_down);
    }
    if (p.draft_head.rows) { m.linear(*p.draft_head.rows); }
    m.tensor(p.draft_head.token_ids);
}

// A call's observable results, read back per decoder layer from the Forward that ran it.
struct Results {
    std::vector<float> logits;
    std::vector<std::vector<std::byte>> residuals; // per layer: the BF16 [S*H, T] tap
    std::vector<std::vector<std::int32_t>> routes; // per layer: the route log's k * T ids
    std::vector<std::vector<std::int32_t>> landed; // per layer: the landed log
    std::vector<std::vector<std::byte>> records;   // verify, per layer: replay records or keys
    std::vector<std::byte> ple_inputs;             // verify: the PLE convolution inputs
};

// What the unsplit Forward or one segment ran, to read its results back.
struct Owner {
    int device = 0;
    q4::LayerRange layers;
    std::uint32_t gdn_base = 0, attention_base = 0;
    const std::vector<Tensor>* taps            = nullptr; // by decoder layer
    const ExpertLogs* logs                     = nullptr; // by decoder layer
    const q4::execution::ForwardVerify* verify = nullptr; // the owner's records (its layers from 0)
};

std::vector<std::byte> read_back(int device, const void* data, std::size_t bytes) {
    std::vector<std::byte> host(bytes);
    const DeviceGuard on(device);
    check(cudaMemcpy(host.data(), data, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy read back");
    return host;
}

std::vector<std::int32_t> read_back_i32(int device, const std::int32_t* data, std::size_t count) {
    std::vector<std::int32_t> host(count);
    const DeviceGuard on(device);
    check(cudaMemcpy(host.data(), data, count * sizeof(std::int32_t), cudaMemcpyDeviceToHost),
          "cudaMemcpy read back");
    return host;
}

// The results of a synchronized call of `batch` sequences of `width` columns over `owners`.
Results collect(const q4::TextConfig& c, std::span<const Owner> owners, const DeviceBuffer& logits,
                int logits_device, std::size_t logit_values, std::int32_t width,
                std::int32_t batch) {
    Results r;
    r.logits.resize(logit_values);
    {
        const DeviceGuard on(logits_device);
        check(cudaMemcpy(r.logits.data(), logits.p, logit_values * sizeof(float),
                         cudaMemcpyDeviceToHost),
              "cudaMemcpy logits");
    }
    const std::size_t T   = static_cast<std::size_t>(width) * static_cast<std::size_t>(batch);
    const std::size_t tap = static_cast<std::size_t>(c.residual_width()) * T * 2;
    for (std::uint32_t l = 0; l < c.num_hidden_layers; ++l) {
        const auto it = std::find_if(owners.begin(), owners.end(),
                                     [&](const Owner& o) { return o.layers.contains(l); });
        if (it == owners.end()) { throw std::logic_error("a layer no Forward ran"); }
        const Owner& o = *it;
        r.residuals.push_back(read_back(o.device, o.taps->at(l).data, tap));
        r.routes.push_back(read_back_i32(
            o.device, static_cast<const std::int32_t*>(o.logs->route.p) + l * o.logs->stride,
            c.moe.top_k * T));
        r.landed.push_back(read_back_i32(o.device,
                                         static_cast<const std::int32_t*>(o.logs->landed.p) +
                                             static_cast<std::size_t>(l) * ExpertFrames::kLanding,
                                         ExpertFrames::kLanding));
        if (o.verify == nullptr) { continue; }
        const std::uint32_t compact = c.compact_layer_indices[l];
        std::vector<std::byte> bytes;
        if (c.layer_types[l] == q4::MixerKind::Gdn) {
            const auto layer =
                o.verify->gdn.layer(static_cast<std::int32_t>(compact - o.gdn_base), batch);
            for (const Tensor* t : {&layer.conv, &layer.key, &layer.value, &layer.gate}) {
                const auto part = read_back(o.device, t->data, t->bytes());
                bytes.insert(bytes.end(), part.begin(), part.end());
            }
        } else {
            const Tensor keys = o.verify->qsa_keys.slice(
                3, static_cast<std::int32_t>(compact - o.attention_base), 1);
            bytes = read_back(o.device, keys.data, keys.bytes());
        }
        r.records.push_back(std::move(bytes));
        if (l == c.ple.layer) {
            r.ple_inputs =
                read_back(o.device, o.verify->ple_inputs.data, o.verify->ple_inputs.bytes());
        }
    }
    return r;
}

template <class T>
bool same_bits(const std::vector<T>& a, const std::vector<T>& b) {
    return a.size() == b.size() &&
           (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0);
}

// Prints whether `got` equals `expected` bit for bit, naming the first differences.
bool compare_results(const Results& expected, const Results& got, const std::string& label) {
    std::vector<std::string> differ;
    if (!same_bits(expected.logits, got.logits)) { differ.push_back("logits"); }
    const auto per_layer = [&](const auto& a, const auto& b, const char* what) {
        if (a.size() != b.size()) {
            differ.push_back(std::string(what) + " (layer count)");
            return;
        }
        for (std::size_t l = 0; l < a.size(); ++l) {
            if (!same_bits(a[l], b[l])) {
                differ.push_back(std::string(what) + " of layer " + std::to_string(l));
            }
        }
    };
    per_layer(expected.residuals, got.residuals, "residual tap");
    per_layer(expected.routes, got.routes, "route log");
    per_layer(expected.landed, got.landed, "landed log");
    per_layer(expected.records, got.records, "verification records");
    if (!same_bits(expected.ple_inputs, got.ple_inputs)) { differ.push_back("PLE inputs"); }
    if (differ.empty()) {
        std::printf("%s: logits, %zu residual taps, route and landing logs%s equal\n",
                    label.c_str(), expected.residuals.size(),
                    expected.records.empty() ? "" : " and records");
        return true;
    }
    std::string first;
    for (std::size_t i = 0; i < differ.size() && i < 6; ++i) {
        first += (i > 0 ? ", " : "") + differ[i];
    }
    std::printf("FAIL %s: %zu differ from the unsplit Forward: %s%s\n", label.c_str(),
                differ.size(), first.c_str(), differ.size() > 6 ? ", ..." : "");
    return false;
}

// The Program CPU-serves the misses of every call of at most kMaxCpuColumns columns (short prompts,
// chunk remainders, forced tokens) and, with the prefill assist (design §19.3.1 P7), the thinnest
// misses of prefill calls up to 255 columns. A CPU-served expert has at most kMaxCpuColumns columns,
// so its output is the same bits on the CPU and the GPU (design §16.2), and so are the call's
// logits. Two sequences run the same prefix, then the last tokens as calls of 1, 2, 5, 8, 24 and 100
// columns, through the Program's MoE configuration (64 staging slots, the prefill overlap stream;
// no resident experts here), one of them with a CPU expert service configured as the Program's
// (decode cap 32, assist cap 256 from 9 columns) as well. With `split`, each K of `splits` then runs
// the same calls through a split pair whose segments are CPU-served through a service with one port
// per segment, in Event and in Gate mode, and they must give the GPU route's logits too.
bool check_cpu_columns(const q4::execution::Parameters& parameters, DeviceContext& device,
                       const q4::NgramVolume& volume, const std::vector<std::int32_t>& tokens,
                       std::int32_t context, KvCacheStorage kv, const SplitSetup* split = nullptr,
                       std::span<const std::uint32_t> splits = {}) {
    namespace moe               = infernix::ops::offloaded_moe;
    const auto& c               = parameters.model.config().text;
    const auto n                = static_cast<std::int32_t>(tokens.size());
    const auto vocab            = static_cast<std::int32_t>(c.vocab_size);
    const std::int32_t widths[] = {1, 2, 5, moe::kMaxCpuColumns, 24, 100};
    const std::int32_t tail     = 1 + 2 + 5 + moe::kMaxCpuColumns + 24 + 100;
    if (n <= tail + moe::kMaxCpuColumns) {
        throw std::invalid_argument("--cpu-columns needs more than 148 tokens");
    }
    const std::int32_t prefix = n - tail;
    Harness harness(parameters, device, context, prefix, 2, kv);
    std::uint64_t record_stride = 0;
    std::vector<moe::CpuMissService::Layer> layers;
    for (const auto& layer : parameters.layers) {
        record_stride = layer.moe.bank->planes.record_stride;
        layers.push_back(
            {.records       = reinterpret_cast<const std::uint8_t*>(layer.moe.bank->planes.records),
             .record_stride = record_stride,
             .scales        = layer.moe.bank->scales.data()});
    }
    constexpr std::int32_t kStagingSlots = 64;
    DeviceBuffer staging(static_cast<std::size_t>(kStagingSlots) * record_stride);
    cudaStream_t overlap = nullptr;
    std::array<cudaEvent_t, 5> events{};
    check(cudaStreamCreateWithFlags(&overlap, cudaStreamNonBlocking), "cudaStreamCreate");
    for (auto& event : events) {
        check(cudaEventCreateWithFlags(&event, cudaEventDisableTiming), "cudaEventCreate");
    }
    bool ok = true;
    constexpr std::int32_t kAssistColumns = 255;
    // The GPU route's logits of the prefix call and of each width's call.
    std::vector<float> gpu_prefix;
    std::vector<std::vector<float>> gpu_logits;
    {
        moe::CpuMissService service(layers, {.workers      = 6,
                                             .max_jobs     = 32,
                                             .max_columns  = kAssistColumns,
                                             .pcie_divisor = 3,
                                             .wide_from    = moe::kMaxCpuColumns + 1,
                                             .wide_jobs    = moe::kMaxCpuJobs,
                                             .cpus         = {}});
        auto sources           = harness.experts_view;
        sources.staging_base   = static_cast<std::uint8_t*>(staging.p);
        sources.staging_slots  = kStagingSlots;
        sources.overlap_stream = overlap;
        sources.overlap_events = events;
        const auto gpu         = harness.forward_with(parameters, device, sources, context);
        for (std::uint32_t l = 0; l < c.num_hidden_layers; ++l) {
            sources.cpu.push_back(service.channel(static_cast<int>(l)));
        }
        const auto cpu = harness.forward_with(parameters, device, sources, context);
        DeviceBuffer logits(static_cast<std::size_t>(vocab) * moe::kMaxCpuColumns * sizeof(float));
        const auto run = [&](q4::execution::Forward& forward, std::int32_t slot, std::int32_t first,
                             std::int32_t width) {
            const bool every = width <= moe::kMaxCpuColumns;
            auto call        = make_call(c, volume, {tokens}, first, width, {slot}, {slot}, every);
            Tensor out(logits.p, DType::FP32, {vocab, every ? width : 1});
            forward.run(call.batch, out);
            device.synchronize();
            return to_float(logits, static_cast<std::size_t>(vocab) * (every ? width : 1));
        };
        gpu_prefix = run(*gpu, 0, 0, prefix);
        (void)run(*cpu, 1, 0, prefix);
        if (prefix > kAssistColumns && service.served_experts() != 0) {
            std::printf("FAIL: a %d-column call reached the CPU service\n", prefix);
            ok = false;
        }
        std::int32_t first = prefix;
        for (const std::int32_t width : widths) {
            const auto before   = service.served_experts();
            const auto expected = run(*gpu, 0, first, width);
            const auto served   = run(*cpu, 1, first, width);
            const auto experts  = service.served_experts() - before;
            const bool same     = expected == served;
            std::printf("%d-column call at position %d: %llu experts CPU-served, logits %s the GPU "
                        "route's\n",
                        width, first, static_cast<unsigned long long>(experts),
                        same ? "equal" : "DIFFER FROM");
            ok = ok && same && experts > 0;
            first += width;
            gpu_logits.push_back(expected);
        }
    }
    for (auto event : events) { cudaEventDestroy(event); }
    cudaStreamDestroy(overlap);
    std::printf(ok ? "CPU-served calls equal the GPU route bit for bit\n"
                   : "FAIL: CPU-served calls differ from the GPU route\n");
    if (split == nullptr || splits.empty()) { return ok; }

    // The same calls through each split, one slot per segment. Calls of the two segments may run on
    // two devices, or on two streams of one, so each segment has its own port (its own device's).
    SplitSetup setup     = *split;
    setup.columns        = prefix;
    const int devices[2] = {setup.devices[0]->device, setup.devices[1]->device};
    std::vector<moe::CpuMissService::PortOptions> ports;
    for (const int on : devices) {
        ports.push_back({.device       = on,
                         .max_jobs     = 32,
                         .max_columns  = kAssistColumns,
                         .pcie_divisor = 3,
                         .wide_from    = moe::kMaxCpuColumns + 1,
                         .wide_jobs    = moe::kMaxCpuJobs});
    }
    moe::CpuMissService service(layers, {.workers = 6, .cpus = {}}, ports);
    OnDevice<SideStream> sides[2] = {OnDevice<SideStream>(devices[0], record_stride),
                                     OnDevice<SideStream>(devices[1], record_stride)};
    OnDevice<DeviceBuffer> split_logits(devices[1], static_cast<std::size_t>(vocab) *
                                                        moe::kMaxCpuColumns * sizeof(float));
    bool split_ok = true;
    for (const std::uint32_t k : splits) {
        for (const HandoffMode mode : {HandoffMode::Event, HandoffMode::Gate}) {
            SplitPair pair(setup, k, mode, [&](std::size_t s, const Harness& h) {
                auto x           = h.experts_view;
                x.staging_base   = static_cast<std::uint8_t*>(sides[s]->staging.p);
                x.staging_slots  = SideStream::kStagingSlots;
                x.overlap_stream = sides[s]->stream;
                x.overlap_events = sides[s]->events;
                // Local entries; each channel names its decoder layer for the service.
                for (std::uint32_t l = h.layers.begin; l < h.layers.end; ++l) {
                    x.cpu.push_back(service.channel(static_cast<int>(l), static_cast<int>(s)));
                }
                return x;
            });
            const auto split_run = [&](std::int32_t first, std::int32_t width) {
                const bool every                    = width <= moe::kMaxCpuColumns;
                std::array<OnDevice<Call>, 2> calls = {
                    call_on(devices[0], c, volume, tokens, first, width, every),
                    call_on(devices[1], c, volume, tokens, first, width, every)};
                Tensor out(split_logits->p, DType::FP32, {vocab, every ? width : 1});
                pair.run({calls[0].get(), calls[1].get()}, &out, nullptr, false);
                std::vector<float> host(static_cast<std::size_t>(vocab) * (every ? width : 1));
                const DeviceGuard on(devices[1]);
                split_logits->copy_to_host(host.data(), host.size() * sizeof(float));
                return host;
            };
            bool same             = split_run(0, prefix) == gpu_prefix;
            std::int32_t first    = prefix;
            std::uint64_t experts = 0;
            for (std::size_t i = 0; i < std::size(widths); ++i) {
                const auto before = service.served_experts();
                same              = split_run(first, widths[i]) == gpu_logits[i] && same;
                const auto served = service.served_experts() - before;
                experts += served;
                same = same && served > 0;
                first += widths[i];
            }
            std::printf("K=%u %s: split CPU-served calls (%llu experts CPU-served) %s the GPU "
                        "route's logits\n",
                        k, mode_name(mode), static_cast<unsigned long long>(experts),
                        same ? "equal" : "DIFFER FROM");
            split_ok = split_ok && same;
        }
    }
    std::printf(split_ok ? "split CPU-served calls equal the GPU route bit for bit\n"
                         : "FAIL: split CPU-served calls differ from the GPU route\n");
    return ok && split_ok;
}

// Layer split (--split): the unsplit Forward, then a split pair for each K of `splits` and each
// handoff mode, run a lane's prefill chunk of min(4096, n - 5) tokens, its next token as a decode
// step and the four after as a verification call (teacher-forced drafts, replay records). The
// logits, every residual tap, the route and landing logs and the records must be equal bit for bit.
// Both use the same kinds of expert sources (ExpertFrames, staging slots, the overlap stream).
// Gate mode launches the decode step's receiving gate first; its waits are reported. Then, per K,
// the prefill again as four chunks from fresh state, enqueued without a host wait and crossing
// through a BulkHandoff (SplitPair::run_bulk): the last chunk's logits, residual taps and route
// log must equal the unsplit Forward's chunked prefill.
bool check_split(const q4::execution::Parameters& parameters, const q4::NgramVolume& volume,
                 const std::vector<std::int32_t>& tokens, SplitSetup setup,
                 std::span<const std::uint32_t> splits) {
    const auto& c                  = parameters.model.config().text;
    const std::uint32_t L          = c.num_hidden_layers;
    const auto n                   = static_cast<std::int32_t>(tokens.size());
    constexpr std::int32_t kChunk  = 4096;
    constexpr std::int32_t kVerify = 4;
    if (n < kVerify + 2) { throw std::invalid_argument("--split needs at least 6 tokens"); }
    const std::int32_t P = std::min(kChunk, n - kVerify - 1);
    if (P < kChunk) { std::printf("note: %d tokens, so the prefill chunk has %d columns\n", n, P); }
    setup.columns = P;

    struct Case {
        const char* name;
        std::int32_t first, width;
        bool verify, receiver_first;
    };

    const Case cases[]          = {{"prefill", 0, P, false, false},
                                   {"decode", P, 1, false, true},
                                   {"verify", P + 1, kVerify, true, false}};
    DeviceContext& device       = *setup.devices[0];
    const int devices[2]        = {setup.devices[0]->device, setup.devices[1]->device};
    const bool one_device       = devices[0] == devices[1];
    const auto W                = static_cast<std::int32_t>(c.residual_width());
    const auto vocab            = static_cast<std::int32_t>(c.vocab_size);
    const std::size_t layer_tap = static_cast<std::size_t>(W) * static_cast<std::size_t>(P) * 2;
    const auto record_stride    = parameters.layers.front().moe.bank->planes.record_stride;

    // Expert frames on each device; route and landing logs for the reference and per device for the
    // split; a side stream per segment (the reference uses segment 0's).
    OnDevice<ExpertFrames> frames0(devices[0], parameters);
    OnDevice<ExpertFrames> frames1   = one_device
                                           ? OnDevice<ExpertFrames>()
                                           : OnDevice<ExpertFrames>(devices[1], *setup.parameters[1]);
    const ExpertFrames* frames_of[2] = {frames0.get(), one_device ? frames0.get() : frames1.get()};
    OnDevice<ExpertLogs> reference_logs(devices[0], c, P), logs0(devices[0], c, P);
    OnDevice<ExpertLogs> logs1 =
        one_device ? OnDevice<ExpertLogs>() : OnDevice<ExpertLogs>(devices[1], c, P);
    ExpertLogs* logs_of[2]        = {logs0.get(), one_device ? logs0.get() : logs1.get()};
    OnDevice<SideStream> sides[2] = {OnDevice<SideStream>(devices[0], record_stride),
                                     OnDevice<SideStream>(devices[1], record_stride)};
    // Residual taps by layer (the reference's, then segment 0's and on one device segment 1's) and
    // the logits.
    OnDevice<DeviceBuffer> taps0(devices[0], static_cast<std::size_t>(L) * layer_tap);
    OnDevice<DeviceBuffer> logits0(devices[0],
                                   static_cast<std::size_t>(vocab) * kVerify * sizeof(float));
    OnDevice<DeviceBuffer> logits1 =
        one_device ? OnDevice<DeviceBuffer>()
                   : OnDevice<DeviceBuffer>(devices[1], static_cast<std::size_t>(vocab) * kVerify *
                                                            sizeof(float));
    const DeviceBuffer& split_logits = one_device ? *logits0 : *logits1;
    std::vector<std::byte*> reference_taps;
    for (std::uint32_t l = 0; l < L; ++l) {
        reference_taps.push_back(static_cast<std::byte*>(taps0->p) + l * layer_tap);
    }
    const auto tap_tensors = [&](const std::vector<std::byte*>& bases, std::int32_t width) {
        std::vector<Tensor> taps;
        for (std::byte* base : bases) {
            taps.emplace_back(base, DType::BF16, std::initializer_list<std::int32_t>{W, width});
        }
        return taps;
    };

    std::vector<Results> expected;
    {
        const DeviceGuard on(devices[0]);
        Harness reference(parameters, device, setup.context, P, 1, setup.kv);
        const auto forward = reference.forward_with(
            parameters, device,
            expert_sources(*frames0, *reference_logs, *sides[0], reference.layers), setup.context);
        const q4::execution::ForwardVerify verify = reference.verify(kVerify, 1);
        for (const Case& kase : cases) {
            Call call =
                make_call(c, volume, {tokens}, kase.first, kase.width, {0}, {0}, kase.verify);
            if (kase.verify) { call.batch.verify = &verify; }
            auto taps = tap_tensors(reference_taps, kase.width);
            q4::execution::ForwardTap tap{.residuals = &taps};
            reference_logs->clear();
            const std::int32_t columns = kase.verify ? kase.width : 1;
            Tensor out(logits0->p, DType::FP32, {vocab, columns});
            const auto t0 = std::chrono::steady_clock::now();
            forward->run(call.batch, out, &tap);
            device.synchronize();
            std::printf("unsplit %s of %d columns: %.1f ms\n", kase.name, kase.width,
                        seconds_since(t0) * 1e3);
            const Owner owner{.device = devices[0],
                              .layers = reference.layers,
                              .taps   = &taps,
                              .logs   = reference_logs.get(),
                              .verify = kase.verify ? &verify : nullptr};
            expected.push_back(collect(c, std::span<const Owner>(&owner, 1), *logits0, devices[0],
                                       static_cast<std::size_t>(vocab) * columns, kase.width, 1));
        }
    }

    // The prefill again from fresh state as kBulkChunks chunks: the results of the last chunk,
    // which reads the KV and state every earlier chunk wrote.
    constexpr std::int32_t kBulkChunks = 4;
    const std::int32_t bulk_width      = P / kBulkChunks;
    const auto bulk_calls              = [&](int device) {
        std::vector<OnDevice<Call>> calls;
        for (std::int32_t i = 0; i < kBulkChunks; ++i) {
            calls.push_back(call_on(device, c, volume, tokens, i * bulk_width, bulk_width, false));
        }
        return calls;
    };
    Results bulk_expected;
    if (bulk_width > 0) {
        const auto calls = bulk_calls(devices[0]); // all staged before the first chunk runs
        const DeviceGuard on(devices[0]);
        Harness reference(parameters, device, setup.context, P, 1, setup.kv);
        const auto forward = reference.forward_with(
            parameters, device,
            expert_sources(*frames0, *reference_logs, *sides[0], reference.layers), setup.context);
        auto taps = tap_tensors(reference_taps, bulk_width);
        q4::execution::ForwardTap tap{.residuals = &taps};
        reference_logs->clear();
        Tensor out(logits0->p, DType::FP32, {vocab, 1});
        for (const auto& call : calls) { forward->run(call->batch, out, &tap); }
        device.synchronize();
        const Owner owner{.device = devices[0], .layers = reference.layers, .taps = &taps,
                          .logs = reference_logs.get()};
        bulk_expected = collect(c, std::span<const Owner>(&owner, 1), *logits0, devices[0],
                                static_cast<std::size_t>(vocab), bulk_width, 1);
    }

    bool ok = true;
    for (const std::uint32_t k : splits) {
        // Segment 1's taps: beside segment 0's on one device, else on its own.
        const OnDevice<DeviceBuffer> taps1 =
            one_device ? OnDevice<DeviceBuffer>()
                       : OnDevice<DeviceBuffer>(devices[1], (L - k) * layer_tap);
        std::vector<std::byte*> split_taps = reference_taps;
        if (!one_device) {
            for (std::uint32_t l = k; l < L; ++l) {
                split_taps[l] = static_cast<std::byte*>(taps1->p) + (l - k) * layer_tap;
            }
        }
        for (const HandoffMode mode : {HandoffMode::Event, HandoffMode::Gate}) {
            SplitPair pair(setup, k, mode, [&](std::size_t s, const Harness& h) {
                return expert_sources(*frames_of[s], *logs_of[s], *sides[s], h.layers);
            });
            std::array<q4::execution::ForwardVerify, 2> verify;
            for (std::size_t s = 0; s < 2; ++s) {
                const DeviceGuard on(devices[s]);
                verify[s] = pair.harness(s).verify(kVerify, 1);
            }
            for (std::size_t i = 0; i < std::size(cases); ++i) {
                const Case& kase                    = cases[i];
                std::array<OnDevice<Call>, 2> calls = {
                    call_on(devices[0], c, volume, tokens, kase.first, kase.width, kase.verify),
                    call_on(devices[1], c, volume, tokens, kase.first, kase.width, kase.verify)};
                for (std::size_t s = 0; s < 2; ++s) {
                    if (kase.verify) { calls[s]->batch.verify = &verify[s]; }
                    const DeviceGuard on(devices[s]);
                    logs_of[s]->clear();
                }
                auto taps = tap_tensors(split_taps, kase.width);
                q4::execution::ForwardTap tap{.residuals = &taps};
                const std::int32_t columns = kase.verify ? kase.width : 1;
                Tensor out(split_logits.p, DType::FP32, {vocab, columns});
                pair.run({calls[0].get(), calls[1].get()}, &out, &tap,
                         mode == HandoffMode::Gate && kase.receiver_first);
                Owner owners[2];
                for (std::size_t s = 0; s < 2; ++s) {
                    const Harness& h = pair.harness(s);
                    owners[s]        = Owner{.device         = devices[s],
                                             .layers         = h.layers,
                                             .gdn_base       = h.gdn_base,
                                             .attention_base = h.attention_base,
                                             .taps           = &taps,
                                             .logs           = logs_of[s],
                                             .verify         = kase.verify ? &verify[s] : nullptr};
                }
                const Results got =
                    collect(c, owners, split_logits, devices[1],
                            static_cast<std::size_t>(vocab) * columns, kase.width, 1);
                const std::string label = "K=" + std::to_string(k) + " " + mode_name(mode) + " " +
                                          kase.name + " (" + std::to_string(kase.width) +
                                          " columns)";
                ok = compare_results(expected[i], got, label) && ok;
                if (mode == HandoffMode::Gate) {
                    const auto [waits, us] = pair.gate_waits();
                    std::printf("  gate waits so far: %llu, %.0f us\n",
                                static_cast<unsigned long long>(waits), us);
                }
            }
        }
        if (bulk_width == 0) { continue; }
        // The chunked prefill through a BulkHandoff whose 2-piece ring is refilled within and
        // across chunks (each chunk's residual is three pieces).
        SplitPair pair(setup, k, HandoffMode::Event, [&](std::size_t s, const Harness& h) {
            return expert_sources(*frames_of[s], *logs_of[s], *sides[s], h.layers);
        });
        const std::size_t residual_bytes = static_cast<std::size_t>(W) * bulk_width * 2;
        BulkHandoff bulk(devices[0], devices[1], (residual_bytes / 3 + 255) / 256 * 256, 2);
        const std::array<std::vector<OnDevice<Call>>, 2> calls = {bulk_calls(devices[0]),
                                                                  bulk_calls(devices[1])};
        std::vector<std::array<Call*, 2>> chunks;
        for (std::size_t i = 0; i < calls[0].size(); ++i) {
            chunks.push_back({calls[0][i].get(), calls[1][i].get()});
        }
        for (std::size_t s = 0; s < 2; ++s) {
            const DeviceGuard on(devices[s]);
            logs_of[s]->clear();
        }
        auto taps = tap_tensors(split_taps, bulk_width);
        q4::execution::ForwardTap tap{.residuals = &taps};
        Tensor out(split_logits.p, DType::FP32, {vocab, 1});
        const auto t0 = std::chrono::steady_clock::now();
        pair.run_bulk(chunks, bulk, &out, &tap);
        std::printf("K=%u BulkHandoff prefill of %d chunks: %.1f ms\n", k, kBulkChunks,
                    seconds_since(t0) * 1e3);
        Owner owners[2];
        for (std::size_t s = 0; s < 2; ++s) {
            const Harness& h = pair.harness(s);
            owners[s]        = Owner{.device         = devices[s],
                                     .layers         = h.layers,
                                     .gdn_base       = h.gdn_base,
                                     .attention_base = h.attention_base,
                                     .taps           = &taps,
                                     .logs           = logs_of[s]};
        }
        const Results got = collect(c, owners, split_logits, devices[1],
                                    static_cast<std::size_t>(vocab), bulk_width, 1);
        ok = compare_results(bulk_expected, got,
                             "K=" + std::to_string(k) + " BulkHandoff prefill, last of " +
                                 std::to_string(kBulkChunks) + " chunks (" +
                                 std::to_string(bulk_width) + " columns)") &&
             ok;
    }
    std::printf(ok ? "split Forwards equal the unsplit Forward bit for bit\n"
                   : "FAIL: split Forwards differ from the unsplit Forward\n");
    return ok;
}

} // namespace

int main(int argc, char** argv) {
    const char* artifact_env = std::getenv("INFERNIX_QWEN4_ARTIFACT");
    if (artifact_env == nullptr || argc < 2) {
        std::printf("SKIP: set INFERNIX_QWEN4_ARTIFACT and pass a token list\n");
        return 77;
    }
    // Each token list takes the options that follow it.
    struct Job {
        std::vector<std::int32_t> tokens;
        std::string logits_path, residuals_path, routes_path, blocks_path, dump_path;
        std::int32_t chunk     = 256;
        std::int32_t dump_from = 0;
        KvCacheStorage kv      = KvCacheStorage::BFloat16;
        bool cpu_columns   = false;
        std::vector<std::uint32_t> splits; // --split: the K of each layer split to check
        int split_device = 0;              // --split-device: segment 1's device
    };
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    try {
        std::vector<Job> jobs;
        models::LoadOptions load; // --rope-yarn-factor F (anywhere): the model's YaRN factor
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--rope-yarn-factor") {
                if (i + 1 >= argc) { throw std::invalid_argument("--rope-yarn-factor needs a value"); }
                load.rope_yarn_factor = std::stof(argv[++i]);
                continue;
            }
            if (arg.rfind("--", 0) != 0) {
                jobs.push_back(Job{parse_tokens(arg)});
                if (jobs.back().tokens.size() < 2) { throw std::invalid_argument("pass at least two tokens"); }
                continue;
            }
            if (arg == "--cpu-columns") {
                if (jobs.empty()) { throw std::invalid_argument("--cpu-columns needs a token list before it"); }
                jobs.back().cpu_columns = true;
                continue;
            }
            if (jobs.empty() || i + 1 >= argc) { throw std::invalid_argument(arg + " needs a token list before it and a value"); }
            const std::string value = argv[++i];
            Job& job = jobs.back();
            if (arg == "--logits") {
                job.logits_path = value;
            } else if (arg == "--residuals") {
                job.residuals_path = value;
            } else if (arg == "--routes") {
                job.routes_path = value;
            } else if (arg == "--blocks") {
                job.blocks_path = value;
            } else if (arg == "--dump-logits") {
                job.dump_path = value;
            } else if (arg == "--chunk") {
                job.chunk = std::stoi(value);
                if (job.chunk <= 0) { throw std::invalid_argument("--chunk must be positive"); }
            } else if (arg == "--dump-from") {
                job.dump_from = std::stoi(value);
                if (job.dump_from < 0) { throw std::invalid_argument("--dump-from must not be negative"); }
            } else if (arg == "--kv") {
                job.kv = infernix::test::parse_kv_cache_storage(value);
            } else if (arg == "--split") {
                job.splits = parse_splits(value);
            } else if (arg == "--split-device") {
                job.split_device = std::stoi(value);
                if (job.split_device < 0) { throw std::invalid_argument("--split-device must not be negative"); }
            } else {
                throw std::invalid_argument("unknown option " + arg);
            }
        }
        const bool scoring = !jobs.front().dump_path.empty();
        for (const Job& job : jobs) {
            if (job.dump_path.empty() != !scoring || (jobs.size() > 1 && job.dump_path.empty())) {
                throw std::invalid_argument("several token lists are scored only, each with its own --dump-logits");
            }
            if (scoring && !job.splits.empty()) { throw std::invalid_argument("--split does not score"); }
        }
        print_memory("start");
        DeviceContext device(0);
        print_memory("device context");
        auto t0 = std::chrono::steady_clock::now();
        artifact::Reader reader(artifact_env);
        if (!q4::is_qwen4_exp(reader)) { throw std::invalid_argument("artifact is not Qwen4Exp"); }
        auto plan  = q4::plan_load(reader, load);
        print_memory("load plan");
        auto model = q4::materialize_model(std::move(plan), device);
        device.synchronize();
        const auto& stats = model->storage_stats();
        std::printf("loaded in %.1f s: %.2f GiB device, %.2f GiB pinned host\n", seconds_since(t0),
                    stats.device_capacity_bytes / 1073741824.0, stats.pinned_bytes / 1073741824.0);
        print_memory("materialized");
        const q4::execution::Parameters parameters(*model);
        print_memory("parameters");
        const auto& c = model->config().text;
        const char* ngram_env = std::getenv("INFERNIX_QWEN4_NGRAM");
        const q4::NgramVolume volume(ngram_env != nullptr ? std::filesystem::path(ngram_env)
                                                          : q4::default_ngram_volume(artifact_env),
                                     c.ple.table);
        const std::size_t vocab = c.vocab_size;
        if (scoring) {
            // Teacher-forced scoring: chunked prefill of each whole text in slot 0 of its own harness.
            for (const Job& job : jobs) {
                const auto& tokens          = job.tokens;
                const auto n                = static_cast<std::int32_t>(tokens.size());
                const std::int32_t context  = std::max<std::int32_t>(256, (n + 63) / 64 * 64);
                const std::int32_t chunk    = std::min(job.chunk, n);
                Harness harness(parameters, device, context, chunk, 1, job.kv);
                DeviceBuffer logits(vocab * chunk * sizeof(float));
                std::ofstream out(job.dump_path, std::ios::binary);
                const std::int32_t from      = std::min(job.dump_from, n);
                const std::int32_t header[2] = {static_cast<std::int32_t>(vocab), n - from};
                out.write(reinterpret_cast<const char*>(header), sizeof(header));
                double nll = 0.0;
                std::int32_t scored = 0, same_top1 = 0;
                t0 = std::chrono::steady_clock::now();
                for (std::int32_t first = 0; first < n; first += chunk) {
                    const std::int32_t width = std::min(chunk, n - first);
                    auto call = make_call(c, volume, {tokens}, first, width, {0}, {0}, true);
                    Tensor chunk_logits(logits.p, DType::FP32, {static_cast<std::int32_t>(vocab), width});
                    harness.forward->run(call.batch, chunk_logits);
                    device.synchronize();
                    if (first + width <= from) { continue; }
                    const auto rows       = to_float(logits, vocab * width);
                    const std::int32_t lo = std::max(0, from - first);
                    out.write(reinterpret_cast<const char*>(rows.data() + static_cast<std::size_t>(lo) * vocab),
                              static_cast<std::streamsize>((rows.size() - static_cast<std::size_t>(lo) * vocab) *
                                                           sizeof(float)));
                    for (std::int32_t i = lo; i < width && first + i + 1 < n; ++i) {
                        const float* row = rows.data() + static_cast<std::size_t>(i) * vocab;
                        const float peak = *std::max_element(row, row + vocab);
                        double sum = 0.0;
                        for (std::size_t v = 0; v < vocab; ++v) { sum += std::exp(double(row[v]) - peak); }
                        const std::int32_t next = tokens[first + i + 1];
                        nll -= double(row[next]) - peak - std::log(sum);
                        same_top1 += std::max_element(row, row + vocab) - row == next;
                        ++scored;
                    }
                }
                if (!out) { throw std::runtime_error("cannot write " + job.dump_path); }
                std::printf("%s: scored %d positions in %.1f s: mean NLL %.5f nats, perplexity %.4f, top-1 = next %.1f%%\n",
                            job.dump_path.c_str(), scored, seconds_since(t0), nll / scored, std::exp(nll / scored),
                            100.0 * same_top1 / scored);
            }
            return 0;
        }
        const Job& job = jobs.front();
        const auto& tokens = job.tokens;
        const auto n = static_cast<std::int32_t>(tokens.size());
        const std::int32_t context = std::max<std::int32_t>(256, (n + 63) / 64 * 64);
        const std::string& logits_path    = job.logits_path;
        const std::string& residuals_path = job.residuals_path;
        const std::string& routes_path    = job.routes_path;
        const std::string& blocks_path    = job.blocks_path;
        if (!job.splits.empty()) {
            for (const std::uint32_t k : job.splits) {
                if (k == 0 || k >= c.num_hidden_layers) {
                    throw std::invalid_argument("--split K needs 1 <= K < " +
                                                std::to_string(c.num_hidden_layers));
                }
            }
            const int other = job.split_device;
            // Segment 1 runs on a second context: of its GPU, or of device 0 in the same-device mode.
            const OnDevice<DeviceContext> second(other, other);
            // On another GPU segment 1 reads a copy of the model's device weights: Parameters built
            // there (their expert scales land there), pointed at the copies.
            DeviceMirror mirror(device.device, other);
            OnDevice<q4::execution::Parameters> mirrored;
            if (other != device.device) {
                mirrored = OnDevice<q4::execution::Parameters>(other, *model);
                rebase(mirror, *mirrored);
                std::printf("segment 1 on device %d reads a %.2f GiB copy of the device weights\n",
                            other, mirror.copied_bytes() / 1073741824.0);
                const int a = numerics_multiprocessor_count(device.device),
                          b = numerics_multiprocessor_count(other);
                if (a != b) {
                    std::printf("note: the devices compute with %d and %d SMs; "
                                "INFERNIX_NUMERICS_SMS=%d makes their results bit-identical\n",
                                a, b, std::min(a, b));
                }
            }
            SplitSetup setup;
            setup.parameters = {&parameters, other != device.device ? mirrored.get() : &parameters};
            setup.devices    = {&device, second.get()};
            setup.context    = context;
            setup.kv         = job.kv;
            bool ok          = check_split(parameters, volume, tokens, setup, job.splits);
            if (job.cpu_columns) {
                ok = check_cpu_columns(parameters, device, volume, tokens, context, job.kv, &setup,
                                       job.splits) &&
                     ok;
            }
            return ok ? 0 : 1;
        }
        if (job.cpu_columns) {
            return check_cpu_columns(parameters, device, volume, tokens, context, job.kv) ? 0 : 1;
        }
        Harness harness(parameters, device, context, n, 3, job.kv);
        DeviceBuffer logits(vocab * 2 * sizeof(float));
        print_memory("harness");

        // 1. All tokens as one prefill chunk in slot/row 0: logits of the last position, and
        //    optionally every block's residual and routed experts.
        auto full = make_call(c, volume, {tokens}, 0, n, {0}, {0});
        Tensor full_logits(logits.p, DType::FP32, {static_cast<std::int32_t>(vocab), 1});
        const auto blocks = static_cast<std::size_t>(c.num_hidden_layers);
        const auto W = static_cast<std::int32_t>(c.residual_width());
        const auto K = static_cast<std::int32_t>(c.moe.top_k);
        const std::size_t residual_bytes = static_cast<std::size_t>(W) * n * 2;
        const std::size_t route_bytes    = static_cast<std::size_t>(K) * n * 4;
        const auto H = static_cast<std::int32_t>(c.hidden_size);
        const std::size_t io_bytes = static_cast<std::size_t>(H) * n * 2;
        DeviceBuffer taps(blocks * (residual_bytes + route_bytes + 4 * io_bytes));
        std::vector<Tensor> residual_taps, route_taps, io_taps[4];
        auto* base = static_cast<std::byte*>(taps.p);
        const std::size_t io_offset = blocks * (residual_bytes + route_bytes);
        for (std::size_t b = 0; b < blocks; ++b) {
            residual_taps.emplace_back(base + b * residual_bytes, DType::BF16, std::initializer_list<std::int32_t>{W, n});
            route_taps.emplace_back(base + blocks * residual_bytes + b * route_bytes, DType::I32,
                                    std::initializer_list<std::int32_t>{K, n});
            for (std::size_t kind = 0; kind < 4; ++kind) {
                io_taps[kind].emplace_back(base + io_offset + (b * 4 + kind) * io_bytes, DType::BF16,
                                           std::initializer_list<std::int32_t>{H, n});
            }
        }
        q4::execution::ForwardTap tap{&residual_taps, &route_taps, &io_taps[0], &io_taps[1], &io_taps[2], &io_taps[3]};
        t0 = std::chrono::steady_clock::now();
        harness.forward->run(full.batch, full_logits, &tap);
        device.synchronize();
        const auto dump = [&](const std::string& path, std::size_t offset, std::size_t bytes) {
            if (path.empty()) { return; }
            std::vector<std::byte> host(bytes);
            taps.copy_to_host(host.data(), bytes, offset);
            std::ofstream file(path, std::ios::binary);
            file.write(reinterpret_cast<const char*>(host.data()), static_cast<std::streamsize>(bytes));
        };
        dump(residuals_path, 0, blocks * residual_bytes);
        dump(routes_path, blocks * residual_bytes, blocks * route_bytes);
        dump(blocks_path, io_offset, blocks * 4 * io_bytes); // [blocks][mixer in, mixer out, moe in, moe out][T][H]
        std::printf("prefill %d tokens: %.1f ms\n", n, seconds_since(t0) * 1e3);
        print_memory("prefill");
        const auto prefill = to_float(logits, vocab);
        const auto best = top(prefill, 0, vocab, 5);
        std::printf("prefill top-5 at position %d:", n - 1);
        for (int id : best) { std::printf(" %d (%.3f)", id, prefill[id]); }
        std::printf("\n");
        if (!logits_path.empty()) {
            std::ofstream file(logits_path, std::ios::binary);
            file.write(reinterpret_cast<const char*>(prefill.data()), static_cast<std::streamsize>(vocab * sizeof(float)));
        }

        // 2. All but the last token in slots/rows 1 and 2, then one batched decode step of both.
        for (std::int32_t slot : {1, 2}) {
            auto head = make_call(c, volume, {tokens}, 0, n - 1, {slot}, {slot});
            Tensor head_logits(logits.p, DType::FP32, {static_cast<std::int32_t>(vocab), 1});
            harness.forward->run(head.batch, head_logits);
        }
        auto step = make_call(c, volume, {tokens, tokens}, n - 1, 1, {1, 2}, {1, 2});
        Tensor step_logits(logits.p, DType::FP32, {static_cast<std::int32_t>(vocab), 2});
        t0 = std::chrono::steady_clock::now();
        harness.forward->run(step.batch, step_logits);
        device.synchronize();
        std::printf("batched decode of 2 sequences: %.1f ms\n", seconds_since(t0) * 1e3);
        const auto decode = to_float(logits, 2 * vocab);
        std::size_t rows_differ = 0;
        double max_diff = 0;
        for (std::size_t i = 0; i < vocab; ++i) {
            rows_differ += decode[i] != decode[vocab + i];
            max_diff = std::max(max_diff, std::abs(double(decode[i]) - prefill[i]));
        }
        const auto decode_best = top(decode, 0, vocab, 5);
        std::printf("decode top-5:");
        for (int id : decode_best) { std::printf(" %d (%.3f)", id, decode[id]); }
        std::printf("\ndecode rows differing between identical sequences: %zu; max |decode - prefill| = %.4f\n",
                    rows_differ, max_diff);
        const bool ok = rows_differ == 0 && decode_best.front() == best.front();
        std::printf(ok ? "forward checks passed\n" : "FAIL: decode disagrees with prefill\n");
        return ok ? 0 : 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
