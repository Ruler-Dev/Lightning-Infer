// Public-Op benchmark of hyper_connection_mix at Qwen3.8-Flash-Next's geometry (S = 4 streams,
// H = 2560, rank 320, Q8_G32_FP16 RowSplit weights as in the Dense8 artifact): W_down [324, 10240]
// with the four injection rows (the 96 block mixers of a forward) or [320, 10240] without them (the
// final mixer), W_up [10240, 320] padded to 384. T <= 16 runs the fused two-kernel route, wider
// calls the composed one.
//
// Each point is one graph-captured call (or --graph-calls calls in one graph, reported per call).
// Cache modes: cold evicts L2 with a 256 MiB read before every sample; warm repeats the graph
// with the weights left in L2, which is the decode case (the previous layer's cpu_wait_kernel
// prefetches the next mixer's weights into L2). --dump writes every call's outputs (x, then the
// injects) in the order of the sweep, so two builds can be compared byte for byte.
//
//   infernix_hyper_connection_mix_bench [--tokens 1,2,...] [--variant inject|final|both]
//       [--cache cold|warm|both] [--graph-calls 1..64] [--warmup N] [--repeat N]
//       [--dump PATH] [--profile]

#include "infernix/ops/hyper_connection.h"

#include "core/device.h"
#include "infernix_bench_common.h"
#include "quantized_weight.cuh"

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace infernix;

namespace {

constexpr int kStreams = 4, kHidden = 2560, kRank = 320, kWidth = kStreams * kHidden;
constexpr int kUpPaddedK          = 384; // the RowSplit K padding to a multiple of 128
constexpr float kEps              = 1.0e-6F;
constexpr std::size_t kFlushBytes = 256ULL << 20;

struct Options {
    std::vector<int> tokens{1, 2, 3, 4, 5, 6, 8, 12, 16};
    std::string variant = "both", cache = "both", dump;
    int graph_calls = 1, warmup = 10, repeat = 100;
    bool profile = false;
};

int integer(const std::string& text, int lower, int upper) {
    std::size_t end = 0;
    const int value = std::stoi(text, &end);
    if (end != text.size() || value < lower || value > upper) {
        throw std::invalid_argument("integer out of range: " + text);
    }
    return value;
}

std::vector<int> list(const std::string& text) {
    std::vector<int> result;
    std::size_t start = 0;
    while (start < text.size()) {
        const auto end = text.find(',', start);
        result.push_back(integer(text.substr(start, end - start), 1, 4096));
        if (end == std::string::npos) { break; }
        start = end + 1;
    }
    if (result.empty()) { throw std::invalid_argument("empty --tokens list"); }
    return result;
}

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--profile") {
            o.profile = true;
            continue;
        }
        if (arg == "--help" || arg == "-h") {
            std::puts("usage: infernix_hyper_connection_mix_bench [--tokens T,...] [--variant "
                      "inject|final|both] "
                      "[--cache cold|warm|both] [--graph-calls 1..64] [--warmup N] [--repeat N] "
                      "[--dump PATH] "
                      "[--profile]");
            std::exit(0);
        }
        if (++i >= argc) { throw std::invalid_argument("missing value of " + arg); }
        const std::string value = argv[i];
        if (arg == "--tokens") {
            o.tokens = list(value);
        } else if (arg == "--variant") {
            o.variant = value;
        } else if (arg == "--cache") {
            o.cache = value;
        } else if (arg == "--graph-calls") {
            o.graph_calls = integer(value, 1, 64);
        } else if (arg == "--warmup") {
            o.warmup = integer(value, 0, 100000);
        } else if (arg == "--repeat") {
            o.repeat = integer(value, 1, 100000);
        } else if (arg == "--dump") {
            o.dump = value;
        } else {
            throw std::invalid_argument("unknown option: " + arg);
        }
    }
    if ((o.variant != "inject" && o.variant != "final" && o.variant != "both") ||
        (o.cache != "cold" && o.cache != "warm" && o.cache != "both")) {
        throw std::invalid_argument("--variant takes inject|final|both and --cache cold|warm|both");
    }
    if (o.profile && (o.tokens.size() != 1 || o.variant == "both" || o.cache == "both")) {
        throw std::invalid_argument("--profile needs one T, one variant and one cache mode");
    }
    return o;
}

struct Mixer {
    bool inject;
    bench::PackedQuantizedWeight down, up;
};

} // namespace

int main(int argc, char** argv) {
    try {
        const Options o = parse(argc, argv);
        DeviceContext device;
        const cudaStream_t stream = device.stream;
        std::printf("# gpu=%s SMs=%d cuda_runtime=%d geometry S=%d H=%d rank=%d q8_g32_fp16 "
                    "graph_calls=%d\n",
                    device.props.name, device.multiprocessor_count(), CUDART_VERSION, kStreams,
                    kHidden, kRank, o.graph_calls);
        const int max_t = *std::max_element(o.tokens.begin(), o.tokens.end());
        bench::L2FlushBuffer flush(kFlushBytes);
        // Represented operands: residual in [-2, 2] (a few units like the model's streams), norm
        // weight offsets in [-0.25, 0.25].
        DeviceBuffer residual =
            bench::make_bf16(static_cast<std::size_t>(kWidth) * max_t, 701U, -2.0F, 2.0F);
        DeviceBuffer norm = bench::make_bf16(kWidth, 702U, -0.25F, 0.25F);
        DeviceBuffer x_out(static_cast<std::size_t>(kHidden) * max_t * 2);
        DeviceBuffer inject_out(static_cast<std::size_t>(kStreams) * max_t * sizeof(float));
        std::vector<Mixer> mixers;
        for (const bool inject : {true, false}) {
            if ((inject && o.variant == "final") || (!inject && o.variant == "inject")) {
                continue;
            }
            mixers.push_back(
                {inject,
                 bench::make_row_split_weight(QType::Q8_G32_FP16, kRank + (inject ? kStreams : 0),
                                              kWidth, kWidth, inject ? 703U : 704U, 0.02F),
                 bench::make_row_split_weight(QType::Q8_G32_FP16, kWidth, kRank, kUpPaddedK,
                                              inject ? 705U : 706U, 0.05F)});
        }
        std::size_t capacity = 0;
        for (const Mixer& m : mixers) {
            capacity = std::max(capacity, ops::hyper_connection_mix_workspace_capacity_bytes(
                                              m.down.weight, m.up.weight,
                                              ops::LinearPolicy::A16Only, kStreams, kRank, max_t));
        }
        WorkspaceArena workspace(capacity);
        std::ofstream dump;
        if (!o.dump.empty()) {
            dump.open(o.dump, std::ios::binary);
            if (!dump) { throw std::runtime_error("cannot open " + o.dump); }
        }
        for (const Mixer& m : mixers) {
            for (const int t : o.tokens) {
                const Tensor r(residual.p, DType::BF16, {kWidth, t});
                const Tensor w(norm.p, DType::BF16, {kWidth});
                Tensor x(x_out.p, DType::BF16, {kHidden, t});
                Tensor inj(inject_out.p, DType::FP32, {kStreams, t});
                const auto call = [&](cudaStream_t s) {
                    ops::hyper_connection_mix(r, w, m.down.weight, m.up.weight,
                                              ops::LinearPolicy::A16Only, kStreams, kRank, kEps, x,
                                              m.inject ? &inj : nullptr, workspace, s);
                };
                call(stream);
                CUDA_CHECK(cudaStreamSynchronize(stream));
                if (dump.is_open()) {
                    std::vector<char> bytes(static_cast<std::size_t>(kHidden) * t * 2);
                    CUDA_CHECK(
                        cudaMemcpy(bytes.data(), x_out.p, bytes.size(), cudaMemcpyDeviceToHost));
                    dump.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
                    if (m.inject) {
                        bytes.resize(static_cast<std::size_t>(kStreams) * t * sizeof(float));
                        CUDA_CHECK(cudaMemcpy(bytes.data(), inject_out.p, bytes.size(),
                                              cudaMemcpyDeviceToHost));
                        dump.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
                    }
                }
                bench::TimedGraph graph;
                graph.capture(stream, [&](cudaStream_t s) {
                    for (int c = 0; c < o.graph_calls; ++c) { call(s); }
                });
                if (o.profile) {
                    for (int i = 0; i < o.warmup; ++i) { graph.launch(stream); }
                    if (o.cache == "cold") { bench::flush_l2(flush, stream); }
                    CUDA_CHECK(cudaStreamSynchronize(stream));
                    CUDA_CHECK(cudaProfilerStart());
                    graph.launch(stream);
                    CUDA_CHECK(cudaStreamSynchronize(stream));
                    CUDA_CHECK(cudaProfilerStop());
                    return 0;
                }
                const double bytes =
                    static_cast<double>(m.down.weight.payload_bytes + m.up.weight.payload_bytes);
                for (const char* mode : {"cold", "warm"}) {
                    if (o.cache != "both" && o.cache != mode) { continue; }
                    const bench::ColdTiming time =
                        std::string(mode) == "cold"
                            ? bench::measure_cold_graph(graph, flush, stream, o.warmup, o.repeat)
                            : bench::measure_graph(graph, stream, o.warmup, o.repeat);
                    const double per = 1.0 / o.graph_calls;
                    std::printf("mixer=%-6s T=%3d route=%-8s cache=%s median=%8.2f us min=%8.2f us "
                                "p95=%8.2f us "
                                "weights=%.2f MB %.0f GB/s\n",
                                m.inject ? "inject" : "final", t, t <= 16 ? "fused" : "composed",
                                mode, time.median_us * per, time.min_us * per, time.p95_us * per,
                                bytes / 1e6, bytes / (time.median_us * per) / 1e3);
                }
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "infernix_hyper_connection_mix_bench: %s\n", error.what());
        return 1;
    }
}
