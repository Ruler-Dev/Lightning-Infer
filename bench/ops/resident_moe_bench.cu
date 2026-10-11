// Public-Op benchmark of resident_moe_experts at the Qwen3.8-Flash-Next MTP drafter's geometry: 512
// experts, top-10, H = 2560, I = 640, Q4_G64_FP16 row-split banks held whole in device memory
// (gate/up [512 * 1280, 2560], down [512 * 2560, 640]). A draft step routes one column;
// verification-shaped calls route a few. Each point times one graph-captured call over random
// distinct experts per column.
//
// Cache modes: cold evicts L2 with a 256 MiB read before every sample (a draft step's experts were
// read a whole forward ago); warm repeats the call with the experts left in L2. --dump writes every
// point's outputs in sweep order, so two builds can be compared byte for byte.
//
//   infernix_resident_moe_bench [--tokens 1,2,4] [--cache cold|warm|both] [--repeat N] [--dump
//   PATH]

#include "infernix/ops/resident_moe.h"

#include "core/device.h"
#include "infernix_bench_common.h"
#include "quantized_weight.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace infernix;

namespace {

constexpr int kExperts = 512, kTopK = 10, kHidden = 2560, kIntermediate = 640;
constexpr std::size_t kFlushBytes = 256ULL << 20;

struct Options {
    std::vector<int> tokens{1, 2, 4};
    std::string cache = "both", dump;
    int warmup = 5, repeat = 100;
};

std::vector<int> list(const std::string& text) {
    std::vector<int> out;
    std::size_t start = 0;
    while (start < text.size()) {
        const auto end  = text.find(',', start);
        const int value = std::stoi(text.substr(start, end - start));
        if (value < 1 || value > 64) { throw std::invalid_argument("--tokens takes 1..64"); }
        out.push_back(value);
        if (end == std::string::npos) { break; }
        start = end + 1;
    }
    return out;
}

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            std::puts("usage: infernix_resident_moe_bench [--tokens 1,2,...] [--cache "
                      "cold|warm|both] [--warmup N] "
                      "[--repeat N] [--dump PATH]");
            std::exit(0);
        }
        if (++i >= argc) { throw std::invalid_argument("missing value of " + arg); }
        const std::string value = argv[i];
        if (arg == "--tokens") {
            o.tokens = list(value);
        } else if (arg == "--cache") {
            o.cache = value;
        } else if (arg == "--warmup") {
            o.warmup = std::stoi(value);
        } else if (arg == "--repeat") {
            o.repeat = std::stoi(value);
        } else if (arg == "--dump") {
            o.dump = value;
        } else {
            throw std::invalid_argument("unknown option: " + arg);
        }
    }
    if (o.cache != "cold" && o.cache != "warm" && o.cache != "both") {
        throw std::invalid_argument("--cache takes cold, warm or both");
    }
    return o;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options o = parse(argc, argv);
        DeviceContext device;
        const cudaStream_t stream = device.stream;
        std::printf("# gpu=%s SMs=%d resident_moe_experts Q4_G64_FP16 E=%d k=%d H=%d I=%d\n",
                    device.props.name, device.multiprocessor_count(), kExperts, kTopK, kHidden,
                    kIntermediate);
        auto gate_up = bench::make_row_split_weight(
            QType::Q4_G64_FP16, kExperts * 2 * kIntermediate, kHidden, kHidden, 901U, 0.05F);
        auto down = bench::make_row_split_weight(QType::Q4_G64_FP16, kExperts * kHidden,
                                                 kIntermediate, kIntermediate, 902U, 0.05F);
        const double expert_bytes =
            static_cast<double>(gate_up.weight.payload_bytes + down.weight.payload_bytes) /
            kExperts;
        const int max_t = *std::max_element(o.tokens.begin(), o.tokens.end());
        DeviceBuffer input =
            bench::make_bf16(static_cast<std::size_t>(kHidden) * max_t, 903U, -1.0F, 1.0F);
        DeviceBuffer ids_buffer(sizeof(std::int32_t) * kTopK * max_t);
        DeviceBuffer outputs(static_cast<std::size_t>(kHidden) * kTopK * max_t * 2);
        DeviceBuffer workspace(ops::resident_moe_workspace_bytes(kTopK * max_t, kIntermediate));
        bench::L2FlushBuffer flush(kFlushBytes);
        std::ofstream dump;
        if (!o.dump.empty()) {
            dump.open(o.dump, std::ios::binary);
            if (!dump) { throw std::runtime_error("cannot open " + o.dump); }
        }
        std::mt19937 rng(20261011U);
        for (const int t : o.tokens) {
            std::vector<std::int32_t> ids(static_cast<std::size_t>(kTopK) * t);
            std::vector<int> pool(kExperts);
            std::iota(pool.begin(), pool.end(), 0);
            int distinct = 0;
            std::vector<bool> used(kExperts, false);
            for (int c = 0; c < t; ++c) {
                std::shuffle(pool.begin(), pool.end(), rng);
                for (int j = 0; j < kTopK; ++j) {
                    ids[static_cast<std::size_t>(c) * kTopK + j] =
                        pool[static_cast<std::size_t>(j)];
                    distinct +=
                        used[static_cast<std::size_t>(pool[static_cast<std::size_t>(j)])] ? 0 : 1;
                    used[static_cast<std::size_t>(pool[static_cast<std::size_t>(j)])] = true;
                }
            }
            CUDA_CHECK(cudaMemcpy(ids_buffer.p, ids.data(), ids.size() * sizeof(std::int32_t),
                                  cudaMemcpyHostToDevice));
            const Tensor x(input.p, DType::BF16, {kHidden, t});
            const Tensor id(ids_buffer.p, DType::I32, {kTopK, t});
            Tensor out(outputs.p, DType::BF16, {kHidden, kTopK * t});
            const auto call = [&](cudaStream_t s) {
                ops::resident_moe_experts(x, id, gate_up.weight, down.weight, kExperts,
                                          kIntermediate, workspace.p, workspace.bytes, out, s);
            };
            call(stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            if (dump.is_open()) {
                std::vector<char> bytes(static_cast<std::size_t>(kHidden) * kTopK * t * 2);
                CUDA_CHECK(
                    cudaMemcpy(bytes.data(), outputs.p, bytes.size(), cudaMemcpyDeviceToHost));
                dump.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
            }
            bench::TimedGraph graph;
            graph.capture(stream, call);
            const double bytes = expert_bytes * distinct;
            for (const char* mode : {"cold", "warm"}) {
                if (o.cache != "both" && o.cache != mode) { continue; }
                const bench::ColdTiming time =
                    std::string(mode) == "cold"
                        ? bench::measure_cold_graph(graph, flush, stream, o.warmup, o.repeat)
                        : bench::measure_graph(graph, stream, o.warmup, o.repeat);
                std::printf("T=%d cache=%s experts %3d (%5.1f MB) median %8.2f us min %8.2f us p95 "
                            "%8.2f us %6.0f GB/s\n",
                            t, mode, distinct, bytes / 1e6, time.median_us, time.min_us,
                            time.p95_us, bytes / time.median_us / 1e3);
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "infernix_resident_moe_bench: %s\n", error.what());
        return 1;
    }
}
