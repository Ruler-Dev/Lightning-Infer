// Decode-shape benchmark of offloaded_sparse_moe's routed experts (moe_experts) for the
// Qwen3.8-Flash-Next layer: 512 experts, top-10, H = 2560, I = 640, `nvfp4_expert_rg16_v1` records.
//
// Each point routes T <= 8 columns (decode and MTP verification widths: every expert is narrow)
// with a fixed skewed expert popularity, then times one graph-captured moe_experts call as decode
// issues it: the fork route (misses staged and computed on a side stream, the resident experts on
// the calling stream), joined before the end of the graph. Every record lives in its own device
// frame, so the timed work is the narrow gate/up and down kernels over resident experts (the
// steady-state hit path); the fork's staged pass finds no job.
//
// Cache modes: cold evicts L2 with a 256 MiB read before every sample (a decode layer's experts
// are not in L2: the previous layers read gigabytes); warm repeats the call with the records left
// in L2 (an upper bound). --dump writes every point's expert outputs in sweep order, so two builds
// can be compared byte for byte.
//
//   infernix_offloaded_moe_narrow_bench [--tokens 1,2,3,4,5,8] [--cache cold|warm|both]
//                                       [--warmup 5] [--repeat 50] [--seed N] [--dump PATH]

#include "infernix/ops/offloaded_sparse_moe.h"

#include "core/device.h"
#include "infernix_bench_common.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace infernix;
namespace moe = infernix::ops::offloaded_moe;

namespace {

constexpr int kExperts            = 512;
constexpr int kTopK               = 10;
constexpr int kDistinct           = 16; // distinct random records, copied into all 512 frames
constexpr int kStagingSlots       = 80; // decode's staging slots cover k * 8 jobs
constexpr std::size_t kFlushBytes = 256ULL << 20;

struct Options {
    std::vector<int> tokens{1, 2, 3, 4, 5, 8};
    std::string cache  = "both", dump;
    int warmup         = 5;
    int repeat         = 50;
    std::uint32_t seed = 20261011U;
};

std::vector<int> parse_list(const std::string& text) {
    std::vector<int> out;
    std::size_t begin = 0;
    while (begin < text.size()) {
        const std::size_t end = text.find(',', begin);
        const int value       = std::stoi(
            text.substr(begin, end == std::string::npos ? std::string::npos : end - begin));
        if (value < 1 || value > moe::kMaxColumns) {
            throw std::invalid_argument("--tokens takes decode widths 1.." +
                                        std::to_string(moe::kMaxColumns));
        }
        out.push_back(value);
        if (end == std::string::npos) { break; }
        begin = end + 1;
    }
    return out;
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto value            = [&]() -> std::string {
            if (i + 1 >= argc) { throw std::invalid_argument(arg + " needs a value"); }
            return argv[++i];
        };
        if (arg == "--tokens") {
            options.tokens = parse_list(value());
        } else if (arg == "--cache") {
            options.cache = value();
        } else if (arg == "--warmup") {
            options.warmup = std::stoi(value());
        } else if (arg == "--repeat") {
            options.repeat = std::stoi(value());
        } else if (arg == "--seed") {
            options.seed = static_cast<std::uint32_t>(std::stoul(value()));
        } else if (arg == "--dump") {
            options.dump = value();
        } else if (arg == "--help" || arg == "-h") {
            std::puts("usage: infernix_offloaded_moe_narrow_bench [--tokens 1,2,...] [--cache "
                      "cold|warm|both] "
                      "[--warmup N] [--repeat N] [--seed N] [--dump PATH]");
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown option " + arg);
        }
    }
    if (options.cache != "cold" && options.cache != "warm" && options.cache != "both") {
        throw std::invalid_argument("--cache takes cold, warm or both");
    }
    if (options.warmup < 0 || options.repeat <= 0) {
        throw std::invalid_argument("invalid --warmup or --repeat");
    }
    return options;
}

// Random codes with scale bytes in the checkpoint's usual range (E4M3 words 40-80).
std::vector<std::uint8_t> random_records(std::mt19937& rng) {
    std::vector<std::uint8_t> records(static_cast<std::size_t>(kDistinct) * moe::kRecordBytes);
    std::uniform_int_distribution<int> byte(0, 255), scale(40, 80);
    for (auto& b : records) { b = static_cast<std::uint8_t>(byte(rng)); }
    for (int r = 0; r < kDistinct; ++r) {
        std::uint8_t* record    = records.data() + static_cast<std::size_t>(r) * moe::kRecordBytes;
        const std::size_t units = static_cast<std::size_t>(moe::kRecordBytes / moe::kUnitBytes);
        for (std::size_t u = 0; u < units; ++u) {
            for (int s = 0; s < 16; ++s) {
                record[u * moe::kUnitBytes + 128 + s] = static_cast<std::uint8_t>(scale(rng));
            }
        }
    }
    return records;
}

// Top-10 distinct experts per column under popularity 1 / (rank + 1)^0.7 over a fixed random
// permutation of the experts (columns of one call share a few experts, as in decode).
std::vector<float> routing_logits(std::mt19937& rng, int columns) {
    std::vector<int> rank(kExperts);
    for (int e = 0; e < kExperts; ++e) { rank[static_cast<std::size_t>(e)] = e; }
    std::shuffle(rank.begin(), rank.end(), rng);
    std::vector<double> weight(kExperts);
    for (int e = 0; e < kExperts; ++e) {
        weight[static_cast<std::size_t>(e)] =
            1.0 / std::pow(rank[static_cast<std::size_t>(e)] + 1.0, 0.7);
    }
    std::normal_distribution<float> n(0.0F, 1.0F);
    std::vector<float> logits(static_cast<std::size_t>(kExperts + 1) * columns);
    for (int t = 0; t < columns; ++t) {
        float* column = &logits[static_cast<std::size_t>(t) * (kExperts + 1)];
        for (int e = 0; e <= kExperts; ++e) { column[e] = n(rng); }
        std::vector<double> w = weight;
        for (int k = 0; k < kTopK; ++k) {
            std::discrete_distribution<int> pick(w.begin(), w.end());
            const int e                    = pick(rng);
            w[static_cast<std::size_t>(e)] = 0.0;
            column[e]                      = 20.0F + static_cast<float>(kTopK - k);
        }
    }
    return logits;
}

struct Fixture {
    DeviceBuffer frames_memory;
    DeviceBuffer frames_table;
    DeviceBuffer scales;
    DeviceBuffer staging;
    std::uint8_t* host        = nullptr;
    std::uint8_t* host_device = nullptr;
    cudaStream_t fork         = nullptr;
    cudaEvent_t events[2]     = {};

    explicit Fixture(std::uint32_t seed) {
        std::mt19937 rng(seed);
        const auto records     = random_records(rng);
        const std::size_t bank = static_cast<std::size_t>(kExperts) * moe::kRecordBytes;
        frames_memory          = DeviceBuffer(bank);
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&host), bank, cudaHostAllocMapped));
        CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&host_device), host, 0));
        for (int e = 0; e < kExperts; ++e) {
            const std::uint8_t* source =
                records.data() + static_cast<std::size_t>(e % kDistinct) * moe::kRecordBytes;
            std::memcpy(host + static_cast<std::size_t>(e) * moe::kRecordBytes, source,
                        moe::kRecordBytes);
        }
        CUDA_CHECK(cudaMemcpy(frames_memory.p, host, bank, cudaMemcpyHostToDevice));
        std::vector<std::int32_t> table(kExperts);
        for (int e = 0; e < kExperts; ++e) { table[static_cast<std::size_t>(e)] = e; }
        frames_table = DeviceBuffer(sizeof(std::int32_t) * kExperts);
        CUDA_CHECK(
            cudaMemcpy(frames_table.p, table.data(), frames_table.bytes, cudaMemcpyHostToDevice));
        std::vector<moe::ExpertScales> s(kExperts);
        std::uniform_real_distribution<float> lg(-11.0F, -7.0F);
        for (auto& e : s) {
            e.input_gate = e.input_up = std::exp2(lg(rng));
            e.input_down              = std::exp2(lg(rng));
            e.alpha_gate = e.alpha_up = std::exp2(lg(rng)) * e.input_gate;
            e.alpha_down              = std::exp2(lg(rng)) * e.input_down;
        }
        scales = DeviceBuffer(sizeof(moe::ExpertScales) * kExperts);
        CUDA_CHECK(cudaMemcpy(scales.p, s.data(), scales.bytes, cudaMemcpyHostToDevice));
        staging = DeviceBuffer(static_cast<std::size_t>(kStagingSlots) * moe::kRecordBytes);
        CUDA_CHECK(cudaStreamCreateWithFlags(&fork, cudaStreamNonBlocking));
        for (auto& event : events) {
            CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
        }
    }

    ~Fixture() {
        for (auto event : events) { cudaEventDestroy(event); }
        if (fork != nullptr) { cudaStreamDestroy(fork); }
        if (host != nullptr) { cudaFreeHost(host); }
    }

    Fixture(const Fixture&)            = delete;
    Fixture& operator=(const Fixture&) = delete;

    ops::MoeExpertSource source() const {
        ops::MoeExpertSource s{.frame_base    = static_cast<const std::uint8_t*>(frames_memory.p),
                               .frames        = static_cast<const std::int32_t*>(frames_table.p),
                               .host_records  = host_device,
                               .record_stride = moe::kRecordBytes,
                               .scales        = static_cast<const moe::ExpertScales*>(scales.p),
                               .staging_base  = static_cast<std::uint8_t*>(staging.p),
                               .staging_slots = kStagingSlots};
        s.fork_stream    = fork;
        s.fork_events[0] = events[0];
        s.fork_events[1] = events[1];
        return s;
    }
};

void run_point(Fixture& fixture, int tokens, const Options& options, bench::L2FlushBuffer& flush,
               std::ofstream& dump, cudaStream_t stream) {
    std::mt19937 rng(options.seed + static_cast<std::uint32_t>(tokens));
    const int entries = kTopK * tokens;
    const auto logits = routing_logits(rng, tokens);
    DeviceBuffer d_logits(logits.size() * sizeof(float));
    CUDA_CHECK(cudaMemcpy(d_logits.p, logits.data(), d_logits.bytes, cudaMemcpyHostToDevice));
    DeviceBuffer ids(sizeof(std::int32_t) * entries), weights(sizeof(float) * entries),
        gate(sizeof(float) * tokens);
    ops::MoeRouting routing{Tensor(ids.p, DType::I32, {kTopK, tokens}),
                            Tensor(weights.p, DType::FP32, {kTopK, tokens}),
                            Tensor(gate.p, DType::FP32, {tokens})};
    ops::moe_route(Tensor(d_logits.p, DType::FP32, {kExperts + 1, tokens}), kTopK, routing, stream);
    DeviceBuffer dispatch_memory(ops::moe_dispatch_bytes(kExperts, entries));
    ops::MoeDispatch dispatch = ops::carve_moe_dispatch(dispatch_memory.p, kExperts, entries);
    ops::moe_dispatch(routing, kExperts, dispatch, nullptr, stream);
    std::int32_t jobs = 0;
    CUDA_CHECK(
        cudaMemcpyAsync(&jobs, dispatch.job_count, sizeof(jobs), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const DeviceBuffer x = bench::make_bf16(static_cast<std::size_t>(moe::kHidden) * tokens,
                                            options.seed + 7, -0.15F, 0.15F);
    DeviceBuffer outputs(sizeof(std::uint16_t) * moe::kHidden * entries);
    const int max_jobs = std::min(kExperts, entries);
    DeviceBuffer workspace(ops::moe_experts_workspace_bytes(max_jobs, entries));
    const ops::MoeExpertSource source = fixture.source();
    const Tensor tx(x.p, DType::BF16, {moe::kHidden, tokens});
    Tensor out(outputs.p, DType::BF16, {moe::kHidden, entries});
    const auto call = [&](cudaStream_t s) {
        ops::moe_experts(tx, dispatch, source, kTopK, max_jobs, workspace.p, out, s);
    };
    CUDA_CHECK(cudaMemset(outputs.p, 0, outputs.bytes));
    call(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (dump.is_open()) {
        std::vector<char> bytes(outputs.bytes);
        CUDA_CHECK(cudaMemcpy(bytes.data(), outputs.p, bytes.size(), cudaMemcpyDeviceToHost));
        dump.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    }
    bench::TimedGraph graph;
    graph.capture(stream, call);
    const double bytes = static_cast<double>(jobs) * static_cast<double>(moe::kRecordBytes);
    for (const char* mode : {"cold", "warm"}) {
        if (options.cache != "both" && options.cache != mode) { continue; }
        const bench::ColdTiming timing =
            std::string(mode) == "cold"
                ? bench::measure_cold_graph(graph, flush, stream, options.warmup, options.repeat)
                : bench::measure_graph(graph, stream, options.warmup, options.repeat);
        std::printf("T=%d cache=%s experts %3d (%5.1f MB of records) | moe_experts median %8.2f us "
                    "(min %8.2f, "
                    "p95 %8.2f) | %6.0f GB/s\n",
                    tokens, mode, jobs, bytes / 1e6, timing.median_us, timing.min_us, timing.p95_us,
                    bytes / timing.median_us / 1e3);
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        DeviceContext context;
        std::printf("# gpu=%s SMs=%d timed=moe_experts fork route, all records resident (routing "
                    "and dispatch "
                    "untimed)\n",
                    context.props.name, context.multiprocessor_count());
        Fixture fixture(options.seed);
        bench::L2FlushBuffer flush(kFlushBytes);
        std::ofstream dump;
        if (!options.dump.empty()) {
            dump.open(options.dump, std::ios::binary);
            if (!dump) { throw std::runtime_error("cannot open " + options.dump); }
        }
        for (const int tokens : options.tokens) {
            run_point(fixture, tokens, options, flush, dump, context.stream);
        }
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "infernix_offloaded_moe_narrow_bench: %s\n", error.what());
        return 1;
    }
}
