// The CPU miss service's ports (multi-GPU WP9; ops/offloaded_sparse_moe/cpu/miss_service.h): one
// service thread and one worker team answer the moe_experts calls of several streams or devices,
// each stream through a port of its own (mapped request, x, y, done, status and heartbeat words, a
// sequence counter on its device).
//
// Oracle: the CPU engine (expert_forward) of every routed (column, expert) pair, bit for bit. The
// calls are decode-shaped (T <= 8), so every expert takes the exact narrow route, whether a device
// frame, a staging slot or the service computed it. Each layer has a bank of its own and each call
// inputs of its own, so an answer computed from another layer's bank, or written to another port's
// outputs, changes bits.
//
// Cases:
//   - options: a one-port service is today's (one port on the current device with the Options'
//     caps); a multi-port service's channels carry each port's caps and the global layer, and its
//     counters sit on the ports' devices; missing ports or layers and bad port options throw.
//   - one GPU, two ports: two streams of device a each run every layer for kRounds rounds (one in
//     layer order, one in reverse, so both ports name the same layers), enqueued interleaved
//     without synchronization; the two ports have different caps.
//   - two GPUs (--two-gpus; --devices a,b, default 0,1; skipped with fewer than two GPUs unless
//     --devices names them, and a == b runs two streams of one device): port 0 on device a runs
//     layers [0, kSplit), port 1 on device b layers [kSplit, kLayers). Once chained as a layer
//     split's decode rounds are (b's round r after a's, a's round r + 1 after b's round r, so each
//     port idles while the other device works), once with both devices running at once.
// After each run every call's error word is clear, each port answered exactly the requests its
// device published (its done word equals its sequence counter, and the service's request count
// grew by their sum), the service computed exactly the CPU jobs the selection rule gives, and every
// port's heartbeat advances. The one-GPU case also reports the service's idle waits as the gaps of
// a heartbeat (about 50 us asked after 20 ms idle, 1 ms after 1 s; a high-resolution waitable timer
// on Windows) with the share of time it polls between them; they are not checked.

#include "core/device_guard.h"
#include "infernix/ops/offloaded_sparse_moe.h"
#include "ops/offloaded_moe_fixtures.h"
#include "ops/offloaded_sparse_moe/cpu/miss_service.h"
#include "ops/offloaded_sparse_moe/cpu/w4a4_expert.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace {

namespace ops      = infernix::ops;
namespace t        = infernix::test;
namespace moe      = infernix::ops::offloaded_moe;
namespace fixtures = infernix::test::offloaded_moe;
using infernix::DeviceGuard;
using infernix::DType;
using infernix::Tensor;
using Service = moe::CpuMissService;

constexpr int kExperts      = 24;
constexpr int kTopK         = 10;
constexpr int kLayers       = 4;
constexpr int kSplit        = 2; // two GPUs: device a runs [0, kSplit), device b [kSplit, kLayers)
constexpr int kRounds       = 16;
constexpr int kStagingSlots = 8; // the GPU's share of a T = 8 call's misses runs in passes
// Columns of layer l's calls: decode and verification shapes (lanes x width).
constexpr int kColumns[kLayers] = {1, 8, 2, 4};

int g_failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++g_failures;
    }
}

template <class E>
void expect_throw(const std::function<void()>& f, const std::string& what) {
    try {
        f();
    } catch (const E&) { return; } catch (const std::exception& e) {
        check(false, what + " (threw another exception: " + e.what() + ")");
        return;
    }
    check(false, what + " (did not throw)");
}

// --------------------------------------------------------------------------------------- memory

// Pinned, mapped, portable host memory: every device reaches it.
class PinnedHost {
public:
    explicit PinnedHost(std::size_t bytes) {
        t::cuda_check(cudaHostAlloc(&p_, bytes, cudaHostAllocMapped | cudaHostAllocPortable),
                      "cudaHostAlloc");
        std::memset(p_, 0, bytes);
    }

    ~PinnedHost() {
        if (p_ != nullptr) { cudaFreeHost(p_); }
    }

    PinnedHost(const PinnedHost&)            = delete;
    PinnedHost& operator=(const PinnedHost&) = delete;

    [[nodiscard]] void* get() const { return p_; }

    // Its address in device `device`'s kernels.
    [[nodiscard]] void* on(int device) const {
        const DeviceGuard guard(device);
        void* address = nullptr;
        t::cuda_check(cudaHostGetDevicePointer(&address, p_, 0), "cudaHostGetDevicePointer");
        return address;
    }

private:
    void* p_ = nullptr;
};

// Memory of one device, allocated, filled and freed with that device current.
class DeviceMemory {
public:
    DeviceMemory(int device, std::size_t bytes) : device_(device) {
        const DeviceGuard guard(device);
        t::cuda_check(cudaMalloc(&p_, std::max<std::size_t>(bytes, 16)), "cudaMalloc");
    }

    template <class T>
    DeviceMemory(int device, const std::vector<T>& contents)
        : DeviceMemory(device, contents.size() * sizeof(T)) {
        const DeviceGuard guard(device);
        t::cuda_check(
            cudaMemcpy(p_, contents.data(), contents.size() * sizeof(T), cudaMemcpyHostToDevice),
            "cudaMemcpy");
    }

    ~DeviceMemory() {
        if (p_ == nullptr) { return; }
        try {
            const DeviceGuard guard(device_);
            cudaFree(p_);
        } catch (...) {}
    }

    DeviceMemory(DeviceMemory&& other) noexcept
        : device_(other.device_), p_(std::exchange(other.p_, nullptr)) {}

    DeviceMemory& operator=(DeviceMemory&&)      = delete;
    DeviceMemory(const DeviceMemory&)            = delete;
    DeviceMemory& operator=(const DeviceMemory&) = delete;

    template <class T = void>
    [[nodiscard]] T* get() const {
        return static_cast<T*>(p_);
    }

    template <class T>
    [[nodiscard]] std::vector<T> download(std::size_t count) const {
        std::vector<T> out(count);
        const DeviceGuard guard(device_);
        t::cuda_check(cudaMemcpy(out.data(), p_, count * sizeof(T), cudaMemcpyDeviceToHost),
                      "cudaMemcpy");
        return out;
    }

private:
    int device_ = 0;
    void* p_    = nullptr;
};

// ---------------------------------------------------------------------------------------- inputs

// A layer's experts: every record in a pinned bank (read by the service and by the GPU's misses),
// every third one also in a device frame.
struct HostLayer {
    std::unique_ptr<PinnedHost> bank;
    std::vector<moe::ExpertScales> scales;
    std::vector<std::int32_t> frames; // expert -> frame, -1 when not resident
    int resident = 0;

    [[nodiscard]] std::uint8_t* record(int expert) const {
        return static_cast<std::uint8_t*>(bank->get()) +
               moe::kRecordBytes * static_cast<std::size_t>(expert);
    }
};

std::vector<HostLayer> make_layers() {
    std::vector<HostLayer> layers(kLayers);
    for (int l = 0; l < kLayers; ++l) {
        std::mt19937 rng(1009U + static_cast<std::uint32_t>(l));
        HostLayer& layer = layers[static_cast<std::size_t>(l)];
        layer.bank       = std::make_unique<PinnedHost>(moe::kRecordBytes * kExperts);
        layer.frames.assign(kExperts, -1);
        for (int e = 0; e < kExperts; ++e) {
            const fixtures::Expert expert = fixtures::random_expert(rng, e % 3 == 0);
            layer.scales.push_back(expert.scales);
            std::memcpy(layer.record(e), expert.record.data(), moe::kRecordBytes);
            if (e % 3 == 0) { layer.frames[static_cast<std::size_t>(e)] = layer.resident++; }
        }
    }
    return layers;
}

std::vector<Service::Layer> service_layers(const std::vector<HostLayer>& layers) {
    std::vector<Service::Layer> out;
    for (const HostLayer& layer : layers) {
        out.push_back({.records       = layer.record(0),
                       .record_stride = moe::kRecordBytes,
                       .scales        = layer.scales.data()});
    }
    return out;
}

// One call's inputs: kTopK distinct experts per column, x, and the CPU engine's outputs.
struct Inputs {
    int layer   = 0;
    int columns = 0;
    std::vector<std::int32_t> ids;       // I32 [k, T]: column t's slot s at t * k + s
    std::vector<std::uint16_t> x;        // BF16 [H, T]
    std::vector<std::uint16_t> expected; // BF16 [H, k * T]
};

Inputs make_inputs(const std::vector<HostLayer>& layers, int layer, std::uint32_t seed) {
    std::mt19937 rng(seed);
    Inputs in;
    in.layer   = layer;
    in.columns = kColumns[layer];
    std::vector<std::int32_t> order(kExperts);
    std::iota(order.begin(), order.end(), 0);
    for (int c = 0; c < in.columns; ++c) {
        std::shuffle(order.begin(), order.end(), rng);
        in.ids.insert(in.ids.end(), order.begin(), order.begin() + kTopK);
    }
    in.x = fixtures::random_activations(rng, in.columns);
    in.expected.resize(static_cast<std::size_t>(moe::kHidden) * in.ids.size());
    const HostLayer& host = layers[static_cast<std::size_t>(layer)];
    for (std::size_t i = 0; i < in.ids.size(); ++i) {
        const int expert           = in.ids[i];
        const std::uint16_t* xp[1] = {&in.x[i / kTopK * moe::kHidden]};
        std::uint16_t* yp[1]       = {&in.expected[i * moe::kHidden]};
        moe::expert_forward(moe::best_cpu_isa(), host.record(expert),
                            host.scales[static_cast<std::size_t>(expert)], 1, xp, yp);
    }
    return in;
}

// The CPU jobs a call hands port `port` (design §19.3.5 S3): want = min(cap, M - M / divisor) of
// its M misses, taken from those of at most max_job_columns columns; none when the call is wider
// than max_columns.
int cpu_jobs(const Inputs& in, const HostLayer& layer, const Service::PortOptions& port) {
    if (in.columns > port.max_columns) { return 0; }
    std::vector<int> columns(kExperts, 0);
    for (const std::int32_t e : in.ids) { ++columns[static_cast<std::size_t>(e)]; }
    int misses = 0, eligible = 0;
    for (int e = 0; e < kExperts; ++e) {
        const int n = columns[static_cast<std::size_t>(e)];
        if (n == 0 || layer.frames[static_cast<std::size_t>(e)] >= 0) { continue; }
        ++misses;
        eligible += n <= port.max_job_columns ? 1 : 0;
    }
    const int cap =
        port.wide_from > 0 && in.columns >= port.wide_from ? port.wide_jobs : port.max_jobs;
    const int want =
        std::min(cap, port.pcie_divisor > 0 ? misses - misses / port.pcie_divisor : misses);
    return std::min(want, eligible);
}

// ----------------------------------------------------------------------------------------- lanes

// One call's device state on its lane's device: the layer's frames and scales, the routing's
// dispatch, x, a workspace, and one output per round (filled with 0xFF: an unwritten output shows).
struct Call {
    Call(int device, const HostLayer& layer, const Inputs& in)
        : inputs(&in), entries(kTopK * in.columns), max_jobs(std::min(kExperts, entries)),
          bank(static_cast<const std::uint8_t*>(layer.bank->on(device))),
          frames(device, layer.frames),
          frame_records(device, moe::kRecordBytes * static_cast<std::size_t>(layer.resident)),
          scales(device, layer.scales), ids(device, in.ids), x(device, in.x),
          dispatch(device, ops::moe_dispatch_bytes(kExperts, entries)),
          workspace(device, ops::moe_experts_workspace_bytes(max_jobs, entries)) {
        const DeviceGuard guard(device);
        for (int e = 0; e < kExperts; ++e) {
            const std::int32_t frame = layer.frames[static_cast<std::size_t>(e)];
            if (frame < 0) { continue; }
            t::cuda_check(cudaMemcpy(frame_records.get<std::uint8_t>() +
                                         moe::kRecordBytes * static_cast<std::size_t>(frame),
                                     layer.record(e), moe::kRecordBytes, cudaMemcpyHostToDevice),
                          "cudaMemcpy");
        }
        view = ops::carve_moe_dispatch(dispatch.get(), kExperts, entries);
        outputs.reserve(kRounds);
        for (int r = 0; r < kRounds; ++r) {
            outputs.emplace_back(device, output_bytes());
            t::cuda_check(cudaMemset(outputs.back().get(), 0xFF, output_bytes()), "cudaMemset");
        }
    }

    [[nodiscard]] std::size_t output_bytes() const {
        return sizeof(std::uint16_t) * moe::kHidden * static_cast<std::size_t>(entries);
    }

    const Inputs* inputs;
    std::int32_t entries, max_jobs;
    const std::uint8_t* bank; // the layer's pinned bank, as the device addresses it
    DeviceMemory frames, frame_records, scales, ids, x, dispatch, workspace;
    ops::MoeDispatch view{};
    std::vector<DeviceMemory> outputs;
};

// One stream's calls, all through one port of the service, with the stream's staging slots, its
// error word and one event per round.
class Lane {
public:
    Lane(int device, int port, const std::vector<HostLayer>& layers,
         const std::vector<const Inputs*>& inputs)
        : device_(device), port_(port), staging_(device, moe::kRecordBytes * kStagingSlots),
          error_(64) {
        const DeviceGuard guard(device);
        t::cuda_check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
                      "cudaStreamCreate");
        for (cudaEvent_t& event : events_) {
            t::cuda_check(cudaEventCreateWithFlags(&event, cudaEventDisableTiming),
                          "cudaEventCreate");
        }
        error_device_ = static_cast<std::uint32_t*>(error_.on(device));
        for (const Inputs* in : inputs) {
            calls_.push_back(
                std::make_unique<Call>(device, layers[static_cast<std::size_t>(in->layer)], *in));
            Call& call = *calls_.back();
            const ops::MoeRouting routing{Tensor(call.ids.get(), DType::I32, {kTopK, in->columns}),
                                          Tensor(), Tensor()};
            ops::moe_dispatch(routing, kExperts, call.view, nullptr, stream_);
        }
        t::cuda_synchronize(stream_);
    }

    ~Lane() {
        try {
            const DeviceGuard guard(device_);
            for (cudaEvent_t event : events_) {
                if (event != nullptr) { cudaEventDestroy(event); }
            }
            if (stream_ != nullptr) { cudaStreamDestroy(stream_); }
        } catch (...) {}
    }

    Lane(const Lane&)            = delete;
    Lane& operator=(const Lane&) = delete;

    [[nodiscard]] int port() const { return port_; }

    [[nodiscard]] std::size_t size() const { return calls_.size(); }

    [[nodiscard]] const Call& call(std::size_t i) const { return *calls_[i]; }

    // Call i's moe_experts of round `round`, on the lane's stream and port.
    void enqueue(const Service& service, std::size_t i, int round) {
        Call& call       = *calls_[i];
        const Inputs& in = *call.inputs;
        const DeviceGuard guard(device_);
        ops::MoeExpertSource source{.frame_base    = call.frame_records.get<const std::uint8_t>(),
                                    .frames        = call.frames.get<const std::int32_t>(),
                                    .host_records  = call.bank,
                                    .record_stride = moe::kRecordBytes,
                                    .scales        = call.scales.get<const moe::ExpertScales>(),
                                    .staging_base  = staging_.get<std::uint8_t>(),
                                    .staging_slots = kStagingSlots,
                                    .cpu           = service.channel(in.layer, port_)};
        source.error = error_device_;
        Tensor x(call.x.get(), DType::BF16, {moe::kHidden, in.columns});
        Tensor out(call.outputs[static_cast<std::size_t>(round)].get(), DType::BF16,
                   {moe::kHidden, call.entries});
        ops::moe_experts(x, call.view, source, kTopK, call.max_jobs, call.workspace.get(), out,
                         stream_);
    }

    void record(int round) {
        const DeviceGuard guard(device_);
        t::cuda_check(cudaEventRecord(events_[static_cast<std::size_t>(round)], stream_),
                      "cudaEventRecord");
    }

    // The lane's later work waits for `other`'s work up to its record(round) (across devices).
    void wait(const Lane& other, int round) {
        const DeviceGuard guard(device_);
        t::cuda_check(
            cudaStreamWaitEvent(stream_, other.events_[static_cast<std::size_t>(round)], 0),
            "cudaStreamWaitEvent");
    }

    void synchronize() {
        const DeviceGuard guard(device_);
        t::cuda_synchronize(stream_);
    }

    [[nodiscard]] std::uint32_t error() const {
        return *static_cast<const volatile std::uint32_t*>(error_.get());
    }

private:
    int device_;
    int port_;
    DeviceMemory staging_;
    PinnedHost error_;
    std::uint32_t* error_device_ = nullptr;
    cudaStream_t stream_         = nullptr;
    cudaEvent_t events_[kRounds] = {};
    std::vector<std::unique_ptr<Call>> calls_;
};

struct LanePlan {
    int device;
    int port;
    std::vector<int> layers;
};

// The device's count of requests published on a port (the channel's sequence counter).
std::uint32_t published(const Service& service, int port, int device) {
    const DeviceGuard guard(device);
    std::uint32_t value = 0;
    t::cuda_check(cudaMemcpy(&value, service.channel(0, port).sequence, sizeof(value),
                             cudaMemcpyDeviceToHost),
                  "cudaMemcpy");
    return value;
}

const volatile std::uint32_t* heartbeat_word(const Service& service, int port) {
    return reinterpret_cast<const volatile std::uint32_t*>(service.channel(0, port).heartbeat);
}

std::uint32_t heartbeat(const Service& service, int port) { return *heartbeat_word(service, port); }

// The service's idle waits, seen from port 0's heartbeat: after 20 ms without a request the
// service waits every 4096 polls, after 1 s after every poll, and the heartbeat stands still for as
// long. Each is reported with the share of the 200 ms the service thread spent polling (outside
// the waits); not checked.
void report_idle_gaps(const Service& service, const char* name, std::chrono::milliseconds idle) {
    using clock = std::chrono::steady_clock;
    std::this_thread::sleep_for(idle); // past the threshold, counted from the last request
    const volatile std::uint32_t* beat = heartbeat_word(service, 0);
    std::vector<double> gaps; // microseconds
    std::uint32_t seen = *beat;
    auto changed       = clock::now();
    const auto begin   = changed;
    const auto end     = changed + std::chrono::milliseconds(200);
    for (auto now = changed; now < end; now = clock::now()) {
        const std::uint32_t value = *beat;
        if (value == seen) { continue; }
        const double us = std::chrono::duration<double, std::micro>(now - changed).count();
        if (us > 20.0) { gaps.push_back(us); }
        seen    = value;
        changed = now;
    }
    if (gaps.empty()) {
        std::printf("%s: no heartbeat gap above 20 us in 200 ms\n", name);
        return;
    }
    const double window = std::chrono::duration<double, std::micro>(clock::now() - begin).count();
    const double waited = std::accumulate(gaps.begin(), gaps.end(), 0.0);
    std::sort(gaps.begin(), gaps.end());
    std::printf("%s: %zu waits in 200 ms, median %.0f us, p90 %.0f us, max %.0f us; polling %.1f%% "
                "of the time\n",
                name, gaps.size(), gaps[gaps.size() / 2], gaps[gaps.size() * 9 / 10], gaps.back(),
                100.0 * std::max(0.0, window - waited) / window);
}

void report_idle_wait(const Service& service) {
    report_idle_gaps(service, "idle wait (50 us asked)", std::chrono::milliseconds(50));
    report_idle_gaps(service, "deep idle wait (1 ms asked)", std::chrono::milliseconds(1100));
}

// Runs every lane's calls for kRounds rounds through one service with `ports` and checks them.
// chained: lane i + 1's round r waits for lane i's, and lane 0's round r + 1 for the last lane's
// round r (a layer split's decode order); otherwise every lane runs at once, its calls enqueued
// interleaved with the other lanes'.
void run_case(const std::string& name, const std::vector<HostLayer>& layers,
              const std::vector<Service::PortOptions>& ports, const std::vector<LanePlan>& plans,
              bool chained, bool report_idle) {
    std::printf("%s\n", name.c_str());
    std::vector<std::unique_ptr<Inputs>> inputs;
    std::vector<std::vector<const Inputs*>> lane_inputs(plans.size());
    for (std::size_t i = 0; i < plans.size(); ++i) {
        for (const int layer : plans[i].layers) {
            inputs.push_back(std::make_unique<Inputs>(make_inputs(
                layers, layer,
                7919U * static_cast<std::uint32_t>(i + 1) + static_cast<std::uint32_t>(layer))));
            lane_inputs[i].push_back(inputs.back().get());
        }
    }
    Service service(service_layers(layers), {.workers = 4, .cpus = {}}, ports);
    std::vector<std::unique_ptr<Lane>> lanes;
    for (std::size_t i = 0; i < plans.size(); ++i) {
        lanes.push_back(
            std::make_unique<Lane>(plans[i].device, plans[i].port, layers, lane_inputs[i]));
    }
    // Expected: the requests each port's device publishes (calls with a CPU job) and the jobs.
    std::vector<std::uint64_t> requests(ports.size(), 0);
    std::uint64_t jobs = 0;
    for (const auto& lane : lanes) {
        const Service::PortOptions& port = ports[static_cast<std::size_t>(lane->port())];
        for (std::size_t i = 0; i < lane->size(); ++i) {
            const Inputs& in = *lane->call(i).inputs;
            const int n      = cpu_jobs(in, layers[static_cast<std::size_t>(in.layer)], port);
            requests[static_cast<std::size_t>(lane->port())] += n > 0 ? kRounds : 0;
            jobs += static_cast<std::uint64_t>(n) * kRounds;
        }
    }
    const std::uint64_t requests_before = service.served_requests();
    const std::uint64_t jobs_before     = service.served_experts();
    std::size_t longest                 = 0;
    for (const auto& lane : lanes) { longest = std::max(longest, lane->size()); }
    const auto start = std::chrono::steady_clock::now();
    for (int r = 0; r < kRounds; ++r) {
        if (chained) {
            for (std::size_t l = 0; l < lanes.size(); ++l) {
                if (l > 0) {
                    lanes[l]->wait(*lanes[l - 1], r);
                } else if (r > 0) {
                    lanes[l]->wait(*lanes.back(), r - 1);
                }
                for (std::size_t i = 0; i < lanes[l]->size(); ++i) {
                    lanes[l]->enqueue(service, i, r);
                }
                lanes[l]->record(r);
            }
        } else {
            for (std::size_t i = 0; i < longest; ++i) {
                for (const auto& lane : lanes) {
                    if (i < lane->size()) { lane->enqueue(service, i, r); }
                }
            }
        }
    }
    for (const auto& lane : lanes) { lane->synchronize(); }
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

    for (std::size_t l = 0; l < lanes.size(); ++l) {
        check(lanes[l]->error() == 0, name + ": lane " + std::to_string(l) + " error word " +
                                          std::to_string(lanes[l]->error()) +
                                          " (layer << 16 | code)");
    }
    // The service counts after answering, so its counts may trail the devices by a moment.
    std::uint64_t all_requests = 0;
    for (const std::uint64_t n : requests) { all_requests += n; }
    const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(1);
    while ((service.served_requests() - requests_before < all_requests ||
            service.served_experts() - jobs_before < jobs) &&
           std::chrono::steady_clock::now() < until) {
        std::this_thread::yield();
    }
    for (std::size_t p = 0; p < ports.size(); ++p) {
        const int port            = static_cast<int>(p);
        const std::uint32_t count = published(service, port, ports[p].device);
        const std::uint32_t done =
            *reinterpret_cast<const volatile std::uint32_t*>(service.channel(0, port).done);
        std::printf("  port %d (device %d): %u requests published, the last answered %u\n", port,
                    ports[p].device, count, done);
        check(requests[p] > 0, name + ": port " + std::to_string(p) + " receives requests");
        check(count == requests[p], name + ": port " + std::to_string(p) +
                                        " publishes one request per call with CPU jobs");
        check(done == count, name + ": port " + std::to_string(p) + " answered its last request");
    }
    const std::uint64_t served_requests = service.served_requests() - requests_before;
    const std::uint64_t served_jobs     = service.served_experts() - jobs_before;
    std::printf("  %zu lanes, %d rounds in %.3f s: %llu requests and %llu CPU jobs served "
                "(expected %llu and %llu)\n",
                lanes.size(), kRounds, seconds, static_cast<unsigned long long>(served_requests),
                static_cast<unsigned long long>(served_jobs),
                static_cast<unsigned long long>(all_requests),
                static_cast<unsigned long long>(jobs));
    check(served_requests == all_requests, name + ": the service answers every published request");
    check(served_jobs == jobs, name + ": the service computes min(cap, M - M / divisor) of the "
                                      "eligible misses of every call");
    check(service.failure().empty(), name + ": the service failed: " + service.failure());

    long mismatches    = 0;
    std::size_t values = 0;
    for (const auto& lane : lanes) {
        for (std::size_t i = 0; i < lane->size(); ++i) {
            const Call& call = lane->call(i);
            for (int r = 0; r < kRounds; ++r) {
                const auto got = call.outputs[static_cast<std::size_t>(r)].download<std::uint16_t>(
                    call.inputs->expected.size());
                for (std::size_t v = 0; v < got.size(); ++v) {
                    mismatches += got[v] != call.inputs->expected[v] ? 1 : 0;
                }
                values += got.size();
            }
        }
    }
    std::printf("  %ld of %zu outputs differ from the CPU engine\n", mismatches, values);
    check(mismatches == 0, name + ": every output equals the CPU engine's, bit for bit");

    // Every port's heartbeat advances (here while the service idles).
    std::vector<std::uint32_t> beats(ports.size());
    for (std::size_t p = 0; p < ports.size(); ++p) {
        beats[p] = heartbeat(service, static_cast<int>(p));
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    for (std::size_t p = 0; p < ports.size(); ++p) {
        check(heartbeat(service, static_cast<int>(p)) != beats[p],
              name + ": port " + std::to_string(p) + "'s heartbeat advances");
    }
    if (report_idle) { report_idle_wait(service); }
}

// ---------------------------------------------------------------------------------------- cases

int device_of(const void* pointer) {
    cudaPointerAttributes attributes{};
    t::cuda_check(cudaPointerGetAttributes(&attributes, pointer), "cudaPointerGetAttributes");
    return attributes.type == cudaMemoryTypeDevice ? attributes.device : -1;
}

bool same_caps(const ops::MoeCpuChannel& channel, const Service::PortOptions& port) {
    return channel.max_jobs == port.max_jobs && channel.max_columns == port.max_columns &&
           channel.pcie_divisor == port.pcie_divisor &&
           channel.max_job_columns == port.max_job_columns && channel.wide_from == port.wide_from &&
           channel.wide_jobs == port.wide_jobs;
}

void options_case(const std::vector<HostLayer>& host_layers, int a, int b) {
    std::printf("options (devices %d and %d)\n", a, b);
    const std::vector<Service::Layer> layers = service_layers(host_layers);
    {
        // One port: today's service, on the current device.
        const DeviceGuard guard(a);
        const Service::Options options{.workers         = 2,
                                       .max_jobs        = 5,
                                       .max_columns     = 33,
                                       .pcie_divisor    = 2,
                                       .max_job_columns = 3,
                                       .wide_from       = 9,
                                       .wide_jobs       = 40,
                                       .cpus            = {}};
        const Service service(layers, options);
        const Service::PortOptions caps{.device          = a,
                                        .max_jobs        = 5,
                                        .max_columns     = 33,
                                        .pcie_divisor    = 2,
                                        .max_job_columns = 3,
                                        .wide_from       = 9,
                                        .wide_jobs       = 40};
        check(service.port_count() == 1, "a service built from Options has one port");
        for (int l = 0; l < kLayers; ++l) {
            const auto channel = service.channel(l);
            check(channel.layer == l && same_caps(channel, caps),
                  "a one-port channel carries its layer and the Options' caps");
            check(channel.request == service.channel(l, 0).request,
                  "channel(layer) is channel(layer, 0)");
        }
        check(device_of(service.channel(0).sequence) == a,
              "a one-port service's sequence counter is on the current device");
        expect_throw<std::out_of_range>([&] { (void)service.channel(0, 1); },
                                        "a one-port service has no port 1");
        expect_throw<std::out_of_range>([&] { (void)service.channel(kLayers); },
                                        "a channel past the last layer");
        expect_throw<std::out_of_range>([&] { (void)service.channel(-1); }, "a negative layer");
    }
    {
        // Two ports on devices a and b, with different caps; the shared Options' caps are unused.
        const std::vector<Service::PortOptions> ports{
            {.device = a, .max_jobs = 4, .max_columns = 64, .pcie_divisor = 3},
            {.device          = b,
             .max_jobs        = 7,
             .max_columns     = 16,
             .pcie_divisor    = 0,
             .max_job_columns = 2,
             .wide_from       = 9,
             .wide_jobs       = 32}};
        const Service service(layers, {.workers = 3, .max_jobs = 1, .cpus = {}}, ports);
        check(service.port_count() == 2, "a service has the ports it was given");
        for (int p = 0; p < 2; ++p) {
            for (int l = 0; l < kLayers; ++l) {
                const auto channel = service.channel(l, p);
                check(channel.layer == l, "a port's channel carries the global layer");
                check(same_caps(channel, ports[static_cast<std::size_t>(p)]),
                      "a port's channel carries that port's caps");
            }
            check(device_of(service.channel(0, p).sequence) ==
                      ports[static_cast<std::size_t>(p)].device,
                  "a port's sequence counter is on the port's device");
        }
        const auto zero = service.channel(0, 0), one = service.channel(0, 1);
        check(zero.request != one.request && zero.x != one.x && zero.y != one.y &&
                  zero.done != one.done && zero.status != one.status &&
                  zero.heartbeat != one.heartbeat && zero.sequence != one.sequence,
              "every port has its own request, x, y, done, status, heartbeat and counter");
        expect_throw<std::out_of_range>([&] { (void)service.channel(0, 2); }, "no port 2");
        expect_throw<std::out_of_range>([&] { (void)service.channel(0, -1); }, "no port -1");
    }
    // Rejected options.
    const auto build = [&](Service::Options shared, std::vector<Service::PortOptions> ports) {
        return [&layers, shared, ports] { Service service(layers, shared, ports); };
    };
    const Service::Options shared{.workers = 2, .cpus = {}};
    expect_throw<std::invalid_argument>(build(shared, {}), "a service without ports");
    expect_throw<std::invalid_argument>(build({.workers = 0, .cpus = {}}, {{.device = a}}),
                                        "a service without workers");
    expect_throw<std::invalid_argument>(
        build(shared, {{.device = a}, {.device = a, .max_jobs = 0}}), "a port without CPU jobs");
    expect_throw<std::invalid_argument>(build(shared, {{.device = -1}}), "a negative device");
    expect_throw<std::invalid_argument>(build(shared, {{.device = a, .wide_from = 9}}),
                                        "a port with wide_from but no wide_jobs");
    expect_throw<std::invalid_argument>(
        build(shared, {{.device = a, .max_job_columns = moe::kMaxCpuColumns + 1}}),
        "a port whose jobs are wider than kMaxCpuColumns");
    int count = 0;
    t::cuda_check(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
    expect_throw<std::runtime_error>(build(shared, {{.device = a}, {.device = count}}),
                                     "a port on a device that does not exist");
}

void one_gpu_case(const std::vector<HostLayer>& layers, int device) {
    std::vector<int> forward(kLayers);
    std::iota(forward.begin(), forward.end(), 0);
    const std::vector<int> backward(forward.rbegin(), forward.rend());
    // Port 1 gives the CPU every miss up to its cap, of at most two columns each.
    run_case("one GPU (device " + std::to_string(device) + "): two streams, two ports", layers,
             {{.device = device, .max_jobs = 4, .max_columns = 64, .pcie_divisor = 3},
              {.device          = device,
               .max_jobs        = 6,
               .max_columns     = 64,
               .pcie_divisor    = 0,
               .max_job_columns = 2}},
             {{device, 0, forward}, {device, 1, backward}}, false, true);
}

void two_gpu_case(const std::vector<HostLayer>& layers, int a, int b) {
    std::vector<int> first(kSplit), second(kLayers - kSplit);
    std::iota(first.begin(), first.end(), 0);
    std::iota(second.begin(), second.end(), kSplit);
    const std::vector<Service::PortOptions> ports{
        {.device = a, .max_jobs = 4, .max_columns = 64, .pcie_divisor = 3},
        {.device = b, .max_jobs = 6, .max_columns = 64, .pcie_divisor = 2}};
    const std::string devices = "devices " + std::to_string(a) + " and " + std::to_string(b);
    run_case("two GPUs (" + devices + "): layer split, chained rounds", layers, ports,
             {{a, 0, first}, {b, 1, second}}, true, false);
    run_case("two GPUs (" + devices + "): both devices at once", layers, ports,
             {{a, 0, first}, {b, 1, second}}, false, false);
}

int usage() {
    std::fprintf(stderr, "usage: infernix_miss_service_ports_test [--two-gpus] [--devices a,b]\n");
    return 2;
}

} // namespace

int main(int argc, char** argv) {
    // Unbuffered, so that a crashing case's last progress line survives.
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    bool two_gpus         = false;
    bool explicit_devices = false;
    int devices[2]        = {0, 1};
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        if (arg == "--two-gpus") {
            two_gpus = true;
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
    if (two_gpus && !explicit_devices && count < 2) {
        std::printf("SKIP: fewer than two CUDA devices (--devices 0,0 runs both ports on one)\n");
        return 77;
    }
    if (!explicit_devices && count < 2) { devices[1] = devices[0]; }
    for (const int device : devices) {
        if (device < 0 || device >= count) {
            std::fprintf(stderr, "FAIL: no CUDA device %d (%d visible)\n", device, count);
            return 1;
        }
    }
    try {
        // The banks' pinned memory is allocated with device a current, so no other GPU gets a
        // context it does not use.
        const DeviceGuard guard(devices[0]);
        const std::vector<HostLayer> layers = make_layers();
        if (two_gpus) {
            two_gpu_case(layers, devices[0], devices[1]);
        } else {
            options_case(layers, devices[0], devices[1]);
            one_gpu_case(layers, devices[0]);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
    if (g_failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("all CPU miss service port checks passed\n");
    return 0;
}
