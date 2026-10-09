// The handoff primitives of a layer split (src/core/handoff.h; the gates in core/device.h).
//
// One GPU (device 0 as sender and receiver, on two non-blocking streams, as the same-device test
// mode of a layer split runs them):
//   - Gate rounds. The producer (a kernel writing the round's data from its word, then send) and
//     the consumer (receive, then a kernel checking every word of the data against its expected
//     word) are captured as graphs, one pair per size from 16 B to 1.3 MiB, and each round's
//     consumer is launched before its producer. --rounds rounds (default 100,000) cycle through
//     the sizes with strictly increasing words. The data of a round differs from every other
//     round's at every word, so a gate that passed early or a copy that read a stale line shows
//     as a mismatch; the sums of sent and received data must agree and no gate may time out.
//   - Event rounds: the same graphs in Event mode (a D2H memcpy node; record_sent and wait_sent
//     between the launches), --rounds / 10 rounds.
//   - The timeout: a gate bounded at 50 ms whose word is never published stores its error value,
//     copies nothing and records its wait, without trapping: the same stream then runs a
//     published round correctly, and a gate released from the host (release) passes.
//   - BulkHandoff of 80 MiB through the default 4 x 16 MiB ring is bit-exact with stream waits and
//     with host waits; transfer() returns while the sender's stream is still held by a kernel
//     (it does not block the host); and the pieces overlap: the transfer hides at least a quarter
//     of the faster side's time against a download and an upload of the same bytes on their own
//     (checked when both devices have two copy engines and the sides are within 2x of each other,
//     as on one device; always reported).
//   - Invalid arguments throw: payloads over the capacity, gate sizes that are not 16-byte
//     multiples, a bound without an error word, record_sent and wait_sent on a capturing stream,
//     empty rings and channels, a device that does not exist.
// Two GPUs, in both directions: the Gate and Event rounds (--rounds / 10 each) and the bulk
// transfer across the devices (bit-exact; its overlap is reported when one link is much slower);
// then one-way latency histograms (p50, p90, p99, max) of Event and
// Gate handoffs of 20, 100 and 320 KiB between graphs, --latency-rounds rounds each (default 2,000),
// beside the WP1 probe's targets (p50 at 100 KiB: Event <= 150 us, Gate <= 45 us). Latency is the
// receiver's %globaltimer after its receive less the sender's before its send, so it assumes that
// the two GPUs' global timers agree (they need not: on the RTX PRO 4000 + RTX 5080 box they differ
// by seconds); the sum of both directions' p50, which cancels any offset between them, is printed
// as well. The histograms are reported, not checked.

#include "core/arena.h"
#include "core/device.h"
#include "core/device_guard.h"
#include "core/handoff.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace {

using infernix::BulkHandoff;
using infernix::DeviceGuard;
using infernix::HandoffChannel;
using infernix::HandoffMode;
using infernix::PinnedGateBound;
using infernix::PinnedHostBuffer;
using Clock = std::chrono::steady_clock;

int g_failures = 0;

void expect(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << '\n';
        ++g_failures;
    }
}

// Whether call() throws an Exception (and not some other exception).
template <class Exception, class Call>
bool throws(Call&& call) {
    try {
        call();
    } catch (const Exception&) { return true; } catch (...) {
        // Another exception type fails the check as well.
    }
    return false;
}

double elapsed_ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

const char* mode_name(HandoffMode mode) { return mode == HandoffMode::Gate ? "Gate" : "Event"; }

std::string direction(int sender, int receiver) { return std::to_string(sender) + "->" + std::to_string(receiver); }

// ---- device side --------------------------------------------------------------------------------

// The value of word `i` of the round with word `round`. Multiplication by an odd constant is a
// bijection, so two rounds differ at every index.
__device__ __forceinline__ std::uint32_t pattern(std::uint32_t round, std::uint32_t i) {
    return (round * 0x9E3779B1U) ^ (i * 0x85EBCA77U + 0x632BE5ABU);
}

__device__ __forceinline__ unsigned long long global_ns() {
    unsigned long long now;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
    return now;
}

// One atomic per warp; every lane of the warp calls it.
__device__ __forceinline__ void add_warp(unsigned long long* total, unsigned long long value) {
    for (int offset = 16; offset > 0; offset /= 2) { value += __shfl_down_sync(0xFFFFFFFFU, value, offset); }
    if ((threadIdx.x & 31U) == 0 && value != 0) { atomicAdd(total, value); }
}

__global__ void advance_word(std::uint32_t* word) { *word += 1; }

__global__ void produce_round(std::uint32_t* data, std::uint32_t words, const std::uint32_t* round,
                              unsigned long long* sum) {
    const std::uint32_t r    = *round;
    unsigned long long local = 0;
    for (std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < words; i += gridDim.x * blockDim.x) {
        const std::uint32_t value = pattern(r, i);
        data[i]                   = value;
        local += value;
    }
    add_warp(sum, local);
}

__global__ void consume_round(const std::uint32_t* data, std::uint32_t words, const std::uint32_t* round,
                              unsigned long long* sum, unsigned long long* mismatches) {
    const std::uint32_t r    = *round;
    unsigned long long local = 0, bad = 0;
    for (std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < words; i += gridDim.x * blockDim.x) {
        const std::uint32_t value = data[i];
        local += value;
        bad += value != pattern(r, i) ? 1 : 0;
    }
    add_warp(sum, local);
    add_warp(mismatches, bad);
}

__global__ void stamp(unsigned long long* stamps, std::uint32_t* count) {
    stamps[*count] = global_ns();
    *count += 1;
}

// Holds its stream for `ns`.
__global__ void hold(unsigned long long ns) {
    const unsigned long long start = global_ns();
    while (global_ns() - start < ns) { __nanosleep(10000); }
}

constexpr int kThreads = 256;

unsigned int blocks_for(std::size_t words) {
    return static_cast<unsigned int>(std::clamp<std::size_t>((words + kThreads - 1) / kThreads, 1, 128));
}

void launch_advance(std::uint32_t* word, cudaStream_t stream) {
    advance_word<<<1, 1, 0, stream>>>(word);
    CUDA_CHECK(cudaGetLastError());
}

void launch_produce(void* data, std::size_t bytes, const std::uint32_t* round, unsigned long long* sum,
                    cudaStream_t stream) {
    const std::size_t words = bytes / 4;
    produce_round<<<blocks_for(words), kThreads, 0, stream>>>(static_cast<std::uint32_t*>(data),
                                                              static_cast<std::uint32_t>(words), round, sum);
    CUDA_CHECK(cudaGetLastError());
}

void launch_consume(const void* data, std::size_t bytes, const std::uint32_t* round, unsigned long long* sum,
                    unsigned long long* mismatches, cudaStream_t stream) {
    const std::size_t words = bytes / 4;
    consume_round<<<blocks_for(words), kThreads, 0, stream>>>(
        static_cast<const std::uint32_t*>(data), static_cast<std::uint32_t>(words), round, sum, mismatches);
    CUDA_CHECK(cudaGetLastError());
}

void launch_stamp(unsigned long long* stamps, std::uint32_t* count, cudaStream_t stream) {
    stamp<<<1, 1, 0, stream>>>(stamps, count);
    CUDA_CHECK(cudaGetLastError());
}

// ---- host resources, each owned with its device current -----------------------------------------

// Device memory, zeroed. cudaMemset and a cudaMemcpy from pageable memory may return before the
// device has written, and the test's non-blocking streams are not ordered after them, so both
// complete on the device before returning; the reads follow a synchronization of the writers.
class DeviceMemory {
public:
    DeviceMemory(int device, std::size_t bytes) : device_(device), bytes_(bytes) {
        const DeviceGuard guard(device);
        CUDA_CHECK(cudaMalloc(&data_, bytes));
        CUDA_CHECK(cudaMemset(data_, 0, bytes));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    ~DeviceMemory() {
        const DeviceGuard guard(device_);
        CUDA_CHECK(cudaFree(data_));
    }

    DeviceMemory(const DeviceMemory&)            = delete;
    DeviceMemory& operator=(const DeviceMemory&) = delete;

    [[nodiscard]] void* get() const noexcept { return data_; }
    [[nodiscard]] std::uint32_t* u32() const noexcept { return static_cast<std::uint32_t*>(data_); }
    [[nodiscard]] std::uint64_t* u64() const noexcept { return static_cast<std::uint64_t*>(data_); }
    // The type of CUDA's 64-bit atomics (std::uint64_t is unsigned long on Linux).
    [[nodiscard]] unsigned long long* ull() const noexcept { return static_cast<unsigned long long*>(data_); }

    // Only while the device runs nothing that waits for the host.
    void set_u32(std::uint32_t value) const {
        const DeviceGuard guard(device_);
        CUDA_CHECK(cudaMemcpy(data_, &value, sizeof(value), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    template <class T>
    [[nodiscard]] T read() const {
        T value{};
        const DeviceGuard guard(device_);
        CUDA_CHECK(cudaMemcpy(&value, data_, sizeof(T), cudaMemcpyDeviceToHost));
        return value;
    }

    [[nodiscard]] std::vector<unsigned char> bytes() const {
        std::vector<unsigned char> out(bytes_);
        const DeviceGuard guard(device_);
        CUDA_CHECK(cudaMemcpy(out.data(), data_, bytes_, cudaMemcpyDeviceToHost));
        return out;
    }

private:
    int device_        = 0;
    std::size_t bytes_ = 0;
    void* data_        = nullptr;
};

class Stream {
public:
    explicit Stream(int device) : device_(device) {
        const DeviceGuard guard(device);
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    }

    ~Stream() {
        const DeviceGuard guard(device_);
        CUDA_CHECK(cudaStreamDestroy(stream_));
    }

    Stream(const Stream&)            = delete;
    Stream& operator=(const Stream&) = delete;

    [[nodiscard]] cudaStream_t get() const noexcept { return stream_; }
    [[nodiscard]] int device() const noexcept { return device_; }

    void synchronize() const { CUDA_CHECK(cudaStreamSynchronize(stream_)); }

    // Submits queued work without waiting for it (WDDM batches launches otherwise).
    void flush() const {
        const cudaError_t status = cudaStreamQuery(stream_);
        if (status == cudaErrorNotReady) {
            (void)cudaGetLastError();
            return;
        }
        CUDA_CHECK(status);
    }

    // Makes this stream wait for the latest record of `event` (of any device).
    void wait(cudaEvent_t event) const {
        const DeviceGuard guard(device_);
        CUDA_CHECK(cudaStreamWaitEvent(stream_, event, 0));
    }

private:
    int device_          = 0;
    cudaStream_t stream_ = nullptr;
};

class Event {
public:
    explicit Event(int device) : device_(device) {
        const DeviceGuard guard(device);
        CUDA_CHECK(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming));
    }

    ~Event() { CUDA_CHECK(cudaEventDestroy(event_)); }

    Event(const Event&)            = delete;
    Event& operator=(const Event&) = delete;

    [[nodiscard]] cudaEvent_t get() const noexcept { return event_; }

    void record(const Stream& stream) const {
        const DeviceGuard guard(device_);
        CUDA_CHECK(cudaEventRecord(event_, stream.get()));
    }

private:
    int device_        = 0;
    cudaEvent_t event_ = nullptr;
};

// A graph captured from `body` on `stream`, with the stream's device current.
class Graph {
public:
    Graph(const Stream& stream, const std::function<void()>& body) : device_(stream.device()) {
        const DeviceGuard guard(device_);
        cudaGraph_t graph = nullptr;
        CUDA_CHECK(cudaStreamBeginCapture(stream.get(), cudaStreamCaptureModeThreadLocal));
        body();
        CUDA_CHECK(cudaStreamEndCapture(stream.get(), &graph));
        CUDA_CHECK(cudaGraphInstantiate(&exec_, graph, 0));
        CUDA_CHECK(cudaGraphDestroy(graph));
    }

    ~Graph() { CUDA_CHECK(cudaGraphExecDestroy(exec_)); }

    Graph(const Graph&)            = delete;
    Graph& operator=(const Graph&) = delete;

    void launch(const Stream& stream) const {
        const DeviceGuard guard(device_);
        CUDA_CHECK(cudaGraphLaunch(exec_, stream.get()));
    }

private:
    int device_           = 0;
    cudaGraphExec_t exec_ = nullptr;
};

// A mapped pinned error word (as ProgramImpl::expert_error_), cleared.
class ErrorWord {
public:
    ErrorWord() : buffer_(64) { clear(); }

    [[nodiscard]] std::uint32_t* get() const noexcept { return static_cast<std::uint32_t*>(buffer_.data()); }
    [[nodiscard]] std::uint32_t value() const noexcept { return *static_cast<const volatile std::uint32_t*>(get()); }
    void clear() const noexcept { std::memset(buffer_.data(), 0, buffer_.size()); }

private:
    PinnedHostBuffer buffer_;
};

// ---- per-round checks ----------------------------------------------------------------------------

// 16 B up to 1.3 MiB (a 16-byte multiple): one column of the 20 KiB residual and more.
constexpr std::size_t kRoundSizes[] = {16, 4096, 20480, 102400, 327680, 1363152};
constexpr std::size_t kMaxRoundBytes = 1363152;

void check_rounds(int sender, int receiver, HandoffMode mode, std::uint64_t rounds) {
    if (rounds == 0) { return; }
    const std::string label = std::string(mode_name(mode)) + " rounds " + direction(sender, receiver);
    HandoffChannel channel(sender, receiver, kMaxRoundBytes);
    const Stream producer(sender);
    const Stream consumer(receiver);
    const DeviceMemory source(sender, kMaxRoundBytes), produced_word(sender, 4), produced_sum(sender, 8);
    const DeviceMemory destination(receiver, kMaxRoundBytes), expected_word(receiver, 4), consumed_sum(receiver, 8),
        mismatches(receiver, 8), stats(receiver, 16);
    const ErrorWord error;
    const PinnedGateBound bound{.timeout_ns  = 1'000'000'000ULL,
                                .error_word  = error.get(),
                                .error_value = (7U << 16) | infernix::kErrorHandoffTimeout};
    // The slot holds one round: a producer starts after the previous round's consumer.
    const Event consumed(receiver);
    std::vector<std::unique_ptr<Graph>> produce, consume;
    for (const std::size_t bytes : kRoundSizes) {
        produce.push_back(std::make_unique<Graph>(producer, [&] {
            launch_advance(produced_word.u32(), producer.get());
            launch_produce(source.get(), bytes, produced_word.u32(), produced_sum.ull(), producer.get());
            channel.send(source.get(), bytes, mode, produced_word.u32(), producer.get());
        }));
        consume.push_back(std::make_unique<Graph>(consumer, [&] {
            launch_advance(expected_word.u32(), consumer.get());
            channel.receive(destination.get(), bytes, mode, expected_word.u32(), bound, stats.u64(), consumer.get());
            launch_consume(destination.get(), bytes, expected_word.u32(), consumed_sum.ull(), mismatches.ull(),
                           consumer.get());
        }));
    }
    const auto start = Clock::now();
    std::uint64_t done = 0;
    while (done < rounds) {
        const std::size_t size = done % std::size(kRoundSizes);
        if (mode == HandoffMode::Gate) {
            consume[size]->launch(consumer); // its gate waits for the producer launched after it
            consumer.flush();
            producer.wait(consumed.get());
            produce[size]->launch(producer);
            producer.flush();
            consumed.record(consumer);
        } else {
            producer.wait(consumed.get());
            produce[size]->launch(producer);
            channel.record_sent(producer.get());
            channel.wait_sent(consumer.get());
            consume[size]->launch(consumer);
            consumed.record(consumer);
            producer.flush();
            consumer.flush();
        }
        ++done;
        if (done % 4096 == 0 || done == rounds) {
            producer.synchronize();
            consumer.synchronize();
            if (error.value() != 0) { break; }
        }
    }
    producer.synchronize();
    consumer.synchronize();
    const double ms                   = elapsed_ms(start);
    const auto bad                    = mismatches.read<unsigned long long>();
    const auto sent                   = produced_sum.read<unsigned long long>();
    const auto received               = consumed_sum.read<unsigned long long>();
    const auto waits                  = stats.read<std::array<unsigned long long, 2>>();
    const std::uint32_t last_sent     = produced_word.read<std::uint32_t>();
    const std::uint32_t last_expected = expected_word.read<std::uint32_t>();
    expect(error.value() == 0, label + ": a gate timed out (error word " + std::to_string(error.value()) + ") after " +
                                   std::to_string(done) + " rounds");
    expect(bad == 0, label + ": " + std::to_string(bad) + " words differ from their round's data");
    expect(sent == received, label + ": the sums of the sent and the received data differ");
    expect(last_sent == done && last_expected == done, label + ": the words did not advance once per round");
    std::cout << label << ": " << done << " rounds, " << std::fixed << std::setprecision(1) << 1000.0 * ms / done
              << " us per round";
    if (mode == HandoffMode::Gate) {
        std::cout << ", " << waits[1] << " gate waits averaging "
                  << (waits[1] != 0 ? static_cast<double>(waits[0]) / waits[1] / 1000.0 : 0.0) << " us";
    }
    std::cout << '\n';
}

void check_timeout(int device) {
    constexpr std::size_t kBytes = 64 * 1024;
    const std::string label      = "the bounded gate on device " + std::to_string(device);
    const DeviceGuard guard(device); // the test's own launches
    HandoffChannel channel(device, device, kBytes);
    const Stream stream(device);
    const DeviceMemory source(device, kBytes), destination(device, kBytes), word(device, 4), sum(device, 8),
        mismatches(device, 8), stats(device, 16);
    const ErrorWord error;
    const PinnedGateBound bound{.timeout_ns  = 50'000'000ULL,
                                .error_word  = error.get(),
                                .error_value = (3U << 16) | infernix::kErrorHandoffTimeout};
    CUDA_CHECK(cudaMemsetAsync(destination.get(), 0xA5, kBytes, stream.get()));
    // Word 5 is never published: the slot's ready word is 0.
    word.set_u32(5);
    const auto start = Clock::now();
    channel.receive(destination.get(), kBytes, HandoffMode::Gate, word.u32(), bound, stats.u64(), stream.get());
    const cudaError_t status = cudaStreamSynchronize(stream.get());
    const double waited      = elapsed_ms(start);
    expect(status == cudaSuccess,
           label + ": a timeout leaves the stream usable (" + std::string(cudaGetErrorName(status)) + ")");
    if (status != cudaSuccess) { return; } // the context is gone; nothing below can run
    expect(error.value() == bound.error_value, label + ": a timeout stores the error value (found " +
                                                   std::to_string(error.value()) + ")");
    expect(waited >= 50.0 && waited < 5000.0,
           label + ": the gate waits for its bound (" + std::to_string(waited) + " ms)");
    const std::vector<unsigned char> untouched = destination.bytes();
    expect(std::all_of(untouched.begin(), untouched.end(), [](unsigned char b) { return b == 0xA5; }),
           label + ": a timed-out gate copies nothing");
    const auto waits = stats.read<std::array<unsigned long long, 2>>();
    expect(waits[1] == 1 && waits[0] >= 50'000'000ULL, label + ": the timed-out wait is recorded once (" +
                                                           std::to_string(waits[1]) + " waits, " +
                                                           std::to_string(waits[0]) + " ns)");

    // The stream and the channel still work: a published round arrives intact.
    error.clear();
    word.set_u32(6);
    launch_produce(source.get(), kBytes, word.u32(), sum.ull(), stream.get());
    channel.send(source.get(), kBytes, HandoffMode::Gate, word.u32(), stream.get());
    channel.receive(destination.get(), kBytes, HandoffMode::Gate, word.u32(), bound, nullptr, stream.get());
    launch_consume(destination.get(), kBytes, word.u32(), sum.ull(), mismatches.ull(), stream.get());
    stream.synchronize();
    expect(error.value() == 0 && mismatches.read<unsigned long long>() == 0,
           label + ": after a timeout, the stream runs a published round correctly");

    // release(): the host opens a gate whose sender will not run.
    CUDA_CHECK(cudaMemsetAsync(destination.get(), 0xA5, kBytes, stream.get()));
    stream.synchronize();
    word.set_u32(7);
    const PinnedGateBound patient{.error_word = error.get(), .error_value = 1}; // 1 s
    channel.receive(destination.get(), kBytes, HandoffMode::Gate, word.u32(), patient, nullptr, stream.get());
    stream.flush();
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    channel.release(7);
    stream.synchronize();
    expect(error.value() == 0, label + ": a gate released from the host passes");
    expect(destination.bytes() == source.bytes(), label + ": a gate released from the host copies the slot");
}

// ---- bulk transfers ------------------------------------------------------------------------------

void check_bulk(int sender, int receiver) {
    constexpr std::size_t kBytes = 80ULL << 20;
    const std::string label      = "BulkHandoff " + direction(sender, receiver);
    BulkHandoff bulk(sender, receiver);
    const Stream producer(sender);
    const Stream consumer(receiver);
    const DeviceMemory source(sender, kBytes), source_word(sender, 4), source_sum(sender, 8);
    const DeviceMemory destination(receiver, kBytes), destination_word(receiver, 4), destination_sum(receiver, 8),
        mismatches(receiver, 8);
    // Pinned well before the timed copies use it: Windows slows host-to-device DMA for a while after
    // a pin.
    const PinnedHostBuffer staging(kBytes);
    source_word.set_u32(0xB01CU);
    destination_word.set_u32(0xB01CU);
    {
        const DeviceGuard guard(sender);
        launch_produce(source.get(), kBytes, source_word.u32(), source_sum.ull(), producer.get());
    }
    producer.synchronize();
    // Checks the destination against the source's pattern, then clears it for the next transfer.
    const auto verify = [&](const std::string& what) {
        {
            const DeviceGuard guard(receiver);
            CUDA_CHECK(cudaMemsetAsync(mismatches.get(), 0, 8, consumer.get()));
            launch_consume(destination.get(), kBytes, destination_word.u32(), destination_sum.ull(),
                           mismatches.ull(), consumer.get());
            CUDA_CHECK(cudaMemsetAsync(destination.get(), 0, kBytes, consumer.get()));
        }
        consumer.synchronize();
        const auto bad = mismatches.read<unsigned long long>();
        expect(bad == 0, label + ": " + what + " is not bit-exact (" + std::to_string(bad) + " words differ)");
    };
    const auto run = [&] {
        const auto start = Clock::now();
        bulk.transfer(source.get(), destination.get(), kBytes, producer.get(), consumer.get());
        producer.flush();
        consumer.flush();
        producer.synchronize();
        consumer.synchronize();
        return elapsed_ms(start);
    };

    (void)run();
    verify("a transfer");

    // The host does not block: the sender's stream is held for 200 ms before the transfer.
    {
        const DeviceGuard guard(sender);
        hold<<<1, 1, 0, producer.get()>>>(200'000'000ULL);
        CUDA_CHECK(cudaGetLastError());
    }
    producer.flush();
    const auto start = Clock::now();
    bulk.transfer(source.get(), destination.get(), kBytes, producer.get(), consumer.get());
    const double enqueue = elapsed_ms(start);
    const cudaError_t pending = cudaStreamQuery(consumer.get());
    (void)cudaGetLastError();
    expect(enqueue < 100.0, label + ": transfer() blocked the host for " + std::to_string(enqueue) +
                                " ms while the sender's stream was held");
    expect(pending == cudaErrorNotReady, label + ": the receiver's uploads did not wait for the held sender");
    producer.flush();
    consumer.flush();
    producer.synchronize();
    consumer.synchronize();
    verify("a transfer behind a held sender");

    // Overlap: the ring against a download and an upload of the same bytes on their own, best of
    // three each. Without overlap the ring takes their sum; ideally it hides all but one piece of
    // the faster side. Hiding a quarter of the faster side's time is checked when the two sides are
    // within 2x of each other (and both devices have two copy engines); against a much slower side
    // the possible gain is within that side's noise, so it is only reported.
    double download = 1e30, upload = 1e30, piped = 1e30;
    for (int repetition = 0; repetition < 3; ++repetition) {
        auto copy_start = Clock::now();
        {
            const DeviceGuard guard(sender);
            CUDA_CHECK(cudaMemcpyAsync(staging.data(), source.get(), kBytes, cudaMemcpyDeviceToHost, producer.get()));
        }
        producer.synchronize();
        download   = std::min(download, elapsed_ms(copy_start));
        copy_start = Clock::now();
        {
            const DeviceGuard guard(receiver);
            CUDA_CHECK(
                cudaMemcpyAsync(destination.get(), staging.data(), kBytes, cudaMemcpyHostToDevice, consumer.get()));
        }
        consumer.synchronize();
        upload = std::min(upload, elapsed_ms(copy_start));
        piped  = std::min(piped, run());
    }
    verify("a timed transfer");
    int sender_engines = 0, receiver_engines = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&sender_engines, cudaDevAttrAsyncEngineCount, sender));
    CUDA_CHECK(cudaDeviceGetAttribute(&receiver_engines, cudaDevAttrAsyncEngineCount, receiver));
    const double faster = std::min(download, upload), slower = std::max(download, upload);
    std::cout << label << ": 80 MiB in " << std::fixed << std::setprecision(2) << piped << " ms through the ring; "
              << download << " ms download and " << upload << " ms upload on their own";
    if (sender_engines >= 2 && receiver_engines >= 2 && faster >= 0.5 * slower) {
        expect(piped < download + upload - 0.25 * faster,
               label + ": the pieces do not overlap (" + std::to_string(piped) + " ms against " +
                   std::to_string(download) + " + " + std::to_string(upload) + " ms)");
    } else {
        std::cout << " [overlap reported only: " << sender_engines << " and " << receiver_engines
                  << " copy engines, sides " << slower / faster << "x apart]";
    }

    // The host-wait fallback.
    bulk.set_host_wait(true);
    const double waited = run();
    bulk.set_host_wait(false);
    verify("a transfer with host waits");
    std::cout << ", " << waited << " ms with host waits\n";
}

// ---- argument checks -----------------------------------------------------------------------------

void check_arguments(int count) {
    constexpr std::size_t kCapacity = 4096;
    HandoffChannel channel(0, 0, kCapacity);
    const Stream stream(0);
    const DeviceMemory buffer(0, 2 * kCapacity), word(0, 4);
    const ErrorWord error;
    const PinnedGateBound bound{.error_word = error.get(), .error_value = 1};
    const auto send_bytes = [&](std::size_t bytes, HandoffMode mode) {
        channel.send(buffer.get(), bytes, mode, word.u32(), stream.get());
    };
    const auto receive_bytes = [&](std::size_t bytes, HandoffMode mode, const PinnedGateBound& with) {
        channel.receive(buffer.get(), bytes, mode, word.u32(), with, nullptr, stream.get());
    };
    expect(channel.capacity() == kCapacity && channel.sender() == 0 && channel.receiver() == 0,
           "a channel reports its capacity and devices");
    expect(throws<std::invalid_argument>([&] { send_bytes(kCapacity + 16, HandoffMode::Event); }),
           "send over the capacity throws");
    expect(throws<std::invalid_argument>([&] { receive_bytes(kCapacity + 16, HandoffMode::Gate, bound); }),
           "receive over the capacity throws");
    expect(throws<std::invalid_argument>([&] { send_bytes(24, HandoffMode::Gate); }),
           "a Gate send of a size that is not a 16-byte multiple throws");
    expect(throws<std::invalid_argument>([&] { receive_bytes(24, HandoffMode::Gate, bound); }),
           "a Gate receive of a size that is not a 16-byte multiple throws");
    expect(throws<std::invalid_argument>([&] { receive_bytes(64, HandoffMode::Gate, PinnedGateBound{}); }),
           "a Gate receive without an error word throws");
    expect(throws<std::invalid_argument>([&] {
               infernix::download_pinned_publish(error.get(), buffer.get(), 16, nullptr, word.u32(), stream.get());
           }),
           "download_pinned_publish without a ready word throws");
    expect(throws<std::invalid_argument>([&] { HandoffChannel empty(0, 0, 0); }), "a channel of 0 bytes throws");
    expect(throws<std::invalid_argument>([&] { HandoffChannel missing(0, count, 64); }),
           "a channel to a device that does not exist throws");
    expect(throws<std::invalid_argument>([&] { BulkHandoff empty(0, 0, 0, 4); }), "a ring of empty pieces throws");
    expect(throws<std::invalid_argument>([&] { BulkHandoff empty(0, 0, 1 << 20, 0); }), "a ring of no pieces throws");
    {
        BulkHandoff ring(0, 0, 1 << 20, 2);
        expect(throws<std::invalid_argument>([&] { ring.transfer(buffer.get(), buffer.get(), 16, nullptr, nullptr); }),
               "a bulk transfer on the legacy default stream throws");
    }
    // The Event mode's host steps outside a capture only.
    const DeviceGuard guard(0);
    for (const bool record : {true, false}) {
        CUDA_CHECK(cudaStreamBeginCapture(stream.get(), cudaStreamCaptureModeThreadLocal));
        const bool threw = throws<std::logic_error>([&] {
            if (record) {
                channel.record_sent(stream.get());
            } else {
                channel.wait_sent(stream.get());
            }
        });
        cudaGraph_t graph = nullptr;
        CUDA_CHECK(cudaStreamEndCapture(stream.get(), &graph));
        if (graph != nullptr) { CUDA_CHECK(cudaGraphDestroy(graph)); }
        expect(threw, std::string(record ? "record_sent" : "wait_sent") + " on a capturing stream throws");
    }
}

// ---- two-GPU latency -----------------------------------------------------------------------------

// One-way latencies (us) of `rounds` handoffs of `bytes` from `sender` to `receiver`, each round on
// its own (both streams idle before it), between a graph per side as a layer split's rounds run.
std::vector<double> one_way_latencies(int sender, int receiver, HandoffMode mode, std::size_t bytes, int rounds) {
    constexpr int kWarmup = 20;
    const int total       = rounds + kWarmup;
    HandoffChannel channel(sender, receiver, bytes);
    const Stream producer(sender);
    const Stream consumer(receiver);
    const DeviceMemory source(sender, bytes), produced_word(sender, 4), send_stamps(sender, 8ULL * total),
        send_count(sender, 4);
    const DeviceMemory destination(receiver, bytes), expected_word(receiver, 4), receive_stamps(receiver, 8ULL * total),
        receive_count(receiver, 4);
    const ErrorWord error;
    const PinnedGateBound bound{.error_word = error.get(), .error_value = (1U << 16) | infernix::kErrorHandoffTimeout};
    const Graph send_graph(producer, [&] {
        launch_advance(produced_word.u32(), producer.get());
        launch_stamp(send_stamps.ull(), send_count.u32(), producer.get());
        channel.send(source.get(), bytes, mode, produced_word.u32(), producer.get());
    });
    const Graph receive_graph(consumer, [&] {
        launch_advance(expected_word.u32(), consumer.get());
        channel.receive(destination.get(), bytes, mode, expected_word.u32(), bound, nullptr, consumer.get());
        launch_stamp(receive_stamps.ull(), receive_count.u32(), consumer.get());
    });
    for (int round = 0; round < total; ++round) {
        if (mode == HandoffMode::Gate) {
            receive_graph.launch(consumer);
            consumer.flush();
            send_graph.launch(producer);
            producer.flush();
        } else {
            send_graph.launch(producer);
            channel.record_sent(producer.get());
            channel.wait_sent(consumer.get());
            receive_graph.launch(consumer);
            producer.flush();
            consumer.flush();
        }
        consumer.synchronize();
        producer.synchronize();
    }
    expect(error.value() == 0, std::string(mode_name(mode)) + " latency rounds " + direction(sender, receiver) +
                                   ": a gate timed out");
    std::vector<unsigned long long> sent(static_cast<std::size_t>(total)), received(static_cast<std::size_t>(total));
    {
        const DeviceGuard guard(sender);
        CUDA_CHECK(cudaMemcpy(sent.data(), send_stamps.get(), 8ULL * total, cudaMemcpyDeviceToHost));
    }
    {
        const DeviceGuard guard(receiver);
        CUDA_CHECK(cudaMemcpy(received.data(), receive_stamps.get(), 8ULL * total, cudaMemcpyDeviceToHost));
    }
    std::vector<double> latencies;
    latencies.reserve(static_cast<std::size_t>(rounds));
    for (int round = kWarmup; round < total; ++round) {
        const auto i = static_cast<std::size_t>(round);
        latencies.push_back(static_cast<double>(static_cast<long long>(received[i] - sent[i])) / 1000.0);
    }
    std::sort(latencies.begin(), latencies.end());
    return latencies;
}

double percentile(const std::vector<double>& sorted, double q) {
    const auto index = std::min(sorted.size() - 1, static_cast<std::size_t>(q * static_cast<double>(sorted.size())));
    return sorted[index];
}

void report_latencies(int rounds) {
    constexpr std::size_t kLatencySizes[] = {20 * 1024, 100 * 1024, 320 * 1024};
    std::cout << "one-way handoff latency between graphs, " << rounds << " rounds each (us; p50 p90 p99 max):\n";
    for (const HandoffMode mode : {HandoffMode::Event, HandoffMode::Gate}) {
        for (const std::size_t bytes : kLatencySizes) {
            double p50_sum = 0.0;
            for (const auto& [sender, receiver] : {std::pair{0, 1}, std::pair{1, 0}}) {
                const std::vector<double> sorted = one_way_latencies(sender, receiver, mode, bytes, rounds);
                const double p50                 = percentile(sorted, 0.50);
                p50_sum += p50;
                std::cout << "  " << std::setw(5) << mode_name(mode) << ' ' << direction(sender, receiver) << ' '
                          << std::setw(3) << bytes / 1024 << " KiB: " << std::fixed << std::setprecision(1)
                          << std::setw(8) << p50 << ' ' << std::setw(8) << percentile(sorted, 0.90) << ' '
                          << std::setw(8) << percentile(sorted, 0.99) << ' ' << std::setw(8) << sorted.back();
                if (bytes == 100 * 1024) {
                    std::cout << "   (WP1 target p50 <= " << (mode == HandoffMode::Gate ? 45 : 150) << ")";
                }
                std::cout << '\n';
            }
            std::cout << "  " << std::setw(5) << mode_name(mode) << " 0->1 + 1->0 " << std::setw(3) << bytes / 1024
                      << " KiB: p50 sum " << p50_sum << " (free of any timer offset)\n";
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    std::uint64_t rounds = 100'000;
    int latency_rounds   = 2'000;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        if (arg == "--rounds" && i + 1 < argc) {
            rounds = std::stoull(argv[++i]);
        } else if (arg == "--latency-rounds" && i + 1 < argc) {
            latency_rounds = std::stoi(argv[++i]);
        } else {
            std::cerr << "usage: infernix_handoff_test [--rounds N] [--latency-rounds N]\n";
            return 2;
        }
    }
    int count               = 0;
    const cudaError_t error = cudaGetDeviceCount(&count);
    if (error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver || (error == cudaSuccess && count == 0)) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    if (error != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(error) << '\n';
        return 1;
    }
    try {
        CUDA_CHECK(cudaSetDevice(0));
        check_arguments(count);
        check_rounds(0, 0, HandoffMode::Gate, rounds);
        check_rounds(0, 0, HandoffMode::Event, rounds / 10);
        check_timeout(0);
        check_bulk(0, 0);
        if (count >= 2) {
            for (const auto& [sender, receiver] : {std::pair{0, 1}, std::pair{1, 0}}) {
                check_rounds(sender, receiver, HandoffMode::Gate, rounds / 10);
                check_rounds(sender, receiver, HandoffMode::Event, rounds / 10);
                check_bulk(sender, receiver);
            }
            if (latency_rounds > 0) { report_latencies(latency_rounds); }
        } else {
            std::cout << "one device: the cross-device rounds, transfers and latencies need two\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
    if (g_failures != 0) {
        std::cerr << g_failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "handoff: all checks passed on " << count << " device(s)\n";
    return 0;
}
