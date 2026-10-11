#include "ops/offloaded_sparse_moe/cpu/miss_service.h"

#include "core/device_guard.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <string>

#if defined(_WIN32)
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#    ifndef CREATE_WAITABLE_TIMER_HIGH_RESOLUTION
#        define CREATE_WAITABLE_TIMER_HIGH_RESOLUTION 0x00000002 // Windows 10 1803 and later
#    endif
#endif
#if defined(__x86_64__) || defined(_M_X64)
#    include <immintrin.h>
#endif

#include <cerrno>

namespace infernix::ops::offloaded_moe {
namespace {

// The service spins while requests arrive; after kIdleAfter without one on any port it waits
// kIdleWait every 4096 polls. After kDeepIdleAfter (an idle server) it waits kDeepIdleWait after
// every poll instead: the 4096 polls between short waits kept a core busy for a tenth or more of
// its time, and the first request after a long idle spell may wait a millisecond (sleep_for's
// floor on Windows before the waitable timer, which could be 15.6 ms).
constexpr auto kIdleAfter     = std::chrono::milliseconds(20);
constexpr auto kIdleWait      = std::chrono::microseconds(50);
constexpr auto kDeepIdleAfter = std::chrono::seconds(1);
constexpr auto kDeepIdleWait  = std::chrono::milliseconds(1);

void cuda_require(cudaError_t error, const char* what) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("CPU miss service ") + what + ": " + cudaGetErrorString(error));
    }
}

void relax() {
#if defined(__x86_64__) || defined(_M_X64)
    _mm_pause();
#endif
}

void pin(int cpu) {
#if defined(_WIN32)
    if (cpu >= 0 && cpu < 64) { (void)SetThreadAffinityMask(GetCurrentThread(), DWORD_PTR{1} << cpu); }
#else
    (void)cpu;
#endif
}

// The idle waits (kIdleWait, kDeepIdleWait). Windows rounds sleep_for up to its timer period (1 ms
// or more, up to 15.6 ms), which delays the first request after an idle spell by as much; a
// high-resolution waitable timer (Windows 10 1803 and later) waits ~0.52 ms for kIdleWait instead
// (median on Windows 11, measured by infernix_miss_service_ports_test: its floor, above the 50 us
// asked). Where none can be created or set, sleep_for remains.
class IdleWait {
public:
    IdleWait() {
#if defined(_WIN32)
        timer_ = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,
                                        TIMER_ALL_ACCESS);
#endif
    }

    ~IdleWait() {
#if defined(_WIN32)
        if (timer_ != nullptr) { CloseHandle(timer_); }
#endif
    }

    IdleWait(const IdleWait&)            = delete;
    IdleWait& operator=(const IdleWait&) = delete;

    void operator()(std::chrono::microseconds wait) const {
#if defined(_WIN32)
        if (timer_ != nullptr) {
            LARGE_INTEGER due{};
            due.QuadPart = -static_cast<LONGLONG>(wait.count()) * 10; // relative, 100 ns units
            if (SetWaitableTimer(timer_, &due, 0, nullptr, nullptr, FALSE) != 0 &&
                WaitForSingleObject(timer_, INFINITE) == WAIT_OBJECT_0) {
                return;
            }
        }
#endif
        std::this_thread::sleep_for(wait);
    }

private:
#if defined(_WIN32)
    HANDLE timer_ = nullptr;
#endif
};

bool valid(const CpuMissService::PortOptions& port) {
    return port.device >= 0 && port.max_jobs >= 1 && port.max_jobs <= kMaxCpuJobs &&
           port.max_columns >= 1 && port.max_columns <= kMaxCpuCallColumns &&
           port.max_job_columns >= 1 && port.max_job_columns <= kMaxCpuColumns &&
           port.wide_from >= 0 && port.wide_jobs >= 0 && port.wide_jobs <= kMaxCpuJobs &&
           (port.wide_from > 0) == (port.wide_jobs > 0);
}

// The one port of a service built from Options: the current device, with the options' caps.
CpuMissService::PortOptions current_port(const CpuMissService::Options& options) {
    int device = 0;
    cuda_require(cudaGetDevice(&device), "cudaGetDevice");
    return {.device          = device,
            .max_jobs        = options.max_jobs,
            .max_columns     = options.max_columns,
            .pcie_divisor    = options.pcie_divisor,
            .max_job_columns = options.max_job_columns,
            .wide_from       = options.wide_from,
            .wide_jobs       = options.wide_jobs};
}

} // namespace

CpuMissService::CpuMissService(std::vector<Layer> layers, Options options)
    : CpuMissService(std::move(layers), options, {current_port(options)}) {}

CpuMissService::CpuMissService(std::vector<Layer> layers, Options shared,
                               std::vector<PortOptions> ports)
    : layers_(std::move(layers)), options_(std::move(shared)) {
    if (options_.workers < 1 || ports.empty() || !std::all_of(ports.begin(), ports.end(), valid)) {
        throw std::invalid_argument("CPU miss service options are out of range");
    }
    for (const auto& layer : layers_) {
        // Without a bank (the SSD tier) every request is tiered: it carries its records' addresses.
        if ((layer.records == nullptr && options_.records == nullptr) || layer.scales == nullptr ||
            layer.record_stride < kRecordBytes) {
            throw std::invalid_argument("CPU miss service layer is incomplete");
        }
    }
    // Portable mapped memory: every device reads and writes it at its host pointer.
    const auto mapped = [](std::size_t bytes) {
        void* p = nullptr;
        cuda_require(cudaHostAlloc(&p, bytes, cudaHostAllocMapped | cudaHostAllocPortable), "cudaHostAlloc");
        std::memset(p, 0, bytes);
        return p;
    };
    ports_.resize(ports.size());
    port_experts_ = std::make_unique<std::atomic<std::uint64_t>[]>(ports.size());
    try {
        for (std::size_t i = 0; i < ports.size(); ++i) {
            Port& port   = ports_[i];
            port.options = ports[i];
            // A port that takes prefill calls sizes for every job a call may hand it: a gated
            // call's cap (MoeCpuChannel::wide_jobs, set per call) may rise to kMaxCpuJobs.
            port.jobs = port.options.wide_jobs > 0 ? kMaxCpuJobs : port.options.max_jobs;
            jobs_     = std::max(jobs_, port.jobs);
            // The sequence counter is the port's device's: its kernels count on it.
            const DeviceGuard guard(port.options.device);
            const std::size_t y_columns = static_cast<std::size_t>(port.jobs) * kMaxCpuColumns;
            port.request   = static_cast<MissRequest*>(mapped(sizeof(MissRequest)));
            port.x         = static_cast<std::uint16_t*>(mapped(sizeof(std::uint16_t) * kHidden * kMaxCpuXColumns));
            port.y         = static_cast<std::uint16_t*>(mapped(sizeof(std::uint16_t) * kHidden * y_columns));
            port.done      = static_cast<std::uint32_t*>(mapped(64));
            port.status    = port.done + 1;
            port.heartbeat = port.done + 2;
            cuda_require(cudaMalloc(&port.sequence, sizeof(std::uint32_t)), "cudaMalloc");
            cuda_require(cudaMemset(port.sequence, 0, sizeof(std::uint32_t)), "cudaMemset");
        }
        thread_ = std::thread([this] { serve(); });
    } catch (...) {
        release();
        throw;
    }
}

CpuMissService::~CpuMissService() {
    stop_.store(true, std::memory_order_release);
    if (thread_.joinable()) { thread_.join(); }
    release();
}

void CpuMissService::release() noexcept {
    for (Port& port : ports_) {
        if (port.sequence != nullptr) {
            try {
                const DeviceGuard guard(port.options.device);
                cudaFree(port.sequence);
            } catch (...) {} // the device cannot be selected: its memory goes with its context
        }
        if (port.done != nullptr) { cudaFreeHost(port.done); }
        if (port.y != nullptr) { cudaFreeHost(port.y); }
        if (port.x != nullptr) { cudaFreeHost(port.x); }
        if (port.request != nullptr) { cudaFreeHost(port.request); }
    }
}

MoeCpuChannel CpuMissService::channel(int global_layer, int port) const {
    if (global_layer < 0 || static_cast<std::size_t>(global_layer) >= layers_.size() || port < 0 ||
        port >= port_count()) {
        throw std::out_of_range("CPU miss service has no layer " + std::to_string(global_layer) +
                                " or port " + std::to_string(port));
    }
    const Port& p = ports_[static_cast<std::size_t>(port)];
    MoeCpuChannel out;
    out.request     = p.request;
    out.x           = p.x;
    out.y           = p.y;
    out.done        = p.done;
    out.status      = p.status;
    out.heartbeat   = p.heartbeat;
    out.sequence    = p.sequence;
    out.layer       = global_layer;
    out.max_jobs    = p.options.max_jobs;
    out.max_columns  = p.options.max_columns;
    out.pcie_divisor = p.options.pcie_divisor;
    out.max_job_columns = p.options.max_job_columns;
    out.wide_from       = p.options.wide_from;
    out.wide_jobs       = p.options.wide_jobs;
    return out;
}

void CpuMissService::serve() {
    try {
        pin(options_.cpus.empty() ? -1 : options_.cpus[0]);
        CpuExpertTeam team({.workers = options_.workers, .max_jobs = jobs_, .cpus = options_.cpus});
        const IdleWait idle_wait;
        std::vector<CpuExpertJob> jobs(static_cast<std::size_t>(jobs_));
        std::vector<CpuExpertJob> ready, read;
        std::vector<std::uint32_t> tickets;
        std::vector<int> read_jobs;
        const std::size_t ports = ports_.size();
        const auto published    = [this](std::size_t p) {
            return *reinterpret_cast<volatile std::uint32_t*>(&ports_[p].request->sequence);
        };
        // Every port's heartbeat advances together: a device waiting on one port also waits while
        // another port's request is computed.
        std::uint32_t beat   = 0;
        const auto heartbeat = [this, &beat] {
            ++beat;
            for (const Port& port : ports_) {
                *reinterpret_cast<volatile std::uint32_t*>(port.heartbeat) = beat;
            }
        };
        std::vector<std::uint32_t> seen(ports);
        for (std::size_t p = 0; p < ports; ++p) { seen[p] = published(p); }
        auto last           = std::chrono::steady_clock::now(); // the last request on any port
        std::uint32_t polls = 0;
        bool deep_idle      = false; // kDeepIdleAfter without a request on any port
        std::size_t next    = 0; // polled first: the port after the one served last (round-robin)
        while (!stop_.load(std::memory_order_acquire)) {
            std::size_t p          = ports; // the first port from `next` with a new request
            std::uint32_t sequence = 0;
            for (std::size_t i = 0; i < ports; ++i) {
                const std::size_t q = next + i < ports ? next + i : next + i - ports;
                sequence            = published(q);
                if (sequence != seen[q]) {
                    p = q;
                    break;
                }
            }
            heartbeat();
            if (p == ports) {
                relax();
                // Spin while any device decodes; after kIdleAfter without a request on any port,
                // wait kIdleWait every 4096 polls, and after kDeepIdleAfter kDeepIdleWait every poll.
                if (deep_idle) {
                    idle_wait(kDeepIdleWait);
                } else if (++polls % 4096 == 0) {
                    const auto idle = std::chrono::steady_clock::now() - last;
                    if (idle > kIdleAfter) { idle_wait(kIdleWait); }
                    deep_idle = idle > kDeepIdleAfter;
                }
                continue;
            }
            std::atomic_thread_fence(std::memory_order_acquire);
            seen[p]              = sequence;
            next                 = p + 1 < ports ? p + 1 : 0;
            const Port& port     = ports_[p];
            const MissRequest& r = *port.request;
            const Layer& layer   = layers_.at(static_cast<std::size_t>(r.layer));
            const int count      = r.jobs;
            std::uint32_t status = 0;
            ready.clear();
            read_jobs.clear();
            tickets.clear();
            for (int j = 0; j < count; ++j) {
                CpuExpertJob& job = jobs[static_cast<std::size_t>(j)];
                const auto expert = static_cast<std::uint64_t>(r.expert[j]);
                job.record        = r.tiered != 0 ? reinterpret_cast<const std::uint8_t*>(r.record[j])
                                                      : layer.records + expert * layer.record_stride;
                job.scales        = layer.scales[expert];
                job.ncols         = r.ncols[j];
                for (int c = 0; c < job.ncols; ++c) {
                    job.x[c] = port.x + static_cast<std::size_t>(r.column[j][c]) * kHidden;
                    job.y[c] = port.y + (static_cast<std::size_t>(j) * kMaxCpuColumns + c) * kHidden;
                }
                if (job.record != nullptr) {
                    ready.push_back(job);
                } else if (options_.records != nullptr) { // its read starts now, under the other jobs
                    read_jobs.push_back(j);
                    tickets.push_back(options_.records->demand(r.layer, static_cast<int>(expert)));
                } else {
                    status = ENOENT;
                }
            }
            team.run(std::span<const CpuExpertJob>(ready.data(), ready.size()));
            if (!read_jobs.empty()) {
                read.clear();
                for (std::size_t i = 0; i < read_jobs.size(); ++i) {
                    std::uint32_t st = 0;
                    while (!options_.records->landed(tickets[i])) {
                        heartbeat();
                        relax();
                    }
                    const std::uint8_t* record = options_.records->wait(tickets[i], st);
                    if (record == nullptr) {
                        status = st != 0 ? st : EIO;
                        continue;
                    }
                    CpuExpertJob job = jobs[static_cast<std::size_t>(read_jobs[i])];
                    job.record       = record;
                    read.push_back(job);
                }
                team.run(std::span<const CpuExpertJob>(read.data(), read.size()));
                for (const std::uint32_t t : tickets) { options_.records->done(t); }
            }
            *reinterpret_cast<volatile std::uint32_t*>(port.status) = status;
            std::atomic_thread_fence(std::memory_order_release);
            *reinterpret_cast<volatile std::uint32_t*>(port.done) = sequence;
            served_.fetch_add(1, std::memory_order_relaxed);
            experts_.fetch_add(static_cast<std::uint64_t>(count), std::memory_order_relaxed);
            port_experts_[p].fetch_add(static_cast<std::uint64_t>(count), std::memory_order_relaxed);
            last      = std::chrono::steady_clock::now();
            polls     = 0;
            deep_idle = false;
        }
    } catch (const std::exception& error) {
        // A failed service never answers: the device wait ends after its timeout instead of hanging,
        // and failure() tells the caller why.
        record_failure(error.what());
    } catch (...) {
        record_failure("unknown exception");
    }
}

void CpuMissService::record_failure(std::string what) noexcept {
    try {
        const std::lock_guard<std::mutex> lock(failure_mutex_);
        failure_ = what.empty() ? std::string("unknown exception") : std::move(what);
    } catch (...) {}
}

std::string CpuMissService::failure() const {
    const std::lock_guard<std::mutex> lock(failure_mutex_);
    return failure_;
}

} // namespace infernix::ops::offloaded_moe
