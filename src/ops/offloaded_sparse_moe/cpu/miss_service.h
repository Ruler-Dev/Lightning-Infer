#pragma once

// The host side of CPU-served misses (docs/maintainer/qwen3_8-flash-next-design.md §10.3): a
// service thread watches the mapped request that moe_experts publishes for a layer call, computes
// the requested experts with the worker team (the service thread is worker 0) from the pinned
// expert banks, writes the BF16 outputs to mapped memory and answers with the request's sequence.
// The arithmetic is expert_forward's, so a CPU-served output equals the GPU's bit for bit.
//
// A layer split (multi-GPU) runs layer calls on several devices at once, so the service has ports:
// each has its own mapped request, x, y, done, status and heartbeat, and a sequence counter on its
// device. A port holds one request at a time, so its calls must run in one stream order; calls that
// may overlap (other devices, or other streams of one device) use other ports. One service thread
// and one worker team serve every port, taking requests round-robin over the ports; layers are
// global, so any port may name any layer.
//
// The service owns the mapped channel buffers and the device sequence counters. It must outlive
// every call that uses its channels, and its destructor must run only when no call is pending.

#include "infernix/ops/offloaded_sparse_moe.h"
#include "ops/offloaded_sparse_moe/cpu/expert_team.h"
#include "ops/offloaded_sparse_moe/cpu/miss_request.h"
#include "ops/offloaded_sparse_moe/cpu/record_provider.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace infernix::ops::offloaded_moe {

class CpuMissService {
public:
    struct Layer {
        const std::uint8_t* records = nullptr; // pinned host bank, record_stride apart
        std::uint64_t record_stride = 0;
        const ExpertScales* scales  = nullptr; // host [E]
    };
    struct Options {
        int workers     = 8;     // including the service thread
        int max_jobs    = 4;     // CPU-served experts per layer call, at most kMaxCpuJobs
        int max_columns = 64;    // calls with more columns stay on the GPU, at most kMaxCpuCallColumns
        int pcie_divisor = 3;    // misses / pcie_divisor stay on the GPU stage (0: none)
        int max_job_columns = kMaxCpuColumns; // misses with more columns stay on the GPU
        int wide_from   = 0;     // calls of at least this many columns (0: none) take wide_jobs as their cap
        int wide_jobs   = 0;     // prefill assist cap, at most kMaxCpuJobs
        std::vector<int> cpus;   // optional CPU of worker i (worker 0 is the service thread)
        // Tiered requests (design §19.3.7): a job without a record pointer is read through `records`
        // (demanded before the other jobs run, computed after them); without a provider, or when a
        // read fails, the request is answered with an errno in the status word. The service keeps its
        // heartbeat word changing while it waits for a record, so the device waits as long as needed.
        RecordProvider* records = nullptr;
    };
    // One port: the device whose calls use it (its sequence counter lives there) and the caps its
    // channels carry, as in Options.
    struct PortOptions {
        int device          = 0;
        int max_jobs        = 4;
        int max_columns     = 64;
        int pcie_divisor    = 3;
        int max_job_columns = kMaxCpuColumns;
        int wide_from       = 0;
        int wide_jobs       = 0;
    };

    // layers: every layer of the model (channel and request layers are global). From `shared` only
    // workers, cpus and records are used; each port's caps come from `ports` (at least one).
    CpuMissService(std::vector<Layer> layers, Options shared, std::vector<PortOptions> ports);
    // One port on the current device with the caps of `options`.
    CpuMissService(std::vector<Layer> layers, Options options);
    ~CpuMissService();
    CpuMissService(const CpuMissService&)            = delete;
    CpuMissService& operator=(const CpuMissService&) = delete;

    // The channel of global layer `global_layer` for calls on port `port`'s device; either out of
    // range throws std::out_of_range.
    [[nodiscard]] MoeCpuChannel channel(int global_layer, int port = 0) const;
    [[nodiscard]] int port_count() const noexcept { return static_cast<int>(ports_.size()); }
    // Totals over every port.
    [[nodiscard]] std::uint64_t served_requests() const noexcept { return served_.load(std::memory_order_relaxed); }
    [[nodiscard]] std::uint64_t served_experts() const noexcept { return experts_.load(std::memory_order_relaxed); }
    // The experts served for port `port`'s calls (0 for a port out of range).
    [[nodiscard]] std::uint64_t served_experts(int port) const noexcept {
        return port >= 0 && port < port_count() ? port_experts_[static_cast<std::size_t>(port)].load(std::memory_order_relaxed)
                                                : 0;
    }
    // Why the service thread stopped serving; empty while it serves. The device's wait for an
    // answer then times out (kErrorHostSilent) on every later call, so the caller treats a failed
    // service as fatal and reports this cause.
    [[nodiscard]] std::string failure() const;

private:
    struct Port {
        PortOptions options;
        int jobs              = 0;       // kMaxCpuJobs with wide_jobs, else max_jobs
        MissRequest* request  = nullptr; // mapped
        std::uint16_t* x      = nullptr; // mapped BF16 [H, kMaxCpuXColumns]
        std::uint16_t* y      = nullptr; // mapped BF16 [H, jobs * kMaxCpuColumns]
        std::uint32_t* done   = nullptr; // mapped
        std::uint32_t* status = nullptr; // mapped, beside done
        // Mapped, beside status: advanced while the service thread lives.
        std::uint32_t* heartbeat = nullptr;
        std::uint32_t* sequence  = nullptr; // on options.device
    };

    void serve();
    // Frees every port's buffers; once, on destruction or when construction fails.
    void release() noexcept;
    void record_failure(std::string what) noexcept;

    std::vector<Layer> layers_;
    Options options_; // workers, cpus and records
    std::vector<Port> ports_;
    int jobs_ = 0; // the most jobs a port's request may carry
    std::atomic<bool> stop_{false};
    std::atomic<std::uint64_t> served_{0}, experts_{0};
    std::unique_ptr<std::atomic<std::uint64_t>[]> port_experts_; // per port, beside experts_
    mutable std::mutex failure_mutex_;
    std::string failure_;
    std::thread thread_;
};

} // namespace infernix::ops::offloaded_moe
