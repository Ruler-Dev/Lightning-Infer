// The expert residency of one layer-split segment (multi-GPU WP11), on a synthetic 48-layer model
// with 8 experts of 4 KiB records in pinned banks (every word names its layer and expert):
//  - a residency given the banks of layers 24-47 serves layers() == 24; table_bytes and max_frames
//    of 24 layers match it; no banks, or more than the model's layers, are refused;
//  - after a round that routes two experts of every local layer, the host and device frame tables
//    are indexed by local layer (table(l) is row l), every routed key is resident, and each frame
//    holds the record of global layer 24 + l, so promotions copy from banks[local];
//  - its saved state is keyed by its 24 layers; it refuses a whole-model state and warm-starts from
//    split_saved_state's part of one, loading the keys of its layers from the right banks;
//  - with two GPUs: a residency for device 1 constructed while device 0 is current reserves device
//    1's whole memory as VMM address space (not device 0's), allocates its tables on device 1,
//    promotes on device 1 under the segment's guard, and is destroyed with device 0 current without
//    a CUDA error; device 0 stays current throughout.

#include "core/device_guard.h"
#include "core/vmm_arena.h"
#include "core/vram_budget.h"
#include "models/qwen4_exp/config.h"
#include "models/qwen4_exp/program/expert_cache/expert_cache.h"
#include "models/qwen4_exp/program/expert_residency.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using infernix::DeviceGuard;
using infernix::models::qwen4_exp::ExpertResidency;
using infernix::models::qwen4_exp::TextConfig;
using infernix::models::qwen4_exp::expert_cache::SavedState;
using infernix::models::qwen4_exp::expert_cache::split_saved_state;

constexpr std::uint32_t kLayers  = 48;
constexpr std::uint32_t kExperts = 8;
constexpr std::uint32_t kTopK    = 2;
constexpr std::uint32_t kFirst   = 24; // the segment's first layer
constexpr std::uint32_t kOwned   = kLayers - kFirst;
constexpr std::uint64_t kStride  = 4096;
constexpr std::int32_t kColumns  = 4;

int g_failures = 0;

void expect(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << '\n';
        ++g_failures;
    }
}

int current() {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    return device;
}

template <class Exception, class Call>
bool throws(Call&& call) {
    try {
        call();
    } catch (const Exception&) { return true; } catch (...) {
        return false;
    }
    return false;
}

TextConfig model() {
    TextConfig c;
    c.num_hidden_layers = kLayers;
    c.moe.experts       = kExperts;
    c.moe.top_k         = kTopK;
    return c;
}

std::uint32_t word_of(std::uint32_t layer, std::uint32_t expert, std::size_t i) {
    return (layer << 24) | (expert << 16) | static_cast<std::uint32_t>(i & 0xFFFFU);
}

// Every layer's pinned records, as the artifact's banks.
class Banks {
public:
    Banks() {
        const std::size_t bytes = static_cast<std::size_t>(kLayers) * kExperts * kStride;
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&records_), bytes));
        auto* words = reinterpret_cast<std::uint32_t*>(records_);
        for (std::uint32_t layer = 0; layer < kLayers; ++layer) {
            for (std::uint32_t expert = 0; expert < kExperts; ++expert) {
                const std::size_t first = (static_cast<std::size_t>(layer) * kExperts + expert) * kStride / 4;
                for (std::size_t i = 0; i < kStride / 4; ++i) { words[first + i] = word_of(layer, expert, i); }
            }
        }
    }
    ~Banks() { (void)cudaFreeHost(records_); }
    Banks(const Banks&)            = delete;
    Banks& operator=(const Banks&) = delete;

    // The banks of layers [begin, end), in local order.
    [[nodiscard]] std::vector<const std::uint8_t*> layers(std::uint32_t begin, std::uint32_t end) const {
        std::vector<const std::uint8_t*> out;
        for (std::uint32_t layer = begin; layer < end; ++layer) {
            out.push_back(records_ + static_cast<std::size_t>(layer) * kExperts * kStride);
        }
        return out;
    }

private:
    std::uint8_t* records_ = nullptr;
};

// Free memory that drops by exactly what the residency's frames occupy, so a resize sees no spill.
class LedgerSource final : public infernix::VramBudgetSource {
public:
    explicit LedgerSource(const ExpertResidency& residency) : residency_(residency) {}
    [[nodiscard]] infernix::VramSnapshot query() override {
        infernix::VramSnapshot snapshot;
        snapshot.device_total = 1ULL << 40;
        snapshot.device_free  = snapshot.device_total - residency_.pool_bytes();
        return snapshot;
    }

private:
    const ExpertResidency& residency_;
};

// Whether `frame` of `residency` holds global layer `layer`'s expert `expert` (its device current).
bool frame_holds(const ExpertResidency& residency, std::int32_t frame, std::uint32_t layer, std::uint32_t expert) {
    if (frame < 0 || static_cast<std::uint32_t>(frame) >= residency.frames()) { return false; }
    std::vector<std::uint32_t> words(kStride / 4);
    CUDA_CHECK(cudaMemcpy(words.data(), residency.frame_base() + static_cast<std::size_t>(frame) * kStride, kStride,
                          cudaMemcpyDeviceToHost));
    for (std::size_t i = 0; i < words.size(); ++i) {
        if (words[i] != word_of(layer, expert, i)) { return false; }
    }
    return true;
}

// Local layer l's experts in the round: 3l and 3l + 1, mod 8.
bool routed(std::uint32_t local, std::uint32_t expert) {
    return expert == (3 * local) % kExperts || expert == (3 * local + 1) % kExperts;
}

// One round with one column routing routed()'s experts of every local layer, run to the point where
// its promotions have landed and the next before_round published them (the residency's device
// current, `stream` on it).
void run_round(ExpertResidency& residency, cudaStream_t stream) {
    std::vector<std::int32_t> routes(residency.route_stride() * residency.layers(), -1);
    for (std::uint32_t local = 0; local < residency.layers(); ++local) {
        routes[local * residency.route_stride()]     = static_cast<std::int32_t>((3 * local) % kExperts);
        routes[local * residency.route_stride() + 1] = static_cast<std::int32_t>((3 * local + 1) % kExperts);
    }
    CUDA_CHECK(cudaMemcpy(residency.route_log(), routes.data(), routes.size() * sizeof(std::int32_t),
                          cudaMemcpyHostToDevice));
    residency.before_round(stream);
    residency.enqueue_route_download(stream, 1);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    residency.after_round(stream, 1, kExperts);
    CUDA_CHECK(cudaDeviceSynchronize()); // the copy stream's promotions
    residency.before_round(stream);      // publishes them and uploads the table
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

// After run_round: the tables are indexed by local layer and every routed key's frame holds the
// record of global layer kFirst + l, copied from banks[l].
void check_promotions(const ExpertResidency& residency, const std::string& where) {
    expect(residency.stats().promotions == 2ULL * kOwned, where + ": one promotion per routed key");
    const std::size_t keys = static_cast<std::size_t>(kOwned) * kExperts;
    std::vector<std::int32_t> device_table(keys);
    CUDA_CHECK(cudaMemcpy(device_table.data(), residency.table(0), keys * sizeof(std::int32_t),
                          cudaMemcpyDeviceToHost));
    const std::int32_t* host_table = residency.host_table();
    expect(std::vector<std::int32_t>(host_table, host_table + keys) == device_table,
           where + ": the device tables equal the host tables");
    expect(residency.table(kOwned - 1) == residency.table(0) + static_cast<std::size_t>(kOwned - 1) * kExperts,
           where + ": table(l) is local row l");
    bool resident = true, sourced = true;
    for (std::uint32_t local = 0; local < kOwned; ++local) {
        for (std::uint32_t expert = 0; expert < kExperts; ++expert) {
            const std::int32_t frame = host_table[local * kExperts + expert];
            resident                 = resident && (frame >= 0) == routed(local, expert);
            if (frame >= 0) { sourced = sourced && frame_holds(residency, frame, kFirst + local, expert); }
        }
    }
    expect(resident, where + ": exactly the routed keys of the local layers are resident");
    expect(sourced, where + ": each frame holds the record of global layer 24 + l (banks[local])");
}

void test_layer_subset(const Banks& banks, int device) {
    const TextConfig c = model();
    expect(throws<std::invalid_argument>([&] { ExpertResidency none(c, {}, kStride, kColumns, device); }),
           "a residency without banks is refused");
    std::vector<const std::uint8_t*> too_many = banks.layers(0, kLayers);
    too_many.push_back(too_many.back());
    expect(throws<std::invalid_argument>([&] { ExpertResidency more(c, too_many, kStride, kColumns, device); }),
           "more banks than the model's layers are refused");
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    {
        ExpertResidency residency(c, banks.layers(kFirst, kLayers), kStride, kColumns, device);
        expect(residency.layers() == kOwned, "a residency over layers 24-47 serves 24 layers");
        expect(residency.device() == device, "the residency records its device");
        expect(ExpertResidency::max_frames(c, kOwned) == kOwned * kExperts - 1 &&
                   ExpertResidency::max_frames(c) == kLayers * kExperts - 1,
               "max_frames counts the given layers (the whole model without them)");
        // Per layer: the frame table, the route log and the landing table and log.
        const std::uint64_t per_layer = kExperts + kTopK * static_cast<std::uint64_t>(kColumns) +
                                        2ULL * ExpertResidency::kLandingSlots;
        expect(ExpertResidency::table_bytes(c, kColumns, kOwned) == kOwned * per_layer * sizeof(std::int32_t) &&
                   ExpertResidency::table_bytes(c, kColumns) == ExpertResidency::table_bytes(c, kColumns, kLayers),
               "table_bytes counts the given layers (the whole model without them)");
        expect(residency.frame_limit() <= ExpertResidency::max_frames(c, kOwned),
               "the frame limit is at most every routed expert of its layers but one");
        expect(residency.route_stride() == static_cast<std::size_t>(kTopK) * kColumns, "route stride is top_k * columns");
        LedgerSource ledger(residency);
        expect(residency.resize(64, stream, ledger).frames == 64, "the subset's cache backs 64 frames");
        run_round(residency, stream);
        check_promotions(residency, "layer subset");
        const SavedState state = residency.saved_state();
        bool counts            = state.counts.size() == static_cast<std::size_t>(kOwned) * kExperts;
        for (std::uint32_t key = 0; counts && key < kOwned * kExperts; ++key) {
            counts = state.counts[key] == (routed(key / kExperts, key % kExperts) ? 1U : 0U);
        }
        expect(counts, "the saved state counts the residency's own keys");
        bool local = state.ranked.size() == 2ULL * kOwned;
        for (const std::uint32_t key : state.ranked) { local = local && key < kOwned * kExperts; }
        expect(local, "the saved state ranks local keys");
    }
    {
        ExpertResidency residency(c, banks.layers(kFirst, kLayers), kStride, kColumns, device);
        LedgerSource ledger(residency);
        (void)residency.resize(16, stream, ledger);
        // Global keys of layers 30, 2, 47, 24 and 10; the first, third and fourth are the segment's.
        const std::uint32_t picks[][2] = {{30, 5}, {2, 1}, {47, 7}, {24, 0}, {10, 3}};
        SavedState global;
        global.counts.assign(static_cast<std::size_t>(kLayers) * kExperts, 0);
        for (const auto& pick : picks) {
            global.ranked.push_back(pick[0] * kExperts + pick[1]);
            global.counts[pick[0] * kExperts + pick[1]] = 4;
        }
        expect(throws<std::invalid_argument>([&] { (void)residency.warm_start(global, 16, 16, stream); }),
               "a whole-model state is refused by a layer subset");
        const SavedState part      = split_saved_state(global, kFirst, kLayers, kExperts);
        const std::uint32_t loaded = residency.warm_start(part, 16, 16, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        expect(loaded == 3, "the warm start loads the segment's three ranked keys");
        bool seeded = true;
        for (const auto& pick : picks) {
            if (pick[0] < kFirst) { continue; }
            const std::int32_t frame = residency.host_table()[(pick[0] - kFirst) * kExperts + pick[1]];
            seeded                   = seeded && frame_holds(residency, frame, pick[0], pick[1]);
        }
        expect(seeded, "seeded frames hold the records of their global layers");
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
}

void test_other_device(const Banks& banks, int count) {
    if (count < 2) { return; }
    CUDA_CHECK(cudaSetDevice(0));
    std::size_t free0 = 0, total0 = 0, free1 = 0, total1 = 0;
    CUDA_CHECK(cudaMemGetInfo(&free0, &total0));
    {
        const DeviceGuard on_device(1);
        CUDA_CHECK(cudaMemGetInfo(&free1, &total1));
    }
    const TextConfig c = model();
    {
        ExpertResidency residency(c, banks.layers(kFirst, kLayers), kStride, kColumns, 1);
        expect(current() == 0, "constructing a residency for device 1 leaves device 0 current");
        expect(residency.device() == 1, "the residency records device 1");
        if (infernix::VmmArena::supported(1)) {
            const std::uint64_t chunk = ExpertResidency::kChunkBytes;
            const std::uint64_t want  = (total1 + chunk - 1) / chunk * chunk;
            expect(residency.reserved_bytes() == want,
                   "the VMM reservation on device 1 is device 1's total memory while device 0 is current");
            std::cout << "VMM reservation for device 1: " << (residency.reserved_bytes() >> 20) << " MiB (device 0 "
                      << (total0 >> 20) << " MiB, device 1 " << (total1 >> 20) << " MiB"
                      << (total0 == total1 ? "; equal totals, so this cannot tell the devices apart" : "") << ")\n";
        } else {
            std::cout << "device 1 lacks VMM: no reservation to check\n";
        }
        cudaPointerAttributes attributes{};
        CUDA_CHECK(cudaPointerGetAttributes(&attributes, residency.table(0)));
        expect(attributes.device == 1, "the frame tables are allocated on device 1");
        CUDA_CHECK(cudaPointerGetAttributes(&attributes, residency.route_log()));
        expect(attributes.device == 1, "the route log is allocated on device 1");
        {
            const DeviceGuard on_device(1); // the segment's guard
            cudaStream_t stream = nullptr;
            CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
            LedgerSource ledger(residency);
            expect(residency.resize(64, stream, ledger).frames == 64, "device 1's cache backs 64 frames");
            run_round(residency, stream);
            check_promotions(residency, "device 1");
            CUDA_CHECK(cudaStreamDestroy(stream));
        }
        expect(current() == 0, "device 0 is current again after the segment's guard");
    }
    expect(current() == 0, "destroying the residency leaves device 0 current");
    const cudaError_t error = cudaGetLastError();
    expect(error == cudaSuccess, std::string("destroying the residency with device 0 current leaves no CUDA error (") +
                                     cudaGetErrorName(error) + ")");
}

} // namespace

int main() {
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
        const Banks banks;
        test_layer_subset(banks, 0);
        test_other_device(banks, count);
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
    if (g_failures != 0) {
        std::cerr << g_failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "expert residency split: all checks passed on " << count << " device(s)"
              << (count < 2 ? " (one device: the cross-device reservation check needs two)" : "") << '\n';
    return 0;
}
