// The Program side of Qwen4Exp's Vision encode window (docs/maintainer/qwen3_8-flash-next-design.md
// §19.3.2): admission plans the window, reserve keeps the media, progress runs the layer-major pass
// in steps, prefill calls scatter the handoff's embeddings, and the handoff returns to the expert
// cache once the prompt's last call is enqueued (or the lane is released).
//
// On a layer split ("On a layer split" in §19.3.2) the tower runs on one segment (place_vision),
// whose cache lends the window and the handoff and whose stream encodes; the first segment's scatter
// and the head's MTP prompt cells read the handoff, each segment on another GPU from its own copy,
// lent with the handoff (vision_lend) and filled once through a pinned crossing after the encode
// (vision_mirror). Segments on the tower's GPU read the handoff in place: their work never overlaps
// the tower's there (SegmentWork). The leases start their pools and, with several media prompts in
// flight, wait rather than split a pool (vision_lend).

#include "models/qwen4_exp/program/program_impl.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace infernix::models::qwen4_exp::detail {

namespace vx = qwen3_5::execution;

namespace {

// The handoff crossing's pinned ring: a 1,024-token image's handoff (5 MiB) in two pieces.
constexpr std::size_t kVisionPieceBytes = 4ULL << 20;
constexpr std::uint32_t kVisionPieces  = 4;

// A Vision lend that would split a pool waits for another lane's leases while the lanes' prefill
// makes steps; after this many polls without one it lends anyway (the stream leases are then sized
// by the longest run, ProgramImpl::stream_chunk), so a lane whose prefill cannot run during a
// binding never holds the encode. One poll per engine boundary, up to three per engine cycle.
constexpr std::uint32_t kVisionWaitPolls = 256;

// INFERNIX_SPLIT_VISION (a measurement knob, a split only): auto (the default) places the tower by
// vision_tower_segment, first and last force the first or the last segment.
enum class VisionPlacement : std::uint8_t { Auto, First, Last };

VisionPlacement vision_placement_from_environment() {
    const char* value = std::getenv("INFERNIX_SPLIT_VISION");
    if (value == nullptr || *value == '\0' || std::strcmp(value, "auto") == 0) { return VisionPlacement::Auto; }
    if (std::strcmp(value, "first") == 0) { return VisionPlacement::First; }
    if (std::strcmp(value, "last") == 0) { return VisionPlacement::Last; }
    throw std::invalid_argument(std::string("INFERNIX_SPLIT_VISION must be auto, first or last, not '") + value + "'");
}

} // namespace

void ProgramImpl::place_vision() {
    vision_segment_ = 0;
    vision_bulk_.clear();
    if (!parameters_.vision || segments_.size() < 2) { return; }
    // An offloaded tower's weights are portable pinned memory any segment's GPU streams; a resident
    // one lives in the first segment's arena (the load places it with the embedding).
    const bool offload = parameters_.model.overlay_vision().has_value();
    std::vector<double> rates;
    for (const auto& segment : segments_) { rates.push_back(segment->policy.bytes_per_second); }
    vision_segment_ = vision_tower_segment(rates, offload);
    switch (vision_placement_from_environment()) {
    case VisionPlacement::Auto:
        break;
    case VisionPlacement::First:
        vision_segment_ = 0;
        break;
    case VisionPlacement::Last:
        if (!offload) {
            throw std::invalid_argument("INFERNIX_SPLIT_VISION=last needs the vision offload (a resident tower runs "
                                        "where the load placed it, on the first segment)");
        }
        vision_segment_ = segments_.size() - 1;
        break;
    }
    const DeviceSegment& tower = vision_seg();
    // The readers on another GPU: the first segment (the scatter) and the head with the MTP drafter
    // (its prompt cells take the next token's embedding).
    vision_bulk_.resize(segments_.size());
    std::string readers;
    for (const DeviceSegment* reader : {&first(), mtp_ ? &head() : nullptr}) {
        if (reader == nullptr || reader->device->device == tower.device->device || vision_bulk_[reader->spec.index]) {
            continue;
        }
        vision_bulk_[reader->spec.index] = std::make_unique<BulkHandoff>(tower.device->device, reader->device->device,
                                                                         kVisionPieceBytes, kVisionPieces);
        readers += (readers.empty() ? "" : ", ") + std::string("segment ") + std::to_string(reader->spec.index);
    }
    char rate[32];
    std::snprintf(rate, sizeof(rate), "%.1f GB/s", tower.policy.bytes_per_second / 1e9);
    diagnostic(segment_name(tower) + "runs the Vision tower (" +
                   (offload ? std::string("weights streamed from pinned RAM over its ") + rate + " link"
                            : std::string("resident weights")) +
                   (readers.empty() ? std::string("; every reader on its GPU)")
                                    : "; the handoff is copied to " + readers + " through pinned memory)"),
               DiagnosticLevel::Debug);
}

bool ProgramImpl::vision_fits(const VisionWindowPlan& window) const noexcept {
    const auto lendable = [](const DeviceSegment& seg) {
        return seg.residency->frames() - seg.residency->frames() / 4;
    };
    if (window.frames > lendable(vision_seg())) { return false; }
    if (window.items.empty()) { return true; }
    for (std::size_t s = 0; s < vision_bulk_.size(); ++s) {
        if (!vision_bulk_[s]) { continue; }
        const DeviceSegment& reader = *segments_[s];
        const std::uint64_t stride  = reader.residency->frame_stride();
        if ((window.handoff_bytes + stride - 1) / stride > lendable(reader)) { return false; }
    }
    return true;
}

std::shared_ptr<const VisionAdmission> ProgramImpl::plan_vision(const qwen3_5::PreparedPromptData& prompt,
                                                                std::uint32_t reused) const {
    if (!parameters_.vision || !parameters_.model.config().vision) {
        throw std::invalid_argument("Qwen4Exp: images and video need an engine started with --vision");
    }
    const qwen3_5::VisionConfig& config = *parameters_.model.config().vision;
    auto out     = std::make_shared<VisionAdmission>();
    out->control = qwen3_5::plan_vision_control(prompt, config);
    std::vector<VisionWindowItem> items;
    items.reserve(out->control.items.size());
    for (std::size_t i = 0; i < out->control.items.size(); ++i) {
        const auto& plan = out->control.items[i];
        items.push_back({.token_begin = plan.token_begin,
                         .token_end   = plan.token_end,
                         .patches     = static_cast<std::uint32_t>(prompt.vision_items[i].patch_count),
                         .merged      = static_cast<std::uint32_t>(plan.merged_count)});
    }
    // The pass encodes only the items ending past the reused prefix.
    std::vector<std::size_t> patches;
    for (const auto& item : items) {
        if (item.token_end > reused) { patches.push_back(item.patches); }
    }
    const std::size_t pass_bytes = patches.empty() ? 0 : vx::plan_vision_pass(config, *parameters_.vision, patches).bytes;
    const auto& overlay          = parameters_.model.overlay_vision();
    const std::size_t staging    = overlay ? vx::vision_staging_align(overlay->layout.staging_bytes) : 0;
    const DeviceSegment& tower   = vision_seg();
    out->window = plan_vision_window(items, reused, pass_bytes, staging, tower.work_capacity,
                                     tower.residency->frame_stride(),
                                     static_cast<std::uint32_t>(parameters_.vision->merger_fc2.weight.n),
                                     config.hidden_size, config.depth);
    return out;
}

void ProgramImpl::vision_reserve(Lane& lane, std::shared_ptr<const VisionAdmission> admission,
                                 const qwen3_5::PreparedPromptData& prompt) {
    auto vision        = std::make_unique<LaneVision>();
    const auto& window = admission->window;
    if (window.items.empty()) {
        vision->encoded = true; // every item lies in the reused prefix
    } else {
        // The encoded items are a suffix of the prompt's items (those ending past the reused prefix).
        const std::uint32_t first = window.items.front();
        vision->control           = qwen3_5::build_vision_control(prompt, admission->control, first);
        for (std::size_t k = 0; k < vision->control.items.size(); ++k) {
            const auto& payload = prompt.media_payloads.at(first + k);
            const auto& control = vision->control.items[k];
            if (!payload || payload->patch_elements !=
                                control.patch_count * static_cast<std::size_t>(parameters_.model.config().vision->patch_width())) {
                throw std::invalid_argument("Qwen4Exp: a Vision item's patches do not match its grid");
            }
            vision->payloads.push_back(payload);
            for (const std::int32_t token : control.scatter_indices) {
                vision->visual.push_back(static_cast<std::uint32_t>(token));
            }
        }
        if (!std::is_sorted(vision->visual.begin(), vision->visual.end()) ||
            vision->visual.size() != window.encoded_tokens) {
            throw std::logic_error("Qwen4Exp: the encoded visual tokens disagree with the window plan");
        }
    }
    vision->admission = std::move(admission);
    lane.vision       = std::move(vision);
}

// Lends the encode's leases, each at the bottom of its pool (choose_run at_start): the tower's
// handoff (W) or handoff and window (L), and each reading segment's copy on another GPU. The handoff
// and the copies stay lent through the prompt's calls, whose stream lease may need most of a pool's
// frames in one run (on a 16 GB card, alone or a split's head, more than half of them): a lease that
// would leave the frames still lendable split (lend_fit: another prompt's stream lease in the way,
// or a hole a released handoff left) waits while another lane prefills, whose calls return its
// leases, once the stream and wide leases went back (between calls nothing reads them; a layer
// walk's span holds them). Nothing is lent while it waits.
bool ProgramImpl::vision_lend(LaneVision& v, std::uint32_t index) {
    using LendFit                  = ExpertResidency::LendFit;
    const VisionWindowPlan& window = v.admission->window;
    DeviceSegment& tower           = vision_seg();
    const std::size_t bytes =
        2ULL * static_cast<std::size_t>(parameters_.vision->merger_fc2.weight.n) * window.encoded_tokens;
    struct Want {
        std::size_t segment;
        std::uint32_t frames;
    };
    std::vector<Want> wants{{vision_segment_, window.frames}};
    for (std::size_t s = 0; s < vision_bulk_.size(); ++s) {
        if (!vision_bulk_[s]) { continue; }
        const std::uint64_t stride = segments_[s]->residency->frame_stride();
        wants.push_back({s, static_cast<std::uint32_t>((bytes + stride - 1) / stride)});
    }
    // The worst fit over the pools, their stream and wide leases counted as coming back.
    const auto fit = [&] {
        LendFit worst = LendFit::Keeps;
        for (const Want& want : wants) {
            DeviceSegment& seg = *segments_[want.segment];
            const DeviceGuard on(seg.device->device);
            const ExpertResidency::FrameLease transient[] = {seg.stream_lease, seg.work_lease};
            worst = std::min(worst, seg.residency->lend_fit(want.frames, seg.device->stream, transient));
        }
        return worst;
    };
    LendFit worst = fit();
    if (worst != LendFit::Keeps && !walk_.span.active) {
        for (const Want& want : wants) {
            DeviceSegment& seg = *segments_[want.segment];
            const DeviceGuard on(seg.device->device);
            return_stream_lease(seg); // the next streamed chunk lends it again
            return_wide_work(seg);
        }
        worst = fit();
    }
    if (worst != LendFit::Keeps && prefilling_besides(index)) {
        if (prefill_steps_ != v.wait_steps) {
            v.wait_steps = prefill_steps_;
            v.wait_polls = 0;
        } else {
            ++v.wait_polls;
        }
        if (!v.wait_reported) {
            diagnostic(segment_name(tower) + "the Vision encode waits for frames: " +
                           (worst == LendFit::None ? std::string("no run holds its leases")
                                                   : std::string("its leases would split a pool")) +
                           " while another lane's prompt holds leases",
                       DiagnosticLevel::Debug);
            v.wait_reported = true;
        }
        if (worst == LendFit::None) {
            if (v.wait_polls == kVisionWaitPolls) {
                diagnostic(segment_name(tower) + "the Vision encode still waits for frames another lane's prompt "
                                                 "holds, and no prefill step ran for " +
                               std::to_string(kVisionWaitPolls) + " polls",
                           DiagnosticLevel::Warning);
            }
            return false; // no lease can be made before they come back
        }
        if (v.wait_polls < kVisionWaitPolls) { return false; }
    }
    if (worst == LendFit::Splits) {
        diagnostic(segment_name(tower) + "the Vision leases are lent although they split a pool (no other lane's "
                                         "prefill step came); stream leases take the longest run meanwhile",
                   DiagnosticLevel::Debug);
    }
    // LendFit::None with no other lane prefilling (nothing else is lent then: the frames shrank since
    // admission): lend throws, as it always did.
    const cudaStream_t writers[] = {tower.device->transfer_stream};
    v.lease = tower.residency->lend(window.frames, tower.device->stream, writers, /*at_start=*/true);
    for (std::size_t k = 1; k < wants.size(); ++k) {
        DeviceSegment& reader = *segments_[wants[k].segment];
        LaneVision::Mirror mirror{.segment = wants[k].segment};
        {
            // Its upload (vision_mirror) runs on the reader's stream, which the lease makes wait for
            // promotions still landing in the run; the reader's calls run after it.
            const DeviceGuard on(reader.device->device);
            mirror.lease = reader.residency->lend(wants[k].frames, reader.device->stream, {}, /*at_start=*/true);
        }
        v.mirrors.push_back(mirror); // released with the lane whatever follows
    }
    return true;
}

bool ProgramImpl::vision_step(Lane& lane, std::uint32_t index) {
    LaneVision& v                        = *lane.vision;
    const VisionWindowPlan& window       = v.admission->window;
    const qwen3_5::VisionConfig& config  = *parameters_.model.config().vision;
    DeviceSegment& tower                 = vision_seg();
    // The tower's device current; on a GPU that runs two segments, after the other one's work.
    const SegmentWork on(*this, tower);
    const cudaStream_t s                 = tower.device->stream;
    const auto out                       = parameters_.vision->merger_fc2.weight.n;
    if (!v.pass) {
        // The handoff (W) or handoff and window (L), the handoff first so the window's frames rejoin
        // the rest when they go back; promotions still landing in the run are waited for by compute
        // and the weight stream's transfer stream.
        if (!vision_lend(v, index)) { return false; }
        v.handoff = Tensor(v.lease.memory.data, DType::BF16, {out, static_cast<std::int32_t>(window.encoded_tokens)});
        DeviceSpan memory;
        if (window.lease) {
            const std::size_t at = align(window.handoff_bytes);
            memory = {static_cast<std::byte*>(v.lease.memory.data) + at, v.lease.memory.bytes - at};
        } else {
            // Placement W: the whole pass runs in this call from the prefill workspace, idle between
            // units and stream-ordered on compute.
            tower.work->reset();
            memory = tower.work->alloc_bytes(window.window_bytes);
        }
        const vx::VisionParameters* weights = &*parameters_.vision;
        std::size_t staging                 = 0;
        if (const auto& overlay = parameters_.model.overlay_vision()) {
            staging   = vx::vision_staging_align(overlay->layout.staging_bytes);
            v.stream  = std::make_unique<vx::VisionWeightStream>(*tower.device, *overlay,
                                                                 static_cast<std::byte*>(memory.data));
            v.weights = v.stream->window_weights(*parameters_.vision);
            weights   = &*v.weights;
        }
        const DeviceSpan arena{static_cast<std::byte*>(memory.data) + staging, memory.bytes - staging};
        std::vector<std::size_t> patches;
        std::vector<vx::VisionTowerItem> items;
        std::size_t column = 0;
        for (std::size_t k = 0; k < v.control.items.size(); ++k) {
            const auto& control = v.control.items[k];
            patches.push_back(control.patch_count);
            const Tensor output(static_cast<std::byte*>(v.handoff.data) + 2ULL * out * column, DType::BF16,
                                {out, static_cast<std::int32_t>(control.merged_count)});
            items.push_back({.patches = v.payloads[k]->span(), .control = &control, .output = output});
            column += control.merged_count;
        }
        v.layout = std::make_unique<vx::VisionPassLayout>(vx::plan_vision_pass(config, *parameters_.vision, patches));
        if (v.layout->bytes > arena.bytes) { throw std::logic_error("Qwen4Exp: the Vision pass exceeds its window"); }
        v.timer = std::make_unique<VisionTimer>();
        CUDA_CHECK(cudaEventCreate(&v.timer->start));
        CUDA_CHECK(cudaEventCreate(&v.timer->stop));
        CUDA_CHECK(cudaEventRecord(v.timer->start, s));
        v.pass = std::make_unique<vx::VisionTowerPass>(*tower.device, config, *weights, std::move(items), arena,
                                                       *v.layout, v.stream.get());
    }
    // Placement W: every stage now. L: `layers_per_step` layers per call; the embedding rides with the
    // first step and the merger with the last.
    const std::uint32_t stages = vx::VisionTowerPass::stages(config.depth);
    std::uint32_t end          = stages;
    if (window.lease) {
        const std::uint32_t next = v.pass->next_stage();
        end                      = (next == 0 ? 1 : next) + window.layers_per_step;
        if (end > config.depth) { end = stages; }
    }
    v.pass->advance(end);
    if (!v.pass->done()) { return true; }
    if (v.stream) { v.stream->finish(s); }
    CUDA_CHECK(cudaEventRecord(v.timer->stop, s));
    if (window.lease) {
        // Only the handoff stays lent; the window's frames go back once compute reaches here.
        const std::uint64_t stride = tower.residency->frame_stride();
        const auto handoff_frames  = static_cast<std::uint32_t>((align(window.handoff_bytes) + stride - 1) / stride);
        if (handoff_frames < v.lease.count) {
            ExpertResidency::FrameLease tail{.first = v.lease.first + handoff_frames, .count = v.lease.count - handoff_frames};
            tower.residency->give_back(tail, s);
            v.lease.count = handoff_frames;
        }
    }
    v.pass.reset();
    v.layout.reset();
    v.weights.reset();
    v.stream.reset();
    v.payloads.clear(); // the media payloads are not needed again
    vision_mirror(v);
    v.encoded = true;
    return true;
}

void ProgramImpl::vision_mirror(LaneVision& v) {
    DeviceSegment& tower    = vision_seg();
    const std::size_t bytes = v.handoff.bytes();
    for (const LaneVision::Mirror& mirror : v.mirrors) {
        DeviceSegment& reader = *segments_[mirror.segment];
        vision_bulk_[mirror.segment]->transfer(v.handoff.data, mirror.lease.memory.data, bytes, tower.device->stream,
                                               reader.device->stream);
        const DeviceGuard on(reader.device->device);
        reader.device->flush();
    }
    if (!v.mirrors.empty()) {
        tower.device->flush();
        diagnostic(segment_name(tower) + "Vision handoff of " + std::to_string(v.handoff.ne[1]) + " tokens (" +
                       std::to_string(bytes >> 10) + " KiB) copied to " + std::to_string(v.mirrors.size()) +
                       " segment(s) on another GPU",
                   DiagnosticLevel::Debug);
    }
}

std::byte* ProgramImpl::vision_memory(const LaneVision& v, const DeviceSegment& seg) const {
    for (const auto& mirror : v.mirrors) {
        if (mirror.segment == seg.spec.index) { return static_cast<std::byte*>(mirror.lease.memory.data); }
    }
    if (seg.device->device != vision_seg().device->device) {
        throw std::logic_error("Qwen4Exp: a segment on another GPU than the Vision tower has no copy of the handoff");
    }
    return static_cast<std::byte*>(v.handoff.data);
}

// Called once the prompt's last call is enqueued and before the prompt's stream leases go back (the
// callers' order): a slow link's cache refills the frames its stream lease held (~2 GB through the
// RTX PRO 4000's x4 link, ~0.6 s), and with this give-back after that refill was queued, the first
// token's synchronization of that GPU ended with the refill's last copy (an nsys trace; image TTFT
// +0.64 s on the split).
void ProgramImpl::vision_release(Lane& lane) noexcept {
    try {
        LaneVision& v        = *lane.vision;
        DeviceSegment& tower = vision_seg();
        {
            // On a GPU that runs two segments, the other one's calls read the handoff in place: the
            // tower's stream follows them before its frames go back.
            const SegmentWork on(*this, tower);
            const cudaStream_t s = tower.device->stream;
            // Compute is ordered after every upload into the window before its memory goes back.
            if (v.stream) { v.stream->finish(s); }
            // The prompt's last call may be enqueued right behind the encode (one call, no wait).
            float ms = 0.0F;
            if (v.encoded && v.timer && cudaEventSynchronize(v.timer->stop) == cudaSuccess &&
                cudaEventElapsedTime(&ms, v.timer->start, v.timer->stop) == cudaSuccess) {
                lane.vision_seconds = ms * 1e-3;
            }
            v.pass.reset();
            v.layout.reset();
            v.weights.reset();
            v.stream.reset();
            v.timer.reset();
            if (v.lease.valid()) { tower.residency->give_back(v.lease, s); }
        }
        // A copy's last reader and its upload ran on the reader's stream.
        for (LaneVision::Mirror& mirror : v.mirrors) {
            DeviceSegment& reader = *segments_[mirror.segment];
            const DeviceGuard on(reader.device->device);
            if (mirror.lease.valid()) { reader.residency->give_back(mirror.lease, reader.device->stream); }
        }
    } catch (...) {}
    lane.vision.reset();
}

const execution::VisionInput* ProgramImpl::stage_vision(Lane& lane, std::uint32_t begin, std::int32_t width) {
    if (!lane.vision || lane.vision->visual.empty()) { return nullptr; }
    const LaneVision& v = *lane.vision;
    auto* words         = reinterpret_cast<std::int32_t*>(host_io() + io_layout_.vision_columns);
    const VisualSlice slice =
        visual_slice(v.visual, begin, begin + static_cast<std::uint32_t>(width), begin,
                     std::span<std::int32_t>(words, static_cast<std::size_t>(columns_)));
    if (slice.count == 0) { return nullptr; }
    const std::int32_t out = v.handoff.ne[0];
    auto* base             = static_cast<std::byte*>(first().io_device.p);
    vision_input_          = {Tensor(vision_memory(v, first()) + 2ULL * out * slice.first, DType::BF16,
                                     {out, static_cast<std::int32_t>(slice.count)}),
                              Tensor(base + io_layout_.vision_columns, DType::I32, {static_cast<std::int32_t>(slice.count)})};
    return &vision_input_;
}

const execution::VisionInput* ProgramImpl::stage_mtp_vision(Lane& lane, std::uint32_t first, std::int32_t cells,
                                                            std::int32_t subchunks) {
    if (!lane.vision || lane.vision->visual.empty() || cells <= 0) { return nullptr; }
    const LaneVision& v    = *lane.vision;
    const std::int32_t out = v.handoff.ne[0];
    auto* words            = reinterpret_cast<std::int32_t*>(host_io() + io_layout_.mtp_vision_columns);
    auto* base             = static_cast<std::byte*>(head().io_device.p);
    std::byte* handoff     = vision_memory(v, head());
    mtp_vision_inputs_.assign(static_cast<std::size_t>(subchunks), execution::VisionInput{});
    for (std::int32_t k = 0; k < subchunks; ++k) {
        // Cell c of sub-chunk k sits at first + 512k + c and takes token first + 512k + c + 1.
        const std::int32_t n    = std::min(execution::kMtpChunkColumns, cells - k * execution::kMtpChunkColumns);
        const std::uint32_t lo  = first + static_cast<std::uint32_t>(k * execution::kMtpChunkColumns) + 1U;
        const std::size_t at    = static_cast<std::size_t>(k) * execution::kMtpChunkColumns;
        const VisualSlice slice = visual_slice(v.visual, lo, lo + static_cast<std::uint32_t>(n), lo,
                                               std::span<std::int32_t>(words + at, static_cast<std::size_t>(n)));
        if (slice.count == 0) { continue; }
        mtp_vision_inputs_[static_cast<std::size_t>(k)] = {
            Tensor(handoff + 2ULL * out * slice.first, DType::BF16, {out, static_cast<std::int32_t>(slice.count)}),
            Tensor(base + io_layout_.mtp_vision_columns + 4ULL * at, DType::I32, {static_cast<std::int32_t>(slice.count)})};
    }
    return mtp_vision_inputs_.data();
}

} // namespace infernix::models::qwen4_exp::detail
