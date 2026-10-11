// The layer walk of Qwen4Exp's prefill (docs/maintainer/qwen3_8-flash-next-design.md §19.3.8 F4):
// a span of consecutive streamed calls of one prompt runs layer-major, so each decoder layer's
// non-resident experts cross PCIe once per span instead of once per call. A chunk-major prompt of
// 4,096-token calls re-streams nearly every non-resident expert for every call (~51 GB at the x8
// link's 27.6 GB/s, ~1.8 s per call), more than the call's compute; a span of 16 calls streams them
// once and is bound by compute.
//
// The span keeps the prompt's call grid: chunk c of layer l reads exactly what the chunk-major
// call c reads at layer l (its own residual columns, and the layer's KV, GDN, PLE and QSA state
// after chunks 0..c-1), so every output bit equals the chunk-major prefill's. Only placement
// differs (resident, streamed or staged experts), which the MoE's arithmetic does not see.
//
// The span's residual stream (BF16 [S*H, tokens]) and every chunk's staged io image live in the
// stream lease after the ring. Each advance_prefill call enqueues about one call's work (a few
// layers of every chunk) and synchronizes, so cancellation and other lanes interleave as between
// chunk-major calls; the walk's own state stays valid across them. A span ends where a prefix tap
// is realized (the captured state must be the boundary's in every layer), before a call too narrow
// to stream, and at kWalkMaxTokens.
//
// A layer split (multi-GPU WP16) walks on every segment: each segment's lease holds its own walk
// area (the span is as long as every segment can lend), every chunk's io image is uploaded to each
// segment's area (the n-gram rows to the PLE segment only), and layer l runs on the segment that
// owns it. After a segment's last layer of chunk c, the boundary's BulkHandoff moves chunk c's
// residual columns into the next segment's walk residual, ordered on both streams without a host
// wait, so the crossing of chunk c overlaps the sender's later chunks. Each segment streams its own
// layers' experts. On two segments of one GPU their kernels never overlap (SegmentWork). A 3.1 GB/s
// x4 segment streams each non-resident expert once per span instead of once per 4,096-token call,
// which bounded its chunk-major prefill (~7 s of link per call at K = 28).

#include "models/qwen4_exp/program/program_impl.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace infernix::models::qwen4_exp::detail {

namespace {

constexpr std::size_t kWalkAlign = 256;

std::size_t walk_align(std::size_t bytes) { return (bytes + kWalkAlign - 1) / kWalkAlign * kWalkAlign; }

} // namespace

std::size_t ProgramImpl::walk_bytes(std::uint32_t tokens, std::size_t chunks) const noexcept {
    return walk_align(2ULL * static_cast<std::size_t>(width_) * tokens) + chunks * walk_align(io_layout_.bytes);
}

DeviceSegment& ProgramImpl::layer_segment(std::uint32_t layer) {
    for (const auto& segment : segments_) {
        if (segment->spec.layers.contains(layer)) { return *segment; }
    }
    throw std::out_of_range("Qwen4Exp: no segment runs layer " + std::to_string(layer));
}

// Each span costs a split a pass over its slowest segment's non-resident experts (~8 s of a 3.1 GB/s
// x4 link at K = 28, against ~2 s for a 27.6 GB/s x8 GPU alone), so on a split (SplitWalk::ExactTaps)
// only an exact tap ends a span: a flexible one (a history ladder tap, which asks for a call
// boundary, not a split) is realized where the span ends instead.
bool ProgramImpl::walk_tap_due(const Lane& lane, std::size_t next_call) const {
    return prefix_tap_due(lane, next_call, segments_.size() > 1 && split_walk_ == SplitWalk::ExactTaps);
}

std::byte* ProgramImpl::walk_slot(const DeviceSegment& seg, std::size_t chunk) const noexcept {
    return walk_area(seg) + walk_.span.residual_bytes + chunk * walk_.span.io_stride;
}

execution::ForwardBatch ProgramImpl::walk_batch(const DeviceSegment& seg, std::size_t c) {
    execution::ForwardBatch fb = io_batch(walk_slot(seg, c), 1, walk_.span.widths[c], 1);
    fb.host_slots              = std::span<const std::int32_t>(&walk_.span.host_slot, 1);
    fb.host_table_rows         = fb.host_slots;
    fb.mtp_chunk               = seg.spec.head && walk_.span.mtp[c] ? &*walk_.span.mtp[c] : nullptr;
    fb.vision                  = seg.spec.embed && walk_.span.vision[c] ? &*walk_.span.vision[c] : nullptr;
    // Chunk 0 waits per layer for the lane's restore and in-flight capture. Resolved here, not at the
    // walk's start: the walk spans engine steps, and a poll between them retires a landed batch and
    // frees its events (a landed batch needs no wait). Each segment's layers wait for their own
    // entries, recorded on that segment's streams.
    if (c == 0) { fb.layer_waits = prefix_waits(lanes_[walk_.span.lane]); }
    fb.stream         = true;
    fb.release_stream = c + 1 == walk_.span.widths.size();
    return fb;
}

// The exclusive end of the span that starts at lane.next_call; lane.next_call when the call runs
// chunk-major. Lends each segment's stream lease (with the walk's area) when none is held, or lends
// it again with the area when the held one has none for the span (another lane's chunk-major calls
// lent it). Another prefilling lane waits while the span runs (context_blocks), so two long prompts
// walk span by span instead of both re-streaming every expert per call. A split's span is as long
// as every segment's area allows, and ends at exact taps only (walk_tap_due); INFERNIX_SPLIT_WALK=0
// keeps a split's prompts chunk-major.
std::size_t ProgramImpl::walk_span_end(const Lane& lane, std::uint32_t index) {
    const std::size_t first = lane.next_call;
    if (walk_.span.active || (segments_.size() > 1 && split_walk_ == SplitWalk::ChunkMajor)) { return first; }
    for (const auto& segment : segments_) {
        if (!segment->expert_stream || segment->residency->frames() == 0) { return first; }
    }
    std::size_t end     = first;
    std::uint32_t from  = lane.state_tokens;
    std::uint32_t total = 0;
    std::vector<std::uint32_t> tokens{0}; // tokens of the span [first, first + i)
    while (end < lane.calls.size()) {
        const std::uint32_t width = lane.calls[end] - from;
        if (width < static_cast<std::uint32_t>(kStreamMinColumns) || total + width > kWalkMaxTokens) { break; }
        total += width;
        from = lane.calls[end];
        ++end;
        tokens.push_back(total);
        if (walk_tap_due(lane, end)) { break; }
    }
    if (end - first < 2) { return first; }
    for (const auto& segment : segments_) {
        DeviceSegment& seg = *segment;
        const SegmentWork on(*this, seg);
        if (seg.stream_lease.valid() && walk_bytes(tokens[2], 2) > seg.stream_walk_bytes) {
            // Between calls nothing reads the lease; its frames go back and come out again with the area.
            return_stream_lease(seg);
        }
        if (!seg.stream_lease.valid()) {
            // Sized for the prompt's largest span: as many tokens as remain (at most the cap) in calls
            // of the configured chunk, plus a remainder call.
            const std::uint32_t remaining = std::min(lane.prefill_end - lane.state_tokens, kWalkMaxTokens);
            const std::size_t chunks      = remaining / static_cast<std::uint32_t>(chunk_) + 2;
            if (!stream_chunk(seg, static_cast<std::int32_t>(lane.calls[first] - lane.state_tokens),
                              walk_bytes(remaining, chunks))) {
                return first;
            }
        }
        while (end - first >= 2 && walk_bytes(tokens[end - first], end - first) > seg.stream_walk_bytes) { --end; }
    }
    return end - first >= 2 ? end : first;
}

// Chunk c's io image (staged at `host_slot`) into `seg`'s walk area on its stream: all of it to the
// PLE segment, which alone reads the n-gram rows, and the rest around the rows to the others
// (IoLayout, upload_io).
void ProgramImpl::walk_upload(DeviceSegment& seg, std::size_t chunk, const std::byte* host_slot) {
    const SegmentWork on(*this, seg);
    const cudaStream_t s = seg.device->stream;
    std::byte* slot      = walk_slot(seg, chunk);
    if (seg.spec.ple) {
        CUDA_CHECK(cudaMemcpyAsync(slot, host_slot, io_layout_.bytes, cudaMemcpyHostToDevice, s));
        return;
    }
    CUDA_CHECK(cudaMemcpyAsync(slot, host_slot, io_layout_.ngram, cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaMemcpyAsync(slot + io_layout_.mtp_ids, host_slot + io_layout_.mtp_ids,
                               io_layout_.bytes - io_layout_.mtp_ids, cudaMemcpyHostToDevice, s));
}

void ProgramImpl::walk_begin(Lane& lane, std::uint32_t index, std::size_t end) {
    walk_.span            = WalkState::LayerWalk{};
    walk_.span.lane       = index;
    walk_.span.first_call = lane.next_call;
    walk_.span.end_call   = end;
    walk_.span.began      = Clock::now();
    const std::size_t n   = end - lane.next_call;
    std::int32_t begin = static_cast<std::int32_t>(lane.state_tokens), offset = 0;
    for (std::size_t c = lane.next_call; c < end; ++c) {
        const auto width = static_cast<std::int32_t>(lane.calls[c]) - begin;
        walk_.span.begins.push_back(begin);
        walk_.span.widths.push_back(width);
        walk_.span.offsets.push_back(offset);
        begin += width;
        offset += width;
    }
    walk_.span.io_stride      = walk_align(io_layout_.bytes);
    walk_.span.residual_bytes = walk_align(2ULL * static_cast<std::size_t>(width_) * static_cast<std::size_t>(offset));
    walk_.span.layers_per_step =
        static_cast<std::uint32_t>((c_.num_hidden_layers + n - 1) / n); // about one call's work per step
    walk_.span.mtp.assign(n, std::nullopt);
    walk_.span.vision.assign(n, std::nullopt);
    walk_.span.mtp_vision.assign(n, {});
    walk_.span.host_slot = host_lanes_[0] = static_cast<std::int32_t>(index);
    if (walk_.host.size() < n * walk_.span.io_stride) {
        walk_.host     = PinnedHostBuffer(n * walk_.span.io_stride);
        walk_.prefetch = WalkState::WalkPrefetch{};
    }
    walk_.span.active = true;

    auto* host_area = static_cast<std::byte*>(walk_.host.data());
    // A tensor (or word pointer) into `seg`'s io image, moved to chunk c's copy in its walk area: the
    // MTP cells point into the head's io, the image columns into the first segment's.
    const auto rebase = [&](const void* p, const DeviceSegment& seg, std::size_t c) -> std::byte* {
        const auto* io = static_cast<const std::byte*>(seg.io_device.p);
        const auto* b  = static_cast<const std::byte*>(p);
        return b >= io && b < io + io_layout_.bytes ? walk_slot(seg, c) + (b - io) : const_cast<std::byte*>(b);
    };
    const auto rebase_tensor = [&](Tensor t, const DeviceSegment& seg, std::size_t c) {
        if (t.data != nullptr) { t.data = rebase(t.data, seg, c); }
        return t;
    };
    for (const auto& segment : segments_) {
        DeviceSegment& seg = *segment;
        const SegmentWork on(*this, seg);
        if (segments_.size() > 1) { CUDA_CHECK(cudaEventRecord(seg.walk_first.event, seg.device->stream)); }
        seg.residency->before_round(seg.device->stream);
        seg.expert_stream->begin(ring_span(seg), seg.residency->host_table(), seg.device->stream);
    }
    DeviceSegment& embedder = layer_segment(0);
    for (std::size_t c = 0; c < n; ++c) {
        const std::int32_t b = walk_.span.begins[c], w = walk_.span.widths[c];
        // Rows the previous span prefetched already stand at this chunk's slot.
        const bool prefetched = walk_prefetched(index, walk_.span.first_call, c, b, w);
        {
            const nvtx::ScopedRange rows_range(nvtx::Name::PrefillPleRows, nvtx::Category::Prefill,
                                               static_cast<std::uint64_t>(w));
            stage_sequence(index, b, w, lane.history, !prefetched);
        }
        std::optional<execution::MtpChunk> chunk = stage_mtp_chunk(lane, index, b, w);
        const execution::VisionInput* vision     = stage_vision(lane, static_cast<std::uint32_t>(b), w);
        std::byte* host_slot = host_area + c * walk_.span.io_stride;
        if (prefetched) {
            const std::size_t rows = static_cast<std::size_t>(w) * hash_.heads() * c_.ple.table.row_bytes;
            const std::size_t tail = io_layout_.ngram + rows;
            std::memcpy(host_slot, io_host_.data(), io_layout_.ngram);
            std::memcpy(host_slot + tail, static_cast<const std::byte*>(io_host_.data()) + tail, io_layout_.bytes - tail);
            ++walk_.prefetched_chunks;
        } else {
            std::memcpy(host_slot, io_host_.data(), io_layout_.bytes);
        }
        for (const auto& segment : segments_) { walk_upload(*segment, c, host_slot); }
        if (chunk) {
            const DeviceSegment& h  = head();
            chunk->ids              = rebase_tensor(chunk->ids, h, c);
            chunk->positions        = rebase_tensor(chunk->positions, h, c);
            chunk->rope_positions   = reinterpret_cast<std::int32_t*>(rebase(chunk->rope_positions, h, c));
            chunk->block_start_rope = reinterpret_cast<std::int32_t*>(rebase(chunk->block_start_rope, h, c));
            if (chunk->vision != nullptr) {
                const std::int32_t cells     = (chunk->prepend ? 1 : 0) + chunk->columns;
                const std::int32_t subchunks = (cells + execution::kMtpChunkColumns - 1) / execution::kMtpChunkColumns;
                auto& copies                 = walk_.span.mtp_vision[c];
                copies.assign(chunk->vision, chunk->vision + subchunks);
                for (auto& input : copies) { input.columns = rebase_tensor(input.columns, h, c); }
                chunk->vision = copies.data();
            }
            walk_.span.mtp[c] = *chunk;
        }
        if (vision != nullptr) {
            walk_.span.vision[c] =
                execution::VisionInput{vision->embeddings, rebase_tensor(vision->columns, first(), c)};
        }
        // The chunk's embedding and first layer run while the host stages the next chunk.
        const SegmentWork on(*this, embedder);
        const WideWork wide(*this, embedder, w);
        walk_pass(c, 0);
    }
    walk_.span.next_layer = 1;
    walk_.prefetch        = WalkState::WalkPrefetch{}; // consumed; the slots now hold this span's chunks
    ++walk_.spans_run;
    walk_.calls_run += n;
}

std::uint64_t ProgramImpl::window_key() const noexcept {
    std::uint64_t h = 1469598103934665603ULL; // FNV-1a over the window's token ids
    for (const std::int32_t t : window_) {
        h ^= static_cast<std::uint32_t>(t);
        h *= 1099511628211ULL;
    }
    return h;
}

bool ProgramImpl::walk_prefetched(std::uint32_t index, std::size_t first_call, std::size_t c, std::int32_t begin,
                                  std::int32_t width) {
    const WalkState::WalkPrefetch& f = walk_.prefetch;
    if (!f.valid || f.lane != index || f.first_call != first_call || c >= f.widths.size() || f.begins[c] != begin ||
        f.widths[c] != width) {
        return false;
    }
    ngram_window(lanes_[index].history, begin, width);
    return window_key() == f.keys[c];
}

// Reads up to `chunks` more chunks of the span after the current one (calls from walk_.span.end_call,
// as walk_span_end would group them) into walk_.host's slots. Only after the span's first device
// synchronization: until then its own slots may still be uploading.
void ProgramImpl::walk_prefetch_rows(Lane& lane, std::uint32_t index, std::size_t chunks) {
    WalkState::WalkPrefetch& f = walk_.prefetch;
    if (!f.valid || f.lane != index || f.first_call != walk_.span.end_call) {
        f            = WalkState::WalkPrefetch{};
        f.valid      = true;
        f.lane       = index;
        f.first_call = walk_.span.end_call;
    }
    const std::size_t slots = walk_.host.size() / walk_.span.io_stride;
    std::uint32_t tokens    = 0;
    for (const auto w : f.widths) { tokens += static_cast<std::uint32_t>(w); }
    for (std::size_t added = 0; added < chunks && !f.done; ++added) {
        const std::size_t k = f.widths.size(), call = f.first_call + k;
        if (k >= slots || call >= lane.calls.size()) {
            f.done = true;
            break;
        }
        const std::uint32_t from  = lane.calls[call - 1];
        const std::uint32_t width = lane.calls[call] - from;
        if (width < static_cast<std::uint32_t>(kStreamMinColumns) || tokens + width > kWalkMaxTokens) {
            f.done = true;
            break;
        }
        const nvtx::ScopedRange rows_range(nvtx::Name::PrefillPleRows, nvtx::Category::Prefill,
                                           static_cast<std::uint64_t>(width));
        ngram_window(lane.history, static_cast<std::int32_t>(from), static_cast<std::int32_t>(width));
        auto* slot = static_cast<std::byte*>(walk_.host.data()) + k * walk_.span.io_stride;
        read_ngram_to(lane, static_cast<std::int32_t>(width), slot + io_layout_.ngram);
        f.begins.push_back(static_cast<std::int32_t>(from));
        f.widths.push_back(static_cast<std::int32_t>(width));
        f.keys.push_back(window_key());
        tokens += width;
        if (walk_tap_due(lane, call + 1)) { f.done = true; }
    }
}

// Layer l of chunk c on the segment that runs it (with the embedding before layer 0, the MTP cells
// and, for the prompt's last chunk, the logits after the last layer, and the crossing into the next
// segment after a segment's last layer). The caller holds the segment's SegmentWork and WideWork.
void ProgramImpl::walk_pass(std::size_t c, std::uint32_t l) {
    const std::size_t n   = walk_.span.widths.size();
    const std::uint32_t L = c_.num_hidden_layers;
    const bool prompt_end = walk_.span.end_call == lanes_[walk_.span.lane].calls.size();
    DeviceSegment& seg    = layer_segment(l);
    auto* area            = walk_area(seg);
    const std::int32_t w  = walk_.span.widths[c];
    execution::ForwardBatch fb = walk_batch(seg, c);
    const std::size_t column_bytes = 2ULL * static_cast<std::size_t>(width_);
    Tensor residual(area + column_bytes * static_cast<std::size_t>(walk_.span.offsets[c]), DType::BF16, {width_, w});
    const bool logits_here = l + 1 == L && prompt_end && c + 1 == n;
    Tensor logits(head().logits32.p, DType::FP32, {vocab_, 1});
    seg.forward->begin_call(fb, logits_here ? &logits : nullptr);
    if (l == 0) { seg.forward->embed(fb, residual); }
    seg.forward->layer(l, fb, residual);
    if (l + 1 == L) { seg.forward->finish(fb, residual, logits_here ? &logits : nullptr); }
    if (l + 1 == seg.spec.layers.end && !seg.spec.head) {
        // The boundary: chunk c's residual columns into the next segment's walk residual, at the
        // same columns; the receiver's first layer of chunk c is ordered after them.
        DeviceSegment& next = *segments_[seg.spec.index + 1];
        seg.bulk_out->transfer(residual.data,
                               walk_area(next) + column_bytes * static_cast<std::size_t>(walk_.span.offsets[c]),
                               residual.bytes(), seg.device->stream, next.device->stream);
    }
}

// Layers [next_layer, to_layer) of every chunk, grouped by segment: each group under its segment's
// SegmentWork (its device current; on a shared GPU after the other segments' kernels) and wide arena.
// A split's groups are submitted at once, as the next segment's stream waits for their crossings.
void ProgramImpl::walk_enqueue(std::uint32_t to_layer) {
    const std::int32_t widest = *std::max_element(walk_.span.widths.begin(), walk_.span.widths.end());
    std::uint32_t l           = walk_.span.next_layer;
    while (l < to_layer) {
        DeviceSegment& seg      = layer_segment(l);
        const std::uint32_t end = std::min(to_layer, seg.spec.layers.end);
        const bool split        = segments_.size() > 1;
        const SegmentWork on(*this, seg);
        const WideWork wide(*this, seg, widest);
        if (split && l == seg.spec.layers.begin && l > 0) {
            CUDA_CHECK(cudaEventRecord(seg.walk_first.event, seg.device->stream));
        }
        for (; l < end; ++l) {
            for (std::size_t c = 0; c < walk_.span.widths.size(); ++c) { walk_pass(c, l); }
        }
        if (split && end == seg.spec.layers.end) {
            CUDA_CHECK(cudaEventRecord(seg.walk_last.event, seg.device->stream));
        }
        if (split) { seg.device->flush(); }
    }
    walk_.span.next_layer = std::max(walk_.span.next_layer, to_layer);
}

PrefillProgress ProgramImpl::walk_step(Lane& lane, std::uint32_t index, Clock::time_point start) {
    const std::uint32_t L = c_.num_hidden_layers;
    std::uint32_t tokens  = 0;
    for (const auto w : walk_.span.widths) { tokens += static_cast<std::uint32_t>(w); }
    // The first step already ran layer 0 while staging.
    const std::uint32_t to =
        std::min(L, walk_.span.next_layer + walk_.span.layers_per_step - (walk_.span.steps == 0 ? 1U : 0U));
    {
        const nvtx::ScopedRange step_range(nvtx::Name::PrefillChunk, nvtx::Category::Prefill,
                                           static_cast<std::uint64_t>(tokens));
        walk_enqueue(to);
    }
    ++walk_.span.steps;
    if (to < L) {
        // While the device runs this step, read the next span's n-gram rows (one span's chunks spread
        // over the remaining steps; ~35-70 ms of NVMe reads per 4,096-token chunk).
        if (walk_.span.steps > 1 && walk_.span.end_call < lane.calls.size()) {
            const std::size_t remaining = (L - to + walk_.span.layers_per_step - 1) / walk_.span.layers_per_step + 1;
            const std::size_t slots     = walk_.host.size() / walk_.span.io_stride;
            const std::size_t have      = walk_.prefetch.valid ? walk_.prefetch.widths.size() : 0;
            walk_prefetch_rows(lane, index, slots > have ? (slots - have + remaining - 1) / remaining : 0);
        }
        {
            const nvtx::ScopedRange wait_range(nvtx::Name::DeviceWait, nvtx::Category::Prefill);
            synchronize();
        }
        for (const auto& segment : segments_) {
            const SegmentWork on(*this, *segment);
            segment->residency->tier_step(segment->device->stream);
        }
        PrefillProgress out;
        out.summary                 = prefill_summary(lane);
        out.processed_prompt_tokens = 0;
        out.complete                = false;
        out.timing.submit_host_ns   = elapsed_ns(start);
        lane.prefill_ns += out.timing.submit_host_ns;
        return out;
    }
    // The span is enqueued: the lane's state is at its end in every layer.
    const std::int32_t begin = walk_.span.begins.back(), width = walk_.span.widths.back();
    const bool last          = walk_.span.end_call == lane.calls.size();
    lane.state_tokens = lane.calls[walk_.span.end_call - 1];
    lane.next_call    = walk_.span.end_call;
    for (const auto& segment : segments_) {
        const SegmentWork on(*this, *segment);
        segment->residency->enqueue_route_download(segment->device->stream, width);
        segment->expert_stream->end();
    }
    walk_.span.active = false;
    host_lanes_[0]    = static_cast<std::int32_t>(index);
    if (last && !prefilling_besides(index)) { return_leases(); }
    prefix_after_prefill_call(lane, index, last);
    if (last && lane.vision) { vision_release(lane); }
    if (!last) {
        {
            const nvtx::ScopedRange wait_range(nvtx::Name::DeviceWait, nvtx::Category::Prefill);
            synchronize();
        }
        walk_report(walk_.span.began);
        const nvtx::ScopedRange residency_range(nvtx::Name::PrefillResidency, nvtx::Category::Moe,
                                                static_cast<std::uint64_t>(width));
        // The route log holds the span's last chunk (every chunk routes nearly every expert).
        trace_round(RouteTraceKind::PrefillChunk, 1, width, tokens, 0, static_cast<std::uint32_t>(begin));
        after_call(width, std::vector<std::size_t>(segments_.size(), 0));
        apply_vram_target(false);
    } else {
        walk_report(walk_.span.began);
    }
    // A walk promotes no experts (its routes still count toward the cache's scores). Between spans
    // the stream lease holds most frames, so promotions would churn through the rest (~12,000
    // records, ~1.2 s of link per 64K span) and the next span would wait for them. After the
    // prompt they would fill the returned frames with the prompt's experts while the answer's own
    // staged misses land there anyway: 16 per layer and chunk measured slower than none for every
    // answer length (128K + 512 tokens: 23.16 -> 21.10 s; design §19.3.17).
    const std::vector<std::size_t> none(segments_.size(), 0);
    return prefill_progress(lane, index, begin, width, tokens, last, none, start);
}

// A split's span report (debug): its tokens and wall time, and each segment's time on its GPU from
// its first to its last pass (with the host steps between them), so where the span's time went.
// Waits for each segment's last pass (the end of the span's work).
void ProgramImpl::walk_report(Clock::time_point began) noexcept {
    if (segments_.size() < 2) { return; }
    try {
        std::uint32_t tokens = 0;
        for (const auto w : walk_.span.widths) { tokens += static_cast<std::uint32_t>(w); }
        std::string line = "layer walk span of " + std::to_string(walk_.span.widths.size()) + " chunks (" +
                           std::to_string(tokens) + " tokens from " + std::to_string(walk_.span.begins.front()) +
                           "): ";
        for (const auto& segment : segments_) {
            const DeviceGuard on(segment->device->device);
            CUDA_CHECK(cudaEventSynchronize(segment->walk_last.event));
            float ms = 0.0F;
            CUDA_CHECK(cudaEventElapsedTime(&ms, segment->walk_first.event, segment->walk_last.event));
            line += segment_name(*segment) + std::to_string(static_cast<int>(ms)) + " ms; ";
        }
        line += "wall " + std::to_string(static_cast<long long>(elapsed_ns(began) / 1000000)) + " ms";
        diagnostic(line, DiagnosticLevel::Debug);
    } catch (...) {}
}

void ProgramImpl::walk_abandon() noexcept {
    if (!walk_.span.active) { return; }
    try {
        synchronize();
    } catch (...) {}
    for (const auto& segment : segments_) {
        try {
            const DeviceGuard on(segment->device->device);
            segment->expert_stream->end();
        } catch (...) {}
    }
    walk_.span     = WalkState::LayerWalk{};
    walk_.prefetch = WalkState::WalkPrefetch{};
}

} // namespace infernix::models::qwen4_exp::detail
