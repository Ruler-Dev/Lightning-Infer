#pragma once

#include "models/qwen4_exp/model.h"

#include <filesystem>
#include <memory>
#include <span>
#include <vector>

namespace infernix::artifact {
class Reader;
}

namespace infernix::models::qwen4_exp {

// Cold load plan; it borrows its Reader until materialization.
//
// Layer split (LoadOptions::segment_ends): each dense weight goes to the device segment that reads
// it (SegmentMap): a decoder layer's to the segment running it, the final mixer, the head, the MTP
// drafter and the proposal head to the last, the Vision tower to the first. The token embedding is
// pinned and shared; each expert bank stays pinned, tagged with its layer's segment.
class LoadPlan {
public:
    ~LoadPlan();
    LoadPlan(LoadPlan&&) noexcept;
    LoadPlan& operator=(LoadPlan&&) noexcept;
    LoadPlan(const LoadPlan&)            = delete;
    LoadPlan& operator=(const LoadPlan&) = delete;

    [[nodiscard]] const Config& config() const;
    [[nodiscard]] const artifact::MaterializationPlan& materialization() const;
    // Pinned bytes of the expert banks and of the model's other pinned weights (token embedding).
    [[nodiscard]] std::uint64_t pinned_expert_bytes() const;
    [[nodiscard]] std::uint64_t pinned_other_bytes() const;
    // Device bytes of the dense weights (every segment's arena), and the tokenizer's public token
    // count.
    [[nodiscard]] std::uint64_t device_bytes() const;
    [[nodiscard]] std::uint32_t public_token_count() const;
    // The Host memory the pinned block must leave free at materialization, and the pins the
    // caller makes after it (both checked before anything is read).
    void set_host_reserve(std::uint64_t reserve_bytes, std::uint64_t later_pinned_bytes);

    // The device segments of the plan (one without a split) and the arena bytes of each.
    [[nodiscard]] std::uint32_t segment_count() const;
    [[nodiscard]] std::uint64_t device_bytes(std::uint32_t segment) const;
    // The device bytes each part of the model adds to the segment holding it, for sizing a split:
    // each decoder layer's (indexed by layer), the head's (final mixer and output head), the MTP
    // drafter's with its proposal head, and the device-resident Vision tower's (0 when it is
    // offloaded or not bound). Each object counts once, with the most padding its alignment can
    // put before it, so a segment's arena never exceeds the sum over the parts it holds.
    [[nodiscard]] std::vector<std::uint64_t> layer_device_bytes() const;
    [[nodiscard]] std::uint64_t head_device_bytes() const;
    [[nodiscard]] std::uint64_t mtp_device_bytes() const;
    [[nodiscard]] std::uint64_t vision_device_bytes() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    explicit LoadPlan(std::unique_ptr<Impl> impl);
    friend LoadPlan plan_load(const artifact::Reader&, LoadOptions);
    friend std::unique_ptr<Model> materialize_model(LoadPlan&&, std::span<DeviceContext* const>,
                                                    const StartupObserver*);
};

// True when the artifact's text component is Qwen4Exp.
[[nodiscard]] bool is_qwen4_exp(const artifact::Reader& reader);

// Throws std::invalid_argument when options.segment_ends do not ascend to the layer count (each
// segment owning at least one layer).
[[nodiscard]] LoadPlan plan_load(const artifact::Reader& reader, LoadOptions options = {});
// devices[s] backs device segment s (one per LoadPlan::segment_count(); two may be contexts of one
// ordinal); the Model's SegmentMap names their ordinals.
[[nodiscard]] std::unique_ptr<Model> materialize_model(LoadPlan&& plan,
                                                       std::span<DeviceContext* const> devices,
                                                       const StartupObserver* observer = nullptr);
// One device segment: materialize_model(plan, {&device}, observer).
[[nodiscard]] std::unique_ptr<Model> materialize_model(LoadPlan&& plan, DeviceContext& device,
                                                       const StartupObserver* observer = nullptr);
[[nodiscard]] std::unique_ptr<Model> load_model(const std::filesystem::path& path,
                                                LoadOptions options, DeviceContext& device,
                                                const StartupObserver* observer = nullptr);

} // namespace infernix::models::qwen4_exp
