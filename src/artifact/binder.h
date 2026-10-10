#pragma once

#include "artifact/materializer.h"
#include "artifact/reader.h"

#include <optional>
#include <span>
#include <string>
#include <vector>

namespace infernix::artifact {

// Streamed: located but never materialized; the runtime reads the object in place from the
// artifact's files (MaterializedArtifact::stream_source, design §19.3.7 SSD tier).
enum class Residency { Device, Host, Values, HostPinned, Streamed };

struct ParameterReference {
    std::string name;
    Shape shape;
    Binding binding;
    Residency residency = Residency::Device;
};

struct HostValues {
    QType format           = QType::FP32;
    std::uint64_t elements = 0;
    std::vector<std::byte> data;

    [[nodiscard]] float scalar_f32() const;
    [[nodiscard]] std::vector<std::int32_t> integers() const;
};

// Collects selected logical demands; neither physical object IDs nor whole-artifact profiles
// determine parameter shapes or native operation support.
//
// Device segments (a layer split): each device object belongs to one segment, whose arena is
// materialized on that segment's device; set_segment_count declares how many there are (one by
// default). A HostPinned object may name the segment that reads it, or kSharedSegment; that tag
// only places it when the plan keeps per-segment pinned chunks (pinned_per_segment).
class Binder {
public:
    explicit Binder(const Reader& reader);

    // `segment` tags the parameter's objects: the arena of a Device parameter (segment 0 when
    // unset), the reading segment of a HostPinned one (kSharedSegment when unset, as for
    // require_pinned); other residencies ignore it. An object keeps one tag per residency: a
    // second demand with another tag throws.
    [[nodiscard]] ParameterReference parameter(std::string_view name, Shape shape,
                                               Residency residency = Residency::Device,
                                               std::optional<QType> exact_format    = {},
                                               std::optional<std::uint8_t> segment = {});
    [[nodiscard]] ParameterReference binding(std::string name, const Binding& binding, Shape shape,
                                             Residency residency,
                                             std::optional<QType> exact_format    = {},
                                             std::optional<std::uint8_t> segment = {});
    [[nodiscard]] const Use& use(std::string_view parameter, std::string_view input) const;
    [[nodiscard]] bool contains(std::string_view parameter) const;

    [[nodiscard]] const Reader& reader() const noexcept { return reader_; }

    // Device segments of the plan, 1 to kSharedSegment - 1. finish() rejects tags at or above it.
    void set_segment_count(std::uint8_t count);

    // A device object belongs to exactly one segment (kSharedSegment is not one).
    void require_device(ObjectHandle object, std::uint64_t alignment = 256,
                        std::uint8_t segment = 0);
    // Rank a device demand for the evictable arena suffix: higher ranks pack closer to the
    // arena end and are evicted first. Requires the object to be a device demand and the plan
    // to have one segment.
    void mark_device_evictable(ObjectHandle object, std::uint32_t evict_rank);
    void require_pinned(ObjectHandle object, std::uint64_t alignment = 256,
                        std::uint8_t segment = kSharedSegment);
    void require_streamed(ObjectHandle object);
    [[nodiscard]] std::span<const std::byte> host_object(ObjectHandle object);
    [[nodiscard]] ObjectHandle resource(std::string_view component, std::string_view role);
    [[nodiscard]] HostValues values(const Binding& binding, std::optional<QType> format = {});
    // Device objects pack in object order into their segment's arena, at offsets relative to it.
    // evictable_alignment (a power of two, typically EvictableWeightPool::kChunkBytes)
    // chunk-aligns the first ranked object so the evictable suffix is chunk-aligned.
    [[nodiscard]] MaterializationPlan finish(std::uint64_t evictable_alignment = 1) &&;

private:
    struct Demand {
        bool device             = false;
        bool host               = false;
        bool pinned             = false;
        bool streamed           = false;
        std::uint64_t alignment = 256;
        std::uint32_t evict_rank = 0;
        std::uint8_t segment        = 0;              // device arena
        std::uint8_t pinned_segment = kSharedSegment; // reading segment of a pinned demand
        std::vector<std::byte> host_data;
    };

    const Reader& reader_;
    std::vector<Demand> demands_;
    std::uint64_t read_bytes_        = 0;
    std::uint64_t owned_value_bytes_ = 0;
    std::uint8_t segment_count_      = 1;
};

} // namespace infernix::artifact
