#pragma once

#include "artifact/schema.h"
#include "core/arena.h"
#include "core/device.h"
#include "core/evictable_weight_pool.h"
#include "core/weight_view.h"
#include "infernix/types.h"

#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace infernix::artifact {

class Reader;

// The tag of a pinned object every device segment reads (Binder::require_pinned); the same value
// as models::qwen4_exp::kSharedSegment. Never a device segment.
inline constexpr std::uint8_t kSharedSegment = 0xFF;

struct DevicePlacement {
    ObjectHandle object;
    std::uint64_t offset    = 0; // inside the arena of `segment`
    std::uint64_t bytes     = 0;
    std::uint64_t alignment = 256;
    std::uint8_t segment    = 0;
};

// The device arena of one segment: materialize() allocates it on the segment's device.
struct DeviceSegmentPlan {
    std::uint64_t capacity_bytes = 0;
};

struct HostPlacement {
    ObjectHandle object;
    // Already-read resources move into final storage without invalidating their byte views.
    std::vector<std::byte> data;
};

struct PinnedPlacement {
    ObjectHandle object;
    std::uint64_t offset = 0;
    std::uint64_t bytes  = 0;
    std::uint8_t segment = kSharedSegment; // the reading segment; placed by pinned_per_segment only
};

struct MaterializationPlan {
    const Reader* source                = nullptr;
    std::size_t object_count            = 0;
    std::uint64_t device_capacity_bytes = 0; // segment 0's arena: device_segments[0].capacity_bytes
    std::uint64_t prior_read_bytes      = 0;
    std::uint64_t owned_value_bytes     = 0;
    std::vector<DevicePlacement> device_objects;
    std::vector<HostPlacement> host_objects;
    std::uint64_t pinned_capacity_bytes = 0;
    std::vector<PinnedPlacement> pinned_objects;
    // Objects read in place at run time; located, never read here.
    std::vector<ObjectHandle> streamed_objects;
    // Physical Host memory the pinned block must leave free, after the caller's later pins
    // (`later_pinned_bytes`, allocated once the model is loaded) are counted against it too.
    std::uint64_t host_reserve_bytes = kDefaultRamHeadroomBytes;
    std::uint64_t later_pinned_bytes = 0;
    // Arena suffix holding the ranked (low-traffic) device weights that an
    // EvictableWeightPool may temporarily evict; chunk-aligned by finish(). One segment only.
    std::uint64_t evictable_tail_bytes = 0;
    // One arena per device segment (Binder::set_segment_count); materialize() takes one device per
    // entry. Empty (a plan not made by a Binder) means one arena of device_capacity_bytes.
    std::vector<DeviceSegmentPlan> device_segments;
    // Off: one pinned space of portable cudaMallocHost chunks that every device reads. On: a chunk
    // holds the objects of one reading segment and is mapped, non-portable memory of that
    // segment's device (cudaHostAlloc while it is current); kSharedSegment objects stay portable.
    // For a platform where the portable chunks cannot be mapped into every device's context.
    bool pinned_per_segment = false;
};

struct MaterializationStats {
    std::uint64_t file_bytes = 0; // Declared container file set, including framing.
    std::uint64_t read_bytes = 0; // Actual payload reads, including direct-I/O alignment.
    std::uint64_t h2d_bytes  = 0;
    std::uint64_t device_capacity_bytes = 0; // all device segments together
    std::uint64_t retained_host_bytes   = 0;
    std::uint64_t owned_value_bytes     = 0;
    std::uint64_t peak_staging_bytes    = 0;
    std::uint64_t pinned_bytes          = 0;
    std::size_t device_object_count     = 0;
    std::size_t host_object_count       = 0;
    std::size_t pinned_object_count     = 0;
    double upload_seconds               = 0;
};

class MaterializedArtifact;

// One contiguous piece of a byte range in one artifact file.
struct FileSegment {
    std::uint32_t file   = 0; // index into StreamSource::files()
    std::uint64_t offset = 0; // byte offset in that file
    std::uint64_t at     = 0; // byte offset inside the requested range
    std::uint64_t bytes  = 0;
};

// Where Streamed objects live in the artifact's files, for reading them in place (design §19.3.7):
// the file paths (the entry and its parts) and each object's segments, kept after the Reader is
// gone. It opens no file; the reader of the bytes opens its own handles.
class StreamSource {
public:
    [[nodiscard]] std::span<const std::filesystem::path> files() const noexcept { return files_; }
    [[nodiscard]] bool contains(ObjectHandle object) const noexcept;
    [[nodiscard]] std::uint64_t object_bytes(ObjectHandle object) const;
    // The file segments of [offset, offset + bytes) of a streamed object.
    [[nodiscard]] std::vector<FileSegment> segments(ObjectHandle object, std::uint64_t offset,
                                                    std::uint64_t bytes) const;

private:
    friend MaterializedArtifact materialize(const Reader&, MaterializationPlan&&,
                                            std::span<DeviceContext* const>, const StartupObserver*,
                                            std::unique_ptr<EvictableWeightPool>);
    struct Object {
        std::uint64_t bytes = 0;
        std::vector<FileSegment> segments; // the whole object, `at` from its start
    };
    std::vector<std::filesystem::path> files_;
    std::vector<std::optional<Object>> objects_;
};

class MaterializedArtifact {
public:
    MaterializedArtifact()                                           = default;
    ~MaterializedArtifact()                                          = default;
    MaterializedArtifact(MaterializedArtifact&&) noexcept            = default;
    MaterializedArtifact& operator=(MaterializedArtifact&&) noexcept = default;
    MaterializedArtifact(const MaterializedArtifact&)                = delete;
    MaterializedArtifact& operator=(const MaterializedArtifact&)     = delete;

    [[nodiscard]] const WeightParent& device_parent(ObjectHandle handle) const;
    [[nodiscard]] const WeightParent& host_parent(ObjectHandle handle) const;
    [[nodiscard]] const WeightParent& pinned_parent(ObjectHandle handle) const;
    // A Streamed object's geometry with no resident data (data == nullptr).
    [[nodiscard]] const WeightParent& streamed_parent(ObjectHandle handle) const;
    [[nodiscard]] const StreamSource& stream_source() const noexcept { return stream_source_; }
    [[nodiscard]] std::span<const std::byte> host_bytes(ObjectHandle handle) const;
    [[nodiscard]] bool has_device(ObjectHandle handle) const noexcept;

    // Device segments: segment s's arena lives on the s-th device given to materialize(), and
    // every device parent points into the arena of its object's segment.
    [[nodiscard]] std::size_t device_segment_count() const noexcept {
        return device_segments_.size();
    }

    // The arena of `segment` ({nullptr, 0} when it holds no object).
    [[nodiscard]] DeviceSpan device_arena(std::size_t segment) const;
    // The segment whose arena holds the device parent of `handle`.
    [[nodiscard]] std::uint8_t device_segment(ObjectHandle handle) const;

    // Pinned host memory backing HostPinned objects (e.g. the vision tower in overlay mode),
    // exposed so the runtime can stream the weights to device staging. The plan's pinned
    // offsets form one contiguous space, but it is backed by several cudaMallocHost chunks
    // split at object boundaries: Windows refuses a single pinned allocation much above half
    // of physical memory, and only cudaMallocHost memory is addressable by the device at its
    // host address there (registered memory is not). pinned_address maps a plan offset range,
    // which must lie inside one object run, to its address. With pinned_per_segment a chunk holds
    // one reading segment's objects, allocated with that segment's device current.
    [[nodiscard]] bool has_pinned() const noexcept { return !pinned_chunks_.empty(); }
    [[nodiscard]] const std::byte* pinned_address(std::uint64_t offset, std::uint64_t bytes) const;

    // VMM arena pool present only when the plan reserved an evictable tail.
    [[nodiscard]] EvictableWeightPool* eviction_pool() noexcept {
        return eviction_pool_.get();
    }
    [[nodiscard]] const EvictableWeightPool* eviction_pool() const noexcept {
        return eviction_pool_.get();
    }

    [[nodiscard]] const MaterializationStats& stats() const noexcept { return stats_; }

private:
    friend MaterializedArtifact materialize(const Reader&, MaterializationPlan&&,
                                            std::span<DeviceContext* const>, const StartupObserver*,
                                            std::unique_ptr<EvictableWeightPool>);

    struct ObjectStorage {
        std::optional<WeightParent> device;
        std::optional<WeightParent> host;
        std::optional<WeightParent> pinned;
        std::optional<WeightParent> streamed;
        std::vector<std::byte> host_data;
        std::uint8_t device_segment = 0;
    };

    // Deleters that release a segment's allocation with the segment's device current.
    struct ArenaRelease {
        int device = 0;
        void operator()(DeviceArena* arena) const noexcept;
    };

    struct MappedRelease {
        int device = 0;
        void operator()(void* pointer) const noexcept;
    };

    struct SegmentArena {
        int device = 0;
        std::unique_ptr<DeviceArena, ArenaRelease> arena; // null: empty, or the eviction pool's
        std::byte* base     = nullptr;
        std::uint64_t bytes = 0;
    };

    std::vector<SegmentArena> device_segments_;
    struct PinnedChunk {
        std::uint64_t begin  = 0; // plan offset of the chunk's first byte
        std::uint64_t end    = 0;
        std::uint8_t segment = kSharedSegment;
        std::unique_ptr<PinnedHostBuffer> buffer;    // portable chunk (kSharedSegment)
        std::unique_ptr<void, MappedRelease> mapped; // a segment's chunk (pinned_per_segment)
        std::byte* base = nullptr; // address of plan offset `begin`
    };
    std::vector<PinnedChunk> pinned_chunks_;
    std::unique_ptr<EvictableWeightPool> eviction_pool_;
    std::vector<ObjectStorage> objects_;
    StreamSource stream_source_;
    MaterializationStats stats_;
};

// devices[s] backs device segment s (plan.device_segments; one device for a plan with one
// segment). Every segment's arena is allocated, with its device current, before any pinned chunk;
// dense uploads run on each owning device's transfer_stream. Two segments may name contexts of
// one ordinal. An eviction pool requires one segment.
[[nodiscard]] MaterializedArtifact materialize(const Reader& reader, MaterializationPlan&& plan,
                                               std::span<DeviceContext* const> devices,
                                               const StartupObserver* startup_observer = nullptr,
                                               std::unique_ptr<EvictableWeightPool> eviction_pool =
                                                   nullptr);

// One device segment: materialize(reader, plan, {&device}, ...).
[[nodiscard]] MaterializedArtifact materialize(const Reader& reader, MaterializationPlan&& plan,
                                               DeviceContext& device,
                                               const StartupObserver* startup_observer = nullptr,
                                               std::unique_ptr<EvictableWeightPool> eviction_pool =
                                                   nullptr);

} // namespace infernix::artifact
