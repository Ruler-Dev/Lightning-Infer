// Device segments in the artifact layer (layer split, multi-GPU WP6): Binder segment tags and the
// per-segment layout of finish(), per-segment device arenas and pinned chunks in materialize().
//
// Without arguments: the plan layout on a synthetic artifact (byte-identical to the one-arena
// layout for one segment, per-segment offsets for two, rejected tags), then materialization on two
// DeviceContexts of device 0 and, with two GPUs, on devices 0 and 1 in both orders. Skips (77)
// without a CUDA device once the layout checks have passed.
//
// --real: INFERNIX_QWEN4_ARTIFACT names a Qwen4Exp artifact; every dense object (6.3 GiB of the
// Dense8 artifact) loaded as segments {24, 48} on two DeviceContexts of device 0 must equal a
// one-segment load byte for byte. Skips (77) when the variable is unset.

#include "artifact/binder.h"
#include "artifact/fixture.h"
#include "core/device.h"
#include "core/device_guard.h"

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <iostream>
#include <map>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace {

using namespace infernix;
using namespace infernix::artifact;
using namespace infernix::test::artifact_fixture;

// BF16 vectors at 256-byte-aligned payload offsets. The first file ends inside d1, so its upload
// crosses files; p* are pinned, x serves the tag conflicts.
struct SyntheticObject {
    std::string_view id;
    std::uint64_t elements = 0;
    std::uint64_t offset   = 0;

    [[nodiscard]] std::uint64_t bytes() const { return 2 * elements; }
};

constexpr std::array kObjects = {
    SyntheticObject{"a0", 500, 0},      SyntheticObject{"b1", 300, 1024},
    SyntheticObject{"c0", 1024, 2048},  SyntheticObject{"d1", 2100, 4096},
    SyntheticObject{"e1", 40, 8448},    SyntheticObject{"p0", 3000, 8704},
    SyntheticObject{"p1", 2048, 14848}, SyntheticObject{"ps", 1000, 18944},
    SyntheticObject{"x", 128, 20992},
};
constexpr std::uint64_t kPayloadBytes   = 21248;
constexpr std::uint64_t kFirstFileBytes = 6000;

const SyntheticObject& object(std::string_view id) {
    for (const auto& item : kObjects) {
        if (item.id == id) { return item; }
    }
    throw std::logic_error("unknown synthetic object");
}

Shape shape(std::string_view id) { return {object(id).elements}; }

void write_synthetic(Fixture& fixture) {
    Json objects  = Json::array();
    Json bindings = Json::object();
    for (const auto& item : kObjects) {
        objects.push_back({{"id", item.id},
                           {"kind", "tensor"},
                           {"shape", Json::array({item.elements})},
                           {"format", "bf16"},
                           {"layout", "contiguous_le_v1"},
                           {"offset", item.offset},
                           {"bytes", item.bytes()}});
        bindings[std::string(item.id)] = {{"object", item.id}};
    }
    fixture.root = {{"components", {{"text", {{"config", Json::object()}}}}},
                    {"objects", objects},
                    {"bindings", bindings},
                    {"uses", Json::array()},
                    {"files", Json::array({{{"path", nullptr}, {"payload_bytes", kFirstFileBytes}},
                                           {{"path", "weights-second.bin"},
                                            {"payload_bytes", kPayloadBytes - kFirstFileBytes}}})}};
    fixture.payload.resize(kPayloadBytes);
    for (std::size_t i = 0; i < fixture.payload.size(); ++i) {
        fixture.payload[i] = std::byte((i * 7 + i / 4096 * 13 + 3) % 251);
    }
    fixture.write();
}

const DevicePlacement& placement_of(const MaterializationPlan& plan, ObjectHandle handle) {
    for (const auto& placement : plan.device_objects) {
        if (placement.object == handle) { return placement; }
    }
    throw std::runtime_error("object has no device placement");
}

const PinnedPlacement& pinned_of(const MaterializationPlan& plan, ObjectHandle handle) {
    for (const auto& placement : plan.pinned_objects) {
        if (placement.object == handle) { return placement; }
    }
    throw std::runtime_error("object has no pinned placement");
}

struct Expected {
    std::string_view id;
    std::uint64_t offset    = 0;
    std::uint64_t alignment = 256;
    std::uint8_t segment    = 0;
};

void require_layout(const Reader& reader, const MaterializationPlan& plan,
                    std::initializer_list<Expected> expected,
                    std::initializer_list<std::uint64_t> capacities, const char* message) {
    require(plan.device_objects.size() == expected.size(), message);
    for (const auto& item : expected) {
        const auto& placement = placement_of(plan, reader.find(item.id));
        require(placement.offset == item.offset && placement.bytes == object(item.id).bytes() &&
                    placement.alignment == item.alignment && placement.segment == item.segment,
                message);
    }
    require(plan.device_segments.size() == capacities.size(), message);
    std::size_t s = 0;
    for (const auto capacity : capacities) {
        require(plan.device_segments[s++].capacity_bytes == capacity, message);
    }
    require(plan.device_capacity_bytes == *capacities.begin(), message);
}

// finish() on one segment keeps the one-arena layout: object order, 256-byte alignment, and the
// evictable tail packed after the unranked prefix in rank order (offsets worked out by hand).
void single_segment_layout(const std::filesystem::path& entry) {
    Reader reader(entry);
    {
        Binder binder(reader);
        for (const auto* id : {"a0", "b1", "c0", "d1", "e1"}) {
            (void)binder.parameter(id, shape(id));
        }
        const auto plan = std::move(binder).finish();
        require_layout(reader, plan,
                       {{"a0", 0}, {"b1", 1024}, {"c0", 1792}, {"d1", 3840}, {"e1", 8192}}, {8272},
                       "one-segment layout differs from the one-arena layout");
        require(plan.evictable_tail_bytes == 0 && !plan.pinned_per_segment,
                "one-segment plan gained an evictable tail or per-segment pins");
    }
    {
        Binder binder(reader);
        binder.set_segment_count(1);
        for (const auto* id : {"a0", "b1", "c0", "d1", "e1"}) {
            (void)binder.parameter(id, shape(id), Residency::Device, {}, 0);
        }
        binder.mark_device_evictable(reader.find("b1"), 1);
        const auto plan = std::move(binder).finish(4096);
        require_layout(reader, plan,
                       {{"a0", 0}, {"c0", 1024}, {"d1", 3072}, {"e1", 7424}, {"b1", 8192, 4096}},
                       {8792}, "one-segment evictable layout differs from the one-arena layout");
        require(plan.evictable_tail_bytes == 600, "evictable tail differs");
    }
}

// Two segments: each object packs into its own segment's arena in object order, offsets relative
// to that arena; pinned offsets do not depend on the tags.
void two_segment_layout(const std::filesystem::path& entry) {
    Reader reader(entry);
    std::vector<std::uint64_t> untagged_pins;
    {
        Binder binder(reader);
        for (const auto* id : {"p0", "p1", "ps"}) {
            (void)binder.parameter(id, shape(id), Residency::HostPinned);
        }
        const auto plan = std::move(binder).finish();
        for (const auto* id : {"p0", "p1", "ps"}) {
            const auto& pinned = pinned_of(plan, reader.find(id));
            require(pinned.segment == kSharedSegment,
                    "an untagged pinned parameter is not tagged shared");
            untagged_pins.push_back(pinned.offset);
        }
    }
    Binder binder(reader);
    binder.set_segment_count(2);
    (void)binder.parameter("a0", shape("a0"), Residency::Device, {}, 0);
    (void)binder.parameter("b1", shape("b1"), Residency::Device, {}, 1);
    (void)binder.parameter("c0", shape("c0"), Residency::Device, {}, 0);
    (void)binder.parameter("d1", shape("d1"), Residency::Device, {}, 1);
    binder.require_device(reader.find("e1"), 4096, 1);
    (void)binder.parameter("p0", shape("p0"), Residency::HostPinned, {}, 0);
    (void)binder.parameter("p1", shape("p1"), Residency::HostPinned, {}, 1);
    (void)binder.parameter("ps", shape("ps"), Residency::HostPinned, {}, kSharedSegment);
    const auto plan = std::move(binder).finish();
    require_layout(reader, plan,
                   {{"a0", 0, 256, 0},
                    {"b1", 0, 256, 1},
                    {"c0", 1024, 256, 0},
                    {"d1", 768, 256, 1},
                    {"e1", 8192, 4096, 1}},
                   {3072, 8272}, "two-segment layout differs from the per-arena packing");
    const std::array<std::uint8_t, 3> tags = {0, 1, kSharedSegment};
    std::size_t i                          = 0;
    for (const auto* id : {"p0", "p1", "ps"}) {
        const auto& pinned = pinned_of(plan, reader.find(id));
        require(pinned.segment == tags[i] && pinned.offset == untagged_pins[i],
                "pinned tags changed the pinned layout or were lost");
        ++i;
    }
}

void rejected_tags(const std::filesystem::path& entry) {
    Reader reader(entry);
    const auto x = reader.find("x");
    {
        Binder binder(reader);
        binder.set_segment_count(2);
        (void)binder.parameter("x", shape("x"), Residency::Device, {}, 0);
        rejects([&] { (void)binder.parameter("x", shape("x"), Residency::Device, {}, 1); },
                "an object was demanded on two device segments");
        (void)binder.parameter("x", shape("x"), Residency::Device, {}, 0); // the same tag again
    }
    {
        Binder binder(reader);
        binder.set_segment_count(2);
        (void)binder.parameter("x", shape("x"), Residency::HostPinned, {}, 0);
        rejects([&] { binder.require_pinned(x, 256, kSharedSegment); },
                "a pinned object was tagged for a segment and as shared");
        // Device and pinned demands of one object are tagged separately (a tied head).
        (void)binder.parameter("x", shape("x"), Residency::Device, {}, 1);
        const auto plan = std::move(binder).finish();
        require(placement_of(plan, x).segment == 1 && pinned_of(plan, x).segment == 0,
                "device and pinned tags of one object interfered");
    }
    {
        Binder binder(reader);
        rejects([&] { binder.require_device(x, 256, kSharedSegment); },
                "a device object was tagged as shared");
        rejects([&] { binder.set_segment_count(0); }, "zero device segments were accepted");
        rejects([&] { binder.set_segment_count(kSharedSegment); },
                "the shared tag was accepted as a segment count");
        binder.set_segment_count(kSharedSegment - 1);
    }
    {
        Binder binder(reader);
        (void)binder.parameter("x", shape("x"), Residency::Device, {}, 1);
        rejects([&] { (void)std::move(binder).finish(); },
                "a device tag beyond the segment count was accepted");
    }
    {
        Binder binder(reader);
        binder.set_segment_count(2);
        (void)binder.parameter("x", shape("x"), Residency::HostPinned, {}, 2);
        rejects([&] { (void)std::move(binder).finish(); },
                "a pinned tag beyond the segment count was accepted");
    }
    {
        Binder binder(reader);
        binder.set_segment_count(2);
        (void)binder.parameter("x", shape("x"));
        binder.mark_device_evictable(x, 1);
        rejects([&] { (void)std::move(binder).finish(); },
                "an evictable tail was accepted with two device segments");
    }
}

std::vector<std::byte> download(int device, const void* source, std::size_t bytes) {
    const DeviceGuard guard(device);
    std::vector<std::byte> out(bytes);
    CUDA_CHECK(cudaMemcpy(out.data(), source, bytes, cudaMemcpyDeviceToHost));
    return out;
}

bool equals_payload(const Fixture& fixture, std::string_view id, std::span<const std::byte> bytes) {
    const auto& item = object(id);
    return bytes.size() == item.bytes() &&
           std::memcmp(bytes.data(), fixture.payload.data() + item.offset, bytes.size()) == 0;
}

// Two segments on `devices`: every device object in its segment's arena at the planned offset,
// on that segment's device, with the source bytes; pinned objects with their bytes, and with
// pinned_per_segment in chunks allocated with their reading segment's device current.
void materialize_two(const Fixture& fixture, std::span<DeviceContext* const> devices,
                     bool pinned_per_segment) {
    Reader reader(fixture.entry);
    Binder binder(reader);
    binder.set_segment_count(2);
    const std::array<std::pair<const char*, std::uint8_t>, 5> dense = {
        {{"a0", 0}, {"b1", 1}, {"c0", 0}, {"d1", 1}, {"e1", 1}}};
    for (const auto& [id, segment] : dense) {
        (void)binder.parameter(id, shape(id), Residency::Device, {}, segment);
    }
    const std::array<std::pair<const char*, std::uint8_t>, 3> pinned = {
        {{"p0", 0}, {"p1", 1}, {"ps", kSharedSegment}}};
    for (const auto& [id, segment] : pinned) {
        (void)binder.parameter(id, shape(id), Residency::HostPinned, {}, segment);
    }
    auto plan               = std::move(binder).finish();
    plan.pinned_per_segment = pinned_per_segment;
    const auto placements   = plan.device_objects;
    const auto segments     = plan.device_segments;
    const auto backing      = materialize(reader, std::move(plan), devices);

    require(backing.device_segment_count() == 2, "materialized segment count differs");
    std::uint64_t bytes = 0;
    for (std::size_t s = 0; s < 2; ++s) {
        const auto arena = backing.device_arena(s);
        require(arena.data != nullptr && arena.bytes == segments[s].capacity_bytes,
                "a segment arena differs from its plan");
        cudaPointerAttributes attributes{};
        CUDA_CHECK(cudaPointerGetAttributes(&attributes, arena.data));
        require(attributes.type == cudaMemoryTypeDevice && attributes.device == devices[s]->device,
                "a segment arena is not on its segment's device");
    }
    for (const auto& placement : placements) {
        const auto& parent = backing.device_parent(placement.object);
        const auto arena   = backing.device_arena(placement.segment);
        require(backing.device_segment(placement.object) == placement.segment &&
                    parent.data == static_cast<std::byte*>(arena.data) + placement.offset,
                "a device object is not at its planned offset in its segment's arena");
        const auto id = object_id(reader.directory().object(placement.object));
        require(equals_payload(fixture, id,
                               download(devices[placement.segment]->device, parent.data,
                                        static_cast<std::size_t>(placement.bytes))),
                "a segment upload differs from the source bytes");
        bytes += placement.bytes;
    }
    const auto& stats = backing.stats();
    require(stats.device_capacity_bytes ==
                    segments[0].capacity_bytes + segments[1].capacity_bytes &&
                stats.h2d_bytes == bytes && stats.device_object_count == dense.size(),
            "segment upload statistics differ");
    for (const auto& [id, segment] : pinned) {
        const auto& parent = backing.pinned_parent(reader.find(id));
        const std::span<const std::byte> host(parent.data,
                                              static_cast<std::size_t>(object(id).bytes()));
        require(equals_payload(fixture, id, host), "a pinned object differs from the source bytes");
        // Its reading device (the first for a shared one) addresses it at its host address.
        const int reader_device = devices[segment == kSharedSegment ? 0 : segment]->device;
        const DeviceGuard guard(reader_device);
        cudaPointerAttributes attributes{};
        CUDA_CHECK(cudaPointerGetAttributes(&attributes, parent.data));
        require(attributes.type == cudaMemoryTypeHost && attributes.devicePointer == parent.data,
                "a pinned object is not device-addressable at its host address");
        if (pinned_per_segment && segment != kSharedSegment) {
            require(attributes.device == reader_device,
                    "a segment's pinned chunk was allocated on another device");
        }
        void* scratch = nullptr;
        CUDA_CHECK(cudaMalloc(&scratch, host.size()));
        CUDA_CHECK(cudaMemcpy(scratch, parent.data, host.size(), cudaMemcpyDefault));
        const auto copied = download(reader_device, scratch, host.size());
        CUDA_CHECK(cudaFree(scratch));
        require(equals_payload(fixture, id, copied),
                "a pinned object read by its segment's device differs");
    }
}

void materialization(const Fixture& fixture, DeviceContext& first, DeviceContext& second) {
    std::array<DeviceContext*, 2> devices = {&first, &second};
    materialize_two(fixture, devices, false);
    materialize_two(fixture, devices, true);

    Reader reader(fixture.entry);
    // The one-device overload: one segment, one arena with the one-arena offsets.
    {
        Binder binder(reader);
        for (const auto* id : {"a0", "b1", "c0", "d1", "e1"}) {
            (void)binder.parameter(id, shape(id));
        }
        const auto backing = materialize(reader, std::move(binder).finish(), first);
        const auto arena   = backing.device_arena(0);
        require(backing.device_segment_count() == 1 && arena.bytes == 8272 &&
                    backing.device_parent(reader.find("e1")).data ==
                        static_cast<std::byte*>(arena.data) + 8192 &&
                    backing.stats().device_capacity_bytes == 8272,
                "the one-device overload changed the one-arena materialization");
    }
    // A segment without device objects has no arena.
    {
        Binder binder(reader);
        binder.set_segment_count(2);
        (void)binder.parameter("a0", shape("a0"));
        const auto backing = materialize(reader, std::move(binder).finish(), devices);
        require(backing.device_arena(1).data == nullptr && backing.device_arena(1).bytes == 0 &&
                    backing.device_segment(reader.find("a0")) == 0,
                "an empty segment allocated an arena");
    }
    // One device per segment.
    {
        Binder binder(reader);
        binder.set_segment_count(2);
        (void)binder.parameter("a0", shape("a0"));
        rejects([&] { (void)materialize(reader, std::move(binder).finish(), first); },
                "a two-segment plan was materialized on one device");
    }
    {
        Binder binder(reader);
        (void)binder.parameter("a0", shape("a0"));
        rejects([&] { (void)materialize(reader, std::move(binder).finish(), devices); },
                "a one-segment plan was materialized on two devices");
    }
}

// --real: the dense objects of a real artifact (expert banks, the pinned embedding and the vision
// tower excluded) by the split's ownership rules: text/layers/i to segment i >= 24, the head, final
// mixer, MTP and proposal head to the last segment, the rest to the first.
int real_artifact() {
    const char* path = std::getenv("INFERNIX_QWEN4_ARTIFACT");
    if (path == nullptr || *path == '\0') {
        std::printf("SKIP: set INFERNIX_QWEN4_ARTIFACT\n");
        return 77;
    }
    constexpr std::uint32_t kSplit = 24;
    Reader reader(path);

    struct Selected {
        std::string name;
        Shape shape;
        std::uint8_t segment = 0;
    };

    std::vector<Selected> selected;
    for (const auto& [name, binding] : reader.directory().bindings) {
        if (name == "text/token_embedding" || name.starts_with("vision/")) { continue; }
        bool dense = true;
        for (const auto& part : binding.parts) {
            dense = dense &&
                    std::holds_alternative<TensorObject>(reader.directory().object(part.object)) &&
                    reader.geometry(part.object).format != QType::NVFP4_MUL;
        }
        if (!dense) { continue; }
        bool last = name == "text/output_head" || name.starts_with("text/final_mixer/") ||
                    name.starts_with("mtp/") || name.starts_with("proposal/");
        if (name.starts_with("text/layers/")) { last = std::stoul(name.substr(12)) >= kSplit; }
        const auto segment = static_cast<std::uint8_t>(last ? 1 : 0);
        auto logical       = binding.whole_object
                                 ? reader.directory().tensor(binding.parts.front().object).shape
                                 : Shape{binding.elements};
        selected.push_back({name, std::move(logical), segment});
    }
    DeviceContext first(0);
    DeviceContext second(0);

    std::map<std::size_t, std::vector<std::byte>> expected;
    {
        Binder binder(reader);
        for (const auto& item : selected) { (void)binder.parameter(item.name, item.shape); }
        auto plan             = std::move(binder).finish();
        const auto placements = plan.device_objects;
        const auto backing    = materialize(reader, std::move(plan), first);
        for (const auto& placement : placements) {
            expected.emplace(placement.object.index,
                             download(first.device, backing.device_parent(placement.object).data,
                                      static_cast<std::size_t>(placement.bytes)));
        }
    }

    Binder binder(reader);
    binder.set_segment_count(2);
    for (const auto& item : selected) {
        (void)binder.parameter(item.name, item.shape, Residency::Device, {}, item.segment);
    }
    auto plan                                   = std::move(binder).finish();
    const auto placements                       = plan.device_objects;
    const auto segments                         = plan.device_segments;
    const std::array<DeviceContext*, 2> devices = {&first, &second};
    const auto backing                          = materialize(reader, std::move(plan), devices);
    require(placements.size() == expected.size(), "the two loads placed different objects");
    std::array<std::uint64_t, 2> bytes{};
    std::array<std::size_t, 2> count{};
    for (const auto& placement : placements) {
        const auto found = expected.find(placement.object.index);
        require(found != expected.end(), "the two loads placed different objects");
        const auto& parent = backing.device_parent(placement.object);
        require(backing.device_segment(placement.object) == placement.segment &&
                    parent.data ==
                        static_cast<std::byte*>(backing.device_arena(placement.segment).data) +
                            placement.offset,
                "a dense object is not at its planned offset in its segment's arena");
        const auto actual = download(devices[placement.segment]->device, parent.data,
                                     static_cast<std::size_t>(placement.bytes));
        if (actual != found->second) {
            throw std::runtime_error(object_id(reader.directory().object(placement.object)) +
                                     ": two-segment bytes differ from the one-segment load");
        }
        bytes[placement.segment] += placement.bytes;
        ++count[placement.segment];
    }
    require(count[0] != 0 && count[1] != 0, "a segment received no dense object");
    constexpr double mib = 1024.0 * 1024.0;
    std::printf("%zu dense objects equal: segment 0 %zu (%.1f MiB of a %.1f MiB arena), "
                "segment 1 %zu (%.1f MiB of a %.1f MiB arena)\n",
                placements.size(), count[0], static_cast<double>(bytes[0]) / mib,
                static_cast<double>(segments[0].capacity_bytes) / mib, count[1],
                static_cast<double>(bytes[1]) / mib,
                static_cast<double>(segments[1].capacity_bytes) / mib);
    return 0;
}

int device_count() {
    int count         = 0;
    const auto result = cudaGetDeviceCount(&count);
    if (result == cudaErrorNoDevice || result == cudaErrorInsufficientDriver) {
        (void)cudaGetLastError();
        return 0;
    }
    CUDA_CHECK(result);
    return count;
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string_view(argv[1]) == "--real") {
            if (device_count() == 0) { return 77; }
            return real_artifact();
        }
        if (argc != 1) { throw std::invalid_argument("expected [--real]"); }
        Fixture fixture;
        write_synthetic(fixture);
        single_segment_layout(fixture.entry);
        two_segment_layout(fixture.entry);
        rejected_tags(fixture.entry);
        std::cout << "segment layout checks passed\n";
        const int devices = device_count();
        if (devices == 0) {
            std::cout << "SKIP: materialization needs a CUDA device\n";
            return 77;
        }
        {
            DeviceContext first(0);
            DeviceContext second(0);
            materialization(fixture, first, second);
            std::cout << "two segments on two contexts of device 0 passed\n";
        }
        if (devices >= 2) {
            DeviceContext zero(0);
            DeviceContext one(1);
            materialization(fixture, zero, one);
            materialization(fixture, one, zero);
            std::cout << "two segments on devices 0 and 1, both orders, passed\n";
        } else {
            std::cout << "one CUDA device: the two-device case did not run\n";
        }
        std::cout << "artifact segment materialization checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
