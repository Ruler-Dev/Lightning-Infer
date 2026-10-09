// The route trace's environment fallback (host only): without the test seam, INFERNIX_ROUTE_TRACE
// names the file construct_qwen4_exp passes to ProgramOptions::route_trace, and a RouteTrace opened
// on it writes the NRTRACE1 header; the seam still wins over the variable, and neither set means no
// trace.

#include "models/qwen4_exp/program/route_trace.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

using namespace infernix::models::qwen4_exp;

namespace {

void require(bool value, const std::string& message) {
    if (!value) { throw std::runtime_error(message); }
}

void set_variable(const std::filesystem::path& value) {
#ifdef _WIN32
    require(_wputenv_s(L"INFERNIX_ROUTE_TRACE", value.c_str()) == 0, "set INFERNIX_ROUTE_TRACE");
#else
    require(setenv("INFERNIX_ROUTE_TRACE", value.c_str(), 1) == 0, "set INFERNIX_ROUTE_TRACE");
#endif
}

void clear_variable() {
#ifdef _WIN32
    require(_wputenv_s(L"INFERNIX_ROUTE_TRACE", L"") == 0, "clear INFERNIX_ROUTE_TRACE");
#else
    require(unsetenv("INFERNIX_ROUTE_TRACE") == 0, "clear INFERNIX_ROUTE_TRACE");
#endif
}

} // namespace

int main() {
    const std::filesystem::path dir  = std::filesystem::temp_directory_path();
    const std::filesystem::path file = dir / "infernix_route_trace_env_test.bin";
    const std::filesystem::path seam = dir / "infernix_route_trace_env_seam.bin";
    try {
        std::filesystem::remove(file);
        testing::set_route_trace({});
        clear_variable();
        require(testing::route_trace().empty(), "no seam and no variable: no trace");

        set_variable(file);
        require(testing::route_trace() == file, "the variable names the trace without the seam");
        testing::set_route_trace(seam);
        require(testing::route_trace() == seam, "the seam wins over the variable");
        testing::set_route_trace({});
        require(testing::route_trace() == file, "the variable again once the seam is cleared");

        {
            const RouteTraceSetup setup{.layers             = 48,
                                        .experts            = 512,
                                        .top_k              = 10,
                                        .frames             = 2257,
                                        .max_columns        = 4096,
                                        .lanes              = 1,
                                        .max_width          = 4,
                                        .mtp_draft_tokens   = 3,
                                        .ngram_draft_tokens = 0,
                                        .prefill_chunk      = 4096};
            RouteTrace trace(testing::route_trace(), setup);
        }
        require(std::filesystem::exists(file), "the fallback path produces a trace file");
        require(std::filesystem::file_size(file) == 64, "the trace holds its 64-byte header");
        std::array<std::uint32_t, 16> header{};
        std::ifstream in(file, std::ios::binary);
        in.read(reinterpret_cast<char*>(header.data()), sizeof(header));
        require(in.good() && std::memcmp(header.data(), "NRTRACE1", 8) == 0, "the header's magic");
        require(header[2] == 1 && header[3] == 48 && header[6] == 2257, "the header's version, layers and frames");
        in.close();

        clear_variable();
        require(testing::route_trace().empty(), "an empty variable means no trace");
        std::filesystem::remove(file);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
    std::printf("qwen4_exp route trace environment checks passed\n");
    return 0;
}
