#include "centerpoint/io/scatter_input_reader.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

namespace centerpoint::io {
namespace {

std::string read_text(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open metadata file: " + path.string());
    }
    return std::string(std::istreambuf_iterator<char>(input),
                       std::istreambuf_iterator<char>());
}

int read_int_field(const std::string& text, const std::string& name) {
    const std::regex pattern("\"" + name + "\"\\s*:\\s*(-?\\d+)");
    std::smatch match;
    if (!std::regex_search(text, match, pattern)) {
        throw std::runtime_error("missing integer metadata field: " + name);
    }
    return std::stoi(match[1].str());
}

std::vector<int> read_int_array(const std::string& text,
                                const std::string& name,
                                int expected_size) {
    const std::regex pattern("\"" + name + "\"\\s*:\\s*\\[([^\\]]+)\\]");
    std::smatch match;
    if (!std::regex_search(text, match, pattern)) {
        throw std::runtime_error("missing integer metadata array: " + name);
    }

    std::vector<int> values;
    const std::string contents = match[1].str();
    const std::regex integer_pattern("-?\\d+");
    for (std::sregex_iterator it(contents.begin(), contents.end(), integer_pattern), end;
         it != end; ++it) {
        values.push_back(std::stoi(it->str()));
    }
    if (static_cast<int>(values.size()) != expected_size) {
        throw std::runtime_error("unexpected metadata array size: " + name);
    }
    return values;
}

template <typename T>
std::vector<T> read_binary(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("failed to open binary file: " + path.string());
    }
    const auto byte_size = input.tellg();
    if (byte_size < 0 || static_cast<std::uintmax_t>(byte_size) % sizeof(T) != 0) {
        throw std::runtime_error("invalid binary file size: " + path.string());
    }

    std::vector<T> values(static_cast<std::size_t>(byte_size) / sizeof(T));
    input.seekg(0, std::ios::beg);
    input.read(reinterpret_cast<char*>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(T)));
    if (!input) {
        throw std::runtime_error("failed to read binary file: " + path.string());
    }
    return values;
}

}  // namespace

ScatterInput read_scatter_input(const std::filesystem::path& pfn_dump_dir,
                                const std::filesystem::path& voxel_dump_dir) {
    const std::string pfn_metadata =
        read_text(pfn_dump_dir / "pillar_features_metadata.json");
    const std::string voxel_metadata = read_text(voxel_dump_dir / "metadata.json");

    ScatterInput input;
    input.num_pillars = read_int_field(pfn_metadata, "num_pillars");
    input.channels = read_int_field(pfn_metadata, "out_channels");
    const int coordinate_pillars = read_int_field(voxel_metadata, "num_pillars");
    if (input.num_pillars != coordinate_pillars) {
        throw std::runtime_error("PFN and coordinate pillar counts do not match");
    }

    const std::vector<int> grid = read_int_array(voxel_metadata, "grid_size_xyz", 3);
    input.grid_x = grid[0];
    input.grid_y = grid[1];
    input.grid_z = grid[2];
    input.pillar_features = read_binary<float>(pfn_dump_dir / "pillar_features.bin");
    input.coordinates = read_binary<std::int32_t>(voxel_dump_dir / "coordinates.bin");

    const std::size_t expected_features =
        static_cast<std::size_t>(input.num_pillars) * input.channels;
    const std::size_t expected_coordinates =
        static_cast<std::size_t>(input.num_pillars) * 4;
    if (input.pillar_features.size() != expected_features) {
        throw std::runtime_error("pillar_features.bin size does not match metadata");
    }
    if (input.coordinates.size() != expected_coordinates) {
        throw std::runtime_error("coordinates.bin size does not match metadata");
    }
    return input;
}

}  // namespace centerpoint::io
