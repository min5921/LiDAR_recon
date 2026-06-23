#include "centerpoint/io/voxel_dump_reader.hpp"

#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>

namespace centerpoint::io {
namespace {

std::string read_text(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open metadata file: " + path.string());
    }
    return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

int read_int_field(const std::string& text, const std::string& name) {
    const std::regex pattern("\"" + name + "\"\\s*:\\s*(-?\\d+)");
    std::smatch match;
    if (!std::regex_search(text, match, pattern)) {
        throw std::runtime_error("missing integer metadata field: " + name);
    }
    return std::stoi(match[1].str());
}

template <typename T, std::size_t N>
std::array<T, N> read_array_field(const std::string& text, const std::string& name) {
    const std::regex pattern("\"" + name + "\"\\s*:\\s*\\[([^\\]]+)\\]");
    std::smatch match;
    if (!std::regex_search(text, match, pattern)) {
        throw std::runtime_error("missing array metadata field: " + name);
    }

    std::array<T, N> values{};
    std::stringstream stream(match[1].str());
    for (std::size_t i = 0; i < N; ++i) {
        std::string token;
        if (!std::getline(stream, token, ',')) {
            throw std::runtime_error("metadata array has too few values: " + name);
        }
        if constexpr (std::is_same_v<T, int>) {
            values[i] = std::stoi(token);
        } else {
            values[i] = std::stof(token);
        }
    }
    return values;
}

template <typename T>
std::vector<T> read_binary_vector(const std::filesystem::path& path) {
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

VoxelDump read_voxel_dump(const std::filesystem::path& dump_dir) {
    const std::string metadata_text = read_text(dump_dir / "metadata.json");

    VoxelDump dump;
    dump.metadata.num_pillars = read_int_field(metadata_text, "num_pillars");
    dump.metadata.max_points_per_pillar = read_int_field(metadata_text, "max_points_per_pillar");
    dump.metadata.feature_dim = read_int_field(metadata_text, "feature_dim");
    dump.metadata.grid_size_xyz = read_array_field<int, 3>(metadata_text, "grid_size_xyz");
    dump.metadata.voxel_size = read_array_field<float, 3>(metadata_text, "voxel_size");
    dump.metadata.point_cloud_range = read_array_field<float, 6>(metadata_text, "point_cloud_range");

    dump.pillars = read_binary_vector<float>(dump_dir / "pillars.bin");
    dump.coordinates = read_binary_vector<int32_t>(dump_dir / "coordinates.bin");
    dump.num_points_per_pillar = read_binary_vector<int32_t>(dump_dir / "num_points.bin");

    const std::size_t expected_pillars =
        static_cast<std::size_t>(dump.metadata.num_pillars) *
        dump.metadata.max_points_per_pillar *
        dump.metadata.feature_dim;
    const std::size_t expected_coordinates = static_cast<std::size_t>(dump.metadata.num_pillars) * 4;
    const std::size_t expected_num_points = static_cast<std::size_t>(dump.metadata.num_pillars);

    if (dump.pillars.size() != expected_pillars ||
        dump.coordinates.size() != expected_coordinates ||
        dump.num_points_per_pillar.size() != expected_num_points) {
        throw std::runtime_error("voxel dump binary sizes do not match metadata");
    }

    return dump;
}

}  // namespace centerpoint::io
