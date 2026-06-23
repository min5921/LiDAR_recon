#include "centerpoint/io/bin_point_reader.hpp"

#include <fstream>
#include <stdexcept>

namespace centerpoint::io {

PointCloud read_float32_point_cloud(const std::filesystem::path& path, int feature_dim) {
    if (feature_dim < 3) {
        throw std::runtime_error("feature_dim must be at least 3");
    }

    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("failed to open point cloud file: " + path.string());
    }

    const auto byte_size = input.tellg();
    if (byte_size < 0 || static_cast<std::uintmax_t>(byte_size) % sizeof(float) != 0) {
        throw std::runtime_error("point cloud file size is not a float32 multiple");
    }

    const auto float_count = static_cast<std::size_t>(byte_size) / sizeof(float);
    if (float_count % static_cast<std::size_t>(feature_dim) != 0) {
        throw std::runtime_error("point cloud float count is not divisible by feature_dim");
    }

    PointCloud cloud;
    cloud.feature_dim = feature_dim;
    cloud.values.resize(float_count);

    input.seekg(0, std::ios::beg);
    input.read(reinterpret_cast<char*>(cloud.values.data()),
               static_cast<std::streamsize>(float_count * sizeof(float)));
    if (!input) {
        throw std::runtime_error("failed to read point cloud file: " + path.string());
    }

    return cloud;
}

}  // namespace centerpoint::io

