#include "centerpoint/waymo_sensor_archive.hpp"

#include <algorithm>
#include <array>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace centerpoint {
namespace {

constexpr std::size_t kFeatureCount = 5;
constexpr std::size_t kPointBytes = sizeof(float) * kFeatureCount;

std::array<float, kFeatureCount> to_array(const WaymoPoint& p) {
    return {p.x, p.y, p.z, p.intensity, p.elongation};
}

}  // namespace

WaymoPointCloud read_waymo_centerpoint_bin(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("failed to open point file: " + path.string());
    }

    const std::streamsize byte_size = input.tellg();
    if (byte_size < 0) {
        throw std::runtime_error("failed to get file size: " + path.string());
    }
    if (static_cast<std::size_t>(byte_size) % kPointBytes != 0U) {
        throw std::runtime_error("file size is not divisible by 5 float32 values: " +
                                 path.string());
    }

    input.seekg(0, std::ios::beg);
    WaymoPointCloud cloud;
    cloud.points.resize(static_cast<std::size_t>(byte_size) / kPointBytes);
    if (!cloud.points.empty()) {
        input.read(reinterpret_cast<char*>(cloud.points.data()), byte_size);
        if (!input) {
            throw std::runtime_error("failed to read all point bytes: " + path.string());
        }
    }
    return cloud;
}

PointStats compute_stats(const WaymoPointCloud& cloud) {
    if (cloud.empty()) {
        throw std::runtime_error("cannot compute stats for an empty point cloud");
    }

    PointStats stats;
    stats.min_values.fill(std::numeric_limits<float>::max());
    stats.max_values.fill(std::numeric_limits<float>::lowest());
    stats.mean_values.fill(0.0);

    for (const WaymoPoint& point : cloud.points) {
        const auto values = to_array(point);
        for (std::size_t i = 0; i < kFeatureCount; ++i) {
            stats.min_values[i] = std::min(stats.min_values[i], values[i]);
            stats.max_values[i] = std::max(stats.max_values[i], values[i]);
            stats.mean_values[i] += values[i];
        }
    }

    for (double& value : stats.mean_values) {
        value /= static_cast<double>(cloud.size());
    }
    return stats;
}

}  // namespace centerpoint

