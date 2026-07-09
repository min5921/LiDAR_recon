#pragma once

#include <array>
#include <cstddef>
#include <filesystem>
#include <vector>

namespace centerpoint {

struct WaymoPoint {
    float x;
    float y;
    float z;
    float intensity;
    float elongation;
};

struct WaymoPointCloud {
    std::vector<WaymoPoint> points;

    std::size_t size() const noexcept {
        return points.size();
    }

    bool empty() const noexcept {
        return points.empty();
    }
};

struct PointStats {
    std::array<float, 5> min_values{};
    std::array<float, 5> max_values{};
    std::array<double, 5> mean_values{};
};

WaymoPointCloud read_waymo_centerpoint_bin(const std::filesystem::path& path);
PointStats compute_stats(const WaymoPointCloud& cloud);

}  // namespace centerpoint

