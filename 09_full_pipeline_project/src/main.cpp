#include "centerpoint/waymo_sensor_archive.hpp"

#include <algorithm>
#include <exception>
#include <iomanip>
#include <iostream>

namespace {

void print_usage(const char* executable) {
    std::cerr << "usage: " << executable << " <waymo_centerpoint_points.bin>\n";
}

void print_row(const char* name,
               const std::array<float, 5>& values,
               int precision = 4) {
    std::cout << name << ": ";
    std::cout << std::fixed << std::setprecision(precision)
              << "x=" << values[0]
              << ", y=" << values[1]
              << ", z=" << values[2]
              << ", intensity=" << values[3]
              << ", elongation=" << values[4] << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        print_usage(argv[0]);
        return 2;
    }

    try {
        const centerpoint::WaymoPointCloud cloud =
            centerpoint::read_waymo_centerpoint_bin(argv[1]);
        const centerpoint::PointStats stats = centerpoint::compute_stats(cloud);

        std::cout << "points: " << cloud.size() << '\n';
        print_row("min", stats.min_values);
        print_row("max", stats.max_values);

        std::array<float, 5> mean{};
        for (std::size_t i = 0; i < mean.size(); ++i) {
            mean[i] = static_cast<float>(stats.mean_values[i]);
        }
        print_row("mean", mean);

        const std::size_t sample_count = std::min<std::size_t>(cloud.size(), 5);
        for (std::size_t i = 0; i < sample_count; ++i) {
            const auto& p = cloud.points[i];
            std::cout << "sample[" << i << "]: "
                      << std::fixed << std::setprecision(4)
                      << p.x << ", " << p.y << ", " << p.z << ", "
                      << p.intensity << ", " << p.elongation << '\n';
        }
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}

