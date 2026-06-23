#include <exception>
#include <filesystem>
#include <iostream>
#include <string>

#include "centerpoint/io/bin_point_reader.hpp"
#include "centerpoint/io/debug_dump.hpp"
#include "centerpoint/voxelization.hpp"

namespace {

void print_usage(const char* executable) {
    std::cerr << "usage: " << executable << " <points.bin> <output_dir> [feature_dim]\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3 || argc > 4) {
        print_usage(argv[0]);
        return 2;
    }

    try {
        const std::filesystem::path input_path = argv[1];
        const std::filesystem::path output_dir = argv[2];
        const int feature_dim = argc == 4 ? std::stoi(argv[3]) : 5;

        centerpoint::VoxelizationConfig config;
        config.feature_dim = feature_dim;

        const centerpoint::PointCloud cloud =
            centerpoint::io::read_float32_point_cloud(input_path, feature_dim);
        const centerpoint::VoxelizationResult result = centerpoint::voxelize_cpu(cloud, config);

        centerpoint::io::write_debug_dump(output_dir, config, result);

        std::cout << "points: " << cloud.num_points() << "\n";
        std::cout << "pillars: " << result.num_pillars << "\n";
        std::cout << "dump: " << output_dir.string() << "\n";
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << "\n";
        return 1;
    }

    return 0;
}

