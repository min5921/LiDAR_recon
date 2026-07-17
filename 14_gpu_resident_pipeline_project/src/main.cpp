#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "centerpoint/gpu_preprocess.hpp"
#include "centerpoint/io.hpp"
#include "centerpoint/pfn_weights.hpp"

namespace {

struct Arguments {
    std::filesystem::path points;
    std::filesystem::path weights;
    std::filesystem::path output_directory;
    bool write_output = false;
};

void print_usage(const char* executable) {
    std::cerr << "usage: " << executable
              << " <points.bin> <pfn_weight_dir> [--output-dir <directory>]\n";
}

Arguments parse_arguments(int argc, char** argv) {
    if (argc != 3 && argc != 5) {
        throw std::invalid_argument("invalid argument count");
    }
    Arguments arguments;
    arguments.points = argv[1];
    arguments.weights = argv[2];
    if (argc == 5) {
        if (std::string(argv[3]) != "--output-dir") {
            throw std::invalid_argument("expected --output-dir");
        }
        arguments.output_directory = argv[4];
        arguments.write_output = true;
    }
    return arguments;
}

void write_summary(const std::filesystem::path& path,
                   const centerpoint::GpuPreprocessStats& stats,
                   const centerpoint::DeviceBevView& bev) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("failed to create summary: " + path.string());
    }
    output << std::fixed << std::setprecision(6)
           << "{\n"
           << "  \"input_points\": " << stats.input_points << ",\n"
           << "  \"valid_points\": " << stats.valid_points << ",\n"
           << "  \"unique_pillars\": " << stats.unique_pillars << ",\n"
           << "  \"selected_pillars\": " << stats.selected_pillars << ",\n"
           << "  \"bev_shape\": [1, " << bev.channels << ", "
           << bev.height << ", " << bev.width << "],\n"
           << "  \"host_to_device_ms\": " << stats.host_to_device_ms << ",\n"
           << "  \"voxelization_ms\": " << stats.voxelization_ms << ",\n"
           << "  \"pfn_ms\": " << stats.pfn_ms << ",\n"
           << "  \"scatter_ms\": " << stats.scatter_ms << ",\n"
           << "  \"total_ms\": " << stats.total_ms << "\n"
           << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Arguments arguments = parse_arguments(argc, argv);
        const centerpoint::GpuPreprocessConfig config;
        const std::vector<float> points =
            centerpoint::read_point_bin(arguments.points,
                                        config.feature_dimension);
        const int point_count = static_cast<int>(
            points.size() / static_cast<std::size_t>(config.feature_dimension));
        const centerpoint::PfnWeights weights =
            centerpoint::load_pfn_weights(arguments.weights);

        centerpoint::GpuPreprocessPipeline pipeline(config, weights);
        const centerpoint::GpuPreprocessStats stats =
            pipeline.run(points.data(), point_count);
        const centerpoint::DeviceBevView bev = pipeline.device_bev();

        std::cout << std::fixed << std::setprecision(3)
                  << "input points: " << stats.input_points << '\n'
                  << "valid points: " << stats.valid_points << '\n'
                  << "unique pillars: " << stats.unique_pillars << '\n'
                  << "selected pillars: " << stats.selected_pillars << '\n'
                  << "BEV shape: [1, " << bev.channels << ", "
                  << bev.height << ", " << bev.width << "]\n"
                  << "H2D: " << stats.host_to_device_ms << " ms\n"
                  << "voxelization: " << stats.voxelization_ms << " ms\n"
                  << "decoration + PFN: " << stats.pfn_ms << " ms\n"
                  << "scatter: " << stats.scatter_ms << " ms\n"
                  << "GPU front-end total: " << stats.total_ms << " ms\n";

        if (arguments.write_output) {
            std::filesystem::create_directories(arguments.output_directory);
            const std::vector<float> host_bev = pipeline.copy_bev_to_host();
            centerpoint::write_float_bin(
                arguments.output_directory / "bev_features.bin", host_bev);
            write_summary(arguments.output_directory / "summary.json", stats, bev);
            std::cout << "verification output: "
                      << arguments.output_directory.string() << '\n';
        }
        return 0;
    } catch (const std::invalid_argument& error) {
        print_usage(argv[0]);
        std::cerr << "error: " << error.what() << '\n';
        return 2;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
