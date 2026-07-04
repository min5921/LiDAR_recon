#include <exception>
#include <filesystem>
#include <iostream>

#include "centerpoint/io/bev_feature_writer.hpp"
#include "centerpoint/io/scatter_input_reader.hpp"
#include "centerpoint/scatter.hpp"

namespace {

void print_usage(const char* executable) {
    std::cerr << "usage: " << executable
              << " <pfn_dump_dir> <voxel_dump_dir> <output_dir>\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        print_usage(argv[0]);
        return 2;
    }
    try {
        const std::filesystem::path pfn_dump_dir = argv[1];
        const std::filesystem::path voxel_dump_dir = argv[2];
        const std::filesystem::path output_dir = argv[3];
        const centerpoint::ScatterInput input =
            centerpoint::io::read_scatter_input(pfn_dump_dir, voxel_dump_dir);
        const centerpoint::BevFeatureResult result =
            centerpoint::scatter_pillars_cpu(input);
        centerpoint::io::write_bev_features(output_dir, result);

        std::cout << "input pillars: " << input.num_pillars << "\n";
        std::cout << "input shape: [" << input.num_pillars << ", "
                  << input.channels << "]\n";
        std::cout << "BEV shape: [" << result.batch_size << ", "
                  << result.channels << ", " << result.height << ", "
                  << result.width << "]\n";
        std::cout << "occupied cells: " << result.occupied_cells << "\n";
        std::cout << "dump: " << output_dir.string() << "\n";
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << "\n";
        return 1;
    }
    return 0;
}
