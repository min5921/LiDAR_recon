#include <exception>
#include <filesystem>
#include <iostream>

#include "centerpoint/io/decorated_dump_writer.hpp"
#include "centerpoint/io/voxel_dump_reader.hpp"
#include "centerpoint/pillar_feature.hpp"

namespace {

void print_usage(const char* executable) {
    std::cerr << "usage: " << executable << " <voxel_dump_dir> <output_dir>\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        print_usage(argv[0]);
        return 2;
    }

    try {
        const std::filesystem::path input_dir = argv[1];
        const std::filesystem::path output_dir = argv[2];

        const centerpoint::VoxelDump dump = centerpoint::io::read_voxel_dump(input_dir);
        const centerpoint::DecoratedPillarResult result = centerpoint::decorate_pillars_cpu(dump);

        centerpoint::io::write_decorated_dump(output_dir, dump.metadata, result);

        std::cout << "pillars: " << result.num_pillars << "\n";
        std::cout << "input feature dim: " << result.input_feature_dim << "\n";
        std::cout << "decorated feature dim: " << result.decorated_feature_dim << "\n";
        std::cout << "dump: " << output_dir.string() << "\n";
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << "\n";
        return 1;
    }

    return 0;
}

