#include <exception>
#include <filesystem>
#include <iostream>
#include <string>

#include "centerpoint/io/decorated_dump_reader.hpp"
#include "centerpoint/io/pillar_feature_writer.hpp"
#include "centerpoint/pfn.hpp"

namespace {

void print_usage(const char* executable) {
    std::cerr << "usage: " << executable << " <decorated_dump_dir> <output_dir> [out_channels]\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3 || argc > 4) {
        print_usage(argv[0]);
        return 2;
    }

    try {
        const std::filesystem::path input_dir = argv[1];
        const std::filesystem::path output_dir = argv[2];
        const int out_channels = argc == 4 ? std::stoi(argv[3]) : 64;

        const centerpoint::DecoratedPillarDump dump =
            centerpoint::io::read_decorated_pillar_dump(input_dir);

        centerpoint::PfnConfig config;
        config.out_channels = out_channels;
        const centerpoint::PfnWeights weights =
            centerpoint::make_dummy_pfn_weights(dump.metadata.decorated_feature_dim, out_channels);
        const centerpoint::PillarFeatureResult result =
            centerpoint::run_pfn_cpu(dump, config, weights);

        centerpoint::io::write_pillar_features(output_dir, result);

        std::cout << "pillars: " << result.num_pillars << "\n";
        std::cout << "in channels: " << dump.metadata.decorated_feature_dim << "\n";
        std::cout << "out channels: " << result.out_channels << "\n";
        std::cout << "dump: " << output_dir.string() << "\n";
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << "\n";
        return 1;
    }

    return 0;
}
