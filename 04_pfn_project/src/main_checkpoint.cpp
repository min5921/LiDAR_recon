#include <chrono>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>

#include "centerpoint/io/decorated_dump_reader.hpp"
#include "centerpoint/io/pfn_checkpoint_reader.hpp"
#include "centerpoint/io/pillar_feature_writer.hpp"
#include "centerpoint/pfn_two_layer.hpp"

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "usage: " << argv[0]
                  << " <decorated_dump_dir> <weight_dir> <output_dir>\n";
        return 2;
    }

    try {
        const centerpoint::DecoratedPillarDump dump =
            centerpoint::io::read_decorated_pillar_dump(argv[1]);
        const centerpoint::TwoLayerPfnCheckpointWeights weights =
            centerpoint::io::read_two_layer_pfn_weights(argv[2]);

        const auto start = std::chrono::steady_clock::now();
        const centerpoint::PillarFeatureResult result =
            centerpoint::run_two_layer_pfn_cpu(dump, weights);
        const auto stop = std::chrono::steady_clock::now();
        const double elapsed_ms =
            std::chrono::duration<double, std::milli>(stop - start).count();

        centerpoint::io::write_pillar_features(argv[3], result);
        std::cout << "input shape: [" << dump.metadata.num_pillars << ", "
                  << dump.metadata.max_points_per_pillar << ", "
                  << dump.metadata.decorated_feature_dim << "]\n";
        std::cout << "PFN layers: [10 -> 32 -> 64] then [64 -> 64]\n";
        std::cout << "output shape: [" << result.num_pillars << ", "
                  << result.out_channels << "]\n";
        std::cout << std::fixed << std::setprecision(3)
                  << "CPU time: " << elapsed_ms << " ms\n";
        std::cout << "dump: " << argv[3] << "\n";
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << "\n";
        return 1;
    }
    return 0;
}
