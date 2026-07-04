#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>

#include "centerpoint/io/bev_dump_reader.hpp"
#include "centerpoint/io/rpn_output_writer.hpp"
#include "centerpoint/io/rpn_weight_reader.hpp"
#include "centerpoint/rpn_full_cuda.hpp"

int main(int argc, char** argv) {
    if (argc < 4 || argc > 5) {
        std::cerr << "usage: " << argv[0]
                  << " <bev_dump_dir> <weight_dir> <output_dir> [--summary-only]\n";
        return 2;
    }

    try {
        const bool write_tensor =
            !(argc == 5 && std::string(argv[4]) == "--summary-only");
        const centerpoint::HostTensor input =
            centerpoint::io::read_bev_dump(argv[1]);
        const centerpoint::FullRpnWeights weights =
            centerpoint::io::read_full_rpn_weights(argv[2]);
        const centerpoint::FullRpnResult result =
            centerpoint::run_full_rpn_cuda(input, weights);
        centerpoint::io::write_full_rpn_output(argv[3], result, write_tensor);

        std::cout << "input: [1, " << input.channels << ", " << input.height
                  << ", " << input.width << "]\n";
        for (int index = 0; index < 3; ++index) {
            const auto& block = result.block_shapes[index];
            const auto& deblock = result.deblock_shapes[index];
            std::cout << "block " << index << ": [1, " << block[0] << ", "
                      << block[1] << ", " << block[2] << "]\n";
            std::cout << "deblock " << index << ": [1, " << deblock[0] << ", "
                      << deblock[1] << ", " << deblock[2] << "]\n";
        }
        std::cout << "output: [1, " << result.output.channels << ", "
                  << result.output.height << ", " << result.output.width << "]\n";
        std::cout << std::fixed << std::setprecision(3)
                  << "CUDA time: " << result.elapsed_ms << " ms\n";
        std::cout << "tensor written: " << (write_tensor ? "yes" : "no") << "\n";
        std::cout << "dump: " << argv[3] << "\n";
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << "\n";
        return 1;
    }
    return 0;
}
