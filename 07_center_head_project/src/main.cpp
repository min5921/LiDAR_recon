#include "centerpoint/head.hpp"
#include <exception>
#include <iomanip>
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "usage: " << argv[0] << " <rpn_output_dir> <weight_dir> <output_dir>\n";
        return 2;
    }
    try {
        const auto input = centerpoint::read_rpn_output(argv[1]);
        const auto weights = centerpoint::read_head_weights(argv[2]);
        const auto result = centerpoint::run_center_head_cuda(input, weights);
        centerpoint::write_head_output(argv[3], result);
        std::cout << "input: [1," << input.channels << ',' << input.height << ',' << input.width << "]\n";
        for (const auto& output : result.outputs)
            std::cout << "output: [1," << output.channels << ',' << output.height << ',' << output.width << "]\n";
        std::cout << std::fixed << std::setprecision(3) << "CUDA time: " << result.elapsed_ms << " ms\n";
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n'; return 1;
    }
    return 0;
}
