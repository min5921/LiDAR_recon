#include <algorithm>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <vector>

#include "centerpoint/rpn_cuda.hpp"
#include "centerpoint/rpn_dump_writer.hpp"

namespace {

std::vector<float> make_input(const centerpoint::ConvBnReluConfig& config) {
    const int count = config.batch * config.in_channels *
                      config.input_height * config.input_width;
    std::vector<float> values(static_cast<std::size_t>(count));
    for (int index = 0; index < count; ++index) {
        values[static_cast<std::size_t>(index)] =
            static_cast<float>((index * 13) % 29 - 14) * 0.1F;
    }
    return values;
}

std::vector<float> make_weights(const centerpoint::ConvBnReluConfig& config) {
    const int count = config.out_channels * config.in_channels *
                      config.kernel_size * config.kernel_size;
    std::vector<float> values(static_cast<std::size_t>(count));
    for (int index = 0; index < count; ++index) {
        values[static_cast<std::size_t>(index)] =
            static_cast<float>((index * 7) % 23 - 11) * 0.01F;
    }
    return values;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: " << argv[0] << " <output_dir>\n";
        return 2;
    }

    try {
        centerpoint::ConvBnReluConfig config;
        config.batch = 1;
        config.in_channels = 3;
        config.out_channels = 4;
        config.input_height = 8;
        config.input_width = 8;
        config.kernel_size = 3;
        config.stride = 1;
        config.padding = 1;

        const std::vector<float> input = make_input(config);
        const std::vector<float> weights = make_weights(config);
        std::vector<float> bn_weight(config.out_channels);
        std::vector<float> bn_bias(config.out_channels);
        std::vector<float> bn_mean(config.out_channels);
        std::vector<float> bn_var(config.out_channels);
        for (int channel = 0; channel < config.out_channels; ++channel) {
            bn_weight[channel] = 1.0F + static_cast<float>(channel) * 0.05F;
            bn_bias[channel] = static_cast<float>(channel - 1) * 0.02F;
            bn_mean[channel] = static_cast<float>(channel) * 0.1F;
            bn_var[channel] = 1.0F + static_cast<float>(channel) * 0.2F;
        }

        const centerpoint::ConvBnReluResult result =
            centerpoint::run_conv_bn_relu_cuda(
                input, weights, bn_weight, bn_bias, bn_mean, bn_var, config);
        centerpoint::io::write_rpn_demo_dump(
            argv[1], input, weights, bn_weight, bn_bias, bn_mean, bn_var,
            config, result);

        std::cout << "input shape: [" << config.batch << ", "
                  << config.in_channels << ", " << config.input_height << ", "
                  << config.input_width << "]\n";
        std::cout << "weight shape: [" << config.out_channels << ", "
                  << config.in_channels << ", " << config.kernel_size << ", "
                  << config.kernel_size << "]\n";
        std::cout << "output shape: [" << result.batch << ", "
                  << result.channels << ", " << result.height << ", "
                  << result.width << "]\n";
        std::cout << std::fixed << std::setprecision(6)
                  << "kernel time: " << result.elapsed_ms << " ms\n";
        std::cout << "first output values:";
        const std::size_t count = std::min<std::size_t>(8, result.output.size());
        for (std::size_t index = 0; index < count; ++index) {
            std::cout << ' ' << result.output[index];
        }
        std::cout << "\ndump: " << argv[1] << "\n";
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << "\n";
        return 1;
    }
    return 0;
}
