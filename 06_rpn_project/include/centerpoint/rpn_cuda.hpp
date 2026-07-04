#pragma once

#include <vector>

namespace centerpoint {

struct ConvBnReluConfig {
    int batch = 1;
    int in_channels = 0;
    int out_channels = 0;
    int input_height = 0;
    int input_width = 0;
    int kernel_size = 3;
    int stride = 1;
    int padding = 1;
    float batch_norm_eps = 1.0e-3F;
};

struct ConvBnReluResult {
    std::vector<float> output;
    int batch = 0;
    int channels = 0;
    int height = 0;
    int width = 0;
    float elapsed_ms = 0.0F;
};

ConvBnReluResult run_conv_bn_relu_cuda(
    const std::vector<float>& input,
    const std::vector<float>& weights,
    const std::vector<float>& bn_weight,
    const std::vector<float>& bn_bias,
    const std::vector<float>& bn_mean,
    const std::vector<float>& bn_var,
    const ConvBnReluConfig& config);

}  // namespace centerpoint
