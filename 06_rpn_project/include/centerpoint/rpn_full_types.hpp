#pragma once

#include <array>
#include <string>
#include <vector>

namespace centerpoint {

struct HostTensor {
    std::vector<float> values;
    int channels = 0;
    int height = 0;
    int width = 0;
};

struct BatchNormWeights {
    std::vector<float> weight;
    std::vector<float> bias;
    std::vector<float> mean;
    std::vector<float> variance;
};

struct ConvLayerWeights {
    std::string name;
    std::vector<float> weight;
    BatchNormWeights batch_norm;
    int in_channels = 0;
    int out_channels = 0;
    int kernel_size = 0;
    int stride = 1;
    int padding = 0;
};

struct TransposedConvLayerWeights {
    std::string name;
    std::vector<float> gemm_weight;  // [out_channels*k*k, in_channels]
    BatchNormWeights batch_norm;
    int in_channels = 0;
    int out_channels = 0;
    int kernel_size = 0;
    int stride = 0;
};

struct FullRpnWeights {
    std::array<std::vector<ConvLayerWeights>, 3> blocks;
    ConvLayerWeights deblock0;
    TransposedConvLayerWeights deblock1;
    TransposedConvLayerWeights deblock2;
    float batch_norm_eps = 1.0e-3F;
};

struct FullRpnResult {
    HostTensor output;
    float elapsed_ms = 0.0F;
    std::array<std::array<int, 3>, 3> block_shapes{};
    std::array<std::array<int, 3>, 3> deblock_shapes{};
};

}  // namespace centerpoint
