#pragma once

#include <array>
#include <filesystem>
#include <string>
#include <vector>

namespace centerpoint {

struct RpnBatchNormWeights {
    std::vector<float> weight;
    std::vector<float> bias;
    std::vector<float> mean;
    std::vector<float> variance;
};

struct RpnConvWeights {
    std::string name;
    std::vector<float> weight;
    RpnBatchNormWeights batch_norm;
    int in_channels = 0;
    int out_channels = 0;
    int kernel_size = 0;
    int stride = 1;
    int padding = 0;
};

struct RpnDeconvWeights {
    std::string name;
    std::vector<float> gemm_weight;
    RpnBatchNormWeights batch_norm;
    int in_channels = 0;
    int out_channels = 0;
    int kernel_size = 0;
    int stride = 0;
};

struct RpnWeights {
    std::array<std::vector<RpnConvWeights>, 3> blocks;
    RpnConvWeights deblock0;
    RpnDeconvWeights deblock1;
    RpnDeconvWeights deblock2;
    float batch_norm_epsilon = 1.0e-3F;
};

RpnWeights load_rpn_weights(const std::filesystem::path& directory);

}  // namespace centerpoint
