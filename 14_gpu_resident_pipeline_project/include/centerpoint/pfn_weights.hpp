#pragma once

#include <filesystem>
#include <vector>

namespace centerpoint {

struct PfnLayerWeights {
    std::vector<float> linear;
    std::vector<float> bn_weight;
    std::vector<float> bn_bias;
    std::vector<float> bn_mean;
    std::vector<float> bn_variance;
    int in_channels = 0;
    int out_channels = 0;
};

struct PfnWeights {
    PfnLayerWeights layer0;
    PfnLayerWeights layer1;
    float batch_norm_epsilon = 1.0e-3F;
};

PfnWeights load_pfn_weights(const std::filesystem::path& directory);

}  // namespace centerpoint
