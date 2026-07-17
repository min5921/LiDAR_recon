#pragma once

#include <array>
#include <filesystem>
#include <string>
#include <vector>

namespace centerpoint {

struct HeadBatchNormWeights {
    std::vector<float> weight;
    std::vector<float> bias;
    std::vector<float> mean;
    std::vector<float> variance;
};

struct HeadConvWeights {
    std::string name;
    std::vector<float> weight;
    std::vector<float> bias;
    HeadBatchNormWeights batch_norm;
    int in_channels = 0;
    int out_channels = 0;
    int kernel_size = 3;
    int padding = 1;
    bool has_batch_norm = false;
};

struct HeadBranchWeights {
    std::string name;
    HeadConvWeights hidden;
    HeadConvWeights output;
};

struct HeadWeights {
    HeadConvWeights shared;
    std::array<HeadBranchWeights, 5> branches;
    float batch_norm_epsilon = 1.0e-3F;
};

HeadWeights load_head_weights(const std::filesystem::path& directory);

}  // namespace centerpoint
