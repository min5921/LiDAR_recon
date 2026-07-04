#pragma once

#include <vector>

#include "centerpoint/types.hpp"

namespace centerpoint {

struct PfnLayerCheckpointWeights {
    std::vector<float> linear_weight;  // [out_channels, in_channels]
    std::vector<float> bn_weight;
    std::vector<float> bn_bias;
    std::vector<float> bn_mean;
    std::vector<float> bn_var;
    int in_channels = 0;
    int out_channels = 0;
};

struct TwoLayerPfnCheckpointWeights {
    PfnLayerCheckpointWeights layer0;
    PfnLayerCheckpointWeights layer1;
    float batch_norm_eps = 1.0e-3F;
};

PillarFeatureResult run_two_layer_pfn_cpu(
    const DecoratedPillarDump& dump,
    const TwoLayerPfnCheckpointWeights& weights);

}  // namespace centerpoint
