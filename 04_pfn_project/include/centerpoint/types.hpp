#pragma once

#include <vector>

namespace centerpoint {

struct DecoratedPillarMetadata {
    int num_pillars = 0;
    int max_points_per_pillar = 0;
    int input_feature_dim = 0;
    int decorated_feature_dim = 0;
};

struct DecoratedPillarDump {
    DecoratedPillarMetadata metadata;
    std::vector<float> decorated_pillars;
};

struct PfnConfig {
    int out_channels = 64;
    float batch_norm_eps = 1.0e-3F;
};

struct PfnWeights {
    std::vector<float> linear_weight;  // [out_channels, in_channels]
    std::vector<float> bn_weight;      // [out_channels]
    std::vector<float> bn_bias;        // [out_channels]
    std::vector<float> bn_mean;        // [out_channels]
    std::vector<float> bn_var;         // [out_channels]
    int in_channels = 0;
    int out_channels = 0;
};

struct PillarFeatureResult {
    std::vector<float> pillar_features;
    int num_pillars = 0;
    int out_channels = 0;
};

}  // namespace centerpoint

