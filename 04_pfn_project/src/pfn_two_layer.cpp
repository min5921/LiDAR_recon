#include "centerpoint/pfn_two_layer.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace centerpoint {
namespace {

float batch_norm_relu(float value,
                      int channel,
                      const PfnLayerCheckpointWeights& weights,
                      float epsilon) {
    const float normalized =
        (value - weights.bn_mean[channel]) /
        std::sqrt(weights.bn_var[channel] + epsilon);
    const float affine =
        normalized * weights.bn_weight[channel] + weights.bn_bias[channel];
    return std::max(affine, 0.0F);
}

void validate_layer(const PfnLayerCheckpointWeights& layer) {
    if (layer.in_channels <= 0 || layer.out_channels <= 0) {
        throw std::invalid_argument("invalid PFN layer dimensions");
    }
    const std::size_t matrix_size =
        static_cast<std::size_t>(layer.out_channels) * layer.in_channels;
    if (layer.linear_weight.size() != matrix_size ||
        layer.bn_weight.size() != static_cast<std::size_t>(layer.out_channels) ||
        layer.bn_bias.size() != static_cast<std::size_t>(layer.out_channels) ||
        layer.bn_mean.size() != static_cast<std::size_t>(layer.out_channels) ||
        layer.bn_var.size() != static_cast<std::size_t>(layer.out_channels)) {
        throw std::invalid_argument("PFN layer weight size mismatch");
    }
}

}  // namespace

PillarFeatureResult run_two_layer_pfn_cpu(
    const DecoratedPillarDump& dump,
    const TwoLayerPfnCheckpointWeights& weights) {
    validate_layer(weights.layer0);
    validate_layer(weights.layer1);

    const int num_pillars = dump.metadata.num_pillars;
    const int max_points = dump.metadata.max_points_per_pillar;
    const int input_channels = dump.metadata.decorated_feature_dim;
    const int local_channels = weights.layer0.out_channels;
    const int concatenated_channels = local_channels * 2;
    const int output_channels = weights.layer1.out_channels;

    if (weights.layer0.in_channels != input_channels) {
        throw std::invalid_argument("layer 0 input does not match decorated feature dimension");
    }
    if (weights.layer1.in_channels != concatenated_channels) {
        throw std::invalid_argument("layer 1 input does not match PFN concatenated dimension");
    }

    PillarFeatureResult result;
    result.num_pillars = num_pillars;
    result.out_channels = output_channels;
    result.pillar_features.assign(
        static_cast<std::size_t>(num_pillars) * output_channels, 0.0F);

    std::vector<float> local_features(
        static_cast<std::size_t>(max_points) * local_channels);
    std::vector<float> local_max(static_cast<std::size_t>(local_channels));

    for (int pillar = 0; pillar < num_pillars; ++pillar) {
        std::fill(local_max.begin(), local_max.end(),
                  -std::numeric_limits<float>::infinity());

        for (int point = 0; point < max_points; ++point) {
            const std::size_t input_offset =
                (static_cast<std::size_t>(pillar) * max_points + point) *
                input_channels;
            const std::size_t local_offset =
                static_cast<std::size_t>(point) * local_channels;

            for (int out = 0; out < local_channels; ++out) {
                float linear = 0.0F;
                const std::size_t weight_offset =
                    static_cast<std::size_t>(out) * input_channels;
                for (int in = 0; in < input_channels; ++in) {
                    linear += dump.decorated_pillars[input_offset + in] *
                              weights.layer0.linear_weight[weight_offset + in];
                }
                const float activated = batch_norm_relu(
                    linear, out, weights.layer0, weights.batch_norm_eps);
                local_features[local_offset + out] = activated;
                local_max[out] = std::max(local_max[out], activated);
            }
        }

        for (int out = 0; out < output_channels; ++out) {
            float pooled = -std::numeric_limits<float>::infinity();
            const std::size_t weight_offset =
                static_cast<std::size_t>(out) * concatenated_channels;

            for (int point = 0; point < max_points; ++point) {
                const std::size_t local_offset =
                    static_cast<std::size_t>(point) * local_channels;
                float linear = 0.0F;
                for (int in = 0; in < local_channels; ++in) {
                    linear += local_features[local_offset + in] *
                              weights.layer1.linear_weight[weight_offset + in];
                }
                for (int in = 0; in < local_channels; ++in) {
                    linear += local_max[in] *
                              weights.layer1.linear_weight[
                                  weight_offset + local_channels + in];
                }
                const float activated = batch_norm_relu(
                    linear, out, weights.layer1, weights.batch_norm_eps);
                pooled = std::max(pooled, activated);
            }
            result.pillar_features[
                static_cast<std::size_t>(pillar) * output_channels + out] = pooled;
        }
    }

    return result;
}

}  // namespace centerpoint
