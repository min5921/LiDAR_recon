#include "centerpoint/pfn.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace centerpoint {

PfnWeights make_dummy_pfn_weights(int in_channels, int out_channels) {
    if (in_channels <= 0 || out_channels <= 0) {
        throw std::runtime_error("invalid PFN weight shape");
    }

    PfnWeights weights;
    weights.in_channels = in_channels;
    weights.out_channels = out_channels;
    weights.linear_weight.resize(static_cast<std::size_t>(out_channels) * in_channels);
    weights.bn_weight.assign(static_cast<std::size_t>(out_channels), 1.0F);
    weights.bn_bias.assign(static_cast<std::size_t>(out_channels), 0.0F);
    weights.bn_mean.assign(static_cast<std::size_t>(out_channels), 0.0F);
    weights.bn_var.assign(static_cast<std::size_t>(out_channels), 1.0F);

    for (int out = 0; out < out_channels; ++out) {
        for (int in = 0; in < in_channels; ++in) {
            const int pattern = ((out + 1) * (in + 3)) % 17;
            weights.linear_weight[static_cast<std::size_t>(out) * in_channels + in] =
                (static_cast<float>(pattern) - 8.0F) * 0.01F;
        }
    }

    return weights;
}

PillarFeatureResult run_pfn_cpu(const DecoratedPillarDump& dump,
                                const PfnConfig& config,
                                const PfnWeights& weights) {
    const int num_pillars = dump.metadata.num_pillars;
    const int max_points = dump.metadata.max_points_per_pillar;
    const int in_channels = dump.metadata.decorated_feature_dim;
    const int out_channels = config.out_channels;

    if (weights.in_channels != in_channels || weights.out_channels != out_channels) {
        throw std::runtime_error("PFN weight shape does not match input shape");
    }

    PillarFeatureResult result;
    result.num_pillars = num_pillars;
    result.out_channels = out_channels;
    result.pillar_features.assign(static_cast<std::size_t>(num_pillars) * out_channels, 0.0F);

    for (int pillar = 0; pillar < num_pillars; ++pillar) {
        for (int out = 0; out < out_channels; ++out) {
            float pooled = -std::numeric_limits<float>::infinity();

            for (int point = 0; point < max_points; ++point) {
                const std::size_t input_offset =
                    (static_cast<std::size_t>(pillar) * max_points + point) * in_channels;

                bool is_padding = true;
                for (int in = 0; in < in_channels; ++in) {
                    if (dump.decorated_pillars[input_offset + in] != 0.0F) {
                        is_padding = false;
                        break;
                    }
                }
                if (is_padding) {
                    continue;
                }

                float linear = 0.0F;
                for (int in = 0; in < in_channels; ++in) {
                    linear +=
                        dump.decorated_pillars[input_offset + in] *
                        weights.linear_weight[static_cast<std::size_t>(out) * in_channels + in];
                }

                const float normalized =
                    (linear - weights.bn_mean[out]) /
                    std::sqrt(weights.bn_var[out] + config.batch_norm_eps);
                const float affine = normalized * weights.bn_weight[out] + weights.bn_bias[out];
                const float activated = std::max(affine, 0.0F);
                pooled = std::max(pooled, activated);
            }

            if (!std::isfinite(pooled)) {
                pooled = 0.0F;
            }

            result.pillar_features[static_cast<std::size_t>(pillar) * out_channels + out] = pooled;
        }
    }

    return result;
}

}  // namespace centerpoint

