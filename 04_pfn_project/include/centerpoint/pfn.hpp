#pragma once

#include "centerpoint/types.hpp"

namespace centerpoint {

PfnWeights make_dummy_pfn_weights(int in_channels, int out_channels);

PillarFeatureResult run_pfn_cpu(const DecoratedPillarDump& dump,
                                const PfnConfig& config,
                                const PfnWeights& weights);

}  // namespace centerpoint

