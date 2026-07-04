#pragma once

#include <filesystem>

#include "centerpoint/pfn_two_layer.hpp"

namespace centerpoint::io {

TwoLayerPfnCheckpointWeights read_two_layer_pfn_weights(
    const std::filesystem::path& weight_dir);

}  // namespace centerpoint::io
