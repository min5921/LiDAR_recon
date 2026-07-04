#pragma once

#include <filesystem>

#include "centerpoint/rpn_full_types.hpp"

namespace centerpoint::io {

FullRpnWeights read_full_rpn_weights(const std::filesystem::path& weight_dir);

}  // namespace centerpoint::io
