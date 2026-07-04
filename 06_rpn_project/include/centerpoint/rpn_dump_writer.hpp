#pragma once

#include <filesystem>
#include <vector>

#include "centerpoint/rpn_cuda.hpp"

namespace centerpoint::io {

void write_rpn_demo_dump(
    const std::filesystem::path& output_dir,
    const std::vector<float>& input,
    const std::vector<float>& weights,
    const std::vector<float>& bn_weight,
    const std::vector<float>& bn_bias,
    const std::vector<float>& bn_mean,
    const std::vector<float>& bn_var,
    const ConvBnReluConfig& config,
    const ConvBnReluResult& result);

}  // namespace centerpoint::io
