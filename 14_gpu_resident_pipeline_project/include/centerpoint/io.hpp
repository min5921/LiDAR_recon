#pragma once

#include <filesystem>
#include <vector>

namespace centerpoint {

std::vector<float> read_point_bin(const std::filesystem::path& path,
                                  int feature_dimension);

void write_float_bin(const std::filesystem::path& path,
                     const std::vector<float>& values);

}  // namespace centerpoint
