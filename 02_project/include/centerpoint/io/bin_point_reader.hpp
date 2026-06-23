#pragma once

#include <filesystem>

#include "centerpoint/types.hpp"

namespace centerpoint::io {

PointCloud read_float32_point_cloud(const std::filesystem::path& path, int feature_dim);

}  // namespace centerpoint::io

