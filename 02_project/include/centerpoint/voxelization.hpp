#pragma once

#include "centerpoint/types.hpp"

namespace centerpoint {

VoxelizationResult voxelize_cpu(const PointCloud& cloud, const VoxelizationConfig& config);

}  // namespace centerpoint

