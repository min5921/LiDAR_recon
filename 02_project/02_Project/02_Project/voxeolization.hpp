#pragma once

#include "type.hpp"

namespace centerpoint {

	VoxelizationResult voxelize_cpu(const PointCloud& cloud, const VoxelizationConfig& config);

}  // namespace centerpoint

