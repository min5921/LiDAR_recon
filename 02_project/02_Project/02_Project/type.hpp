#pragma once

#include <vector>
#include <cstdint>
#include <array>


namespace centerpoint {

	struct PointCloud {
		std::vector<float> values;
		int feature_dim = 0;

		int num_points() const {
			return feature_dim > 0 ? static_cast<int>(values.size() / feature_dim) : 0;
		}
	};

	struct VoxelizationConfig {
		std::array<float, 3> voxel_size{ 0.32F, 0.32F, 6.0F };
		std::array<float, 6> point_cloud_range{ -74.88F, -74.88F, -2.0F, 74.88F, 74.88F, 4.0F };
		int max_points_per_voxel = 20;
		int max_voxels = 60000;
		int feature_dim = 5;
	};

	struct VoxelizationResult {
		std::vector<float> pillars;
		std::vector<int32_t> coordinates;
		std::vector<int32_t> num_points_per_pillar;
		std::array<int, 3> grid_size_xyz{ 0, 0, 0 };
		int num_pillars = 0;
		int max_points_per_pillar = 0;
		int feature_dim = 0;
	};
}