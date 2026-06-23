#pragma once

#include <array>
#include <cstdint>
#include <vector>

namespace centerpoint {

struct VoxelDumpMetadata {
    int num_pillars = 0;
    int max_points_per_pillar = 0;
    int feature_dim = 0;
    std::array<int, 3> grid_size_xyz{0, 0, 0};
    std::array<float, 3> voxel_size{0.0F, 0.0F, 0.0F};
    std::array<float, 6> point_cloud_range{0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
};

struct VoxelDump {
    VoxelDumpMetadata metadata;
    std::vector<float> pillars;
    std::vector<int32_t> coordinates;
    std::vector<int32_t> num_points_per_pillar;
};

struct DecoratedPillarResult {
    std::vector<float> decorated_pillars;
    int num_pillars = 0;
    int max_points_per_pillar = 0;
    int input_feature_dim = 0;
    int decorated_feature_dim = 0;
};

}  // namespace centerpoint

