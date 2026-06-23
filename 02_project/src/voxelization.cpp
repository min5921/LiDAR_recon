#include "centerpoint/voxelization.hpp"

#include <cmath>
#include <stdexcept>

namespace centerpoint {
namespace {

int flatten_zyx(int z, int y, int x, const std::array<int, 3>& grid_size_xyz) {
    const int nx = grid_size_xyz[0];
    const int ny = grid_size_xyz[1];
    return (z * ny + y) * nx + x;
}

std::array<int, 3> compute_grid_size_xyz(const VoxelizationConfig& config) {
    std::array<int, 3> grid{};
    for (int axis = 0; axis < 3; ++axis) {
        const float extent = config.point_cloud_range[axis + 3] - config.point_cloud_range[axis];
        grid[axis] = static_cast<int>(std::round(extent / config.voxel_size[axis]));
    }
    return grid;
}

}  // namespace

VoxelizationResult voxelize_cpu(const PointCloud& cloud, const VoxelizationConfig& config) {
    if (cloud.feature_dim != config.feature_dim) {
        throw std::runtime_error("point cloud feature_dim does not match voxelization config");
    }
    if (config.feature_dim < 3) {
        throw std::runtime_error("feature_dim must be at least 3");
    }

    VoxelizationResult result;
    result.grid_size_xyz = compute_grid_size_xyz(config);
    result.max_points_per_pillar = config.max_points_per_voxel;
    result.feature_dim = config.feature_dim;

    const int nx = result.grid_size_xyz[0];
    const int ny = result.grid_size_xyz[1];
    const int nz = result.grid_size_xyz[2];
    if (nx <= 0 || ny <= 0 || nz <= 0) {
        throw std::runtime_error("invalid voxel grid size");
    }

    std::vector<int32_t> coord_to_voxel_idx(static_cast<std::size_t>(nx) * ny * nz, -1);
    result.pillars.assign(static_cast<std::size_t>(config.max_voxels) *
                              config.max_points_per_voxel * config.feature_dim,
                          0.0F);
    result.coordinates.assign(static_cast<std::size_t>(config.max_voxels) * 4, 0);
    result.num_points_per_pillar.assign(static_cast<std::size_t>(config.max_voxels), 0);

    int voxel_count = 0;
    const int num_points = cloud.num_points();

    for (int point_idx = 0; point_idx < num_points; ++point_idx) {
        const float* point = cloud.values.data() + static_cast<std::size_t>(point_idx) * config.feature_dim;

        int coord_xyz[3]{};
        bool outside = false;
        for (int axis = 0; axis < 3; ++axis) {
            const float coord_f =
                std::floor((point[axis] - config.point_cloud_range[axis]) / config.voxel_size[axis]);
            const int coord = static_cast<int>(coord_f);
            if (coord < 0 || coord >= result.grid_size_xyz[axis]) {
                outside = true;
                break;
            }
            coord_xyz[axis] = coord;
        }

        if (outside) {
            continue;
        }

        const int x = coord_xyz[0];
        const int y = coord_xyz[1];
        const int z = coord_xyz[2];
        const int map_index = flatten_zyx(z, y, x, result.grid_size_xyz);

        int voxel_idx = coord_to_voxel_idx[map_index];
        if (voxel_idx == -1) {
            if (voxel_count >= config.max_voxels) {
                continue;
            }

            voxel_idx = voxel_count;
            coord_to_voxel_idx[map_index] = voxel_idx;

            const std::size_t coord_offset = static_cast<std::size_t>(voxel_idx) * 4;
            result.coordinates[coord_offset + 0] = 0;
            result.coordinates[coord_offset + 1] = z;
            result.coordinates[coord_offset + 2] = y;
            result.coordinates[coord_offset + 3] = x;

            ++voxel_count;
        }

        const int point_count = result.num_points_per_pillar[voxel_idx];
        if (point_count >= config.max_points_per_voxel) {
            continue;
        }

        const std::size_t pillar_offset =
            (static_cast<std::size_t>(voxel_idx) * config.max_points_per_voxel + point_count) *
            config.feature_dim;
        for (int feature = 0; feature < config.feature_dim; ++feature) {
            result.pillars[pillar_offset + feature] = point[feature];
        }
        result.num_points_per_pillar[voxel_idx] += 1;
    }

    result.num_pillars = voxel_count;
    result.pillars.resize(static_cast<std::size_t>(voxel_count) *
                          config.max_points_per_voxel * config.feature_dim);
    result.coordinates.resize(static_cast<std::size_t>(voxel_count) * 4);
    result.num_points_per_pillar.resize(static_cast<std::size_t>(voxel_count));

    return result;
}

}  // namespace centerpoint

