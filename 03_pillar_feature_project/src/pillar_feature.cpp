#include "centerpoint/pillar_feature.hpp"

#include <algorithm>
#include <stdexcept>

namespace centerpoint {

DecoratedPillarResult decorate_pillars_cpu(const VoxelDump& dump) {
    const VoxelDumpMetadata& metadata = dump.metadata;
    if (metadata.feature_dim < 3) {
        throw std::runtime_error("feature_dim must be at least 3");
    }

    DecoratedPillarResult result;
    result.num_pillars = metadata.num_pillars;
    result.max_points_per_pillar = metadata.max_points_per_pillar;
    result.input_feature_dim = metadata.feature_dim;
    result.decorated_feature_dim = metadata.feature_dim + 5;

    result.decorated_pillars.assign(
        static_cast<std::size_t>(result.num_pillars) *
            result.max_points_per_pillar *
            result.decorated_feature_dim,
        0.0F);

    const float x_offset = metadata.voxel_size[0] * 0.5F + metadata.point_cloud_range[0];
    const float y_offset = metadata.voxel_size[1] * 0.5F + metadata.point_cloud_range[1];

    for (int pillar_idx = 0; pillar_idx < metadata.num_pillars; ++pillar_idx) {
        const int point_count = std::clamp(
            dump.num_points_per_pillar[static_cast<std::size_t>(pillar_idx)],
            0,
            metadata.max_points_per_pillar);
        if (point_count == 0) {
            continue;
        }

        float mean_x = 0.0F;
        float mean_y = 0.0F;
        float mean_z = 0.0F;
        for (int point_idx = 0; point_idx < point_count; ++point_idx) {
            const std::size_t input_offset =
                (static_cast<std::size_t>(pillar_idx) * metadata.max_points_per_pillar + point_idx) *
                metadata.feature_dim;
            mean_x += dump.pillars[input_offset + 0];
            mean_y += dump.pillars[input_offset + 1];
            mean_z += dump.pillars[input_offset + 2];
        }

        mean_x /= static_cast<float>(point_count);
        mean_y /= static_cast<float>(point_count);
        mean_z /= static_cast<float>(point_count);

        const std::size_t coord_offset = static_cast<std::size_t>(pillar_idx) * 4;
        const int y_coord = dump.coordinates[coord_offset + 2];
        const int x_coord = dump.coordinates[coord_offset + 3];
        const float pillar_center_x = static_cast<float>(x_coord) * metadata.voxel_size[0] + x_offset;
        const float pillar_center_y = static_cast<float>(y_coord) * metadata.voxel_size[1] + y_offset;

        for (int point_idx = 0; point_idx < point_count; ++point_idx) {
            const std::size_t input_offset =
                (static_cast<std::size_t>(pillar_idx) * metadata.max_points_per_pillar + point_idx) *
                metadata.feature_dim;
            const std::size_t output_offset =
                (static_cast<std::size_t>(pillar_idx) * metadata.max_points_per_pillar + point_idx) *
                result.decorated_feature_dim;

            const float x = dump.pillars[input_offset + 0];
            const float y = dump.pillars[input_offset + 1];
            const float z = dump.pillars[input_offset + 2];

            for (int feature = 0; feature < metadata.feature_dim; ++feature) {
                result.decorated_pillars[output_offset + feature] =
                    dump.pillars[input_offset + feature];
            }

            result.decorated_pillars[output_offset + metadata.feature_dim + 0] = x - mean_x;
            result.decorated_pillars[output_offset + metadata.feature_dim + 1] = y - mean_y;
            result.decorated_pillars[output_offset + metadata.feature_dim + 2] = z - mean_z;
            result.decorated_pillars[output_offset + metadata.feature_dim + 3] = x - pillar_center_x;
            result.decorated_pillars[output_offset + metadata.feature_dim + 4] = y - pillar_center_y;
        }
    }

    return result;
}

}  // namespace centerpoint

