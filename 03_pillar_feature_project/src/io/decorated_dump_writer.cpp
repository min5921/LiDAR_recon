#include "centerpoint/io/decorated_dump_writer.hpp"

#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace centerpoint::io {

void write_decorated_dump(const std::filesystem::path& output_dir,
                          const VoxelDumpMetadata& input_metadata,
                          const DecoratedPillarResult& result) {
    std::filesystem::create_directories(output_dir);

    std::ofstream output(output_dir / "decorated_pillars.bin", std::ios::binary);
    if (!output) {
        throw std::runtime_error("failed to open decorated_pillars.bin");
    }

    output.write(reinterpret_cast<const char*>(result.decorated_pillars.data()),
                 static_cast<std::streamsize>(result.decorated_pillars.size() * sizeof(float)));
    if (!output) {
        throw std::runtime_error("failed to write decorated_pillars.bin");
    }

    std::ofstream meta(output_dir / "decorated_metadata.json");
    if (!meta) {
        throw std::runtime_error("failed to open decorated_metadata.json");
    }

    meta << "{\n";
    meta << "  \"num_pillars\": " << result.num_pillars << ",\n";
    meta << "  \"max_points_per_pillar\": " << result.max_points_per_pillar << ",\n";
    meta << "  \"input_feature_dim\": " << result.input_feature_dim << ",\n";
    meta << "  \"decorated_feature_dim\": " << result.decorated_feature_dim << ",\n";
    meta << "  \"feature_order\": [\"raw_features\", \"cluster_offset_xyz\", \"center_offset_xy\"],\n";
    meta << "  \"coordinate_order\": \"batch,z,y,x\",\n";
    meta << "  \"voxel_size\": ["
         << input_metadata.voxel_size[0] << ", "
         << input_metadata.voxel_size[1] << ", "
         << input_metadata.voxel_size[2] << "],\n";
    meta << "  \"point_cloud_range\": ["
         << input_metadata.point_cloud_range[0] << ", "
         << input_metadata.point_cloud_range[1] << ", "
         << input_metadata.point_cloud_range[2] << ", "
         << input_metadata.point_cloud_range[3] << ", "
         << input_metadata.point_cloud_range[4] << ", "
         << input_metadata.point_cloud_range[5] << "]\n";
    meta << "}\n";
}

}  // namespace centerpoint::io

