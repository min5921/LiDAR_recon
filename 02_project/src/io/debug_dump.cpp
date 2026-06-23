#include "centerpoint/io/debug_dump.hpp"

#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace centerpoint::io {
namespace {

template <typename T>
void write_binary(const std::filesystem::path& path, const std::vector<T>& values) {
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("failed to open output file: " + path.string());
    }

    output.write(reinterpret_cast<const char*>(values.data()),
                 static_cast<std::streamsize>(values.size() * sizeof(T)));
    if (!output) {
        throw std::runtime_error("failed to write output file: " + path.string());
    }
}

}  // namespace

void write_debug_dump(const std::filesystem::path& output_dir,
                      const VoxelizationConfig& config,
                      const VoxelizationResult& result) {
    std::filesystem::create_directories(output_dir);

    write_binary(output_dir / "pillars.bin", result.pillars);
    write_binary(output_dir / "coordinates.bin", result.coordinates);
    write_binary(output_dir / "num_points.bin", result.num_points_per_pillar);

    std::ofstream meta(output_dir / "metadata.json");
    if (!meta) {
        throw std::runtime_error("failed to open metadata.json");
    }

    meta << "{\n";
    meta << "  \"num_pillars\": " << result.num_pillars << ",\n";
    meta << "  \"max_points_per_pillar\": " << result.max_points_per_pillar << ",\n";
    meta << "  \"feature_dim\": " << result.feature_dim << ",\n";
    meta << "  \"coordinate_order\": \"batch,z,y,x\",\n";
    meta << "  \"grid_size_xyz\": ["
         << result.grid_size_xyz[0] << ", "
         << result.grid_size_xyz[1] << ", "
         << result.grid_size_xyz[2] << "],\n";
    meta << "  \"voxel_size\": ["
         << config.voxel_size[0] << ", "
         << config.voxel_size[1] << ", "
         << config.voxel_size[2] << "],\n";
    meta << "  \"point_cloud_range\": ["
         << config.point_cloud_range[0] << ", "
         << config.point_cloud_range[1] << ", "
         << config.point_cloud_range[2] << ", "
         << config.point_cloud_range[3] << ", "
         << config.point_cloud_range[4] << ", "
         << config.point_cloud_range[5] << "]\n";
    meta << "}\n";
}

}  // namespace centerpoint::io

