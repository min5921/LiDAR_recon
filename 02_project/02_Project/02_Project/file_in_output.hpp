#pragma once

#include <filesystem>

#include "type.hpp"

namespace centerpoint::io {

    void write_debug_dump(const std::filesystem::path& output_dir,
        const VoxelizationConfig& config,
        const VoxelizationResult& result);

    PointCloud read_float32_point_cloud(const std::filesystem::path& path, int feature_dim);


}  // namespace centerpoint::io

