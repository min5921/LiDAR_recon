#pragma once

#include <filesystem>

#include "centerpoint/types.hpp"

namespace centerpoint::io {

void write_debug_dump(const std::filesystem::path& output_dir,
                      const VoxelizationConfig& config,
                      const VoxelizationResult& result);

}  // namespace centerpoint::io

