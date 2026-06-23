#pragma once

#include <filesystem>

#include "centerpoint/types.hpp"

namespace centerpoint::io {

VoxelDump read_voxel_dump(const std::filesystem::path& dump_dir);

}  // namespace centerpoint::io

