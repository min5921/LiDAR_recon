#pragma once

#include <filesystem>

#include "centerpoint/types.hpp"

namespace centerpoint::io {

ScatterInput read_scatter_input(const std::filesystem::path& pfn_dump_dir,
                                const std::filesystem::path& voxel_dump_dir);

}  // namespace centerpoint::io
