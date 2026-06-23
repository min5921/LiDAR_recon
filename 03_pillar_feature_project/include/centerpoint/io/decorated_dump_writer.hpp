#pragma once

#include <filesystem>

#include "centerpoint/types.hpp"

namespace centerpoint::io {

void write_decorated_dump(const std::filesystem::path& output_dir,
                          const VoxelDumpMetadata& input_metadata,
                          const DecoratedPillarResult& result);

}  // namespace centerpoint::io

