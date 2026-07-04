#pragma once

#include <filesystem>

#include "centerpoint/rpn_full_types.hpp"

namespace centerpoint::io {

HostTensor read_bev_dump(const std::filesystem::path& dump_dir);

}  // namespace centerpoint::io
