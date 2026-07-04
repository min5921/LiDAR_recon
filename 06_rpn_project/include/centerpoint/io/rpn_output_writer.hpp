#pragma once

#include <filesystem>

#include "centerpoint/rpn_full_types.hpp"

namespace centerpoint::io {

void write_full_rpn_output(const std::filesystem::path& output_dir,
                           const FullRpnResult& result,
                           bool write_tensor);

}  // namespace centerpoint::io
