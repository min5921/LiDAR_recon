#pragma once

#include <filesystem>

#include "centerpoint/types.hpp"

namespace centerpoint::io {

void write_bev_features(const std::filesystem::path& output_dir,
                        const BevFeatureResult& result);

}  // namespace centerpoint::io
