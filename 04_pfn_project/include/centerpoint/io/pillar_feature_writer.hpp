#pragma once

#include <filesystem>

#include "centerpoint/types.hpp"

namespace centerpoint::io {

void write_pillar_features(const std::filesystem::path& output_dir,
                           const PillarFeatureResult& result);

}  // namespace centerpoint::io

