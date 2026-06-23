#pragma once

#include <filesystem>

#include "centerpoint/types.hpp"

namespace centerpoint::io {

DecoratedPillarDump read_decorated_pillar_dump(const std::filesystem::path& dump_dir);

}  // namespace centerpoint::io

