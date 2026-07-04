#pragma once

#include <cstdint>
#include <vector>

namespace centerpoint {

struct ScatterInput {
    std::vector<float> pillar_features;  // [num_pillars, channels]
    std::vector<std::int32_t> coordinates;  // [num_pillars, 4], batch,z,y,x
    int num_pillars = 0;
    int channels = 0;
    int grid_x = 0;
    int grid_y = 0;
    int grid_z = 0;
};

struct BevFeatureResult {
    std::vector<float> features;  // NCHW
    int batch_size = 0;
    int channels = 0;
    int height = 0;
    int width = 0;
    int occupied_cells = 0;
};

}  // namespace centerpoint
