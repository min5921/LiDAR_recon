#include "centerpoint/scatter.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace centerpoint {

BevFeatureResult scatter_pillars_cpu(const ScatterInput& input) {
    if (input.num_pillars < 0 || input.channels <= 0 ||
        input.grid_x <= 0 || input.grid_y <= 0 || input.grid_z <= 0) {
        throw std::invalid_argument("invalid scatter dimensions");
    }

    const std::size_t expected_features =
        static_cast<std::size_t>(input.num_pillars) * input.channels;
    const std::size_t expected_coordinates =
        static_cast<std::size_t>(input.num_pillars) * 4;
    if (input.pillar_features.size() != expected_features) {
        throw std::invalid_argument("pillar feature count does not match dimensions");
    }
    if (input.coordinates.size() != expected_coordinates) {
        throw std::invalid_argument("coordinate count does not match num_pillars");
    }

    int batch_size = 0;
    for (int pillar = 0; pillar < input.num_pillars; ++pillar) {
        const int batch = input.coordinates[static_cast<std::size_t>(pillar) * 4];
        if (batch < 0) {
            throw std::invalid_argument("negative batch coordinate");
        }
        batch_size = std::max(batch_size, batch + 1);
    }
    if (input.num_pillars == 0) {
        batch_size = 1;
    }

    BevFeatureResult result;
    result.batch_size = batch_size;
    result.channels = input.channels;
    result.height = input.grid_y;
    result.width = input.grid_x;
    result.features.assign(
        static_cast<std::size_t>(batch_size) * input.channels * input.grid_y * input.grid_x,
        0.0F);

    const std::size_t spatial_size =
        static_cast<std::size_t>(input.grid_y) * input.grid_x;
    std::vector<std::uint8_t> occupied(
        static_cast<std::size_t>(batch_size) * spatial_size, 0);

    for (int pillar = 0; pillar < input.num_pillars; ++pillar) {
        const std::size_t coord_offset = static_cast<std::size_t>(pillar) * 4;
        const int batch = input.coordinates[coord_offset];
        const int z = input.coordinates[coord_offset + 1];
        const int y = input.coordinates[coord_offset + 2];
        const int x = input.coordinates[coord_offset + 3];

        if (z < 0 || z >= input.grid_z || y < 0 || y >= input.grid_y ||
            x < 0 || x >= input.grid_x) {
            throw std::out_of_range("pillar coordinate is outside the configured grid");
        }

        const std::size_t cell =
            static_cast<std::size_t>(batch) * spatial_size +
            static_cast<std::size_t>(y) * input.grid_x + x;
        if (occupied[cell] != 0) {
            throw std::invalid_argument("duplicate pillar coordinate");
        }
        occupied[cell] = 1;

        for (int channel = 0; channel < input.channels; ++channel) {
            const std::size_t source =
                static_cast<std::size_t>(pillar) * input.channels + channel;
            const std::size_t destination =
                ((static_cast<std::size_t>(batch) * input.channels + channel) *
                     input.grid_y + y) * input.grid_x + x;
            result.features[destination] = input.pillar_features[source];
        }
    }

    result.occupied_cells = input.num_pillars;
    return result;
}

}  // namespace centerpoint
