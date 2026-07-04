#include "centerpoint/io/rpn_output_writer.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace centerpoint::io {

void write_full_rpn_output(const std::filesystem::path& output_dir,
                           const FullRpnResult& result,
                           bool write_tensor) {
    std::filesystem::create_directories(output_dir);
    if (write_tensor) {
        std::ofstream output(output_dir / "rpn_features.bin", std::ios::binary);
        if (!output) {
            throw std::runtime_error("failed to open rpn_features.bin");
        }
        output.write(reinterpret_cast<const char*>(result.output.values.data()),
                     static_cast<std::streamsize>(
                         result.output.values.size() * sizeof(float)));
    }

    float minimum = std::numeric_limits<float>::infinity();
    float maximum = -std::numeric_limits<float>::infinity();
    double sum = 0.0;
    std::size_t non_finite = 0;
    for (float value : result.output.values) {
        if (!std::isfinite(value)) {
            ++non_finite;
            continue;
        }
        minimum = std::min(minimum, value);
        maximum = std::max(maximum, value);
        sum += value;
    }

    std::ofstream metadata(output_dir / "rpn_features_metadata.json");
    if (!metadata) {
        throw std::runtime_error("failed to open RPN metadata");
    }
    metadata << "{\n";
    metadata << "  \"layout\": \"NCHW\",\n";
    metadata << "  \"shape\": [1, " << result.output.channels << ", "
             << result.output.height << ", " << result.output.width << "],\n";
    metadata << "  \"elapsed_ms\": " << result.elapsed_ms << ",\n";
    metadata << "  \"minimum\": " << minimum << ",\n";
    metadata << "  \"maximum\": " << maximum << ",\n";
    metadata << "  \"sum\": " << sum << ",\n";
    metadata << "  \"non_finite_count\": " << non_finite << ",\n";
    metadata << "  \"tensor_written\": " << (write_tensor ? "true" : "false") << ",\n";
    metadata << "  \"block_shapes\": [\n";
    for (int index = 0; index < 3; ++index) {
        const auto& shape = result.block_shapes[index];
        metadata << "    [" << shape[0] << ", " << shape[1] << ", " << shape[2] << "]"
                 << (index == 2 ? "\n" : ",\n");
    }
    metadata << "  ],\n";
    metadata << "  \"deblock_shapes\": [\n";
    for (int index = 0; index < 3; ++index) {
        const auto& shape = result.deblock_shapes[index];
        metadata << "    [" << shape[0] << ", " << shape[1] << ", " << shape[2] << "]"
                 << (index == 2 ? "\n" : ",\n");
    }
    metadata << "  ],\n";
    metadata << "  \"first_values\": [";
    const std::size_t sample_count =
        std::min<std::size_t>(16, result.output.values.size());
    for (std::size_t index = 0; index < sample_count; ++index) {
        metadata << result.output.values[index]
                 << (index + 1 == sample_count ? "" : ", ");
    }
    metadata << "]\n";
    metadata << "}\n";
}

}  // namespace centerpoint::io
