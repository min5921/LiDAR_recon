#include "centerpoint/io/bev_feature_writer.hpp"

#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace centerpoint::io {

void write_bev_features(const std::filesystem::path& output_dir,
                        const BevFeatureResult& result) {
    std::filesystem::create_directories(output_dir);

    std::ofstream output(output_dir / "bev_features.bin", std::ios::binary);
    if (!output) {
        throw std::runtime_error("failed to open bev_features.bin");
    }
    output.write(reinterpret_cast<const char*>(result.features.data()),
                 static_cast<std::streamsize>(result.features.size() * sizeof(float)));
    if (!output) {
        throw std::runtime_error("failed to write bev_features.bin");
    }

    std::ofstream metadata(output_dir / "bev_features_metadata.json");
    if (!metadata) {
        throw std::runtime_error("failed to open bev_features_metadata.json");
    }
    metadata << "{\n";
    metadata << "  \"layout\": \"NCHW\",\n";
    metadata << "  \"batch_size\": " << result.batch_size << ",\n";
    metadata << "  \"channels\": " << result.channels << ",\n";
    metadata << "  \"height\": " << result.height << ",\n";
    metadata << "  \"width\": " << result.width << ",\n";
    metadata << "  \"occupied_cells\": " << result.occupied_cells << ",\n";
    metadata << "  \"shape\": [" << result.batch_size << ", "
             << result.channels << ", " << result.height << ", "
             << result.width << "]\n";
    metadata << "}\n";
}

}  // namespace centerpoint::io
