#include "centerpoint/io/pillar_feature_writer.hpp"

#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace centerpoint::io {

void write_pillar_features(const std::filesystem::path& output_dir,
                           const PillarFeatureResult& result) {
    std::filesystem::create_directories(output_dir);

    std::ofstream output(output_dir / "pillar_features.bin", std::ios::binary);
    if (!output) {
        throw std::runtime_error("failed to open pillar_features.bin");
    }
    output.write(reinterpret_cast<const char*>(result.pillar_features.data()),
                 static_cast<std::streamsize>(result.pillar_features.size() * sizeof(float)));
    if (!output) {
        throw std::runtime_error("failed to write pillar_features.bin");
    }

    std::ofstream meta(output_dir / "pillar_features_metadata.json");
    if (!meta) {
        throw std::runtime_error("failed to open pillar_features_metadata.json");
    }

    meta << "{\n";
    meta << "  \"num_pillars\": " << result.num_pillars << ",\n";
    meta << "  \"out_channels\": " << result.out_channels << ",\n";
    meta << "  \"shape\": ["
         << result.num_pillars << ", "
         << result.out_channels << "]\n";
    meta << "}\n";
}

}  // namespace centerpoint::io

