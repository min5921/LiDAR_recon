#include "centerpoint/rpn_dump_writer.hpp"

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace centerpoint::io {
namespace {

void write_binary(const std::filesystem::path& path,
                  const std::vector<float>& values) {
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("failed to open output file: " + path.string());
    }
    output.write(reinterpret_cast<const char*>(values.data()),
                 static_cast<std::streamsize>(values.size() * sizeof(float)));
    if (!output) {
        throw std::runtime_error("failed to write output file: " + path.string());
    }
}

}  // namespace

void write_rpn_demo_dump(
    const std::filesystem::path& output_dir,
    const std::vector<float>& input,
    const std::vector<float>& weights,
    const std::vector<float>& bn_weight,
    const std::vector<float>& bn_bias,
    const std::vector<float>& bn_mean,
    const std::vector<float>& bn_var,
    const ConvBnReluConfig& config,
    const ConvBnReluResult& result) {
    std::filesystem::create_directories(output_dir);
    write_binary(output_dir / "input.bin", input);
    write_binary(output_dir / "conv_weight.bin", weights);
    write_binary(output_dir / "bn_weight.bin", bn_weight);
    write_binary(output_dir / "bn_bias.bin", bn_bias);
    write_binary(output_dir / "bn_mean.bin", bn_mean);
    write_binary(output_dir / "bn_var.bin", bn_var);
    write_binary(output_dir / "output.bin", result.output);

    std::ofstream metadata(output_dir / "metadata.json");
    if (!metadata) {
        throw std::runtime_error("failed to open metadata.json");
    }
    metadata << "{\n";
    metadata << "  \"layout\": \"NCHW\",\n";
    metadata << "  \"input_shape\": [" << config.batch << ", "
             << config.in_channels << ", " << config.input_height << ", "
             << config.input_width << "],\n";
    metadata << "  \"weight_shape\": [" << config.out_channels << ", "
             << config.in_channels << ", " << config.kernel_size << ", "
             << config.kernel_size << "],\n";
    metadata << "  \"output_shape\": [" << result.batch << ", "
             << result.channels << ", " << result.height << ", "
             << result.width << "],\n";
    metadata << "  \"stride\": " << config.stride << ",\n";
    metadata << "  \"padding\": " << config.padding << ",\n";
    metadata << "  \"batch_norm_eps\": " << config.batch_norm_eps << ",\n";
    metadata << "  \"kernel_elapsed_ms\": " << result.elapsed_ms << "\n";
    metadata << "}\n";
}

}  // namespace centerpoint::io
