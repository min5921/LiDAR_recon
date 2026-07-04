#include "centerpoint/io/pfn_checkpoint_reader.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

namespace centerpoint::io {
namespace {

std::string read_text(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open weight metadata: " + path.string());
    }
    return std::string(std::istreambuf_iterator<char>(input),
                       std::istreambuf_iterator<char>());
}

int read_int(const std::string& text, const std::string& name) {
    const std::regex pattern("\"" + name + "\"\\s*:\\s*(\\d+)");
    std::smatch match;
    if (!std::regex_search(text, match, pattern)) {
        throw std::runtime_error("missing weight metadata field: " + name);
    }
    return std::stoi(match[1].str());
}

float read_float(const std::string& text, const std::string& name) {
    const std::regex pattern(
        "\"" + name + "\"\\s*:\\s*([-+0-9.eE]+)");
    std::smatch match;
    if (!std::regex_search(text, match, pattern)) {
        throw std::runtime_error("missing weight metadata field: " + name);
    }
    return std::stof(match[1].str());
}

std::vector<float> read_floats(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("failed to open weight file: " + path.string());
    }
    const auto bytes = input.tellg();
    if (bytes < 0 || static_cast<std::uintmax_t>(bytes) % sizeof(float) != 0) {
        throw std::runtime_error("invalid float weight file: " + path.string());
    }
    std::vector<float> values(static_cast<std::size_t>(bytes) / sizeof(float));
    input.seekg(0, std::ios::beg);
    input.read(reinterpret_cast<char*>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(float)));
    if (!input) {
        throw std::runtime_error("failed to read weight file: " + path.string());
    }
    return values;
}

PfnLayerCheckpointWeights read_layer(const std::filesystem::path& dir,
                                     const std::string& prefix,
                                     int in_channels,
                                     int out_channels) {
    PfnLayerCheckpointWeights layer;
    layer.in_channels = in_channels;
    layer.out_channels = out_channels;
    layer.linear_weight = read_floats(dir / (prefix + "_linear_weight.bin"));
    layer.bn_weight = read_floats(dir / (prefix + "_bn_weight.bin"));
    layer.bn_bias = read_floats(dir / (prefix + "_bn_bias.bin"));
    layer.bn_mean = read_floats(dir / (prefix + "_bn_mean.bin"));
    layer.bn_var = read_floats(dir / (prefix + "_bn_var.bin"));
    return layer;
}

}  // namespace

TwoLayerPfnCheckpointWeights read_two_layer_pfn_weights(
    const std::filesystem::path& weight_dir) {
    const std::string metadata = read_text(weight_dir / "weights_metadata.json");
    TwoLayerPfnCheckpointWeights weights;
    weights.batch_norm_eps = read_float(metadata, "batch_norm_eps");
    weights.layer0 = read_layer(
        weight_dir, "layer0", read_int(metadata, "layer0_in_channels"),
        read_int(metadata, "layer0_out_channels"));
    weights.layer1 = read_layer(
        weight_dir, "layer1", read_int(metadata, "layer1_in_channels"),
        read_int(metadata, "layer1_out_channels"));
    return weights;
}

}  // namespace centerpoint::io
