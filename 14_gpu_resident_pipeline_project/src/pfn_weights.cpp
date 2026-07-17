#include "centerpoint/pfn_weights.hpp"

#include <fstream>
#include <regex>
#include <stdexcept>
#include <string>

namespace centerpoint {
namespace {

std::string read_text(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open PFN metadata: " + path.string());
    }
    return std::string(std::istreambuf_iterator<char>(input),
                       std::istreambuf_iterator<char>());
}

int read_int(const std::string& text, const std::string& name) {
    const std::regex pattern("\"" + name + "\"\\s*:\\s*(\\d+)");
    std::smatch match;
    if (!std::regex_search(text, match, pattern)) {
        throw std::runtime_error("missing PFN metadata field: " + name);
    }
    return std::stoi(match[1].str());
}

float read_float(const std::string& text, const std::string& name) {
    const std::regex pattern(
        "\"" + name + "\"\\s*:\\s*([-+0-9.eE]+)");
    std::smatch match;
    if (!std::regex_search(text, match, pattern)) {
        throw std::runtime_error("missing PFN metadata field: " + name);
    }
    return std::stof(match[1].str());
}

std::vector<float> read_floats(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("failed to open PFN tensor: " + path.string());
    }
    const std::streamoff bytes = input.tellg();
    if (bytes < 0 || bytes % static_cast<std::streamoff>(sizeof(float)) != 0) {
        throw std::runtime_error("invalid PFN tensor size: " + path.string());
    }
    std::vector<float> values(
        static_cast<std::size_t>(bytes) / sizeof(float));
    input.seekg(0, std::ios::beg);
    input.read(reinterpret_cast<char*>(values.data()), bytes);
    if (!input) {
        throw std::runtime_error("failed to read PFN tensor: " + path.string());
    }
    return values;
}

PfnLayerWeights read_layer(const std::filesystem::path& directory,
                           const std::string& prefix,
                           int in_channels,
                           int out_channels) {
    PfnLayerWeights layer;
    layer.in_channels = in_channels;
    layer.out_channels = out_channels;
    layer.linear = read_floats(directory / (prefix + "_linear_weight.bin"));
    layer.bn_weight = read_floats(directory / (prefix + "_bn_weight.bin"));
    layer.bn_bias = read_floats(directory / (prefix + "_bn_bias.bin"));
    layer.bn_mean = read_floats(directory / (prefix + "_bn_mean.bin"));
    layer.bn_variance = read_floats(directory / (prefix + "_bn_var.bin"));

    const std::size_t matrix_size =
        static_cast<std::size_t>(in_channels) * out_channels;
    const std::size_t channel_count = static_cast<std::size_t>(out_channels);
    if (layer.linear.size() != matrix_size ||
        layer.bn_weight.size() != channel_count ||
        layer.bn_bias.size() != channel_count ||
        layer.bn_mean.size() != channel_count ||
        layer.bn_variance.size() != channel_count) {
        throw std::runtime_error("PFN tensor shape mismatch for " + prefix);
    }
    return layer;
}

}  // namespace

PfnWeights load_pfn_weights(const std::filesystem::path& directory) {
    const std::string metadata =
        read_text(directory / "weights_metadata.json");
    PfnWeights weights;
    weights.batch_norm_epsilon = read_float(metadata, "batch_norm_eps");
    weights.layer0 = read_layer(
        directory, "layer0", read_int(metadata, "layer0_in_channels"),
        read_int(metadata, "layer0_out_channels"));
    weights.layer1 = read_layer(
        directory, "layer1", read_int(metadata, "layer1_in_channels"),
        read_int(metadata, "layer1_out_channels"));
    return weights;
}

}  // namespace centerpoint
