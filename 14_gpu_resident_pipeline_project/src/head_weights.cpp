#include "centerpoint/head_weights.hpp"

#include <fstream>
#include <stdexcept>

namespace centerpoint {
namespace {

std::vector<float> read_floats(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("failed to open CenterHead weight: " +
                                 path.string());
    }
    const std::streamoff bytes = input.tellg();
    if (bytes < 0 || bytes % static_cast<std::streamoff>(sizeof(float)) != 0) {
        throw std::runtime_error("invalid CenterHead weight size: " +
                                 path.string());
    }
    std::vector<float> values(
        static_cast<std::size_t>(bytes) / sizeof(float));
    input.seekg(0, std::ios::beg);
    input.read(reinterpret_cast<char*>(values.data()), bytes);
    if (!input) {
        throw std::runtime_error("failed to read CenterHead weight: " +
                                 path.string());
    }
    return values;
}

void require_size(const std::vector<float>& values,
                  std::size_t expected,
                  const std::string& name) {
    if (values.size() != expected) {
        throw std::runtime_error(name + " element count mismatch");
    }
}

HeadBatchNormWeights read_batch_norm(const std::filesystem::path& directory,
                                     const std::string& prefix,
                                     int channels) {
    HeadBatchNormWeights result;
    result.weight = read_floats(directory / (prefix + "_bn_weight.bin"));
    result.bias = read_floats(directory / (prefix + "_bn_bias.bin"));
    result.mean = read_floats(directory / (prefix + "_bn_mean.bin"));
    result.variance = read_floats(directory / (prefix + "_bn_var.bin"));
    const std::size_t expected = static_cast<std::size_t>(channels);
    require_size(result.weight, expected, prefix + " BN weight");
    require_size(result.bias, expected, prefix + " BN bias");
    require_size(result.mean, expected, prefix + " BN mean");
    require_size(result.variance, expected, prefix + " BN variance");
    return result;
}

HeadConvWeights read_conv(const std::filesystem::path& directory,
                          const std::string& prefix,
                          int in_channels,
                          int out_channels,
                          bool has_batch_norm) {
    HeadConvWeights result;
    result.name = prefix;
    result.in_channels = in_channels;
    result.out_channels = out_channels;
    result.has_batch_norm = has_batch_norm;
    result.weight = read_floats(directory / (prefix + "_weight.bin"));
    result.bias = read_floats(directory / (prefix + "_bias.bin"));
    require_size(result.weight,
                 static_cast<std::size_t>(out_channels) * in_channels * 9,
                 prefix + " Conv weight");
    require_size(result.bias, static_cast<std::size_t>(out_channels),
                 prefix + " Conv bias");
    if (has_batch_norm) {
        result.batch_norm =
            read_batch_norm(directory, prefix, out_channels);
    }
    return result;
}

}  // namespace

HeadWeights load_head_weights(const std::filesystem::path& directory) {
    HeadWeights result;
    result.shared = read_conv(directory, "shared", 384, 64, true);
    const std::array<std::string, 5> names = {
        "reg", "height", "dim", "rot", "hm"};
    const std::array<int, 5> channels = {2, 1, 3, 2, 3};
    for (int index = 0; index < 5; ++index) {
        HeadBranchWeights& branch = result.branches[index];
        branch.name = names[index];
        branch.hidden = read_conv(
            directory, names[index] + "_hidden", 64, 64, true);
        branch.output = read_conv(
            directory, names[index] + "_output", 64, channels[index], false);
    }
    return result;
}

}  // namespace centerpoint
