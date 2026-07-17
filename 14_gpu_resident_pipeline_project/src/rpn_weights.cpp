#include "centerpoint/rpn_weights.hpp"

#include <fstream>
#include <stdexcept>

namespace centerpoint {
namespace {

std::vector<float> read_floats(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("failed to open RPN weight: " + path.string());
    }
    const std::streamoff bytes = input.tellg();
    if (bytes < 0 || bytes % static_cast<std::streamoff>(sizeof(float)) != 0) {
        throw std::runtime_error("invalid RPN weight size: " + path.string());
    }
    std::vector<float> values(
        static_cast<std::size_t>(bytes) / sizeof(float));
    input.seekg(0, std::ios::beg);
    input.read(reinterpret_cast<char*>(values.data()), bytes);
    if (!input) {
        throw std::runtime_error("failed to read RPN weight: " + path.string());
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

RpnBatchNormWeights read_batch_norm(const std::filesystem::path& directory,
                                    const std::string& prefix,
                                    int channels) {
    RpnBatchNormWeights batch_norm;
    batch_norm.weight = read_floats(directory / (prefix + "_bn_weight.bin"));
    batch_norm.bias = read_floats(directory / (prefix + "_bn_bias.bin"));
    batch_norm.mean = read_floats(directory / (prefix + "_bn_mean.bin"));
    batch_norm.variance = read_floats(directory / (prefix + "_bn_var.bin"));
    const std::size_t expected = static_cast<std::size_t>(channels);
    require_size(batch_norm.weight, expected, prefix + " BN weight");
    require_size(batch_norm.bias, expected, prefix + " BN bias");
    require_size(batch_norm.mean, expected, prefix + " BN mean");
    require_size(batch_norm.variance, expected, prefix + " BN variance");
    return batch_norm;
}

RpnConvWeights read_conv(const std::filesystem::path& directory,
                         const std::string& prefix,
                         int in_channels,
                         int out_channels,
                         int kernel_size,
                         int stride,
                         int padding) {
    RpnConvWeights layer;
    layer.name = prefix;
    layer.in_channels = in_channels;
    layer.out_channels = out_channels;
    layer.kernel_size = kernel_size;
    layer.stride = stride;
    layer.padding = padding;
    layer.weight = read_floats(directory / (prefix + "_weight.bin"));
    require_size(layer.weight,
                 static_cast<std::size_t>(out_channels) * in_channels *
                     kernel_size * kernel_size,
                 prefix + " Conv weight");
    layer.batch_norm = read_batch_norm(directory, prefix, out_channels);
    return layer;
}

RpnDeconvWeights read_deconv(const std::filesystem::path& directory,
                             const std::string& prefix,
                             int in_channels,
                             int out_channels,
                             int kernel_size) {
    RpnDeconvWeights layer;
    layer.name = prefix;
    layer.in_channels = in_channels;
    layer.out_channels = out_channels;
    layer.kernel_size = kernel_size;
    layer.stride = kernel_size;
    layer.gemm_weight =
        read_floats(directory / (prefix + "_weight_gemm.bin"));
    require_size(layer.gemm_weight,
                 static_cast<std::size_t>(out_channels) * kernel_size *
                     kernel_size * in_channels,
                 prefix + " transposed Conv weight");
    layer.batch_norm = read_batch_norm(directory, prefix, out_channels);
    return layer;
}

}  // namespace

RpnWeights load_rpn_weights(const std::filesystem::path& directory) {
    RpnWeights weights;
    weights.blocks[0].push_back(
        read_conv(directory, "block0_conv0", 64, 64, 3, 1, 1));
    for (int index = 1; index < 4; ++index) {
        weights.blocks[0].push_back(read_conv(
            directory, "block0_conv" + std::to_string(index),
            64, 64, 3, 1, 1));
    }

    weights.blocks[1].push_back(
        read_conv(directory, "block1_conv0", 64, 128, 3, 2, 1));
    for (int index = 1; index < 6; ++index) {
        weights.blocks[1].push_back(read_conv(
            directory, "block1_conv" + std::to_string(index),
            128, 128, 3, 1, 1));
    }

    weights.blocks[2].push_back(
        read_conv(directory, "block2_conv0", 128, 256, 3, 2, 1));
    for (int index = 1; index < 6; ++index) {
        weights.blocks[2].push_back(read_conv(
            directory, "block2_conv" + std::to_string(index),
            256, 256, 3, 1, 1));
    }

    weights.deblock0 = read_conv(directory, "deblock0", 64, 128, 1, 1, 0);
    weights.deblock1 = read_deconv(directory, "deblock1", 128, 128, 2);
    weights.deblock2 = read_deconv(directory, "deblock2", 256, 128, 4);
    return weights;
}

}  // namespace centerpoint
