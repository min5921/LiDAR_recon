#include "centerpoint/io/rpn_weight_reader.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace centerpoint::io {
namespace {

std::vector<float> read_floats(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("failed to open RPN weight: " + path.string());
    }
    const auto bytes = input.tellg();
    if (bytes < 0 || static_cast<std::uintmax_t>(bytes) % sizeof(float) != 0) {
        throw std::runtime_error("invalid RPN weight size: " + path.string());
    }
    std::vector<float> values(static_cast<std::size_t>(bytes) / sizeof(float));
    input.seekg(0, std::ios::beg);
    input.read(reinterpret_cast<char*>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(float)));
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

BatchNormWeights read_bn(const std::filesystem::path& dir,
                         const std::string& prefix,
                         int channels) {
    BatchNormWeights bn;
    bn.weight = read_floats(dir / (prefix + "_bn_weight.bin"));
    bn.bias = read_floats(dir / (prefix + "_bn_bias.bin"));
    bn.mean = read_floats(dir / (prefix + "_bn_mean.bin"));
    bn.variance = read_floats(dir / (prefix + "_bn_var.bin"));
    require_size(bn.weight, channels, prefix + " BN weight");
    require_size(bn.bias, channels, prefix + " BN bias");
    require_size(bn.mean, channels, prefix + " BN mean");
    require_size(bn.variance, channels, prefix + " BN variance");
    return bn;
}

ConvLayerWeights read_conv(const std::filesystem::path& dir,
                           const std::string& prefix,
                           int in_channels,
                           int out_channels,
                           int kernel,
                           int stride,
                           int padding) {
    ConvLayerWeights layer;
    layer.name = prefix;
    layer.in_channels = in_channels;
    layer.out_channels = out_channels;
    layer.kernel_size = kernel;
    layer.stride = stride;
    layer.padding = padding;
    layer.weight = read_floats(dir / (prefix + "_weight.bin"));
    require_size(layer.weight,
                 static_cast<std::size_t>(out_channels) * in_channels *
                     kernel * kernel,
                 prefix + " Conv weight");
    layer.batch_norm = read_bn(dir, prefix, out_channels);
    return layer;
}

TransposedConvLayerWeights read_deconv(const std::filesystem::path& dir,
                                       const std::string& prefix,
                                       int in_channels,
                                       int out_channels,
                                       int kernel) {
    TransposedConvLayerWeights layer;
    layer.name = prefix;
    layer.in_channels = in_channels;
    layer.out_channels = out_channels;
    layer.kernel_size = kernel;
    layer.stride = kernel;
    layer.gemm_weight = read_floats(dir / (prefix + "_weight_gemm.bin"));
    require_size(layer.gemm_weight,
                 static_cast<std::size_t>(out_channels) * kernel * kernel *
                     in_channels,
                 prefix + " transposed Conv weight");
    layer.batch_norm = read_bn(dir, prefix, out_channels);
    return layer;
}

}  // namespace

FullRpnWeights read_full_rpn_weights(const std::filesystem::path& weight_dir) {
    FullRpnWeights weights;

    weights.blocks[0].push_back(read_conv(weight_dir, "block0_conv0", 64, 64, 3, 1, 1));
    for (int index = 1; index < 4; ++index) {
        weights.blocks[0].push_back(read_conv(
            weight_dir, "block0_conv" + std::to_string(index), 64, 64, 3, 1, 1));
    }

    weights.blocks[1].push_back(read_conv(weight_dir, "block1_conv0", 64, 128, 3, 2, 1));
    for (int index = 1; index < 6; ++index) {
        weights.blocks[1].push_back(read_conv(
            weight_dir, "block1_conv" + std::to_string(index), 128, 128, 3, 1, 1));
    }

    weights.blocks[2].push_back(read_conv(weight_dir, "block2_conv0", 128, 256, 3, 2, 1));
    for (int index = 1; index < 6; ++index) {
        weights.blocks[2].push_back(read_conv(
            weight_dir, "block2_conv" + std::to_string(index), 256, 256, 3, 1, 1));
    }

    weights.deblock0 = read_conv(weight_dir, "deblock0", 64, 128, 1, 1, 0);
    weights.deblock1 = read_deconv(weight_dir, "deblock1", 128, 128, 2);
    weights.deblock2 = read_deconv(weight_dir, "deblock2", 256, 128, 4);
    return weights;
}

}  // namespace centerpoint::io
