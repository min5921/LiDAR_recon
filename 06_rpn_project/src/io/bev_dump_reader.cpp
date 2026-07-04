#include "centerpoint/io/bev_dump_reader.hpp"

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
        throw std::runtime_error("failed to open BEV metadata: " + path.string());
    }
    return std::string(std::istreambuf_iterator<char>(input),
                       std::istreambuf_iterator<char>());
}

int read_int(const std::string& text, const std::string& name) {
    const std::regex pattern("\"" + name + "\"\\s*:\\s*(\\d+)");
    std::smatch match;
    if (!std::regex_search(text, match, pattern)) {
        throw std::runtime_error("missing BEV metadata field: " + name);
    }
    return std::stoi(match[1].str());
}

std::vector<float> read_floats(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("failed to open BEV tensor: " + path.string());
    }
    const auto bytes = input.tellg();
    if (bytes < 0 || static_cast<std::uintmax_t>(bytes) % sizeof(float) != 0) {
        throw std::runtime_error("invalid BEV tensor size");
    }
    std::vector<float> values(static_cast<std::size_t>(bytes) / sizeof(float));
    input.seekg(0, std::ios::beg);
    input.read(reinterpret_cast<char*>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(float)));
    if (!input) {
        throw std::runtime_error("failed to read BEV tensor");
    }
    return values;
}

}  // namespace

HostTensor read_bev_dump(const std::filesystem::path& dump_dir) {
    const std::string metadata =
        read_text(dump_dir / "bev_features_metadata.json");
    const int batch_size = read_int(metadata, "batch_size");
    if (batch_size != 1) {
        throw std::runtime_error("full RPN currently supports batch size 1");
    }

    HostTensor tensor;
    tensor.channels = read_int(metadata, "channels");
    tensor.height = read_int(metadata, "height");
    tensor.width = read_int(metadata, "width");
    tensor.values = read_floats(dump_dir / "bev_features.bin");
    const std::size_t expected =
        static_cast<std::size_t>(tensor.channels) * tensor.height * tensor.width;
    if (tensor.values.size() != expected) {
        throw std::runtime_error("BEV tensor size does not match metadata");
    }
    return tensor;
}

}  // namespace centerpoint::io
