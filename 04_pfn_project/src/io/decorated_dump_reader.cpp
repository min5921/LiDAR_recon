#include "centerpoint/io/decorated_dump_reader.hpp"

#include <filesystem>
#include <fstream>
#include <regex>
#include <stdexcept>
#include <string>

namespace centerpoint::io {
namespace {

std::string read_text(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open metadata file: " + path.string());
    }
    return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

int read_int_field(const std::string& text, const std::string& name) {
    const std::regex pattern("\"" + name + "\"\\s*:\\s*(-?\\d+)");
    std::smatch match;
    if (!std::regex_search(text, match, pattern)) {
        throw std::runtime_error("missing integer metadata field: " + name);
    }
    return std::stoi(match[1].str());
}

std::vector<float> read_float_vector(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("failed to open binary file: " + path.string());
    }

    const auto byte_size = input.tellg();
    if (byte_size < 0 || static_cast<std::uintmax_t>(byte_size) % sizeof(float) != 0) {
        throw std::runtime_error("invalid float binary file size: " + path.string());
    }

    std::vector<float> values(static_cast<std::size_t>(byte_size) / sizeof(float));
    input.seekg(0, std::ios::beg);
    input.read(reinterpret_cast<char*>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(float)));
    if (!input) {
        throw std::runtime_error("failed to read binary file: " + path.string());
    }
    return values;
}

}  // namespace

DecoratedPillarDump read_decorated_pillar_dump(const std::filesystem::path& dump_dir) {
    const std::string metadata_text = read_text(dump_dir / "decorated_metadata.json");

    DecoratedPillarDump dump;
    dump.metadata.num_pillars = read_int_field(metadata_text, "num_pillars");
    dump.metadata.max_points_per_pillar = read_int_field(metadata_text, "max_points_per_pillar");
    dump.metadata.input_feature_dim = read_int_field(metadata_text, "input_feature_dim");
    dump.metadata.decorated_feature_dim = read_int_field(metadata_text, "decorated_feature_dim");
    dump.decorated_pillars = read_float_vector(dump_dir / "decorated_pillars.bin");

    const std::size_t expected =
        static_cast<std::size_t>(dump.metadata.num_pillars) *
        dump.metadata.max_points_per_pillar *
        dump.metadata.decorated_feature_dim;
    if (dump.decorated_pillars.size() != expected) {
        throw std::runtime_error("decorated_pillars.bin size does not match metadata");
    }

    return dump;
}

}  // namespace centerpoint::io

