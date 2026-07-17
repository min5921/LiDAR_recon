#include "centerpoint/io.hpp"

#include <fstream>
#include <stdexcept>

namespace centerpoint {

std::vector<float> read_point_bin(const std::filesystem::path& path,
                                  int feature_dimension) {
    if (feature_dimension <= 0) {
        throw std::invalid_argument("feature_dimension must be positive");
    }

    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("failed to open point file: " + path.string());
    }
    const std::streamoff byte_count = input.tellg();
    const std::streamoff point_bytes =
        static_cast<std::streamoff>(feature_dimension * sizeof(float));
    if (byte_count < 0 || byte_count % point_bytes != 0) {
        throw std::runtime_error(
            "point file size is not divisible by feature_dimension");
    }

    std::vector<float> points(
        static_cast<std::size_t>(byte_count) / sizeof(float));
    input.seekg(0, std::ios::beg);
    input.read(reinterpret_cast<char*>(points.data()), byte_count);
    if (!input) {
        throw std::runtime_error("failed to read point file: " + path.string());
    }
    return points;
}

void write_float_bin(const std::filesystem::path& path,
                     const std::vector<float>& values) {
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("failed to create float output: " + path.string());
    }
    output.write(reinterpret_cast<const char*>(values.data()),
                 static_cast<std::streamsize>(values.size() * sizeof(float)));
    if (!output) {
        throw std::runtime_error("failed to write float output: " + path.string());
    }
}

}  // namespace centerpoint
