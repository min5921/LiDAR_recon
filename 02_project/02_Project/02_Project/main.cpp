#include <exception>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>
#include <cstdint>
#include <array>
#include <fstream>


// namespace structure -----------------------------------------------------------------

namespace centerpoint {

	struct PointCloud {
		std::vector<float> values;
		int feature_dim = 0;

		int num_points() const {
			return feature_dim > 0 ? static_cast<int>(values.size() / feature_dim) : 0;
		}
	};

	struct VoxelizationConfig {
		std::array<float, 3> voxel_size{ 0.32F, 0.32F, 6.0F };
		std::array<float, 6> point_cloud_range{ -74.88F, -74.88F, -2.0F, 74.88F, 74.88F, 4.0F };
		int max_points_per_voxel = 20;
		int max_voxels = 60000;
		int feature_dim = 5;
	};

	struct VoxelizationResult {
		std::vector<float> pillars;
		std::vector<int32_t> coordinates;
		std::vector<int32_t> num_points_per_voxel;
		std::array<int, 3> grid_size_xyz{ 0, 0, 0 };
		int num_pillars = 0;
		int max_points_per_voxel = 0;
		int feature_dim = 0;
	};
}

// File input -----------------------------------------------------------------

namespace centerpoint::io {

	PointCloud read_float32_point_cloud(const std::filesystem::path& path, int feature_dim) {
		if (feature_dim < 3) {
			throw std::invalid_argument("feature_dim must be at least 3");
		}

		std::ifstream input(path, std::ios::binary | std::ios::ate);
		if (!input) {
			throw std::runtime_error("Failed to open file: " + path.string());
		}

		const auto file_size = input.tellg();
		if (file_size < 0 || static_cast<std::uintmax_t>(file_size) % sizeof(float) != 0) {
			throw std::runtime_error("PointCloud data is not float32 and INVALID");
		};

		const auto float_count = static_cast<std::size_t>(file_size) / sizeof(float);
		if (float_count % feature_dim != 0) {


	}
}


int main() {
	try {
		const std::filesystem::path input_path = "input.txt";
		const std::filesystem::path output_path = "output.txt";

		std::cout << "input path: " << input_path.string() << "\n";
		std::cout << "output path: " << output_path.string() << "\n";
	}
	catch (const std::exception& e) {
		std::cerr << "Error: " << e.what() << std::endl;
		return EXIT_FAILURE;
	}
	return 0;
}