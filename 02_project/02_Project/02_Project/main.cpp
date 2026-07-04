#include <exception>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>
#include <cstdint>
#include <array>
#include <fstream>

#include "type.hpp"
#include "file_in_output.hpp"
#include "voxeolization.hpp"


int main() {
	try {
		const std::filesystem::path input_path = "C:/Users/USER/Desktop/Coding/Cpp/new/LiDAR_recon/00_reference/sample_data/kitti/000000.bin";
		const std::filesystem::path output_path = "output.txt";

		std::cout << "input path: " << input_path.string() << "\n";
		std::cout << "output path: " << output_path.string() << "\n";

		int feature_dim = 4;

		centerpoint::VoxelizationConfig config;
		config.feature_dim = feature_dim;

		const centerpoint::PointCloud cloud =
			centerpoint::io::read_float32_point_cloud(input_path, feature_dim);
		const centerpoint::VoxelizationResult result = centerpoint::voxelize_cpu(cloud, config);

		centerpoint::io::write_debug_dump(output_path, config, result);

	}
	catch (const std::exception& e) {
		std::cerr << "Error: " << e.what() << std::endl;
		return EXIT_FAILURE;
	}
	return 0;
}