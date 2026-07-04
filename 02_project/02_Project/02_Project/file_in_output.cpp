#include <filesystem>
#include <fstream>

#include "type.hpp"

namespace centerpoint::io {

	namespace {

		template <typename T>
		void write_binary(const std::filesystem::path& path, const std::vector<T>& values) {
			std::ofstream output(path, std::ios::binary);
			if (!output) {
				throw std::runtime_error("failed to open output file: " + path.string());
			}

			output.write(reinterpret_cast<const char*>(values.data()),
				static_cast<std::streamsize>(values.size() * sizeof(T)));
			if (!output) {
				throw std::runtime_error("failed to write output file: " + path.string());
			}
		}
	}

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
			throw std::runtime_error("Point cloud float count is not divisble by feature_dime");
		}

		PointCloud cloud;
		cloud.feature_dim = feature_dim;
		cloud.values.resize(float_count);

		input.seekg(0, std::ios::beg);
		input.read(reinterpret_cast<char*>(cloud.values.data()),
			static_cast<std::streamsize>(float_count * sizeof(float)));

		if (!input) {
			throw std::runtime_error("failed to read point cloud file: " + path.string());
		}

		return cloud;
	}

	void write_debug_dump(const std::filesystem::path& output_dir,
		const VoxelizationConfig& config,
		const VoxelizationResult& result) {
		std::filesystem::create_directories(output_dir);

		write_binary(output_dir / "pillars.bin", result.pillars);
		write_binary(output_dir / "coordinates.bin", result.coordinates);
		write_binary(output_dir / "num_points.bin", result.num_points_per_pillar);

		std::ofstream meta(output_dir / "metadata.json");
		if (!meta) {
			throw std::runtime_error("failed to open metadata.json");
		}

		meta << "{\n";
		meta << "  \"num_pillars\": " << result.num_pillars << ",\n";
		meta << "  \"max_points_per_pillar\": " << result.max_points_per_pillar << ",\n";
		meta << "  \"feature_dim\": " << result.feature_dim << ",\n";
		meta << "  \"coordinate_order\": \"batch,z,y,x\",\n";
		meta << "  \"grid_size_xyz\": ["
			<< result.grid_size_xyz[0] << ", "
			<< result.grid_size_xyz[1] << ", "
			<< result.grid_size_xyz[2] << "],\n";
		meta << "  \"voxel_size\": ["
			<< config.voxel_size[0] << ", "
			<< config.voxel_size[1] << ", "
			<< config.voxel_size[2] << "],\n";
		meta << "  \"point_cloud_range\": ["
			<< config.point_cloud_range[0] << ", "
			<< config.point_cloud_range[1] << ", "
			<< config.point_cloud_range[2] << ", "
			<< config.point_cloud_range[3] << ", "
			<< config.point_cloud_range[4] << ", "
			<< config.point_cloud_range[5] << "]\n";
		meta << "}\n";
	}

}