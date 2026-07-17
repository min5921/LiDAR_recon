#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "centerpoint/gpu_preprocess.hpp"
#include "centerpoint/gpu_rpn.hpp"
#include "centerpoint/io.hpp"
#include "centerpoint/pfn_weights.hpp"
#include "centerpoint/rpn_weights.hpp"

namespace {

struct Arguments {
    std::filesystem::path points;
    std::filesystem::path weights_root;
    std::filesystem::path output_directory;
    bool has_output_directory = false;
    bool collect_probes = false;
};

void print_usage(const char* executable) {
    std::cerr << "usage: " << executable
              << " <points.bin> <weights_root> "
                 "[--output-dir <directory>] [--probes]\n";
}

Arguments parse_arguments(int argc, char** argv) {
    if (argc < 3) {
        throw std::invalid_argument("missing required arguments");
    }
    Arguments arguments;
    arguments.points = argv[1];
    arguments.weights_root = argv[2];
    for (int index = 3; index < argc; ++index) {
        const std::string option = argv[index];
        if (option == "--output-dir") {
            if (++index >= argc) {
                throw std::invalid_argument("--output-dir requires a path");
            }
            arguments.output_directory = argv[index];
            arguments.has_output_directory = true;
        } else if (option == "--probes") {
            arguments.collect_probes = true;
        } else {
            throw std::invalid_argument("unknown option: " + option);
        }
    }
    if (arguments.collect_probes && !arguments.has_output_directory) {
        throw std::invalid_argument("--probes requires --output-dir");
    }
    return arguments;
}

void write_integer_array(std::ostream& output, const std::array<int, 3>& values) {
    output << '[' << values[0] << ", " << values[1] << ", " << values[2] << ']';
}

void write_probes(const std::filesystem::path& path,
                  const std::vector<centerpoint::RpnLayerProbe>& probes,
                  float batch_norm_epsilon) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("failed to create RPN probe JSON");
    }
    output << std::setprecision(9)
           << "{\n  \"batch_norm_epsilon\": " << batch_norm_epsilon
           << ",\n  \"probes\": [\n";
    for (std::size_t index = 0; index < probes.size(); ++index) {
        const centerpoint::RpnLayerProbe& probe = probes[index];
        output << "    {\n"
               << "      \"name\": \"" << probe.name << "\",\n"
               << "      \"operation\": \"" << probe.operation << "\",\n"
               << "      \"input_shape\": ";
        write_integer_array(output, probe.input_shape);
        output << ",\n      \"output_shape\": ";
        write_integer_array(output, probe.output_shape);
        output << ",\n"
               << "      \"kernel_size\": " << probe.kernel_size << ",\n"
               << "      \"stride\": " << probe.stride << ",\n"
               << "      \"padding\": " << probe.padding << ",\n"
               << "      \"output_index\": ";
        write_integer_array(output, probe.output_index);
        output << ",\n      \"input_values\": [";
        for (std::size_t value = 0; value < probe.input_values.size(); ++value) {
            if (value != 0) {
                output << ", ";
            }
            output << probe.input_values[value];
        }
        output << "],\n"
               << "      \"output_value\": " << probe.output_value << "\n"
               << "    }" << (index + 1 == probes.size() ? "\n" : ",\n");
    }
    output << "  ]\n}\n";
}

void write_summary(const std::filesystem::path& path,
                   const centerpoint::GpuPreprocessStats& preprocess,
                   const centerpoint::GpuRpnStats& rpn,
                   const centerpoint::DeviceRpnView& output_view) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("failed to create GPU RPN summary");
    }
    output << std::fixed << std::setprecision(6)
           << "{\n"
           << "  \"input_points\": " << preprocess.input_points << ",\n"
           << "  \"valid_points\": " << preprocess.valid_points << ",\n"
           << "  \"selected_pillars\": " << preprocess.selected_pillars << ",\n"
           << "  \"preprocess_ms\": " << preprocess.total_ms << ",\n"
           << "  \"rpn_ms\": " << rpn.elapsed_ms << ",\n"
           << "  \"rpn_shape\": [1, " << output_view.channels << ", "
           << output_view.height << ", " << output_view.width << "],\n"
           << "  \"probe_count\": " << rpn.probe_count << "\n"
           << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Arguments arguments = parse_arguments(argc, argv);
        const centerpoint::GpuPreprocessConfig preprocess_config;
        const std::vector<float> points = centerpoint::read_point_bin(
            arguments.points, preprocess_config.feature_dimension);
        const int point_count = static_cast<int>(
            points.size() /
            static_cast<std::size_t>(preprocess_config.feature_dimension));

        const centerpoint::PfnWeights pfn_weights =
            centerpoint::load_pfn_weights(arguments.weights_root / "04_pfn");
        const centerpoint::RpnWeights rpn_weights =
            centerpoint::load_rpn_weights(arguments.weights_root / "06_rpn");
        centerpoint::GpuPreprocessPipeline preprocess_pipeline(
            preprocess_config, pfn_weights);
        centerpoint::GpuRpnPipeline rpn_pipeline(rpn_weights);

        const centerpoint::GpuPreprocessStats preprocess_stats =
            preprocess_pipeline.run(points.data(), point_count);
        const centerpoint::GpuRpnStats rpn_stats = rpn_pipeline.run(
            preprocess_pipeline.device_bev(), arguments.collect_probes);
        const centerpoint::DeviceRpnView rpn_output =
            rpn_pipeline.device_output();

        std::cout << std::fixed << std::setprecision(3)
                  << "points: " << preprocess_stats.input_points << '\n'
                  << "pillars: " << preprocess_stats.selected_pillars << '\n'
                  << "BEV device input: [1, 64, 468, 468]\n"
                  << "RPN device output: [1, " << rpn_output.channels << ", "
                  << rpn_output.height << ", " << rpn_output.width << "]\n"
                  << "preprocess: " << preprocess_stats.total_ms << " ms\n"
                  << "RPN: " << rpn_stats.elapsed_ms << " ms\n"
                  << "layer probes: " << rpn_stats.probe_count << '\n'
                  << "intermediate tensor files: 0\n";

        if (arguments.has_output_directory) {
            std::filesystem::create_directories(arguments.output_directory);
            write_summary(arguments.output_directory / "summary.json",
                          preprocess_stats, rpn_stats, rpn_output);
            if (arguments.collect_probes) {
                write_probes(arguments.output_directory / "rpn_probes.json",
                             rpn_pipeline.probes(),
                             rpn_weights.batch_norm_epsilon);
            }
            std::cout << "summary output: "
                      << arguments.output_directory.string() << '\n';
        }
        return 0;
    } catch (const std::invalid_argument& error) {
        print_usage(argv[0]);
        std::cerr << "error: " << error.what() << '\n';
        return 2;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
