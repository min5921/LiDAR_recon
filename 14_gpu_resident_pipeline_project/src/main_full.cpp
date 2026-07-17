#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "centerpoint/gpu_center_head.hpp"
#include "centerpoint/gpu_postprocess.hpp"
#include "centerpoint/gpu_preprocess.hpp"
#include "centerpoint/gpu_rpn.hpp"
#include "centerpoint/head_weights.hpp"
#include "centerpoint/io.hpp"
#include "centerpoint/pfn_weights.hpp"
#include "centerpoint/rpn_weights.hpp"

namespace {

struct Arguments {
    std::filesystem::path points;
    std::filesystem::path weights_root;
    std::filesystem::path output_directory;
    bool has_output_directory = false;
    bool validation = false;
    centerpoint::GpuPostprocessConfig postprocess;
};

float parse_unit_float(const char* text, const char* name) {
    char* end = nullptr;
    const float value = std::strtof(text, &end);
    if (end == text || *end != '\0' || value < 0.0F || value > 1.0F) {
        throw std::invalid_argument(std::string(name) +
                                    " must be in [0,1]");
    }
    return value;
}

void print_usage(const char* executable) {
    std::cerr
        << "usage: " << executable << " <points.bin> <weights_root> "
        << "[--output-dir <directory>] [--validation] "
        << "[--score-threshold <value>] [--nms-iou <value>] "
        << "[--nms-convention pcdet|current] "
        << "[--class-thresholds <vehicle> <pedestrian> <cyclist>]\n";
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
        } else if (option == "--validation") {
            arguments.validation = true;
        } else if (option == "--score-threshold") {
            if (++index >= argc) {
                throw std::invalid_argument(
                    "--score-threshold requires a value");
            }
            arguments.postprocess.score_threshold =
                parse_unit_float(argv[index], "score threshold");
        } else if (option == "--nms-iou") {
            if (++index >= argc) {
                throw std::invalid_argument("--nms-iou requires a value");
            }
            arguments.postprocess.nms_iou_threshold =
                parse_unit_float(argv[index], "NMS IoU");
        } else if (option == "--nms-convention") {
            if (++index >= argc) {
                throw std::invalid_argument(
                    "--nms-convention requires a value");
            }
            const std::string convention = argv[index];
            if (convention == "pcdet") {
                arguments.postprocess.use_pcdet_nms_convention = true;
            } else if (convention == "current" || convention == "cpp") {
                arguments.postprocess.use_pcdet_nms_convention = false;
            } else {
                throw std::invalid_argument(
                    "NMS convention must be pcdet or current");
            }
        } else if (option == "--class-thresholds") {
            if (index + 3 >= argc) {
                throw std::invalid_argument(
                    "--class-thresholds requires three values");
            }
            for (int label = 0; label < 3; ++label) {
                arguments.postprocess.class_score_thresholds[label] =
                    parse_unit_float(argv[++index], "class threshold");
            }
            arguments.postprocess.use_class_score_thresholds = true;
        } else {
            throw std::invalid_argument("unknown option: " + option);
        }
    }
    if (arguments.validation && !arguments.has_output_directory) {
        throw std::invalid_argument(
            "--validation requires --output-dir");
    }
    return arguments;
}

void write_shape(std::ostream& output, const std::array<int, 3>& shape) {
    output << '[' << shape[0] << ", " << shape[1] << ", " << shape[2]
           << ']';
}

void write_head_probes(
    const std::filesystem::path& path,
    const std::vector<centerpoint::HeadLayerProbe>& probes,
    float epsilon) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("failed to create CenterHead probe JSON");
    }
    output << std::setprecision(9)
           << "{\n  \"batch_norm_epsilon\": " << epsilon
           << ",\n  \"probes\": [\n";
    for (std::size_t index = 0; index < probes.size(); ++index) {
        const auto& probe = probes[index];
        output << "    {\n"
               << "      \"name\": \"" << probe.name << "\",\n"
               << "      \"has_batch_norm\": "
               << (probe.has_batch_norm ? "true" : "false") << ",\n"
               << "      \"input_shape\": ";
        write_shape(output, probe.input_shape);
        output << ",\n      \"output_shape\": ";
        write_shape(output, probe.output_shape);
        output << ",\n      \"output_index\": ";
        write_shape(output, probe.output_index);
        output << ",\n      \"input_values\": [";
        for (std::size_t value = 0; value < probe.input_values.size();
             ++value) {
            if (value != 0) {
                output << ", ";
            }
            output << probe.input_values[value];
        }
        output << "],\n      \"output_value\": " << probe.output_value
               << "\n    }"
               << (index + 1 == probes.size() ? "\n" : ",\n");
    }
    output << "  ]\n}\n";
}

void write_detection_rows(const std::filesystem::path& path,
                          const std::vector<centerpoint::Detection>& values) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("failed to create detection CSV");
    }
    output << std::setprecision(9)
           << "x,y,z,dx,dy,dz,yaw,score,label,source_index\n";
    for (const auto& value : values) {
        output << value.x << ',' << value.y << ',' << value.z << ','
               << value.dx << ',' << value.dy << ',' << value.dz << ','
               << value.yaw << ',' << value.score << ',' << value.label << ','
               << value.source_index << '\n';
    }
}

void write_summary(
    const std::filesystem::path& path,
    const centerpoint::GpuPreprocessStats& preprocess,
    const centerpoint::GpuRpnStats& rpn,
    const centerpoint::GpuCenterHeadStats& head,
    const centerpoint::GpuPostprocessStats& postprocess,
    const centerpoint::GpuPostprocessConfig& config,
    bool validation) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("failed to create full-pipeline summary");
    }
    output << std::fixed << std::setprecision(6)
           << "{\n"
           << "  \"input_points\": " << preprocess.input_points << ",\n"
           << "  \"valid_points\": " << preprocess.valid_points << ",\n"
           << "  \"selected_pillars\": " << preprocess.selected_pillars
           << ",\n"
           << "  \"preprocess_ms\": " << preprocess.total_ms << ",\n"
           << "  \"rpn_ms\": " << rpn.elapsed_ms << ",\n"
           << "  \"center_head_ms\": " << head.elapsed_ms << ",\n"
           << "  \"postprocess_ms\": " << postprocess.elapsed_ms << ",\n"
           << "  \"rpn_shape\": [1, 384, 468, 468],\n"
           << "  \"head_shapes\": {\"reg\": [1,2,468,468], "
              "\"height\": [1,1,468,468], \"dim\": [1,3,468,468], "
              "\"rot\": [1,2,468,468], \"hm\": [1,3,468,468]},\n"
           << "  \"candidates_before_nms\": "
           << postprocess.candidates_before_nms << ",\n"
           << "  \"candidates_after_pre_max\": "
           << postprocess.candidates_after_pre_max << ",\n"
           << "  \"detections_after_nms\": "
           << postprocess.detections_after_nms << ",\n"
           << "  \"score_threshold\": " << config.score_threshold << ",\n"
           << "  \"nms_iou_threshold\": " << config.nms_iou_threshold
           << ",\n"
           << "  \"nms_convention\": \""
           << (config.use_pcdet_nms_convention ? "pcdet" : "current")
           << "\",\n"
           << "  \"head_probe_count\": " << head.probe_count << ",\n"
           << "  \"validation_host_copies\": "
           << (validation ? "true" : "false") << ",\n"
           << "  \"intermediate_tensor_files\": 0\n"
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
        const centerpoint::HeadWeights head_weights =
            centerpoint::load_head_weights(arguments.weights_root / "07_head");

        centerpoint::GpuPreprocessPipeline preprocess_pipeline(
            preprocess_config, pfn_weights);
        centerpoint::GpuRpnPipeline rpn_pipeline(rpn_weights);
        centerpoint::GpuCenterHeadPipeline head_pipeline(head_weights);
        centerpoint::GpuPostprocessPipeline postprocess_pipeline(
            arguments.postprocess);

        const auto preprocess_stats =
            preprocess_pipeline.run(points.data(), point_count);
        const auto rpn_stats =
            rpn_pipeline.run(preprocess_pipeline.device_bev(), false);
        const auto head_stats = head_pipeline.run(
            rpn_pipeline.device_output(), arguments.validation);
        const auto postprocess_stats =
            postprocess_pipeline.run(head_pipeline.device_maps());
        const std::vector<centerpoint::Detection> detections =
            postprocess_pipeline.copy_detections_to_host();

        std::cout << std::fixed << std::setprecision(3)
                  << "points: " << preprocess_stats.input_points << '\n'
                  << "pillars: " << preprocess_stats.selected_pillars << '\n'
                  << "RPN device output: [1, 384, 468, 468]\n"
                  << "CenterHead device maps: reg=2 height=1 dim=3 rot=2 hm=3\n"
                  << "candidates before NMS: "
                  << postprocess_stats.candidates_before_nms << '\n'
                  << "detections after GPU NMS: " << detections.size() << '\n'
                  << "preprocess: " << preprocess_stats.total_ms << " ms\n"
                  << "RPN: " << rpn_stats.elapsed_ms << " ms\n"
                  << "CenterHead: " << head_stats.elapsed_ms << " ms\n"
                  << "GPU decode/sort/NMS: " << postprocess_stats.elapsed_ms
                  << " ms\n"
                  << "intermediate tensor files: 0\n";
        for (std::size_t index = 0;
             index < detections.size() && index < 5; ++index) {
            const auto& detection = detections[index];
            std::cout << index << ": class=" << detection.label
                      << " score=" << detection.score << " xyz=("
                      << detection.x << ',' << detection.y << ','
                      << detection.z << ") size=(" << detection.dx << ','
                      << detection.dy << ',' << detection.dz << ") yaw="
                      << detection.yaw << '\n';
        }

        if (arguments.has_output_directory) {
            std::filesystem::create_directories(arguments.output_directory);
            write_detection_rows(
                arguments.output_directory / "detections.csv", detections);
            write_summary(arguments.output_directory / "summary.json",
                          preprocess_stats, rpn_stats, head_stats,
                          postprocess_stats, postprocess_pipeline.config(),
                          arguments.validation);
            if (arguments.validation) {
                write_head_probes(
                    arguments.output_directory / "head_probes.json",
                    head_pipeline.probes(),
                    head_weights.batch_norm_epsilon);
                write_detection_rows(
                    arguments.output_directory / "pre_nms_candidates.csv",
                    postprocess_pipeline.copy_pre_nms_to_host());
            }
            std::cout << "output: "
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
