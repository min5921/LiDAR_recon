#include "centerpoint/decode.hpp"
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

namespace {
float parse_threshold(const char *text, const char *name) {
  char *end = nullptr;
  const float value = std::strtof(text, &end);
  if (end == text || *end != '\0' || !(value >= 0.0F && value <= 1.0F))
    throw std::runtime_error(std::string(name) +
                             " must be a number in [0, 1]");
  return value;
}

bool parse_nms_convention(const char *text) {
  const std::string value(text);
  if (value == "current" || value == "cpp")
    return false;
  if (value == "pcdet")
    return true;
  throw std::runtime_error(
      "nms_convention must be one of: current, cpp, pcdet");
}

void write_decode_config(const std::filesystem::path &output_dir,
                         const centerpoint::DecodeConfig &config) {
  std::ofstream file(output_dir / "decode_config.json");
  file << "{\n"
       << "  \"score_threshold\": " << config.score_threshold << ",\n"
       << "  \"nms_iou_threshold\": " << config.nms_iou_threshold << ",\n"
       << "  \"use_pcdet_nms_convention\": "
       << (config.use_pcdet_nms_convention ? "true" : "false") << ",\n"
       << "  \"use_class_score_thresholds\": "
       << (config.use_class_score_thresholds ? "true" : "false") << ",\n"
       << "  \"class_score_thresholds\": ["
       << config.class_score_thresholds[0] << ", "
       << config.class_score_thresholds[1] << ", "
       << config.class_score_thresholds[2] << "],\n"
       << "  \"pre_max_size\": " << config.pre_max_size << ",\n"
       << "  \"post_max_size\": " << config.post_max_size << "\n"
       << "}\n";
}
} // namespace

int main(int argc, char **argv) {
  if (argc < 3 || argc > 9 || argc == 7 || argc == 8) {
    std::cerr << "usage: " << argv[0]
              << " <center_head_output_dir> <output_dir> "
                 "[nms_iou_threshold] [score_threshold] [nms_convention] "
                 "[vehicle_score pedestrian_score cyclist_score]\n";
    return 2;
  }
  try {
    const auto maps = centerpoint::read_head_maps(argv[1]);
    centerpoint::DecodeConfig config;
    if (argc >= 4)
      config.nms_iou_threshold = parse_threshold(argv[3], "nms_iou_threshold");
    if (argc >= 5)
      config.score_threshold = parse_threshold(argv[4], "score_threshold");
    if (argc >= 6)
      config.use_pcdet_nms_convention = parse_nms_convention(argv[5]);
    if (argc == 9) {
      config.class_score_thresholds[0] =
          parse_threshold(argv[6], "vehicle_score");
      config.class_score_thresholds[1] =
          parse_threshold(argv[7], "pedestrian_score");
      config.class_score_thresholds[2] =
          parse_threshold(argv[8], "cyclist_score");
      config.use_class_score_thresholds = true;
    }
    const auto result = centerpoint::decode_and_nms(maps, config);
    centerpoint::write_detections(argv[2], result);
    write_decode_config(argv[2], config);
    std::cout << "score/range candidates: " << result.before_nms.size() << '\n';
    std::cout << "detections after rotated NMS: " << result.detections.size()
              << '\n';
    std::cout << "NMS IoU threshold: " << config.nms_iou_threshold << '\n';
    std::cout << "score threshold: " << config.score_threshold << '\n';
    std::cout << "NMS convention: "
              << (config.use_pcdet_nms_convention ? "pcdet" : "current")
              << '\n';
    std::cout << std::fixed << std::setprecision(3)
              << "CUDA decode: " << result.cuda_ms
              << " ms\nC++ rotated NMS: " << result.nms_ms << " ms\n";
    for (std::size_t i = 0; i < result.detections.size() && i < 10; ++i) {
      const auto &d = result.detections[i];
      std::cout << i << ": class=" << d.label << " score=" << d.score
                << " xyz=(" << d.x << ',' << d.y << ',' << d.z << ") size=("
                << d.dx << ',' << d.dy << ',' << d.dz << ") yaw=" << d.yaw
                << '\n';
    }
  } catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << '\n';
    return 1;
  }
  return 0;
}
