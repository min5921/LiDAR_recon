#include "centerpoint/decode.hpp"
#include <exception>
#include <iomanip>
#include <iostream>

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "usage: " << argv[0]
              << " <center_head_output_dir> <output_dir>\n";
    return 2;
  }
  try {
    const auto maps = centerpoint::read_head_maps(argv[1]);
    const centerpoint::DecodeConfig config;
    const auto result = centerpoint::decode_and_nms(maps, config);
    centerpoint::write_detections(argv[2], result);
    std::cout << "score/range candidates: " << result.before_nms.size() << '\n';
    std::cout << "detections after rotated NMS: " << result.detections.size()
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
