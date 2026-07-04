#pragma once
#include <array>
#include <filesystem>
#include <vector>

namespace centerpoint {
struct HeadMaps {
  std::vector<float> reg, height, dim, rot, hm;
  int height_size = 468, width_size = 468;
};
struct Detection {
  float x = 0, y = 0, z = 0, dx = 0, dy = 0, dz = 0, yaw = 0, score = 0;
  int label = 0, source_index = 0;
};
struct DecodeConfig {
  float score_threshold = 0.1F, pc_x = -74.88F, pc_y = -74.88F;
  float voxel_x = 0.32F, voxel_y = 0.32F;
  std::array<float, 6> post_range{-80, -80, -10, 80, 80, 10};
  float nms_iou_threshold = 0.7F;
  int pre_max_size = 4096, post_max_size = 500;
};
struct DecodeResult {
  std::vector<Detection> before_nms, detections;
  float cuda_ms = 0, nms_ms = 0;
};
HeadMaps read_head_maps(const std::filesystem::path &dir);
DecodeResult decode_and_nms(const HeadMaps &maps, const DecodeConfig &config);
void write_detections(const std::filesystem::path &dir,
                      const DecodeResult &result);
} // namespace centerpoint
