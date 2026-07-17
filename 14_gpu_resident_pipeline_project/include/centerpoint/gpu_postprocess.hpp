#pragma once

#include <array>
#include <memory>
#include <vector>

#include "centerpoint/gpu_center_head.hpp"

namespace centerpoint {

struct Detection {
    float x = 0.0F;
    float y = 0.0F;
    float z = 0.0F;
    float dx = 0.0F;
    float dy = 0.0F;
    float dz = 0.0F;
    float yaw = 0.0F;
    float score = 0.0F;
    int label = 0;
    int source_index = 0;
};

struct GpuPostprocessConfig {
    float score_threshold = 0.35F;
    std::array<float, 3> class_score_thresholds{0.35F, 0.35F, 0.35F};
    bool use_class_score_thresholds = false;
    float nms_iou_threshold = 0.5F;
    bool use_pcdet_nms_convention = true;
    float point_cloud_x = -74.88F;
    float point_cloud_y = -74.88F;
    float cell_x = 0.32F;
    float cell_y = 0.32F;
    std::array<float, 6> post_center_range{
        -80.0F, -80.0F, -10.0F, 80.0F, 80.0F, 10.0F};
    int pre_max_size = 4096;
    int post_max_size = 500;
};

struct DeviceDetectionView {
    const Detection* data = nullptr;
    int count = 0;
};

struct GpuPostprocessStats {
    float elapsed_ms = 0.0F;
    int candidates_before_nms = 0;
    int candidates_after_pre_max = 0;
    int detections_after_nms = 0;
};

class GpuPostprocessPipeline {
public:
    explicit GpuPostprocessPipeline(const GpuPostprocessConfig& config = {});
    ~GpuPostprocessPipeline();

    GpuPostprocessPipeline(const GpuPostprocessPipeline&) = delete;
    GpuPostprocessPipeline& operator=(const GpuPostprocessPipeline&) = delete;
    GpuPostprocessPipeline(GpuPostprocessPipeline&&) noexcept;
    GpuPostprocessPipeline& operator=(GpuPostprocessPipeline&&) noexcept;

    GpuPostprocessStats run(const DeviceHeadMaps& maps);
    DeviceDetectionView device_detections() const;
    std::vector<Detection> copy_detections_to_host() const;
    std::vector<Detection> copy_pre_nms_to_host() const;
    const GpuPostprocessConfig& config() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace centerpoint
