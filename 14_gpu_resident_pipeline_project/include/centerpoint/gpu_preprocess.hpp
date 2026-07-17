#pragma once

#include <array>
#include <memory>
#include <vector>

#include "centerpoint/pfn_weights.hpp"

namespace centerpoint {

struct GpuPreprocessConfig {
    std::array<float, 3> voxel_size{0.32F, 0.32F, 6.0F};
    std::array<float, 6> point_cloud_range{
        -74.88F, -74.88F, -2.0F, 74.88F, 74.88F, 4.0F};
    int feature_dimension = 5;
    int max_points_per_pillar = 20;
    int max_pillars = 60000;
};

struct DeviceBevView {
    const float* data = nullptr;
    int channels = 0;
    int height = 0;
    int width = 0;
};

struct GpuPreprocessStats {
    int input_points = 0;
    int valid_points = 0;
    int unique_pillars = 0;
    int selected_pillars = 0;
    float host_to_device_ms = 0.0F;
    float voxelization_ms = 0.0F;
    float pfn_ms = 0.0F;
    float scatter_ms = 0.0F;
    float total_ms = 0.0F;
};

class GpuPreprocessPipeline {
public:
    GpuPreprocessPipeline(const GpuPreprocessConfig& config,
                          const PfnWeights& weights);
    ~GpuPreprocessPipeline();

    GpuPreprocessPipeline(const GpuPreprocessPipeline&) = delete;
    GpuPreprocessPipeline& operator=(const GpuPreprocessPipeline&) = delete;
    GpuPreprocessPipeline(GpuPreprocessPipeline&&) noexcept;
    GpuPreprocessPipeline& operator=(GpuPreprocessPipeline&&) noexcept;

    GpuPreprocessStats run(const float* host_points, int point_count);
    DeviceBevView device_bev() const;
    std::vector<float> copy_bev_to_host() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace centerpoint
