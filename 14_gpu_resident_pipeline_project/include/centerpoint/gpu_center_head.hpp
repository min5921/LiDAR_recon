#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "centerpoint/gpu_rpn.hpp"
#include "centerpoint/head_weights.hpp"

namespace centerpoint {

struct DeviceHeadMaps {
    const float* reg = nullptr;
    const float* height = nullptr;
    const float* dim = nullptr;
    const float* rot = nullptr;
    const float* heatmap = nullptr;
    int height_size = 0;
    int width_size = 0;
};

struct HostHeadMaps {
    std::array<std::vector<float>, 5> values;
    int height_size = 0;
    int width_size = 0;
};

struct HeadLayerProbe {
    std::string name;
    std::array<int, 3> input_shape{};
    std::array<int, 3> output_shape{};
    std::array<int, 3> output_index{};
    std::vector<float> input_values;
    float output_value = 0.0F;
    bool has_batch_norm = false;
};

struct GpuCenterHeadStats {
    float elapsed_ms = 0.0F;
    int probe_count = 0;
};

class GpuCenterHeadPipeline {
public:
    explicit GpuCenterHeadPipeline(const HeadWeights& weights);
    ~GpuCenterHeadPipeline();

    GpuCenterHeadPipeline(const GpuCenterHeadPipeline&) = delete;
    GpuCenterHeadPipeline& operator=(const GpuCenterHeadPipeline&) = delete;
    GpuCenterHeadPipeline(GpuCenterHeadPipeline&&) noexcept;
    GpuCenterHeadPipeline& operator=(GpuCenterHeadPipeline&&) noexcept;

    GpuCenterHeadStats run(const DeviceRpnView& input,
                           bool collect_probes = false);
    DeviceHeadMaps device_maps() const;
    HostHeadMaps copy_maps_to_host() const;
    const std::vector<HeadLayerProbe>& probes() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace centerpoint
