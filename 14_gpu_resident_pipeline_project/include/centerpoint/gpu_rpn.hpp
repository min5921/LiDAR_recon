#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "centerpoint/gpu_preprocess.hpp"
#include "centerpoint/rpn_weights.hpp"

namespace centerpoint {

struct DeviceRpnView {
    const float* data = nullptr;
    int channels = 0;
    int height = 0;
    int width = 0;
};

struct RpnLayerProbe {
    std::string name;
    std::string operation;
    std::array<int, 3> input_shape{};
    std::array<int, 3> output_shape{};
    int kernel_size = 0;
    int stride = 0;
    int padding = 0;
    std::array<int, 3> output_index{};
    std::vector<float> input_values;
    float output_value = 0.0F;
};

struct GpuRpnStats {
    float elapsed_ms = 0.0F;
    std::array<std::array<int, 3>, 3> block_shapes{};
    std::array<std::array<int, 3>, 3> deblock_shapes{};
    int probe_count = 0;
};

class GpuRpnPipeline {
public:
    explicit GpuRpnPipeline(const RpnWeights& weights);
    ~GpuRpnPipeline();

    GpuRpnPipeline(const GpuRpnPipeline&) = delete;
    GpuRpnPipeline& operator=(const GpuRpnPipeline&) = delete;
    GpuRpnPipeline(GpuRpnPipeline&&) noexcept;
    GpuRpnPipeline& operator=(GpuRpnPipeline&&) noexcept;

    GpuRpnStats run(const DeviceBevView& input, bool collect_probes = false);
    DeviceRpnView device_output() const;
    const std::vector<RpnLayerProbe>& probes() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace centerpoint
