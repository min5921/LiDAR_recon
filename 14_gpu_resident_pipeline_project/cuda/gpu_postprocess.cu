#include "centerpoint/gpu_postprocess.hpp"

#include <cub/device/device_radix_sort.cuh>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace centerpoint {
namespace {

constexpr int kThreads = 256;
constexpr int kNmsThreads = 64;
constexpr int kHeight = 468;
constexpr int kWidth = 468;
constexpr int kSpatial = kHeight * kWidth;
constexpr int kMaximumPreNms = 4096;
constexpr int kMaximumPostNms = 500;
constexpr int kMaskBlocks =
    (kMaximumPreNms + kNmsThreads - 1) / kNmsThreads;

struct DevicePostprocessConfig {
    float score_threshold;
    float class_score_thresholds[3];
    bool use_class_score_thresholds;
    float nms_iou_threshold;
    bool use_pcdet_nms_convention;
    float point_cloud_x;
    float point_cloud_y;
    float cell_x;
    float cell_y;
    float post_center_range[6];
    int pre_max_size;
    int post_max_size;
};

DevicePostprocessConfig make_device_config(
    const GpuPostprocessConfig& source) {
    DevicePostprocessConfig result{};
    result.score_threshold = source.score_threshold;
    for (int index = 0; index < 3; ++index) {
        result.class_score_thresholds[index] =
            source.class_score_thresholds[index];
    }
    result.use_class_score_thresholds = source.use_class_score_thresholds;
    result.nms_iou_threshold = source.nms_iou_threshold;
    result.use_pcdet_nms_convention = source.use_pcdet_nms_convention;
    result.point_cloud_x = source.point_cloud_x;
    result.point_cloud_y = source.point_cloud_y;
    result.cell_x = source.cell_x;
    result.cell_y = source.cell_y;
    for (int index = 0; index < 6; ++index) {
        result.post_center_range[index] = source.post_center_range[index];
    }
    result.pre_max_size = source.pre_max_size;
    result.post_max_size = source.post_max_size;
    return result;
}

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::ostringstream message;
        message << operation << " failed: " << cudaGetErrorString(status);
        throw std::runtime_error(message.str());
    }
}

int block_count(std::size_t count) {
    return static_cast<int>((count + kThreads - 1) / kThreads);
}

class CudaEvent {
public:
    CudaEvent() { check_cuda(cudaEventCreate(&event_), "cudaEventCreate"); }
    ~CudaEvent() {
        if (event_ != nullptr) {
            cudaEventDestroy(event_);
        }
    }
    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;
    void record() { check_cuda(cudaEventRecord(event_), "cudaEventRecord"); }
    void synchronize() {
        check_cuda(cudaEventSynchronize(event_), "cudaEventSynchronize");
    }
    cudaEvent_t get() const { return event_; }

private:
    cudaEvent_t event_ = nullptr;
};

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t count) { allocate(count); }
    ~DeviceBuffer() { reset(); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept
        : data_(other.data_), count_(other.count_) {
        other.data_ = nullptr;
        other.count_ = 0;
    }
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            reset();
            data_ = other.data_;
            count_ = other.count_;
            other.data_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    T* data() { return data_; }
    const T* data() const { return data_; }

private:
    void allocate(std::size_t count) {
        if (count == 0) {
            return;
        }
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&data_),
                              count * sizeof(T)),
                   "cudaMalloc postprocess buffer");
        count_ = count;
    }
    void reset() {
        if (data_ != nullptr) {
            cudaFree(data_);
            data_ = nullptr;
        }
        count_ = 0;
    }

    T* data_ = nullptr;
    std::size_t count_ = 0;
};

__device__ float sigmoid(float value) {
    return 1.0F / (1.0F + expf(-value));
}

__device__ float class_threshold(int label,
                                 float default_threshold,
                                 float threshold0,
                                 float threshold1,
                                 float threshold2,
                                 bool use_class_thresholds) {
    if (!use_class_thresholds) {
        return default_threshold;
    }
    return label == 0 ? threshold0 : (label == 1 ? threshold1 : threshold2);
}

__device__ bool decode_at_source(const DeviceHeadMaps maps,
                                 int source_index,
                                 float score,
                                 int label,
                                 const DevicePostprocessConfig config,
                                 Detection* output) {
    const int y = source_index / maps.width_size;
    const int x = source_index % maps.width_size;
    const int spatial = maps.height_size * maps.width_size;
    Detection result;
    result.x = (static_cast<float>(x) + maps.reg[source_index]) *
                   config.cell_x +
               config.point_cloud_x;
    result.y =
        (static_cast<float>(y) + maps.reg[spatial + source_index]) *
            config.cell_y +
        config.point_cloud_y;
    result.z = maps.height[source_index];
    result.dx = expf(maps.dim[source_index]);
    result.dy = expf(maps.dim[spatial + source_index]);
    result.dz = expf(maps.dim[2 * spatial + source_index]);
    result.yaw = atan2f(maps.rot[source_index],
                        maps.rot[spatial + source_index]);
    result.score = score;
    result.label = label;
    result.source_index = source_index;

    const bool finite =
        isfinite(result.x) && isfinite(result.y) && isfinite(result.z) &&
        isfinite(result.dx) && isfinite(result.dy) && isfinite(result.dz) &&
        isfinite(result.yaw) && isfinite(result.score);
    const bool in_range =
        result.x >= config.post_center_range[0] &&
        result.y >= config.post_center_range[1] &&
        result.z >= config.post_center_range[2] &&
        result.x <= config.post_center_range[3] &&
        result.y <= config.post_center_range[4] &&
        result.z <= config.post_center_range[5];
    if (!finite || !in_range) {
        return false;
    }
    *output = result;
    return true;
}

__global__ void generate_candidate_scores_kernel(
    DeviceHeadMaps maps,
    DevicePostprocessConfig config,
    float* scores,
    int* source_indices,
    int* candidate_count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= kSpatial) {
        return;
    }
    source_indices[index] = index;
    int label = 0;
    float logit = maps.heatmap[index];
    for (int channel = 1; channel < 3; ++channel) {
        const float candidate = maps.heatmap[channel * kSpatial + index];
        if (candidate > logit) {
            logit = candidate;
            label = channel;
        }
    }
    const float score = sigmoid(logit);
    const float threshold = class_threshold(
        label, config.score_threshold, config.class_score_thresholds[0],
        config.class_score_thresholds[1], config.class_score_thresholds[2],
        config.use_class_score_thresholds);
    Detection decoded;
    if (score > threshold &&
        decode_at_source(maps, index, score, label, config, &decoded)) {
        scores[index] = score;
        atomicAdd(candidate_count, 1);
    } else {
        scores[index] = -1.0F;
    }
}

__global__ void prepare_pre_nms_kernel(DeviceHeadMaps maps,
                                       DevicePostprocessConfig config,
                                       const float* sorted_scores,
                                       const int* sorted_indices,
                                       Detection* boxes) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= config.pre_max_size) {
        return;
    }
    const float score = sorted_scores[index];
    Detection result;
    result.score = -1.0F;
    if (score >= 0.0F) {
        const int source_index = sorted_indices[index];
        int label = 0;
        float logit = maps.heatmap[source_index];
        for (int channel = 1; channel < 3; ++channel) {
            const float candidate =
                maps.heatmap[channel * kSpatial + source_index];
            if (candidate > logit) {
                logit = candidate;
                label = channel;
            }
        }
        decode_at_source(maps, source_index, score, label, config, &result);
    }
    boxes[index] = result;
}

struct DevicePoint {
    double x;
    double y;
};

__device__ double cross(DevicePoint a, DevicePoint b, DevicePoint c) {
    return (b.x - a.x) * (c.y - a.y) -
           (b.y - a.y) * (c.x - a.x);
}

__device__ DevicePoint intersection(DevicePoint a,
                                    DevicePoint b,
                                    DevicePoint p,
                                    DevicePoint q) {
    const double first = cross(p, q, a);
    const double second = cross(p, q, b);
    const double denominator = first - second;
    if (fabs(denominator) < 1.0e-12) {
        return b;
    }
    const double scale = first / denominator;
    return {a.x + (b.x - a.x) * scale,
            a.y + (b.y - a.y) * scale};
}

__device__ void box_corners(const Detection& box,
                            bool pcdet_convention,
                            DevicePoint* output) {
    double yaw = box.yaw;
    double half_x = box.dx * 0.5;
    double half_y = box.dy * 0.5;
    if (pcdet_convention) {
        constexpr double kHalfPi = 1.57079632679489661923;
        yaw = -box.yaw - kHalfPi;
        half_x = box.dy * 0.5;
        half_y = box.dx * 0.5;
    }
    const double cosine = cos(yaw);
    const double sine = sin(yaw);
    const DevicePoint local[4] = {
        {-half_x, -half_y}, {half_x, -half_y},
        {half_x, half_y}, {-half_x, half_y}};
    for (int index = 0; index < 4; ++index) {
        output[index] = {
            box.x + local[index].x * cosine - local[index].y * sine,
            box.y + local[index].x * sine + local[index].y * cosine};
    }
}

__device__ double polygon_area(const DevicePoint* points, int count) {
    double area = 0.0;
    for (int index = 0; index < count; ++index) {
        const DevicePoint next = points[(index + 1) % count];
        area += points[index].x * next.y - next.x * points[index].y;
    }
    return fabs(area) * 0.5;
}

__device__ double rotated_iou(const Detection& first,
                              const Detection& second,
                              bool pcdet_convention) {
    const double broad_limit =
        (first.dx + first.dy + second.dx + second.dy) * 0.5;
    if (fabs(static_cast<double>(first.x) - second.x) > broad_limit ||
        fabs(static_cast<double>(first.y) - second.y) > broad_limit) {
        return 0.0;
    }

    DevicePoint first_corners[4];
    DevicePoint second_corners[4];
    box_corners(first, pcdet_convention, first_corners);
    box_corners(second, pcdet_convention, second_corners);
    DevicePoint polygon[16];
    DevicePoint next_polygon[16];
    int polygon_count = 4;
    for (int index = 0; index < 4; ++index) {
        polygon[index] = first_corners[index];
    }

    for (int edge = 0; edge < 4 && polygon_count > 0; ++edge) {
        const DevicePoint clip_start = second_corners[edge];
        const DevicePoint clip_end = second_corners[(edge + 1) % 4];
        int next_count = 0;
        for (int index = 0; index < polygon_count; ++index) {
            const DevicePoint current = polygon[index];
            const DevicePoint previous =
                polygon[(index + polygon_count - 1) % polygon_count];
            const bool current_inside =
                cross(clip_start, clip_end, current) >= -1.0e-9;
            const bool previous_inside =
                cross(clip_start, clip_end, previous) >= -1.0e-9;
            if (current_inside != previous_inside) {
                next_polygon[next_count++] = intersection(
                    previous, current, clip_start, clip_end);
            }
            if (current_inside) {
                next_polygon[next_count++] = current;
            }
        }
        polygon_count = next_count;
        for (int index = 0; index < polygon_count; ++index) {
            polygon[index] = next_polygon[index];
        }
    }
    const double intersection_area =
        polygon_count == 0 ? 0.0 : polygon_area(polygon, polygon_count);
    const double union_area =
        static_cast<double>(first.dx) * first.dy +
        static_cast<double>(second.dx) * second.dy - intersection_area;
    return union_area > 0.0 ? intersection_area / union_area : 0.0;
}

__global__ void rotated_nms_mask_kernel(const Detection* boxes,
                                        std::uint64_t* masks,
                                        int pre_max_size,
                                        float iou_threshold,
                                        bool pcdet_convention) {
    const int row_block = blockIdx.y;
    const int column_block = blockIdx.x;
    if (row_block > column_block) {
        return;
    }
    __shared__ Detection column_boxes[kNmsThreads];
    const int column_index = column_block * kNmsThreads + threadIdx.x;
    if (column_index < pre_max_size) {
        column_boxes[threadIdx.x] = boxes[column_index];
    } else {
        column_boxes[threadIdx.x].score = -1.0F;
    }
    __syncthreads();

    const int row_index = row_block * kNmsThreads + threadIdx.x;
    if (row_index >= pre_max_size) {
        return;
    }
    const Detection row_box = boxes[row_index];
    std::uint64_t bits = 0;
    if (row_box.score >= 0.0F) {
        const int start =
            row_block == column_block ? threadIdx.x + 1 : 0;
        for (int index = start; index < kNmsThreads; ++index) {
            if (column_boxes[index].score >= 0.0F &&
                rotated_iou(row_box, column_boxes[index],
                            pcdet_convention) > iou_threshold) {
                bits |= std::uint64_t{1} << index;
            }
        }
    }
    masks[static_cast<std::size_t>(row_index) * kMaskBlocks +
          column_block] = bits;
}

__global__ void select_nms_kernel(const Detection* boxes,
                                  const std::uint64_t* masks,
                                  std::uint64_t* removed,
                                  Detection* output,
                                  int* output_count,
                                  int pre_max_size,
                                  int post_max_size) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    int count = 0;
    for (int index = 0; index < pre_max_size; ++index) {
        if (boxes[index].score < 0.0F) {
            break;
        }
        const int block = index / kNmsThreads;
        const int bit = index % kNmsThreads;
        if ((removed[block] & (std::uint64_t{1} << bit)) != 0) {
            continue;
        }
        output[count++] = boxes[index];
        if (count >= post_max_size) {
            break;
        }
        const std::uint64_t* row =
            masks + static_cast<std::size_t>(index) * kMaskBlocks;
        for (int mask_block = block; mask_block < kMaskBlocks;
             ++mask_block) {
            removed[mask_block] |= row[mask_block];
        }
    }
    *output_count = count;
}

}  // namespace

class GpuPostprocessPipeline::Impl {
public:
    explicit Impl(const GpuPostprocessConfig& config)
        : config_(config),
          device_config_(make_device_config(config)),
          scores_input_(kSpatial),
          scores_output_(kSpatial),
          indices_input_(kSpatial),
          indices_output_(kSpatial),
          candidate_count_(1),
          pre_nms_boxes_(kMaximumPreNms),
          nms_masks_(static_cast<std::size_t>(kMaximumPreNms) * kMaskBlocks),
          removed_masks_(kMaskBlocks),
          detections_(kMaximumPostNms),
          detection_count_(1) {
        if (config_.pre_max_size <= 0 ||
            config_.pre_max_size > kMaximumPreNms ||
            config_.post_max_size <= 0 ||
            config_.post_max_size > kMaximumPostNms) {
            throw std::invalid_argument(
                "postprocess max sizes exceed CUDA pipeline limits");
        }
        std::size_t temporary_bytes = 0;
        check_cuda(cub::DeviceRadixSort::SortPairsDescending(
                       nullptr, temporary_bytes, scores_input_.data(),
                       scores_output_.data(), indices_input_.data(),
                       indices_output_.data(), kSpatial),
                   "query postprocess radix-sort storage");
        sort_temporary_ = DeviceBuffer<unsigned char>(temporary_bytes);
        sort_temporary_bytes_ = temporary_bytes;
    }

    GpuPostprocessStats run(const DeviceHeadMaps& maps) {
        if (maps.reg == nullptr || maps.height == nullptr ||
            maps.dim == nullptr || maps.rot == nullptr ||
            maps.heatmap == nullptr || maps.height_size != kHeight ||
            maps.width_size != kWidth) {
            throw std::invalid_argument(
                "GPU postprocess expects five device head maps at 468x468");
        }
        check_cuda(cudaMemset(candidate_count_.data(), 0, sizeof(int)),
                   "reset candidate count");
        check_cuda(cudaMemset(detection_count_.data(), 0, sizeof(int)),
                   "reset detection count");
        check_cuda(cudaMemset(removed_masks_.data(), 0,
                              kMaskBlocks * sizeof(std::uint64_t)),
                   "reset NMS removed mask");

        CudaEvent start;
        CudaEvent stop;
        start.record();
        generate_candidate_scores_kernel<<<block_count(kSpatial), kThreads>>>(
            maps, device_config_, scores_input_.data(), indices_input_.data(),
            candidate_count_.data());
        check_cuda(cudaGetLastError(), "launch candidate score kernel");

        check_cuda(cub::DeviceRadixSort::SortPairsDescending(
                       sort_temporary_.data(), sort_temporary_bytes_,
                       scores_input_.data(), scores_output_.data(),
                       indices_input_.data(), indices_output_.data(), kSpatial),
                   "sort postprocess candidates");

        prepare_pre_nms_kernel<<<block_count(config_.pre_max_size), kThreads>>>(
            maps, device_config_, scores_output_.data(), indices_output_.data(),
            pre_nms_boxes_.data());
        check_cuda(cudaGetLastError(), "launch pre-NMS decode kernel");

        const dim3 nms_grid(kMaskBlocks, kMaskBlocks);
        rotated_nms_mask_kernel<<<nms_grid, kNmsThreads>>>(
            pre_nms_boxes_.data(), nms_masks_.data(), config_.pre_max_size,
            config_.nms_iou_threshold, config_.use_pcdet_nms_convention);
        check_cuda(cudaGetLastError(), "launch rotated NMS mask kernel");
        select_nms_kernel<<<1, 1>>>(
            pre_nms_boxes_.data(), nms_masks_.data(), removed_masks_.data(),
            detections_.data(), detection_count_.data(), config_.pre_max_size,
            config_.post_max_size);
        check_cuda(cudaGetLastError(), "launch rotated NMS selection kernel");

        stop.record();
        stop.synchronize();
        check_cuda(cudaEventElapsedTime(&stats_.elapsed_ms,
                                        start.get(), stop.get()),
                   "measure postprocess elapsed time");
        check_cuda(cudaMemcpy(&stats_.candidates_before_nms,
                              candidate_count_.data(), sizeof(int),
                              cudaMemcpyDeviceToHost),
                   "copy candidate count");
        check_cuda(cudaMemcpy(&stats_.detections_after_nms,
                              detection_count_.data(), sizeof(int),
                              cudaMemcpyDeviceToHost),
                   "copy detection count");
        stats_.candidates_after_pre_max =
            std::min(stats_.candidates_before_nms, config_.pre_max_size);
        return stats_;
    }

    DeviceDetectionView device_detections() const {
        return {detections_.data(), stats_.detections_after_nms};
    }

    std::vector<Detection> copy_detections_to_host() const {
        std::vector<Detection> result(stats_.detections_after_nms);
        check_cuda(cudaMemcpy(result.data(), detections_.data(),
                              result.size() * sizeof(Detection),
                              cudaMemcpyDeviceToHost),
                   "copy final GPU detections");
        return result;
    }

    std::vector<Detection> copy_pre_nms_to_host() const {
        std::vector<Detection> result(stats_.candidates_after_pre_max);
        check_cuda(cudaMemcpy(result.data(), pre_nms_boxes_.data(),
                              result.size() * sizeof(Detection),
                              cudaMemcpyDeviceToHost),
                   "copy validation pre-NMS candidates");
        return result;
    }

    const GpuPostprocessConfig& config() const { return config_; }

private:
    GpuPostprocessConfig config_;
    DevicePostprocessConfig device_config_;
    DeviceBuffer<float> scores_input_;
    DeviceBuffer<float> scores_output_;
    DeviceBuffer<int> indices_input_;
    DeviceBuffer<int> indices_output_;
    DeviceBuffer<int> candidate_count_;
    DeviceBuffer<unsigned char> sort_temporary_;
    std::size_t sort_temporary_bytes_ = 0;
    DeviceBuffer<Detection> pre_nms_boxes_;
    DeviceBuffer<std::uint64_t> nms_masks_;
    DeviceBuffer<std::uint64_t> removed_masks_;
    DeviceBuffer<Detection> detections_;
    DeviceBuffer<int> detection_count_;
    GpuPostprocessStats stats_;
};

GpuPostprocessPipeline::GpuPostprocessPipeline(
    const GpuPostprocessConfig& config)
    : impl_(std::make_unique<Impl>(config)) {}

GpuPostprocessPipeline::~GpuPostprocessPipeline() = default;
GpuPostprocessPipeline::GpuPostprocessPipeline(
    GpuPostprocessPipeline&&) noexcept = default;
GpuPostprocessPipeline& GpuPostprocessPipeline::operator=(
    GpuPostprocessPipeline&&) noexcept = default;

GpuPostprocessStats GpuPostprocessPipeline::run(const DeviceHeadMaps& maps) {
    return impl_->run(maps);
}

DeviceDetectionView GpuPostprocessPipeline::device_detections() const {
    return impl_->device_detections();
}

std::vector<Detection> GpuPostprocessPipeline::copy_detections_to_host() const {
    return impl_->copy_detections_to_host();
}

std::vector<Detection> GpuPostprocessPipeline::copy_pre_nms_to_host() const {
    return impl_->copy_pre_nms_to_host();
}

const GpuPostprocessConfig& GpuPostprocessPipeline::config() const {
    return impl_->config();
}

}  // namespace centerpoint
