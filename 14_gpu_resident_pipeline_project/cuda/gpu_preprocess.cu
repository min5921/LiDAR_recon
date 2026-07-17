#include "centerpoint/gpu_preprocess.hpp"

#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace centerpoint {
namespace {

constexpr int kPfnInputChannels = 10;
constexpr int kPfnLocalChannels = 32;
constexpr int kPfnOutputChannels = 64;
constexpr int kRequiredPointFeatures = 5;
constexpr int kRequiredMaxPoints = 20;
constexpr int kThreads = 256;

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::ostringstream message;
        message << operation << " failed: " << cudaGetErrorString(status);
        throw std::runtime_error(message.str());
    }
}

int block_count(std::size_t count, int threads = kThreads) {
    return static_cast<int>((count + threads - 1) / threads);
}

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t count) { reserve(count); }
    ~DeviceBuffer() { reset(); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : data_(other.data_), capacity_(other.capacity_) {
        other.data_ = nullptr;
        other.capacity_ = 0;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            reset();
            data_ = other.data_;
            capacity_ = other.capacity_;
            other.data_ = nullptr;
            other.capacity_ = 0;
        }
        return *this;
    }

    void reserve(std::size_t count) {
        if (count <= capacity_) {
            return;
        }
        reset();
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)),
                   "cudaMalloc");
        capacity_ = count;
    }

    T* data() { return data_; }
    const T* data() const { return data_; }
    std::size_t capacity() const { return capacity_; }

private:
    void reset() {
        if (data_ != nullptr) {
            cudaFree(data_);
            data_ = nullptr;
        }
        capacity_ = 0;
    }

    T* data_ = nullptr;
    std::size_t capacity_ = 0;
};

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

float elapsed_ms(const CudaEvent& start, const CudaEvent& stop) {
    float elapsed = 0.0F;
    check_cuda(cudaEventElapsedTime(&elapsed, start.get(), stop.get()),
               "cudaEventElapsedTime");
    return elapsed;
}

template <typename T>
void copy_vector_to_device(DeviceBuffer<T>& destination,
                           const std::vector<T>& source,
                           const char* operation) {
    destination.reserve(source.size());
    check_cuda(cudaMemcpy(destination.data(), source.data(),
                          source.size() * sizeof(T), cudaMemcpyHostToDevice),
               operation);
}

void validate_weights(const PfnWeights& weights) {
    if (weights.layer0.in_channels != kPfnInputChannels ||
        weights.layer0.out_channels != kPfnLocalChannels ||
        weights.layer1.in_channels != kPfnLocalChannels * 2 ||
        weights.layer1.out_channels != kPfnOutputChannels) {
        throw std::invalid_argument(
            "GPU PFN expects layer shapes [32,10] and [64,64]");
    }
}

__global__ void compute_point_keys_kernel(
    const float* points,
    std::uint32_t* keys,
    std::uint32_t* point_indices,
    int* valid_count,
    int point_count,
    int feature_dimension,
    float voxel_x,
    float voxel_y,
    float voxel_z,
    float min_x,
    float min_y,
    float min_z,
    int grid_x,
    int grid_y,
    int grid_z) {
    const int point_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (point_index >= point_count) {
        return;
    }

    const float* point = points +
        static_cast<std::size_t>(point_index) * feature_dimension;
    const int x = __float2int_rd((point[0] - min_x) / voxel_x);
    const int y = __float2int_rd((point[1] - min_y) / voxel_y);
    const int z = __float2int_rd((point[2] - min_z) / voxel_z);
    point_indices[point_index] = static_cast<std::uint32_t>(point_index);
    if (x < 0 || x >= grid_x || y < 0 || y >= grid_y ||
        z < 0 || z >= grid_z) {
        keys[point_index] = UINT32_MAX;
        return;
    }

    keys[point_index] = static_cast<std::uint32_t>(
        (z * grid_y + y) * grid_x + x);
    atomicAdd(valid_count, 1);
}

__global__ void mark_group_starts_kernel(const std::uint32_t* sorted_keys,
                                         int* group_start_flags,
                                         int valid_count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= valid_count) {
        return;
    }
    group_start_flags[index] =
        index > 0 && sorted_keys[index] != sorted_keys[index - 1] ? 1 : 0;
}

__global__ void collect_group_metadata_kernel(
    const std::uint32_t* sorted_keys,
    const std::uint32_t* sorted_point_indices,
    const int* group_start_flags,
    const int* group_ids,
    std::uint32_t* group_first_point,
    int* group_identity,
    std::uint32_t* group_key,
    int* group_start,
    int valid_count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= valid_count ||
        (index != 0 && group_start_flags[index] == 0)) {
        return;
    }
    const int group = group_ids[index];
    group_first_point[group] = sorted_point_indices[index];
    group_identity[group] = group;
    group_key[group] = sorted_keys[index];
    group_start[group] = index;
}

__global__ void setup_selected_pillars_kernel(
    const int* ordered_groups,
    const std::uint32_t* group_keys,
    const int* group_starts,
    int* group_to_pillar,
    int* coordinates,
    int* num_points,
    int selected_pillars,
    int unique_pillars,
    int valid_points,
    int grid_x,
    int grid_y,
    int max_points) {
    const int pillar = blockIdx.x * blockDim.x + threadIdx.x;
    if (pillar >= selected_pillars) {
        return;
    }

    const int group = ordered_groups[pillar];
    group_to_pillar[group] = pillar;
    const int start = group_starts[group];
    const int end = group + 1 < unique_pillars
        ? group_starts[group + 1]
        : valid_points;
    num_points[pillar] = min(end - start, max_points);

    std::uint32_t key = group_keys[group];
    const int x = static_cast<int>(key % static_cast<std::uint32_t>(grid_x));
    key /= static_cast<std::uint32_t>(grid_x);
    const int y = static_cast<int>(key % static_cast<std::uint32_t>(grid_y));
    const int z = static_cast<int>(key / static_cast<std::uint32_t>(grid_y));
    const std::size_t offset = static_cast<std::size_t>(pillar) * 4;
    coordinates[offset + 0] = 0;
    coordinates[offset + 1] = z;
    coordinates[offset + 2] = y;
    coordinates[offset + 3] = x;
}

__global__ void fill_pillars_kernel(
    const float* points,
    const std::uint32_t* sorted_point_indices,
    const int* group_ids,
    const int* group_starts,
    const int* group_to_pillar,
    float* pillars,
    int valid_points,
    int feature_dimension,
    int max_points) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= valid_points) {
        return;
    }
    const int group = group_ids[index];
    const int pillar = group_to_pillar[group];
    const int point_in_pillar = index - group_starts[group];
    if (pillar < 0 || point_in_pillar >= max_points) {
        return;
    }

    const std::uint32_t source_point = sorted_point_indices[index];
    const std::size_t source =
        static_cast<std::size_t>(source_point) * feature_dimension;
    const std::size_t destination =
        (static_cast<std::size_t>(pillar) * max_points + point_in_pillar) *
        feature_dimension;
    for (int feature = 0; feature < feature_dimension; ++feature) {
        pillars[destination + feature] = points[source + feature];
    }
}

__device__ float batch_norm_relu(float value,
                                 int channel,
                                 const float* weight,
                                 const float* bias,
                                 const float* mean,
                                 const float* variance,
                                 float epsilon) {
    const float normalized =
        (value - mean[channel]) / sqrtf(variance[channel] + epsilon);
    return fmaxf(normalized * weight[channel] + bias[channel], 0.0F);
}

__global__ void decorate_and_pfn_kernel(
    const float* pillars,
    const int* coordinates,
    const int* num_points,
    float* pillar_features,
    int pillar_count,
    float voxel_x,
    float voxel_y,
    float min_x,
    float min_y,
    const float* layer0_linear,
    const float* layer0_bn_weight,
    const float* layer0_bn_bias,
    const float* layer0_bn_mean,
    const float* layer0_bn_variance,
    const float* layer1_linear,
    const float* layer1_bn_weight,
    const float* layer1_bn_bias,
    const float* layer1_bn_mean,
    const float* layer1_bn_variance,
    float epsilon) {
    const int pillar = blockIdx.x;
    if (pillar >= pillar_count) {
        return;
    }

    __shared__ float mean_xyz[3];
    __shared__ float local_features[kRequiredMaxPoints * kPfnLocalChannels];
    __shared__ float local_max[kPfnLocalChannels];

    const int count = num_points[pillar];
    if (threadIdx.x == 0) {
        float mean_x = 0.0F;
        float mean_y = 0.0F;
        float mean_z = 0.0F;
        for (int point = 0; point < count; ++point) {
            const std::size_t offset =
                (static_cast<std::size_t>(pillar) * kRequiredMaxPoints + point) *
                kRequiredPointFeatures;
            mean_x += pillars[offset + 0];
            mean_y += pillars[offset + 1];
            mean_z += pillars[offset + 2];
        }
        const float divisor = static_cast<float>(count);
        mean_xyz[0] = count > 0 ? mean_x / divisor : 0.0F;
        mean_xyz[1] = count > 0 ? mean_y / divisor : 0.0F;
        mean_xyz[2] = count > 0 ? mean_z / divisor : 0.0F;
    }
    __syncthreads();

    const std::size_t coordinate_offset =
        static_cast<std::size_t>(pillar) * 4;
    const int y_coordinate = coordinates[coordinate_offset + 2];
    const int x_coordinate = coordinates[coordinate_offset + 3];
    const float center_x =
        static_cast<float>(x_coordinate) * voxel_x + voxel_x * 0.5F + min_x;
    const float center_y =
        static_cast<float>(y_coordinate) * voxel_y + voxel_y * 0.5F + min_y;

    const int local_value_count = kRequiredMaxPoints * kPfnLocalChannels;
    for (int index = threadIdx.x; index < local_value_count;
         index += blockDim.x) {
        const int point = index / kPfnLocalChannels;
        const int output_channel = index % kPfnLocalChannels;
        float decorated[kPfnInputChannels] = {};
        if (point < count) {
            const std::size_t input_offset =
                (static_cast<std::size_t>(pillar) * kRequiredMaxPoints + point) *
                kRequiredPointFeatures;
            for (int feature = 0; feature < kRequiredPointFeatures; ++feature) {
                decorated[feature] = pillars[input_offset + feature];
            }
            decorated[5] = decorated[0] - mean_xyz[0];
            decorated[6] = decorated[1] - mean_xyz[1];
            decorated[7] = decorated[2] - mean_xyz[2];
            decorated[8] = decorated[0] - center_x;
            decorated[9] = decorated[1] - center_y;
        }

        float linear = 0.0F;
        const std::size_t weight_offset =
            static_cast<std::size_t>(output_channel) * kPfnInputChannels;
        for (int input_channel = 0; input_channel < kPfnInputChannels;
             ++input_channel) {
            linear += decorated[input_channel] *
                      layer0_linear[weight_offset + input_channel];
        }
        local_features[index] = batch_norm_relu(
            linear, output_channel, layer0_bn_weight, layer0_bn_bias,
            layer0_bn_mean, layer0_bn_variance, epsilon);
    }
    __syncthreads();

    if (threadIdx.x < kPfnLocalChannels) {
        float maximum = -FLT_MAX;
        for (int point = 0; point < kRequiredMaxPoints; ++point) {
            maximum = fmaxf(
                maximum,
                local_features[point * kPfnLocalChannels + threadIdx.x]);
        }
        local_max[threadIdx.x] = maximum;
    }
    __syncthreads();

    if (threadIdx.x < kPfnOutputChannels) {
        const int output_channel = threadIdx.x;
        const std::size_t weight_offset =
            static_cast<std::size_t>(output_channel) *
            (kPfnLocalChannels * 2);
        float pooled = -FLT_MAX;
        for (int point = 0; point < kRequiredMaxPoints; ++point) {
            float linear = 0.0F;
            const int local_offset = point * kPfnLocalChannels;
            for (int input_channel = 0; input_channel < kPfnLocalChannels;
                 ++input_channel) {
                linear += local_features[local_offset + input_channel] *
                          layer1_linear[weight_offset + input_channel];
            }
            for (int input_channel = 0; input_channel < kPfnLocalChannels;
                 ++input_channel) {
                linear += local_max[input_channel] *
                          layer1_linear[weight_offset + kPfnLocalChannels +
                                        input_channel];
            }
            const float activated = batch_norm_relu(
                linear, output_channel, layer1_bn_weight, layer1_bn_bias,
                layer1_bn_mean, layer1_bn_variance, epsilon);
            pooled = fmaxf(pooled, activated);
        }
        pillar_features[
            static_cast<std::size_t>(pillar) * kPfnOutputChannels +
            output_channel] = pooled;
    }
}

__global__ void scatter_bev_kernel(const float* pillar_features,
                                   const int* coordinates,
                                   float* bev,
                                   int pillar_count,
                                   int grid_x,
                                   int grid_y) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int value_count = pillar_count * kPfnOutputChannels;
    if (index >= value_count) {
        return;
    }
    const int pillar = index / kPfnOutputChannels;
    const int channel = index % kPfnOutputChannels;
    const std::size_t coordinate_offset =
        static_cast<std::size_t>(pillar) * 4;
    const int y = coordinates[coordinate_offset + 2];
    const int x = coordinates[coordinate_offset + 3];
    const std::size_t destination =
        (static_cast<std::size_t>(channel) * grid_y + y) * grid_x + x;
    bev[destination] = pillar_features[index];
}

}  // namespace

class GpuPreprocessPipeline::Impl {
public:
    Impl(const GpuPreprocessConfig& config, const PfnWeights& weights)
        : config_(config),
          grid_x_(static_cast<int>(std::lround(
              (config.point_cloud_range[3] - config.point_cloud_range[0]) /
              config.voxel_size[0]))),
          grid_y_(static_cast<int>(std::lround(
              (config.point_cloud_range[4] - config.point_cloud_range[1]) /
              config.voxel_size[1]))),
          grid_z_(static_cast<int>(std::lround(
              (config.point_cloud_range[5] - config.point_cloud_range[2]) /
              config.voxel_size[2]))),
          batch_norm_epsilon_(weights.batch_norm_epsilon) {
        validate_weights(weights);
        if (config.feature_dimension != kRequiredPointFeatures ||
            config.max_points_per_pillar != kRequiredMaxPoints ||
            grid_x_ <= 0 || grid_y_ <= 0 || grid_z_ <= 0) {
            throw std::invalid_argument(
                "GPU front-end expects 5 point features and 20 points per pillar");
        }

        const std::size_t max_pillars =
            static_cast<std::size_t>(config_.max_pillars);
        d_pillars_.reserve(max_pillars * kRequiredMaxPoints *
                           kRequiredPointFeatures);
        d_coordinates_.reserve(max_pillars * 4);
        d_num_points_.reserve(max_pillars);
        d_pillar_features_.reserve(max_pillars * kPfnOutputChannels);
        d_bev_.reserve(static_cast<std::size_t>(kPfnOutputChannels) *
                       grid_y_ * grid_x_);
        d_valid_count_.reserve(1);

        copy_vector_to_device(d_layer0_linear_, weights.layer0.linear,
                              "copy PFN layer0 linear");
        copy_vector_to_device(d_layer0_bn_weight_, weights.layer0.bn_weight,
                              "copy PFN layer0 BN weight");
        copy_vector_to_device(d_layer0_bn_bias_, weights.layer0.bn_bias,
                              "copy PFN layer0 BN bias");
        copy_vector_to_device(d_layer0_bn_mean_, weights.layer0.bn_mean,
                              "copy PFN layer0 BN mean");
        copy_vector_to_device(d_layer0_bn_variance_,
                              weights.layer0.bn_variance,
                              "copy PFN layer0 BN variance");
        copy_vector_to_device(d_layer1_linear_, weights.layer1.linear,
                              "copy PFN layer1 linear");
        copy_vector_to_device(d_layer1_bn_weight_, weights.layer1.bn_weight,
                              "copy PFN layer1 BN weight");
        copy_vector_to_device(d_layer1_bn_bias_, weights.layer1.bn_bias,
                              "copy PFN layer1 BN bias");
        copy_vector_to_device(d_layer1_bn_mean_, weights.layer1.bn_mean,
                              "copy PFN layer1 BN mean");
        copy_vector_to_device(d_layer1_bn_variance_,
                              weights.layer1.bn_variance,
                              "copy PFN layer1 BN variance");
    }

    GpuPreprocessStats run(const float* host_points, int point_count) {
        if (point_count < 0 || (point_count > 0 && host_points == nullptr)) {
            throw std::invalid_argument("invalid host point buffer");
        }
        ensure_point_capacity(point_count);

        CudaEvent total_start;
        CudaEvent upload_done;
        CudaEvent voxel_done;
        CudaEvent pfn_done;
        CudaEvent scatter_done;
        total_start.record();

        if (point_count > 0) {
            check_cuda(cudaMemcpyAsync(
                           d_points_.data(), host_points,
                           static_cast<std::size_t>(point_count) *
                               config_.feature_dimension * sizeof(float),
                           cudaMemcpyHostToDevice),
                       "copy points to GPU");
        }
        upload_done.record();

        check_cuda(cudaMemset(d_valid_count_.data(), 0, sizeof(int)),
                   "clear valid point count");
        if (point_count > 0) {
            compute_point_keys_kernel<<<block_count(point_count), kThreads>>>(
                d_points_.data(), d_point_keys_in_.data(),
                d_point_indices_in_.data(), d_valid_count_.data(), point_count,
                config_.feature_dimension, config_.voxel_size[0],
                config_.voxel_size[1], config_.voxel_size[2],
                config_.point_cloud_range[0], config_.point_cloud_range[1],
                config_.point_cloud_range[2], grid_x_, grid_y_, grid_z_);
            check_cuda(cudaGetLastError(), "launch point key kernel");
            sort_points(point_count);
        }

        int valid_points = 0;
        check_cuda(cudaMemcpy(&valid_points, d_valid_count_.data(), sizeof(int),
                              cudaMemcpyDeviceToHost),
                   "copy valid point count");

        int unique_pillars = 0;
        int selected_pillars = 0;
        check_cuda(cudaMemset(
                       d_pillars_.data(), 0,
                       static_cast<std::size_t>(config_.max_pillars) *
                           kRequiredMaxPoints * kRequiredPointFeatures *
                           sizeof(float)),
                   "clear pillar buffer");

        if (valid_points > 0) {
            mark_group_starts_kernel<<<block_count(valid_points), kThreads>>>(
                d_point_keys_sorted_.data(), d_group_start_flags_.data(),
                valid_points);
            check_cuda(cudaGetLastError(), "launch group start kernel");
            scan_group_ids(valid_points);

            int last_group_id = 0;
            check_cuda(cudaMemcpy(
                           &last_group_id,
                           d_point_group_ids_.data() + valid_points - 1,
                           sizeof(int), cudaMemcpyDeviceToHost),
                       "copy last group id");
            unique_pillars = last_group_id + 1;

            collect_group_metadata_kernel<<<block_count(valid_points), kThreads>>>(
                d_point_keys_sorted_.data(), d_point_indices_sorted_.data(),
                d_group_start_flags_.data(), d_point_group_ids_.data(),
                d_group_first_point_.data(), d_group_identity_.data(),
                d_group_key_.data(), d_group_start_.data(), valid_points);
            check_cuda(cudaGetLastError(), "launch group metadata kernel");
            sort_groups(unique_pillars);

            check_cuda(cudaMemset(d_group_to_pillar_.data(), 0xFF,
                                  static_cast<std::size_t>(unique_pillars) *
                                      sizeof(int)),
                       "clear group-to-pillar map");
            selected_pillars =
                std::min(unique_pillars, config_.max_pillars);
            setup_selected_pillars_kernel<<<block_count(selected_pillars),
                                             kThreads>>>(
                d_group_order_.data(), d_group_key_.data(),
                d_group_start_.data(), d_group_to_pillar_.data(),
                d_coordinates_.data(), d_num_points_.data(), selected_pillars,
                unique_pillars, valid_points, grid_x_, grid_y_,
                config_.max_points_per_pillar);
            check_cuda(cudaGetLastError(), "launch selected pillar setup kernel");
            fill_pillars_kernel<<<block_count(valid_points), kThreads>>>(
                d_points_.data(), d_point_indices_sorted_.data(),
                d_point_group_ids_.data(), d_group_start_.data(),
                d_group_to_pillar_.data(), d_pillars_.data(), valid_points,
                config_.feature_dimension, config_.max_points_per_pillar);
            check_cuda(cudaGetLastError(), "launch pillar fill kernel");
        }
        voxel_done.record();

        if (selected_pillars > 0) {
            decorate_and_pfn_kernel<<<selected_pillars, kPfnOutputChannels>>>(
                d_pillars_.data(), d_coordinates_.data(), d_num_points_.data(),
                d_pillar_features_.data(), selected_pillars,
                config_.voxel_size[0], config_.voxel_size[1],
                config_.point_cloud_range[0], config_.point_cloud_range[1],
                d_layer0_linear_.data(), d_layer0_bn_weight_.data(),
                d_layer0_bn_bias_.data(), d_layer0_bn_mean_.data(),
                d_layer0_bn_variance_.data(), d_layer1_linear_.data(),
                d_layer1_bn_weight_.data(), d_layer1_bn_bias_.data(),
                d_layer1_bn_mean_.data(), d_layer1_bn_variance_.data(),
                batch_norm_epsilon_);
            check_cuda(cudaGetLastError(), "launch decoration and PFN kernel");
        }
        pfn_done.record();

        const std::size_t bev_count =
            static_cast<std::size_t>(kPfnOutputChannels) * grid_y_ * grid_x_;
        check_cuda(cudaMemset(d_bev_.data(), 0, bev_count * sizeof(float)),
                   "clear BEV tensor");
        if (selected_pillars > 0) {
            const int scatter_values = selected_pillars * kPfnOutputChannels;
            scatter_bev_kernel<<<block_count(scatter_values), kThreads>>>(
                d_pillar_features_.data(), d_coordinates_.data(), d_bev_.data(),
                selected_pillars, grid_x_, grid_y_);
            check_cuda(cudaGetLastError(), "launch BEV scatter kernel");
        }
        scatter_done.record();
        scatter_done.synchronize();

        stats_.input_points = point_count;
        stats_.valid_points = valid_points;
        stats_.unique_pillars = unique_pillars;
        stats_.selected_pillars = selected_pillars;
        stats_.host_to_device_ms = elapsed_ms(total_start, upload_done);
        stats_.voxelization_ms = elapsed_ms(upload_done, voxel_done);
        stats_.pfn_ms = elapsed_ms(voxel_done, pfn_done);
        stats_.scatter_ms = elapsed_ms(pfn_done, scatter_done);
        stats_.total_ms = elapsed_ms(total_start, scatter_done);
        return stats_;
    }

    DeviceBevView device_bev() const {
        return {d_bev_.data(), kPfnOutputChannels, grid_y_, grid_x_};
    }

    std::vector<float> copy_bev_to_host() const {
        const std::size_t count =
            static_cast<std::size_t>(kPfnOutputChannels) * grid_y_ * grid_x_;
        std::vector<float> host(count);
        check_cuda(cudaMemcpy(host.data(), d_bev_.data(), count * sizeof(float),
                              cudaMemcpyDeviceToHost),
                   "copy final BEV to host");
        return host;
    }

private:
    void ensure_point_capacity(int point_count) {
        const std::size_t count = static_cast<std::size_t>(point_count);
        d_points_.reserve(count * config_.feature_dimension);
        d_point_keys_in_.reserve(count);
        d_point_keys_sorted_.reserve(count);
        d_point_indices_in_.reserve(count);
        d_point_indices_sorted_.reserve(count);
        d_group_start_flags_.reserve(count);
        d_point_group_ids_.reserve(count);
        d_group_first_point_.reserve(count);
        d_group_first_point_sorted_.reserve(count);
        d_group_identity_.reserve(count);
        d_group_order_.reserve(count);
        d_group_key_.reserve(count);
        d_group_start_.reserve(count);
        d_group_to_pillar_.reserve(count);
    }

    void sort_points(int point_count) {
        std::size_t temporary_bytes = 0;
        check_cuda(cub::DeviceRadixSort::SortPairs(
                       nullptr, temporary_bytes, d_point_keys_in_.data(),
                       d_point_keys_sorted_.data(), d_point_indices_in_.data(),
                       d_point_indices_sorted_.data(), point_count),
                   "query point radix sort storage");
        d_temporary_.reserve(temporary_bytes);
        check_cuda(cub::DeviceRadixSort::SortPairs(
                       d_temporary_.data(), temporary_bytes,
                       d_point_keys_in_.data(), d_point_keys_sorted_.data(),
                       d_point_indices_in_.data(),
                       d_point_indices_sorted_.data(), point_count),
                   "sort points by pillar key");
    }

    void scan_group_ids(int valid_points) {
        std::size_t temporary_bytes = 0;
        check_cuda(cub::DeviceScan::InclusiveSum(
                       nullptr, temporary_bytes, d_group_start_flags_.data(),
                       d_point_group_ids_.data(), valid_points),
                   "query group scan storage");
        d_temporary_.reserve(temporary_bytes);
        check_cuda(cub::DeviceScan::InclusiveSum(
                       d_temporary_.data(), temporary_bytes,
                       d_group_start_flags_.data(),
                       d_point_group_ids_.data(), valid_points),
                   "scan point group ids");
    }

    void sort_groups(int unique_pillars) {
        std::size_t temporary_bytes = 0;
        check_cuda(cub::DeviceRadixSort::SortPairs(
                       nullptr, temporary_bytes, d_group_first_point_.data(),
                       d_group_first_point_sorted_.data(),
                       d_group_identity_.data(), d_group_order_.data(),
                       unique_pillars),
                   "query group radix sort storage");
        d_temporary_.reserve(temporary_bytes);
        check_cuda(cub::DeviceRadixSort::SortPairs(
                       d_temporary_.data(), temporary_bytes,
                       d_group_first_point_.data(),
                       d_group_first_point_sorted_.data(),
                       d_group_identity_.data(), d_group_order_.data(),
                       unique_pillars),
                   "sort pillars by first point");
    }

    GpuPreprocessConfig config_;
    int grid_x_ = 0;
    int grid_y_ = 0;
    int grid_z_ = 0;
    float batch_norm_epsilon_ = 1.0e-3F;
    GpuPreprocessStats stats_;

    DeviceBuffer<float> d_points_;
    DeviceBuffer<std::uint32_t> d_point_keys_in_;
    DeviceBuffer<std::uint32_t> d_point_keys_sorted_;
    DeviceBuffer<std::uint32_t> d_point_indices_in_;
    DeviceBuffer<std::uint32_t> d_point_indices_sorted_;
    DeviceBuffer<int> d_valid_count_;
    DeviceBuffer<int> d_group_start_flags_;
    DeviceBuffer<int> d_point_group_ids_;
    DeviceBuffer<std::uint32_t> d_group_first_point_;
    DeviceBuffer<std::uint32_t> d_group_first_point_sorted_;
    DeviceBuffer<int> d_group_identity_;
    DeviceBuffer<int> d_group_order_;
    DeviceBuffer<std::uint32_t> d_group_key_;
    DeviceBuffer<int> d_group_start_;
    DeviceBuffer<int> d_group_to_pillar_;
    DeviceBuffer<float> d_pillars_;
    DeviceBuffer<int> d_coordinates_;
    DeviceBuffer<int> d_num_points_;
    DeviceBuffer<float> d_pillar_features_;
    DeviceBuffer<float> d_bev_;
    DeviceBuffer<unsigned char> d_temporary_;

    DeviceBuffer<float> d_layer0_linear_;
    DeviceBuffer<float> d_layer0_bn_weight_;
    DeviceBuffer<float> d_layer0_bn_bias_;
    DeviceBuffer<float> d_layer0_bn_mean_;
    DeviceBuffer<float> d_layer0_bn_variance_;
    DeviceBuffer<float> d_layer1_linear_;
    DeviceBuffer<float> d_layer1_bn_weight_;
    DeviceBuffer<float> d_layer1_bn_bias_;
    DeviceBuffer<float> d_layer1_bn_mean_;
    DeviceBuffer<float> d_layer1_bn_variance_;
};

GpuPreprocessPipeline::GpuPreprocessPipeline(
    const GpuPreprocessConfig& config, const PfnWeights& weights)
    : impl_(std::make_unique<Impl>(config, weights)) {}

GpuPreprocessPipeline::~GpuPreprocessPipeline() = default;
GpuPreprocessPipeline::GpuPreprocessPipeline(GpuPreprocessPipeline&&) noexcept =
    default;
GpuPreprocessPipeline& GpuPreprocessPipeline::operator=(
    GpuPreprocessPipeline&&) noexcept = default;

GpuPreprocessStats GpuPreprocessPipeline::run(const float* host_points,
                                              int point_count) {
    return impl_->run(host_points, point_count);
}

DeviceBevView GpuPreprocessPipeline::device_bev() const {
    return impl_->device_bev();
}

std::vector<float> GpuPreprocessPipeline::copy_bev_to_host() const {
    return impl_->copy_bev_to_host();
}

}  // namespace centerpoint
