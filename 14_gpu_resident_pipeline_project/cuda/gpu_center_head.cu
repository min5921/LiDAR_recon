#include "centerpoint/gpu_center_head.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace centerpoint {
namespace {

constexpr int kThreads = 256;

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::ostringstream message;
        message << operation << " failed: " << cudaGetErrorString(status);
        throw std::runtime_error(message.str());
    }
}

void check_cublas(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::ostringstream message;
        message << operation << " failed with cuBLAS status "
                << static_cast<int>(status);
        throw std::runtime_error(message.str());
    }
}

int block_count(std::size_t count) {
    return static_cast<int>((count + kThreads - 1) / kThreads);
}

class CublasHandle {
public:
    CublasHandle() { check_cublas(cublasCreate(&handle_), "cublasCreate"); }
    ~CublasHandle() {
        if (handle_ != nullptr) {
            cublasDestroy(handle_);
        }
    }
    CublasHandle(const CublasHandle&) = delete;
    CublasHandle& operator=(const CublasHandle&) = delete;
    cublasHandle_t get() const { return handle_; }

private:
    cublasHandle_t handle_ = nullptr;
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
    std::size_t count() const { return count_; }

private:
    void allocate(std::size_t count) {
        if (count == 0) {
            return;
        }
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&data_),
                              count * sizeof(T)),
                   "cudaMalloc CenterHead buffer");
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

DeviceBuffer<float> upload(const std::vector<float>& source,
                           const char* operation) {
    DeviceBuffer<float> result(source.size());
    check_cuda(cudaMemcpy(result.data(), source.data(),
                          source.size() * sizeof(float),
                          cudaMemcpyHostToDevice),
               operation);
    return result;
}

struct TensorView {
    const float* values = nullptr;
    int channels = 0;
    int height = 0;
    int width = 0;
};

struct DeviceTensor {
    DeviceBuffer<float> values;
    int channels = 0;
    int height = 0;
    int width = 0;

    std::size_t count() const {
        return static_cast<std::size_t>(channels) * height * width;
    }
    TensorView view() const {
        return {values.data(), channels, height, width};
    }
};

DeviceTensor allocate_tensor(int channels, int height, int width) {
    DeviceTensor result;
    result.channels = channels;
    result.height = height;
    result.width = width;
    result.values = DeviceBuffer<float>(result.count());
    return result;
}

struct DeviceBatchNorm {
    DeviceBuffer<float> weight;
    DeviceBuffer<float> bias;
    DeviceBuffer<float> mean;
    DeviceBuffer<float> variance;
};

struct DeviceConvLayer {
    std::string name;
    DeviceBuffer<float> weight;
    DeviceBuffer<float> bias;
    DeviceBatchNorm batch_norm;
    int in_channels = 0;
    int out_channels = 0;
    int kernel_size = 3;
    int padding = 1;
    bool has_batch_norm = false;
};

DeviceBatchNorm upload_batch_norm(const HeadBatchNormWeights& weights) {
    DeviceBatchNorm result;
    result.weight = upload(weights.weight, "upload CenterHead BN weight");
    result.bias = upload(weights.bias, "upload CenterHead BN bias");
    result.mean = upload(weights.mean, "upload CenterHead BN mean");
    result.variance =
        upload(weights.variance, "upload CenterHead BN variance");
    return result;
}

DeviceConvLayer upload_conv(const HeadConvWeights& weights) {
    DeviceConvLayer result;
    result.name = weights.name;
    result.weight = upload(weights.weight, "upload CenterHead Conv weight");
    result.bias = upload(weights.bias, "upload CenterHead Conv bias");
    if (weights.has_batch_norm) {
        result.batch_norm = upload_batch_norm(weights.batch_norm);
    }
    result.in_channels = weights.in_channels;
    result.out_channels = weights.out_channels;
    result.kernel_size = weights.kernel_size;
    result.padding = weights.padding;
    result.has_batch_norm = weights.has_batch_norm;
    return result;
}

__global__ void im2col_same_kernel(const float* input,
                                   float* columns,
                                   int channels,
                                   int height,
                                   int width,
                                   int kernel_size,
                                   int padding,
                                   std::size_t total) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= total) {
        return;
    }
    const int spatial = height * width;
    const int output_offset = static_cast<int>(index % spatial);
    int position = static_cast<int>(index / spatial);
    const int kernel_x = position % kernel_size;
    position /= kernel_size;
    const int kernel_y = position % kernel_size;
    const int channel = position / kernel_size;
    const int output_y = output_offset / width;
    const int output_x = output_offset % width;
    const int input_y = output_y + kernel_y - padding;
    const int input_x = output_x + kernel_x - padding;
    float value = 0.0F;
    if (input_y >= 0 && input_y < height &&
        input_x >= 0 && input_x < width) {
        value = input[(static_cast<std::size_t>(channel) * height + input_y) *
                      width + input_x];
    }
    columns[index] = value;
}

__global__ void bias_bn_relu_kernel(float* values,
                                    const float* conv_bias,
                                    const float* bn_weight,
                                    const float* bn_bias,
                                    const float* mean,
                                    const float* variance,
                                    int spatial,
                                    float epsilon,
                                    std::size_t total) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= total) {
        return;
    }
    const int channel = static_cast<int>(index / spatial);
    const float biased = values[index] + conv_bias[channel];
    const float normalized =
        (biased - mean[channel]) * rsqrtf(variance[channel] + epsilon);
    values[index] =
        fmaxf(normalized * bn_weight[channel] + bn_bias[channel], 0.0F);
}

__global__ void add_bias_kernel(float* values,
                                const float* bias,
                                int spatial,
                                std::size_t total) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < total) {
        values[index] += bias[index / spatial];
    }
}

__global__ void gather_patch_kernel(const float* input,
                                    float* patch,
                                    int channels,
                                    int height,
                                    int width,
                                    int kernel_size,
                                    int padding,
                                    int output_y,
                                    int output_x,
                                    std::size_t total) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= total) {
        return;
    }
    int position = static_cast<int>(index);
    const int kernel_x = position % kernel_size;
    position /= kernel_size;
    const int kernel_y = position % kernel_size;
    const int channel = position / kernel_size;
    const int input_y = output_y + kernel_y - padding;
    const int input_x = output_x + kernel_x - padding;
    float value = 0.0F;
    if (input_y >= 0 && input_y < height &&
        input_x >= 0 && input_x < width) {
        value = input[(static_cast<std::size_t>(channel) * height + input_y) *
                      width + input_x];
    }
    patch[index] = value;
}

void conv_same(CublasHandle& handle,
               const TensorView& input,
               const DeviceConvLayer& layer,
               float epsilon,
               float* column_workspace,
               std::size_t workspace_count,
               DeviceTensor& output) {
    if (input.channels != layer.in_channels) {
        throw std::runtime_error(layer.name + " input channel mismatch");
    }
    const int spatial = input.height * input.width;
    const int reduction =
        input.channels * layer.kernel_size * layer.kernel_size;
    const std::size_t column_count =
        static_cast<std::size_t>(reduction) * spatial;
    if (column_count > workspace_count ||
        output.channels != layer.out_channels ||
        output.height != input.height || output.width != input.width) {
        throw std::runtime_error(layer.name + " workspace/output mismatch");
    }
    im2col_same_kernel<<<block_count(column_count), kThreads>>>(
        input.values, column_workspace, input.channels, input.height, input.width,
        layer.kernel_size, layer.padding, column_count);
    check_cuda(cudaGetLastError(), "launch CenterHead im2col kernel");

    const float alpha = 1.0F;
    const float beta = 0.0F;
    check_cublas(
        cublasSgemm(handle.get(), CUBLAS_OP_N, CUBLAS_OP_N,
                    spatial, output.channels, reduction,
                    &alpha, column_workspace, spatial,
                    layer.weight.data(), reduction,
                    &beta, output.values.data(), spatial),
        layer.name.c_str());

    if (layer.has_batch_norm) {
        bias_bn_relu_kernel<<<block_count(output.count()), kThreads>>>(
            output.values.data(), layer.bias.data(),
            layer.batch_norm.weight.data(), layer.batch_norm.bias.data(),
            layer.batch_norm.mean.data(), layer.batch_norm.variance.data(),
            spatial, epsilon, output.count());
    } else {
        add_bias_kernel<<<block_count(output.count()), kThreads>>>(
            output.values.data(), layer.bias.data(), spatial, output.count());
    }
    check_cuda(cudaGetLastError(), "launch CenterHead post-Conv kernel");
}

float copy_scalar(const DeviceTensor& tensor, int channel, int y, int x) {
    const std::size_t index =
        (static_cast<std::size_t>(channel) * tensor.height + y) *
        tensor.width + x;
    float value = 0.0F;
    check_cuda(cudaMemcpy(&value, tensor.values.data() + index, sizeof(float),
                          cudaMemcpyDeviceToHost),
               "copy CenterHead probe output");
    return value;
}

void append_probe(const TensorView& input,
                  const DeviceTensor& output,
                  const DeviceConvLayer& layer,
                  int sample,
                  std::vector<HeadLayerProbe>& probes) {
    HeadLayerProbe probe;
    probe.name = layer.name;
    probe.input_shape = {input.channels, input.height, input.width};
    probe.output_shape = {output.channels, output.height, output.width};
    probe.has_batch_norm = layer.has_batch_norm;
    if (sample == 0) {
        probe.output_index = {0, output.height / 2, output.width / 2};
    } else {
        probe.output_index = {std::min(7, output.channels - 1), 0, 0};
    }

    const std::size_t value_count =
        static_cast<std::size_t>(input.channels) * layer.kernel_size *
        layer.kernel_size;
    DeviceBuffer<float> device_patch(value_count);
    gather_patch_kernel<<<block_count(value_count), kThreads>>>(
        input.values, device_patch.data(), input.channels, input.height,
        input.width, layer.kernel_size, layer.padding, probe.output_index[1],
        probe.output_index[2], value_count);
    check_cuda(cudaGetLastError(), "launch CenterHead probe gather");
    probe.input_values.resize(value_count);
    check_cuda(cudaMemcpy(probe.input_values.data(), device_patch.data(),
                          value_count * sizeof(float), cudaMemcpyDeviceToHost),
               "copy CenterHead probe input");
    probe.output_value = copy_scalar(
        output, probe.output_index[0], probe.output_index[1],
        probe.output_index[2]);
    probes.push_back(std::move(probe));
}

}  // namespace

class GpuCenterHeadPipeline::Impl {
public:
    explicit Impl(const HeadWeights& weights)
        : shared_(upload_conv(weights.shared)),
          batch_norm_epsilon_(weights.batch_norm_epsilon) {
        for (int index = 0; index < 5; ++index) {
            branches_[index].name = weights.branches[index].name;
            branches_[index].hidden =
                upload_conv(weights.branches[index].hidden);
            branches_[index].output =
                upload_conv(weights.branches[index].output);
        }
        constexpr int kSpatial = 468 * 468;
        im2col_workspace_ = DeviceBuffer<float>(
            static_cast<std::size_t>(384) * 3 * 3 * kSpatial);
        shared_output_ = allocate_tensor(64, 468, 468);
        hidden_output_ = allocate_tensor(64, 468, 468);
        const std::array<int, 5> output_channels = {2, 1, 3, 2, 3};
        for (int index = 0; index < 5; ++index) {
            outputs_[index] =
                allocate_tensor(output_channels[index], 468, 468);
        }
    }

    GpuCenterHeadStats run(const DeviceRpnView& input, bool collect_probes) {
        if (input.data == nullptr || input.channels != 384 ||
            input.height != 468 || input.width != 468) {
            throw std::invalid_argument(
                "GPU CenterHead expects device input [1,384,468,468]");
        }
        probes_.clear();
        CudaEvent start;
        CudaEvent stop;
        start.record();

        const TensorView input_view{
            input.data, input.channels, input.height, input.width};
        conv_same(handle_, input_view, shared_, batch_norm_epsilon_,
                  im2col_workspace_.data(), im2col_workspace_.count(),
                  shared_output_);
        if (collect_probes) {
            append_probe(input_view, shared_output_, shared_, 0, probes_);
            append_probe(input_view, shared_output_, shared_, 1, probes_);
        }

        for (int index = 0; index < 5; ++index) {
            const TensorView shared_view = shared_output_.view();
            conv_same(handle_, shared_view, branches_[index].hidden,
                      batch_norm_epsilon_, im2col_workspace_.data(),
                      im2col_workspace_.count(), hidden_output_);
            if (collect_probes) {
                append_probe(shared_view, hidden_output_,
                             branches_[index].hidden, 0, probes_);
                append_probe(shared_view, hidden_output_,
                             branches_[index].hidden, 1, probes_);
            }
            const TensorView hidden_view = hidden_output_.view();
            conv_same(handle_, hidden_view, branches_[index].output,
                      batch_norm_epsilon_, im2col_workspace_.data(),
                      im2col_workspace_.count(), outputs_[index]);
            if (collect_probes) {
                append_probe(hidden_view, outputs_[index],
                             branches_[index].output, 0, probes_);
                append_probe(hidden_view, outputs_[index],
                             branches_[index].output, 1, probes_);
            }
        }

        stop.record();
        stop.synchronize();
        check_cuda(cudaEventElapsedTime(&stats_.elapsed_ms,
                                        start.get(), stop.get()),
                   "measure CenterHead elapsed time");
        stats_.probe_count = static_cast<int>(probes_.size());
        return stats_;
    }

    DeviceHeadMaps device_maps() const {
        return {outputs_[0].values.data(), outputs_[1].values.data(),
                outputs_[2].values.data(), outputs_[3].values.data(),
                outputs_[4].values.data(), outputs_[0].height,
                outputs_[0].width};
    }

    HostHeadMaps copy_maps_to_host() const {
        HostHeadMaps result;
        result.height_size = outputs_[0].height;
        result.width_size = outputs_[0].width;
        for (int index = 0; index < 5; ++index) {
            result.values[index].resize(outputs_[index].count());
            check_cuda(cudaMemcpy(result.values[index].data(),
                                  outputs_[index].values.data(),
                                  outputs_[index].count() * sizeof(float),
                                  cudaMemcpyDeviceToHost),
                       "copy CenterHead validation map");
        }
        return result;
    }

    const std::vector<HeadLayerProbe>& probes() const { return probes_; }

private:
    struct DeviceBranch {
        std::string name;
        DeviceConvLayer hidden;
        DeviceConvLayer output;
    };

    CublasHandle handle_;
    DeviceConvLayer shared_;
    std::array<DeviceBranch, 5> branches_;
    DeviceBuffer<float> im2col_workspace_;
    DeviceTensor shared_output_;
    DeviceTensor hidden_output_;
    std::array<DeviceTensor, 5> outputs_;
    float batch_norm_epsilon_ = 1.0e-3F;
    GpuCenterHeadStats stats_;
    std::vector<HeadLayerProbe> probes_;
};

GpuCenterHeadPipeline::GpuCenterHeadPipeline(const HeadWeights& weights)
    : impl_(std::make_unique<Impl>(weights)) {}

GpuCenterHeadPipeline::~GpuCenterHeadPipeline() = default;
GpuCenterHeadPipeline::GpuCenterHeadPipeline(
    GpuCenterHeadPipeline&&) noexcept = default;
GpuCenterHeadPipeline& GpuCenterHeadPipeline::operator=(
    GpuCenterHeadPipeline&&) noexcept = default;

GpuCenterHeadStats GpuCenterHeadPipeline::run(
    const DeviceRpnView& input, bool collect_probes) {
    return impl_->run(input, collect_probes);
}

DeviceHeadMaps GpuCenterHeadPipeline::device_maps() const {
    return impl_->device_maps();
}

HostHeadMaps GpuCenterHeadPipeline::copy_maps_to_host() const {
    return impl_->copy_maps_to_host();
}

const std::vector<HeadLayerProbe>& GpuCenterHeadPipeline::probes() const {
    return impl_->probes();
}

}  // namespace centerpoint
