#include "centerpoint/gpu_rpn.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

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
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)),
                   "cudaMalloc");
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
    DeviceBuffer<float> destination(source.size());
    check_cuda(cudaMemcpy(destination.data(), source.data(),
                          source.size() * sizeof(float), cudaMemcpyHostToDevice),
               operation);
    return destination;
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

struct DeviceBatchNorm {
    DeviceBuffer<float> weight;
    DeviceBuffer<float> bias;
    DeviceBuffer<float> mean;
    DeviceBuffer<float> variance;
};

struct DeviceConvLayer {
    std::string name;
    DeviceBuffer<float> weight;
    DeviceBatchNorm batch_norm;
    int in_channels = 0;
    int out_channels = 0;
    int kernel_size = 0;
    int stride = 1;
    int padding = 0;
};

struct DeviceDeconvLayer {
    std::string name;
    DeviceBuffer<float> gemm_weight;
    DeviceBatchNorm batch_norm;
    int in_channels = 0;
    int out_channels = 0;
    int kernel_size = 0;
    int stride = 0;
};

DeviceBatchNorm upload_batch_norm(const RpnBatchNormWeights& weights) {
    DeviceBatchNorm result;
    result.weight = upload(weights.weight, "upload RPN BN weight");
    result.bias = upload(weights.bias, "upload RPN BN bias");
    result.mean = upload(weights.mean, "upload RPN BN mean");
    result.variance = upload(weights.variance, "upload RPN BN variance");
    return result;
}

DeviceConvLayer upload_conv(const RpnConvWeights& weights) {
    DeviceConvLayer result;
    result.name = weights.name;
    result.weight = upload(weights.weight, "upload RPN Conv weight");
    result.batch_norm = upload_batch_norm(weights.batch_norm);
    result.in_channels = weights.in_channels;
    result.out_channels = weights.out_channels;
    result.kernel_size = weights.kernel_size;
    result.stride = weights.stride;
    result.padding = weights.padding;
    return result;
}

DeviceDeconvLayer upload_deconv(const RpnDeconvWeights& weights) {
    DeviceDeconvLayer result;
    result.name = weights.name;
    result.gemm_weight =
        upload(weights.gemm_weight, "upload RPN transposed Conv weight");
    result.batch_norm = upload_batch_norm(weights.batch_norm);
    result.in_channels = weights.in_channels;
    result.out_channels = weights.out_channels;
    result.kernel_size = weights.kernel_size;
    result.stride = weights.stride;
    return result;
}

__global__ void im2col_nchw_kernel(const float* input,
                                   float* columns,
                                   int in_channels,
                                   int input_height,
                                   int input_width,
                                   int output_height,
                                   int output_width,
                                   int kernel_size,
                                   int stride,
                                   int padding,
                                   std::size_t total) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= total) {
        return;
    }

    const int spatial = output_height * output_width;
    const int output_index = static_cast<int>(index % spatial);
    int kernel_index = static_cast<int>(index / spatial);
    const int kernel_x = kernel_index % kernel_size;
    kernel_index /= kernel_size;
    const int kernel_y = kernel_index % kernel_size;
    const int input_channel = kernel_index / kernel_size;
    const int output_y = output_index / output_width;
    const int output_x = output_index % output_width;
    const int input_y = output_y * stride + kernel_y - padding;
    const int input_x = output_x * stride + kernel_x - padding;

    float value = 0.0F;
    if (input_y >= 0 && input_y < input_height &&
        input_x >= 0 && input_x < input_width) {
        value = input[(static_cast<std::size_t>(input_channel) * input_height +
                       input_y) * input_width + input_x];
    }
    columns[index] = value;
}

__global__ void batch_norm_relu_kernel(float* values,
                                       const float* weight,
                                       const float* bias,
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
    const float normalized =
        (values[index] - mean[channel]) * rsqrtf(variance[channel] + epsilon);
    values[index] =
        fmaxf(normalized * weight[channel] + bias[channel], 0.0F);
}

__global__ void deconv_columns_to_nchw_kernel(const float* columns,
                                              float* output,
                                              int input_height,
                                              int input_width,
                                              int kernel_size,
                                              int output_height,
                                              int output_width,
                                              std::size_t total) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= total) {
        return;
    }
    const int output_spatial = output_height * output_width;
    const int channel = static_cast<int>(index / output_spatial);
    const int output_index = static_cast<int>(index % output_spatial);
    const int output_y = output_index / output_width;
    const int output_x = output_index % output_width;
    const int kernel_y = output_y % kernel_size;
    const int kernel_x = output_x % kernel_size;
    const int input_y = output_y / kernel_size;
    const int input_x = output_x / kernel_size;
    const int input_spatial = input_height * input_width;
    const int column_row =
        (channel * kernel_size + kernel_y) * kernel_size + kernel_x;
    const int column_column = input_y * input_width + input_x;
    output[index] =
        columns[static_cast<std::size_t>(column_row) * input_spatial +
                column_column];
}

__global__ void concat_three_nchw_kernel(const float* first,
                                         const float* second,
                                         const float* third,
                                         float* output,
                                         int channels_per_input,
                                         int spatial,
                                         std::size_t total) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= total) {
        return;
    }
    const int channel = static_cast<int>(index / spatial);
    const int offset = static_cast<int>(index % spatial);
    if (channel < channels_per_input) {
        output[index] = first[static_cast<std::size_t>(channel) * spatial + offset];
    } else if (channel < channels_per_input * 2) {
        output[index] = second[
            static_cast<std::size_t>(channel - channels_per_input) * spatial +
            offset];
    } else {
        output[index] = third[
            static_cast<std::size_t>(channel - channels_per_input * 2) * spatial +
            offset];
    }
}

__global__ void gather_conv_patch_kernel(const float* input,
                                         float* patch,
                                         int channels,
                                         int input_height,
                                         int input_width,
                                         int kernel_size,
                                         int stride,
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
    const int input_y = output_y * stride + kernel_y - padding;
    const int input_x = output_x * stride + kernel_x - padding;
    float value = 0.0F;
    if (input_y >= 0 && input_y < input_height &&
        input_x >= 0 && input_x < input_width) {
        value = input[(static_cast<std::size_t>(channel) * input_height +
                       input_y) * input_width + input_x];
    }
    patch[index] = value;
}

__global__ void gather_deconv_input_kernel(const float* input,
                                           float* values,
                                           int channels,
                                           int input_height,
                                           int input_width,
                                           int input_y,
                                           int input_x) {
    const int channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= channels) {
        return;
    }
    values[channel] = input[
        (static_cast<std::size_t>(channel) * input_height + input_y) *
            input_width + input_x];
}

void launch_batch_norm_relu(DeviceTensor& tensor,
                            const DeviceBatchNorm& batch_norm,
                            float epsilon) {
    const int spatial = tensor.height * tensor.width;
    batch_norm_relu_kernel<<<block_count(tensor.count()), kThreads>>>(
        tensor.values.data(), batch_norm.weight.data(), batch_norm.bias.data(),
        batch_norm.mean.data(), batch_norm.variance.data(), spatial, epsilon,
        tensor.count());
    check_cuda(cudaGetLastError(), "launch RPN BatchNorm-ReLU kernel");
}

DeviceTensor conv_bn_relu(CublasHandle& handle,
                          const TensorView& input,
                          const DeviceConvLayer& layer,
                          float epsilon) {
    if (input.channels != layer.in_channels) {
        throw std::runtime_error(layer.name + " input channel mismatch");
    }
    const int output_height =
        (input.height + 2 * layer.padding - layer.kernel_size) /
            layer.stride + 1;
    const int output_width =
        (input.width + 2 * layer.padding - layer.kernel_size) /
            layer.stride + 1;
    const int spatial = output_height * output_width;
    const int reduction =
        layer.in_channels * layer.kernel_size * layer.kernel_size;
    const std::size_t column_count =
        static_cast<std::size_t>(reduction) * spatial;
    DeviceBuffer<float> columns(column_count);
    im2col_nchw_kernel<<<block_count(column_count), kThreads>>>(
        input.values, columns.data(), layer.in_channels, input.height,
        input.width, output_height, output_width, layer.kernel_size,
        layer.stride, layer.padding, column_count);
    check_cuda(cudaGetLastError(), "launch RPN im2col kernel");

    DeviceTensor output;
    output.channels = layer.out_channels;
    output.height = output_height;
    output.width = output_width;
    output.values = DeviceBuffer<float>(output.count());

    const float alpha = 1.0F;
    const float beta = 0.0F;
    check_cublas(
        cublasSgemm(handle.get(), CUBLAS_OP_N, CUBLAS_OP_N,
                    spatial, layer.out_channels, reduction,
                    &alpha, columns.data(), spatial,
                    layer.weight.data(), reduction,
                    &beta, output.values.data(), spatial),
        layer.name.c_str());
    launch_batch_norm_relu(output, layer.batch_norm, epsilon);
    return output;
}

DeviceTensor deconv_bn_relu(CublasHandle& handle,
                            const TensorView& input,
                            const DeviceDeconvLayer& layer,
                            float epsilon) {
    if (input.channels != layer.in_channels ||
        layer.stride != layer.kernel_size) {
        throw std::runtime_error(layer.name + " configuration mismatch");
    }
    const int input_spatial = input.height * input.width;
    const int expanded_channels =
        layer.out_channels * layer.kernel_size * layer.kernel_size;
    const std::size_t column_count =
        static_cast<std::size_t>(expanded_channels) * input_spatial;
    DeviceBuffer<float> columns(column_count);

    const float alpha = 1.0F;
    const float beta = 0.0F;
    check_cublas(
        cublasSgemm(handle.get(), CUBLAS_OP_N, CUBLAS_OP_N,
                    input_spatial, expanded_channels, layer.in_channels,
                    &alpha, input.values, input_spatial,
                    layer.gemm_weight.data(), layer.in_channels,
                    &beta, columns.data(), input_spatial),
        layer.name.c_str());

    DeviceTensor output;
    output.channels = layer.out_channels;
    output.height = input.height * layer.stride;
    output.width = input.width * layer.stride;
    output.values = DeviceBuffer<float>(output.count());
    deconv_columns_to_nchw_kernel<<<block_count(output.count()), kThreads>>>(
        columns.data(), output.values.data(), input.height, input.width,
        layer.kernel_size, output.height, output.width, output.count());
    check_cuda(cudaGetLastError(), "launch RPN deconvolution rearrange kernel");
    launch_batch_norm_relu(output, layer.batch_norm, epsilon);
    return output;
}

DeviceTensor concatenate(const DeviceTensor& first,
                         const DeviceTensor& second,
                         const DeviceTensor& third) {
    if (first.channels != second.channels || first.channels != third.channels ||
        first.height != second.height || first.height != third.height ||
        first.width != second.width || first.width != third.width) {
        throw std::runtime_error("RPN deblock shapes cannot be concatenated");
    }
    DeviceTensor output;
    output.channels = first.channels * 3;
    output.height = first.height;
    output.width = first.width;
    output.values = DeviceBuffer<float>(output.count());
    concat_three_nchw_kernel<<<block_count(output.count()), kThreads>>>(
        first.values.data(), second.values.data(), third.values.data(),
        output.values.data(), first.channels, first.height * first.width,
        output.count());
    check_cuda(cudaGetLastError(), "launch RPN concat kernel");
    return output;
}

float copy_scalar(const DeviceTensor& tensor, int channel, int y, int x) {
    const std::size_t index =
        (static_cast<std::size_t>(channel) * tensor.height + y) *
        tensor.width + x;
    float value = 0.0F;
    check_cuda(cudaMemcpy(&value, tensor.values.data() + index, sizeof(float),
                          cudaMemcpyDeviceToHost),
               "copy RPN probe output");
    return value;
}

void append_conv_probe(const TensorView& input,
                       const DeviceTensor& output,
                       const DeviceConvLayer& layer,
                       int sample,
                       std::vector<RpnLayerProbe>& probes) {
    RpnLayerProbe probe;
    probe.name = layer.name;
    probe.operation = "conv2d";
    probe.input_shape = {input.channels, input.height, input.width};
    probe.output_shape = {output.channels, output.height, output.width};
    probe.kernel_size = layer.kernel_size;
    probe.stride = layer.stride;
    probe.padding = layer.padding;
    if (sample == 0) {
        probe.output_index = {0, output.height / 2, output.width / 2};
    } else {
        probe.output_index = {std::min(7, output.channels - 1), 0, 0};
    }

    const std::size_t value_count =
        static_cast<std::size_t>(input.channels) * layer.kernel_size *
        layer.kernel_size;
    DeviceBuffer<float> device_patch(value_count);
    gather_conv_patch_kernel<<<block_count(value_count), kThreads>>>(
        input.values, device_patch.data(), input.channels, input.height,
        input.width, layer.kernel_size, layer.stride, layer.padding,
        probe.output_index[1], probe.output_index[2], value_count);
    check_cuda(cudaGetLastError(), "launch RPN conv probe gather");
    probe.input_values.resize(value_count);
    check_cuda(cudaMemcpy(probe.input_values.data(), device_patch.data(),
                          value_count * sizeof(float), cudaMemcpyDeviceToHost),
               "copy RPN conv probe input");
    probe.output_value = copy_scalar(
        output, probe.output_index[0], probe.output_index[1],
        probe.output_index[2]);
    probes.push_back(std::move(probe));
}

void append_deconv_probe(const TensorView& input,
                         const DeviceTensor& output,
                         const DeviceDeconvLayer& layer,
                         int sample,
                         std::vector<RpnLayerProbe>& probes) {
    RpnLayerProbe probe;
    probe.name = layer.name;
    probe.operation = "conv_transpose2d";
    probe.input_shape = {input.channels, input.height, input.width};
    probe.output_shape = {output.channels, output.height, output.width};
    probe.kernel_size = layer.kernel_size;
    probe.stride = layer.stride;
    if (sample == 0) {
        probe.output_index = {0, output.height / 2, output.width / 2};
    } else {
        probe.output_index = {std::min(7, output.channels - 1), 0, 0};
    }

    const int input_y = probe.output_index[1] / layer.stride;
    const int input_x = probe.output_index[2] / layer.stride;
    DeviceBuffer<float> device_values(input.channels);
    gather_deconv_input_kernel<<<block_count(input.channels), kThreads>>>(
        input.values, device_values.data(), input.channels, input.height,
        input.width, input_y, input_x);
    check_cuda(cudaGetLastError(), "launch RPN deconv probe gather");
    probe.input_values.resize(input.channels);
    check_cuda(cudaMemcpy(probe.input_values.data(), device_values.data(),
                          static_cast<std::size_t>(input.channels) * sizeof(float),
                          cudaMemcpyDeviceToHost),
               "copy RPN deconv probe input");
    probe.output_value = copy_scalar(
        output, probe.output_index[0], probe.output_index[1],
        probe.output_index[2]);
    probes.push_back(std::move(probe));
}

std::array<int, 3> shape_of(const DeviceTensor& tensor) {
    return {tensor.channels, tensor.height, tensor.width};
}

}  // namespace

class GpuRpnPipeline::Impl {
public:
    explicit Impl(const RpnWeights& weights)
        : batch_norm_epsilon_(weights.batch_norm_epsilon) {
        for (int block = 0; block < 3; ++block) {
            device_blocks_[block].reserve(weights.blocks[block].size());
            for (const RpnConvWeights& layer : weights.blocks[block]) {
                device_blocks_[block].push_back(upload_conv(layer));
            }
        }
        deblock0_ = upload_conv(weights.deblock0);
        deblock1_ = upload_deconv(weights.deblock1);
        deblock2_ = upload_deconv(weights.deblock2);
    }

    GpuRpnStats run(const DeviceBevView& input, bool collect_probes) {
        if (input.data == nullptr || input.channels != 64 ||
            input.height != 468 || input.width != 468) {
            throw std::invalid_argument(
                "GPU RPN expects device input [1,64,468,468]");
        }
        probes_.clear();
        CudaEvent start;
        CudaEvent stop;
        start.record();

        TensorView current{input.data, input.channels, input.height, input.width};
        DeviceTensor current_owner;
        for (const DeviceConvLayer& layer : device_blocks_[0]) {
            DeviceTensor next =
                conv_bn_relu(handle_, current, layer, batch_norm_epsilon_);
            if (collect_probes) {
                append_conv_probe(current, next, layer, 0, probes_);
                append_conv_probe(current, next, layer, 1, probes_);
            }
            current_owner = std::move(next);
            current = current_owner.view();
        }
        stats_.block_shapes[0] = shape_of(current_owner);
        DeviceTensor up0 =
            conv_bn_relu(handle_, current, deblock0_, batch_norm_epsilon_);
        if (collect_probes) {
            append_conv_probe(current, up0, deblock0_, 0, probes_);
            append_conv_probe(current, up0, deblock0_, 1, probes_);
        }
        stats_.deblock_shapes[0] = shape_of(up0);

        for (const DeviceConvLayer& layer : device_blocks_[1]) {
            DeviceTensor next =
                conv_bn_relu(handle_, current, layer, batch_norm_epsilon_);
            if (collect_probes) {
                append_conv_probe(current, next, layer, 0, probes_);
                append_conv_probe(current, next, layer, 1, probes_);
            }
            current_owner = std::move(next);
            current = current_owner.view();
        }
        stats_.block_shapes[1] = shape_of(current_owner);
        DeviceTensor up1 =
            deconv_bn_relu(handle_, current, deblock1_, batch_norm_epsilon_);
        if (collect_probes) {
            append_deconv_probe(current, up1, deblock1_, 0, probes_);
            append_deconv_probe(current, up1, deblock1_, 1, probes_);
        }
        stats_.deblock_shapes[1] = shape_of(up1);

        for (const DeviceConvLayer& layer : device_blocks_[2]) {
            DeviceTensor next =
                conv_bn_relu(handle_, current, layer, batch_norm_epsilon_);
            if (collect_probes) {
                append_conv_probe(current, next, layer, 0, probes_);
                append_conv_probe(current, next, layer, 1, probes_);
            }
            current_owner = std::move(next);
            current = current_owner.view();
        }
        stats_.block_shapes[2] = shape_of(current_owner);
        DeviceTensor up2 =
            deconv_bn_relu(handle_, current, deblock2_, batch_norm_epsilon_);
        if (collect_probes) {
            append_deconv_probe(current, up2, deblock2_, 0, probes_);
            append_deconv_probe(current, up2, deblock2_, 1, probes_);
        }
        stats_.deblock_shapes[2] = shape_of(up2);

        output_ = concatenate(up0, up1, up2);
        stop.record();
        stop.synchronize();
        check_cuda(cudaEventElapsedTime(&stats_.elapsed_ms, start.get(), stop.get()),
                   "measure RPN elapsed time");
        stats_.probe_count = static_cast<int>(probes_.size());
        return stats_;
    }

    DeviceRpnView device_output() const {
        const TensorView view = output_.view();
        return {view.values, view.channels, view.height, view.width};
    }

    const std::vector<RpnLayerProbe>& probes() const { return probes_; }

private:
    CublasHandle handle_;
    std::array<std::vector<DeviceConvLayer>, 3> device_blocks_;
    DeviceConvLayer deblock0_;
    DeviceDeconvLayer deblock1_;
    DeviceDeconvLayer deblock2_;
    float batch_norm_epsilon_ = 1.0e-3F;
    DeviceTensor output_;
    GpuRpnStats stats_;
    std::vector<RpnLayerProbe> probes_;
};

GpuRpnPipeline::GpuRpnPipeline(const RpnWeights& weights)
    : impl_(std::make_unique<Impl>(weights)) {}

GpuRpnPipeline::~GpuRpnPipeline() = default;
GpuRpnPipeline::GpuRpnPipeline(GpuRpnPipeline&&) noexcept = default;
GpuRpnPipeline& GpuRpnPipeline::operator=(GpuRpnPipeline&&) noexcept = default;

GpuRpnStats GpuRpnPipeline::run(const DeviceBevView& input,
                                bool collect_probes) {
    return impl_->run(input, collect_probes);
}

DeviceRpnView GpuRpnPipeline::device_output() const {
    return impl_->device_output();
}

const std::vector<RpnLayerProbe>& GpuRpnPipeline::probes() const {
    return impl_->probes();
}

}  // namespace centerpoint
