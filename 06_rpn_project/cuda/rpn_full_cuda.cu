#include "centerpoint/rpn_full_cuda.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace centerpoint {
namespace {

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

class CublasHandle {
public:
    CublasHandle() {
        check_cublas(cublasCreate(&handle_), "cublasCreate");
    }
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

class DeviceArray {
public:
    DeviceArray() = default;
    explicit DeviceArray(std::size_t count) : count_(count) {
        check_cuda(cudaMalloc(&data_, count_ * sizeof(float)), "cudaMalloc");
    }
    ~DeviceArray() { reset(); }

    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;

    DeviceArray(DeviceArray&& other) noexcept {
        data_ = other.data_;
        count_ = other.count_;
        other.data_ = nullptr;
        other.count_ = 0;
    }

    DeviceArray& operator=(DeviceArray&& other) noexcept {
        if (this != &other) {
            reset();
            data_ = other.data_;
            count_ = other.count_;
            other.data_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    float* data() { return data_; }
    const float* data() const { return data_; }

private:
    void reset() {
        if (data_ != nullptr) {
            cudaFree(data_);
            data_ = nullptr;
        }
        count_ = 0;
    }

    float* data_ = nullptr;
    std::size_t count_ = 0;
};

struct DeviceTensor {
    DeviceArray values;
    int channels = 0;
    int height = 0;
    int width = 0;

    std::size_t count() const {
        return static_cast<std::size_t>(channels) * height * width;
    }
};

DeviceArray copy_to_device(const std::vector<float>& host,
                           const char* operation) {
    DeviceArray device(host.size());
    check_cuda(cudaMemcpy(device.data(), host.data(), host.size() * sizeof(float),
                          cudaMemcpyHostToDevice), operation);
    return device;
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
    const int in_channel = kernel_index / kernel_size;
    const int output_y = output_index / output_width;
    const int output_x = output_index % output_width;
    const int input_y = output_y * stride + kernel_y - padding;
    const int input_x = output_x * stride + kernel_x - padding;

    float value = 0.0F;
    if (input_y >= 0 && input_y < input_height &&
        input_x >= 0 && input_x < input_width) {
        const std::size_t input_offset =
            (static_cast<std::size_t>(in_channel) * input_height + input_y) *
                input_width + input_x;
        value = input[input_offset];
    }
    columns[index] = value;
}

__global__ void batch_norm_relu_kernel(float* values,
                                       const float* weight,
                                       const float* bias,
                                       const float* mean,
                                       const float* variance,
                                       int channels,
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
                                              int out_channels,
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
    const int column_col = input_y * input_width + input_x;
    output[index] =
        columns[static_cast<std::size_t>(column_row) * input_spatial +
                column_col];
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
        const int source_channel = channel - channels_per_input;
        output[index] = second[static_cast<std::size_t>(source_channel) * spatial + offset];
    } else {
        const int source_channel = channel - channels_per_input * 2;
        output[index] = third[static_cast<std::size_t>(source_channel) * spatial + offset];
    }
}

int block_count(std::size_t total, int threads) {
    return static_cast<int>((total + threads - 1) / threads);
}

void launch_batch_norm_relu(DeviceTensor& tensor,
                            const BatchNormWeights& bn,
                            float epsilon) {
    DeviceArray weight = copy_to_device(bn.weight, "copy BN weight");
    DeviceArray bias = copy_to_device(bn.bias, "copy BN bias");
    DeviceArray mean = copy_to_device(bn.mean, "copy BN mean");
    DeviceArray variance = copy_to_device(bn.variance, "copy BN variance");
    constexpr int threads = 256;
    const int spatial = tensor.height * tensor.width;
    batch_norm_relu_kernel<<<block_count(tensor.count(), threads), threads>>>(
        tensor.values.data(), weight.data(), bias.data(), mean.data(),
        variance.data(), tensor.channels, spatial, epsilon, tensor.count());
    check_cuda(cudaGetLastError(), "launch BatchNorm-ReLU kernel");
}

DeviceTensor conv_bn_relu(CublasHandle& handle,
                          const DeviceTensor& input,
                          const ConvLayerWeights& layer,
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

    DeviceArray columns(column_count);
    constexpr int threads = 256;
    im2col_nchw_kernel<<<block_count(column_count, threads), threads>>>(
        input.values.data(), columns.data(), layer.in_channels, input.height,
        input.width, output_height, output_width, layer.kernel_size,
        layer.stride, layer.padding, column_count);
    check_cuda(cudaGetLastError(), "launch im2col kernel");

    DeviceArray weight = copy_to_device(layer.weight, "copy Conv weight");
    DeviceTensor output;
    output.channels = layer.out_channels;
    output.height = output_height;
    output.width = output_width;
    output.values = DeviceArray(output.count());

    const float alpha = 1.0F;
    const float beta = 0.0F;
    check_cublas(
        cublasSgemm(handle.get(), CUBLAS_OP_N, CUBLAS_OP_N,
                    spatial, layer.out_channels, reduction,
                    &alpha, columns.data(), spatial, weight.data(), reduction,
                    &beta, output.values.data(), spatial),
        layer.name.c_str());
    launch_batch_norm_relu(output, layer.batch_norm, epsilon);
    return output;
}

DeviceTensor deconv_bn_relu(CublasHandle& handle,
                            const DeviceTensor& input,
                            const TransposedConvLayerWeights& layer,
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
    DeviceArray columns(column_count);
    DeviceArray weight =
        copy_to_device(layer.gemm_weight, "copy transposed Conv weight");

    const float alpha = 1.0F;
    const float beta = 0.0F;
    check_cublas(
        cublasSgemm(handle.get(), CUBLAS_OP_N, CUBLAS_OP_N,
                    input_spatial, expanded_channels, layer.in_channels,
                    &alpha, input.values.data(), input_spatial,
                    weight.data(), layer.in_channels,
                    &beta, columns.data(), input_spatial),
        layer.name.c_str());

    DeviceTensor output;
    output.channels = layer.out_channels;
    output.height = input.height * layer.stride;
    output.width = input.width * layer.stride;
    output.values = DeviceArray(output.count());
    constexpr int threads = 256;
    deconv_columns_to_nchw_kernel<<<block_count(output.count(), threads), threads>>>(
        columns.data(), output.values.data(), output.channels,
        input.height, input.width, layer.kernel_size,
        output.height, output.width, output.count());
    check_cuda(cudaGetLastError(), "launch transposed Conv rearrange kernel");
    launch_batch_norm_relu(output, layer.batch_norm, epsilon);
    return output;
}

DeviceTensor run_block(CublasHandle& handle,
                       DeviceTensor input,
                       const std::vector<ConvLayerWeights>& layers,
                       float epsilon) {
    for (const ConvLayerWeights& layer : layers) {
        input = conv_bn_relu(handle, input, layer, epsilon);
    }
    return input;
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
    output.values = DeviceArray(output.count());
    constexpr int threads = 256;
    concat_three_nchw_kernel<<<block_count(output.count(), threads), threads>>>(
        first.values.data(), second.values.data(), third.values.data(),
        output.values.data(), first.channels, first.height * first.width,
        output.count());
    check_cuda(cudaGetLastError(), "launch RPN concat kernel");
    return output;
}

std::array<int, 3> shape_of(const DeviceTensor& tensor) {
    return {tensor.channels, tensor.height, tensor.width};
}

}  // namespace

FullRpnResult run_full_rpn_cuda(const HostTensor& input,
                               const FullRpnWeights& weights) {
    if (input.channels != 64 || input.height != 468 || input.width != 468) {
        throw std::invalid_argument("full RPN expects input shape [1,64,468,468]");
    }
    if (input.values.size() !=
        static_cast<std::size_t>(input.channels) * input.height * input.width) {
        throw std::invalid_argument("full RPN input element count mismatch");
    }

    CublasHandle handle;
    DeviceTensor current;
    current.channels = input.channels;
    current.height = input.height;
    current.width = input.width;
    current.values = copy_to_device(input.values, "copy RPN input");

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    check_cuda(cudaEventCreate(&start), "create RPN start event");
    check_cuda(cudaEventCreate(&stop), "create RPN stop event");
    check_cuda(cudaEventRecord(start), "record RPN start event");

    try {
        FullRpnResult result;

        current = run_block(handle, std::move(current), weights.blocks[0],
                            weights.batch_norm_eps);
        result.block_shapes[0] = shape_of(current);
        DeviceTensor up0 = conv_bn_relu(handle, current, weights.deblock0,
                                        weights.batch_norm_eps);
        result.deblock_shapes[0] = shape_of(up0);

        current = run_block(handle, std::move(current), weights.blocks[1],
                            weights.batch_norm_eps);
        result.block_shapes[1] = shape_of(current);
        DeviceTensor up1 = deconv_bn_relu(handle, current, weights.deblock1,
                                          weights.batch_norm_eps);
        result.deblock_shapes[1] = shape_of(up1);

        current = run_block(handle, std::move(current), weights.blocks[2],
                            weights.batch_norm_eps);
        result.block_shapes[2] = shape_of(current);
        DeviceTensor up2 = deconv_bn_relu(handle, current, weights.deblock2,
                                          weights.batch_norm_eps);
        result.deblock_shapes[2] = shape_of(up2);

        DeviceTensor output = concatenate(up0, up1, up2);
        check_cuda(cudaEventRecord(stop), "record RPN stop event");
        check_cuda(cudaEventSynchronize(stop), "synchronize RPN stop event");
        check_cuda(cudaEventElapsedTime(&result.elapsed_ms, start, stop),
                   "measure RPN time");

        result.output.channels = output.channels;
        result.output.height = output.height;
        result.output.width = output.width;
        result.output.values.resize(output.count());
        check_cuda(cudaMemcpy(result.output.values.data(), output.values.data(),
                              output.count() * sizeof(float),
                              cudaMemcpyDeviceToHost),
                   "copy RPN output to host");

        cudaEventDestroy(stop);
        cudaEventDestroy(start);
        return result;
    } catch (...) {
        cudaEventDestroy(stop);
        cudaEventDestroy(start);
        throw;
    }
}

}  // namespace centerpoint
