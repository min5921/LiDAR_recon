#include "centerpoint/rpn_cuda.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <string>
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

class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t bytes) : bytes_(bytes) {
        check_cuda(cudaMalloc(&data_, bytes_), "cudaMalloc");
    }

    ~DeviceBuffer() {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    void* data() { return data_; }
    std::size_t bytes() const { return bytes_; }

private:
    void* data_ = nullptr;
    std::size_t bytes_ = 0;
};

__global__ void conv2d_nchw_kernel(
    const float* input,
    const float* weights,
    float* output,
    int batch,
    int in_channels,
    int out_channels,
    int input_height,
    int input_width,
    int output_height,
    int output_width,
    int kernel_size,
    int stride,
    int padding) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch * out_channels * output_height * output_width;
    if (index >= total) {
        return;
    }

    int remaining = index;
    const int out_x = remaining % output_width;
    remaining /= output_width;
    const int out_y = remaining % output_height;
    remaining /= output_height;
    const int out_channel = remaining % out_channels;
    const int batch_index = remaining / out_channels;

    float sum = 0.0F;
    for (int in_channel = 0; in_channel < in_channels; ++in_channel) {
        for (int kernel_y = 0; kernel_y < kernel_size; ++kernel_y) {
            const int input_y = out_y * stride + kernel_y - padding;
            if (input_y < 0 || input_y >= input_height) {
                continue;
            }
            for (int kernel_x = 0; kernel_x < kernel_size; ++kernel_x) {
                const int input_x = out_x * stride + kernel_x - padding;
                if (input_x < 0 || input_x >= input_width) {
                    continue;
                }

                const std::size_t input_offset =
                    ((static_cast<std::size_t>(batch_index) * in_channels + in_channel) *
                         input_height + input_y) * input_width + input_x;
                const std::size_t weight_offset =
                    ((static_cast<std::size_t>(out_channel) * in_channels + in_channel) *
                         kernel_size + kernel_y) * kernel_size + kernel_x;
                sum += input[input_offset] * weights[weight_offset];
            }
        }
    }
    output[index] = sum;
}

__global__ void batch_norm_relu_nchw_kernel(
    float* values,
    const float* weight,
    const float* bias,
    const float* mean,
    const float* variance,
    int total,
    int channels,
    int spatial_size,
    float epsilon) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total) {
        return;
    }

    const int channel = (index / spatial_size) % channels;
    const float normalized =
        (values[index] - mean[channel]) * rsqrtf(variance[channel] + epsilon);
    const float affine = normalized * weight[channel] + bias[channel];
    values[index] = fmaxf(affine, 0.0F);
}

void validate_size(const std::vector<float>& values,
                   std::size_t expected,
                   const char* name) {
    if (values.size() != expected) {
        std::ostringstream message;
        message << name << " size mismatch: expected " << expected
                << ", got " << values.size();
        throw std::invalid_argument(message.str());
    }
}

}  // namespace

ConvBnReluResult run_conv_bn_relu_cuda(
    const std::vector<float>& input,
    const std::vector<float>& weights,
    const std::vector<float>& bn_weight,
    const std::vector<float>& bn_bias,
    const std::vector<float>& bn_mean,
    const std::vector<float>& bn_var,
    const ConvBnReluConfig& config) {
    if (config.batch <= 0 || config.in_channels <= 0 || config.out_channels <= 0 ||
        config.input_height <= 0 || config.input_width <= 0 ||
        config.kernel_size <= 0 || config.stride <= 0 || config.padding < 0) {
        throw std::invalid_argument("invalid Conv-BN-ReLU configuration");
    }

    const int output_height =
        (config.input_height + 2 * config.padding - config.kernel_size) /
            config.stride + 1;
    const int output_width =
        (config.input_width + 2 * config.padding - config.kernel_size) /
            config.stride + 1;
    if (output_height <= 0 || output_width <= 0) {
        throw std::invalid_argument("convolution output shape is not positive");
    }

    const std::size_t input_count =
        static_cast<std::size_t>(config.batch) * config.in_channels *
        config.input_height * config.input_width;
    const std::size_t weight_count =
        static_cast<std::size_t>(config.out_channels) * config.in_channels *
        config.kernel_size * config.kernel_size;
    const std::size_t output_count =
        static_cast<std::size_t>(config.batch) * config.out_channels *
        output_height * output_width;
    validate_size(input, input_count, "input");
    validate_size(weights, weight_count, "weights");
    validate_size(bn_weight, config.out_channels, "bn_weight");
    validate_size(bn_bias, config.out_channels, "bn_bias");
    validate_size(bn_mean, config.out_channels, "bn_mean");
    validate_size(bn_var, config.out_channels, "bn_var");

    DeviceBuffer device_input(input_count * sizeof(float));
    DeviceBuffer device_weights(weight_count * sizeof(float));
    DeviceBuffer device_output(output_count * sizeof(float));
    DeviceBuffer device_bn_weight(bn_weight.size() * sizeof(float));
    DeviceBuffer device_bn_bias(bn_bias.size() * sizeof(float));
    DeviceBuffer device_bn_mean(bn_mean.size() * sizeof(float));
    DeviceBuffer device_bn_var(bn_var.size() * sizeof(float));

    check_cuda(cudaMemcpy(device_input.data(), input.data(), device_input.bytes(),
                          cudaMemcpyHostToDevice), "copy input to device");
    check_cuda(cudaMemcpy(device_weights.data(), weights.data(), device_weights.bytes(),
                          cudaMemcpyHostToDevice), "copy weights to device");
    check_cuda(cudaMemcpy(device_bn_weight.data(), bn_weight.data(), device_bn_weight.bytes(),
                          cudaMemcpyHostToDevice), "copy BN weight to device");
    check_cuda(cudaMemcpy(device_bn_bias.data(), bn_bias.data(), device_bn_bias.bytes(),
                          cudaMemcpyHostToDevice), "copy BN bias to device");
    check_cuda(cudaMemcpy(device_bn_mean.data(), bn_mean.data(), device_bn_mean.bytes(),
                          cudaMemcpyHostToDevice), "copy BN mean to device");
    check_cuda(cudaMemcpy(device_bn_var.data(), bn_var.data(), device_bn_var.bytes(),
                          cudaMemcpyHostToDevice), "copy BN variance to device");

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    check_cuda(cudaEventCreate(&start), "create start event");
    try {
        check_cuda(cudaEventCreate(&stop), "create stop event");
        check_cuda(cudaEventRecord(start), "record start event");

        constexpr int threads = 256;
        const int blocks = static_cast<int>((output_count + threads - 1) / threads);
        conv2d_nchw_kernel<<<blocks, threads>>>(
            static_cast<const float*>(device_input.data()),
            static_cast<const float*>(device_weights.data()),
            static_cast<float*>(device_output.data()),
            config.batch,
            config.in_channels,
            config.out_channels,
            config.input_height,
            config.input_width,
            output_height,
            output_width,
            config.kernel_size,
            config.stride,
            config.padding);
        check_cuda(cudaGetLastError(), "launch Conv2D kernel");

        batch_norm_relu_nchw_kernel<<<blocks, threads>>>(
            static_cast<float*>(device_output.data()),
            static_cast<const float*>(device_bn_weight.data()),
            static_cast<const float*>(device_bn_bias.data()),
            static_cast<const float*>(device_bn_mean.data()),
            static_cast<const float*>(device_bn_var.data()),
            static_cast<int>(output_count),
            config.out_channels,
            output_height * output_width,
            config.batch_norm_eps);
        check_cuda(cudaGetLastError(), "launch BatchNorm-ReLU kernel");

        check_cuda(cudaEventRecord(stop), "record stop event");
        check_cuda(cudaEventSynchronize(stop), "synchronize stop event");

        ConvBnReluResult result;
        result.output.resize(output_count);
        result.batch = config.batch;
        result.channels = config.out_channels;
        result.height = output_height;
        result.width = output_width;
        check_cuda(cudaEventElapsedTime(&result.elapsed_ms, start, stop),
                   "measure kernel time");
        check_cuda(cudaMemcpy(result.output.data(), device_output.data(),
                              device_output.bytes(), cudaMemcpyDeviceToHost),
                   "copy output to host");

        cudaEventDestroy(stop);
        cudaEventDestroy(start);
        return result;
    } catch (...) {
        if (stop != nullptr) {
            cudaEventDestroy(stop);
        }
        cudaEventDestroy(start);
        throw;
    }
}

}  // namespace centerpoint
