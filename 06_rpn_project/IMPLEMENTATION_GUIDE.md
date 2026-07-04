# 06 RPN CUDA 기본 연산 구현 코드 가이드

## 현재 마일스톤

전체 RPN을 한 번에 구현하지 않고 가장 기본 단위부터 CUDA로 검증한다.

```text
input [1,3,8,8]
  -> Conv2D, weight [4,3,3,3]
  -> BatchNorm inference
  -> ReLU
output [1,4,8,8]
```

핵심 파일은 `cuda/rpn_kernels.cu`다.

## CMake에서 CUDA 활성화

```cmake
project(centerpoint_rpn_cuda LANGUAGES CXX CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 120)
```

`120`은 RTX 5080의 `sm_120` 코드를 생성한다. 현재 PC는 Visual Studio CUDA 등록이 없어 configure할 때 toolkit 경로를 직접 전달한다.

```powershell
cmake -S . -B build_cuda `
  -T "cuda=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1"
```

## C++ main 흐름

`src/main.cpp`는 작은 입력과 weight를 deterministic하게 만든다.

```cpp
input[index] = ((index * 13) % 29 - 14) * 0.1F;
weight[index] = ((index * 7) % 23 - 11) * 0.01F;
```

그 다음 CUDA 함수와 dump writer를 호출한다.

```cpp
ConvBnReluResult result = run_conv_bn_relu_cuda(
    input, weights, bn_weight, bn_bias, bn_mean, bn_var, config);
write_rpn_demo_dump(output_dir, ..., result);
```

## DeviceBuffer

```cpp
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t bytes) {
        cudaMalloc(&data_, bytes);
    }
    ~DeviceBuffer() {
        cudaFree(data_);
    }
};
```

함수가 정상 종료되거나 예외가 발생해도 destructor가 GPU memory를 해제하는 RAII 구조다.

## Host에서 Device로 복사

```cpp
cudaMemcpy(device_input.data(), input.data(), bytes,
           cudaMemcpyHostToDevice);
```

현재 demo 흐름:

```text
CPU vector 생성
  -> cudaMalloc
  -> HostToDevice copy
  -> CUDA kernels
  -> DeviceToHost copy
  -> binary dump
```

## CUDA thread가 output 하나를 담당

```cpp
const int index = blockIdx.x * blockDim.x + threadIdx.x;
if (index >= total) return;
```

1차원 `index`를 NCHW 좌표로 되돌린다.

```cpp
out_x = index % output_width;
out_y = (index / output_width) % output_height;
out_channel = ... % out_channels;
batch_index = ... / out_channels;
```

즉 thread 하나가 `output[n,c,y,x]` 하나를 계산한다.

## Conv2D kernel

```cpp
for (int in_channel = 0; in_channel < in_channels; ++in_channel) {
    for (int kernel_y = 0; kernel_y < kernel_size; ++kernel_y) {
        for (int kernel_x = 0; kernel_x < kernel_size; ++kernel_x) {
            sum += input[input_offset] * weights[weight_offset];
        }
    }
}
output[index] = sum;
```

Padding 영역은 실제 input memory를 읽지 않고 건너뛴다.

```cpp
if (input_y < 0 || input_y >= input_height) continue;
if (input_x < 0 || input_x >= input_width) continue;
```

## BatchNorm-ReLU kernel

두 연산을 하나의 kernel에 넣었다.

```cpp
normalized = (value - mean[channel]) * rsqrtf(var[channel] + eps);
affine = normalized * weight[channel] + bias[channel];
value = fmaxf(affine, 0.0F);
```

`rsqrtf(x)`는 `1/sqrt(x)`를 GPU에서 계산한다.

## Kernel 실행

```cpp
constexpr int threads = 256;
const int blocks = (output_count + threads - 1) / threads;

conv2d_nchw_kernel<<<blocks, threads>>>(...);
batch_norm_relu_nchw_kernel<<<blocks, threads>>>(...);
```

올림 나눗셈으로 모든 output을 처리할 block 수를 만든다. 남는 thread는 `index >= total`에서 종료한다.

## CUDA 오류 처리

```cpp
check_cuda(cudaGetLastError(), "launch Conv2D kernel");
```

모든 CUDA API 상태를 확인하고 실패하면 `std::runtime_error`로 변환한다.

## 시간 측정

```cpp
cudaEventRecord(start);
// kernels
cudaEventRecord(stop);
cudaEventSynchronize(stop);
cudaEventElapsedTime(&elapsed_ms, start, stop);
```

현재 작은 demo의 시간은 GPU 상태와 최초 실행 overhead 영향을 크게 받으므로 성능 benchmark로 해석하면 안 된다.

## Python 비교

`tools/compare_python_cpp_rpn_cuda.py`가 같은 loop 순서로 Conv2D를 계산한다.

```text
Python/CUDA allclose: True
max abs diff: 0.00000009
```

## 전체 RPN으로 가는 다음 코드

```text
1. stride=2 Conv2D로 Downsample
2. Conv-BN-ReLU block 반복
3. ConvTranspose2D로 2배/4배 Upsample
4. 세 branch를 channel 방향으로 Concat
5. 실제 Scatter [1,64,468,468] 입력 연결
6. 실제 checkpoint weight loader 또는 TensorRT 연결
```

현재 직접 작성한 Conv kernel은 학습과 정확도 기준용이다. 전체 RPN 성능 구현은 cuDNN 또는 TensorRT가 적절하다.
