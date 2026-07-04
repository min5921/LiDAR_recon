# 전체 RPN CUDA 구현 가이드

## 목표

실제 PointPillars Scatter 출력을 checkpoint의 `neck.*` weight로 처리한다.

```text
입력  [1,64,468,468]
출력  [1,384,468,468]
```

PyTorch runtime은 사용하지 않는다. 일반 convolution은 CUDA `im2col`과 cuBLAS SGEMM, 나머지 연산은 직접 작성한 CUDA kernel로 실행한다.

## 실제 RPN 구조

```text
Block 0: 4 x Conv-BN-ReLU
  [64,468,468]
  -> 1x1 Conv deblock [128,468,468]

Block 1: stride 2 Conv + 5 x Conv-BN-ReLU
  [128,234,234]
  -> 2x2 Transposed Conv [128,468,468]

Block 2: stride 2 Conv + 5 x Conv-BN-ReLU
  [256,117,117]
  -> 4x4 Transposed Conv [128,468,468]

세 branch channel concat
  -> [384,468,468]
```

## 주요 파일

```text
cuda/rpn_full_cuda.cu
  GPU memory, im2col, SGEMM, BN/ReLU, deconvolution, concat

src/main_full.cpp
  BEV와 weight를 읽고 전체 RPN 실행

src/io/rpn_weight_reader.cpp
  추출된 95개 float tensor 검증과 로드

tools/extract_rpn_weights.py
  .pth의 neck.* tensor를 PyTorch 없이 추출

tools/check_selected_rpn_values.py
  실제 weight를 CPU scalar 연산으로 재계산
```

## Conv2D

### 1. im2col

입력 주변의 `3x3` 영역을 GEMM용 matrix로 펼친다.

```text
input:   [Cin,H,W]
columns: [Cin*3*3, Hout*Wout]
```

CUDA thread 하나가 `columns[k,n]` 하나를 담당한다. Padding 위치는 0을 기록한다.

### 2. cuBLAS SGEMM

```text
weight  [Cout, Cin*3*3]
columns [Cin*3*3, Hout*Wout]
output  [Cout, Hout*Wout]
```

Tensor data는 NCHW row-major다. cuBLAS의 column-major 규칙에 맞춰 transpose view로 SGEMM parameter를 전달한다.

### 3. BatchNorm + ReLU

```cpp
normalized = (value - mean[c]) * rsqrtf(var[c] + eps);
output = fmaxf(normalized * gamma[c] + beta[c], 0.0F);
```

두 연산을 kernel 하나에서 처리한다.

## Transposed Conv2D

PyTorch weight shape:

```text
[input_channels, output_channels, kernel_y, kernel_x]
```

Exporter가 다음 GEMM shape로 변환한다.

```text
[output_channels * kernel * kernel, input_channels]
```

SGEMM 결과를 CUDA kernel이 NCHW output 위치로 재배치한다. 현재 config는 `kernel_size == stride`이므로 output cell이 겹치지 않아 atomic addition이 필요 없다.

## Concat

세 deblock은 모두 `[128,468,468]`이다. CUDA kernel이 channel 범위에 따라 source branch를 선택한다.

```text
output channel   0..127 -> deblock0
output channel 128..255 -> deblock1
output channel 256..383 -> deblock2
```

## GPU memory

가장 큰 임시 buffer는 Block 0의 im2col matrix다.

```text
64 * 3 * 3 * 468 * 468 float32
약 504 MB
```

각 layer가 끝나면 RAII `DeviceArray`가 이전 임시 memory를 해제한다. 세 deblock 출력은 마지막 concat까지 유지한다.

## Build

```powershell
cmake -S . -B build_cuda `
  -T "cuda=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1"

cmake --build build_cuda --config Release `
  --target centerpoint_rpn_full_cuda
```

## Weight 추출

```powershell
python .\tools\extract_rpn_weights.py `
  --checkpoint "..\00_reference\checkpoints\waymo\centerpoint_waymo_pointpillars_50_novelocity.pth" `
  --output-dir ".\weights\waymo_pointpillars_50_novelocity"
```

## 실행

```powershell
.\build_cuda\Release\centerpoint_rpn_full_cuda.exe `
  <bev_dump_dir> `
  .\weights\waymo_pointpillars_50_novelocity `
  <output_dir>
```

전체 tensor 저장이 필요 없으면 마지막에 `--summary-only`를 붙인다.

## 현재 범위

Batch size 1과 고정 입력 `[1,64,468,468]`을 지원한다. 다음 단계 CenterHead에서는 RPN output을 host로 복사하지 않고 GPU memory에서 직접 전달하는 통합 API로 변경하는 것이 좋다.
