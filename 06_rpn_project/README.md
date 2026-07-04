# 06 RPN CUDA Project

CenterPoint PointPillars RPN의 기본 연산인 `Conv2D -> BatchNorm -> ReLU`를 CUDA로 구현하는 첫 단계다.

현재 마일스톤은 전체 RPN이 아니라, 작은 deterministic tensor를 사용해 CUDA 연산과 NCHW memory layout을 검증한다.

## 현재 연산

```text
input  [1, 3, 8, 8]
weight [4, 3, 3, 3]
  -> Conv2D, stride=1, padding=1
  -> BatchNorm inference, eps=1e-3
  -> ReLU
output [1, 4, 8, 8]
```

## 파일

```text
cuda/rpn_kernels.cu                 CUDA kernels와 device memory 관리
src/main.cpp                        deterministic 입력/weight 생성과 실행
src/io/rpn_dump_writer.cpp          입력, weight, 출력 binary 저장
tools/compare_python_cpp_rpn_cuda.py Python 기준 연산과 비교
```

## Build

```powershell
cmake -S . -B build_cuda `
  -T "cuda=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1"
cmake --build build_cuda --config Release
```

현재 PC에는 CUDA toolkit이 설치돼 있지만 Visual Studio Build Customization 검색 경로에는 등록되지 않았다. 그래서 `-T cuda=<toolkit path>`로 CUDA 위치를 직접 전달한다.

## Run

```powershell
.\build_cuda\Release\centerpoint_rpn_cuda_demo.exe .\dump\conv_bn_relu_demo
```

## Compare

```powershell
python .\tools\compare_python_cpp_rpn_cuda.py --dump .\dump\conv_bn_relu_demo
```

## 원본 RPN으로 확장할 shape

```text
Scatter input: [1, 64, 468, 468]

Block 0 -> [1,  64, 468, 468] -> deblock [1,128,468,468]
Block 1 -> [1, 128, 234, 234] -> deblock [1,128,468,468]
Block 2 -> [1, 256, 117, 117] -> deblock [1,128,468,468]

Concat -> [1,384,468,468]
```

현재 kernel은 연산 이해와 정확도 검증을 위한 직접 구현이다. 전체 RPN 성능 구현에서는 cuDNN 또는 TensorRT를 사용하는 편이 적절하다.
