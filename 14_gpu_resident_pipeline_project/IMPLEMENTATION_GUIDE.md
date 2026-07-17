# GPU-Resident CenterPoint 전체 구현 설명

## 1. 독립 프로젝트 경계

`centerpoint_gpu_full.exe`는 이 프로젝트 안의 source만으로 빌드된다.

```text
src/main_full.cpp            CLI와 전체 실행 순서
src/io.cpp                   points.bin reader
src/pfn_weights.cpp          PFN binary weight reader
src/rpn_weights.cpp          RPN binary weight reader
src/head_weights.cpp         CenterHead binary weight reader
cuda/gpu_preprocess.cu       전처리 CUDA 구현
cuda/gpu_rpn.cu              RPN CUDA/cuBLAS 구현
cuda/gpu_center_head.cu      CenterHead CUDA/cuBLAS 구현
cuda/gpu_postprocess.cu      decode, 정렬, rotated NMS
include/centerpoint/*.hpp    단계 사이의 public contract
```

이전 프로젝트는 구현 수식과 weight 형식을 파악하는 참조였을 뿐 빌드 및 실행
의존성이 아니다.

## 2. 전체 실행 순서

`src/main_full.cpp`의 핵심 흐름은 다음과 같다.

1. `points.bin`과 `04_pfn`, `06_rpn`, `07_head` weight를 host에서 읽는다.
2. 네 pipeline 객체를 생성하며 weight와 작업 공간을 device에 준비한다.
3. `GpuPreprocessPipeline::run()`이 BEV를 device에 만든다.
4. `device_bev()`가 `DeviceBevView`를 반환한다.
5. `GpuRpnPipeline::run()`이 같은 pointer를 직접 입력으로 사용한다.
6. `GpuCenterHeadPipeline::run()`이 `DeviceRpnView`를 직접 사용한다.
7. `GpuPostprocessPipeline::run()`이 head map을 decode하고 NMS를 수행한다.
8. 마지막 `DeviceDetectionView`만 결과 출력을 위해 host로 복사한다.

```cpp
const auto preprocess_stats = preprocess_pipeline.run(points.data(), point_count);
const auto rpn_stats = rpn_pipeline.run(preprocess_pipeline.device_bev());
const auto head_stats = head_pipeline.run(rpn_pipeline.device_output(), validation);
const auto post_stats = postprocess_pipeline.run(head_pipeline.device_maps());
const auto detections = postprocess_pipeline.copy_detections_to_host();
```

일반 실행에는 stage별 `cudaMemcpyDeviceToHost`와 tensor 파일 출력이 없다.

## 3. Deterministic CUDA voxelization

단순 atomic hash는 thread 실행 순서에 따라 pillar와 내부 point 순서가 달라질 수
있다. 현재 구현은 CUB stable radix sort와 scan으로 Python hard voxelization의
순서를 복원한다.

```text
각 point의 flattened cell key 계산
  -> (cell key, original point index) stable radix sort
  -> 같은 key의 run 시작점 표시
  -> inclusive scan으로 group id 생성
  -> group의 첫 original point index 수집
  -> group을 첫 point index 순으로 다시 sort
  -> group 안 original 순서의 앞 20개 point 복사
```

따라서 `max_pillars=60000`, `max_points_per_pillar=20` 절단도 결정적이다.

## 4. Fused decoration, PFN, scatter

CUDA block 하나가 pillar 하나를 처리한다. 실제 point의 xyz 평균을 구한 후 각
point의 10차원 decoration을 register에서 바로 생성한다.

```text
[x, y, z, intensity, elongation,
 x-mean_x, y-mean_y, z-mean_z,
 x-center_x, y-center_y]
```

별도 decorated tensor를 global memory에 저장하지 않고 두 PFN layer를 이어서
계산한다.

```text
[20,10]
  -> Linear [10,32] -> BN -> ReLU
  -> pillar max [32]
  -> local [32] + repeated max [32]
  -> Linear [64,64] -> BN -> ReLU
  -> pillar max [64]
```

마지막 scatter는 pillar feature를 NCHW BEV에 직접 기록한다.

```text
bev[(channel * 468 + y) * 468 + x] = pillar_feature[channel]
```

## 5. RPN 네트워크 구조

RPN은 세 해상도의 block feature를 모두 `468 x 468`로 맞춘 뒤 channel 방향으로
합친다.

| 경로 | 연산 | Block 출력 | Deblock 출력 |
|---|---|---|---|
| 0 | Conv 3x3 네 번, stride 1 | `[64,468,468]` | Conv 1x1 `[128,468,468]` |
| 1 | 첫 Conv stride 2 + Conv 다섯 번 | `[128,234,234]` | Deconv 2x2 `[128,468,468]` |
| 2 | 첫 Conv stride 2 + Conv 다섯 번 | `[256,117,117]` | Deconv 4x4 `[128,468,468]` |

세 deblock을 concat하면 최종 feature는 `[1,384,468,468]`이다.

## 6. CUDA Conv 구현

일반 Conv는 `im2col_nchw_kernel`과 `cublasSgemm`으로 나눈다.

```text
input [Cin,H,W]
  -> im2col [Cin*K*K, Hout*Wout]
  -> weight [Cout, Cin*K*K]와 GEMM
  -> output [Cout,Hout,Wout]
  -> BatchNorm + ReLU CUDA kernel
```

`stride`와 `padding`은 im2col에서 적용하고, 범위를 벗어난 위치는 0으로 채운다.
BN은 checkpoint의 running mean/variance, affine weight/bias와
`epsilon=1e-3`을 사용한다.

## 7. CUDA Transposed Conv 구현

`deblock1`, `deblock2`는 kernel size와 stride가 각각 `2/2`, `4/4`다. weight는
미리 GEMM용 `[Cout*K*K, Cin]` 형식으로 export되어 있다.

```text
weight_gemm [Cout*K*K,Cin] x input [Cin,H*W]
  -> columns [Cout*K*K,H*W]
  -> kernel offset에 따라 [Cout,H*K,W*K]로 재배열
  -> BatchNorm + ReLU
```

이 설정에서는 deconvolution kernel들이 겹치지 않으므로 scatter-add가 아니라
각 출력 위치에 대응하는 column 하나를 직접 배치할 수 있다.

## 8. GPU 메모리 소유권

- `GpuPreprocessPipeline::Impl`: point scratch, CUB storage, pillar, PFN, BEV
- `GpuRpnPipeline::Impl`: RPN weight, 임시 layer tensor, 최종 RPN feature
- `GpuCenterHeadPipeline::Impl`: head weight, Conv workspace, 5개 출력 map
- `GpuPostprocessPipeline::Impl`: score/index 정렬, 후보, NMS mask, detection
- `DeviceBevView`: 전처리 소유 BEV의 읽기 전용 view
- `DeviceRpnView`: RPN 소유 최종 feature의 읽기 전용 view
- `DeviceHeadMaps`: CenterHead 소유 5개 출력의 읽기 전용 view
- `DeviceDetectionView`: postprocess 소유 최종 detection의 읽기 전용 view

view는 메모리를 소유하지 않으므로 해당 pipeline 객체보다 오래 보관하면 안 된다.
각 다음 단계는 이전 단계 view의 device pointer를 그대로 받는다.

## 9. CenterHead 구현

RPN의 `[384,468,468]` feature에 shared `3x3 Conv 384->64`를 적용한 후 다섯
branch가 각각 hidden `3x3 Conv 64->64`와 output `3x3 Conv`를 수행한다.

| Branch | 출력 channel | 의미 |
|---|---:|---|
| `reg` | 2 | cell 내부 x/y offset |
| `height` | 1 | box 중심 z |
| `dim` | 3 | log box size |
| `rot` | 2 | `sin(yaw)`, `cos(yaw)` |
| `hm` | 3 | vehicle, pedestrian, cyclist logit |

shared와 hidden Conv는 `Conv -> bias -> BN -> ReLU`, 각 output Conv는
`Conv -> bias`만 수행한다. Conv는 RPN과 같은 `im2col + cublasSgemm`을 사용하며,
가장 큰 column workspace와 branch 출력 buffer를 실행마다 재할당하지 않고
재사용한다. 이 변경으로 같은 frame의 관찰값에서 CenterHead가 약 `2280 ms`에서
`22~42 ms` 범위로 줄었다.

## 10. CUDA decode와 score 정렬

feature-map cell마다 CUDA thread 하나가 다음 연산을 수행한다.

```text
label, score = argmax(sigmoid(hm[:, y, x]))
center_x = (x + reg_x) * 0.32 - 74.88
center_y = (y + reg_y) * 0.32 - 74.88
center_z = height
size = exp(dim)
yaw = atan2(rot_sin, rot_cos)
```

score와 post-center range 조건을 통과하지 못한 cell은 score `-1`로 표시한다.
모든 `468*468` score/index 쌍을 CUB stable radix sort로 내림차순 정렬하고 앞의
최대 4096개만 pre-NMS 후보로 만든다. 후보 개수를 먼저 host로 읽어 정렬 크기를
정하는 왕복은 없다.

## 11. CUDA rotated NMS

각 box의 네 꼭짓점을 회전시킨 뒤 convex polygon clipping으로 교집합 면적과
rotated IoU를 계산한다. 64개 후보 단위 block pair마다 suppression bitmask를
GPU에서 만들고, 정렬된 앞 후보부터 하나의 CUDA thread가 mask를 합치며 최대
500개를 선택한다. 현재 checkpoint는 세 class가 하나의 CenterHead task에 있으므로
기존 구현과 동일하게 class-agnostic NMS를 사용한다.

`pcdet` convention은 Python/OpenPCDet 기준과 yaw 방향 및 길이/폭 해석을 맞춘다.
검증에서 Python rotated NMS와 선택한 index 및 값이 정확히 일치했다.

## 12. 검증용 probe

전체 tensor를 Python으로 복사하면 단계별 파일 저장 방식으로 되돌아가므로,
검증 모드만 RPN과 CenterHead 각 layer에서 두 출력 위치를 선택한다.

```text
sample 0: channel 0, feature map 중앙
sample 1: channel 7, 좌상단 경계
```

Conv probe는 해당 출력의 `Cin*K*K` 입력 patch를, Deconv probe는 대응 입력 위치의
`Cin` 값을 복사한다. Python은 weight와 BN 파일을 독립적으로 읽어 출력 scalar를
다시 계산한다. CenterHead는 11개 Conv의 22개 probe를 검사한다. 일반 실행에서는
이 복사와 probe 생성 자체가 비활성화된다.

## 13. 검증 경계와 한계

전처리는 최종 BEV 전체를 NumPy와 비교했다. RPN은 19개 layer의 38개 대표
지점을 비교했으므로 shape, weight 방향, padding/stride, BN/ReLU 오류를 강하게
검사하지만 최종 RPN tensor 전체 비교는 아니다. CenterHead도 11개 layer의 22개
대표 지점을 비교했다. 마지막에는 전체 head map에서 Python이 독립 decode한 모든
pre-NMS 후보와 Python rotated NMS 결과를 비교해 stage-local probe의 빈틈을
보완했다. 실제 Waymo label과의 AP 평가는 numerical parity와 별개의 다음 검증이다.
