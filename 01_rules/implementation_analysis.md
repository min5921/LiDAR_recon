# CenterPoint C++/CUDA 구현 분석

## 원본 코드 위치

원본 프레임워크 위치:

`00_reference/centerpoint_original/CenterPoint-master`

현재 원본은 PyTorch 기반 CenterPoint 구현이다. 크게 두 가지 detector 계열을 지원한다.

- `VoxelNet`: `spconv`를 사용하는 sparse 3D convolution 구조
- `PointPillars`: pillar feature를 만든 뒤 BEV pseudo image로 scatter하고, 이후 dense 2D CNN을 사용하는 구조

첫 C++/CUDA 구현은 `PointPillars` 경로로 가는 것이 좋다. `VoxelNet`은 sparse convolution 엔진 의존도가 높아서 C++/CUDA로 옮길 때 `spconv`에 해당하는 영역까지 함께 해결해야 한다. 반면 `PointPillars`는 전처리, 네트워크 추론, 후처리를 비교적 명확하게 나눌 수 있다.

## 1차 구현 추천 대상

첫 구현 대상 config:

`configs/waymo/pp/waymo_centerpoint_pp_two_pfn_stride1_3x.py`

선택 이유:

- 하나의 task만 사용한다.
- 클래스가 `VEHICLE`, `PEDESTRIAN`, `CYCLIST` 3개라 단순하다.
- velocity head가 없다.
- sparse voxelnet보다 pillar 입력 처리가 단순하다.
- pillar scatter 이후에는 dense 2D feature map이 되므로 TensorRT/ONNX로 넘기기 좋다.

NuScenes PointPillars는 Waymo 버전이 동작한 뒤 확장하는 것이 좋다. NuScenes는 task head가 6개이고 velocity head도 있어서 decode와 post-processing이 더 복잡하다.

## 추론 흐름

원본 Python 기준 흐름:

1. `tools/simple_inference_waymo.py`
2. `VoxelGenerator.generate(points)`
3. `PointPillars.forward`
4. `PillarFeatureNet.forward`
5. `PointPillarsScatter.forward`
6. `RPN.forward`
7. `CenterHead.forward`
8. `CenterHead.predict`
9. `iou3d_nms_cuda.nms_gpu`를 통한 rotated NMS

C++/CUDA 구현 기준 흐름:

1. point cloud를 host memory로 로드한다.
2. points를 GPU로 복사한다.
3. CUDA에서 voxelization/pillarization을 수행한다.
4. pillar feature decoration을 수행한다. PFN까지 CUDA/C++로 구현하거나, PFN을 neural network export에 포함한다.
5. pillar feature를 BEV pseudo image로 scatter한다.
6. neural network inference를 실행한다.
7. GPU에서 CenterHead 출력을 decode한다.
8. score/range filter를 적용한다.
9. score 기준 정렬 후 rotated NMS를 수행한다.
10. 최종 detection 결과만 host로 복사한다.

## 모듈 분리

실제 구현은 `02_project` 아래에 둔다.

추천 파일 구조:

```text
02_project/
  CMakeLists.txt
  README.md
  include/
    centerpoint/config.hpp
    centerpoint/types.hpp
    centerpoint/pipeline.hpp
    centerpoint/tensorrt_engine.hpp
    centerpoint/cuda/voxelization.cuh
    centerpoint/cuda/scatter.cuh
    centerpoint/cuda/decode.cuh
    centerpoint/cuda/nms.cuh
  src/
    main.cpp
    config.cpp
    pipeline.cpp
    tensorrt_engine.cpp
  cuda/
    voxelization.cu
    scatter.cu
    decode.cu
    nms.cu
  configs/
    waymo_pointpillars.yaml
  tests/
    test_decode.cpp
    test_voxelization.cpp
  tools/
    export_onnx.py
    compare_python_cpp.py
```

## CUDA 구현 항목

### Voxelization

참조 파일:

`det3d/ops/point_cloud/point_cloud_ops.py`

원본 좌표 규칙:

- 입력 point는 `x, y, z, ...` 순서다.
- `reverse_index=True`일 때 출력 coordinate는 `z, y, x` 순서다.
- 이후 batch 차원이 앞에 붙어서 최종 coordinate는 `batch, z, y, x`가 된다.

Waymo PointPillars 기준 값:

- range: `[-74.88, -74.88, -2, 74.88, 74.88, 4.0]`
- voxel size: `[0.32, 0.32, 6.0]`
- max points per voxel: `20`
- max voxels inference: `60000`
- grid size: `[468, 468, 1]` (`x, y, z` 기준)

구현 전략:

- Phase 1: Python 원본과 비교하기 쉽도록 C++ CPU voxelization부터 deterministic하게 만든다.
- Phase 2: hash table 또는 dense coordinate map 기반 CUDA voxelization으로 확장한다.

### Pillar Feature Decoration

참조 파일:

`det3d/models/readers/pillar_encoder.py`

각 pillar 안의 point마다 다음 feature를 만든다.

- 원본 point feature
- `f_cluster = point_xyz - mean_xyz`
- `f_center_x = point_x - (coord_x * voxel_x + x_offset)`
- `f_center_y = point_y - (coord_y * voxel_y + y_offset)`

Waymo config 기준:

- `num_input_features=5`
- `with_distance=False`
- PFN filters: `[64, 64]`

실용적인 추천:

- 가능하면 `PillarFeatureNet + RPN + CenterHead.forward`를 ONNX로 export한다.
- C++/CUDA에서는 voxelization과 최종 decode/NMS를 담당한다.
- dynamic pillar count 때문에 ONNX export가 어렵다면 PFN을 CUDA/C++로 구현하고, dense BEV 2D network부터 TensorRT로 넘긴다.

### Scatter

참조 함수:

`PointPillarsScatter.forward`

각 pillar feature를 다음 위치에 배치한다.

```text
canvas[channel, y * nx + x] = pillar_feature[channel]
```

최종 BEV tensor shape:

```text
[batch, channels, ny, nx]
```

Waymo PointPillars 기준:

```text
[1, 64, 468, 468]
```

### Decode

참조 파일:

`det3d/models/bbox_heads/center_head.py`

각 grid cell마다 다음 계산을 수행한다.

- `score = sigmoid(hm)`
- `dim = exp(dim)`
- `rot = atan2(rot_sin, rot_cos)`
- `x = (grid_x + reg_x) * out_size_factor * voxel_size_x + pc_range_x`
- `y = (grid_y + reg_y) * out_size_factor * voxel_size_y + pc_range_y`
- NMS 전 box format: `[x, y, z, l, w, h, rot]`

Waymo PointPillars 기준:

- score threshold: `0.1`
- NMS pre max: `4096`
- NMS post max: `500`
- NMS IoU threshold: `0.7`
- decode pc range: `[-74.88, -74.88]`
- decode voxel size: `[0.32, 0.32]`

### Rotated NMS

참조 파일:

- `det3d/core/bbox/box_torch_ops.py`
- `det3d/ops/iou3d_nms/src/iou3d_nms_kernel.cu`

원본은 NMS 전에 box convention을 다음처럼 변환한다.

```text
boxes = boxes[:, [0, 1, 2, 4, 3, 5, -1]]
boxes[:, -1] = -boxes[:, -1] - pi / 2
```

원본 CUDA NMS 로직은 재사용할 수 있다. 다만 PyTorch binding을 제거하고 일반 C++/CUDA API로 노출해야 한다.

## Neural Network 전략

추천 runtime:

- C++ 추론용으로 TensorRT 사용
- 모델 교환 형식으로 ONNX 사용

Phase 1:

- 검증 가능한 Waymo PointPillars checkpoint를 준비한다.
- voxel tensor부터 head output까지 full ONNX export를 시도한다.
- dynamic voxel count 때문에 막히면 BEV pseudo image 이후부터 분리한다.

Phase 2:

- 전처리/후처리 CUDA 모듈과 TensorRT engine wrapper를 구현한다.
- 고정 sample에 대해 PyTorch output tensor와 C++ output tensor를 비교한다.

Phase 3:

- CUDA voxelization, scatter, decode, NMS를 최적화한다.

## Parity Test 순서

검증은 아래 순서로 진행한다.

1. `points_to_voxel` 대비 voxelization 결과 비교
2. `PointPillarsScatter.forward` 대비 scatter 결과 비교
3. NMS 전 `CenterHead.predict` decode 결과 비교
4. `iou3d_nms_cuda.nms_gpu` 대비 rotated NMS 결과 비교
5. `simple_inference_waymo.py` 대비 end-to-end detection 결과 비교

## 첫 코딩 마일스톤

처음부터 전체 네트워크를 붙이지 않는다. 첫 목표는 아래 흐름까지다.

```text
point_cloud.bin
  -> C++ loader
  -> voxelization
  -> pillar tensor
  -> coordinate tensor
  -> debug dump
```

이후 dump된 tensor를 Python 원본 voxelization 결과와 비교한다. 이 단계가 맞아야 뒤의 TensorRT 추론과 decode/NMS 검증도 안정적으로 진행할 수 있다.

