# 14 GPU-Resident Full Pipeline Project

기존 `02~09` 마일스톤의 실행 파일이나 library를 호출하지 않고, CenterPoint
PointPillars 추론 전체를 새로 구현한 독립 C++/CUDA 프로젝트다. 입력 point를
한 번 GPU에 올린 뒤 최종 detection까지 device pointer로 이어서 처리한다.

```text
points.bin [N,5]
  -> deterministic CUDA voxelization
  -> fused pillar decoration + two-layer PFN
  -> CUDA scatter: BEV [1,64,468,468]
  -> CUDA RPN: feature [1,384,468,468]
  -> CUDA CenterHead: reg/height/dim/rot/hm
  -> CUDA decode + CUB score sort
  -> CUDA rotated NMS
  -> detections [M,10]
```

Production 실행 중에는 중간 tensor를 host로 복사하거나 파일로 저장하지 않는다.
최종 detection만 CPU로 복사하며, 검증 모드에서만 작은 layer probe와 pre-NMS
후보를 복사한다.

## 입력과 weight 계약

- point 형식: float32 `[x, y, z, intensity, elongation]`
- Waymo intensity: loader에서 미리 `tanh`가 적용된 값
- voxel size: `[0.32, 0.32, 6.0]`
- point cloud range: `[-74.88, -74.88, -2, 74.88, 74.88, 4]`
- max points per pillar: `20`
- max pillars: `60000`
- weight root: `04_pfn`, `06_rpn`, `07_head` 디렉터리 필요
- 기본 score threshold: `0.35`
- 기본 rotated NMS IoU: `0.5`, `pcdet` 좌표 convention

## 주요 파일

```text
src/main_full.cpp                  전체 GPU 추론 실행 흐름
cuda/gpu_preprocess.cu             voxelization, decoration/PFN, scatter
cuda/gpu_rpn.cu                    RPN Conv/Deconv, BN/ReLU, concat
cuda/gpu_center_head.cu            shared Conv와 5개 head branch
cuda/gpu_postprocess.cu            decode, CUB sort, rotated NMS
src/*_weights.cpp                  독립 binary weight reader
include/centerpoint/*.hpp          단계 사이의 device-view 계약
tools/compare_python_gpu_*.py      NumPy layer probe 비교
tools/validate_gpu_detections.py   decode/NMS/기존 결과 비교
```

## 빌드

```powershell
cmake -S . -B build_cuda `
  -T "cuda=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1"
cmake --build build_cuda --config Release
```

RTX 5080용으로 `CMAKE_CUDA_ARCHITECTURES=120`과 `--fmad=false`를 사용한다.

## 전체 실행

```powershell
.\build_cuda\Release\centerpoint_gpu_full.exe `
  <points.bin> <weights_root> `
  --output-dir <output_directory>
```

생성 파일은 최종 `detections.csv`와 실행 정보 `summary.json`뿐이다. 출력 폴더를
생략하면 detection을 화면에만 출력한다.

주요 선택 옵션:

```text
--score-threshold <0..1>
--nms-iou <0..1>
--nms-convention pcdet|current
--class-thresholds <vehicle> <pedestrian> <cyclist>
```

## 검증 실행

```powershell
.\build_cuda\Release\centerpoint_gpu_full.exe `
  <points.bin> <weights_root> `
  --output-dir <verification_output> --validation

python -B tools\compare_python_gpu_head_probes.py `
  <verification_output>\head_probes.json <weights_root>\07_head

python -B tools\validate_gpu_detections.py `
  --pre-nms <verification_output>\pre_nms_candidates.csv `
  --detections <verification_output>\detections.csv `
  --reference-head-dir <reference_frame>\07_head `
  --reference-detections <reference_frame>\08_detections\detections.csv
```

Waymo 5개 frame에서 CenterHead probe `110/110`, decode 후보 `143/143`, 최종
detection `28/28`이 Python 및 기존 기준 결과와 일치했다. 자세한 구현과 수치는
`IMPLEMENTATION_GUIDE.md`, `VALIDATION_REPORT.md`에 정리되어 있다.
