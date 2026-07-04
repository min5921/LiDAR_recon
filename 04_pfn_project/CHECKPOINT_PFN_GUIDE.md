# 실제 Waymo 2단 PFN 실행 가이드

기존 `centerpoint_pfn_dummy`는 구조 학습용으로 유지한다. 새 `centerpoint_pfn_checkpoint`는 다음 체크포인트의 실제 PFN weight를 사용한다.

```text
centerpoint_waymo_pointpillars_50_novelocity.pth
```

## 구조

```text
decorated input [P,20,10]

PFN layer 0
  Linear [32,10]
  BatchNorm [32]
  ReLU
  MaxPool [32]
  Local과 반복된 Max를 concat
  -> [P,20,64]

PFN layer 1
  Linear [64,64]
  BatchNorm [64]
  ReLU
  MaxPool
  -> [P,64]
```

## Weight 추출

PyTorch 설치 없이 ZIP checkpoint의 contiguous FloatStorage를 읽는다.

```powershell
python .\tools\extract_pfn_weights.py `
  --checkpoint "..\00_reference\checkpoints\waymo\centerpoint_waymo_pointpillars_50_novelocity.pth" `
  --output-dir ".\weights\waymo_pointpillars_50_novelocity"
```

## 10차원 검증 입력 준비

현재 KITTI sample의 4개 feature 뒤에 `elongation=0`을 추가한다.

```powershell
python .\tools\make_kitti_feature5_fixture.py `
  --input "..\00_reference\sample_data\kitti\000000.bin" `
  --output ".\fixtures\kitti_000000_feature5.bin"
```

기존 Voxelization과 Decoration을 그대로 실행한다.

```powershell
..\02_project\build\Release\centerpoint_voxel_dump.exe `
  ".\fixtures\kitti_000000_feature5.bin" `
  ".\dump\kitti_000000_waymo5_voxel" 5

..\03_pillar_feature_project\build\Release\centerpoint_decorate_pillars.exe `
  ".\dump\kitti_000000_waymo5_voxel" `
  ".\dump\kitti_000000_waymo10_decorated"
```

## Build와 실행

```powershell
cmake -S . -B build
cmake --build build --config Release

.\build\Release\centerpoint_pfn_checkpoint.exe `
  ".\dump\kitti_000000_waymo10_decorated" `
  ".\weights\waymo_pointpillars_50_novelocity" `
  ".\dump\kitti_000000_checkpoint_pfn"
```

## NumPy 비교

```powershell
python .\tools\compare_python_cpp_pfn_checkpoint.py `
  --decorated-dump ".\dump\kitti_000000_waymo10_decorated" `
  --weight-dir ".\weights\waymo_pointpillars_50_novelocity" `
  --pfn-dump ".\dump\kitti_000000_checkpoint_pfn"
```

이 fixture는 tensor 연산 검증용이다. `elongation=0`인 KITTI point이므로 Waymo detection 정확도를 평가하는 데이터는 아니다.
