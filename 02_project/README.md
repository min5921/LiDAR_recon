# CenterPoint C++/CUDA 실험 프로젝트

첫 마일스톤은 원본 Python voxelization과 비교 가능한 C++ 기준 구현을 만드는 것이다.

현재 실행파일:

```text
centerpoint_voxel_dump
```

역할:

```text
point_cloud.bin
  -> C++ float32 point loader
  -> Waymo PointPillars voxelization
  -> pillars / coordinates / num_points dump
```

입력 point cloud는 float32 row-major binary 형식이어야 한다.

기본 feature dimension은 5이다.

```text
[x, y, z, intensity, elongation]
```

## 빌드

```powershell
cmake -S . -B build
cmake --build build --config Release
```

## 실행

```powershell
.\build\Release\centerpoint_voxel_dump.exe <points.bin> <output_dir> [feature_dim]
```

예:

```powershell
.\build\Release\centerpoint_voxel_dump.exe .\sample_data\points.bin .\dump 5
```

## 출력

```text
pillars.bin
coordinates.bin
num_points.bin
metadata.json
```

`coordinates.bin`은 int32 `[num_pillars, 4]`이며 순서는 원본 CenterPoint 입력과 맞춰 `batch, z, y, x`이다.

