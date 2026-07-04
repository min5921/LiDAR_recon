# 05 Scatter Project

PFN이 생성한 pillar feature를 좌표에 따라 dense BEV pseudo-image로 배치하는 CPU C++ 구현입니다.

## Input

- PFN dump: `pillar_features.bin`, shape `[num_pillars, channels]`
- Voxel dump: `coordinates.bin`, order `[batch, z, y, x]`
- Voxel metadata: `grid_size_xyz`

## Output

- `bev_features.bin`: float32, NCHW layout
- `bev_features_metadata.json`

현재 KITTI 예제의 출력 shape는 `[1, 64, 468, 468]`입니다. 좌표가 없는 BEV cell은 0이며, 각 pillar의 64차원 feature는 `BEV[batch, :, y, x]`에 복사됩니다.

## Build

```powershell
cmake -S . -B build
cmake --build build --config Release
```

## Run

```powershell
.\build\Release\centerpoint_scatter.exe `
  ..\04_pfn_project\dump\kitti_000000_pfn `
  ..\02_project\dump\kitti_000000 `
  .\dump\kitti_000000_scatter
```

## Compare

```powershell
python .\tools\compare_python_cpp_scatter.py `
  --pfn-dump ..\04_pfn_project\dump\kitti_000000_pfn `
  --voxel-dump ..\02_project\dump\kitti_000000 `
  --scatter-dump .\dump\kitti_000000_scatter
```
