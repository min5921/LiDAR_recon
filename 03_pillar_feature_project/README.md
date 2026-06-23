# Pillar Feature Decoration

이 프로젝트는 `02_project`에서 만든 raw pillar dump를 읽고 PointPillars의 feature decoration 결과를 만든다.

입력:

```text
pillars.bin
coordinates.bin
num_points.bin
metadata.json
```

출력:

```text
decorated_pillars.bin
decorated_metadata.json
```

KITTI 샘플 기준 입력 point feature는 `[x, y, z, intensity]`이고, 출력 feature는 다음 9개다.

```text
x, y, z, intensity,
x - mean_x, y - mean_y, z - mean_z,
x - pillar_center_x, y - pillar_center_y
```

## 빌드

```powershell
cmake -S . -B build
cmake --build build --config Release
```

## 실행

```powershell
.\build\Release\centerpoint_decorate_pillars.exe ..\02_project\dump\kitti_000000 dump\kitti_000000_decorated
```

