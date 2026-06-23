# PFN Dummy Project

이 프로젝트는 decorated pillar tensor를 PFN 구조로 통과시켜 pillar feature를 만든다.

현재 단계에서는 실제 checkpoint weight가 없으므로 deterministic dummy weight를 사용한다.

입력:

```text
decorated_pillars.bin
decorated_metadata.json
```

출력:

```text
pillar_features.bin
pillar_features_metadata.json
```

현재 구현하는 PFN 연산:

```text
Linear
BatchNorm 형태의 affine/normalize
ReLU
max pooling over points
```

KITTI 샘플 기준:

```text
[10404, 20, 9] -> [10404, 64]
```

## 빌드

```powershell
cmake -S . -B build
cmake --build build --config Release
```

## 실행

```powershell
.\build\Release\centerpoint_pfn_dummy.exe ..\03_pillar_feature_project\dump\kitti_000000_decorated dump\kitti_000000_pfn
```

