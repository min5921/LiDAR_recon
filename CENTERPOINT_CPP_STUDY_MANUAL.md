# CenterPoint PointPillars C++ 구현 학습 매뉴얼

이 문서는 현재 프로젝트를 처음 보는 사람도 전체 구조를 이해하고, 코드를 읽고, 빌드하고, 같은 결과를 재현할 수 있도록 작성한 통합 매뉴얼이다.

현재 구현은 **Waymo용 CenterPoint PointPillars 추론 구조를 C++/CUDA로 옮긴 단계별 학습 프로젝트**다. Voxelization부터 PFN, Scatter, 전체 RPN, CenterHead, Box Decode, Rotated NMS까지 구현되어 최종 3D bounding box 파일을 만들 수 있다. Waymo derived sensor archive 5프레임의 전체 추론, GT 평가, 단계별 독립 수치 검증, False Negative 원인 분석도 완료했다. 현재 남은 큰 작업은 더 많은 프레임의 통계 평가, Waymo 공식 metric 연동, 그리고 GPU 메모리 안에서 단계를 직접 연결하는 최적화다.

## 이 문서를 읽는 방법

목적에 따라 다음 순서로 읽으면 된다.

```text
처음 실행하려는 경우:
  0 -> 4 -> 5 -> 6 -> 17

연산 원리를 공부하려는 경우:
  0 -> 2 -> 3 -> 7 -> 8 -> 9 -> 10 -> 10A -> 10B -> 10C -> 11 -> 19

실제 weight를 연결하려는 경우:
  9.8 -> 12 -> 13 -> 16

다음 C++/CUDA 개발을 이어가려는 경우:
  14 -> 15 -> 16 -> 18
```

## 목차

0. 초심자를 위한 준비 개념
1. 현재 목표
2. 기준 모델
3. PointPillars와 Sparse Voxel의 차이
4. 프로젝트 폴더 구조
5. 개발 환경
6. 전체 파이프라인 빠른 실행
7. 02단계: Voxelization / Pillarization
8. 03단계: Pillar Feature Decoration
9. 04단계: PFN
10. 05단계: Scatter to BEV
10A. 06단계: 전체 RPN CUDA
10B. 07단계: CenterHead CUDA
10C. 08단계: Box Decode와 Rotated NMS
11. 단계별 tensor 흐름
12. Weight 파일 선택 방법
13. Waymo 데이터 준비 상태
14. 왜 Python 결과와 비교하는가
15. CPU C++와 CUDA의 역할
16. 다음 구현 순서
17. 자주 발생하는 문제
18. 코드를 공부하는 추천 순서
19. 학습용 확인 문제
20. 현재 상태 요약

---

## 0. 초심자를 위한 준비 개념

### 0.1 Point cloud란 무엇인가

LiDAR는 카메라처럼 2차원 사진을 바로 만들지 않는다. 레이저가 반사되어 돌아온 위치를 점으로 기록한다.

KITTI의 실제 첫 point는 다음과 같다.

```text
[x, y, z, intensity]
[52.897940, 0.022990, 1.997995, 0.080000]
```

```text
x = 52.897940 m     차량 앞/뒤 방향 위치
y = 0.022990 m      차량 좌/우 방향 위치
z = 1.997995 m      높이
intensity = 0.08    레이저 반사 강도
```

Waymo는 여기에 `elongation`을 더해 point당 5개 feature를 사용한다.

### 0.2 Tensor와 shape 읽는 법

Tensor는 여러 숫자를 규칙적으로 정리한 배열이다. Shape는 각 방향으로 숫자가 몇 개 있는지를 나타낸다.

```text
[10404, 20, 4]

pillar가 10,404개
pillar 하나에 point 공간이 20개
point 하나에 feature가 4개
```

`pillars[3,5,0]`은 4번째 pillar의 6번째 point에서 x feature를 선택한다. 컴퓨터 배열 index는 0부터 시작한다.

### 0.3 Batch란 무엇인가

Batch는 여러 LiDAR frame을 한 번에 처리하기 위한 차원이다. 현재 sample은 한 frame만 처리하므로 batch index는 항상 `0`이다.

```text
coordinate = [batch, z, y, x]
              [  0,   0,234,399]
```

### 0.4 Binary와 metadata를 함께 저장하는 이유

`.bin` 파일에는 숫자만 연속해서 들어 있고 shape 정보는 없다. 그래서 JSON metadata가 필요하다.

```text
pillar_features.bin
  -> float 숫자들

pillar_features_metadata.json
  -> 숫자를 [10404, 64]로 읽으라는 설명
```

Binary를 잘못된 shape로 읽으면 프로그램은 실행되더라도 전혀 다른 tensor가 된다. 이 프로젝트의 reader가 binary 크기와 metadata를 함께 검사하는 이유다.

### 0.5 자주 쓰는 기호

```text
N = batch size
C = channel 수
H = height, Y 방향 grid 수
W = width, X 방향 grid 수
F = point feature 수
P = pillar 수
M = pillar당 최대 point 수
```

`BEV [1,64,468,468]`은 `NCHW` 순서이며 `N=1`, `C=64`, `H=468`, `W=468`이다.

---

## 1. 현재 목표

최종 목표는 다음 추론 흐름을 C++/CUDA에서 실행하는 것이다.

```text
LiDAR points
  -> Voxelization / Pillarization
  -> Pillar Feature Decoration
  -> PFN (Pillar Feature Network)
  -> Scatter to BEV
  -> 2D RPN Backbone
  -> CenterHead
  -> Decode
  -> Rotated NMS
  -> 3D Bounding Boxes
```

현재까지 구현된 범위는 다음과 같다.

```text
[완료] 02 Voxelization
[완료] 03 Pillar Feature Decoration
[완료] 04 실제 체크포인트 two-layer PFN
[완료] 05 Scatter
[완료] 06 전체 2D RPN CUDA
[완료] 07 CenterHead CUDA
[완료] 08 Decode CUDA / Rotated NMS C++
[완료] 09 Waymo derived sensor archive -> 5-feature point bin bridge
[완료] 09 Waymo 5프레임 전체 02~08 실행과 GT 평가
[완료] 10 CenterHead GT peak 독립 수치 검증
[완료] 11 raw/tanh 전처리 및 PFN~Head 단계 비교
[완료] 12 공식 CCW geometry 검증과 FN 원인 분석
[다음] 수백 프레임 threshold/거리/point-count 통계 평가
[다음] Waymo 공식 metric 입력과 평가 도구 연결
[다음] 단계 사이 Host-GPU 복사 제거와 kernel 최적화
```

`04_pfn_project`에는 학습용 dummy 경로와 실제 `centerpoint_waymo_pointpillars_50_novelocity.pth`의 two-layer PFN 경로가 함께 있다. 06과 07 역시 같은 체크포인트의 `neck.*`, `bbox_head.*` tensor를 추출해 PyTorch runtime 없이 실행한다.

---

## 2. 기준 모델

현재 C++ 구현의 기준이 되는 원본 config는 다음 파일이다.

```text
00_reference/centerpoint_original/CenterPoint-master/
  configs/waymo/pp/waymo_centerpoint_pp_two_pfn_stride1_3x.py
```

주요 설정은 다음과 같다.

| 항목 | 값 |
|---|---:|
| Dataset | Waymo |
| Detector | PointPillars + CenterHead |
| Classes | VEHICLE, PEDESTRIAN, CYCLIST |
| Point feature 수 | 5 |
| Voxel size | `[0.32, 0.32, 6.0]` |
| Point cloud range | `[-74.88, -74.88, -2, 74.88, 74.88, 4]` |
| Grid size | `[468, 468, 1]` |
| 최대 point/pillar | 20 |
| 최대 pillar 수 | 60000 |
| PFN filters | `[64, 64]` |
| Scatter 출력 | `[batch, 64, 468, 468]` |

Waymo point 한 개는 일반적으로 다음 5개 값을 사용한다.

```text
[x, y, z, intensity, elongation]
```

현재 단계별 검증에는 KITTI sample을 사용한다. KITTI sample은 다음 4개 값을 가진다.

```text
[x, y, z, intensity]
```

그래서 KITTI 검증에서는 decoration 이후 feature가 9차원이고, Waymo에서는 10차원이 된다.

---

## 3. PointPillars와 Sparse Voxel의 차이

우리가 구현하는 구조는 **PointPillars 기반 CenterPoint**다.

### PointPillars

XY 평면을 격자로 나누고, 각 칸을 Z축 전체를 포함하는 긴 기둥으로 취급한다.

```text
위에서 본 XY 평면

┌───┬───┬───┐
│   │ ● │   │
├───┼───┼───┤
│ ● │ ● │   │
├───┼───┼───┤
│   │   │ ● │
└───┴───┴───┘
```

각 pillar 안의 여러 point를 PFN이 하나의 64차원 feature로 압축한다. 그 후 Scatter를 통해 2D BEV tensor를 만든다.

### Sparse Voxel

XYZ 공간을 작은 3D voxel로 나누고, point가 있는 voxel만 Sparse 3D Convolution으로 처리한다.

```text
높은 voxel:  □ ■ □
중간 voxel:  ■ □ □
낮은 voxel:  ■ ■ □
```

Sparse Voxel 방식은 높이 구조를 더 직접적으로 표현하지만 `spconv`와 같은 sparse convolution 구현이 필요하다. 현재 프로젝트가 PointPillars부터 시작한 이유는 전처리와 tensor 흐름을 더 명확하게 학습할 수 있기 때문이다.

---

## 4. 프로젝트 폴더 구조

프로젝트 루트:

```text
C:\Users\user\Desktop\Onechip\Codex\my project
```

```text
my project/
├─ 00_reference/
│  ├─ centerpoint_original/       원본 Python CenterPoint
│  ├─ checkpoints/waymo/          다운로드한 체크포인트
│  └─ sample_data/kitti/          KITTI sample
├─ 01_rules/
│  └─ implementation_analysis.md  구현 방향 분석
├─ 02_project/                    CPU Voxelization
├─ 03_pillar_feature_project/     Feature Decoration
├─ 04_pfn_project/                Dummy + 실제 two-layer PFN
├─ 05_scatter_project/            Scatter to BEV
├─ 06_rpn_project/                전체 RPN CUDA
├─ 07_center_head_project/        CenterHead CUDA
├─ 08_decode_project/             Box Decode + Rotated NMS
├─ 09_full_pipeline_project/      Waymo 입력 bridge와 02~08 전체 실행
├─ 10_head_validation_project/    GT 위치의 CenterHead 출력 검증
├─ 11_reference_comparison_project/ raw/tanh 및 단계별 독립 비교
├─ 12_waymo_fn_analysis_project/  FN별 점군/heatmap/geometry 원인 분석
├─ 000_waymo_training_project/    Waymo 변환/학습 준비
└─ CENTERPOINT_CPP_STUDY_MANUAL.md
```

각 구현 프로젝트는 Visual Studio에서 보기 편하도록 다음 구조를 공통으로 사용한다.

```text
project/
├─ CMakeLists.txt
├─ README.md
├─ include/centerpoint/  헤더
├─ src/                  C++ 구현
├─ tools/                Python 기준 구현과 비교 도구
├─ dump/                 실행 결과
└─ build/                Visual Studio solution과 실행 파일
```

---

## 5. 개발 환경

### 필수 도구

```text
Windows 10/11
Visual Studio 2022 또는 Build Tools
C++ Desktop workload
CMake 3.18 이상
Python 3
NumPy
```

버전 확인:

```powershell
cmake --version
python --version
python -c "import numpy; print(numpy.__version__)"
```

### 공통 빌드 방법

각 프로젝트 폴더에서 다음 명령을 실행한다.

```powershell
cmake -S . -B build
cmake --build build --config Release
```

실행 파일은 다음 위치에 생성된다.

```text
build/Release/*.exe
```

Visual Studio에서는 `build` 폴더 안의 `.sln` 파일을 열면 된다.

---

## 6. 전체 파이프라인 빠른 실행

다음 명령들은 프로젝트 루트에서 차례대로 실행한다.

### 6.1 Voxelization

```powershell
cd "C:\Users\user\Desktop\Onechip\Codex\my project\02_project"

.\build\Release\centerpoint_voxel_dump.exe `
  "..\00_reference\sample_data\kitti\000000.bin" `
  ".\dump\kitti_000000" `
  4
```

### 6.2 Pillar Feature Decoration

```powershell
cd "C:\Users\user\Desktop\Onechip\Codex\my project\03_pillar_feature_project"

.\build\Release\centerpoint_decorate_pillars.exe `
  "..\02_project\dump\kitti_000000" `
  ".\dump\kitti_000000_decorated"
```

### 6.3 실제 체크포인트 PFN

```powershell
cd "C:\Users\user\Desktop\Onechip\Codex\my project\04_pfn_project"

.\build\Release\centerpoint_pfn_checkpoint.exe `
  "..\03_pillar_feature_project\dump\kitti_000000_decorated" `
  ".\weights\waymo_pointpillars_50_novelocity" `
  ".\dump\kitti_000000_checkpoint_pfn"
```

### 6.4 Scatter

```powershell
cd "C:\Users\user\Desktop\Onechip\Codex\my project\05_scatter_project"

.\build\Release\centerpoint_scatter.exe `
  "..\04_pfn_project\dump\kitti_000000_checkpoint_pfn" `
  "..\02_project\dump\kitti_000000" `
  ".\dump\kitti_000000_scatter"
```

최종 출력:

```text
05_scatter_project/dump/kitti_000000_scatter/
├─ bev_features.bin
└─ bev_features_metadata.json
```

### 6.5 전체 RPN CUDA

```powershell
cd "..\06_rpn_project"
.\build_cuda\Release\centerpoint_rpn_full_cuda.exe `
  "..\05_scatter_project\dump\kitti_000000_scatter" `
  ".\weights\waymo_pointpillars_50_novelocity" `
  ".\dump\kitti_000000_rpn"
```

### 6.6 CenterHead CUDA

```powershell
cd "..\07_center_head_project"
.\build_cuda\Release\centerpoint_head_cuda.exe `
  "..\06_rpn_project\dump\kitti_000000_rpn" `
  ".\weights" `
  ".\dump\kitti_000000_head"
```

### 6.7 Decode와 Rotated NMS

```powershell
cd "..\08_decode_project"
.\build_cuda\Release\centerpoint_decode.exe `
  "..\07_center_head_project\dump\kitti_000000_head" `
  ".\dump\kitti_000000_detections"
```

최종 결과는 `detections.csv`, `detections.bin`, `detections_metadata.json`이다.

---

## 7. 02단계: Voxelization / Pillarization

### 7.1 목적

LiDAR는 point 개수가 프레임마다 다르다. Neural Network에 입력하려면 point들을 일정한 XY 격자에 모아 구조화해야 한다.

입력 point:

```text
point = [x, y, z, feature...]
```

각 point의 voxel 좌표는 다음과 같이 계산한다.

```text
voxel_x = floor((x - x_min) / voxel_size_x)
voxel_y = floor((y - y_min) / voxel_size_y)
voxel_z = floor((z - z_min) / voxel_size_z)
```

Waymo 설정에서는 Z voxel 크기가 전체 높이 범위와 같은 `6.0`이므로 `grid_z=1`이다. 즉 사실상 3D voxel이 아니라 XY pillar가 된다.

### 7.2 범위 검사

다음 조건을 만족하지 않는 point는 버린다.

```text
x_min <= x < x_max
y_min <= y < y_max
z_min <= z < z_max
```

최댓값을 `<`로 검사하는 이유는 `x == x_max`이면 계산된 voxel index가 grid 밖이 되기 때문이다.

### 7.3 제한

```text
pillar 한 개에 최대 20 points
전체 최대 60000 pillars
```

pillar에 20개보다 많은 point가 들어오면 뒤의 point는 저장하지 않는다. 새로운 pillar가 60000개를 넘으면 더 이상 생성하지 않는다.

### 7.4 출력 tensor

```text
pillars.bin
  float32 [num_pillars, max_points, feature_dim]

coordinates.bin
  int32 [num_pillars, 4]
  order = [batch, z, y, x]

num_points.bin
  int32 [num_pillars]
```

KITTI sample 결과:

```text
입력 points: 124,668
생성 pillars: 10,404
pillar tensor: [10404, 20, 4]
```

### 7.5 실제 point가 pillar 좌표로 바뀌는 예

KITTI 파일의 첫 point:

```text
point[0] = [52.897940, 0.022990, 1.997995, 0.080000]
```

각 좌표의 grid index를 계산한다.

```text
x index = floor((52.897940 - (-74.88)) / 0.32) = 399
y index = floor(( 0.022990 - (-74.88)) / 0.32) = 234
z index = floor(( 1.997995 - ( -2.00)) / 6.00) =   0
```

계산은 `x,y,z = 399,234,0` 순서지만 저장 좌표는 원본 CenterPoint 규칙에 맞춰 다음처럼 기록한다.

```text
coordinates[0] = [batch, z, y, x]
               = [0, 0, 234, 399]
```

첫 pillar에는 실제 point가 한 개 들어 있다.

```text
num_points[0] = 1

pillars[0,0] =
[52.897940, 0.022990, 1.997995, 0.080000]

pillars[0,1]부터 pillars[0,19]까지 =
[0, 0, 0, 0]
```

### 7.6 중요한 flat memory index

C++의 `std::vector<float>`는 1차원이다. `[pillar, point, feature]`의 offset은 다음과 같다.

```cpp
offset = (pillar * max_points + point) * feature_dim + feature;
```

이 식을 이해하면 이후 모든 tensor 코드의 indexing을 쉽게 읽을 수 있다.

### 7.7 출력 파일을 Python에서 직접 보는 방법

```python
import json
import numpy as np

metadata = json.load(open("dump/kitti_000000/metadata.json"))
pillars = np.fromfile(
    "dump/kitti_000000/pillars.bin", dtype=np.float32
).reshape(
    metadata["num_pillars"],
    metadata["max_points_per_pillar"],
    metadata["feature_dim"],
)
coordinates = np.fromfile(
    "dump/kitti_000000/coordinates.bin", dtype=np.int32
).reshape(-1, 4)

print(coordinates[0])
print(pillars[0, 0])
```

예상 출력:

```text
[  0   0 234 399]
[52.89794  0.02299  1.997995 0.08]
```

### 7.8 Python 비교

```powershell
cd "C:\Users\user\Desktop\Onechip\Codex\my project\02_project"

python .\tools\compare_python_cpp_voxelization.py `
  --points "..\00_reference\sample_data\kitti\000000.bin" `
  --cpp-dump ".\dump\kitti_000000" `
  --feature-dim 4
```

검증 결과:

```text
coordinates equal: True
num_points equal: True
pillars equal: True
max abs diff: 0
```

---

## 8. 03단계: Pillar Feature Decoration

### 8.1 목적

raw point의 절대좌표만으로는 point가 pillar 안에서 어디에 모여 있는지 알기 어렵다. 그래서 각 point에 상대 위치 정보를 추가한다.

### 8.2 추가 feature

입력 feature를 `F`개라고 하면 출력은 `F+5`개다.

```text
원본 features
+ cluster offset 3개
+ pillar center offset 2개
```

KITTI:

```text
[x, y, z, intensity]                    4개
[x-mean_x, y-mean_y, z-mean_z]          3개
[x-center_x, y-center_y]                 2개
합계                                      9개
```

Waymo:

```text
[x, y, z, intensity, elongation]         5개
+ decoration                             5개
합계                                     10개
```

### 8.3 Cluster offset

pillar 안의 유효 point가 `N`개일 때 평균은 다음과 같다.

```text
mean_x = sum(x_i) / N
mean_y = sum(y_i) / N
mean_z = sum(z_i) / N
```

각 point에는 다음 값을 붙인다.

```text
f_cluster = [x-mean_x, y-mean_y, z-mean_z]
```

이 feature는 point가 같은 pillar 안에서 point 군집의 중심보다 어느 방향에 있는지 표현한다.

### 8.4 Pillar center offset

pillar의 실제 중심 좌표는 다음과 같다.

```text
x_offset = voxel_size_x / 2 + x_min
y_offset = voxel_size_y / 2 + y_min

center_x = coord_x * voxel_size_x + x_offset
center_y = coord_y * voxel_size_y + y_offset
```

각 point에는 다음 값을 붙인다.

```text
f_center = [x-center_x, y-center_y]
```

### 8.5 Padding 처리

pillar는 항상 20개 point 공간을 가지지만 실제 point는 더 적을 수 있다. 유효하지 않은 padding row는 모든 feature를 0으로 유지해야 한다.

### 8.6 Shape 변화

```text
KITTI: [10404, 20, 4]
                -> [10404, 20, 9]
```

### 8.7 실제 네 point의 decoration 예

14번 pillar에는 유효 point가 4개 있다.

```text
coordinates[14] = [0, 0, 271, 431]
num_points[14] = 4
```

원본 point:

```text
[63.340088, 12.013950, 2.387444, 0.00]
[63.302227, 11.926991, 1.976460, 0.06]
[63.298840, 12.132906, 1.977420, 0.00]
[63.282320, 11.868020, 1.672471, 0.00]
```

네 point의 XYZ 평균:

```text
mean_xyz = [63.305870, 11.985466, 2.003449]
```

첫 point의 cluster offset:

```text
x - mean_x = 63.340088 - 63.305870 = 0.034218
y - mean_y = 12.013950 - 11.985466 = 0.028484
z - mean_z =  2.387444 -  2.003449 = 0.383995
```

첫 point의 최종 decorated feature:

```text
[63.340088, 12.013950, 2.387444, 0.000000,
  0.034218,  0.028484, 0.383995,
  0.140083,  0.013943]
```

```text
앞 4개    = 원본 x, y, z, intensity
다음 3개  = cluster 중심으로부터의 차이
마지막 2개 = pillar XY 중심으로부터의 차이
```

이 pillar에는 실제 point가 4개뿐이므로 다섯 번째 row부터는 padding이다.

```text
decorated[14,4] = [0,0,0,0,0,0,0,0,0]
```

### 8.8 Python 비교

```powershell
cd "C:\Users\user\Desktop\Onechip\Codex\my project\03_pillar_feature_project"

python .\tools\compare_python_cpp_pillar_feature.py `
  --voxel-dump "..\02_project\dump\kitti_000000" `
  --decorated-dump ".\dump\kitti_000000_decorated"
```

검증 결과:

```text
shape equal: True
decorated values equal: True
max abs diff: 0
```

---

## 9. 04단계: PFN

### 9.1 PFN의 역할

pillar마다 point 수가 다르지만, Scatter에 넣으려면 pillar 하나를 고정 길이 vector 하나로 만들어야 한다.

```text
한 pillar의 points: [20, input_channels]
PFN 출력:           [64]
```

현재 dummy PFN의 연산은 다음과 같다.

```text
Linear
  -> BatchNorm inference
  -> ReLU
  -> point 방향 Max Pooling
```

### 9.2 Linear

각 point feature `x`와 weight `W`를 곱한다.

```text
y[out] = sum(x[in] * W[out, in])
```

현재 weight는 checkpoint에서 읽은 값이 아니라 Python과 C++의 구조 비교를 위한 deterministic dummy 값이다.

```text
W[out,in] = ((((out+1)*(in+3)) mod 17) - 8) * 0.01
```

### 9.3 BatchNorm inference

학습이 끝난 BatchNorm은 저장된 통계를 사용한다.

```text
normalized = (x - running_mean) / sqrt(running_var + eps)
output = normalized * gamma + beta
```

현재 dummy 설정:

```text
gamma = 1
beta = 0
running_mean = 0
running_var = 1
eps = 1e-3
```

### 9.4 ReLU

```text
ReLU(x) = max(0, x)
```

음수 feature를 0으로 만든다.

### 9.5 Max Pooling

pillar 안의 모든 유효 point에 대해 channel별 최댓값을 선택한다.

```text
pillar_feature[channel]
  = max(point_feature[0][channel], ..., point_feature[N-1][channel])
```

이 과정으로 point 순서와 관계없이 pillar 하나가 64차원 vector 하나로 압축된다.

### 9.6 현재 shape

```text
[10404, 20, 9]
  -> Linear/BN/ReLU
  -> Max Pool
[10404, 64]
```

### 9.7 실제 PFN 출력 예

첫 pillar의 9차원 decorated feature가 dummy Linear, BN, ReLU, Max Pooling을 지나면 64차원 vector가 된다. 앞의 16개 값은 다음과 같다.

```text
pillar_features[0, 0:16] =
[0.000000, 0.000000, 0.649769, 2.014551,
 3.704370, 0.000000, 0.000000, 0.000000,
 1.105608, 2.799333, 4.164114, 0.000000,
 0.000000, 0.000000, 1.554807, 3.259078]
```

`0`이 많은 것은 ReLU가 음수 출력을 0으로 바꾸었기 때문이다. 이 숫자들은 학습된 객체 특징이 아니라 현재 deterministic dummy weight로 만들어진 구조 검증용 값이다.

### 9.8 원본 Waymo PFN과 현재 구현의 차이

원본 config는 PFN이 두 층이다.

```text
입력: 10

PFN layer 0:
  Linear [32, 10]
  BN [32]
  ReLU
  Max [32]
  point feature와 max feature를 concat
  출력 [point, 64]

PFN layer 1:
  Linear [64, 64]
  BN [64]
  ReLU
  Max
  출력 [pillar, 64]
```

현재 `04_pfn_project`는 위 구조의 실제 two-layer PFN을 구현했다. 체크포인트 weight를 직접 읽는 대신 Python exporter가 tensor별 float32 binary와 metadata를 만들고 C++ reader가 shape를 검증한다. KITTI 입력은 feature가 4개라 Waymo의 `elongation` 위치를 0으로 채워 10차원 decoration으로 맞춘다.

### 9.9 Python 비교

```powershell
cd "C:\Users\user\Desktop\Onechip\Codex\my project\04_pfn_project"

python .\tools\compare_python_cpp_pfn_dummy.py `
  --decorated-dump "..\03_pillar_feature_project\dump\kitti_000000_decorated" `
  --pfn-dump ".\dump\kitti_000000_pfn"
```

검증 결과:

```text
Python PFN shape: [10404, 64]
C++ PFN shape:    [10404, 64]
allclose: True
max abs diff: 약 0.00000191
```

부동소수점 덧셈 순서 때문에 아주 작은 차이가 있지만 허용 오차 안에서 동일하다.

---

## 10. 05단계: Scatter to BEV

### 10.1 목적

PFN 출력은 pillar 목록이다.

```text
pillar_features: [num_pillars, 64]
coordinates:     [num_pillars, 4]
```

2D CNN은 고정 크기 image 형태가 필요하다. Scatter는 각 pillar feature를 원래 XY 좌표로 되돌려 dense BEV tensor를 만든다.

### 10.2 핵심 연산

좌표가 다음과 같다고 하자.

```text
coordinate = [batch, z, y, x]
```

복사 연산은 다음 한 줄로 표현할 수 있다.

```text
BEV[batch, :, y, x] = pillar_features[pillar, :]
```

좌표가 없는 cell은 0이다.

### 10.3 NCHW layout

출력 layout은 일반적인 convolution 입력 형식인 NCHW다.

```text
N = batch
C = channels
H = grid_y
W = grid_x
```

Waymo 기준:

```text
[1, 64, 468, 468]
```

### 10.4 C++ flat index

`BEV[batch, channel, y, x]`의 1차원 offset은 다음과 같다.

```cpp
offset = ((batch * channels + channel) * height + y) * width + x;
```

현재 구현은 다음 오류도 검사한다.

```text
좌표가 grid 밖인지
batch index가 음수인지
동일한 XY 좌표가 중복되는지
PFN pillar 수와 coordinate pillar 수가 같은지
binary 크기가 metadata와 일치하는지
```

### 10.5 아주 작은 Scatter 예제

실제 `468 x 468` grid를 보기 전에 `3 x 4` grid를 생각하면 쉽다.

```text
channels = 2
grid height = 3
grid width = 4

pillar feature = [10, 20]
coordinate = [0, 0, 1, 2]
```

이 feature는 batch 0의 `y=1, x=2`에 들어간다.

```text
channel 0
0  0  0  0
0  0 10  0
0  0  0  0

channel 1
0  0  0  0
0  0 20  0
0  0  0  0
```

Scatter는 `[10,20]`을 계산하거나 변경하지 않고 위치만 정한다.

### 10.6 실제 프로젝트 Scatter 예

앞에서 본 첫 pillar의 정보는 다음과 같다.

```text
coordinate = [0, 0, 234, 399]

pillar_features[0, 0:16] =
[0.000000, 0.000000, 0.649769, 2.014551,
 3.704370, 0.000000, 0.000000, 0.000000,
 1.105608, 2.799333, 4.164114, 0.000000,
 0.000000, 0.000000, 1.554807, 3.259078]
```

Scatter 이후 같은 값이 다음 위치에 그대로 들어간다.

```text
BEV[0, 0:16, 234, 399] =
[0.000000, 0.000000, 0.649769, 2.014551,
 3.704370, 0.000000, 0.000000, 0.000000,
 1.105608, 2.799333, 4.164114, 0.000000,
 0.000000, 0.000000, 1.554807, 3.259078]
```

pillar가 없는 바로 옆 cell은 0이다.

```text
BEV[0, 0:8, 234, 400] = [0,0,0,0,0,0,0,0]
```

PFN 출력값은 바뀌지 않고 XY 위치만 부여된다는 것을 숫자로 확인할 수 있다.

### 10.7 현재 전체 결과

```text
입력: [10404, 64]
출력: [1, 64, 468, 468]
occupied cells: 10404
출력 binary 크기: 56,070,144 bytes
```

### 10.8 Python 비교

```powershell
cd "C:\Users\user\Desktop\Onechip\Codex\my project\05_scatter_project"

python .\tools\compare_python_cpp_scatter.py `
  --pfn-dump "..\04_pfn_project\dump\kitti_000000_pfn" `
  --voxel-dump "..\02_project\dump\kitti_000000" `
  --scatter-dump ".\dump\kitti_000000_scatter"
```

검증 결과:

```text
Python BEV shape:  [1, 64, 468, 468]
C++ BEV shape:     [1, 64, 468, 468]
exactly equal: True
max abs diff: 0
```

Scatter는 값을 계산하지 않고 복사하므로 Python과 C++ 결과가 bitwise exact하게 일치한다.

---

## 10A. 06단계: 전체 RPN CUDA

RPN은 Scatter의 `[1,64,468,468]` BEV를 2D convolution으로 처리해 더 넓은 주변 문맥을 학습된 feature로 만든다.

```text
Block 0 [64,468,468]  -> Deblock [128,468,468]
Block 1 [128,234,234] -> Deblock [128,468,468]
Block 2 [256,117,117] -> Deblock [128,468,468]
Channel concat         -> [384,468,468]
```

일반 convolution은 `im2col CUDA kernel -> cuBLAS SGEMM -> BN/ReLU CUDA kernel` 순서다. Downsample은 stride 2 convolution, upsample은 transposed convolution으로 구현했다. 체크포인트의 `neck.*` 95개 float tensor를 사용하며 PyTorch runtime은 필요 없다.

실제 검증에서 출력은 `[1,384,468,468]`, 반복 SHA-256은 동일했고 선택한 CPU scalar 기준값의 최대 오차는 약 `2.21e-6`이었다. 자세한 코드는 `06_rpn_project/FULL_RPN_IMPLEMENTATION_GUIDE.md`에서 읽는다.

---

## 10B. 07단계: CenterHead CUDA

CenterHead는 RPN feature를 최종 box 구성 요소별 raw prediction map으로 바꾼다.

```text
입력 [1,384,468,468]
  -> Shared Conv 384 -> 64 + BN + ReLU
  -> 다섯 개 독립 branch

reg     [1,2,468,468]  중심의 cell 내부 offset
height  [1,1,468,468]  box 중심 높이
dim     [1,3,468,468]  log 공간의 dx,dy,dz
rot     [1,2,468,468]  sin/cos 회전 표현
hm      [1,3,468,468]  세 클래스 heatmap logit
```

각 branch는 `Conv 64->64 + BN + ReLU -> Conv 64->출력 채널`이다. 이 체크포인트는 `novelocity` 모델이므로 velocity branch가 없다. 선택 지점 15개의 CPU reference 최대 오차는 `4.768e-7`이며 모든 출력에 non-finite 값이 없었다.

중요한 점은 이 출력이 아직 box가 아니라는 것이다. 예를 들어 `hm=-3.0`은 확률이 아니며 다음 단계에서 sigmoid가 필요하다. 실제 코드 설명은 `07_center_head_project/IMPLEMENTATION_GUIDE.md`를 본다.

---

## 10C. 08단계: Box Decode와 Rotated NMS

CUDA decode kernel은 feature-map의 각 cell을 독립적으로 처리한다.

```text
score = sigmoid(max(hm_logits))
x = (grid_x + reg_x) * 0.32 - 74.88
y = (grid_y + reg_y) * 0.32 - 74.88
z = height
(dx,dy,dz) = exp(dim)
yaw = atan2(rot_sin, rot_cos)
```

그 뒤 score `0.1` 이하와 공간 범위 밖 후보를 제거한다. C++ rotated NMS는 점수순 최대 4096개에 대해 회전 사각형의 polygon intersection과 BEV IoU를 계산하고 IoU `0.7` 초과 중복을 억제한다. 최종 출력은 최대 500개다.

```text
detections.csv 한 행:
x,y,z,dx,dy,dz,yaw,score,label,source_index

label 0 = VEHICLE
label 1 = PEDESTRIAN
label 2 = CYCLIST
```

KITTI 샘플을 Waymo weight에 넣은 구현 검증에서는 후보 1380개와 최종 500개가 생성됐다. 독립 NumPy/Python decode와 rotated NMS의 인덱스·순서·label이 모두 일치했고 최대 수치 차이는 `6.636e-6`이었다. 500개가 나왔다는 사실은 정확도가 좋다는 의미가 아니다. 데이터셋이 다르므로 실제 품질 평가는 Waymo frame과 annotation으로 해야 한다.

---

## 11. 단계별 tensor 흐름

현재 KITTI sample 기준:

```text
000000.bin
  [124668, 4]

Voxelization
  pillars       [10404, 20, 4]
  coordinates   [10404, 4]
  num_points    [10404]

Decoration
  decorated     [10404, 20, 9]

Checkpoint two-layer PFN
  pillar feature [10404, 64]

Scatter
  BEV            [1, 64, 468, 468]

RPN
  feature        [1, 384, 468, 468]

CenterHead
  reg            [1, 2, 468, 468]
  height         [1, 1, 468, 468]
  dim            [1, 3, 468, 468]
  rot            [1, 2, 468, 468]
  hm             [1, 3, 468, 468]

Decode + NMS
  detections     [num_detections, 9]
```

Waymo 실제 입력을 사용할 때 예상 흐름:

```text
Waymo points
  [num_points, 5]

Voxelization
  [num_pillars, 20, 5]

Decoration
  [num_pillars, 20, 10]

Two-layer PFN
  [num_pillars, 64]

Scatter
  [1, 64, 468, 468]

RPN
  [1, 384, 468, 468]

CenterHead
  raw maps [2,1,3,2,3] x [468,468]

Decode + NMS
  [num_detections, x,y,z,dx,dy,dz,yaw,score,label]
```

---

## 12. Weight 파일 선택 방법

### 12.1 정확히 필요한 체크포인트

다음 config로 학습된 **한 개의 전체 `.pth` 파일**이 필요하다.

```text
configs/waymo/pp/waymo_centerpoint_pp_two_pfn_stride1_3x.py
```

검색하거나 요청할 때 사용할 표현:

```text
CenterPoint Waymo PointPillars
one-stage, three-class
two PFN layers
config: waymo_centerpoint_pp_two_pfn_stride1_3x.py
```

체크포인트에는 다음 top-level weight가 있어야 한다.

```text
reader.pfn_layers.*
neck.*
bbox_head.*
```

Scatter는 학습 parameter가 없으므로 `backbone` 또는 Scatter weight가 따로 필요하지 않다.

### 12.2 필요한 PFN shape

```text
reader.pfn_layers.0.linear.weight       [32, 10]
reader.pfn_layers.0.norm.weight         [32]
reader.pfn_layers.0.norm.bias           [32]
reader.pfn_layers.0.norm.running_mean   [32]
reader.pfn_layers.0.norm.running_var    [32]

reader.pfn_layers.1.linear.weight       [64, 64]
reader.pfn_layers.1.norm.weight         [64]
reader.pfn_layers.1.norm.bias           [64]
reader.pfn_layers.1.norm.running_mean   [64]
reader.pfn_layers.1.norm.running_var    [64]
```

### 12.3 현재 보유한 체크포인트 판정

#### `centerpoint_waymo_50.pth`

```text
크기: 약 93.36 MB
내부 archive: epoch_12
parameter prefix: backbone, neck, bbox_head
PFN parameter: 없음
```

이 파일은 Sparse Voxel 기반 CenterPoint로 판단된다. 현재 PointPillars PFN에 사용할 수 없다.

#### `pointpillar_7728.pth`

```text
크기: 약 19.38 MB
epoch: 80
iteration: 9280
PFN Linear: [64, 10]
backbone_2d 포함
dense_head 포함
```

PointPillars 체크포인트는 맞지만 PFN이 한 층이고 `dense_head.conv_cls`를 사용하는 Anchor Head 모델이다. 현재 목표인 two-PFN CenterHead 전체 모델과는 다르다.

PFN 연산을 공부하거나 weight 추출을 연습하는 참고 자료로는 사용할 수 있지만, 전체 모델 weight로 바로 연결하면 안 된다.

---

## 13. Waymo 데이터 준비 상태

현재 Windows C++/CUDA 추론과 검증에 실제 사용한 derived archive:

```text
E:\Waymo_datset\derived_v1_4_3\sensor_archives\train\
  segment-10017090168044687777_6380_000_6400_000_with_camera_labels.zip
```

이 archive는 frame별 6-feature LiDAR bin과 `laser_labels.json`을 이미 포함한다.
09 exporter가 `nlz_flag`를 제외하고 intensity에 `tanh`를 적용해 모델의
5-feature 입력을 만든다. 따라서 아래 TFRecord 변환 과정은 **재학습 데이터나
원본 framework 실행을 준비할 때** 필요한 별도 흐름이다.

현재 training TFRecord:

```text
C:\Users\user\Desktop\Onechip\archived_files_training_training_0000
```

확인된 파일:

```text
27 TFRecord files
총 약 26.84 GB
```

CenterPoint용 dataset root:

```text
C:\Users\user\Desktop\Onechip\Waymo_CenterPoint
```

연결 상태:

```text
Waymo_CenterPoint\tfrecord_training
  -> archived_files_training_training_0000

CenterPoint-master\data\Waymo
  -> Waymo_CenterPoint
```

### 필요한 데이터 변환

원본 Waymo TFRecord를 CenterPoint가 직접 읽지는 않는다. 다음 변환이 필요하다.

```text
TFRecord segment
  -> frame별 lidar/*.pkl
  -> frame별 annos/*.pkl
  -> infos_train_01sweeps_filter_zero_gt.pkl
  -> ground-truth database
```

원본 명령:

```bash
python det3d/datasets/waymo/waymo_converter.py \
  --record_path 'data/Waymo/tfrecord_training/*.tfrecord' \
  --root_path 'data/Waymo/train/'

python tools/create_data.py waymo_data_prep \
  --root_path=data/Waymo \
  --split train \
  --nsweeps=1
```

### Windows 주의점

원본 CenterPoint가 요구하는 오래된 Waymo devkit은 Windows pip에서 설치되지 않았다.

```text
waymo-open-dataset-tf-1-15-0==1.2.0
```

따라서 데이터 변환과 학습은 다음 환경이 현실적이다.

```text
WSL2 Ubuntu
Ubuntu/Linux PC
Docker 기반 Linux 환경
```

C++ 전처리와 추론 구현은 Windows에서 계속 진행할 수 있다.

---

## 14. 왜 Python 결과와 비교하는가

C++로 다시 작성할 때 가장 위험한 부분은 문법이 아니라 다음과 같은 작은 규칙 차이다.

```text
좌표 순서 xyz 또는 zyx
range 경계에서 < 또는 <=
NCHW 또는 NHWC
padding mask 적용 시점
BatchNorm epsilon
weight matrix transpose
pillar 생성 순서
float 연산 순서
```

그래서 각 단계마다 다음 절차를 사용한다.

```text
1. 같은 binary 입력을 Python과 C++에 넣는다.
2. 중간 tensor를 binary로 저장한다.
3. shape를 비교한다.
4. coordinate와 count를 비교한다.
5. 모든 float 값의 최대 절대 오차를 계산한다.
6. 차이가 있으면 최초 mismatch index를 확인한다.
```

이 방식의 장점은 최종 bounding box가 이상할 때 전체 모델을 한 번에 디버깅하지 않아도 된다는 것이다.

---

## 15. CPU C++와 CUDA의 역할

C++로 작성했다고 자동으로 빨라지는 것은 아니다. 현재 CPU 구현의 가장 중요한 목적은 다음과 같다.

```text
원본 동작을 명확하게 이해한다.
memory layout을 확정한다.
Python과 정확히 비교한다.
CUDA kernel의 기준 결과를 만든다.
```

현재 역할 분담은 다음과 같다.

```text
CPU C++: Voxelization, Decoration, PFN, Scatter, weight/file I/O
CUDA: RPN convolution, CenterHead convolution, box decode
cuBLAS: RPN/CenterHead의 matrix multiplication
CPU C++: 후보 정렬과 rotated polygon IoU NMS
```

현재 구현은 정확성 검증을 위해 단계마다 binary를 저장하며 Host-GPU 복사가 존재한다. 최종 최적화에서는 PFN/Scatter도 CUDA로 옮기고 RPN 출력 device pointer를 CenterHead와 Decode에 직접 전달하는 것이 핵심이다. 후보가 최대 4096개인 NMS는 먼저 CPU 정확성을 유지하고, 전체 병목 측정 후 GPU 이전 여부를 판단한다.

권장 순서는 항상 다음과 같다.

```text
정확한 CPU 기준 구현
  -> Python 비교 통과
  -> CUDA 구현
  -> CPU/CUDA 비교
  -> 성능 측정
```

---

## 16. 구현 완료 기록과 다음 순서

### 완료 1: 실제 two-layer PFN

```text
Waymo 10차원 decorated input 지원
PFN layer 0: 10 -> 32 -> concat -> 64
PFN layer 1: 64 -> 64 -> max
실제 BN parameter loader
Python 원본과 비교
```

### 완료 2: Checkpoint weight 추출

`.pth` ZIP archive를 읽는 경량 Python exporter로 다음 독립 binary와 JSON을 생성한다.

```text
weight manifest JSON
tensor name
shape
dtype
raw float data
```

C++에서는 PyTorch 자체를 링크하지 않고 변환된 weight를 읽는 방식이 단순하다.

### 완료 3: 2D RPN Backbone

Scatter 출력 `[1,64,468,468]`을 입력으로 받아 Conv2D/BN/ReLU/downsample/upsample을 수행한다.

TensorRT 없이 직접 작성한 CUDA im2col kernel, cuBLAS SGEMM, BN/ReLU, transposed convolution으로 구현했다.

### 완료 4: CenterHead

세 클래스에 대해 다음 출력을 만든다.

```text
heatmap
reg offset
height
dimensions
rotation sin/cos
```

### 완료 5: Decode

```text
score = sigmoid(heatmap)
dimensions = exp(dimensions)
rotation = atan2(rot_sin, rot_cos)
x = (grid_x + reg_x) * out_size_factor * voxel_x + x_min
y = (grid_y + reg_y) * out_size_factor * voxel_y + y_min
```

### 완료 6: Rotated NMS

score가 낮거나 range 밖인 box를 제거하고, 겹치는 rotated box를 억제한다.

Waymo config 기준:

```text
score threshold: 0.1
NMS pre max: 4096
NMS post max: 500
NMS IoU threshold: 0.7
```

### 완료 7: Waymo derived sensor archive 입력 bridge

`E:\Waymo_datset\derived_v1_4_3\sensor_archives` 아래의 segment zip은 이미 frame별 lidar bin을 포함한다.

```text
frame_000/lidar/TOP_return1.bin
schema: [x, y, z, intensity, elongation, nlz_flag]
```

`09_full_pipeline_project/tools/export_waymo_frame.py`는 zip 안의 lidar bin을 읽고 마지막 `nlz_flag`를 제외해 현재 C++ CenterPoint 입력인 5-feature point bin을 만든다.

```text
[x, y, z, intensity, elongation]
```

첫 검증 frame은 `153830`개 point를 만들었고 C++ `waymo_frame_inspect`가 같은 min/max/mean과 sample 값을 읽었다.

주의할 점은 C++17 표준 라이브러리에는 zip reader가 없다는 것이다. 그래서 현재는 Python exporter가 zip을 풀고, C++는 추론 파이프라인 입력과 같은 raw float32 bin을 읽는다. 순수 C++ zip 직접 읽기는 나중에 miniz/libzip/libarchive 같은 의존성을 붙여 확장할 수 있다.

### 완료 8: Waymo 전체 파이프라인과 GT 평가

```text
derived sensor archive zip 읽기
09 exporter로 [x,y,z,intensity,elongation] 저장
02 voxelization feature_dim=5로 실행
03~08 단계를 순서대로 연결
detections.csv 생성
```

09 runner가 위 과정을 5프레임에 자동 적용하고, 같은 class의 rotated BEV IoU
`>= 0.5`로 TP/FP/FN을 계산한다. Waymo GT는 공식 CCW heading을 사용하고,
CenterPoint prediction은 `-yaw - pi/2`로 변환한다.

### 완료 9: 원본 전처리와 단계별 독립 비교

원본 loader의 `tanh(intensity)`를 적용하자 5프레임 recall이 `0.3243`에서
`0.6757`로 상승했다. PFN, Scatter, RPN, CenterHead는 독립 NumPy 계산과
각각 허용 오차 안에서 일치한다.

### 완료 10: False Negative 원인 분석

수정된 공식 geometry 기준 결과는 `TP=25`, `FP=3`, `FN=12`다. 남은 FN은
`LOW_MODEL_SCORE` 9개와 `LOW_POINT_COUNT` 3개로 분류됐다. 상세 입력 점 수,
heatmap score, IoU와 BEV 그림은 `12_waymo_fn_analysis_project`에서 확인한다.

### 다음 1: 대규모 통계와 공식 평가

수백 프레임에서 score threshold, 거리, GT 내부 point count 구간별
precision/recall을 계산한 뒤 Waymo 공식 metric 형식으로 확장한다.

### 다음 2: 단일 GPU 파이프라인 최적화

단계별 dump를 없애고 `PFN -> Scatter -> RPN -> CenterHead -> Decode`를 device memory에서 직접 연결한다. 현재 측정값은 학습용 독립 실행기의 시간이며 최종 실시간 성능으로 해석하면 안 된다.

---

## 17. 자주 발생하는 문제

### `point cloud float count is not divisible by feature_dim`

`.bin` 파일의 실제 feature 수와 실행 argument가 다르다.

```text
KITTI: feature_dim=4
Waymo 변환 결과: 일반적으로 feature_dim=5
```

### PFN과 coordinate pillar 수가 다름

서로 다른 실행 결과 폴더를 섞어 사용했을 가능성이 높다. 같은 voxelization 결과에서 이어진 dump를 사용해야 한다.

### BEV 파일이 매우 큼

`[1,64,468,468]` float32 tensor는 약 56 MB가 맞다.

```text
1 * 64 * 468 * 468 * 4 bytes = 56,070,144 bytes
```

### `.pth` 파일 이름은 PointPillar인데 연결되지 않음

파일 이름보다 내부 architecture가 중요하다. 다음을 반드시 확인한다.

```text
PFN layer 수
PFN tensor shape
CenterHead 또는 Anchor Head
Waymo 또는 KITTI
class 수
voxel size와 range
```

### CMake build 후 `pwsh.exe` 경고

실행 파일이 정상 생성되었다면 vcpkg 후처리 단계의 환경 경고일 수 있다. 먼저 다음 파일이 존재하는지 확인한다.

```text
build/Release/<target>.exe
```

---

## 18. 코드를 공부하는 추천 순서

각 프로젝트에서 다음 순서로 읽으면 흐름을 놓치지 않는다.

```text
1. README.md
2. CMakeLists.txt
3. include/centerpoint/types.hpp
4. src/main.cpp
5. 핵심 연산 cpp
6. src/io/*
7. tools/compare_*.py
```

특히 다음 파일이 핵심이다.

```text
02_project/src/voxelization.cpp
03_pillar_feature_project/src/pillar_feature.cpp
04_pfn_project/src/pfn.cpp
05_scatter_project/src/scatter.cpp
```

원본 Python과 비교해서 읽을 파일:

```text
det3d/ops/point_cloud/point_cloud_ops.py
det3d/models/readers/pillar_encoder.py
det3d/models/necks/rpn.py
det3d/models/bbox_heads/center_head.py
```

---

## 19. 학습용 확인 문제

### Voxelization

1. `x == x_max`인 point를 제외해야 하는 이유는 무엇인가?
2. coordinate를 `[batch,z,y,x]` 순서로 저장하는 이유는 무엇인가?
3. pillar당 point가 20개를 넘으면 현재 코드는 어떤 point를 남기는가?
4. point 순서가 바뀌면 pillar 생성 순서가 바뀔 수 있는가?

### Decoration

1. `f_cluster`와 `f_center`는 각각 어떤 정보를 표현하는가?
2. KITTI는 왜 9차원이고 Waymo는 왜 10차원인가?
3. padding feature를 0으로 다시 mask하지 않으면 어떤 문제가 생기는가?

### PFN

1. Max Pooling을 사용하면 point 순서에 대해 어떤 성질을 얻는가?
2. BatchNorm의 training 연산과 inference 연산은 무엇이 다른가?
3. 첫 PFN layer에서 local feature와 max feature를 concat하는 이유는 무엇인가?
4. PyTorch `Linear` weight shape가 `[out,in]`인 점이 C++ 구현에 어떤 영향을 주는가?

### Scatter

1. Scatter에 학습 weight가 필요하지 않은 이유는 무엇인가?
2. `grid_x`가 width이고 `grid_y`가 height인 이유는 무엇인가?
3. `NCHW` flat offset 식을 직접 유도할 수 있는가?
4. Scatter 전후에 실제 feature 값이 변하는가?

---

## 20. 현재 상태 요약

현재 프로젝트는 LiDAR point cloud부터 최종 3D box 파일까지의 추론 연산을 독립된 C++/CUDA 프로그램으로 나누어 구현했다.

```text
Voxelization: Python과 exact match
Decoration:   Python과 exact match
Checkpoint PFN: NumPy와 allclose, 최대 오차 약 3.34e-6
Scatter:      Python과 bitwise exact match
RPN CUDA:     선택 CPU scalar reference 통과, 반복 hash 일치
CenterHead:   선택 CPU reference 최대 오차 4.768e-7
Decode/NMS:   독립 Python 전체 결과 일치, 최대 오차 6.636e-6
Waymo preprocessing: 5프레임 exact match
Waymo PFN:     독립 NumPy 최대 오차 4.649e-6
Waymo Scatter: 전체 tensor exact match
Waymo RPN:     38 layer probe 최대 오차 8.343e-7
Waymo Head:    GT peak 37개 최대 오차 1.907e-6
Waymo 평가:   TP 25 / FP 3 / FN 12, evaluator와 독립 geometry 일치
```

가장 중요한 다음 작업은 **5프레임 검증을 수백 프레임 통계 평가로 확장하는
것**이다. threshold를 낮췄을 때 FN이 얼마나 복구되고 FP가 얼마나 늘어나는지,
거리와 박스 내부 point count에 따라 recall이 어떻게 달라지는지 확인한 뒤
Waymo 공식 metric과 연결한다.
