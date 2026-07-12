# 09 Full Pipeline Diagnostic Report

Generated on 2026-07-10.

## 목적

Waymo 한 프레임에서 예측 박스가 여러 개 겹쳐 보이는 이유를 숫자로 확인하고,
현재 C++/CUDA 구현이 원본 CenterPoint PointPillars와 얼마나 유사한지 정리한다.

검사 기준 프레임:

```text
E:\Waymo_datset\derived_v1_4_3\sensor_archives\train\segment-10017090168044687777_6380_000_6400_000_with_camera_labels.zip
frame_000
```

## NMS 중복 박스 진단

새 도구:

```text
09_full_pipeline_project/tools/debug_nms_iou.py
```

이 도구는 `detections.csv`를 읽고 같은 클래스 예측 박스끼리 BEV rotated IoU를
계산한다. Waymo `laser_labels.json`이 주어지면 상위 예측과 가장 가까운 GT의
BEV IoU도 함께 계산한다.

### full_novelocity checkpoint 결과

입력:

```text
C:\Users\user\Documents\객체인지\waymo_detection_run_full_novelocity\08_detections\detections.csv
```

요약:

| 항목 | 값 |
|---|---:|
| 사용한 prediction 수(score >= 0.1) | 48 |
| 같은 클래스이며 8m 이내인 pair | 190 |
| IoU > 0.7 pair | 0 |
| IoU > 0.5 pair | 56 |
| IoU > 0.3 pair | 92 |

대표 중복 예:

| label | score A | score B | center distance | IoU |
|---|---:|---:|---:|---:|
| PEDESTRIAN | 0.319 | 0.232 | 0.146 m | 0.689 |
| VEHICLE | 0.696 | 0.312 | 0.319 m | 0.689 |
| VEHICLE | 0.129 | 0.109 | 0.295 m | 0.695 |

해석:

현재 NMS threshold는 0.7이다. 위 중복 후보들은 화면상으로는 거의 같은
물체처럼 보이지만 IoU가 0.7을 아주 조금 넘지 못한다. 따라서 NMS가 실패한
것이라기보다, 현재 threshold가 중복 후보를 많이 남기는 설정이다. threshold를
0.5로 낮추면 위 중복 후보 상당수가 제거된다.

상위 vehicle 예측의 GT IoU:

| rank | score | center distance to GT | BEV IoU to GT |
|---:|---:|---:|---:|
| 0 | 0.802 | 0.206 m | 0.658 |
| 1 | 0.696 | 0.118 m | 0.656 |
| 2 | 0.374 | 0.283 m | 0.528 |
| 4 | 0.312 | 0.423 m | 0.604 |

상위 vehicle은 GT 근처에 있으나 IoU가 0.7 이상으로 안정적으로 붙지는 않는다.
즉 중심점은 맞지만 box 크기, 회전, NMS 후처리, 입력 전처리 중 하나 이상이
원본과 완전히 같지 않을 가능성이 있다.

### 기존 50_novelocity all-lidar 결과

입력:

```text
C:\Users\user\Documents\객체인지\waymo_detection_run_all_lidars\08_detections\detections.csv
```

요약:

| 항목 | 값 |
|---|---:|
| 사용한 prediction 수(score >= 0.1) | 94 |
| 같은 클래스이며 8m 이내인 pair | 324 |
| IoU > 0.7 pair | 0 |
| IoU > 0.5 pair | 85 |
| IoU > 0.3 pair | 177 |

기존 checkpoint도 같은 현상을 보인다. 가까운 중복 후보가 많지만 0.7을 넘는
pair가 없어 현재 NMS 기준에서는 제거되지 않는다.

## 원본 CenterPoint PointPillars와 같은 부분

현재 구현은 큰 구조상 CenterPoint Waymo PointPillars 경로와 같다.

1. Waymo point input
   - 원본 설정의 `num_input_features=5`와 맞게 `[x, y, z, intensity, elongation]`
     5개 feature를 사용한다.

2. Voxel/Pillar 설정
   - voxel size: `(0.32, 0.32, 6.0)`
   - point cloud range: `(-74.88, -74.88, -2, 74.88, 74.88, 4.0)`
   - max points per voxel: `20`
   - BEV grid: `468 x 468`

3. Pillar feature path
   - pillar decoration
   - PFN `num_filters=[64, 64]`
   - scatter to BEV feature map

4. Backbone/head path
   - RPN layer structure:
     - down blocks: `[64, 128, 256]`
     - up blocks: `[128, 128, 128]`
     - final BEV feature channels: `384`
   - CenterHead heads:
     - `reg`
     - `height`
     - `dim`
     - `rot`
     - `hm`

5. Decode 설정
   - score threshold: `0.1`
   - post center range: `[-80, -80, -10, 80, 80, 10]`
   - NMS pre max: `4096`
   - NMS post max: `500`
   - NMS IoU threshold: `0.7`

6. Weight
   - `.pth` checkpoint에서 실제 PFN/RPN/Head weight를 추출해 C++/CUDA 경로에
     넣고 있다.

## 원본과 아직 다른 부분

1. Waymo preprocessing
   - 원본은 Waymo TFRecord/range image 기반 converter와 `infos_*.pkl` 흐름을
     사용한다.
   - 현재는 우리가 만든 derived sensor archive에서 lidar bin을 읽는다.
   - NLZ filtering, sensor ordering, return 선택, pose 보정, intensity/elongation
     처리 방식이 원본과 완전히 같은지 아직 검증하지 않았다.

2. Voxelization ordering
   - 원본은 `points_to_voxel` 계열 구현을 사용한다.
   - 현재 CPU 구현은 같은 수식을 목표로 하지만, voxel 생성 순서, point 삽입 순서,
     max voxel 도달 시 잘리는 순서가 다를 수 있다.

3. NMS 좌표 변환
   - 원본 CenterPoint는 NMS 직전에 `rotate_nms_pcdet()`를 호출한다.
   - 그 함수는 box를 PCDet CUDA NMS 형식으로 바꾸기 위해 `[w, l]`을 `[l, w]`로
     바꾸고 yaw를 `-yaw - pi/2`로 변환한다.
   - 현재 C++ NMS는 직접 polygon IoU를 계산한다. 유사한 결과를 목표로 하지만,
     이 PCDet 변환과 CUDA kernel 동작을 1:1로 재현한 것은 아니다.

4. Evaluation
   - 현재는 한 프레임 시각화와 BEV IoU 진단이다.
   - Waymo 공식 metric 또는 여러 프레임 mAP/precision/recall 평가는 아직 없다.

5. 속도/구조
   - RPN/Head는 CUDA/cuBLAS 기반으로 구현했지만, 중간 파일 dump/load가 많다.
   - PFN, scatter, decode 사이도 아직 하나의 GPU 메모리 pipeline으로 연결되어
     있지 않다.

## 문제점과 개선 방안

### 1. 중복 예측 박스

문제:

NMS threshold 0.7에서는 IoU 0.5~0.69 수준의 가까운 중복 박스가 남는다.

개선:

- `debug_nms_iou.py`로 threshold sweep을 계속 비교한다.
- 실험용으로 NMS threshold `0.5` 결과 이미지를 만들어 본다.
- 단, 원본 Waymo PointPillars 설정은 0.7이므로 기본값을 바로 바꾸기보다는
  debug 옵션으로 먼저 비교한다.

### 1-1. NMS 0.5 + score 0.35 실험

중복 박스와 낮은 confidence false positive를 줄이기 위해 decode 실행부에
optional argument를 추가했다.

```text
centerpoint_decode.exe <07_head_dir> <output_dir> [nms_iou_threshold] [score_threshold]
```

실험 명령:

```text
centerpoint_decode.exe ...\07_head ...\08_detections_nms05_score035 0.5 0.35
```

결과:

| 항목 | 기존 0.7/0.1 | NMS 0.5 | NMS 0.5 + score 0.35 |
|---|---:|---:|---:|
| score/range candidates | 112 | 112 | 11 |
| final detections | 48 | 22 | 3 |
| same-class pairs within 8m | 190 | 27 | 0 |
| prediction classes | VEHICLE 35, PEDESTRIAN 13 | VEHICLE 13, PEDESTRIAN 9 | VEHICLE 3 |

최종 3개 prediction:

| class | score | nearest GT center distance | BEV IoU |
|---|---:|---:|---:|
| VEHICLE | 0.802 | 0.206 m | 0.658 |
| VEHICLE | 0.696 | 0.118 m | 0.656 |
| VEHICLE | 0.374 | 0.283 m | 0.528 |

생성 파일:

```text
C:\Users\user\Documents\객체인지\waymo_detection_run_full_novelocity\08_detections_nms05_score035\detections.csv
C:\Users\user\Documents\객체인지\waymo_detection_run_full_novelocity\visualization\bev_predictions_vs_labels_decode_nms05_score035.png
C:\Users\user\Documents\객체인지\waymo_detection_run_full_novelocity\nms_debug\nms_debug_decode_nms05_score035.json
```

해석:

이 설정은 현재 frame에서 눈에 띄는 이상 예측을 제거한다. 다만 recall은 낮아진다.
Waymo label은 vehicle 7개인데 최종 prediction은 3개만 남는다. 따라서 이 값은
"깔끔한 시각화/고신뢰 detection"에는 적합하지만, 실제 성능 평가용 기본값으로
확정하기 전에 여러 frame에서 precision/recall을 같이 봐야 한다.

### 2. GT와 box IoU가 낮은 예측

문제:

상위 vehicle center는 GT와 가깝지만 BEV IoU가 0.5~0.66 정도로 머무는 경우가
있다.

개선:

- prediction box convention을 원본 `rotate_nms_pcdet()` 변환과 맞춰 재검증한다.
- Waymo label의 width/length/heading과 prediction의 dim/yaw를 같은 기준으로
  맞추는 unit test를 만든다.
- 시각화뿐 아니라 수치 IoU matching으로 TP/FP/FN을 표시한다.

### 3. Pedestrian false positive

문제:

현재 frame의 Waymo label은 vehicle만 7개인데, prediction에는 pedestrian이
여러 개 나온다. 이는 낮은 score threshold에서 보이는 false positive일 수 있다.

개선:

- class별 threshold를 분리한다.
- `score >= 0.2`, `0.3`, `0.4` 이미지와 통계를 같이 저장한다.
- 여러 frame에서 반복되는 현상인지 확인한다.

### 4. 공식 preprocessing과의 차이

문제:

모델 weight가 기대한 입력 분포와 현재 derived bin 입력 분포가 다를 수 있다.

개선:

- 원본 CenterPoint converter로 같은 segment/frame을 변환한 결과와 현재 bin을
  point 수, feature 범위, NLZ 제거 여부 기준으로 비교한다.
- PFN 입력 직전의 decorated feature 일부를 Python 원본과 C++ 구현에서 나란히
  dump해 차이를 확인한다.

### 5. 성능

문제:

전체가 아직 실시간 pipeline이라기보다 milestone들을 파일로 연결한 검증용
pipeline이다.

개선:

- PFN과 scatter도 CUDA로 옮긴다.
- milestone 사이 파일 저장을 제거하고 device buffer를 직접 넘긴다.
- NMS는 CPU polygon 방식에서 CUDA bitmask 방식으로 교체한다.

## 다음 추천 작업

1. NMS threshold 0.7/0.5/0.3 결과를 같은 이미지로 비교한다.
2. 원본 `rotate_nms_pcdet()`와 동일한 convention을 C++ NMS에 맞춘다.
3. GT matching report를 class별 TP/FP/FN 형태로 확장한다.
4. 5~10개 Waymo frame을 자동으로 돌려 같은 문제가 반복되는지 확인한다.
5. 그 다음 PFN/scatter를 CUDA로 연결해 파일 기반 milestone을 실제 pipeline으로
   합친다.
