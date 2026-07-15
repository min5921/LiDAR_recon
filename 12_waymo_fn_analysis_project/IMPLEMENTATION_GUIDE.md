# 12 Implementation Guide: False Negative Evidence

## 1. 실행 흐름

진입점은 `tools/analyze_waymo_false_negatives.py`의 `main()`이다.

```text
main()
  -> aggregate_report.json과 frame manifest 검증
  -> archive에서 GT와 센서별 원본 점 로드
  -> detections.csv에서 예측 박스 로드
  -> 공식 Waymo 좌표계로 IoU 독립 재계산
  -> 각 GT 박스 내부 점 수 계산
  -> GT 주변 CenterHead heatmap 최대값 계산
  -> FN 원인 분류
  -> JSON / CSV / Markdown / BEV PNG 저장
```

## 2. 평가 실행 계약 확인

`validate_frame_manifest()`는 모든 프레임이 같은 archive, 전처리, decode 설정,
소스 SHA256, weight SHA256으로 생성됐는지 확인한다. 예전 캐시와 새 실행 결과가
섞이면 분석을 중단한다.

실제 검증된 핵심 설정은 다음과 같다.

```json
{
  "intensity_transform": "tanh",
  "drop_nlz": false,
  "returns": ["return1", "return2"],
  "score_threshold": 0.35,
  "nms_iou": 0.5,
  "match_iou": 0.5
}
```

## 3. Waymo 박스 회전

Waymo label은 `length` 축을 heading 방향으로 두고 반시계 방향(CCW)을
양수로 사용한다. 로컬 점 `(lx, ly)`를 전역으로 옮기는 식은 다음과 같다.

```text
x = center_x + lx * cos(heading) - ly * sin(heading)
y = center_y + lx * sin(heading) + ly * cos(heading)
```

CenterPoint가 decode한 prediction yaw는 Waymo heading과 표현이 다르므로
평가 직전에 다음과 같이 변환한다.

```text
waymo_heading = -prediction_yaw - pi / 2
```

`box_corners()`, `polygon_intersection_area()`, `rotated_iou()`가 이 규칙으로
BEV IoU를 독립 계산한다. 같은 변환은 09 평가기의 `corners()`에도 적용됐다.

## 4. 박스 내부 점 계산

`load_source_points()`는 archive의 각 점을 다음 6개 값으로 읽는다.

```text
[x, y, z, intensity, elongation, nlz]
```

`points_in_waymo_box()`는 점을 GT 중심 기준으로 이동한 뒤 heading의 역회전을
적용한다.

```text
local_x =  dx * cos(heading) + dy * sin(heading)
local_y = -dx * sin(heading) + dy * cos(heading)
```

그 다음 아래 세 조건을 모두 만족하는 점만 박스 내부로 센다.

```text
abs(local_x) <= length / 2
abs(local_y) <= width  / 2
abs(z - center_z) <= height / 2
```

분석 결과에는 다음 수가 따로 남는다.

- `all_archive_points_in_box`: 모든 LiDAR와 return의 점
- `selected_points_in_box`: 실행 설정이 선택한 센서와 return의 점
- `selected_points_in_model_range`: voxel 범위 안에 남은 점
- `effective_model_points`: NLZ 설정까지 적용해 실제 모델이 본 점

## 5. Heatmap 증거

`load_heatmap()`은 `07_head`의 heatmap tensor를 읽는다. GT 중심은 모델의
point-cloud range, voxel 크기, output stride를 이용해 feature-map cell로 바꾼다.
`heatmap_evidence()`는 해당 cell 주변 기본 `5 x 5` 영역의 최대 sigmoid score를
구한다.

예를 들어 score threshold가 0.35인데 주변 최대값이 0.1266이면 decode 전에
이미 후보가 되지 못한 것이다.

## 6. FN 분류 순서

`classify_false_negative()`는 아래 순서로 가장 앞선 원인을 선택한다.

1. `EVALUATION_GEOMETRY_MISMATCH`: 독립 공식 좌표계에서는 매칭됨
2. `OUT_OF_RANGE`: GT 중심 또는 높이가 모델 범위 밖
3. `LOW_POINT_COUNT`: 실제 입력 점이 기준보다 적음
4. `PREPROCESSING_SENSITIVE`: 센서/return/NLZ/range 선택에서 점이 크게 감소
5. `BOX_REGRESSION_ERROR`: heatmap은 높지만 최종 IoU가 부족함
6. `LOW_MODEL_SCORE`: GT 주변 heatmap이 score threshold 미만

이번 수정 후 첫 번째와 다섯 번째 분류는 0개다. 남은 12개는 점 부족 3개와
낮은 score 9개로만 구성된다.

## 7. 한 행의 실제 출력 예

```text
frame=frame_000
label_id=dx-1oA
distance_m=44.1
official_num_lidar_points=10
effective_model_points=10
heatmap_local_max_score=0.1266
classification=LOW_MODEL_SCORE
```

이 구조 덕분에 FN을 보면서 “모델이 점을 충분히 받았는가”, “Head가 반응했는가”,
“박스 회귀나 평가 좌표계가 문제인가”를 한 행에서 구분할 수 있다.

