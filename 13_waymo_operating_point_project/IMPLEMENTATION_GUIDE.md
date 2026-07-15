# 13 Implementation Guide: Operating Point Study

## 1. 전체 실행 흐름

```text
Waymo frame
  -> 02~07 C++/CUDA inference
  -> minimum score Decode + rotated NMS
  -> GT matching report 저장
  -> 큰 intermediate tensor 정리
  -> minimum-score detections를 메모리에 한 번 로드
  -> threshold별 score filtering
  -> 공식 CCW geometry로 GT 재매칭
  -> 전체/class/거리/point-count 지표 저장
```

전체 추론은 `09_full_pipeline_project/tools/run_waymo_multiframe_eval.py`가
담당하고, threshold 분석은 `tools/analyze_operating_points.py`가 담당한다.

## 2. 왜 Decode를 매번 실행하지 않는가

Rotated NMS의 입력은 score 내림차순으로 정렬된다.

```text
score 0.90 box
score 0.70 box
score 0.20 box
score 0.05 box
```

낮은 score 박스는 자신보다 먼저 선택된 높은 score 박스를 억제할 수 없다.
따라서 threshold `0.05`에서 NMS를 실행한 결과를 `score >= 0.35`로 필터링한
집합은 처음부터 threshold `0.35`로 실행한 결과와 같다. 이 규칙은 NMS IoU,
NMS convention, pre/post max가 동일할 때만 사용한다.

실제 5프레임의 모든 detection 행을 비교한 결과:

```text
frame_000  exact
frame_001  exact
frame_002  exact
frame_003  exact
frame_004  exact
```

## 3. Compact 출력

일반 실행은 프레임당 약 397MB를 사용한다.

```text
02 voxel       3.82 MB
03 decorated   7.27 MB
04 PFN         2.33 MB
05 scatter    53.47 MB
06 RPN       320.84 MB
07 Head        9.19 MB
```

198프레임을 모두 보관하면 약 78GB가 필요하다. `compact_frame_outputs()`는
매 프레임의 GT matching이 끝난 뒤 위 stage와 `points.bin`만 제거한다.

보존되는 항목:

```text
08_detections/
logs/
export_summary.json
match_report.json
pipeline_cache_manifest.json
```

삭제 대상은 항상 `frame_dir` 바로 아래의 고정된 이름만 사용하며, resolve된
부모 경로가 `frame_dir`인지 검사한다.

## 4. Threshold 평가

`prepare_frame_data()`는 archive에서 GT를 읽고 compact evaluation 폴더에서
최소 threshold prediction을 한 번 읽는다. `evaluate_threshold()`는 다음
조건으로 prediction을 선택한다.

```python
float(prediction["score"]) >= threshold
```

그 뒤 12단계의 공식 Waymo CCW rotated IoU 구현을 재사용해 같은 class끼리
score 순서로 greedy matching한다.

```text
TP = IoU >= 0.5로 매칭된 prediction
FP = prediction - TP
FN = GT - TP
precision = TP / (TP + FP)
recall = TP / (TP + FN)
F1 = 2 * precision * recall / (precision + recall)
```

threshold가 증가할 때 prediction과 TP가 증가하지 않는지도
`validate_monotonic_results()`에서 검사한다.

## 5. 거리와 Point-count 구간

거리:

```text
0-30 m
30-50 m
50-75 m
75+ m
```

Point count:

```text
0-4
5-9
10-19
20-49
50+
```

point count는 Waymo label의 `num_lidar_points_in_box`를 사용한다. 대규모
분석에서 매 threshold마다 원본 점을 다시 세지 않아도 되며, 모든 GT에 같은
기준을 적용할 수 있다.

거리와 point-count 구간은 prediction의 precision을 계산하는 분류가 아니라,
GT를 분모로 하는 recall 분석이다.

## 6. 운영점 선택

`select_operating_points()`는 두 값을 따로 저장한다.

1. `best_f1`: 전체 F1이 가장 높은 threshold
2. `best_recall_at_precision_floor`: 지정 precision 이상에서 recall이 가장 높은 threshold

두 기준을 분리하는 이유는 실제 제품이 원하는 FP 허용량에 따라 적절한
threshold가 달라지기 때문이다.

## 7. 실제 출력 예

```text
threshold=0.30
predictions=3012
labels=4337
tp=2745
fp=267
fn=1592
precision=0.9114
recall=0.6329
f1=0.7470
```

## 8. 제한 사항

- 한 개 Waymo training segment의 198프레임 결과다.
- 이 segment의 model 대상 GT는 모두 VEHICLE이다.
- Waymo 공식 mAP/mAPH가 아니라 BEV IoU 0.5 greedy matching이다.
- 연속 프레임은 서로 완전히 독립인 표본이 아니므로 여러 segment 검증이 필요하다.

