# 13 Waymo Operating Point Project

이 마일스톤은 score threshold를 바꾸었을 때 precision, recall, F1이 어떻게
변하는지 실제 Waymo segment 전체에서 측정한다. 거리와 GT 박스 내부 point
수에 따른 recall도 함께 계산한다.

## 핵심 아이디어

가장 낮은 threshold로 Decode와 NMS를 한 번 실행한다. NMS는 score가 높은
박스부터 처리하므로 낮은 score 박스가 높은 score 박스를 제거할 수 없다.
따라서 이 결과에서 score만 필터링하면 같은 NMS 설정의 더 높은 threshold
결과를 재추론 없이 얻을 수 있다.

5프레임에서 최소 threshold `0.05` 결과를 `0.35`로 필터링했을 때 기존
`0.35` 직접 Decode의 모든 `detections.csv` 행이 exact match했다.

## 198프레임 compact 실행

```powershell
python -B 09_full_pipeline_project/tools/run_waymo_multiframe_eval.py `
  --project-root "C:\Users\user\Desktop\Onechip\Codex\my project" `
  --archive "E:\Waymo_datset\derived_v1_4_3\sensor_archives\train\segment-10017090168044687777_6380_000_6400_000_with_camera_labels.zip" `
  --output-dir "C:\Users\user\Documents\객체인지\waymo_eval_tanh_pcdet_score005_198frames_compact" `
  --weights-root "C:\Users\user\Documents\객체인지\weights_full_novelocity" `
  --max-frames 198 `
  --intensity-transform tanh `
  --nms-iou 0.5 `
  --score-threshold 0.05 `
  --nms-convention pcdet `
  --match-iou 0.5 `
  --skip-existing `
  --compact-output `
  --summary-only
```

`--compact-output`은 프레임 평가가 끝난 뒤 `02_voxel`부터 `07_head`까지의
대형 중간 tensor와 `points.bin`을 지운다. detection, match report, 실행 계약,
로그는 남긴다. 실제 198프레임 결과는 약 `126.2MB`다.

## Threshold 분석

```powershell
python -B tools/analyze_operating_points.py `
  --eval-dir "C:\Users\user\Documents\객체인지\waymo_eval_tanh_pcdet_score005_198frames_compact" `
  --output-dir "C:\Users\user\Documents\객체인지\waymo_operating_point_198frames" `
  --precision-floor 0.8
```

출력:

```text
operating_point_analysis.json  전체 threshold와 frame별 수치
threshold_summary.csv          threshold별 전체 지표
stratified_recall.csv          class/거리/point-count별 지표
operating_point_report.md      자동 요약
operating_point_study.png      비교 그래프
```

## 실제 결과

| 선택 기준 | Threshold | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| 최대 F1 | 0.30 | 0.9114 | 0.6329 | 0.7470 |
| Precision >= 0.80에서 최대 recall | 0.25 | 0.8400 | 0.6502 | 0.7330 |
| 기존 기준 | 0.35 | 0.9478 | 0.6154 | 0.7463 |

이 segment의 4,337개 GT는 모두 `VEHICLE`이다. 따라서 pedestrian와 cyclist
threshold는 이 결과만으로 확정할 수 없다.

## 테스트

```powershell
python -B -m unittest discover -s tests -v
```

