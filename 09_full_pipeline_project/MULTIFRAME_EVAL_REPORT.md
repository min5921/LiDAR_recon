# Multi-Frame Evaluation Report

Generated on 2026-07-10.

> 2026-07-15 correction: Waymo label polygon을 공식 CCW heading으로
> 회전하도록 평가기를 수정했다. 아래 기본 5프레임 표는 수정 후 값이다.
> 뒤쪽 threshold sweep의 과거 표는 수정 전 evaluator로 만든 실험 기록이므로
> 최신 성능 판단에는 11, 12 마일스톤 결과를 사용한다.

## Purpose

이 문서는 한 프레임 시각화에서 벗어나, 여러 Waymo frame에 대해 현재
C++/CUDA CenterPoint PointPillars pipeline을 실제로 실행하고 GT와 매칭한
결과를 정리한다.

## Added Tool

```text
09_full_pipeline_project/tools/run_waymo_multiframe_eval.py
```

이 스크립트는 다음 과정을 자동 실행한다.

```text
Waymo archive frame
  -> export 5-feature point bin
  -> voxelization
  -> pillar feature decoration
  -> PFN
  -> scatter
  -> RPN
  -> CenterHead
  -> decode + NMS
  -> GT matching by BEV IoU
```

GT matching은 같은 class끼리 prediction score 순서로 greedy하게 수행한다.
기본 match 기준은 BEV IoU `>= 0.5`다.

## Dataset Slice

Archive:

```text
E:\Waymo_datset\derived_v1_4_3\sensor_archives\train\segment-10017090168044687777_6380_000_6400_000_with_camera_labels.zip
```

5-frame run:

```text
frame_000
frame_001
frame_002
frame_003
frame_004
```

## Main Run

Settings:

```text
NMS IoU threshold: 0.5
score threshold: 0.35
match IoU threshold: 0.5
lidars: TOP FRONT SIDE_LEFT SIDE_RIGHT REAR
returns: return1 return2
```

Command:

```text
python 09_full_pipeline_project/tools/run_waymo_multiframe_eval.py ^
  --project-root "C:\Users\user\Desktop\Onechip\Codex\my project" ^
  --archive "E:\Waymo_datset\derived_v1_4_3\sensor_archives\train\segment-10017090168044687777_6380_000_6400_000_with_camera_labels.zip" ^
  --output-dir "C:\Users\user\Documents\객체인지\waymo_multiframe_eval_nms05_score035_5frames" ^
  --weights-root "C:\Users\user\Documents\객체인지\weights_full_novelocity" ^
  --max-frames 5 ^
  --nms-iou 0.5 ^
  --score-threshold 0.35 ^
  --match-iou 0.5
```

Aggregate result:

| Metric | Value |
|---|---:|
| Frames | 5 |
| Predictions | 14 |
| GT labels | 37 |
| TP | 12 |
| FP | 2 |
| FN | 25 |
| Precision | 0.857 |
| Recall | 0.324 |

Per-frame result:

| Frame | Predictions | Labels | TP | FP | FN | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| frame_000 | 3 | 7 | 3 | 0 | 4 | 1.000 | 0.429 |
| frame_001 | 2 | 7 | 2 | 0 | 5 | 1.000 | 0.286 |
| frame_002 | 2 | 7 | 2 | 0 | 5 | 1.000 | 0.286 |
| frame_003 | 3 | 7 | 3 | 0 | 4 | 1.000 | 0.429 |
| frame_004 | 4 | 9 | 2 | 2 | 7 | 0.500 | 0.222 |

Output files:

```text
C:\Users\user\Documents\객체인지\waymo_multiframe_eval_nms05_score035_5frames\aggregate_report.json
C:\Users\user\Documents\객체인지\waymo_multiframe_eval_nms05_score035_5frames\frame_summary.csv
```

## Score Threshold Sweep

All threshold sweep results below use `NMS IoU = 0.5`, `match IoU = 0.5`.

### 3-frame comparison

| Score threshold | Predictions | TP | FP | FN | Precision | Recall |
|---:|---:|---:|---:|---:|---:|---:|
| 0.35 | 7 | 7 | 0 | 14 | 1.000 | 0.333 |
| 0.30 | 11 | 8 | 3 | 13 | 0.727 | 0.381 |
| 0.25 | 13 | 8 | 5 | 13 | 0.615 | 0.381 |

Interpretation:

Lowering the score threshold from `0.35` to `0.30` recovers one more true
positive on the first 3 frames, but it also introduces 3 false positives.
Lowering further to `0.25` does not recover more true positives in this slice;
it only increases false positives.

## Follow-Up Fixes From Review

The decoder and evaluator were hardened after the first multi-frame run.

### Decode reproducibility

`centerpoint_decode.exe` now validates threshold arguments. Invalid values such
as `nope` or values outside `[0, 1]` fail immediately instead of being parsed as
zero.

Each decode output directory now also writes:

```text
decode_config.json
```

This records:

```text
score_threshold
nms_iou_threshold
use_pcdet_nms_convention
use_class_score_thresholds
class_score_thresholds
pre_max_size
post_max_size
```

The multi-frame runner uses this file when `--skip-existing` is enabled, so a
cached result is reused only when its decode settings match the requested run.

### PCDet-style NMS convention option

The decoder now supports a selectable NMS convention:

```text
current
pcdet
```

`pcdet` applies the same geometric conversion used by original CenterPoint
before calling PCDet NMS:

```text
dx/dy swap
yaw = -yaw - pi/2
```

The command shape is:

```text
centerpoint_decode.exe <07_head_dir> <output_dir> [nms_iou] [score] [nms_convention]
```

Example:

```text
centerpoint_decode.exe 07_head 08_detections 0.5 0.35 pcdet
```

On the 5-frame slice, `current` and `pcdet` produced the same aggregate result:

| NMS convention | Predictions | TP | FP | FN | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| current | 14 | 10 | 4 | 27 | 0.714 | 0.270 |
| pcdet | 14 | 10 | 4 | 27 | 0.714 | 0.270 |

This means the visible error on this slice is not fixed by the NMS convention
alone, but the option is now available for broader comparisons.

### Class-wise score thresholds

The decoder can now receive class-specific thresholds:

```text
centerpoint_decode.exe <07_head_dir> <output_dir> <nms_iou> <score> <nms_convention> <vehicle_score> <pedestrian_score> <cyclist_score>
```

The multi-frame runner exposes the same knobs:

```text
--vehicle-score-threshold
--pedestrian-score-threshold
--cyclist-score-threshold
```

Tested setting:

```text
NMS IoU: 0.5
base score: 0.30
NMS convention: pcdet
vehicle: 0.30
pedestrian: 0.45
cyclist: 0.45
```

5-frame result:

| Setting | Predictions | TP | FP | FN | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| score 0.35, pcdet | 14 | 10 | 4 | 27 | 0.714 | 0.270 |
| class-wise 0.30/0.45/0.45, pcdet | 17 | 11 | 6 | 26 | 0.647 | 0.297 |

The class-wise run recovered one more true positive, but it added two more false
positives. It is useful for experimentation, but not better as a default on
this slice.

### FP/FN diagnostics

The evaluator now records detailed false positive and false negative lists.

False positives include:

```text
score
best_same_class_iou
nearest_same_class_center_distance_m
pred_xy
```

False negatives include:

```text
best_prediction_iou
best_prediction_score
nearest_prediction_center_distance_m
nearest_prediction_score
gt_xy
```

This is important because it tells whether a missed GT had a nearby low-IoU
prediction or whether the model failed to place any prediction near it.

## Repository Hygiene

The build folders were already listed in `.gitignore`, but some old generated
files under these paths were still tracked:

```text
02_project/build
04_pfn_project/build
```

They have now been removed from the git index with `git rm --cached`, while the
local files remain on disk. Future builds should no longer dirty the repository
through those folders.

## What This Proves

1. The pipeline is now evaluable over multiple frames.
   - We no longer rely only on one BEV image.
   - Each frame produces prediction/label/TP/FP/FN numbers.

2. The clean one-frame setting is not globally clean.
   - `score=0.35` looked clean on `frame_000`.
   - On 5 frames, false positives appear in `frame_003` and `frame_004`.

3. The current detector is conservative and low-recall.
   - At `score=0.35`, precision is acceptable on this slice, but recall is low.
   - Many GT vehicles are missed.

4. Simply lowering score threshold is not enough.
   - `score=0.30` and `0.25` add false positives quickly.
   - Recall only improves slightly or not at all.

## Likely Causes To Investigate Next

1. Preprocessing mismatch
   - Current derived archive input may not exactly match original CenterPoint
     Waymo preprocessing.
   - NLZ filtering, feature scaling, return handling, and sensor aggregation
     need direct comparison with the official converter output.

2. NMS convention mismatch
   - Original CenterPoint calls `rotate_nms_pcdet()` and converts box convention
     before PCDet CUDA NMS.
   - Our NMS is a direct polygon implementation. It is useful, but not yet a
     byte-for-byte equivalent of the original postprocess.

3. Box dimension/yaw mismatch
   - Matched centers are close, but IoU often stays around `0.50~0.69`.
   - This points to width/length/yaw convention or dimension regression
     differences.

4. Threshold needs class-wise treatment
   - The GT slice contains only `VEHICLE`, while false positives include
     `PEDESTRIAN`.
   - Class-wise score thresholds may be necessary.

## Recommended Next Step

The next coding milestone should be a formal postprocess/evaluation pass:

1. Implement PCDet-style NMS convention in C++:
   - swap dimensions like original `rotate_nms_pcdet()`
   - apply yaw transform `-yaw - pi/2`
   - compare selected indices with the current polygon NMS

2. Add class-wise threshold options:
   - vehicle threshold
   - pedestrian threshold
   - cyclist threshold

3. Extend the evaluator to save FP/FN lists:
   - false positive prediction boxes
   - missed GT boxes
   - nearest prediction per missed GT

That will tell whether the next error source is mostly postprocess convention,
thresholding, or upstream feature/preprocessing mismatch.
