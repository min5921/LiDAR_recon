# Preprocessing and Geometry Audit

Generated on 2026-07-12.

> 2026-07-15 correction: 기존 evaluator가 Waymo label heading을 반대
> 방향으로 회전한 문제를 수정했다. all-LiDAR 두 행은 수정 후 지표이며,
> TOP-only 행은 당시 전처리 비교를 남긴 historical 값이다.

## Purpose

This audit checks whether the poor visual result is mainly caused by:

1. preprocessing choices such as NLZ filtering or TOP-only LiDAR input
2. NMS convention
3. prediction/GT box geometry convention
4. low proposal/heatmap recall

The tested archive was:

```text
E:\Waymo_datset\derived_v1_4_3\sensor_archives\train\segment-10017090168044687777_6380_000_6400_000_with_camera_labels.zip
```

Frames:

```text
frame_000
frame_001
frame_002
frame_003
frame_004
```

## Added Tool

```text
09_full_pipeline_project/tools/compare_waymo_eval_runs.py
```

The tool compares several evaluation output folders by reading:

```text
aggregate_report.json
frame_*/export_summary.json
```

Example:

```text
python 09_full_pipeline_project/tools/compare_waymo_eval_runs.py ^
  waymo_eval_review_pcdet_5frames ^
  waymo_eval_preprocess_drop_nlz_5frames ^
  waymo_eval_preprocess_top_only_5frames ^
  --output-json 09_full_pipeline_project/preprocessing_comparison_5frames.json
```

## Preprocessing Comparison

All runs used:

```text
NMS IoU: 0.5
score threshold: 0.35
NMS convention: pcdet
match IoU: 0.5
```

| Run | Points | Dropped by NLZ | Predictions | TP | FP | FN | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| all LiDAR + both returns | 917,322 | 0 | 14 | 12 | 2 | 25 | 0.857 | 0.324 |
| all LiDAR + both returns + drop NLZ | 917,322 | 0 | 14 | 12 | 2 | 25 | 0.857 | 0.324 |
| TOP return1 only (historical) | 768,040 | 0 | 12 | 8 | 4 | 29 | 0.667 | 0.216 |

Interpretation:

- `drop_nlz` did not remove any points in this derived archive slice.
- `drop_nlz` also produced exactly the same aggregate detection result.
- TOP-only input reduced point count, but it also reduced TP and recall.
- Therefore, the current error is not solved by NLZ filtering or TOP-only input.

## Geometry Diagnostics Added

The evaluator now saves richer geometry fields for each match:

```text
pred_box
gt_box
dx_abs_error_m
dy_abs_error_m
raw_yaw_abs_error_rad
waymo_converted_yaw_abs_error_rad
```

For false positives and false negatives it also stores nearby candidate boxes:

```text
nearest_gt_box
best_iou_gt_box
best_prediction_box
nearest_prediction_box
```

This is useful because raw CenterPoint yaw and Waymo label yaw are not directly
in the same convention. The Waymo-style comparison uses:

```text
pred_yaw_as_waymo = -pred_yaw - pi / 2
```

## Matched Box Geometry Summary

On the 10 matched vehicles from the 5-frame pcdet run:

| Field | Mean | Min | Max |
|---|---:|---:|---:|
| center distance | 0.192 m | 0.062 m | 0.283 m |
| dx error | 0.190 m | 0.097 m | 0.250 m |
| dy error | 0.124 m | 0.002 m | 0.359 m |
| raw yaw error | 2.017 rad | 1.906 rad | 2.245 rad |
| Waymo-converted yaw error | 0.034 rad | 0.000 rad | 0.090 rad |
| BEV IoU | 0.625 | 0.503 | 0.689 |

Interpretation:

- The large raw yaw error is mostly a convention artifact.
- After converting prediction yaw into Waymo label convention, yaw error is small.
- Matched centers are also close.
- The visible issue is therefore not primarily a yaw convention failure.
- The main unresolved problem is low recall: many GT vehicles never receive a
  nearby prediction.

## Current Diagnosis

The implementation now behaves like this:

1. When it predicts a vehicle, the center and yaw are usually reasonable.
2. NMS convention change did not alter aggregate metrics on this slice.
3. NLZ filtering did not affect this slice.
4. TOP-only input made recall worse.
5. The dominant issue is that the model produces too few valid high-score boxes.

Most likely next checks:

1. Heatmap / CenterHead output inspection before decode
   - count peaks per class
   - inspect top heatmap locations before score threshold
   - check whether missed GT centers have heatmap responses nearby

2. Coordinate mapping from BEV cell to metric box center
   - verify `x = (cell_x + reg_x) * voxel_size_x * out_size_factor + pc_range_x`
   - verify `y = (cell_y + reg_y) * voxel_size_y * out_size_factor + pc_range_y`

3. Training preprocessing parity
   - compare this derived archive point distribution with the exact official
     CenterPoint Waymo converter output if available

## Recommended Next Coding Step

Add a heatmap audit tool for `07_head` output.

The tool should:

1. read head output tensors
2. list top-K heatmap peaks per class before decode
3. project Waymo GT centers into BEV heatmap cells
4. report whether each GT has a nearby heatmap peak

This will tell whether recall is lost before decode or during decode.
