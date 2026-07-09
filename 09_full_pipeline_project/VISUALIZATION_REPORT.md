# Waymo Visualization Report

Generated on 2026-07-09.

## Purpose

This report checks whether the C++/CUDA detection output is in the same BEV
coordinate frame as Waymo `laser_labels.json`.

The visualization overlays:

- Gray points: exported Waymo lidar points
- Orange boxes: C++/CUDA detections from `08_decode_project`
- Blue boxes: Waymo laser labels from the same frame

## Input

```text
archive:
E:\Waymo_datset\derived_v1_4_3\sensor_archives\train\segment-10017090168044687777_6380_000_6400_000_with_camera_labels.zip

frame:
frame_000
```

## Generated Images

```text
C:\Users\user\Documents\객체인지\waymo_detection_run\visualization\bev_predictions_vs_labels.png
C:\Users\user\Documents\객체인지\waymo_detection_run_all_lidars\visualization\bev_predictions_vs_labels.png
C:\Users\user\Documents\객체인지\waymo_detection_run_all_lidars\visualization\bev_predictions_vs_labels_score010.png
```

## All-Lidar Summary

Score threshold:

```text
0.2
```

Drawn predictions:

```text
23
```

Drawn Waymo labels:

```text
7
```

Prediction class counts:

```text
VEHICLE: 16
PEDESTRIAN: 7
```

Waymo label class counts:

```text
VEHICLE: 7
```

## Top Prediction Center Distance To Nearest Same-Class Label

The strongest predictions are close to same-class Waymo labels in BEV:

| Prediction | Score | Nearest GT center distance |
|---|---:|---:|
| VEHICLE | 0.844101 | 0.243 m |
| VEHICLE | 0.655414 | 0.028 m |
| VEHICLE | 0.568739 | 0.273 m |
| VEHICLE | 0.428607 | 0.269 m |
| VEHICLE | 0.376134 | 0.255 m |

## Interpretation

The strongest vehicle detections are spatially aligned with Waymo labels. This
is a good sanity check for the input bridge, voxel coordinates, BEV scatter,
decode formula, and yaw convention.

This is not a full metric evaluation yet. The next step is to replace this
nearest-center sanity check with proper BEV/3D IoU matching and class-wise
precision/recall over multiple frames.

