# Waymo Detection Run Comparison

Generated on 2026-07-08.

## Input Frame

Archive:

```text
E:\Waymo_datset\derived_v1_4_3\sensor_archives\train\segment-10017090168044687777_6380_000_6400_000_with_camera_labels.zip
```

Frame:

```text
frame_000
```

## Runs

| Run | Input points | Pillars | Candidates before NMS | Detections after NMS | Max score | Mean score | Class counts |
|---|---:|---:|---:|---:|---:|---:|---|
| Waymo TOP return1 | 153,830 | 7,933 | 198 | 111 | 0.831798 | 0.163835 | VEHICLE 65, PEDESTRIAN 43, CYCLIST 3 |
| Waymo all lidars return1+2 | 183,680 | 9,529 | 154 | 94 | 0.844101 | 0.176294 | VEHICLE 51, PEDESTRIAN 38, CYCLIST 5 |
| Previous KITTI sample run | n/a | n/a | 1,380 | 500 | n/a | n/a | n/a |

## Top Detection: Waymo TOP return1

```text
class=VEHICLE score=0.831798
xyz=(20.9490929, -4.64883661, 2.20161819)
size=(2.03517652, 4.46683502, 1.70010483)
yaw=0.243592575
```

## Top Detection: Waymo All Lidars

```text
class=VEHICLE score=0.844101
xyz=(20.9699, -4.675, 2.143)
size=(2.071, 4.568, 1.739)
yaw=0.274
```

## Interpretation

The previous KITTI run was mainly a code-path validation because KITTI data was
being passed through Waymo weights. It produced many more candidates and hit the
post-NMS cap of 500 detections, so it should not be interpreted as model quality.

The Waymo-derived frame is a more meaningful input for this checkpoint. The
TOP-only and all-lidar runs produce similar strongest boxes near the same
locations, which is a useful sanity check that the full pipeline is connected
consistently.

The next useful check is visualization: draw the point cloud, predicted boxes,
and `laser_labels.json` boxes together for this same frame.

