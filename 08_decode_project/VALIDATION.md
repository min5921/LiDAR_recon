# 08 Validation

## End-to-end input

The test uses the existing pipeline output:

```text
KITTI sample -> PFN -> Scatter -> RPN CUDA -> CenterHead CUDA -> Decode/NMS
```

The network weights are trained for Waymo, so this mixed-dataset run validates implementation behavior and determinism rather than detection quality.

## Measured result

- Score/range candidates: `1380`
- Candidates after pre-NMS top-k: `1380` (limit is `4096`)
- Final detections: `500` (post-NMS limit reached)
- Warm CUDA decode: `0.062 ms`
- C++ rotated NMS: `67.261 ms`
- Non-finite decoded candidates: rejected by the CUDA kernel

## Independent reference

`tools/validate_reference.py` independently repeats the following in NumPy/Python:

1. class argmax and sigmoid
2. center, dimensions, and yaw decode
3. score and post-center range filtering
4. score ordering and pre-NMS top-k
5. rotated rectangle polygon IoU and NMS

The reference produced the same `1380` candidates and the same ordered `500` final source indices and labels. Maximum numeric difference was `6.63617554e-06`.

## Important interpretation

Reaching the 500-box cap is not evidence of a good detection result. This frame uses a KITTI point cloud with a Waymo-trained model and therefore has a dataset mismatch. A semantic check requires a converted Waymo frame and its calibration/annotation data.
