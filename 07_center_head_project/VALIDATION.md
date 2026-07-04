# CenterHead CUDA Validation

## Test path

`KITTI sample -> existing PFN/Scatter output -> 06 RPN CUDA -> 07 CenterHead CUDA`

The checkpoint is `centerpoint_waymo_pointpillars_50_novelocity.pth`. The KITTI frame is used only as a deterministic numeric input; the resulting detections are not semantically meaningful because the checkpoint was trained for Waymo.

## Result

- RPN input: `[1, 64, 468, 468]`
- RPN output / Head input: `[1, 384, 468, 468]`
- Head outputs: `reg=2`, `height=1`, `dim=3`, `rot=2`, `hm=3`, all `468x468`
- RPN CUDA time: `162.535 ms`
- Head CUDA time: `186.948 ms` (repeat: `183.349 ms`)
- Non-finite values: `0` in every output
- Selected CPU reference: 15 samples, maximum absolute difference `4.76837158203125e-7`
- Repeat execution: all five output SHA-256 values matched exactly

The selected reference checks the first channel of every branch at `(0,0)`, `(234,234)`, and `(467,467)`. This covers both zero-padding boundaries and an interior location.

## Meaning of outputs

- `reg`: sub-cell center offsets
- `height`: object center height
- `dim`: log-space box dimensions before `exp`
- `rot`: sine/cosine-style rotation pair used by decode
- `hm`: class heatmap logits before `sigmoid`

These are raw prediction maps, not final 3D boxes. Box decode and NMS belong to the next milestone.
