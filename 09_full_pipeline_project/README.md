# 09 Full Pipeline Project

This milestone starts the full Waymo-based inference path.

Current scope:

1. Read one derived Waymo sensor archive zip.
2. Export a selected lidar frame to the current CenterPoint point format.
3. Verify the exported point cloud from C++.

The derived archive lidar schema is:

```text
x, y, z, intensity, elongation, nlz_flag
```

The current CenterPoint PointPillars input keeps:

```text
x, y, z, intensity, elongation
```

Waymo intensity is normalized with `tanh(intensity)` by default. This matches
the original CenterPoint `read_single_waymo()` loader. Use
`--intensity-transform none` only for an explicit raw-intensity comparison.

Evaluation converts decoded CenterPoint yaw with
`waymo_heading = -prediction_yaw - pi/2` and rotates both prediction and Waymo
label polygons counter-clockwise. This convention is covered by the tests in
`12_waymo_fn_analysis_project`.

## Multi-frame Cache Contract

`run_waymo_multiframe_eval.py --skip-existing` reuses a frame only when its
`pipeline_cache_manifest.json` matches the current run. The manifest includes:

- archive path, size, and modification time
- frame name and point preprocessing settings
- score threshold and NMS settings
- SHA-256 signatures of the pipeline scripts, executables, and weight files
- Python and NumPy versions

If any value differs, the old frame directory is removed and every pipeline
stage is recomputed. The aggregate report stores the same information in
`run_contract` so that raw/tanh comparisons can reject mixed experiments.

## Export One Frame

```powershell
python tools/export_waymo_frame.py `
  "E:\Waymo_datset\derived_v1_4_3\sensor_archives\train\segment-10017090168044687777_6380_000_6400_000_with_camera_labels.zip" `
  "C:\Users\user\Documents\객체인지\waymo_frame_000_top_return1.bin" `
  --frame frame_000 `
  --lidars TOP `
  --returns return1 `
  --summary-json "C:\Users\user\Documents\객체인지\waymo_frame_000_top_return1.summary.json"
```

## Build And Inspect

```powershell
cmake -S . -B build
cmake --build build --config Release
.\build\Release\waymo_frame_inspect.exe "C:\Users\user\Documents\객체인지\waymo_frame_000_top_return1.bin"
```

If MSVC linking fails under a non-ASCII build path, keep the source folder here
but put the build directory in an ASCII-only path:

```powershell
cmake -S . -B "C:\Users\user\AppData\Local\Temp\codex_09_full_pipeline_build"
cmake --build "C:\Users\user\AppData\Local\Temp\codex_09_full_pipeline_build" --config Release
```

## Next

After this loader is verified, the exported 5-feature bin can be passed to
`02_project` with `feature_dim=5`, then connected through PFN, scatter, RPN,
CenterHead, and Decode.
