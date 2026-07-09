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
