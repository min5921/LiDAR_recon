# 09 Implementation Guide: Waymo Input Bridge

## What This Code Does

The Waymo derived archives already contain per-frame lidar bins. Each lidar bin
stores six `float32` values per point:

```text
[x, y, z, intensity, elongation, nlz_flag]
```

The current C++ CenterPoint pipeline expects five values per point:

```text
[x, y, z, intensity, elongation]
```

원본 CenterPoint의 Waymo loader는 다섯 feature를 결합하기 전에 intensity에
`tanh`를 적용한다. `export_waymo_frame.py`의 기본값도 이 규칙을 따른다.
비교 실험을 위해서만 `--intensity-transform none`을 사용한다.

So this milestone builds a small bridge:

1. `tools/export_waymo_frame.py` opens one segment zip.
2. It reads selected lidar entries such as `frame_000/lidar/TOP_return1.bin`.
3. It checks that the data is shaped as `N x 6`.
4. It removes the last column, `nlz_flag`.
5. It writes a new raw `float32` bin with `N x 5` values.
6. `waymo_frame_inspect` reads that `N x 5` bin in C++ and prints stats.

## Why Zip Export Is Python First

C++17 can read normal files, but it does not include a standard zip reader.
Using Python for zip extraction lets the C++ side stay focused on the exact
data layout that the inference pipeline will consume.

Later, if we want a pure C++ archive reader, we can add a small dependency such
as miniz, libzip, or libarchive.

## C++ Data Structure

`WaymoPoint` is intentionally simple:

```cpp
struct WaymoPoint {
    float x;
    float y;
    float z;
    float intensity;
    float elongation;
};
```

This is exactly 20 bytes per point:

```text
5 float32 values * 4 bytes = 20 bytes
```

The C++ reader verifies this by checking:

```text
file_size % 20 == 0
```

If this check fails, the file is not a valid 5-feature CenterPoint input.

## Sample Input

From the archive:

```text
frame_000/lidar/TOP_return1.bin
```

The source point layout is:

```text
x, y, z, intensity, elongation, nlz_flag
```

Example conceptual row:

```text
25.1, -3.2, 1.4, 0.52, 0.08, -1
```

## Sample Output

The exported row becomes:

```text
25.1, -3.2, 1.4, 0.52, 0.08
```

That is the form we can feed into voxelization with `feature_dim=5`.

## Connection To The Previous Milestones

The next handoff is:

```text
Waymo zip
  -> export_waymo_frame.py
  -> 5-feature points.bin
  -> 02 voxelization
  -> 03 feature decoration
  -> 04 PFN
  -> 05 scatter
  -> 06 RPN
  -> 07 CenterHead
  -> 08 Decode/NMS
```

