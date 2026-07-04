# 02 Voxelization 구현 코드 가이드

## 입출력

```text
points [124668,4]
  -> pillars [10404,20,4]
  -> coordinates [10404,4], order=[batch,z,y,x]
  -> num_points [10404]
```

핵심 파일은 `src/voxelization.cpp`, 핵심 함수는 `voxelize_cpu()`다.

## main 호출 흐름

```cpp
PointCloud cloud = read_float32_point_cloud(input_path, feature_dim);
VoxelizationResult result = voxelize_cpu(cloud, config);
write_debug_dump(output_dir, config, result);
```

`.bin`을 읽고, pillar를 만든 뒤, 중간 tensor를 다시 binary로 저장한다.

## Grid 크기

```cpp
const float extent = range_max - range_min;
grid[axis] = static_cast<int>(std::round(extent / voxel_size));
```

Waymo X축은 `(74.88 - (-74.88)) / 0.32 = 468`이다.

## Point 좌표 계산

```cpp
const int coord = static_cast<int>(std::floor(
    (point[axis] - point_cloud_range[axis]) / voxel_size[axis]));
```

`coord < 0` 또는 `coord >= grid_size`이면 range 밖 point이므로 `continue`한다.

## 좌표 map

```cpp
int flatten_zyx(int z, int y, int x, const Grid& grid) {
    return (z * grid_y + y) * grid_x + x;
}
```

`coord_to_voxel_idx[map_index]`가 `-1`이면 새 pillar를 만든다. 0 이상이면 기존 pillar에 point를 추가한다.

```cpp
coordinates[offset + 0] = 0;
coordinates[offset + 1] = z;
coordinates[offset + 2] = y;
coordinates[offset + 3] = x;
```

## Point 복사와 offset

```cpp
const std::size_t offset =
    (voxel_idx * max_points + point_count) * feature_dim;

for (int feature = 0; feature < feature_dim; ++feature) {
    pillars[offset + feature] = point[feature];
}
```

`[pillar,point,feature]`를 1차원 `std::vector`에 저장하는 식이다. 이미 20개 point가 있으면 추가 point는 버린다.

## 마지막 resize

처음에는 최대 60000 pillar 공간을 만들고, 처리 후 실제 `voxel_count`만 남긴다.

```cpp
pillars.resize(voxel_count * max_points * feature_dim);
coordinates.resize(voxel_count * 4);
num_points.resize(voxel_count);
```

## 검증

`tools/compare_python_cpp_voxelization.py` 결과:

```text
coordinates equal: True
num_points equal: True
pillars equal: True
max abs diff: 0
```

## 읽을 순서

```text
include/centerpoint/types.hpp
src/main.cpp
src/io/bin_point_reader.cpp
src/voxelization.cpp
src/io/debug_dump.cpp
tools/compare_python_cpp_voxelization.py
```
