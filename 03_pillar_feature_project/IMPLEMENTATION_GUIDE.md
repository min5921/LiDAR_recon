# 03 Pillar Feature Decoration 구현 코드 가이드

## 입출력

```text
raw pillars [10404,20,4]
  -> decorated pillars [10404,20,9]
```

핵심 함수는 `src/pillar_feature.cpp`의 `decorate_pillars_cpu()`다.

## 출력 초기화

```cpp
result.decorated_feature_dim = metadata.feature_dim + 5;
result.decorated_pillars.assign(
    num_pillars * max_points * decorated_feature_dim, 0.0F);
```

추가되는 값은 cluster offset 3개와 pillar center offset 2개다. 0으로 초기화하므로 padding row는 별도 복사 없이 0으로 남는다.

## 실제 point 수

```cpp
const int point_count = std::clamp(
    num_points_per_pillar[pillar_idx], 0, max_points_per_pillar);
```

20개 전체가 아니라 유효 point까지만 처리한다.

## XYZ 평균

```cpp
mean_x += pillars[input_offset + 0];
mean_y += pillars[input_offset + 1];
mean_z += pillars[input_offset + 2];

mean_x /= point_count;
mean_y /= point_count;
mean_z /= point_count;
```

## Pillar 중심

```cpp
x_offset = voxel_x * 0.5F + x_min;
y_offset = voxel_y * 0.5F + y_min;

pillar_center_x = x_coord * voxel_x + x_offset;
pillar_center_y = y_coord * voxel_y + y_offset;
```

좌표가 `[batch,z,y,x]`이므로 `y=coordinates[offset+2]`, `x=coordinates[offset+3]`에서 읽는다.

## Decoration 기록

```cpp
output[out + feature_dim + 0] = x - mean_x;
output[out + feature_dim + 1] = y - mean_y;
output[out + feature_dim + 2] = z - mean_z;
output[out + feature_dim + 3] = x - pillar_center_x;
output[out + feature_dim + 4] = y - pillar_center_y;
```

입력과 출력 feature 수가 다르므로 offset도 따로 계산한다.

```cpp
input_offset  = (pillar * max_points + point) * input_feature_dim;
output_offset = (pillar * max_points + point) * decorated_feature_dim;
```

## 검증

```text
Python shape: [10404,20,9]
C++ shape:    [10404,20,9]
decorated equal: True
max abs diff: 0
```

## 현재 한계

CPU 기준 구현이다. CUDA 버전은 pillar별 평균 reduction과 point별 decoration kernel로 나눌 수 있다.
