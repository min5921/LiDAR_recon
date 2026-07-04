# 05 Scatter 구현 코드 가이드

## 입출력

```text
pillar_features [10404,64]
coordinates     [10404,4]
  -> BEV [1,64,468,468], NCHW
```

핵심 함수는 `src/scatter.cpp`의 `scatter_pillars_cpu()`다. Scatter에는 학습 weight가 없다.

## 입력 검증

```cpp
expected_features = num_pillars * channels;
expected_coordinates = num_pillars * 4;
```

binary 원소 수가 metadata와 다르면 예외를 발생시킨다.

## Batch 크기

```cpp
for (int pillar = 0; pillar < num_pillars; ++pillar) {
    const int batch = coordinates[pillar * 4];
    batch_size = std::max(batch_size, batch + 1);
}
```

현재 sample은 모든 batch index가 0이므로 `batch_size=1`이다.

## BEV 초기화

```cpp
features.assign(
    batch_size * channels * grid_y * grid_x,
    0.0F);
```

pillar가 없는 cell은 이 초기값 0을 유지한다.

## 좌표 읽기

```cpp
const int batch = coordinates[coord_offset + 0];
const int z = coordinates[coord_offset + 1];
const int y = coordinates[coord_offset + 2];
const int x = coordinates[coord_offset + 3];
```

Grid 밖 좌표와 동일한 cell의 중복 pillar는 예외로 처리한다.

## 핵심 복사

```cpp
for (int channel = 0; channel < channels; ++channel) {
    source = pillar * channels + channel;
    destination =
        ((batch * channels + channel) * grid_y + y) * grid_x + x;
    bev[destination] = pillar_features[source];
}
```

수식으로 쓰면 다음 한 줄이다.

```text
BEV[batch,:,y,x] = pillar_features[pillar,:]
```

## 왜 Z를 출력 index에 쓰지 않는가

PointPillars 설정은 `grid_z=1`이고 PFN이 높이 정보를 64차원 feature 안에 이미 압축했다. Scatter 출력은 2D CNN용 BEV이므로 Y와 X만 공간축으로 사용한다.

## 검증

```text
Python shape: [1,64,468,468]
C++ shape:    [1,64,468,468]
exactly equal: True
max abs diff: 0
```

Scatter는 산술 계산 없이 값을 복사하므로 bitwise exact 비교가 가능하다.

## 현재 한계

CPU 구현이며 dense BEV 전체를 host memory에 생성한다. CUDA 파이프라인에서는 PFN 출력부터 BEV까지 device memory 안에서 바로 연결해야 복사 비용을 줄일 수 있다.
