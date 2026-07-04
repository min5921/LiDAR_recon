# 08 Box Decode와 Rotated NMS 구현 코드 가이드

## 1. 이 단계의 역할

CenterHead raw map을 실제 metric 좌표의 3D box로 바꾸고 중복 후보를 제거한다. 진입점은 `cuda/decode_cuda.cu`의 `decode_and_nms()`다.

## 2. CUDA decode kernel

CUDA thread 하나가 feature map의 cell 하나를 담당한다. 총 thread 대상은 `468*468=219024`개다.

### 2.1 클래스와 점수

```cpp
label = argmax(hm[0:3, y, x]);
score = 1 / (1 + exp(-hm[label, y, x]));
```

원본 CenterPoint는 cell마다 세 클래스 중 최대값 하나를 선택한다. `score > 0.1`인 경우만 다음 계산을 수행한다.

### 2.2 중심 좌표

```cpp
x = (grid_x + reg_x) * 0.32F - 74.88F;
y = (grid_y + reg_y) * 0.32F - 74.88F;
z = height;
```

`reg`는 정수 cell 안의 소수 offset이다. `0.32`는 Waymo PointPillars voxel 크기이고 `-74.88`은 point-cloud range의 시작 좌표다. 이 config의 `out_size_factor`는 1이다.

### 2.3 크기와 회전

```cpp
dx = exp(dim_x);
dy = exp(dim_y);
dz = exp(dim_z);
yaw = atan2(rot_sin, rot_cos);
```

학습 시 크기를 log로 예측했기 때문에 exp로 양수 크기를 복원한다. 회전을 sin/cos 두 성분으로 예측하면 `-pi`와 `pi` 경계가 연속적이다.

### 2.4 후보 저장

공간 범위와 finite 검사를 통과한 thread가 `atomicAdd(count, 1)`로 출력 위치를 얻는다. Atomic 저장 순서는 실행마다 달라질 수 있으므로 CPU에서 `(score 내림차순, source_index 오름차순)`으로 다시 정렬해 결과를 결정적으로 만든다.

## 3. Rotated IoU

NMS는 Z 높이가 아닌 위에서 본 BEV 회전 사각형 IoU를 사용한다.

1. `(x,y,dx,dy,yaw)`로 네 corner를 만든다.
2. Sutherland-Hodgman 방식으로 두 convex polygon의 교집합을 자른다.
3. Shoelace 공식으로 교집합 넓이를 구한다.
4. `IoU = intersection / (area_a + area_b - intersection)`을 계산한다.

두 box의 IoU가 `0.7`보다 크면 점수가 낮은 box를 억제한다.

## 4. 원본과 같은 filtering 순서

```text
모든 468x468 cell
  -> class argmax + sigmoid
  -> score > 0.1
  -> center가 [-80,-80,-10] ~ [80,80,10]
  -> score순 최대 4096개
  -> class-agnostic rotated NMS, IoU 0.7
  -> 최대 500개
```

class-agnostic은 서로 다른 label 후보도 서로 억제할 수 있다는 뜻이다. 원본 `CenterHead.post_processing()`도 max class를 선택한 후 하나의 NMS에 전달한다.

## 5. 출력 파일

`detections.csv`는 사람이 읽고 비교하기 위한 파일이다.

```text
x,y,z,dx,dy,dz,yaw,score,label,source_index
```

`detections.bin`은 detection마다 float32 9개를 저장한다.

```text
[x,y,z,dx,dy,dz,yaw,score,label_as_float]
```

`source_index = grid_y * 468 + grid_x`이며 reference 비교와 디버깅에 사용한다.

## 6. 독립 검증

`tools/validate_reference.py`는 CUDA/C++ 결과를 그대로 복사하지 않고 NumPy/Python으로 전체 decode와 polygon NMS를 다시 수행한다.

```text
후보 수: 1380 == 1380
최종 수: 500 == 500
모든 source_index, label, 순서 일치
최대 수치 차이: 6.63617554e-6
반복 binary SHA-256 일치
```

## 7. 결과 해석 주의

현재 검증 입력은 KITTI point cloud이고 weight는 Waymo에서 학습됐다. 따라서 500개 cap에 도달한 결과를 검출 성능으로 해석하면 안 된다. 이 검증이 증명하는 것은 수식, tensor layout, NMS, 결정성이 reference와 일치한다는 점이다.

## 8. 다음 최적화

CUDA decode는 약 0.5ms 이하이며 현재 rotated NMS는 약 67ms다. 실제 Waymo 입력에서 후보 수를 다시 측정한 뒤 공간 prefilter, AABB reject, CUDA NMS bitmask 방식 등을 검토한다. 정확한 CPU 구현을 기준으로 유지해야 최적화 회귀를 잡을 수 있다.
