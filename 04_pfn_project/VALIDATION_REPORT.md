# Waymo PointPillars Checkpoint PFN 검증 결과

## Checkpoint

```text
centerpoint_waymo_pointpillars_50_novelocity.pth
epoch: 12
iteration: 26784
```

## 추출된 구조

```text
PFN layer 0 Linear: [32,10]
PFN layer 0 BN:     [32]
PFN layer 1 Linear: [64,64]
PFN layer 1 BN:     [64]
BatchNorm eps:      0.001
```

## 검증 입력

KITTI `[x,y,z,intensity]`에 `elongation=0`을 추가하고 기존 Voxelization과 Decoration을 실행했다.

```text
points:          [124668,5]
pillars:         [10404,20,5]
decorated input: [10404,20,10]
```

이 입력은 PFN 수치 검증용이며 Waymo detection 정확도 평가용이 아니다.

## C++ 실제 2단 PFN

```text
input:  [10404,20,10]
output: [10404,64]
CPU time: 205.883 ms
```

CPU 시간은 검증 당시 환경의 1회 측정값이며 정식 benchmark가 아니다.

## NumPy 비교

```text
allclose:      True
max abs diff:  0.00000334
mean abs diff: 0.00000007
```

첫 pillar의 앞 8개 output:

```text
NumPy:
[0.0000000, 1.7672750, 1.7743196, 0.0000000,
 0.9929256, 2.4505057, 1.1214296, 0.0000000]

C++:
[0.0000000, 1.7672750, 1.7743196, 0.0000000,
 0.9929254, 2.4505060, 1.1214294, 0.0000000]
```

## Scatter 연결 검증

실제 checkpoint PFN 출력을 `05_scatter_project`에 입력했다.

```text
PFN input:       [10404,64]
BEV output:      [1,64,468,468]
occupied cells:  10404
exactly equal:   True
max abs diff:    0
```

## 판정

체크포인트 PFN tensor 추출, 2단 PFN C++ 연산, NumPy 비교, Scatter 연결이 모두 통과했다.
