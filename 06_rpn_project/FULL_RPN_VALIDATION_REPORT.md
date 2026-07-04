# 전체 RPN CUDA 검증 보고서

## 사용한 입력

```text
Checkpoint:
centerpoint_waymo_pointpillars_50_novelocity.pth

BEV:
실제 checkpoint 2단 PFN 출력 + 기존 Scatter
shape [1,64,468,468]
```

## Weight 추출

```text
RPN layers: 19
float tensors: 95
총 binary 크기: 19,249,152 bytes
```

`num_batches_tracked` 19개는 inference 계산에 필요하지 않아 제외했다.

## Shape 검증

```text
Block 0:   [1,64,468,468]
Deblock 0: [1,128,468,468]

Block 1:   [1,128,234,234]
Deblock 1: [1,128,468,468]

Block 2:   [1,256,117,117]
Deblock 2: [1,128,468,468]

Output:    [1,384,468,468]
```

## 출력 통계

```text
minimum: 0
maximum: 70.2753
sum: 1.98693e+07
non-finite values: 0
```

## 반복 실행 검증

전체 `84,105,216`개 float output을 두 번 생성해 비교했다.

```text
exactly equal: True
max abs diff: 0

SHA-256:
d3a8a47868bb9af9ec61683fe074645b68e710b1855d9133671114dd4e5f3bd7
```

## CPU scalar 선택 위치 검증

실제 BEV와 실제 checkpoint weight를 사용해 Python이 필요한 receptive field만 scalar로 재귀 계산했다.

```text
deblock0 c=0 y=0 x=0
CPU  0.35024709
CUDA 0.35024720
diff 0.00000012

deblock0 c=7 y=10 x=11
diff 0.00000036

deblock1 c=9 y=10 x=11
CPU  0.62342685
CUDA 0.62342465
diff 0.00000221

deblock2 c=1 y=0 x=1
CPU  0.20457134
CUDA 0.20457174
diff 0.00000040
```

세 branch 모두 선택 위치 검증을 통과했다. 이 검사는 Conv, stride/padding, BatchNorm, ReLU, 1x/2x/4x deblock weight 방향과 concat channel 위치를 포함한다.

## 실행 시간

RTX 5080에서 측정한 단일 실행:

```text
warm-up 포함: 188.651 ms
run 1:        150.723 ms
run 2:        147.257 ms
```

현재 구현은 layer마다 weight upload, `cudaMalloc/cudaFree`, im2col allocation을 반복하므로 최적화 전 기준이다. 정식 benchmark가 아니다.

## 판정

실제 checkpoint weight 추출, 전체 `.cu` RPN 실행, shape, finite, deterministic hash, 세 branch의 독립 scalar 비교가 모두 통과했다.
