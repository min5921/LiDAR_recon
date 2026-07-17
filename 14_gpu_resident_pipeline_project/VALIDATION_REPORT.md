# GPU-Resident CenterPoint 전체 검증 결과

## 검증 환경

```text
GPU: RTX 5080, 16 GB reported by nvidia-smi
CUDA: 13.1
input: Waymo tanh-intensity frame_000 ~ frame_004
point feature: [x,y,z,intensity,elongation]
checkpoint: centerpoint_waymo_pointpillars_full_novelocity
default threshold: score 0.35, rotated NMS IoU 0.5, pcdet convention
```

## 1. 전처리 전체 BEV 비교

`tools/compare_python_gpu_preprocess.py`가 동일 point와 PFN weight에서
voxelization, decoration, PFN, scatter를 NumPy로 다시 계산했다. 비교 대상은
전체 BEV `[1,64,468,468]`이다.

| Frame | Input points | Valid points | Pillars | Max abs diff | Result |
|---|---:|---:|---:|---:|---|
| `frame_000` | 183,680 | 183,505 | 9,529 | 0.00000381 | PASS |
| `frame_001` | 183,408 | 183,242 | 9,542 | 0.00000453 | PASS |
| `frame_002` | 182,276 | 182,105 | 9,585 | 0.00000525 | PASS |
| `frame_003` | 183,651 | 183,464 | 9,583 | 0.00000358 | PASS |
| `frame_004` | 184,307 | 184,118 | 9,793 | 0.00000381 | PASS |

모든 frame이 `rtol=1e-5`, `atol=2e-5`를 통과했다. 전체 최대 절대 오차는
`5.25e-6`이다.

## 2. RPN layer-local NumPy 비교

CUDA RPN 19개 layer에서 두 위치씩 총 38개 probe를 수집했다. Python은 원본
binary weight와 probe 입력으로 `Conv/Deconv -> BN -> ReLU`를 직접 계산했다.

| Frame | Probes | Failed | Max abs diff | Result |
|---|---:|---:|---:|---|
| `frame_000` | 38 | 0 | 0.000000685 | PASS |
| `frame_001` | 38 | 0 | 0.000000685 | PASS |
| `frame_002` | 38 | 0 | 0.000000954 | PASS |
| `frame_003` | 38 | 0 | 0.000001192 | PASS |
| `frame_004` | 38 | 0 | 0.000000685 | PASS |

5개 frame의 `190/190` probe가 `rtol=1e-5`, `atol=2e-4`를 통과했다.

## 3. CenterHead layer-local NumPy 비교

shared Conv, 다섯 hidden Conv, 다섯 output Conv에서 각각 두 위치를 검사했다.
Python은 Conv bias와 BN/ReLU 포함 여부를 branch 계약대로 독립 계산했다.

| Frame | Probes | Failed | Max abs diff | Result |
|---|---:|---:|---:|---|
| `frame_000` | 22 | 0 | 0.000003815 | PASS |
| `frame_001` | 22 | 0 | 0.000003815 | PASS |
| `frame_002` | 22 | 0 | 0.000003815 | PASS |
| `frame_003` | 22 | 0 | 0.000003815 | PASS |
| `frame_004` | 22 | 0 | 0.000003815 | PASS |

5개 frame의 `110/110` probe가 `rtol=1e-5`, `atol=2e-4`를 통과했다.

## 4. Decode와 rotated NMS 비교

`tools/validate_gpu_detections.py`가 기존 `07_head` map 전체에서 NumPy decode를
수행해 GPU pre-NMS 후보와 비교했다. 이어서 새 CUDA 후보에 Python rotated NMS를
적용하고, 마지막으로 기존 `08_detections` 결과와 전체 pipeline 출력을 비교했다.

| Frame | Pre-NMS | Final | Decode max diff | NMS max diff | Reference max diff |
|---|---:|---:|---:|---:|---:|
| `frame_000` | 28/28 | 5/5 | 0.000008924 | 0 | 0.000007150 |
| `frame_001` | 32/32 | 7/7 | 0.000007780 | 0 | 0.000004770 |
| `frame_002` | 26/26 | 6/6 | 0.000007799 | 0 | 0.000005841 |
| `frame_003` | 28/28 | 5/5 | 0.000008956 | 0 | 0.000004770 |
| `frame_004` | 29/29 | 5/5 | 0.000007986 | 0 | 0.000003900 |

decode 후보 `143/143`, 최종 detection `28/28`의 개수, score 순서, label,
source index가 모두 일치했다. Python NMS와 CUDA NMS의 최종 row 값은 정확히
같았고, 전체 pipeline과 기존 기준의 최대 절대 오차는 `7.15e-6`이었다.

## 5. GPU-resident 흐름과 실행 시간

Production 실행은 다음 두 파일만 선택적으로 출력한다.

```text
detections.csv
summary.json
intermediate_tensor_files: 0
```

검증 모드에서는 `head_probes.json`, `pre_nms_candidates.csv`를 추가하지만 전체
BEV, RPN, head tensor는 저장하지 않는다.

| Frame | Preprocess ms | RPN ms | CenterHead ms | Decode/NMS ms |
|---|---:|---:|---:|---:|
| `frame_000` | 1.868 | 183.327 | 24.029 | 4.477 |
| `frame_001` | 1.993 | 246.275 | 23.769 | 4.607 |
| `frame_002` | 1.923 | 232.481 | 24.289 | 4.338 |
| `frame_003` | 2.559 | 195.203 | 24.128 | 6.257 |
| `frame_004` | 1.994 | 178.915 | 41.809 | 19.119 |

이는 단일 실행 관찰값이며 정식 benchmark가 아니다. 측정 중 다른 프로세스가
GPU 사용률 `100%`, VRAM `15639/16303 MiB`를 사용한 시점도 확인되어 성능 비교는
GPU가 유휴 상태일 때 별도로 반복해야 한다. 재사용 Conv workspace 적용 전후의
CenterHead 관찰값은 약 `2280 ms -> 22~42 ms`였지만 이것도 동일 부하 조건의
benchmark로 해석하면 안 된다.

## 판정

`points -> voxelization -> PFN -> scatter -> RPN -> CenterHead -> decode ->
rotated NMS -> detections`가 하나의 독립 실행 파일 안에서 GPU-resident 흐름으로
연결되었다. 5개 Waymo frame에서 stage-local NumPy 계산과 기존 전체 결과에 대한
numerical parity를 통과했다. 남은 검증은 실제 Waymo ground truth를 사용하는
정량 AP 평가와 유휴 GPU에서의 반복 성능 측정이다.
