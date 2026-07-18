# LiDAR_recon 재현 로그

이 문서는 보고서 작성 중 실제로 실행한 핵심 명령, 위치, 종료 코드, 결과와 대체 검증을 기록한다. 원본 알고리즘 파일은 수정하지 않았고 build/run 출력은 `%LOCALAPPDATA%/Temp/lidar_recon_report_be012d4`에 만들었다.

## 저장소 기준 상태

실행 위치: `C:\Users\user\Desktop\Onechip\Codex\my project`

### `git status --short`

- 종료 코드: `0`
- 출력: 없음
- 판정: 성공, 작업 시작 시 tracked worktree clean `[직접 실행]`

### `git rev-parse HEAD`

- 종료 코드: `0`
- 출력: `be012d4d065073a3e2e0e647a620abde5d535296`
- 판정: 성공 `[직접 실행]`

### `git branch --show-current`

- 종료 코드: `0`
- 출력: `main`
- 판정: 성공 `[직접 실행]`

### `git log --reverse --date=short --format="%h | %ad | %s"`

- 종료 코드: `0`
- 판정: 성공 `[직접 실행]`

```text
b352b64 | 2026-06-23 | first
acbc08e | 2026-07-04 | second
613b9e0 | 2026-07-04 | Add center head and decode projects
cb23607 | 2026-07-04 | Add Visual Studio voxelization project
143bcd3 | 2026-07-05 | 02_project_vs
c7a4555 | 2026-07-09 | Add Waymo full pipeline bridge and visualization
1ea3317 | 2026-07-12 | Harden Waymo decode diagnostics
b92bd06 | 2026-07-12 | Add CenterHead heatmap validation
21a7a27 | 2026-07-15 | Add reproducible CenterPoint reference validation
e850dfd | 2026-07-15 | Add Waymo false negative analysis
d0510fb | 2026-07-15 | Add Waymo operating point study
be012d4 | 2026-07-18 | Add GPU-resident CenterPoint inference pipeline
```

### 원격 저장소 확인

명령:

```powershell
git remote -v
```

- 종료 코드: `0`
- 핵심 출력: `origin https://github.com/min5921/LiDAR_recon.git`
- 판정: 사용자가 지정한 저장소와 로컬 저장소가 일치 `[직접 실행]`

## 경로 별칭

아래 로그에서 사용한 경로다.

```text
$REPO    = C:\Users\user\Desktop\Onechip\Codex\my project
$TMP     = C:\Users\user\AppData\Local\Temp\lidar_recon_report_be012d4
$PY      = C:\Users\user\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe
$WEIGHTS = C:\Users\user\Documents\객체인지\weights_full_novelocity
$WAYMO   = C:\Users\user\Documents\객체인지\waymo_eval_tanh_pcdet_5frames\frame_000
```

`$WEIGHTS`와 `$WAYMO`는 로컬 외부 자산이며 git tracked가 아니다. fresh clone 재현자는 직접 준비해야 한다.

## 환경 확인

| 명령 | 위치 | 종료 코드 | 핵심 출력 | 판정 |
|---|---|---:|---|---|
| `cmake --version` | `$REPO` | `0` | CMake `4.3.2` | 성공 `[직접 실행]` |
| `nvcc --version` | `$REPO` | `0` | CUDA `13.1`, V13.1.80 | 성공 `[직접 실행]` |
| `nvidia-smi` query | `$REPO` | `0` | RTX 5080, `16303 MiB`, driver `591.44` | 성공 `[직접 실행]` |
| `$PY --version` | `$REPO` | `0` | Python `3.12.13` | 성공 `[직접 실행]` |
| NumPy version 출력 | `$REPO` | `0` | NumPy `2.3.5` | 성공 `[직접 실행]` |
| Matplotlib import | `$REPO` | `1` | `ModuleNotFoundError: matplotlib` | 실패 `[직접 실행]` |
| Pillow import | `$REPO` | `0` | 사용 가능 | BEV 생성 대체 도구 `[직접 실행]` |

Matplotlib 부재 때문에 package를 설치하거나 원본 환경을 변경하지 않았다. NumPy와 Pillow만 사용하는 `report_tools/inspect_outputs.py`로 BEV를 생성했다.

## CMake 구성 시행착오

### 기본 Visual Studio generator

대표 명령:

```powershell
cmake -S 02_project -B $TMP\build_02
```

- 위치: `$REPO`
- 종료 코드: 실패
- 핵심 오류: MSBuild가 환경 변수 dictionary에 `PATH`와 `Path` 중복 key가 있다고 예외 발생
- 원인: 현재 Codex Windows process 환경의 대소문자 중복 변수
- 대체 방법: Visual Studio `VsDevCmd.bat`를 호출하고 Ninja generator와 Ninja executable을 명시

### 성공한 Ninja 구성 형식

```powershell
cmd /c "call C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat -arch=x64 -host_arch=x64 && cmake -S <project> -B <temp_build> -G Ninja -DCMAKE_MAKE_PROGRAM=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"
```

- 종료 코드: `0`
- 적용 대상: voxelization, pillar decoration, PFN, scatter, RPN CUDA, CenterHead CUDA, decode CUDA, Waymo inspector, GPU-resident project
- 판정: 성공 `[직접 실행]`

CUDA target은 필요할 때 다음 compiler도 명시했다.

```text
-DCMAKE_CUDA_COMPILER=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\nvcc.exe
```

빌드 중 CUDA header에서 Windows codepage `949` 관련 `C4819` 경고가 반복됐지만 link와 실행은 성공했다. 이번 작업에서는 warning으로 기록하고 원본 build option이나 source encoding을 수정하지 않았다.

### GPU-resident full target 후속 빌드

첫 전체 build 호출은 `centerpoint_gpu_preprocess`와 `centerpoint_gpu_rpn`을 만든 뒤 command session이 끝나 `centerpoint_gpu_full`이 남았다. developer environment 없이 target만 다시 빌드한 명령은 `cl.exe`를 찾지 못해 종료 코드 `1`이었다. 다음 명령으로 다시 실행했다.

```powershell
cmd /c "call <VsDevCmd.bat> -arch=x64 -host_arch=x64 && cmake --build $TMP\build_14_ninja --target centerpoint_gpu_full -j 2"
```

- 종료 코드: `0`
- 핵심 출력: device link 후 `centerpoint_gpu_full.exe` link 성공
- 판정: 대체 빌드 성공 `[직접 실행]`

## 커밋 아티팩트 독립 검산

명령:

```powershell
$PY report_tools\inspect_outputs.py `
  --repo-root $REPO `
  --json-out report\artifact_inspection.json `
  --bev-out report\assets\kitti_000000_bev.png
```

- 실행 위치: 보고서 staging 디렉터리
- 종료 코드: `0`
- 핵심 결과:
  - KITTI sample `[124668,4] float32`, `1994688 bytes`
  - committed voxel pillar `[10404,20,4]`, shape/dtype byte 크기 일치
  - committed decorated pillar `[10404,20,9]`, shape/dtype byte 크기 일치
  - committed PFN dump `[10404,64]`, dummy PFN으로 분류
  - project 13 threshold CSV 전 행의 precision/recall/F1 공식 일치
- 판정: 성공 `[아티팩트 검산]`, `[수식 검산]`

## 단계형 파이프라인 직접 실행

모든 출력은 `$TMP\run` 아래에 생성했다.

### KITTI feature 4 voxelization

```powershell
$TMP\build_02_v4\centerpoint_voxel_dump.exe `
  $REPO\00_reference\sample_data\kitti\000000.bin `
  $TMP\run\02_f4 4

$PY 02_project\tools\compare_python_cpp_voxelization.py `
  --points 00_reference\sample_data\kitti\000000.bin `
  --dump $TMP\run\02_f4 --feature-dim 4
```

- 위치: `$REPO`
- 종료 코드: 모두 `0`
- 핵심 출력: point `124668`, pillar `10404`, coordinates/num_points/pillars 모두 같음, max diff `0`
- 판정: 성공 `[직접 실행]`

### KITTI feature 5 fixture와 voxelization

```powershell
$PY 04_pfn_project\tools\make_kitti_feature5_fixture.py <input> <output>
$TMP\build_02_v4\centerpoint_voxel_dump.exe <feature5.bin> $TMP\run\02_f5 5
$PY 02_project\tools\compare_python_cpp_voxelization.py <same-contract-arguments>
```

- 종료 코드: 모두 `0`
- 핵심 출력: 입력 `[124668,4]`에서 elongation 영 값을 붙인 `[124668,5]`, point `124668`, pillar `10404`, max diff `0`
- 판정: 성공 `[직접 실행]`

### Pillar decoration

```powershell
$TMP\build_03_ninja\centerpoint_decorate_pillars.exe $TMP\run\02_f5 $TMP\run\03_f5
$PY 03_pillar_feature_project\tools\compare_python_cpp_pillar_feature.py --voxel-dump $TMP\run\02_f5 --decorated-dump $TMP\run\03_f5
```

- 종료 코드: 모두 `0`
- 핵심 출력: `[10404,20,5] -> [10404,20,10]`, NumPy equality `true`, max diff `0`
- 판정: 성공 `[직접 실행]`

### Checkpoint PFN

```powershell
$TMP\build_04_ninja\centerpoint_pfn_checkpoint.exe `
  $TMP\run\03_f5 `
  $REPO\04_pfn_project\weights\waymo_pointpillars_50_novelocity `
  $TMP\run\04_f5

$PY 04_pfn_project\tools\compare_python_cpp_pfn_checkpoint.py `
  --decorated-dump $TMP\run\03_f5 `
  --weight-dir $REPO\04_pfn_project\weights\waymo_pointpillars_50_novelocity `
  --pfn-dump $TMP\run\04_f5
```

- 종료 코드: 모두 `0`
- 핵심 출력: `[10404,20,10] -> [10404,64]`, allclose `true`, max diff `0.00000286`, mean diff `0.00000008`
- 단일 CPU 실행 시간: `3615.995 ms`, benchmark가 아님
- 주의: weight 디렉터리는 로컬 git-ignored 자산
- 판정: 성공 `[직접 실행]`

### BEV scatter

```powershell
$TMP\build_05_ninja\centerpoint_scatter.exe $TMP\run\04_f5 $TMP\run\02_f5 $TMP\run\05_f5
$PY 05_scatter_project\tools\compare_python_cpp_scatter.py --pfn-dump $TMP\run\04_f5 --voxel-dump $TMP\run\02_f5 --bev-dump $TMP\run\05_f5
```

- 종료 코드: 모두 `0`
- 핵심 출력: `[10404,64] -> [1,64,468,468]`, occupied `10404`, exact equality `true`
- 판정: 성공 `[직접 실행]`

### RPN CUDA

```powershell
$TMP\build_06_ninja\centerpoint_rpn_full_cuda.exe `
  $TMP\run\05_f5 `
  $REPO\06_rpn_project\weights\waymo_pointpillars_50_novelocity `
  $TMP\run\06_f5 --probes

$PY 06_rpn_project\tools\check_selected_rpn_values.py `
  --bev-dump $TMP\run\05_f5 `
  --weight-dir $REPO\06_rpn_project\weights\waymo_pointpillars_50_novelocity `
  --rpn-dump $TMP\run\06_f5
```

- 실행 종료 코드: `0`
- NumPy 선택값 검증 종료 코드: `0`
- 핵심 출력: 최종 `[1,384,468,468]`, selected checks `true`, 확인값 최대 diff `0.00000221`
- 단일 CUDA 실행 시간: `223.657 ms`, benchmark가 아님
- NumPy 재계산은 약 수 분 소요됐지만 process 종료까지 기다려 결과를 회수함
- 판정: 성공 `[직접 실행]`

### CenterHead CUDA

```powershell
$TMP\build_07_ninja\centerpoint_head_cuda.exe $TMP\run\06_f5 $REPO\07_center_head_project\weights $TMP\run\07_f5
$PY 07_center_head_project\tools\validate_selected.py --rpn-dir $TMP\run\06_f5 --weight-dir $REPO\07_center_head_project\weights --output-dir $TMP\run\07_f5
```

- 종료 코드: 모두 `0`
- 핵심 출력: reg `2`, height `1`, dim `3`, rot `2`, hm `3` channel, 공간 `[468,468]`, 선택값 max diff `3.5762786865234375e-7`
- 단일 CUDA 실행 시간: `247.378 ms`, benchmark가 아님
- 판정: 성공 `[직접 실행]`

### Decode legacy 기본 설정

```powershell
$TMP\build_08_ninja\centerpoint_decode.exe $TMP\run\07_f5 $TMP\run\08_f5_default
$PY 08_decode_project\tools\validate_reference.py --head-dir $TMP\run\07_f5 --decode-dir $TMP\run\08_f5_default
```

- 종료 코드: 모두 `0`
- 핵심 출력: score `0.1`, NMS `0.7`, convention `current`, 후보 `811`, 최종 `500`, max diff `7.62816163e-6`
- 판정: 성공 `[직접 실행]`

### Decode 운영 설정과 legacy validator 실패

```powershell
$TMP\build_08_ninja\centerpoint_decode.exe $TMP\run\07_f5 $TMP\run\08_f5 0.5 0.35 pcdet
$PY 08_decode_project\tools\validate_reference.py --head-dir $TMP\run\07_f5 --decode-dir $TMP\run\08_f5
```

- decode 종료 코드: `0`
- decode 핵심 출력: 후보 `54`, 최종 `20`
- validator 종료 코드: `1`
- 오류: `AssertionError: (20, 500)`
- 원인: validator가 score `0.1`, NMS `0.7`, `current`를 고정 사용해 실행기의 명시적 `0.35`, `0.5`, `pcdet` 설정을 반영하지 않음
- 대체 검증: legacy 설정은 같은 validator로 통과시켰고, 운영 설정을 인자로 받는 project 14 detection validator를 GPU-resident frame에 사용
- 판정: 예상 가능한 계약 불일치, 구현 결과 오류로 해석하지 않음

## GPU-resident 직접 실행

### 전체 pipeline, Waymo frame 000

```powershell
$TMP\build_14_ninja\centerpoint_gpu_full.exe `
  $WAYMO\points.bin $WEIGHTS `
  --output-dir $TMP\run\14_waymo_f0 --validation
```

- 위치: `$REPO`
- 종료 코드: `0`
- 핵심 출력: point `183680`, pillar `9529`, pre-NMS `28`, final detection `5`, intermediate tensor files `0`
- 출력: `detections.csv`, `summary.json`, `head_probes.json`, `pre_nms_candidates.csv`
- 판정: 성공 `[직접 실행]`

### GPU preprocess 전체 BEV 비교

```powershell
$PY 14_gpu_resident_pipeline_project\tools\compare_python_gpu_preprocess.py `
  --gpu-exe $TMP\build_14_ninja\centerpoint_gpu_preprocess.exe `
  --points $WAYMO\points.bin `
  --weight-dir $WEIGHTS\04_pfn `
  --output-dir $TMP\run\14_preprocess_f0
```

- 종료 코드: `0`
- 핵심 출력: input `183680`, valid `183505`, pillar `9529`, BEV `[1,64,468,468]`, strict allclose `true`, max diff `0.00000620`
- 판정: 성공 `[직접 실행]`

### GPU RPN probe 비교

```powershell
$TMP\build_14_ninja\centerpoint_gpu_rpn.exe $WAYMO\points.bin $WEIGHTS --output-dir $TMP\run\14_rpn_f0 --probes
$PY 14_gpu_resident_pipeline_project\tools\compare_python_gpu_rpn_probes.py $TMP\run\14_rpn_f0\rpn_probes.json $WEIGHTS\06_rpn
```

- 종료 코드: 모두 `0`
- 핵심 출력: `38/38` probe 통과, max diff `6.85453415e-7`
- 판정: 성공 `[직접 실행]`

전체 pipeline validation output에는 RPN probe 파일이 없다. 처음 `rpn_layer_probes.json`을 찾는 비교 명령은 `FileNotFoundError`로 종료 코드 `1`이었다. RPN 검증은 위의 전용 `centerpoint_gpu_rpn --probes` 경로로 대체했다.

### GPU CenterHead와 detection 비교

```powershell
$PY 14_gpu_resident_pipeline_project\tools\compare_python_gpu_head_probes.py `
  $TMP\run\14_waymo_f0\head_probes.json $WEIGHTS\07_head

$PY 14_gpu_resident_pipeline_project\tools\validate_gpu_detections.py `
  --pre-nms $TMP\run\14_waymo_f0\pre_nms_candidates.csv `
  --detections $TMP\run\14_waymo_f0\detections.csv `
  --reference-head-dir $WAYMO\07_head `
  --reference-detections $WAYMO\08_detections\detections.csv `
  --score-threshold 0.35 --nms-iou 0.5 --nms-convention pcdet
```

- 종료 코드: 모두 `0`
- CenterHead: `22/22` probe, max diff `3.81469727e-6`
- Decode: `28/28`, max diff `8.92439954e-6`
- GPU NMS vs Python: `5/5`, max diff `0`
- Full pipeline vs staged reference: `5/5`, max diff `7.15e-6`
- 판정: 성공 `[직접 실행]`

## Python unit test

```powershell
$PY -m unittest 11_reference_comparison_project\tests\test_validation_contracts.py
$PY -m unittest 12_waymo_fn_analysis_project\tests\test_fn_analysis.py
$PY -m unittest 13_waymo_operating_point_project\tests\test_operating_points.py
```

| test | 종료 코드 | 결과 |
|---|---:|---|
| reference comparison contracts | `0` | `7` tests, OK `[직접 실행]` |
| false-negative analysis | `0` | `9` tests, OK `[직접 실행]` |
| operating-point analysis | `0` | `6` tests, OK `[직접 실행]` |

## 값 검증과 문서 QA

```powershell
$PY -m json.tool report\ACTUAL_VALUES.json
$PY -m py_compile report_tools\inspect_outputs.py
```

- JSON parse 종료 코드: `0`
- Python in-memory compile 종료 코드: `0`
- `report_tools/inspect_outputs.py`를 저장소 루트에서 기본 인자로 재실행한 종료 코드: `0`
- Markdown 로컬 링크 검사: 누락 `0`, 종료 코드 `0`
- 보고서의 주요 표는 `검증 상태`와 `출처` 열을 포함하도록 검토
- 숫자 상태는 지정된 다섯 종류만 사용하도록 검색 검토

## 최종 작업 트리 확인

최종 복사 후 아래 명령을 실행했다.

```powershell
git status --short
git diff --stat
git diff --check
```

- `git status --short` 종료 코드: `0`
- 출력: `?? report/`, `?? report_tools/`
- `git diff --stat` 종료 코드: `0`, 출력 없음. 이유는 모든 산출물이 아직 untracked이고 기존 tracked 파일에는 diff가 없기 때문
- `git diff --check` 종료 코드: `0`, 출력 없음
- 최종 변경 범위: `report/`, `report_tools/`만
- 기존 C++/CUDA/Python 알고리즘 tracked 파일 변경: 없음
- commit, push, branch 생성: 수행하지 않음
