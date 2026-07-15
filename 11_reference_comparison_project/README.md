# 11 Reference Comparison Project

이 프로젝트는 현재 C++/CUDA PointPillars 파이프라인을 원본 CenterPoint의 Waymo 입력 규칙과 checkpoint 연산에 맞춰 단계별로 비교한다.

## 이번에 찾은 핵심 차이

원본 `det3d/datasets/pipelines/loading.py`의 `read_single_waymo()`는 다음 전처리를 수행한다.

```text
intensity = tanh(intensity)
```

기존 `09` exporter는 raw intensity를 그대로 사용했다. 현재 exporter와 다중 프레임 runner는 `tanh`를 기본값으로 사용한다.

## 비교 계약과 캐시 검증

다중 프레임 runner는 각 프레임에 `pipeline_cache_manifest.json`을 기록한다. archive, frame, 전처리, Decode 설정, 실행 파일 SHA-256, weight SHA-256이 모두 같을 때만 `--skip-existing` 결과를 재사용한다.

전체 비교기는 raw 실행이 `none`, reference 실행이 `tanh`인지 확인하고 다음 조건이 동일한 경우에만 비교한다.

- archive와 frame 목록
- lidar/return/NLZ 설정
- score threshold, NMS, match IoU
- Python/NumPy, 실행 파일, weight 서명
- RPN/Head 보조 검증의 eval 및 weight 출처

조건이 다르거나 전처리 tensor 비교가 하나라도 실패하면 JSON의 `stage_validation`이 실패하고 프로그램도 0이 아닌 종료 코드를 반환한다.

## RPN Probe 실행

```powershell
06_rpn_project\build_cuda\Release\centerpoint_rpn_full_cuda.exe `
  <05_scatter_dir> <06_rpn_weight_dir> <probe_output_dir> `
  --summary-only --probes

python tools\validate_rpn_layer_probes.py `
  --probe-json <probe_output_dir>\rpn_layer_probes.json `
  --weight-dir <06_rpn_weight_dir> `
  --output-json <probe_output_dir>\rpn_probe_validation.json
```

`--probes`는 각 RPN Conv/Deconv 레이어에서 중앙과 경계 위치의 입력값과 CUDA 출력값을 저장한다. 일반 추론에서는 사용하지 않으므로 성능에 영향을 주지 않는다.

## 전체 비교 보고서 생성

`tools/run_reference_comparison.py`는 다음 결과를 하나의 JSON으로 묶는다.

- raw intensity와 tanh intensity의 5프레임 detection 지표
- 원본 전처리 적용 여부
- 5프레임 PFN NumPy 비교
- 5프레임 Scatter exact 비교
- RPN layer probe 비교
- CenterHead GT peak 비교

회귀 테스트는 다음 명령으로 실행한다.

```powershell
python -B -m unittest discover `
  -s 11_reference_comparison_project\tests -v
```

실행 인수와 결과 해석은 `IMPLEMENTATION_GUIDE.md`와 `COMPARISON_REPORT.md`를 참고한다.
