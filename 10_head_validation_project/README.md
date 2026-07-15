# 10 Head Validation Project

이 프로젝트는 `07_center_head_project`의 raw heatmap 출력과 Waymo 정답 중심을 직접 비교한다.
목적은 낮은 recall이 CenterHead 이전에서 발생했는지, `08_decode_project` 이후에서 발생했는지 구분하는 것이다.

## 입력

- `09_full_pipeline_project/tools/run_waymo_multiframe_eval.py`가 만든 평가 폴더
- 각 프레임의 `07_head/hm.bin`: `[1, 3, 468, 468]` float32 raw logit
- 각 프레임의 `08_detections/decode_config.json`
- Waymo derived sensor archive의 `labels/laser_labels.json`

## 실행

```powershell
python tools/audit_head_heatmap.py `
  --eval-dir "C:\Users\user\Documents\객체인지\waymo_eval_review_pcdet_5frames" `
  --output-dir "C:\Users\user\Documents\객체인지\waymo_head_validation_5frames"
```

`--archive`를 생략하면 첫 프레임의 `export_summary.json`에서 원본 archive 경로를 읽는다.

CUDA 결과를 GT peak 위치에서 NumPy로 독립 재계산하려면 다음 명령을 실행한다.

```powershell
python tools/validate_gt_heatmap_reference.py `
  --eval-dir "C:\Users\user\Documents\객체인지\waymo_eval_review_pcdet_5frames" `
  --weight-dir "C:\Users\user\Documents\객체인지\weights_full_novelocity\07_head" `
  --audit-csv "C:\Users\user\Documents\객체인지\waymo_head_validation_5frames\gt_heatmap_audit.csv" `
  --output-json "C:\Users\user\Documents\객체인지\waymo_head_validation_5frames\gt_peak_reference_validation.json"
```

## 출력

- `gt_heatmap_audit.csv`: 모든 GT 중심의 center score, 주변 최고 score, 최종 검출 여부
- `top_heatmap_peaks.csv`: 프레임·클래스별 상위 local peak
- `head_validation_summary.json`: 원인 분류 집계
- `gt_peak_reference_validation.json`: GT peak의 NumPy 재계산과 CUDA 출력 오차
- `frame_*_<class>_heatmap.png`: GT와 상위 peak를 표시한 heatmap

두 JSON 보고서에는 입력 eval의 `run_contract`가 함께 저장된다. 이후 비교기는 이 값으로 같은 폴더에 남아 있던 이전 Head 결과가 섞이지 않았는지 확인한다.

## 원인 분류

- `DETECTED`: 최종 IoU 매칭까지 성공
- `LOW_HEATMAP_SCORE`: GT 주변 heatmap이 Decode threshold보다 낮음
- `HIGH_HEATMAP_EMITTED_UNMATCHED`: 같은 heatmap cell의 박스가 출력됐지만 GT IoU 매칭 실패
- `HIGH_HEATMAP_NOT_EMITTED`: heatmap은 높지만 해당 cell이 최종 출력에서 사라짐
- `CLASS_CONFLICT_AT_PEAK`: 같은 cell에서 다른 클래스 logit이 더 높음
- `OUT_OF_RANGE`: GT 중심이 현재 468 x 468 BEV 범위 밖

이 검사는 이미 CUDA로 계산된 `hm.bin`을 읽는 독립 검증 단계다. GPU 연산을 다시 구현하지 않으므로 `.cu` 파일은 필요하지 않다.
