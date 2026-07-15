# 12 Waymo False Negative Analysis

이 마일스톤은 모델이 놓친 Waymo 차량을 단순히 `FN`으로만 세지 않고,
**어느 단계에서 왜 놓쳤는지** 실제 입력 점과 Head 출력으로 분류한다.

## 분석 대상

- 입력: `09_full_pipeline_project`가 만든 5프레임 평가 폴더
- GT: Waymo sensor archive의 `laser_labels.json`
- 점군: 각 LiDAR와 return의 6-feature 원본 bin
- 모델 출력: `07_head` heatmap과 최종 `detections.csv`
- 기본 평가: 같은 class의 rotated BEV IoU `>= 0.5`

## 실행

```powershell
python -B tools/analyze_waymo_false_negatives.py `
  --eval-dir "C:\Users\user\Documents\객체인지\waymo_eval_tanh_pcdet_5frames" `
  --output-dir "C:\Users\user\Documents\객체인지\waymo_fn_analysis_tanh_5frames"
```

`--low-point-threshold`의 기본값은 5다. GT 박스 안에서 실제 모델 입력으로
남은 점이 이 값보다 적으면 `LOW_POINT_COUNT`로 분류한다.

## 출력

```text
fn_analysis.json       전체 근거와 실행 계약
fn_analysis.csv        GT 한 개당 한 행인 표
fn_analysis_report.md  사람이 읽기 쉬운 요약
figures/*.png          프레임별 BEV 시각화
```

저장소에는 동일 실행의 압축 결과인 `fn_analysis_5frames.json`과
`fn_analysis_5frames.csv`도 보관한다. PNG는 용량 때문에 실행 결과 폴더에 둔다.

## 실제 5프레임 결과

| 항목 | 결과 |
|---|---:|
| Prediction | 28 |
| Waymo VEHICLE GT | 37 |
| TP / FP / FN | 25 / 3 / 12 |
| Precision | 0.8929 |
| Recall | 0.6757 |
| `LOW_MODEL_SCORE` | 9 |
| `LOW_POINT_COUNT` | 3 |

평가기와 독립 구현이 모두 같은 `25 / 3 / 12`를 냈다. 따라서 현재 남은
12개는 IoU 좌표계 오류가 아니라 실제 미검출이다.

## 테스트

```powershell
python -B -m unittest discover -s tests -v
```

테스트는 rotated IoU, Waymo CCW heading, prediction yaw 변환, 점군 필터링,
FN 분류 우선순위를 검사한다.

