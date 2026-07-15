# 10 Head Validation 결과

> 후속 검증: `11_reference_comparison_project`에서 원본 Waymo loader의
> `tanh(intensity)` 누락을 발견했다. 이를 적용한 5프레임 결과는 TP 23,
> recall 0.6216이며 `LOW_HEATMAP_SCORE`가 25개에서 12개로 감소했다.
> 이 문서 아래의 수치는 수정 전 raw intensity 실행을 기록한 것이다.

## 검증 대상

- 평가 입력: `waymo_eval_review_pcdet_5frames`
- 프레임: `frame_000`부터 `frame_004`
- Waymo GT: VEHICLE 37개
- Decode score threshold: 0.35
- GT 주변 검사 범위: 반경 2 cell, 즉 최대 5 x 5 영역

## Heatmap 원인 분류

| 결과 | 개수 | 의미 |
|---|---:|---|
| `DETECTED` | 10 | 최종 IoU 0.5 매칭 성공 |
| `LOW_HEATMAP_SCORE` | 25 | GT 주변 최대 score가 0.35 미만 |
| `HIGH_HEATMAP_EMITTED_UNMATCHED` | 2 | 박스는 출력됐지만 IoU 0.5 미만 |
| `HIGH_HEATMAP_NOT_EMITTED` | 0 | 높은 heatmap이 Decode/NMS에서 사라진 사례 없음 |
| `CLASS_CONFLICT_AT_PEAK` | 0 | 다른 클래스 argmax로 바뀐 사례 없음 |
| `OUT_OF_RANGE` | 0 | 현재 BEV 범위 밖 GT 없음 |

낮은 heatmap 25개 중 14개는 score 0.1 미만이고, 4개는 score 0.01 미만이다. 나머지 11개는 0.1 이상 0.35 미만이다. threshold를 낮추면 일부 후보는 복구되지만, 14개는 0.1 threshold에서도 후보가 되지 않는다.

`LOW_HEATMAP_SCORE`의 주변 최대 score 통계:

```text
min  = 0.00004987
mean = 0.10475138
max  = 0.34261072
```

## 높은 Heatmap이지만 IoU 실패한 두 사례

```text
frame_003: score=0.54475, center error=0.350 m, best IoU=0.48478
frame_004: score=0.71504, center error=0.349 m, best IoU=0.47643
```

두 박스 모두 최종 `detections.csv`에 존재한다. NMS가 삭제한 것이 아니라 중심·크기·회전 오차가 합쳐져 평가 기준 IoU 0.5에 조금 못 미친 사례다.

## CUDA CenterHead 독립 수치 검증

GT 주변 peak 37개에서 RPN feature와 CenterHead weight를 사용해 NumPy로 heatmap branch를 다시 계산했다.

```text
samples      = 37
frames       = 5
tolerance    = 0.0002
max_abs_diff = 0.00000667572
passed       = true
```

CUDA 출력과 독립 계산이 충분히 일치하므로 `07_center_head_project`의 convolution 구현 오류가 낮은 heatmap의 주원인일 가능성은 낮다.

## 결론

현재 낮은 recall의 가장 큰 원인은 NMS가 아니다. 27개 FN 중 25개는 Decode 전에 GT 주변 heatmap이 threshold보다 낮다. 나머지 2개는 최종 박스까지 생성됐지만 IoU 0.5에 근소하게 미달했다.

다음 분석 우선순위는 다음과 같다.

1. 현재 full checkpoint가 기대하는 Waymo point preprocessing과 입력 feature scale을 Python reference와 비교한다.
2. `04 PFN -> 05 Scatter -> 06 RPN`의 중간 tensor를 checkpoint 원본 프레임워크와 비교한다.
3. threshold 0.1 이상 0.35 미만인 GT 11개는 box regression과 FP 증가량을 함께 보며 별도로 검토한다.
