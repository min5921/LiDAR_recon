# 10 Head Validation 결과

## 검증 대상

- 평가 입력: `waymo_eval_tanh_pcdet_5frames`
- 프레임: `frame_000`부터 `frame_004`
- Waymo GT: VEHICLE 37개
- Decode score threshold: 0.35
- GT 주변 검사 범위: 반경 2 cell, 즉 최대 `5 x 5`
- 평가 geometry: Waymo 공식 CCW heading

## Heatmap 원인 분류

| 결과 | 개수 | 의미 |
|---|---:|---|
| `DETECTED` | 25 | 최종 rotated IoU 0.5 매칭 성공 |
| `LOW_HEATMAP_SCORE` | 12 | GT 주변 최대 score가 0.35 미만 |
| `HIGH_HEATMAP_EMITTED_UNMATCHED` | 0 | 높은 heatmap 박스의 미매칭 사례 없음 |
| `HIGH_HEATMAP_NOT_EMITTED` | 0 | 높은 heatmap이 Decode/NMS에서 사라진 사례 없음 |
| `CLASS_CONFLICT_AT_PEAK` | 0 | 다른 class argmax로 바뀐 사례 없음 |
| `OUT_OF_RANGE` | 0 | 현재 BEV 범위 밖 GT 없음 |

raw intensity 실행에서는 `DETECTED=12`, `LOW_HEATMAP_SCORE=25`였다. 원본
CenterPoint 규칙인 `tanh(intensity)`를 적용해 낮은 heatmap 사례 13개가
정상 검출로 바뀌었다.

## 평가 Geometry 수정

기존 평가기는 Waymo GT heading을 반대 방향으로 회전했다. 다음 두 박스는
Head에서 높은 score로 출력됐지만 잘못된 GT polygon 때문에 IoU가 낮게 보였다.

```text
frame_001: score=0.7074, old IoU=0.4999, official IoU=0.6918
frame_004: score=0.7163, old IoU=0.4753, official IoU=0.6898
```

Waymo label은 공식 CCW heading을 그대로 쓰고, prediction은
`-prediction_yaw - pi/2`로 변환하도록 수정한 뒤 두 박스 모두 TP가 됐다.

## CUDA CenterHead 독립 수치 검증

GT 주변 peak 37개에서 RPN feature와 CenterHead weight를 사용해 NumPy로
heatmap branch를 다시 계산했다.

```text
samples      = 37
frames       = 5
tolerance    = 0.0002
max_abs_diff = 0.00000190735
passed       = true
```

CUDA 출력과 독립 계산이 충분히 일치하므로 CenterHead convolution 구현 오류가
낮은 heatmap의 주원인일 가능성은 낮다.

## 결론

최종 `tanh` 실행은 `TP=25`, `FP=3`, `FN=12`, recall `0.6757`이다. 남은
12개 FN은 12번째 마일스톤에서 원본 점군까지 추적했고, 낮은 model score 9개와
유효 점 부족 3개로 분류됐다.
