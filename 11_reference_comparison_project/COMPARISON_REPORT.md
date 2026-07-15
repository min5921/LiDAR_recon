# 원본 CenterPoint 비교 결과

## 1. Detection 지표

동일한 Waymo 5프레임, 동일 weight, score threshold 0.35, NMS IoU 0.5 조건이다.

| 입력 intensity | Prediction | TP | FP | FN | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| Raw | 14 | 10 | 4 | 27 | 0.7143 | 0.2703 |
| `tanh(intensity)` | 28 | 23 | 5 | 14 | 0.8214 | 0.6216 |
| 변화 | +14 | +13 | +1 | -13 | +0.1071 | +0.3514 |

원본 전처리 적용 후 TP가 13개 늘고 recall이 약 35.1%p 상승했다. FP 증가는 1개뿐이며 precision도 상승했다.

## 2. Heatmap 원인 분류

| 결과 | Raw | tanh |
|---|---:|---:|
| `DETECTED` | 10 | 23 |
| `LOW_HEATMAP_SCORE` | 25 | 12 |
| `HIGH_HEATMAP_EMITTED_UNMATCHED` | 2 | 2 |

낮은 heatmap GT가 25개에서 12개로 감소했다. Detection TP 증가량 13개와 정확히 대응한다.

## 3. 단계별 수치 검증

| 단계 | 범위 | 최대 절대 오차 | 결과 |
|---|---|---:|---|
| 실행 계약 | archive/config/executable/weight | - | 통과 |
| 결과 출처 | raw/tanh eval, RPN/Head weight | - | 통과 |
| 전처리 | 5프레임 XYZ/intensity/좌표/point count | 0 | 통과 |
| PFN | 5프레임 전체 pillar x 64 channel | 0.000004649 | 통과 |
| Scatter | 5프레임 `[1,64,468,468]` 전체 | 0 | exact 통과 |
| RPN | 19개 레이어 x 2 probe = 38개 | 0.000000834 | 통과 |
| CenterHead | 5프레임 GT peak 37개 | 0.000001907 | 통과 |

## 4. 결론

현재 C++/CUDA의 PFN, Scatter, RPN, CenterHead 연산은 checkpoint 수식의 독립 NumPy 계산과 모두 일치한다. 이전 낮은 recall의 가장 큰 원인은 Waymo intensity의 `tanh` 정규화 누락이었다.

아직 남은 14개 FN 중 12개는 tanh 적용 후에도 GT 주변 heatmap이 0.35 미만이고, 2개는 박스가 출력됐지만 IoU 0.5에 미달한다. 다음 비교 대상은 원본 Waymo point 생성 과정의 lidar return/NLZ 선택과 remaining GT의 거리·point count 분포다.

## 5. 제한 사항

이 Windows 환경에는 PyTorch/MMCV가 없어 원본 framework 자체의 end-to-end tensor dump와 비교하지는 못했다. 이번 결과는 원본 source preprocessing 규칙, checkpoint에서 직접 추출한 weight, 독립 NumPy 수식을 기준으로 한다.
