# Waymo 5프레임 False Negative 분석 결과

## 1. 이번에 발견한 평가 오류

기존 09 평가기는 Waymo label heading을 시계 방향처럼 회전해 GT 박스를
좌우 반전된 방향으로 만들었다. 중심이 거의 같은 예측 2개가 IoU `0.4999`,
`0.4753`으로 계산되어 FN과 FP로 잘못 분류됐다.

공식 CCW 회전과 CenterPoint prediction yaw 변환을 적용한 결과는 다음과 같다.

| 평가 상태 | TP | FP | FN | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| 수정 전 evaluator | 23 | 5 | 14 | 0.8214 | 0.6216 |
| 공식 좌표계 독립 재계산 | 25 | 3 | 12 | 0.8929 | 0.6757 |
| 수정 후 evaluator | 25 | 3 | 12 | 0.8929 | 0.6757 |

수정 후 평가기와 독립 분석기의 지표 차이는 정확히 0이다.

## 2. 회전 방향의 추가 증거

Waymo label이 제공하는 `num_lidar_points_in_box`와 직접 센 점 수를 비교했다.

| 회전 방법 | 전체 점 수 | 공식 점 수와 MAE | exact match |
|---|---:|---:|---:|
| Waymo 공식 label | 7,857 | - | 37개 기준 |
| CCW heading | 7,584 | 7.378 | 12 |
| 기존 mirrored heading | 5,430 | 70.622 | 2 |

archive의 파생 점과 공식 label 생성 입력이 완전히 같지는 않아 exact match가
37개는 아니지만, CCW 방식의 오차가 기존 방식보다 약 9.6배 작다.

## 3. Raw와 원본 전처리 비교

동일한 5프레임과 weight에서 intensity만 바꾸었다.

| 입력 | Prediction | TP | FP | FN | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| raw intensity | 14 | 12 | 2 | 25 | 0.8571 | 0.3243 |
| `tanh(intensity)` | 28 | 25 | 3 | 12 | 0.8929 | 0.6757 |
| 변화 | +14 | +13 | +1 | -13 | +0.0357 | +0.3514 |

원본 CenterPoint loader의 `tanh` 규칙이 recall 개선의 가장 큰 요인이라는 결론은
그대로 유지된다.

## 4. 남은 12개 FN

| 분류 | 개수 | 의미 |
|---|---:|---|
| `LOW_MODEL_SCORE` | 9 | 점은 있지만 GT 주변 heatmap이 0.35 미만 |
| `LOW_POINT_COUNT` | 3 | 실제 모델 입력으로 남은 박스 내부 점이 5개 미만 |

FN의 effective point 수는 최소 0, 중앙값 6, 평균 6.83, 최대 16이다.

대표 사례:

| Frame | 거리 | 유효 점 | 주변 최대 score | 분류 |
|---|---:|---:|---:|---|
| frame_000 | 44.1 m | 10 | 0.1266 | `LOW_MODEL_SCORE` |
| frame_001 | 26.5 m | 6 | 0.1856 | `LOW_MODEL_SCORE` |
| frame_003 | 27.4 m | 3 | 0.1801 | `LOW_POINT_COUNT` |
| frame_004 | 60.6 m | 0 | 0.0101 | `LOW_POINT_COUNT` |

## 5. 단계별 수치 검증

| 단계 | 검증 범위 | 최대 절대 오차 | 결과 |
|---|---|---:|---|
| 전처리 | 5프레임 | 0 | 통과 |
| PFN | 모든 pillar x 64 channel | 0.000004649 | 통과 |
| Scatter | `[1,64,468,468]` 전체 | 0 | exact 통과 |
| RPN | 38개 layer probe | 0.000000834 | 통과 |
| CenterHead | GT peak 37개 | 0.000001907 | 통과 |
| 평가 geometry | evaluator vs 독립 구현 | 지표 차이 0 | 통과 |

## 6. 결론과 다음 단계

CUDA 연산, checkpoint weight 연결, decode, 공식 좌표계 평가까지 수치적으로
일관된 상태다. 현재 5프레임에서 남은 오차는 구현 버그보다 sparse object와
낮은 model confidence에 집중된다.

다음 마일스톤은 더 많은 프레임에서 score threshold를 구간별로 sweep하고,
거리와 point count별 precision/recall을 계산하는 것이다. 5프레임만 보고
threshold를 낮추면 우연히 FP가 적게 보일 수 있으므로 최소 수백 프레임에서
정한 뒤 Waymo 공식 metric 형식으로 확장하는 편이 안전하다.

