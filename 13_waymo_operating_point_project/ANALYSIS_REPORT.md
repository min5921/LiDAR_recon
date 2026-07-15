# Waymo 198프레임 Operating Point 분석

## 1. 데이터와 조건

```text
segment frames: 198
GT labels: 4,337
GT class: VEHICLE 4,337
intensity: tanh
NMS IoU: 0.5
NMS convention: pcdet
match IoU: 0.5
source score threshold: 0.05
```

이번 결과는 Waymo 공식 mAP/mAPH가 아니라 현재 C++/CUDA pipeline의
threshold 선택과 미검출 특성을 확인하기 위한 BEV IoU 평가다.

## 2. 전체 Threshold Sweep

| Threshold | Prediction | TP | FP | FN | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 48,503 | 3,351 | 45,152 | 986 | 0.0691 | 0.7727 | 0.1268 |
| 0.10 | 10,971 | 3,142 | 7,829 | 1,195 | 0.2864 | 0.7245 | 0.4105 |
| 0.15 | 5,547 | 2,989 | 2,558 | 1,348 | 0.5388 | 0.6892 | 0.6048 |
| 0.20 | 3,970 | 2,901 | 1,069 | 1,436 | 0.7307 | 0.6689 | 0.6984 |
| 0.25 | 3,357 | 2,820 | 537 | 1,517 | 0.8400 | 0.6502 | 0.7330 |
| 0.30 | 3,012 | 2,745 | 267 | 1,592 | 0.9114 | 0.6329 | **0.7470** |
| 0.35 | 2,816 | 2,669 | 147 | 1,668 | 0.9478 | 0.6154 | 0.7463 |
| 0.40 | 2,683 | 2,596 | 87 | 1,741 | 0.9676 | 0.5986 | 0.7396 |
| 0.50 | 2,435 | 2,404 | 31 | 1,933 | 0.9873 | 0.5543 | 0.7100 |
| 0.60 | 2,185 | 2,177 | 8 | 2,160 | 0.9963 | 0.5020 | 0.6676 |
| 0.70 | 1,812 | 1,812 | 0 | 2,525 | 1.0000 | 0.4178 | 0.5894 |

5프레임에서는 `0.50`이 가장 좋아 보였지만, 198프레임에서는 `0.30`이 최대
F1이다. 표본을 늘리기 전에 threshold를 확정하면 안 되는 이유가 실제로
확인됐다.

## 3. 추천 가능한 두 운영점

### 균형 우선

```text
threshold = 0.30
precision = 0.9114
recall    = 0.6329
F1        = 0.7470
```

기존 `0.35`보다 recall이 약 1.75%p 높고 precision은 약 3.64%p 낮다. F1
차이는 약 0.0008로 매우 작으므로 FP 비용이 크다면 기존 `0.35`도 합리적이다.

### Recall 우선, precision 0.80 이상

```text
threshold = 0.25
precision = 0.8400
recall    = 0.6502
F1        = 0.7330
```

`0.30`보다 TP 75개를 더 복구하지만 FP가 270개 더 생긴다.

## 4. 거리별 Recall at 0.25

| 거리 | GT | TP | FN | Recall |
|---|---:|---:|---:|---:|
| 0-30 m | 2,087 | 1,799 | 288 | 0.8620 |
| 30-50 m | 1,088 | 585 | 503 | 0.5377 |
| 50-75 m | 1,060 | 434 | 626 | 0.4094 |
| 75+ m | 102 | 2 | 100 | 0.0196 |

50m 이후 recall이 크게 떨어지고 75m 이상은 사실상 검출되지 않는다.

## 5. GT Point-count별 Recall at 0.25

| GT 내부 점 | GT | TP | FN | Recall |
|---|---:|---:|---:|---:|
| 0-4 | 932 | 23 | 909 | 0.0247 |
| 5-9 | 204 | 39 | 165 | 0.1912 |
| 10-19 | 324 | 127 | 197 | 0.3920 |
| 20-49 | 549 | 371 | 178 | 0.6758 |
| 50+ | 2,328 | 2,260 | 68 | 0.9708 |

박스 내부 점이 50개 이상이면 거의 모두 검출하지만 5개 미만이면 threshold를
`0.25`로 낮춰도 2.47%만 검출된다. 단순 threshold 조정보다 sparse-object
학습과 point aggregation 개선이 더 중요한 이유다.

## 6. Class 주의사항

threshold `0.25`에서 prediction은 VEHICLE 3,062개와 PEDESTRIAN 295개다.
하지만 이 segment의 평가 대상 GT에는 pedestrian가 0개라 295개가 모두 FP로
계산된다. 이 결과만 보고 pedestrian threshold를 높이는 것은 위험하다.
pedestrian와 cyclist GT가 충분한 다른 segment에서 class-wise threshold를
별도로 측정해야 한다.

## 7. Compact 출력 검증

- 일반 중간 tensor 예상량: 약 78GB
- 실제 compact 198프레임 결과: 126.2MB
- 남은 heavy stage directory: 0개
- 5프레임 `0.05 -> 0.35` 필터와 직접 `0.35` Decode: 모든 행 exact match

## 8. 결론

현재 segment에서 전역 threshold 기본 후보는 `0.30~0.35`다. recall을 더
중시하고 precision `0.80`을 허용한다면 `0.25`를 실험 후보로 둘 수 있다.
하지만 가장 큰 FN 집단은 threshold 문제가 아니라 먼 거리와 낮은 point
count다.

다음 검증은 여러 validation segment에서 같은 분석을 반복하고, class별 GT가
충분히 모인 뒤 Waymo 공식 mAP/mAPH와 class-wise threshold로 확장해야 한다.

