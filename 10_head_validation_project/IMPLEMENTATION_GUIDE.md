# CenterHead Heatmap 검증 구현 설명

## 1. 검증하려는 질문

최종 검출 누락은 크게 두 위치에서 발생한다.

1. CenterHead가 GT 위치에 충분한 heatmap 점수를 만들지 못했다.
2. heatmap 점수는 충분했지만 Decode, box regression 또는 NMS에서 사라졌다.

최종 `detections.csv`만 보면 두 경우를 구분할 수 없다. 따라서 Decode 이전의 `hm.bin`을 GT 위치에서 직접 읽는다.

## 2. Heatmap 좌표 변환

현재 Waymo PointPillars 설정은 다음과 같다.

```text
grid shape       = [3, 468, 468]
point cloud min  = [-74.88, -74.88]
cell size        = [0.32, 0.32] meter
```

Waymo GT 중심 `(x, y)`를 heatmap cell로 변환한다.

```text
cell_x_float = (x - (-74.88)) / 0.32
cell_y_float = (y - (-74.88)) / 0.32
cell_x = floor(cell_x_float)
cell_y = floor(cell_y_float)
```

`hm.bin`은 logit이므로 점수 비교 전에 sigmoid를 적용한다.

```text
score = 1 / (1 + exp(-logit))
```

## 3. 한 GT에서 읽는 값

- GT가 위치한 cell의 raw logit과 sigmoid score
- GT cell 주변 `5 x 5` 영역의 최대 score
- 주변 최대 cell과 GT 중심의 meter 거리
- 해당 score의 클래스 전체 순위
- 상위 local peak 중 GT에 가장 가까운 peak 거리
- `match_report.json` 기준 최종 검출 성공 여부

주변 영역은 기본 반경 2 cell이며 `--local-radius`로 바꿀 수 있다.

## 4. 결과 해석

`LOW_HEATMAP_SCORE`가 대부분이면 NMS threshold를 계속 조정해도 recall은 회복되지 않는다. PFN, scatter, RPN, CenterHead weight 매핑 또는 입력 전처리를 확인해야 한다.

`HIGH_HEATMAP_EMITTED_UNMATCHED`는 해당 cell의 박스가 최종 출력까지 살아 있지만 IoU 기준을 통과하지 못한 경우다. regression된 중심, 크기, 회전을 GT와 비교해야 한다.

`HIGH_HEATMAP_NOT_EMITTED`가 많으면 heatmap은 정상이며 class 선택, Decode 유효성 검사 또는 NMS에서 객체가 사라지는 것이다.

`CLASS_CONFLICT_AT_PEAK`는 관심 클래스 점수는 높지만 같은 cell의 다른 클래스 점수가 더 높아 Decode의 argmax에서 클래스가 바뀐 경우다.

검출된 GT의 peak 위치는 맞지만 누락 GT의 score만 낮다면 좌표계 문제보다는 입력 분포나 모델 출력 자체를 우선 의심할 수 있다.

## 5. CUDA 구현값 독립 재계산

`validate_gt_heatmap_reference.py`는 audit에서 선택한 GT 주변 peak cell에 대해서만 다음 연산을 NumPy로 다시 수행한다.

```text
RPN feature
  -> shared 3x3 Conv + BN + ReLU
  -> hm hidden 3x3 Conv + BN + ReLU
  -> hm output 3x3 Conv + bias
```

전체 `[3,468,468]`을 다시 계산하지 않고 필요한 receptive field만 계산하며, 겹치는 shared/hidden cell은 cache로 재사용한다. CUDA 출력과의 최대 절대 오차가 `2e-4` 미만이면 CenterHead 계산 구현은 통과로 판정한다.

## 6. 시각화 읽는 법

배경은 `log10(sigmoid score)`다. 매우 작은 점수도 보이도록 `-6`부터 `0` 범위로 표시한다.

- 초록 원: 최종 검출 성공 GT
- 파란 원: GT 주변 heatmap 점수가 threshold 미만
- 빨간 원: 박스는 출력됐지만 GT IoU 매칭 실패
- 주황 원: heatmap은 threshold 이상이지만 최종 박스가 출력되지 않음
- 보라 원: 같은 cell의 다른 클래스가 선택됨
- 흰색 X: 클래스별 상위 local peak
