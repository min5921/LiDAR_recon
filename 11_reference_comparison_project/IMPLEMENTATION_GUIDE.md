# 원본 CenterPoint 단계별 비교 구현

## 1. 비교 기준

이 PC에는 PyTorch와 원본 CenterPoint의 CUDA extension이 설치되어 있지 않다. 따라서 원본 framework 전체를 직접 실행한 비교는 아니다.

대신 다음 세 가지를 기준으로 사용한다.

1. 원본 repository의 Waymo loader와 PointPillars source code
2. `centerpoint_waymo_pointpillars_full_novelocity.pth`에서 직접 추출한 weight
3. NumPy로 독립 구현한 동일 수식

이 방식은 framework 설치 없이 C++/CUDA 계산값을 수치 검증할 수 있지만, 원본 PyTorch end-to-end 출력 dump가 생기면 마지막으로 full tensor 교차 검증을 추가해야 한다.

## 2. 전처리 비교

원본 Waymo loader는 `points_feature[:, 0] = tanh(points_feature[:, 0])`을 적용한다. 현재 derived archive의 raw intensity 최대값은 29.5이므로 이 변환을 누락하면 PFN 입력 분포가 크게 달라진다.

비교기는 다음을 확인한다.

- XYZ와 elongation은 raw/tanh 실행에서 동일
- reference 실행의 intensity가 raw intensity의 tanh와 동일
- intensity는 voxel 좌표를 바꾸지 않으므로 coordinates와 num_points가 동일

이 다섯 조건은 단순 참고값이 아니다. 하나라도 `false`이면 `preprocessing_all_frames_passed`가 실패하고 비교 프로그램도 0이 아닌 종료 코드를 반환한다.

## 3. 실행 계약 검증

`09` runner의 `run_contract`를 사용해 raw와 tanh 실행에서 intensity 변환 외의 조건이 모두 동일한지 먼저 확인한다.

- archive 경로, 크기, 수정 시각
- frame 목록과 lidar/return/NLZ 선택
- score threshold, NMS IoU/규칙, match IoU
- 실행 파일과 Python script의 SHA-256
- PFN/RPN/Head weight 전체의 SHA-256

RPN probe JSON은 reference eval의 첫 프레임 내부에 있어야 한다. Head audit와 독립 reference 결과도 현재 raw/reference eval 폴더와 현재 weight 폴더를 가리켜야 한다. 이 계약을 통과하기 전에는 지표 차이를 `tanh` 효과로 해석하지 않는다.

## 4. PFN 비교

`decorated_pillars.bin`과 checkpoint PFN weight를 사용해 NumPy에서 두 PFN layer를 전체 재계산한다.

```text
Linear -> BN -> ReLU -> max
-> local/max concatenate
-> Linear -> BN -> ReLU -> max
```

모든 pillar와 64개 출력 channel을 C++ 결과와 비교한다.

## 5. Scatter 비교

PFN feature와 `[batch,z,y,x]` coordinate를 사용해 NumPy BEV tensor를 만들고 C++의 `[1,64,468,468]` 전체 tensor와 exact 비교한다.

## 6. RPN 비교

기존 재귀 검사기는 이전 레이어의 중간 tensor가 없어 120초를 초과했다. 새 `--probes` 모드는 19개 RPN 레이어마다 두 위치를 저장한다.

```text
normal Conv: input patch [Cin,K,K] + output scalar
deconvolution: input vector [Cin] + output scalar
```

NumPy validator는 checkpoint weight와 BN parameter를 읽어 각 probe의 Conv/Deconv, BN, ReLU를 독립 재계산한다.

## 7. CenterHead 비교

10번 프로젝트의 validator를 사용해 37개 Waymo GT 주변 heatmap peak를 NumPy로 다시 계산하고 CUDA `hm.bin`과 비교한다.

## 8. 결과 해석

PFN부터 Head까지 수치 검증이 모두 통과하면서 tanh 적용만으로 detection 지표가 크게 개선되면, 낮은 recall의 주원인은 CUDA 레이어 구현보다 입력 전처리 불일치였다고 판단할 수 있다.
