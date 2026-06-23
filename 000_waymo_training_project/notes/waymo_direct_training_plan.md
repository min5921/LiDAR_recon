# Waymo 직접 학습 계획

## 목표

Waymo 원본 데이터를 CenterPoint 형식으로 변환하고, `waymo_centerpoint_pp_two_pfn_stride1_3x.py` 기반 PointPillars 모델을 직접 학습한다.

첫 목표는 최고 성능이 아니라 C++/CUDA 포팅 검증에 사용할 checkpoint를 확보하는 것이다.

## 추천 순서

1. 다운로드 완료 확인
2. `.tar` 무결성 확인
3. `.tar`를 풀어서 `.tfrecord` 확보
4. `WAYMO_DATASET_ROOT/tfrecord_training`, `tfrecord_validation`, `tfrecord_testing`로 정리
5. `waymo_converter.py`로 frame pkl 생성
6. `tools/create_data.py waymo_data_prep`로 info pkl 생성
7. PointPillars config로 소규모 학습 시작
8. checkpoint에서 PFN/RPN/CenterHead weight 추출
9. C++ 구현과 PyTorch forward 결과 비교

## 처음에는 전체 학습보다 subset 권장

전체 Waymo 학습은 시간이 길다. 처음에는 training tar 1개 또는 일부 tfrecord만 변환해서 pipeline을 검증한다.

예상 흐름:

```text
training_0000.tar
  -> tfrecord_training/*.tfrecord
  -> train/lidar/*.pkl, train/annos/*.pkl
  -> infos_train_01sweeps_filter_zero_gt.pkl
  -> 짧은 epoch 학습
```

## 환경 메모

CenterPoint 원본은 오래된 dependency를 사용한다.

- TensorFlow 1.15 계열 Waymo package
- old spconv
- PyTorch 버전 호환 이슈 가능

PointPillars 경로는 spconv 의존이 낮기 때문에 VoxelNet보다 시작하기 좋다.

