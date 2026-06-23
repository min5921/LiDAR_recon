# Waymo Training Project

이 폴더는 Waymo 원본 데이터를 직접 준비하고, CenterPoint 학습까지 연결하기 위한 작업 공간이다.

현재 기준 원본 repo:

```text
../00_reference/centerpoint_original/CenterPoint-master
```

CenterPoint 문서 기준 Waymo 데이터 최종 구조:

```text
WAYMO_DATASET_ROOT/
  tfrecord_training/
  tfrecord_validation/
  tfrecord_testing/
  train/
    lidar/
    annos/
  val/
    lidar/
    annos/
  test/
    lidar/
    annos/
  infos_train_01sweeps_filter_zero_gt.pkl
  infos_val_01sweeps_filter_zero_gt.pkl
  infos_test_01sweeps_filter_zero_gt.pkl
  dbinfos_train_1sweeps_withvelo.pkl
```

## 다운로드 후 1차 확인

스크린샷 기준 다운로드 중인 파일은 정상 `.tar` 형태다.

예:

```text
training_0000.tar
training_0001.tar
...
```

다운로드가 끝난 뒤 먼저 아카이브 상태를 확인한다.

```powershell
python .\scripts\scan_waymo_files.py --root "E:\Waymo_datset\Perception_Dataset_v1.4.3_with_map\archived_files"
```

## 압축 해제 목표

압축을 푼 뒤에는 `.tfrecord` 파일들이 아래처럼 모이면 된다.

```text
WAYMO_DATASET_ROOT/
  tfrecord_training/*.tfrecord
  tfrecord_validation/*.tfrecord
  tfrecord_testing/*.tfrecord
```

## CenterPoint 변환 명령

원본 CenterPoint 문서 기준:

```powershell
cd "..\00_reference\centerpoint_original\CenterPoint-master"

python det3d\datasets\waymo\waymo_converter.py --record_path "WAYMO_DATASET_ROOT\tfrecord_training\*.tfrecord" --root_path "WAYMO_DATASET_ROOT\train"
python det3d\datasets\waymo\waymo_converter.py --record_path "WAYMO_DATASET_ROOT\tfrecord_validation\*.tfrecord" --root_path "WAYMO_DATASET_ROOT\val"
python det3d\datasets\waymo\waymo_converter.py --record_path "WAYMO_DATASET_ROOT\tfrecord_testing\*.tfrecord" --root_path "WAYMO_DATASET_ROOT\test"
```

그 다음 info 파일을 만든다.

```powershell
python tools\create_data.py waymo_data_prep --root_path=data\Waymo --split train --nsweeps=1
python tools\create_data.py waymo_data_prep --root_path=data\Waymo --split val --nsweeps=1
python tools\create_data.py waymo_data_prep --root_path=data\Waymo --split test --nsweeps=1
```

## 현재 주의점

현재 로컬 Python에는 다음 패키지가 아직 없다.

```text
tensorflow
waymo_open_dataset
```

CenterPoint 원본 문서는 다음 패키지를 요구한다.

```text
waymo-open-dataset-tf-1-15-0==1.2.0
```

이 부분은 Python/CUDA/PyTorch 환경과 충돌 가능성이 있어 별도 conda 환경에서 준비하는 것이 좋다.

## 현재 로컬 점검 결과

Windows에서 `centerpoint_waymo` conda 환경을 만들었다.

```text
python=3.7
env name=centerpoint_waymo
```

하지만 원본 문서가 요구하는 Waymo devkit은 Windows pip에서 찾을 수 없었다.

```text
waymo-open-dataset-tf-1-15-0==1.2.0
No matching distribution found
```

최신 TF2 계열 Waymo 패키지도 현재 Windows pip 환경에서는 잡히지 않았다.

또한 현재 PC에는 WSL 배포판이 설치되어 있지 않다.

따라서 Waymo TFRecord 변환 단계는 다음 중 하나가 필요하다.

```text
1. WSL2 Ubuntu 설치 후 Linux 환경에서 변환
2. 별도 Ubuntu/Linux 머신에서 변환
3. Docker/WSL 기반 Linux 컨테이너에서 변환
```

데이터 경로 연결은 완료되어 있다.

```text
C:\Users\user\Desktop\Onechip\Waymo_CenterPoint\tfrecord_training
  -> C:\Users\user\Desktop\Onechip\archived_files_training_training_0000
```

CenterPoint repo 안에서도 아래 경로로 보인다.

```text
CenterPoint-master\data\Waymo\tfrecord_training
```
