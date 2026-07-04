# 08 Box Decode and Rotated NMS

07 CenterHead의 raw prediction map을 실제 3D box 후보로 복원하고 중복 box를 제거한다.

## 입력과 출력

입력은 NCHW float32 파일 `reg.bin`, `height.bin`, `dim.bin`, `rot.bin`, `hm.bin`이다. 출력 `detections.csv`의 한 행은 다음과 같다.

```text
x,y,z,dx,dy,dz,yaw,score,label,source_index
```

클래스는 `0=VEHICLE`, `1=PEDESTRIAN`, `2=CYCLIST`이다.

## 핵심 수식

각 feature-map 셀 `(grid_x, grid_y)`에서:

```text
score = sigmoid(max(hm_logits))
x = (grid_x + reg_x) * 0.32 - 74.88
y = (grid_y + reg_y) * 0.32 - 74.88
z = height
(dx, dy, dz) = exp(dim)
yaw = atan2(rot_sin, rot_cos)
```

CUDA가 모든 셀을 병렬 decode하고 score와 공간 범위를 통과한 후보만 만든다. C++은 점수순 최대 4096개 후보에 대해 BEV 회전 사각형 IoU를 계산하고, IoU가 `0.7`보다 큰 중복 후보를 억제한 뒤 최대 500개를 남긴다. 이는 원본 Waymo PointPillars 설정과 같다.

## 빌드와 실행

```powershell
cmake -S . -B build_cuda -T "cuda=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1"
cmake --build build_cuda --config Release
.\build_cuda\Release\centerpoint_decode.exe <07-output-dir> <output-dir>
python .\tools\validate_reference.py --head-dir <07-output-dir> --decode-dir <output-dir>
```

현재는 box decode까지 완성된 상태다. 실제 Waymo frame에서 결과를 시각화하고 정답 annotation과 평가하려면 Waymo 데이터 변환/좌표계 연결이 다음 작업이다.
