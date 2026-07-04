# 07 CenterHead CUDA

RPN의 `[1,384,468,468]` BEV feature를 CenterPoint의 raw box prediction map으로 바꾸는 단계다.

## 계산 순서

1. 공유 `3x3 Conv(384->64) + BN + ReLU`
2. 다섯 branch에서 `3x3 Conv(64->64) + BN + ReLU`
3. 각 branch 마지막 `3x3 Conv`: `reg=2`, `height=1`, `dim=3`, `rot=2`, `hm=3`

모든 tensor는 NCHW다. 현재 checkpoint는 `novelocity` 모델이므로 velocity branch가 없다. 출력은 아직 3D box가 아니라 decode 전 raw map이다.

## 실행

```powershell
python tools/extract_head_weights.py --checkpoint <pth> --output-dir weights
cmake -S . -B build_cuda -T "cuda=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1"
cmake --build build_cuda --config Release
.\build_cuda\Release\centerpoint_head_cuda.exe <rpn-output> weights output
python tools/validate_selected.py --rpn-dir <rpn-output> --weight-dir weights --output-dir output
```

구체적인 코드 흐름은 `IMPLEMENTATION_GUIDE.md`, 검증 결과는 `VALIDATION.md`를 본다. 다음 단계인 raw map decode와 rotated NMS는 `08_decode_project`에 구현되어 있다.
