# 07 CenterHead CUDA 구현 코드 가이드

## 1. 이 단계의 역할

RPN이 만든 `[1,384,468,468]` feature를 box의 구성 요소별 raw prediction map으로 바꾼다. 핵심 진입점은 `cuda/center_head_cuda.cu`의 `run_center_head_cuda()`다.

```text
RPN feature
  -> Shared Conv-BN-ReLU
  -> reg / height / dim / rot / hm branch
```

## 2. 입출력

입력 `rpn_features.bin`은 NCHW float32다. 출력도 모두 NCHW다.

| 출력 | Shape | 의미 |
|---|---|---|
| `reg.bin` | `[1,2,468,468]` | cell 내부 X/Y offset |
| `height.bin` | `[1,1,468,468]` | box 중심 Z |
| `dim.bin` | `[1,3,468,468]` | log-space DX/DY/DZ |
| `rot.bin` | `[1,2,468,468]` | 회전 sin/cos 성분 |
| `hm.bin` | `[1,3,468,468]` | VEHICLE/PEDESTRIAN/CYCLIST logit |

이 값들은 확률이나 최종 box가 아니다. sigmoid, exp, 좌표 복원은 08에서 수행한다.

## 3. 실제 checkpoint 구조

`tools/extract_head_weights.py`는 다음 tensor를 추출한다.

```text
bbox_head.shared_conv.0  Conv [64,384,3,3] + bias
bbox_head.shared_conv.1  BatchNorm [64]

각 branch:
  Conv [64,64,3,3] + bias
  BatchNorm [64]
  ReLU
  Conv [output_channels,64,3,3] + bias
```

`novelocity` 체크포인트이므로 `vel` branch는 없다. Exporter는 PyTorch runtime 없이 `.pth` ZIP 안의 `data.pkl`과 storage를 읽어 contiguous float32 binary를 만든다.

## 4. CUDA convolution 흐름

`conv()`은 모든 3x3 convolution이 공유하는 함수다.

```text
input [Cin,H,W]
  -> im2col [Cin*9,H*W]
  -> cuBLAS SGEMM
  -> output [Cout,H,W]
```

첫 두 convolution 종류는 `bias_bn_relu` kernel을 호출한다.

```cpp
value = convolution + conv_bias[channel];
value = (value - mean[channel]) / sqrt(variance[channel] + epsilon);
value = value * bn_weight[channel] + bn_bias[channel];
value = max(value, 0.0F);
```

branch 마지막 convolution은 BN/ReLU 없이 `add_bias`만 수행한다. 음수 heatmap logit도 다음 sigmoid에 필요하므로 ReLU로 지우면 안 된다.

## 5. 실행 순서

```cpp
shared = conv(input, weights.shared);

for (branch : branches) {
    hidden = conv(shared, branch.hidden);
    output = conv(hidden, branch.output);
    download(output);
}
```

다섯 branch를 순차 실행해 같은 shared tensor를 재사용한다. 현재는 학습과 검증을 위해 각 결과를 host로 내려받는다.

## 6. 메모리에서 주의할 점

가장 큰 임시 tensor는 shared convolution의 im2col이다.

```text
384 * 9 * 468 * 468 * 4 bytes
약 3.0 GB
```

`Buffer`는 RAII 방식으로 scope가 끝날 때 `cudaFree`를 호출한다. Weight와 BN tensor도 계층 실행이 끝나면 해제된다.

## 7. 검증

`tools/validate_selected.py`는 모서리 `(0,0)`, 중앙 `(234,234)`, 반대 모서리 `(467,467)`에서 각 branch 첫 채널을 NumPy로 재계산한다. 이 위치 선택으로 zero padding과 일반 내부 convolution을 함께 검사한다.

```text
검사 sample: 15개
최대 절대 오차: 4.76837158203125e-7
반복 실행: 다섯 binary SHA-256 일치
```

## 8. 현재 한계

RPN 결과를 host binary로 읽고 Head 결과도 host에 저장한다. 최종 통합에서는 RPN의 device output을 shared convolution에 직접 넘겨 약 336MB의 왕복 복사를 없애야 한다.
