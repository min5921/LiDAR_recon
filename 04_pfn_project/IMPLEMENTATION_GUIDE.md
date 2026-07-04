# 04 Dummy PFN 구현 코드 가이드

## 입출력

```text
decorated [10404,20,9]
  -> pillar_features [10404,64]
```

핵심 파일은 `src/pfn.cpp`다. 현재는 실제 checkpoint가 아니라 deterministic dummy weight를 사용한다.

## main 호출 흐름

```cpp
DecoratedPillarDump dump = read_decorated_pillar_dump(input_dir);
PfnWeights weights = make_dummy_pfn_weights(in_channels, out_channels);
PillarFeatureResult result = run_pfn_cpu(dump, config, weights);
write_pillar_features(output_dir, result);
```

## Dummy weight 생성

```cpp
const int pattern = ((out + 1) * (in + 3)) % 17;
linear_weight[out * in_channels + in] =
    (static_cast<float>(pattern) - 8.0F) * 0.01F;
```

Python과 C++이 같은 weight를 별도 파일 없이 만들기 위한 규칙이다.

Dummy BatchNorm parameter:

```text
gamma=1, beta=0, running_mean=0, running_var=1
```

## 반복문 구조

```cpp
for (int pillar = 0; pillar < num_pillars; ++pillar) {
    for (int out = 0; out < out_channels; ++out) {
        for (int point = 0; point < max_points; ++point) {
            for (int in = 0; in < in_channels; ++in) {
                // Linear 누적
            }
        }
    }
}
```

한 pillar의 한 output channel을 계산한 뒤 point 방향 최댓값을 선택한다.

## Padding 판별

```cpp
bool is_padding = true;
for (int in = 0; in < in_channels; ++in) {
    if (decorated[input_offset + in] != 0.0F) {
        is_padding = false;
        break;
    }
}
```

모든 decorated 값이 0인 row는 PFN 계산에서 제외한다.

## Linear

```cpp
linear += input[input_offset + in] *
          weight[out * in_channels + in];
```

PyTorch `Linear` weight shape는 `[out_channels,in_channels]`이다.

## BatchNorm, ReLU, Max Pool

```cpp
normalized = (linear - mean[out]) / sqrt(var[out] + eps);
affine = normalized * gamma[out] + beta[out];
activated = std::max(affine, 0.0F);
pooled = std::max(pooled, activated);
```

최종 저장 offset:

```cpp
pillar_features[pillar * out_channels + out] = pooled;
```

## 검증

```text
Python/C++ allclose: True
max abs diff: 0.00000191
```

## 중요한 한계

현재 코드는 한 층 `9 -> 64` PFN이다. 목표 Waymo 모델은 10차원 입력과 두 PFN layer를 사용한다.

```text
layer 0: 10 -> 32, max concat 후 64
layer 1: 64 -> 64, max 후 pillar feature
```
