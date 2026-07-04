import argparse
import functools
import json
import sys
from pathlib import Path

import numpy as np


def read_bn(weight_dir, prefix, channels):
    return {
        "weight": np.fromfile(weight_dir / f"{prefix}_bn_weight.bin", np.float32).reshape(channels),
        "bias": np.fromfile(weight_dir / f"{prefix}_bn_bias.bin", np.float32).reshape(channels),
        "mean": np.fromfile(weight_dir / f"{prefix}_bn_mean.bin", np.float32).reshape(channels),
        "var": np.fromfile(weight_dir / f"{prefix}_bn_var.bin", np.float32).reshape(channels),
    }


def bn_relu(value, bn, channel, eps=np.float32(1.0e-3)):
    value = np.float32(
        (value - bn["mean"][channel])
        / np.sqrt(np.float32(bn["var"][channel] + eps))
    )
    value = np.float32(value * bn["weight"][channel] + bn["bias"][channel])
    return np.maximum(value, np.float32(0.0))


def make_conv(weight_dir, prefix, in_channels, out_channels, stride, input_shape):
    weight = np.fromfile(weight_dir / f"{prefix}_weight.bin", np.float32).reshape(
        out_channels, in_channels, 3, 3
    )
    output_shape = (
        out_channels,
        (input_shape[1] + 2 - 3) // stride + 1,
        (input_shape[2] + 2 - 3) // stride + 1,
    )
    return {
        "prefix": prefix,
        "weight": weight,
        "bn": read_bn(weight_dir, prefix, out_channels),
        "stride": stride,
        "input_shape": input_shape,
        "output_shape": output_shape,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bev-dump", required=True, type=Path)
    parser.add_argument("--weight-dir", required=True, type=Path)
    parser.add_argument("--rpn-dump", required=True, type=Path)
    args = parser.parse_args()

    bev_meta = json.loads(
        (args.bev_dump / "bev_features_metadata.json").read_text(encoding="utf-8")
    )
    bev = np.memmap(
        args.bev_dump / "bev_features.bin",
        dtype=np.float32,
        mode="r",
        shape=tuple(bev_meta["shape"]),
    )[0]
    rpn_meta = json.loads(
        (args.rpn_dump / "rpn_features_metadata.json").read_text(encoding="utf-8")
    )
    rpn = np.memmap(
        args.rpn_dump / "rpn_features.bin",
        dtype=np.float32,
        mode="r",
        shape=tuple(rpn_meta["shape"]),
    )[0]

    layers = []
    shape = (64, 468, 468)
    for index in range(4):
        layer = make_conv(args.weight_dir, f"block0_conv{index}", 64, 64, 1, shape)
        layers.append(layer)
        shape = layer["output_shape"]
    block0_last = len(layers) - 1

    layer = make_conv(args.weight_dir, "block1_conv0", 64, 128, 2, shape)
    layers.append(layer)
    shape = layer["output_shape"]
    for index in range(1, 6):
        layer = make_conv(args.weight_dir, f"block1_conv{index}", 128, 128, 1, shape)
        layers.append(layer)
        shape = layer["output_shape"]
    block1_last = len(layers) - 1

    layer = make_conv(args.weight_dir, "block2_conv0", 128, 256, 2, shape)
    layers.append(layer)
    shape = layer["output_shape"]
    for index in range(1, 6):
        layer = make_conv(args.weight_dir, f"block2_conv{index}", 256, 256, 1, shape)
        layers.append(layer)
        shape = layer["output_shape"]
    block2_last = len(layers) - 1

    @functools.lru_cache(maxsize=None)
    def evaluate(layer_index, channel, y, x):
        layer = layers[layer_index]
        weight = layer["weight"]
        stride = layer["stride"]
        input_channels, input_height, input_width = layer["input_shape"]
        value = np.float32(0.0)
        for in_channel in range(input_channels):
            for kernel_y in range(3):
                input_y = y * stride + kernel_y - 1
                if input_y < 0 or input_y >= input_height:
                    continue
                for kernel_x in range(3):
                    input_x = x * stride + kernel_x - 1
                    if input_x < 0 or input_x >= input_width:
                        continue
                    if layer_index == 0:
                        source = bev[in_channel, input_y, input_x]
                    else:
                        source = evaluate(
                            layer_index - 1, in_channel, input_y, input_x
                        )
                    value = np.float32(
                        value + source * weight[channel, in_channel, kernel_y, kernel_x]
                    )
        return bn_relu(value, layer["bn"], channel)

    deblock0_weight = np.fromfile(
        args.weight_dir / "deblock0_weight.bin", np.float32
    ).reshape(128, 64)
    deblock0_bn = read_bn(args.weight_dir, "deblock0", 128)

    def evaluate_deblock0(channel, y, x):
        value = np.float32(0.0)
        for in_channel in range(64):
            value = np.float32(
                value
                + evaluate(block0_last, in_channel, y, x)
                * deblock0_weight[channel, in_channel]
            )
        return bn_relu(value, deblock0_bn, channel)

    deblock1_weight = np.fromfile(
        args.weight_dir / "deblock1_weight_gemm.bin", np.float32
    ).reshape(128 * 2 * 2, 128)
    deblock1_bn = read_bn(args.weight_dir, "deblock1", 128)

    def evaluate_deblock1(channel, y, x):
        input_y, kernel_y = divmod(y, 2)
        input_x, kernel_x = divmod(x, 2)
        row = (channel * 2 + kernel_y) * 2 + kernel_x
        value = np.float32(0.0)
        for in_channel in range(128):
            value = np.float32(
                value
                + evaluate(block1_last, in_channel, input_y, input_x)
                * deblock1_weight[row, in_channel]
            )
        return bn_relu(value, deblock1_bn, channel)

    deblock2_weight = np.fromfile(
        args.weight_dir / "deblock2_weight_gemm.bin", np.float32
    ).reshape(128 * 4 * 4, 256)
    deblock2_bn = read_bn(args.weight_dir, "deblock2", 128)

    def evaluate_deblock2(channel, y, x):
        input_y, kernel_y = divmod(y, 4)
        input_x, kernel_x = divmod(x, 4)
        row = (channel * 4 + kernel_y) * 4 + kernel_x
        value = np.float32(0.0)
        for in_channel in range(256):
            value = np.float32(
                value
                + evaluate(block2_last, in_channel, input_y, input_x)
                * deblock2_weight[row, in_channel]
            )
        return bn_relu(value, deblock2_bn, channel)

    checks = [
        ("deblock0", 0, 0, 0, evaluate_deblock0),
        ("deblock0", 7, 10, 11, evaluate_deblock0),
        ("deblock1", 0, 0, 0, evaluate_deblock1),
        ("deblock1", 3, 1, 1, evaluate_deblock1),
        ("deblock1", 9, 10, 11, evaluate_deblock1),
        ("deblock2", 1, 0, 1, evaluate_deblock2),
    ]
    passed = True
    for branch, channel, y, x, evaluator in checks:
        if branch == "deblock0":
            output_channel = channel
        elif branch == "deblock1":
            output_channel = 128 + channel
        else:
            output_channel = 256 + channel
        reference = float(evaluator(channel, y, x))
        actual = float(rpn[output_channel, y, x])
        difference = abs(reference - actual)
        close = np.isclose(reference, actual, rtol=2.0e-4, atol=2.0e-4)
        passed = passed and bool(close)
        print(
            f"{branch} c={channel} y={y} x={x}: "
            f"CPU={reference:.8f} CUDA={actual:.8f} "
            f"diff={difference:.8f} close={bool(close)}"
        )
    print(f"selected checks passed: {passed}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
