import argparse
import json
import sys
from pathlib import Path

import numpy as np


def conv2d_nchw(input_values, weights, stride, padding):
    batch, in_channels, input_height, input_width = input_values.shape
    out_channels, weight_channels, kernel_height, kernel_width = weights.shape
    assert in_channels == weight_channels
    output_height = (input_height + 2 * padding - kernel_height) // stride + 1
    output_width = (input_width + 2 * padding - kernel_width) // stride + 1
    output = np.zeros(
        (batch, out_channels, output_height, output_width), dtype=np.float32
    )

    for n in range(batch):
        for out_channel in range(out_channels):
            for out_y in range(output_height):
                for out_x in range(output_width):
                    value = np.float32(0.0)
                    for in_channel in range(in_channels):
                        for kernel_y in range(kernel_height):
                            input_y = out_y * stride + kernel_y - padding
                            if input_y < 0 or input_y >= input_height:
                                continue
                            for kernel_x in range(kernel_width):
                                input_x = out_x * stride + kernel_x - padding
                                if input_x < 0 or input_x >= input_width:
                                    continue
                                value = np.float32(
                                    value
                                    + input_values[n, in_channel, input_y, input_x]
                                    * weights[out_channel, in_channel, kernel_y, kernel_x]
                                )
                    output[n, out_channel, out_y, out_x] = value
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", required=True, type=Path)
    args = parser.parse_args()

    metadata = json.loads((args.dump / "metadata.json").read_text(encoding="utf-8"))
    input_shape = tuple(metadata["input_shape"])
    weight_shape = tuple(metadata["weight_shape"])
    output_shape = tuple(metadata["output_shape"])

    input_values = np.fromfile(args.dump / "input.bin", np.float32).reshape(input_shape)
    weights = np.fromfile(args.dump / "conv_weight.bin", np.float32).reshape(weight_shape)
    bn_weight = np.fromfile(args.dump / "bn_weight.bin", np.float32)
    bn_bias = np.fromfile(args.dump / "bn_bias.bin", np.float32)
    bn_mean = np.fromfile(args.dump / "bn_mean.bin", np.float32)
    bn_var = np.fromfile(args.dump / "bn_var.bin", np.float32)
    cpp = np.fromfile(args.dump / "output.bin", np.float32).reshape(output_shape)

    conv = conv2d_nchw(
        input_values, weights, metadata["stride"], metadata["padding"]
    )
    scale = bn_weight / np.sqrt(bn_var + np.float32(metadata["batch_norm_eps"]))
    python = (conv - bn_mean[None, :, None, None]) * scale[None, :, None, None]
    python += bn_bias[None, :, None, None]
    python = np.maximum(python, np.float32(0.0)).astype(np.float32)

    max_abs_diff = float(np.max(np.abs(python - cpp)))
    close = np.allclose(python, cpp, rtol=1.0e-5, atol=1.0e-6)
    print(f"input shape:       {input_values.shape}")
    print(f"weight shape:      {weights.shape}")
    print(f"python output:     {python.shape}")
    print(f"CUDA output:       {cpp.shape}")
    print(f"allclose:          {close}")
    print(f"max abs diff:      {max_abs_diff:.8f}")
    print(f"CUDA first values: {cpp.reshape(-1)[:8]}")
    print(f"Python first vals: {python.reshape(-1)[:8]}")
    return 0 if close else 1


if __name__ == "__main__":
    sys.exit(main())
