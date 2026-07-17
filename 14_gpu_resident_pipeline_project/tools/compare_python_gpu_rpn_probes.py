#!/usr/bin/env python3
"""Validate CUDA RPN layer probes with independent NumPy calculations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare CUDA RPN probe values with NumPy."
    )
    parser.add_argument("probes_json", type=Path)
    parser.add_argument("weights_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--atol", type=float, default=2.0e-4)
    parser.add_argument("--rtol", type=float, default=1.0e-5)
    return parser.parse_args()


def read_f32(path: Path, expected_count: int | None = None) -> np.ndarray:
    values = np.fromfile(path, dtype=np.float32)
    if expected_count is not None and values.size != expected_count:
        raise ValueError(
            f"{path.name}: expected {expected_count} floats, got {values.size}"
        )
    return values


def batch_norm_relu(
    raw_value: np.float32,
    channel: int,
    prefix: str,
    weights_dir: Path,
    epsilon: np.float32,
) -> np.float32:
    weight = read_f32(weights_dir / f"{prefix}_bn_weight.bin")[channel]
    bias = read_f32(weights_dir / f"{prefix}_bn_bias.bin")[channel]
    mean = read_f32(weights_dir / f"{prefix}_bn_mean.bin")[channel]
    variance = read_f32(weights_dir / f"{prefix}_bn_var.bin")[channel]

    inverse_std = np.float32(1.0) / np.sqrt(
        np.float32(variance + epsilon), dtype=np.float32
    )
    normalized = np.float32(np.float32(raw_value - mean) * inverse_std)
    value = np.float32(np.float32(normalized * weight) + bias)
    return np.maximum(value, np.float32(0.0))


def calculate_probe(
    probe: dict,
    layer: dict,
    weights_dir: Path,
    epsilon: np.float32,
) -> tuple[np.float32, np.float32]:
    prefix = probe["name"]
    channel, output_y, output_x = probe["output_index"]
    input_values = np.asarray(probe["input_values"], dtype=np.float32)
    runtime_shape = tuple(layer["runtime_weight_shape"])
    weight = read_f32(
        weights_dir / layer["weight_file"], int(np.prod(runtime_shape))
    ).reshape(runtime_shape)

    if probe["operation"] == "conv2d":
        selected_weight = weight[channel].reshape(-1)
    elif probe["operation"] == "conv_transpose2d":
        kernel_size = int(probe["kernel_size"])
        kernel_y = output_y % kernel_size
        kernel_x = output_x % kernel_size
        row = (channel * kernel_size + kernel_y) * kernel_size + kernel_x
        selected_weight = weight[row]
    else:
        raise ValueError(f"unsupported operation: {probe['operation']}")

    if selected_weight.size != input_values.size:
        raise ValueError(
            f"{prefix}: weight/input mismatch "
            f"({selected_weight.size} != {input_values.size})"
        )

    raw_value = np.sum(
        selected_weight * input_values, dtype=np.float32
    )
    expected = batch_norm_relu(
        raw_value, channel, prefix, weights_dir, epsilon
    )
    return raw_value, expected


def main() -> int:
    args = parse_args()
    probe_document = json.loads(args.probes_json.read_text(encoding="utf-8"))
    metadata_path = args.weights_dir / "rpn_weights_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    layers = {layer["prefix"]: layer for layer in metadata["layers"]}
    epsilon = np.float32(probe_document["batch_norm_epsilon"])

    comparisons = []
    for index, probe in enumerate(probe_document["probes"]):
        prefix = probe["name"]
        if prefix not in layers:
            raise KeyError(f"missing metadata for {prefix}")
        raw_value, expected = calculate_probe(
            probe, layers[prefix], args.weights_dir, epsilon
        )
        actual = np.float32(probe["output_value"])
        absolute_error = float(np.abs(np.float32(actual - expected)))
        tolerance = float(args.atol + args.rtol * abs(float(expected)))
        passed = absolute_error <= tolerance
        comparisons.append(
            {
                "probe_index": index,
                "name": prefix,
                "operation": probe["operation"],
                "output_index": probe["output_index"],
                "raw_numpy": float(raw_value),
                "expected_numpy": float(expected),
                "actual_cuda": float(actual),
                "absolute_error": absolute_error,
                "tolerance": tolerance,
                "passed": passed,
            }
        )

    failed = [item for item in comparisons if not item["passed"]]
    max_error_item = max(comparisons, key=lambda item: item["absolute_error"])
    result = {
        "passed": not failed,
        "probe_count": len(comparisons),
        "failed_count": len(failed),
        "max_abs_diff": max_error_item["absolute_error"],
        "max_abs_diff_probe": {
            "name": max_error_item["name"],
            "output_index": max_error_item["output_index"],
        },
        "atol": args.atol,
        "rtol": args.rtol,
        "comparisons": comparisons,
    }

    output_path = args.output or args.probes_json.with_name(
        "rpn_probe_comparison.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    status = "PASS" if result["passed"] else "FAIL"
    print(f"RPN probe comparison: {status}")
    print(f"probes: {len(comparisons)}, failed: {len(failed)}")
    print(f"max abs diff: {result['max_abs_diff']:.9g}")
    print(
        "max diff probe: "
        f"{max_error_item['name']} {max_error_item['output_index']}"
    )
    print(f"report: {output_path}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
