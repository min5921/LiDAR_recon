#!/usr/bin/env python3
"""Validate CUDA CenterHead layer probes with NumPy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("probes_json", type=Path)
    parser.add_argument("weights_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--atol", type=float, default=2.0e-4)
    parser.add_argument("--rtol", type=float, default=1.0e-5)
    return parser.parse_args()


def read_f32(path: Path, count: int | None = None) -> np.ndarray:
    values = np.fromfile(path, dtype=np.float32)
    if count is not None and values.size != count:
        raise ValueError(f"{path.name}: expected {count}, got {values.size}")
    return values


def calculate(probe: dict, layer: dict, weights_dir: Path,
              epsilon: np.float32) -> tuple[np.float32, np.float32]:
    prefix = probe["name"]
    shape = tuple(layer["weight_shape"])
    weight = read_f32(
        weights_dir / f"{prefix}_weight.bin", int(np.prod(shape))
    ).reshape(shape)
    channel = int(probe["output_index"][0])
    inputs = np.asarray(probe["input_values"], dtype=np.float32)
    raw = np.sum(weight[channel].reshape(-1) * inputs, dtype=np.float32)
    conv_bias = read_f32(weights_dir / f"{prefix}_bias.bin")[channel]
    biased = np.float32(raw + conv_bias)
    if not probe["has_batch_norm"]:
        return raw, biased

    bn_weight = read_f32(weights_dir / f"{prefix}_bn_weight.bin")[channel]
    bn_bias = read_f32(weights_dir / f"{prefix}_bn_bias.bin")[channel]
    mean = read_f32(weights_dir / f"{prefix}_bn_mean.bin")[channel]
    variance = read_f32(weights_dir / f"{prefix}_bn_var.bin")[channel]
    inverse_std = np.float32(1.0) / np.sqrt(
        np.float32(variance + epsilon), dtype=np.float32
    )
    normalized = np.float32(np.float32(biased - mean) * inverse_std)
    output = np.float32(np.float32(normalized * bn_weight) + bn_bias)
    return raw, np.maximum(output, np.float32(0.0))


def main() -> int:
    args = parse_args()
    document = json.loads(args.probes_json.read_text(encoding="utf-8"))
    metadata = json.loads(
        (args.weights_dir / "head_weights_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    layers = {item["prefix"]: item for item in metadata["layers"]}
    epsilon = np.float32(document["batch_norm_epsilon"])
    comparisons = []
    for index, probe in enumerate(document["probes"]):
        raw, expected = calculate(
            probe, layers[probe["name"]], args.weights_dir, epsilon
        )
        actual = np.float32(probe["output_value"])
        difference = float(np.abs(np.float32(actual - expected)))
        tolerance = float(args.atol + args.rtol * abs(float(expected)))
        comparisons.append(
            {
                "probe_index": index,
                "name": probe["name"],
                "output_index": probe["output_index"],
                "raw_numpy": float(raw),
                "expected_numpy": float(expected),
                "actual_cuda": float(actual),
                "absolute_error": difference,
                "tolerance": tolerance,
                "passed": difference <= tolerance,
            }
        )

    failures = [item for item in comparisons if not item["passed"]]
    maximum = max(comparisons, key=lambda item: item["absolute_error"])
    result = {
        "passed": not failures,
        "probe_count": len(comparisons),
        "failed_count": len(failures),
        "max_abs_diff": maximum["absolute_error"],
        "max_abs_diff_probe": maximum["name"],
        "atol": args.atol,
        "rtol": args.rtol,
        "comparisons": comparisons,
    }
    output = args.output or args.probes_json.with_name(
        "head_probe_comparison.json"
    )
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"CenterHead probe comparison: {'PASS' if result['passed'] else 'FAIL'}")
    print(f"probes: {len(comparisons)}, failed: {len(failures)}")
    print(f"max abs diff: {result['max_abs_diff']:.9g}")
    print(f"report: {output}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
