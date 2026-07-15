#!/usr/bin/env python3
"""Validate CUDA RPN layer probes with independent NumPy calculations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-json", required=True, type=Path)
    parser.add_argument("--weight-dir", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--absolute-tolerance", type=float, default=2.0e-4)
    parser.add_argument("--relative-tolerance", type=float, default=2.0e-4)
    return parser.parse_args()


def load(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    values = np.fromfile(path, dtype="<f4")
    expected = int(np.prod(shape))
    if values.size != expected:
        raise ValueError(f"{path} has {values.size} floats, expected {expected}")
    return values.reshape(shape)


def find_pipeline_manifest(probe_json: Path) -> tuple[Path, dict[str, object]]:
    for directory in (probe_json.parent, *probe_json.parents):
        candidate = directory / "pipeline_cache_manifest.json"
        if candidate.is_file():
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                break
            return candidate, payload
    raise FileNotFoundError(
        f"no pipeline_cache_manifest.json found above {probe_json}"
    )


def load_bn(weight_dir: Path, name: str, channels: int) -> dict[str, np.ndarray]:
    return {
        key: load(weight_dir / f"{name}_bn_{file_key}.bin", (channels,))
        for key, file_key in (
            ("weight", "weight"),
            ("bias", "bias"),
            ("mean", "mean"),
            ("variance", "var"),
        )
    }


def raw_convolution(probe: dict[str, object], weight_dir: Path) -> np.float32:
    name = str(probe["name"])
    operation = str(probe["operation"])
    input_channels = int(probe["input_shape"][0])
    output_channels = int(probe["output_shape"][0])
    kernel_size = int(probe["kernel_size"])
    output_channel, output_y, output_x = [
        int(value) for value in probe["output_index"]
    ]
    values = np.asarray(probe["input_values"], dtype=np.float32)

    if operation == "conv2d":
        weights = load(
            weight_dir / f"{name}_weight.bin",
            (output_channels, input_channels, kernel_size, kernel_size),
        )
        patch = values.reshape(input_channels, kernel_size, kernel_size)
        return np.sum(weights[output_channel] * patch, dtype=np.float32)

    if operation == "conv_transpose2d":
        weights = load(
            weight_dir / f"{name}_weight_gemm.bin",
            (output_channels * kernel_size * kernel_size, input_channels),
        )
        kernel_y = output_y % kernel_size
        kernel_x = output_x % kernel_size
        row = (output_channel * kernel_size + kernel_y) * kernel_size + kernel_x
        return np.sum(weights[row] * values, dtype=np.float32)

    raise ValueError(f"unsupported probe operation: {operation}")


def expected_output(
    probe: dict[str, object], weight_dir: Path, epsilon: np.float32
) -> tuple[float, float]:
    name = str(probe["name"])
    output_channels = int(probe["output_shape"][0])
    output_channel = int(probe["output_index"][0])
    raw = raw_convolution(probe, weight_dir)
    bn = load_bn(weight_dir, name, output_channels)
    normalized = np.float32(
        (raw - bn["mean"][output_channel])
        / np.sqrt(np.float32(bn["variance"][output_channel] + epsilon))
    )
    value = np.float32(
        normalized * bn["weight"][output_channel] + bn["bias"][output_channel]
    )
    return float(raw), float(np.maximum(value, np.float32(0.0)))


def main() -> int:
    args = parse_args()
    payload = json.loads(args.probe_json.read_text(encoding="utf-8"))
    manifest_path, pipeline_manifest = find_pipeline_manifest(args.probe_json)
    epsilon = np.float32(payload["batch_norm_eps"])
    results: list[dict[str, object]] = []

    for index, probe in enumerate(payload["probes"]):
        raw, expected = expected_output(probe, args.weight_dir, epsilon)
        actual = float(probe["output_value"])
        difference = abs(expected - actual)
        close = bool(
            np.isclose(
                expected,
                actual,
                rtol=args.relative_tolerance,
                atol=args.absolute_tolerance,
            )
        )
        results.append(
            {
                "index": index,
                "name": probe["name"],
                "operation": probe["operation"],
                "output_index": probe["output_index"],
                "raw_convolution": raw,
                "expected": expected,
                "actual": actual,
                "abs_diff": difference,
                "passed": close,
            }
        )

    maximum = max((float(row["abs_diff"]) for row in results), default=0.0)
    failed = [row for row in results if not bool(row["passed"])]
    report = {
        "probe_json": str(args.probe_json.resolve()),
        "weight_dir": str(args.weight_dir.resolve()),
        "pipeline_manifest_path": str(manifest_path.resolve()),
        "pipeline_cache_manifest": pipeline_manifest,
        "probes": len(results),
        "passed_probes": len(results) - len(failed),
        "failed_probes": len(failed),
        "max_abs_diff": maximum,
        "absolute_tolerance": args.absolute_tolerance,
        "relative_tolerance": args.relative_tolerance,
        "passed": not failed,
        "results": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "results"},
            indent=2,
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
