#!/usr/bin/env python3
"""Recompute CenterHead heatmap logits at audited GT peak cells with NumPy."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


HEIGHT = 468
WIDTH = 468
RPN_CHANNELS = 384
HIDDEN_CHANNELS = 64
HEATMAP_CHANNELS = 3
BN_EPSILON = np.float32(1.0e-3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate CUDA CenterHead heatmap values at audited GT peak cells."
    )
    parser.add_argument("--eval-dir", required=True, type=Path)
    parser.add_argument("--weight-dir", required=True, type=Path)
    parser.add_argument("--audit-csv", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--tolerance", type=float, default=2.0e-4)
    return parser.parse_args()


def load(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    values = np.fromfile(path, dtype="<f4")
    expected = int(np.prod(shape))
    if values.size != expected:
        raise ValueError(f"{path} has {values.size} floats, expected {expected}")
    return values.reshape(shape)


def conv_vector(
    source: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    y: int,
    x: int,
) -> np.ndarray:
    patch = np.zeros((source.shape[0], 3, 3), dtype=np.float32)
    y0 = max(0, y - 1)
    y1 = min(source.shape[1], y + 2)
    x0 = max(0, x - 1)
    x1 = min(source.shape[2], x + 2)
    patch[:, y0 - y + 1 : y1 - y + 1, x0 - x + 1 : x1 - x + 1] = source[
        :, y0:y1, x0:x1
    ]
    value = np.tensordot(
        weight.astype(np.float64),
        patch.astype(np.float64),
        axes=((1, 2, 3), (0, 1, 2)),
    ) + bias
    return np.asarray(value, dtype=np.float32)


def apply_bn_relu(value: np.ndarray, bn: list[np.ndarray]) -> np.ndarray:
    normalized = (value - bn[2]) / np.sqrt(bn[3] + BN_EPSILON)
    return np.maximum(normalized * bn[0] + bn[1], 0.0).astype(np.float32)


def read_audit_cells(path: Path) -> dict[str, list[dict[str, str]]]:
    frames: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["in_range"].lower() != "true":
                continue
            frames[row["frame"]].append(row)
    return dict(frames)


def load_weights(weight_dir: Path) -> dict[str, object]:
    shared_weight = load(weight_dir / "shared_weight.bin", (64, 384, 3, 3))
    shared_bias = load(weight_dir / "shared_bias.bin", (64,))
    shared_bn = [
        load(weight_dir / f"shared_bn_{name}.bin", (64,))
        for name in ("weight", "bias", "mean", "var")
    ]
    hidden_weight = load(weight_dir / "hm_hidden_weight.bin", (64, 64, 3, 3))
    hidden_bias = load(weight_dir / "hm_hidden_bias.bin", (64,))
    hidden_bn = [
        load(weight_dir / f"hm_hidden_bn_{name}.bin", (64,))
        for name in ("weight", "bias", "mean", "var")
    ]
    output_weight = load(weight_dir / "hm_output_weight.bin", (3, 64, 3, 3))
    output_bias = load(weight_dir / "hm_output_bias.bin", (3,))
    return {
        "shared_weight": shared_weight,
        "shared_bias": shared_bias,
        "shared_bn": shared_bn,
        "hidden_weight": hidden_weight,
        "hidden_bias": hidden_bias,
        "hidden_bn": hidden_bn,
        "output_weight": output_weight,
        "output_bias": output_bias,
    }


def validate_frame(
    eval_dir: Path,
    frame: str,
    audit_rows: list[dict[str, str]],
    weights: dict[str, object],
) -> list[dict[str, object]]:
    frame_dir = eval_dir / frame
    rpn = load(frame_dir / "06_rpn" / "rpn_features.bin", (384, 468, 468))
    actual = load(frame_dir / "07_head" / "hm.bin", (3, 468, 468))
    shared_cache: dict[tuple[int, int], np.ndarray] = {}
    hidden_cache: dict[tuple[int, int], np.ndarray] = {}

    def shared_at(y: int, x: int) -> np.ndarray:
        if not (0 <= y < HEIGHT and 0 <= x < WIDTH):
            return np.zeros(HIDDEN_CHANNELS, dtype=np.float32)
        key = (y, x)
        if key not in shared_cache:
            value = conv_vector(
                rpn,
                weights["shared_weight"],
                weights["shared_bias"],
                y,
                x,
            )
            shared_cache[key] = apply_bn_relu(value, weights["shared_bn"])
        return shared_cache[key]

    def hidden_at(y: int, x: int) -> np.ndarray:
        if not (0 <= y < HEIGHT and 0 <= x < WIDTH):
            return np.zeros(HIDDEN_CHANNELS, dtype=np.float32)
        key = (y, x)
        if key not in hidden_cache:
            patch = np.zeros((HIDDEN_CHANNELS, 3, 3), dtype=np.float32)
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    patch[:, dy + 1, dx + 1] = shared_at(y + dy, x + dx)
            value = conv_vector(
                patch,
                weights["hidden_weight"],
                weights["hidden_bias"],
                1,
                1,
            )
            hidden_cache[key] = apply_bn_relu(value, weights["hidden_bn"])
        return hidden_cache[key]

    results: list[dict[str, object]] = []
    computed: dict[tuple[int, int], np.ndarray] = {}
    for row in audit_rows:
        cell_x = int(row["local_max_cell_x"])
        cell_y = int(row["local_max_cell_y"])
        key = (cell_y, cell_x)
        if key not in computed:
            hidden_patch = np.zeros((HIDDEN_CHANNELS, 3, 3), dtype=np.float32)
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    hidden_patch[:, dy + 1, dx + 1] = hidden_at(
                        cell_y + dy, cell_x + dx
                    )
            computed[key] = conv_vector(
                hidden_patch,
                weights["output_weight"],
                weights["output_bias"],
                1,
                1,
            )

        expected = computed[key]
        class_index = int(row["class_index"])
        got = float(actual[class_index, cell_y, cell_x])
        want = float(expected[class_index])
        results.append(
            {
                "frame": frame,
                "class_index": class_index,
                "class_name": row["class_name"],
                "gt_x": float(row["gt_x"]),
                "gt_y": float(row["gt_y"]),
                "cell_x": cell_x,
                "cell_y": cell_y,
                "expected_logit": want,
                "actual_logit": got,
                "abs_diff": abs(want - got),
            }
        )
    return results


def main() -> int:
    args = parse_args()
    aggregate = json.loads(
        (args.eval_dir / "aggregate_report.json").read_text(encoding="utf-8")
    )
    run_contract = aggregate.get("run_contract")
    if not isinstance(run_contract, dict):
        raise ValueError("aggregate report has no run_contract; rerun the evaluator")
    audit_by_frame = read_audit_cells(args.audit_csv)
    weights = load_weights(args.weight_dir)
    results: list[dict[str, object]] = []
    for frame, rows in sorted(audit_by_frame.items()):
        results.extend(validate_frame(args.eval_dir, frame, rows, weights))
    maximum = max((float(row["abs_diff"]) for row in results), default=0.0)
    report = {
        "eval_dir": str(args.eval_dir.resolve()),
        "weight_dir": str(args.weight_dir.resolve()),
        "audit_csv": str(args.audit_csv.resolve()),
        "frame_names": sorted(audit_by_frame),
        "run_contract": run_contract,
        "samples": len(results),
        "frames": len(audit_by_frame),
        "tolerance": args.tolerance,
        "max_abs_diff": maximum,
        "passed": maximum < args.tolerance,
        "results": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
