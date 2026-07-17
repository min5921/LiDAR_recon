#!/usr/bin/env python3
"""Independently validate GPU decode, ordering, and rotated NMS."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


FIELDS = ("x", "y", "z", "dx", "dy", "dz", "yaw", "score")
H = W = 468
SPATIAL = H * W


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre-nms", required=True, type=Path)
    parser.add_argument("--detections", required=True, type=Path)
    parser.add_argument("--reference-head-dir", required=True, type=Path)
    parser.add_argument("--reference-detections", type=Path)
    parser.add_argument("--score-threshold", type=float, default=0.35)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument(
        "--nms-convention", choices=("pcdet", "current"), default="pcdet"
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    result = []
    for row in rows:
        parsed = {name: float(row[name]) for name in FIELDS}
        parsed["label"] = int(row["label"])
        parsed["source_index"] = int(row["source_index"])
        result.append(parsed)
    return result


def load_map(directory: Path, name: str, channels: int) -> np.ndarray:
    values = np.fromfile(directory / f"{name}.bin", dtype=np.float32)
    expected = channels * SPATIAL
    if values.size != expected:
        raise ValueError(f"{name}.bin: expected {expected}, got {values.size}")
    return values.reshape(channels, SPATIAL)


def numpy_decode(head_dir: Path, threshold: float) -> list[dict]:
    reg = load_map(head_dir, "reg", 2)
    height = load_map(head_dir, "height", 1)
    dim = load_map(head_dir, "dim", 3)
    rot = load_map(head_dir, "rot", 2)
    heatmap = load_map(head_dir, "hm", 3)
    labels = np.argmax(heatmap, axis=0)
    source = np.arange(SPATIAL, dtype=np.int32)
    logits = heatmap[labels, source]
    scores = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
    selected = source[scores > threshold]
    result = []
    for index in selected.tolist():
        label = int(labels[index])
        y, x = divmod(index, W)
        values = {
            "x": (x + float(reg[0, index])) * 0.32 - 74.88,
            "y": (y + float(reg[1, index])) * 0.32 - 74.88,
            "z": float(height[0, index]),
            "dx": float(np.exp(np.float64(dim[0, index]))),
            "dy": float(np.exp(np.float64(dim[1, index]))),
            "dz": float(np.exp(np.float64(dim[2, index]))),
            "yaw": float(np.arctan2(rot[0, index], rot[1, index])),
            "score": float(scores[index]),
            "label": label,
            "source_index": index,
        }
        finite = all(math.isfinite(values[name]) for name in FIELDS)
        in_range = (
            -80.0 <= values["x"] <= 80.0
            and -80.0 <= values["y"] <= 80.0
            and -10.0 <= values["z"] <= 10.0
        )
        if finite and in_range:
            result.append(values)
    result.sort(key=lambda item: (-item["score"], item["source_index"]))
    return result[:4096]


def corners(box: dict, convention: str) -> list[tuple[float, float]]:
    yaw = box["yaw"]
    half_x = box["dx"] * 0.5
    half_y = box["dy"] * 0.5
    if convention == "pcdet":
        yaw = -yaw - math.pi * 0.5
        half_x = box["dy"] * 0.5
        half_y = box["dx"] * 0.5
    cosine, sine = math.cos(yaw), math.sin(yaw)
    return [
        (
            box["x"] + x * cosine - y * sine,
            box["y"] + x * sine + y * cosine,
        )
        for x, y in (
            (-half_x, -half_y),
            (half_x, -half_y),
            (half_x, half_y),
            (-half_x, half_y),
        )
    ]


def cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def intersection(a, b, p, q):
    first, second = cross(p, q, a), cross(p, q, b)
    denominator = first - second
    scale = first / denominator if abs(denominator) >= 1.0e-12 else 1.0
    return (
        a[0] + (b[0] - a[0]) * scale,
        a[1] + (b[1] - a[1]) * scale,
    )


def rotated_iou(first: dict, second: dict, convention: str) -> float:
    broad = (
        first["dx"] + first["dy"] + second["dx"] + second["dy"]
    ) * 0.5
    if abs(first["x"] - second["x"]) > broad or abs(
        first["y"] - second["y"]
    ) > broad:
        return 0.0
    polygon = corners(first, convention)
    clip = corners(second, convention)
    for edge in range(4):
        p, q = clip[edge], clip[(edge + 1) % 4]
        previous_polygon, polygon = polygon, []
        if not previous_polygon:
            break
        for index, current in enumerate(previous_polygon):
            previous = previous_polygon[index - 1]
            current_inside = cross(p, q, current) >= -1.0e-9
            previous_inside = cross(p, q, previous) >= -1.0e-9
            if current_inside != previous_inside:
                polygon.append(intersection(previous, current, p, q))
            if current_inside:
                polygon.append(current)
    if not polygon:
        return 0.0
    intersection_area = abs(
        sum(
            polygon[index][0] * polygon[(index + 1) % len(polygon)][1]
            - polygon[(index + 1) % len(polygon)][0] * polygon[index][1]
            for index in range(len(polygon))
        )
    ) * 0.5
    union = (
        first["dx"] * first["dy"]
        + second["dx"] * second["dy"]
        - intersection_area
    )
    return intersection_area / union if union > 0.0 else 0.0


def nms(candidates: list[dict], threshold: float,
        convention: str) -> list[dict]:
    kept = []
    for candidate in candidates:
        if all(
            rotated_iou(previous, candidate, convention) <= threshold
            for previous in kept
        ):
            kept.append(candidate)
            if len(kept) == 500:
                break
    return kept


def compare(actual: list[dict], expected: list[dict]) -> dict:
    same_count = len(actual) == len(expected)
    same_indices = same_count and all(
        first["source_index"] == second["source_index"]
        and first["label"] == second["label"]
        for first, second in zip(actual, expected)
    )
    maximum = 0.0
    if same_count:
        for first, second in zip(actual, expected):
            maximum = max(
                maximum,
                max(abs(first[name] - second[name]) for name in FIELDS),
            )
    return {
        "actual_count": len(actual),
        "expected_count": len(expected),
        "same_count": same_count,
        "same_indices_and_labels": same_indices,
        "max_abs_diff": maximum,
        "passed": same_count and same_indices and maximum <= 2.0e-4,
    }


def main() -> int:
    args = parse_args()
    gpu_pre_nms = read_csv(args.pre_nms)
    gpu_detections = read_csv(args.detections)
    numpy_pre_nms = numpy_decode(
        args.reference_head_dir, args.score_threshold
    )
    numpy_detections = nms(
        gpu_pre_nms, args.nms_iou, args.nms_convention
    )
    result = {
        "decode": compare(gpu_pre_nms, numpy_pre_nms),
        "gpu_nms_vs_python": compare(gpu_detections, numpy_detections),
    }
    if args.reference_detections:
        result["full_pipeline_vs_reference"] = compare(
            gpu_detections, read_csv(args.reference_detections)
        )
    result["passed"] = all(
        item["passed"] for key, item in result.items() if key != "passed"
    )
    output = args.output or args.detections.with_name(
        "detection_validation.json"
    )
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"GPU detection validation: {'PASS' if result['passed'] else 'FAIL'}")
    for name, comparison in result.items():
        if name == "passed":
            continue
        print(
            f"{name}: {comparison['actual_count']}/"
            f"{comparison['expected_count']}, max diff "
            f"{comparison['max_abs_diff']:.9g}, "
            f"{'PASS' if comparison['passed'] else 'FAIL'}"
        )
    print(f"report: {output}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
