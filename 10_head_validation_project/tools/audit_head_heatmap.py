#!/usr/bin/env python3
"""Audit CenterHead heatmap responses at Waymo ground-truth centers."""

from __future__ import annotations

import argparse
import csv
import json
import math
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np


CLASS_NAMES = {0: "VEHICLE", 1: "PEDESTRIAN", 2: "CYCLIST"}
WAYMO_CLASS_INDEX = {
    "TYPE_VEHICLE": 0,
    "TYPE_PEDESTRIAN": 1,
    "TYPE_CYCLIST": 2,
}
GRID_HEIGHT = 468
GRID_WIDTH = 468
POINT_CLOUD_X = -74.88
POINT_CLOUD_Y = -74.88
VOXEL_X = 0.32
VOXEL_Y = 0.32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare CenterHead heatmap peaks with Waymo labels."
    )
    parser.add_argument("--eval-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--archive",
        type=Path,
        default=None,
        help="Waymo derived archive. Inferred from export_summary.json when omitted.",
    )
    parser.add_argument("--frames", nargs="*", default=None)
    parser.add_argument("--local-radius", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=None,
        help="Override the threshold recorded by 08_decode_project.",
    )
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    if args.local_radius < 0:
        parser.error("--local-radius must be non-negative")
    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    if args.score_threshold is not None and not 0.0 <= args.score_threshold <= 1.0:
        parser.error("--score-threshold must be in [0, 1]")
    return args


def sigmoid(values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values, dtype=np.float32)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result


def list_frames(eval_dir: Path, requested: list[str] | None) -> list[str]:
    available = sorted(
        path.name
        for path in eval_dir.glob("frame_*")
        if path.is_dir() and (path / "07_head" / "hm.bin").is_file()
    )
    if requested is None:
        return available
    missing = sorted(set(requested) - set(available))
    if missing:
        raise FileNotFoundError(f"missing evaluated frames: {', '.join(missing)}")
    return [frame for frame in requested if frame in available]


def infer_archive(eval_dir: Path, frames: list[str]) -> Path:
    if not frames:
        raise ValueError("no evaluated frames found")
    summary_path = eval_dir / frames[0] / "export_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    archive = Path(summary["archive"])
    if not archive.is_file():
        raise FileNotFoundError(f"archive from export summary does not exist: {archive}")
    return archive


def read_heatmap(frame_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    path = frame_dir / "07_head" / "hm.bin"
    logits = np.fromfile(path, dtype="<f4")
    expected = len(CLASS_NAMES) * GRID_HEIGHT * GRID_WIDTH
    if logits.size != expected:
        raise ValueError(f"{path} has {logits.size} floats, expected {expected}")
    logits = logits.reshape(len(CLASS_NAMES), GRID_HEIGHT, GRID_WIDTH)
    if not np.isfinite(logits).all():
        raise ValueError(f"{path} contains non-finite values")
    return logits, sigmoid(logits)


def read_labels(zf: zipfile.ZipFile, frame: str) -> list[dict[str, float | int | str]]:
    entry = f"{frame}/labels/laser_labels.json"
    items = json.loads(zf.read(entry).decode("utf-8"))
    labels: list[dict[str, float | int | str]] = []
    for item in items:
        class_index = WAYMO_CLASS_INDEX.get(item.get("type"))
        if class_index is None:
            continue
        box = item["box"]
        labels.append(
            {
                "class_index": class_index,
                "class_name": CLASS_NAMES[class_index],
                "x": float(box["center_x"]),
                "y": float(box["center_y"]),
            }
        )
    return labels


def read_thresholds(frame_dir: Path, override: float | None) -> list[float]:
    if override is not None:
        return [override] * len(CLASS_NAMES)
    config_path = frame_dir / "08_detections" / "decode_config.json"
    if not config_path.is_file():
        return [0.1] * len(CLASS_NAMES)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("use_class_score_thresholds"):
        values = [float(value) for value in config["class_score_thresholds"]]
        if len(values) != len(CLASS_NAMES):
            raise ValueError(f"invalid class thresholds in {config_path}")
        return values
    return [float(config.get("score_threshold", 0.1))] * len(CLASS_NAMES)


def read_detection_keys(frame_dir: Path) -> set[tuple[str, float, float]]:
    path = frame_dir / "match_report.json"
    if not path.is_file():
        return set()
    report = json.loads(path.read_text(encoding="utf-8"))
    keys: set[tuple[str, float, float]] = set()
    for match in report.get("matches", []):
        box = match.get("gt_box")
        if box is None:
            continue
        keys.add((str(box["label"]), round(float(box["x"]), 5), round(float(box["y"]), 5)))
    return keys


def read_detections(frame_dir: Path) -> dict[int, dict[str, float | int]]:
    path = frame_dir / "08_detections" / "detections.csv"
    if not path.is_file():
        return {}
    detections: dict[int, dict[str, float | int]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            source_index = int(row["source_index"])
            detections[source_index] = {
                "source_index": source_index,
                "label": int(row["label"]),
                "x": float(row["x"]),
                "y": float(row["y"]),
                "dx": float(row["dx"]),
                "dy": float(row["dy"]),
                "yaw": float(row["yaw"]),
                "score": float(row["score"]),
            }
    return detections


def is_detected(
    label: dict[str, float | int | str], keys: set[tuple[str, float, float]]
) -> bool:
    key = (
        str(label["class_name"]),
        round(float(label["x"]), 5),
        round(float(label["y"]), 5),
    )
    return key in keys


def top_local_peaks(scores: np.ndarray, top_k: int) -> list[tuple[int, int, float]]:
    padded = np.pad(scores, 1, mode="constant", constant_values=-np.inf)
    neighbors = [
        padded[dy : dy + GRID_HEIGHT, dx : dx + GRID_WIDTH]
        for dy in range(3)
        for dx in range(3)
    ]
    local_max = np.maximum.reduce(neighbors)
    indices = np.flatnonzero(scores.reshape(-1) >= local_max.reshape(-1))
    if indices.size == 0:
        return []
    values = scores.reshape(-1)[indices]
    count = min(top_k, indices.size)
    selected = np.argpartition(values, -count)[-count:]
    selected = selected[np.argsort(values[selected])[::-1]]
    result: list[tuple[int, int, float]] = []
    for selected_index in selected:
        flat_index = int(indices[selected_index])
        cell_y, cell_x = divmod(flat_index, GRID_WIDTH)
        result.append((cell_x, cell_y, float(scores[cell_y, cell_x])))
    return result


def world_x(cell_x: int) -> float:
    return POINT_CLOUD_X + cell_x * VOXEL_X


def world_y(cell_y: int) -> float:
    return POINT_CLOUD_Y + cell_y * VOXEL_Y


def audit_label(
    frame: str,
    label: dict[str, float | int | str],
    logits: np.ndarray,
    scores: np.ndarray,
    peaks: list[tuple[int, int, float]],
    threshold: float,
    radius: int,
    detected_keys: set[tuple[str, float, float]],
    detections: dict[int, dict[str, float | int]],
) -> dict[str, object]:
    class_index = int(label["class_index"])
    x = float(label["x"])
    y = float(label["y"])
    cell_float_x = (x - POINT_CLOUD_X) / VOXEL_X
    cell_float_y = (y - POINT_CLOUD_Y) / VOXEL_Y
    cell_x = math.floor(cell_float_x)
    cell_y = math.floor(cell_float_y)
    in_range = 0 <= cell_x < GRID_WIDTH and 0 <= cell_y < GRID_HEIGHT
    detected = is_detected(label, detected_keys)

    row: dict[str, object] = {
        "frame": frame,
        "class_index": class_index,
        "class_name": label["class_name"],
        "gt_x": x,
        "gt_y": y,
        "cell_float_x": cell_float_x,
        "cell_float_y": cell_float_y,
        "cell_x": cell_x,
        "cell_y": cell_y,
        "in_range": in_range,
        "detected": detected,
        "score_threshold": threshold,
        "center_logit": None,
        "center_score": None,
        "local_max_logit": None,
        "local_max_score": None,
        "local_max_cell_x": None,
        "local_max_cell_y": None,
        "local_max_x": None,
        "local_max_y": None,
        "local_max_distance_m": None,
        "global_score_rank": None,
        "nearest_top_peak_distance_m": None,
        "local_peak_source_index": None,
        "local_peak_winning_class": None,
        "emitted": False,
        "emitted_label": None,
        "emitted_x": None,
        "emitted_y": None,
        "emitted_dx": None,
        "emitted_dy": None,
        "emitted_yaw": None,
        "emitted_score": None,
        "emitted_center_distance_m": None,
        "outcome": "OUT_OF_RANGE",
    }
    if not in_range:
        return row

    class_logits = logits[class_index]
    class_scores = scores[class_index]
    x0 = max(0, cell_x - radius)
    x1 = min(GRID_WIDTH, cell_x + radius + 1)
    y0 = max(0, cell_y - radius)
    y1 = min(GRID_HEIGHT, cell_y + radius + 1)
    window = class_scores[y0:y1, x0:x1]
    local_flat = int(np.argmax(window))
    local_y_offset, local_x_offset = divmod(local_flat, window.shape[1])
    local_x = x0 + local_x_offset
    local_y = y0 + local_y_offset
    local_score = float(class_scores[local_y, local_x])
    local_logit = float(class_logits[local_y, local_x])
    source_index = local_y * GRID_WIDTH + local_x
    winning_class = int(np.argmax(logits[:, local_y, local_x]))
    emitted = detections.get(source_index)
    local_world_x = world_x(local_x)
    local_world_y = world_y(local_y)
    nearest_top_distance = None
    if peaks:
        nearest_top_distance = min(
            math.hypot(world_x(px) - x, world_y(py) - y) for px, py, _ in peaks
        )

    if detected:
        outcome = "DETECTED"
    elif local_score < threshold:
        outcome = "LOW_HEATMAP_SCORE"
    elif winning_class != class_index:
        outcome = "CLASS_CONFLICT_AT_PEAK"
    elif emitted is not None and int(emitted["label"]) == class_index:
        outcome = "HIGH_HEATMAP_EMITTED_UNMATCHED"
    else:
        outcome = "HIGH_HEATMAP_NOT_EMITTED"

    row.update(
        {
            "center_logit": float(class_logits[cell_y, cell_x]),
            "center_score": float(class_scores[cell_y, cell_x]),
            "local_max_logit": local_logit,
            "local_max_score": local_score,
            "local_max_cell_x": local_x,
            "local_max_cell_y": local_y,
            "local_max_x": local_world_x,
            "local_max_y": local_world_y,
            "local_max_distance_m": math.hypot(local_world_x - x, local_world_y - y),
            "global_score_rank": int(np.count_nonzero(class_scores > local_score) + 1),
            "nearest_top_peak_distance_m": nearest_top_distance,
            "local_peak_source_index": source_index,
            "local_peak_winning_class": CLASS_NAMES[winning_class],
            "emitted": emitted is not None,
            "emitted_label": (
                CLASS_NAMES.get(int(emitted["label"]), str(emitted["label"]))
                if emitted is not None
                else None
            ),
            "emitted_x": emitted["x"] if emitted is not None else None,
            "emitted_y": emitted["y"] if emitted is not None else None,
            "emitted_dx": emitted["dx"] if emitted is not None else None,
            "emitted_dy": emitted["dy"] if emitted is not None else None,
            "emitted_yaw": emitted["yaw"] if emitted is not None else None,
            "emitted_score": emitted["score"] if emitted is not None else None,
            "emitted_center_distance_m": (
                math.hypot(float(emitted["x"]) - x, float(emitted["y"]) - y)
                if emitted is not None
                else None
            ),
            "outcome": outcome,
        }
    )
    return row


def peak_rows(
    frame: str,
    class_index: int,
    logits: np.ndarray,
    peaks: list[tuple[int, int, float]],
    threshold: float,
) -> list[dict[str, object]]:
    rows = []
    for rank, (cell_x, cell_y, score) in enumerate(peaks, start=1):
        rows.append(
            {
                "frame": frame,
                "class_index": class_index,
                "class_name": CLASS_NAMES[class_index],
                "rank": rank,
                "cell_x": cell_x,
                "cell_y": cell_y,
                "x": world_x(cell_x),
                "y": world_y(cell_y),
                "logit": float(logits[class_index, cell_y, cell_x]),
                "score": score,
                "above_threshold": score >= threshold,
                "score_threshold": threshold,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_heatmap(
    output_path: Path,
    frame: str,
    class_index: int,
    scores: np.ndarray,
    labels: list[dict[str, float | int | str]],
    rows: list[dict[str, object]],
    peaks: list[tuple[int, int, float]],
) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError:
        return

    class_labels = [label for label in labels if int(label["class_index"]) == class_index]
    if not class_labels:
        return
    row_by_position = {
        (round(float(row["gt_x"]), 5), round(float(row["gt_y"]), 5)): row
        for row in rows
        if int(row["class_index"]) == class_index
    }
    log_scores = np.log10(np.clip(scores[class_index], 1.0e-6, 1.0))
    fig, axis = plt.subplots(figsize=(9, 8))
    image = axis.imshow(
        log_scores,
        origin="lower",
        extent=[
            POINT_CLOUD_X,
            POINT_CLOUD_X + GRID_WIDTH * VOXEL_X,
            POINT_CLOUD_Y,
            POINT_CLOUD_Y + GRID_HEIGHT * VOXEL_Y,
        ],
        vmin=-6.0,
        vmax=0.0,
        cmap="magma",
        interpolation="nearest",
    )
    colors = {
        "DETECTED": "#35d07f",
        "LOW_HEATMAP_SCORE": "#55aaff",
        "HIGH_HEATMAP_EMITTED_UNMATCHED": "#ff4d4d",
        "HIGH_HEATMAP_NOT_EMITTED": "#ff9f43",
        "CLASS_CONFLICT_AT_PEAK": "#d980fa",
        "OUT_OF_RANGE": "#aaaaaa",
    }
    for label in class_labels:
        key = (round(float(label["x"]), 5), round(float(label["y"]), 5))
        outcome = str(row_by_position[key]["outcome"])
        axis.scatter(
            [float(label["x"])],
            [float(label["y"])],
            s=52,
            facecolors="none",
            edgecolors=colors[outcome],
            linewidths=1.8,
        )
    if peaks:
        axis.scatter(
            [world_x(px) for px, _, _ in peaks],
            [world_y(py) for _, py, _ in peaks],
            marker="x",
            s=22,
            c="white",
            linewidths=0.8,
        )
    axis.set_title(f"{frame} {CLASS_NAMES[class_index]} heatmap")
    axis.set_xlabel("Waymo vehicle X [m]")
    axis.set_ylabel("Waymo vehicle Y [m]")
    axis.set_aspect("equal")
    legend_items = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="none",
            markeredgecolor=color,
            label=outcome,
        )
        for outcome, color in colors.items()
        if any(str(row["outcome"]) == outcome for row in rows)
    ]
    legend_items.append(
        Line2D([0], [0], marker="x", linestyle="none", color="white", label="TOP PEAK")
    )
    axis.legend(handles=legend_items, loc="lower left", fontsize=7, framealpha=0.85)
    fig.colorbar(image, ax=axis, label="log10(sigmoid heatmap score)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    frames = list_frames(args.eval_dir, args.frames)
    aggregate = json.loads(
        (args.eval_dir / "aggregate_report.json").read_text(encoding="utf-8")
    )
    run_contract = aggregate.get("run_contract")
    if not isinstance(run_contract, dict):
        raise ValueError("aggregate report has no run_contract; rerun the evaluator")
    archive = args.archive if args.archive is not None else infer_archive(args.eval_dir, frames)
    if not archive.is_file():
        raise FileNotFoundError(f"archive does not exist: {archive}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    audit_rows: list[dict[str, object]] = []
    all_peak_rows: list[dict[str, object]] = []
    frame_summaries: list[dict[str, object]] = []

    with zipfile.ZipFile(archive) as zf:
        for frame in frames:
            frame_dir = args.eval_dir / frame
            logits, scores = read_heatmap(frame_dir)
            labels = read_labels(zf, frame)
            thresholds = read_thresholds(frame_dir, args.score_threshold)
            detected_keys = read_detection_keys(frame_dir)
            detections = read_detections(frame_dir)
            peaks_by_class = {
                class_index: top_local_peaks(scores[class_index], args.top_k)
                for class_index in CLASS_NAMES
            }
            frame_rows = [
                audit_label(
                    frame,
                    label,
                    logits,
                    scores,
                    peaks_by_class[int(label["class_index"])],
                    thresholds[int(label["class_index"])],
                    args.local_radius,
                    detected_keys,
                    detections,
                )
                for label in labels
            ]
            audit_rows.extend(frame_rows)
            for class_index in CLASS_NAMES:
                all_peak_rows.extend(
                    peak_rows(
                        frame,
                        class_index,
                        logits,
                        peaks_by_class[class_index],
                        thresholds[class_index],
                    )
                )
                if not args.no_plots:
                    plot_heatmap(
                        args.output_dir / f"{frame}_{CLASS_NAMES[class_index].lower()}_heatmap.png",
                        frame,
                        class_index,
                        scores,
                        labels,
                        frame_rows,
                        peaks_by_class[class_index],
                    )
            frame_summaries.append(
                {
                    "frame": frame,
                    "labels": len(frame_rows),
                    "outcomes": dict(Counter(str(row["outcome"]) for row in frame_rows)),
                    "thresholds": {
                        CLASS_NAMES[index]: thresholds[index] for index in CLASS_NAMES
                    },
                    "heatmap_max_scores": {
                        CLASS_NAMES[index]: float(scores[index].max()) for index in CLASS_NAMES
                    },
                }
            )

    write_csv(args.output_dir / "gt_heatmap_audit.csv", audit_rows)
    write_csv(args.output_dir / "top_heatmap_peaks.csv", all_peak_rows)
    summary = {
        "eval_dir": str(args.eval_dir.resolve()),
        "archive": str(archive.resolve()),
        "run_contract": run_contract,
        "grid": {
            "shape": [len(CLASS_NAMES), GRID_HEIGHT, GRID_WIDTH],
            "point_cloud_start_xy": [POINT_CLOUD_X, POINT_CLOUD_Y],
            "cell_size_xy": [VOXEL_X, VOXEL_Y],
        },
        "local_radius_cells": args.local_radius,
        "top_k": args.top_k,
        "frames": len(frames),
        "labels": len(audit_rows),
        "outcome_counts": dict(Counter(str(row["outcome"]) for row in audit_rows)),
        "class_counts": dict(Counter(str(row["class_name"]) for row in audit_rows)),
        "frame_summaries": frame_summaries,
    }
    (args.output_dir / "head_validation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
