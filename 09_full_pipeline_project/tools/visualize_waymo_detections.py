#!/usr/bin/env python3
"""Draw Waymo point cloud, predicted boxes, and laser labels in BEV.

This is a lightweight visual sanity check for the C++/CUDA pipeline output.
It reads:
  - CenterPoint 5-feature point bin: x, y, z, intensity, elongation
  - detections.csv from 08_decode_project
  - frame_XXX/labels/laser_labels.json from a derived Waymo segment zip
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CLASS_NAMES = {
    0: "VEHICLE",
    1: "PEDESTRIAN",
    2: "CYCLIST",
}

WAYMO_TO_MODEL = {
    "TYPE_VEHICLE": "VEHICLE",
    "TYPE_PEDESTRIAN": "PEDESTRIAN",
    "TYPE_CYCLIST": "CYCLIST",
}


@dataclass
class Box:
    x: float
    y: float
    dx: float
    dy: float
    yaw: float
    label: str
    convention: str
    score: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--points-bin", required=True, type=Path)
    parser.add_argument("--detections-csv", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--frame", default="frame_000")
    parser.add_argument("--output-png", required=True, type=Path)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--score-threshold", type=float, default=0.2)
    parser.add_argument("--max-points", type=int, default=120_000)
    parser.add_argument("--range", type=float, default=80.0)
    return parser.parse_args()


def read_points(path: Path) -> np.ndarray:
    values = np.fromfile(path, dtype="<f4")
    if values.size % 5 != 0:
        raise ValueError(f"{path} is not Nx5 float32")
    return values.reshape(-1, 5)


def read_predictions(path: Path, score_threshold: float) -> list[Box]:
    boxes: list[Box] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            score = float(row["score"])
            if score < score_threshold:
                continue
            label = CLASS_NAMES.get(int(row["label"]), f"class_{row['label']}")
            boxes.append(
                Box(
                    x=float(row["x"]),
                    y=float(row["y"]),
                    dx=float(row["dx"]),
                    dy=float(row["dy"]),
                    yaw=float(row["yaw"]),
                    label=label,
                    convention="prediction",
                    score=score,
                )
            )
    return boxes


def read_labels(archive_path: Path, frame: str) -> list[Box]:
    entry = f"{frame}/labels/laser_labels.json"
    with zipfile.ZipFile(archive_path) as zf:
        labels = json.loads(zf.read(entry).decode("utf-8"))

    boxes: list[Box] = []
    for item in labels:
        label = WAYMO_TO_MODEL.get(item.get("type"))
        if label is None:
            continue
        box = item["box"]
        boxes.append(
            Box(
                x=float(box["center_x"]),
                y=float(box["center_y"]),
                dx=float(box["width"]),
                dy=float(box["length"]),
                yaw=float(box["heading"]),
                label=label,
                convention="waymo_label",
            )
        )
    return boxes


def corners(box: Box) -> np.ndarray:
    if box.convention == "waymo_label":
        # Waymo label JSON stores width, length, heading. The heading points
        # along the length axis and is positive clockwise from vehicle +x.
        half_x = box.dy / 2.0
        half_y = box.dx / 2.0
    elif box.convention == "prediction":
        # CenterPoint Waymo predictions in this repo are written as
        # x,y,z,dx,dy,dz,yaw where dx is width-like and dy is length-like.
        # The NMS code uses the same clockwise-positive rotation convention.
        half_x = box.dx / 2.0
        half_y = box.dy / 2.0
    else:
        raise ValueError(f"unknown box convention: {box.convention}")
    pts = np.array(
        [
            [half_x, half_y],
            [half_x, -half_y],
            [-half_x, -half_y],
            [-half_x, half_y],
            [half_x, half_y],
        ],
        dtype=np.float32,
    )
    c = math.cos(box.yaw)
    s = math.sin(box.yaw)
    rot_mat_t = np.array([[c, -s], [s, c]], dtype=np.float32)
    return pts @ rot_mat_t + np.array([box.x, box.y], dtype=np.float32)


def class_counts(boxes: list[Box]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for box in boxes:
        counts[box.label] = counts.get(box.label, 0) + 1
    return dict(sorted(counts.items()))


def nearest_label_distances(preds: list[Box], labels: list[Box], top_k: int = 10) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for pred in preds[:top_k]:
        same = [gt for gt in labels if gt.label == pred.label]
        if not same:
            continue
        nearest = min(same, key=lambda gt: math.hypot(pred.x - gt.x, pred.y - gt.y))
        rows.append(
            {
                "pred_label": pred.label,
                "pred_score": pred.score,
                "pred_xy": [pred.x, pred.y],
                "nearest_gt_xy": [nearest.x, nearest.y],
                "center_distance_m": math.hypot(pred.x - nearest.x, pred.y - nearest.y),
            }
        )
    return rows


def draw(args: argparse.Namespace) -> dict[str, object]:
    points = read_points(args.points_bin)
    preds = read_predictions(args.detections_csv, args.score_threshold)
    labels = read_labels(args.archive, args.frame)

    if len(points) > args.max_points:
        rng = np.random.default_rng(7)
        points_to_draw = points[rng.choice(len(points), args.max_points, replace=False)]
    else:
        points_to_draw = points

    fig, ax = plt.subplots(figsize=(12, 12), dpi=160)
    ax.scatter(points_to_draw[:, 0], points_to_draw[:, 1], s=0.08, c="#222222", alpha=0.22)

    for box in labels:
        pts = corners(box)
        ax.plot(pts[:, 0], pts[:, 1], color="#2563EB", linewidth=1.0, alpha=0.75)

    for box in preds:
        pts = corners(box)
        ax.plot(pts[:, 0], pts[:, 1], color="#F97316", linewidth=1.6, alpha=0.95)
        ax.text(box.x, box.y, f"{box.label[0]} {box.score:.2f}", color="#C2410C", fontsize=5)

    lim = args.range
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.4, color="#DDDDDD")
    ax.set_xlabel("x forward (m)")
    ax.set_ylabel("y left (m)")
    ax.set_title(
        f"{args.frame} BEV: predictions score>={args.score_threshold} "
        f"(orange) vs Waymo labels (blue)"
    )
    ax.text(
        0.01,
        0.01,
        "Orange: C++/CUDA detections  Blue: Waymo laser_labels",
        transform=ax.transAxes,
        fontsize=8,
        color="#333333",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )

    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output_png)
    plt.close(fig)

    return {
        "points": int(points.shape[0]),
        "predictions_drawn": len(preds),
        "labels_drawn": len(labels),
        "prediction_class_counts": class_counts(preds),
        "label_class_counts": class_counts(labels),
        "nearest_label_distances_top_predictions": nearest_label_distances(preds, labels),
        "output_png": str(args.output_png),
    }


def main() -> int:
    args = parse_args()
    summary = draw(args)
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
