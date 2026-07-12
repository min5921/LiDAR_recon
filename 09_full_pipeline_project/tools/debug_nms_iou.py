#!/usr/bin/env python3
"""Inspect duplicate CenterPoint detections with rotated BEV IoU.

The C++ decoder already runs rotated NMS, but a BEV picture can still show
multiple boxes near one object. This tool gives numbers for those overlaps:

  - prediction/prediction pairs by center distance and IoU
  - nearest Waymo ground-truth label for each top prediction
  - threshold-sweep hints for NMS settings
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


CLASS_NAMES = {0: "VEHICLE", 1: "PEDESTRIAN", 2: "CYCLIST"}
WAYMO_TO_MODEL = {
    "TYPE_VEHICLE": "VEHICLE",
    "TYPE_PEDESTRIAN": "PEDESTRIAN",
    "TYPE_CYCLIST": "CYCLIST",
}


@dataclass(frozen=True)
class Box:
    x: float
    y: float
    dx: float
    dy: float
    yaw: float
    label: str
    convention: str
    score: float = 1.0
    source_index: int = -1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detections-csv", required=True, type=Path)
    parser.add_argument("--archive", type=Path, default=None)
    parser.add_argument("--frame", default="frame_000")
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--score-threshold", type=float, default=0.1)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-center-distance", type=float, default=8.0)
    return parser.parse_args()


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
                    source_index=int(row.get("source_index", -1)),
                )
            )
    return sorted(boxes, key=lambda box: box.score, reverse=True)


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


def center_distance(a: Box, b: Box) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def yaw_difference(a: float, b: float) -> float:
    diff = (a - b + math.pi) % (2.0 * math.pi) - math.pi
    return abs(diff)


def corners(box: Box, mode: str) -> list[tuple[float, float]]:
    if box.convention == "waymo_label":
        half_x = box.dy / 2.0
        half_y = box.dx / 2.0
        clockwise = True
    elif box.convention == "prediction":
        half_x = box.dx / 2.0
        half_y = box.dy / 2.0
        clockwise = mode == "centerpoint_clockwise"
    else:
        raise ValueError(f"unknown box convention: {box.convention}")

    local = [
        (half_x, half_y),
        (half_x, -half_y),
        (-half_x, -half_y),
        (-half_x, half_y),
    ]
    c = math.cos(box.yaw)
    s = math.sin(box.yaw)
    pts: list[tuple[float, float]] = []
    for lx, ly in local:
        if clockwise:
            x = box.x + lx * c + ly * s
            y = box.y - lx * s + ly * c
        else:
            x = box.x + lx * c - ly * s
            y = box.y + lx * s + ly * c
        pts.append((x, y))
    return pts


def signed_area(poly: Iterable[tuple[float, float]]) -> float:
    pts = list(poly)
    area = 0.0
    for i, p in enumerate(pts):
        q = pts[(i + 1) % len(pts)]
        area += p[0] * q[1] - q[0] * p[1]
    return area * 0.5


def polygon_area(poly: Iterable[tuple[float, float]]) -> float:
    return abs(signed_area(poly))


def cross(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def line_intersection(
    a: tuple[float, float],
    b: tuple[float, float],
    p: tuple[float, float],
    q: tuple[float, float],
) -> tuple[float, float]:
    a1 = cross(p, q, a)
    a2 = cross(p, q, b)
    den = a1 - a2
    if abs(den) < 1e-12:
        return b
    t = a1 / den
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def inside(
    point: tuple[float, float],
    edge_a: tuple[float, float],
    edge_b: tuple[float, float],
    clip_is_ccw: bool,
) -> bool:
    value = cross(edge_a, edge_b, point)
    return value >= -1e-9 if clip_is_ccw else value <= 1e-9


def intersect_area(
    subject: list[tuple[float, float]], clip: list[tuple[float, float]]
) -> float:
    if not subject or not clip:
        return 0.0
    clip_is_ccw = signed_area(clip) > 0.0
    poly = subject[:]
    for i, edge_a in enumerate(clip):
        if not poly:
            return 0.0
        edge_b = clip[(i + 1) % len(clip)]
        next_poly: list[tuple[float, float]] = []
        for j, cur in enumerate(poly):
            prev = poly[(j - 1) % len(poly)]
            cur_inside = inside(cur, edge_a, edge_b, clip_is_ccw)
            prev_inside = inside(prev, edge_a, edge_b, clip_is_ccw)
            if cur_inside != prev_inside:
                next_poly.append(line_intersection(prev, cur, edge_a, edge_b))
            if cur_inside:
                next_poly.append(cur)
        poly = next_poly
    return polygon_area(poly)


def rotated_iou(a: Box, b: Box, mode: str) -> float:
    ca = corners(a, mode)
    cb = corners(b, mode)
    inter = intersect_area(ca, cb)
    union = a.dx * a.dy + b.dx * b.dy - inter
    return inter / union if union > 0.0 else 0.0


def prediction_pairs(preds: list[Box], max_distance: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for i, a in enumerate(preds):
        for j in range(i + 1, len(preds)):
            b = preds[j]
            if a.label != b.label:
                continue
            dist = center_distance(a, b)
            if dist > max_distance:
                continue
            iou_cpp = rotated_iou(a, b, "cpp_nms_like")
            iou_cp = rotated_iou(a, b, "centerpoint_clockwise")
            rows.append(
                {
                    "i": i,
                    "j": j,
                    "label": a.label,
                    "score_i": a.score,
                    "score_j": b.score,
                    "center_distance_m": dist,
                    "yaw_diff_rad": yaw_difference(a.yaw, b.yaw),
                    "iou_cpp_nms_like": iou_cpp,
                    "iou_centerpoint_clockwise": iou_cp,
                    "suppressed_at_0_7": iou_cpp > 0.7,
                    "suppressed_at_0_5": iou_cpp > 0.5,
                    "suppressed_at_0_3": iou_cpp > 0.3,
                }
            )
    return sorted(rows, key=lambda row: (row["center_distance_m"], -row["score_j"]))


def nearest_gt_rows(preds: list[Box], labels: list[Box], top_k: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for i, pred in enumerate(preds[:top_k]):
        same = [gt for gt in labels if gt.label == pred.label]
        if not same:
            rows.append(
                {
                    "rank": i,
                    "label": pred.label,
                    "score": pred.score,
                    "nearest_gt_center_distance_m": None,
                    "nearest_gt_iou_bev": None,
                }
            )
            continue
        nearest = max(same, key=lambda gt: rotated_iou(pred, gt, "centerpoint_clockwise"))
        rows.append(
            {
                "rank": i,
                "label": pred.label,
                "score": pred.score,
                "pred_xy": [pred.x, pred.y],
                "nearest_gt_xy": [nearest.x, nearest.y],
                "nearest_gt_center_distance_m": center_distance(pred, nearest),
                "nearest_gt_iou_bev": rotated_iou(pred, nearest, "centerpoint_clockwise"),
            }
        )
    return rows


def write_pair_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "i",
        "j",
        "label",
        "score_i",
        "score_j",
        "center_distance_m",
        "yaw_diff_rad",
        "iou_cpp_nms_like",
        "iou_centerpoint_clockwise",
        "suppressed_at_0_7",
        "suppressed_at_0_5",
        "suppressed_at_0_3",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    args = parse_args()
    preds = read_predictions(args.detections_csv, args.score_threshold)
    labels = read_labels(args.archive, args.frame) if args.archive else []
    pairs = prediction_pairs(preds[: args.top_k], args.max_center_distance)
    gt_rows = nearest_gt_rows(preds, labels, args.top_k) if labels else []

    summary = {
        "detections_csv": str(args.detections_csv),
        "score_threshold": args.score_threshold,
        "predictions_used": len(preds),
        "top_k_for_pair_scan": min(args.top_k, len(preds)),
        "same_class_pairs_within_distance": len(pairs),
        "pairs_iou_cpp_gt_0_7": sum(1 for row in pairs if row["iou_cpp_nms_like"] > 0.7),
        "pairs_iou_cpp_gt_0_5": sum(1 for row in pairs if row["iou_cpp_nms_like"] > 0.5),
        "pairs_iou_cpp_gt_0_3": sum(1 for row in pairs if row["iou_cpp_nms_like"] > 0.3),
        "nearest_gt_available": bool(labels),
        "gt_labels": len(labels),
        "top_prediction_pairs": pairs[:20],
        "top_predictions_nearest_gt": gt_rows[:20],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_pair_csv(args.output_csv, pairs)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
