#!/usr/bin/env python3
"""Run the C++/CUDA CenterPoint pipeline on multiple Waymo frames and score it.

This script is intentionally practical rather than abstract:

1. export a frame from a derived Waymo sensor archive
2. call the existing milestone executables
3. decode with configurable NMS/score thresholds
4. greedily match predictions to Waymo laser labels by class and BEV IoU
5. write per-frame and aggregate reports
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
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
DEFAULT_LIDARS = ["TOP", "FRONT", "SIDE_LEFT", "SIDE_RIGHT", "REAR"]
DEFAULT_RETURNS = ["return1", "return2"]


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--weights-root", required=True, type=Path)
    parser.add_argument("--frames", nargs="*", default=None)
    parser.add_argument("--max-frames", type=int, default=3)
    parser.add_argument("--lidars", nargs="+", default=DEFAULT_LIDARS)
    parser.add_argument("--returns", nargs="+", default=DEFAULT_RETURNS)
    parser.add_argument("--drop-nlz", action="store_true")
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--score-threshold", type=float, default=0.35)
    parser.add_argument("--nms-convention", choices=["current", "pcdet"], default="current")
    parser.add_argument("--vehicle-score-threshold", type=float, default=None)
    parser.add_argument("--pedestrian-score-threshold", type=float, default=None)
    parser.add_argument("--cyclist-score-threshold", type=float, default=None)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def list_frames(archive: Path, requested: list[str] | None, max_frames: int) -> list[str]:
    if requested:
        return requested[:max_frames]
    frames: set[str] = set()
    with zipfile.ZipFile(archive) as zf:
        for name in zf.namelist():
            if name.startswith("frame_") and "/lidar/" in name:
                frames.add(name.split("/", 1)[0])
    return sorted(frames)[:max_frames]


def run_command(command: list[str], cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        joined = " ".join(command)
        raise RuntimeError(f"command failed ({completed.returncode}): {joined}\n{completed.stdout}")


def class_thresholds(args: argparse.Namespace) -> list[float] | None:
    values = [
        args.vehicle_score_threshold,
        args.pedestrian_score_threshold,
        args.cyclist_score_threshold,
    ]
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError(
            "class-wise thresholds require vehicle, pedestrian, and cyclist values"
        )
    return [float(value) for value in values]


def decode_config_matches(config_path: Path, args: argparse.Namespace) -> bool:
    if not config_path.exists():
        return False
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if abs(float(config.get("nms_iou_threshold", -1.0)) - args.nms_iou) >= 1e-6:
        return False
    if abs(float(config.get("score_threshold", -1.0)) - args.score_threshold) >= 1e-6:
        return False
    expected_pcdet = args.nms_convention == "pcdet"
    if bool(config.get("use_pcdet_nms_convention", False)) != expected_pcdet:
        return False
    expected_class = class_thresholds(args)
    if expected_class is None:
        return not bool(config.get("use_class_score_thresholds", False))
    actual = config.get("class_score_thresholds")
    if not isinstance(actual, list) or len(actual) != 3:
        return False
    return all(abs(float(a) - b) < 1e-6 for a, b in zip(actual, expected_class))


def exe(project_root: Path, rel: str) -> str:
    return str(project_root / rel)


def run_pipeline(args: argparse.Namespace, frame: str, frame_dir: Path) -> Path:
    project = args.project_root
    weights = args.weights_root
    points_bin = frame_dir / "points.bin"
    logs = frame_dir / "logs"
    detections_csv = frame_dir / "08_detections" / "detections.csv"
    decode_config = frame_dir / "08_detections" / "decode_config.json"

    if args.skip_existing and detections_csv.exists() and decode_config_matches(
        decode_config, args
    ):
        return detections_csv

    export_cmd = [
        sys.executable,
        str(project / "09_full_pipeline_project" / "tools" / "export_waymo_frame.py"),
        str(args.archive),
        str(points_bin),
        "--frame",
        frame,
        "--summary-json",
        str(frame_dir / "export_summary.json"),
        "--lidars",
        *args.lidars,
        "--returns",
        *args.returns,
    ]
    if args.drop_nlz:
        export_cmd.append("--drop-nlz")

    decode_cmd = [
        exe(project, "08_decode_project/build_cuda/Release/centerpoint_decode.exe"),
        str(frame_dir / "07_head"),
        str(frame_dir / "08_detections"),
        str(args.nms_iou),
        str(args.score_threshold),
        args.nms_convention,
    ]
    thresholds = class_thresholds(args)
    if thresholds is not None:
        decode_cmd.extend(str(value) for value in thresholds)

    steps = [
        (
            "01_export",
            export_cmd,
        ),
        (
            "02_voxel",
            [
                exe(project, "02_project/build/Release/centerpoint_voxel_dump.exe"),
                str(points_bin),
                str(frame_dir / "02_voxel"),
                "5",
            ],
        ),
        (
            "03_decorate",
            [
                exe(project, "03_pillar_feature_project/build/Release/centerpoint_decorate_pillars.exe"),
                str(frame_dir / "02_voxel"),
                str(frame_dir / "03_decorated"),
            ],
        ),
        (
            "04_pfn",
            [
                exe(project, "04_pfn_project/build/Release/centerpoint_pfn_checkpoint.exe"),
                str(frame_dir / "03_decorated"),
                str(weights / "04_pfn"),
                str(frame_dir / "04_pfn"),
            ],
        ),
        (
            "05_scatter",
            [
                exe(project, "05_scatter_project/build/Release/centerpoint_scatter.exe"),
                str(frame_dir / "04_pfn"),
                str(frame_dir / "02_voxel"),
                str(frame_dir / "05_scatter"),
            ],
        ),
        (
            "06_rpn",
            [
                exe(project, "06_rpn_project/build_cuda/Release/centerpoint_rpn_full_cuda.exe"),
                str(frame_dir / "05_scatter"),
                str(weights / "06_rpn"),
                str(frame_dir / "06_rpn"),
            ],
        ),
        (
            "07_head",
            [
                exe(project, "07_center_head_project/build_cuda/Release/centerpoint_head_cuda.exe"),
                str(frame_dir / "06_rpn"),
                str(weights / "07_head"),
                str(frame_dir / "07_head"),
            ],
        ),
        (
            "08_decode",
            decode_cmd,
        ),
    ]

    for step_name, command in steps:
        run_command(command, project, logs / f"{step_name}.log")
    return detections_csv


def read_predictions(path: Path) -> list[Box]:
    boxes: list[Box] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            boxes.append(
                Box(
                    x=float(row["x"]),
                    y=float(row["y"]),
                    dx=float(row["dx"]),
                    dy=float(row["dy"]),
                    yaw=float(row["yaw"]),
                    label=CLASS_NAMES.get(int(row["label"]), f"class_{row['label']}"),
                    convention="prediction",
                    score=float(row["score"]),
                )
            )
    return sorted(boxes, key=lambda box: box.score, reverse=True)


def read_labels(archive: Path, frame: str) -> list[Box]:
    entry = f"{frame}/labels/laser_labels.json"
    with zipfile.ZipFile(archive) as zf:
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


def normalize_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def box_dict(box: Box) -> dict[str, float | str]:
    return {
        "x": box.x,
        "y": box.y,
        "dx": box.dx,
        "dy": box.dy,
        "yaw": box.yaw,
        "label": box.label,
        "score": box.score,
    }


def geometry_errors(pred: Box, gt: Box) -> dict[str, float]:
    # CenterPoint decoded yaw uses the prediction convention. Waymo labels use
    # the label convention handled in corners(), so keep both values visible.
    pred_yaw_as_waymo = -pred.yaw - math.pi / 2.0
    return {
        "center_distance_m": center_distance(pred, gt),
        "dx_abs_error_m": abs(pred.dx - gt.dx),
        "dy_abs_error_m": abs(pred.dy - gt.dy),
        "raw_yaw_abs_error_rad": abs(normalize_angle(pred.yaw - gt.yaw)),
        "waymo_converted_yaw_abs_error_rad": abs(
            normalize_angle(pred_yaw_as_waymo - gt.yaw)
        ),
    }


def corners(box: Box) -> list[tuple[float, float]]:
    if box.convention == "waymo_label":
        half_x = box.dy / 2.0
        half_y = box.dx / 2.0
    else:
        half_x = box.dx / 2.0
        half_y = box.dy / 2.0
    local = [
        (half_x, half_y),
        (half_x, -half_y),
        (-half_x, -half_y),
        (-half_x, half_y),
    ]
    c = math.cos(box.yaw)
    s = math.sin(box.yaw)
    return [(box.x + lx * c + ly * s, box.y - lx * s + ly * c) for lx, ly in local]


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


def rotated_iou(a: Box, b: Box) -> float:
    inter = intersect_area(corners(a), corners(b))
    union = a.dx * a.dy + b.dx * b.dy - inter
    return inter / union if union > 0.0 else 0.0


def evaluate_frame(
    frame: str, predictions: list[Box], labels: list[Box], match_iou: float
) -> dict[str, object]:
    matches: list[dict[str, object]] = []
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    for pred_index, pred in enumerate(predictions):
        best_index = -1
        best_iou = 0.0
        for gt_index, gt in enumerate(labels):
            if gt_index in used_gt or gt.label != pred.label:
                continue
            iou = rotated_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_index = gt_index
        if best_index >= 0 and best_iou >= match_iou:
            used_pred.add(pred_index)
            used_gt.add(best_index)
            gt = labels[best_index]
            errors = geometry_errors(pred, gt)
            matches.append(
                {
                    "frame": frame,
                    "label": pred.label,
                    "score": pred.score,
                    "iou": best_iou,
                    **errors,
                    "pred_xy": [pred.x, pred.y],
                    "gt_xy": [gt.x, gt.y],
                    "pred_box": box_dict(pred),
                    "gt_box": box_dict(gt),
                }
            )

    false_positives = []
    for pred_index, pred in enumerate(predictions):
        if pred_index in used_pred:
            continue
        best_iou = 0.0
        nearest_distance = None
        nearest_gt = None
        best_gt = None
        for gt in labels:
            if gt.label != pred.label:
                continue
            iou = rotated_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_gt = gt
            distance = center_distance(pred, gt)
            if nearest_distance is None or distance < nearest_distance:
                nearest_distance = distance
                nearest_gt = gt
        false_positives.append(
            {
                "frame": frame,
                "label": pred.label,
                "score": pred.score,
                "best_same_class_iou": best_iou,
                "nearest_same_class_center_distance_m": nearest_distance,
                "pred_xy": [pred.x, pred.y],
                "pred_box": box_dict(pred),
                "nearest_gt_box": box_dict(nearest_gt) if nearest_gt else None,
                "best_iou_gt_box": box_dict(best_gt) if best_gt else None,
            }
        )

    false_negatives = []
    for gt_index, gt in enumerate(labels):
        if gt_index in used_gt:
            continue
        best_iou = 0.0
        nearest_distance = None
        best_score = None
        nearest_score = None
        best_pred = None
        nearest_pred = None
        for pred in predictions:
            if pred.label != gt.label:
                continue
            iou = rotated_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_score = pred.score
                best_pred = pred
            distance = center_distance(pred, gt)
            if nearest_distance is None or distance < nearest_distance:
                nearest_distance = distance
                nearest_score = pred.score
                nearest_pred = pred
        false_negatives.append(
            {
                "frame": frame,
                "label": gt.label,
                "best_prediction_iou": best_iou,
                "best_prediction_score": best_score,
                "nearest_prediction_center_distance_m": nearest_distance,
                "nearest_prediction_score": nearest_score,
                "gt_xy": [gt.x, gt.y],
                "gt_box": box_dict(gt),
                "best_prediction_box": box_dict(best_pred) if best_pred else None,
                "nearest_prediction_box": box_dict(nearest_pred)
                if nearest_pred
                else None,
            }
        )

    tp = len(matches)
    fp = len(predictions) - tp
    fn = len(labels) - tp
    return {
        "frame": frame,
        "predictions": len(predictions),
        "labels": len(labels),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "matches": matches,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "prediction_class_counts": count_by_label(predictions),
        "label_class_counts": count_by_label(labels),
    }


def count_by_label(boxes: list[Box]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for box in boxes:
        counts[box.label] = counts.get(box.label, 0) + 1
    return dict(sorted(counts.items()))


def write_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["frame", "predictions", "labels", "tp", "fp", "fn", "precision", "recall"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def main() -> int:
    args = parse_args()
    frames = list_frames(args.archive, args.frames, args.max_frames)
    if not frames:
        raise RuntimeError("no frames selected")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame_reports: list[dict[str, object]] = []
    all_matches: list[dict[str, object]] = []
    all_false_positives: list[dict[str, object]] = []
    all_false_negatives: list[dict[str, object]] = []

    for frame in frames:
        frame_dir = args.output_dir / frame
        if frame_dir.exists() and not args.skip_existing:
            shutil.rmtree(frame_dir)
        frame_dir.mkdir(parents=True, exist_ok=True)

        detections_csv = run_pipeline(args, frame, frame_dir)
        preds = read_predictions(detections_csv)
        labels = read_labels(args.archive, frame)
        report = evaluate_frame(frame, preds, labels, args.match_iou)
        frame_reports.append(report)
        all_matches.extend(report["matches"])
        all_false_positives.extend(report["false_positives"])
        all_false_negatives.extend(report["false_negatives"])
        (frame_dir / "match_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )

    total_tp = sum(int(row["tp"]) for row in frame_reports)
    total_fp = sum(int(row["fp"]) for row in frame_reports)
    total_fn = sum(int(row["fn"]) for row in frame_reports)
    aggregate = {
        "archive": str(args.archive),
        "frames": frames,
        "nms_iou": args.nms_iou,
        "score_threshold": args.score_threshold,
        "nms_convention": args.nms_convention,
        "class_score_thresholds": class_thresholds(args),
        "match_iou": args.match_iou,
        "total_predictions": sum(int(row["predictions"]) for row in frame_reports),
        "total_labels": sum(int(row["labels"]) for row in frame_reports),
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0,
        "recall": total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0,
        "frame_reports": frame_reports,
        "matches": all_matches,
        "false_positives": all_false_positives,
        "false_negatives": all_false_negatives,
    }

    (args.output_dir / "aggregate_report.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )
    write_summary_csv(args.output_dir / "frame_summary.csv", frame_reports)
    print(json.dumps(aggregate, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
