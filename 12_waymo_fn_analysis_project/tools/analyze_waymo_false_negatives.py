#!/usr/bin/env python3
"""Analyze Waymo false negatives with point, heatmap, and geometry evidence."""

from __future__ import annotations

import argparse
import csv
import json
import math
import zipfile
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np


WAYMO_TO_CLASS = {
    "TYPE_VEHICLE": "VEHICLE",
    "TYPE_PEDESTRIAN": "PEDESTRIAN",
    "TYPE_CYCLIST": "CYCLIST",
}
CLASS_INDEX = {"VEHICLE": 0, "PEDESTRIAN": 1, "CYCLIST": 2}
LIDARS = ["TOP", "FRONT", "SIDE_LEFT", "SIDE_RIGHT", "REAR"]
RETURNS = ["return1", "return2"]
SCHEMA_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--archive", type=Path, default=None)
    parser.add_argument("--frames", nargs="*", default=None)
    parser.add_argument("--local-radius", type=int, default=2)
    parser.add_argument("--low-point-threshold", type=int, default=5)
    parser.add_argument("--point-sample-limit", type=int, default=80000)
    parser.add_argument("--no-figures", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_path(path: Path) -> str:
    return str(path.resolve())


def sigmoid(values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values, dtype=np.float32)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def normalize_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def source_name(lidar: str, return_name: str) -> str:
    return f"{lidar}_{return_name}"


def source_parts(name: str) -> tuple[str, str]:
    lidar, return_number = name.rsplit("_return", 1)
    return lidar, f"return{return_number}"


def load_source_points(
    archive: zipfile.ZipFile, frame: str
) -> dict[str, np.ndarray]:
    names = set(archive.namelist())
    sources = {}
    for lidar in LIDARS:
        for return_name in RETURNS:
            name = source_name(lidar, return_name)
            entry = f"{frame}/lidar/{name}.bin"
            if entry not in names:
                continue
            values = np.frombuffer(archive.read(entry), dtype="<f4")
            if values.size % 6 != 0:
                raise ValueError(f"{entry} has {values.size} floats, expected Nx6")
            sources[name] = values.reshape(-1, 6)
    if not sources:
        raise FileNotFoundError(f"no lidar sources found for {frame}")
    return sources


def load_labels(archive: zipfile.ZipFile, frame: str) -> list[dict[str, object]]:
    entry = f"{frame}/labels/laser_labels.json"
    items = json.loads(archive.read(entry).decode("utf-8"))
    labels = []
    for item in items:
        class_name = WAYMO_TO_CLASS.get(str(item.get("type")))
        if class_name is None:
            continue
        box = item["box"]
        labels.append(
            {
                "id": str(item["id"]),
                "class_name": class_name,
                "box": {
                    "x": float(box["center_x"]),
                    "y": float(box["center_y"]),
                    "z": float(box["center_z"]),
                    "length": float(box["length"]),
                    "width": float(box["width"]),
                    "height": float(box["height"]),
                    "heading": float(box["heading"]),
                },
                "official_num_lidar_points": int(
                    item.get("num_lidar_points_in_box", 0)
                ),
                "official_num_top_lidar_points": int(
                    item.get("num_top_lidar_points_in_box", 0)
                ),
            }
        )
    return labels


def load_predictions(frame_dir: Path) -> list[dict[str, object]]:
    path = frame_dir / "08_detections" / "detections.csv"
    predictions = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            class_index = int(row["label"])
            predictions.append(
                {
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "width": float(row["dx"]),
                    "length": float(row["dy"]),
                    "heading": normalize_angle(
                        -float(row["yaw"]) - math.pi / 2.0
                    ),
                    "raw_yaw": float(row["yaw"]),
                    "class_name": next(
                        name for name, index in CLASS_INDEX.items() if index == class_index
                    ),
                    "score": float(row["score"]),
                }
            )
    return sorted(predictions, key=lambda row: float(row["score"]), reverse=True)


def box_corners(box: dict[str, object]) -> list[tuple[float, float]]:
    half_length = float(box["length"]) / 2.0
    half_width = float(box["width"]) / 2.0
    heading = float(box["heading"])
    cosine = math.cos(heading)
    sine = math.sin(heading)
    local = [
        (half_length, half_width),
        (-half_length, half_width),
        (-half_length, -half_width),
        (half_length, -half_width),
    ]
    result = [
        (
            float(box["x"]) + local_x * cosine - local_y * sine,
            float(box["y"]) + local_x * sine + local_y * cosine,
        )
        for local_x, local_y in local
    ]
    if signed_area(result) < 0.0:
        result.reverse()
    return result


def signed_area(polygon: Iterable[tuple[float, float]]) -> float:
    points = list(polygon)
    area = 0.0
    for index, point in enumerate(points):
        following = points[(index + 1) % len(points)]
        area += point[0] * following[1] - following[0] * point[1]
    return area * 0.5


def polygon_area(polygon: Iterable[tuple[float, float]]) -> float:
    return abs(signed_area(polygon))


def cross(
    first: tuple[float, float],
    second: tuple[float, float],
    third: tuple[float, float],
) -> float:
    return (second[0] - first[0]) * (third[1] - first[1]) - (
        second[1] - first[1]
    ) * (third[0] - first[0])


def line_intersection(
    first: tuple[float, float],
    second: tuple[float, float],
    clip_first: tuple[float, float],
    clip_second: tuple[float, float],
) -> tuple[float, float]:
    first_dx = second[0] - first[0]
    first_dy = second[1] - first[1]
    clip_dx = clip_second[0] - clip_first[0]
    clip_dy = clip_second[1] - clip_first[1]
    denominator = first_dx * clip_dy - first_dy * clip_dx
    if abs(denominator) < 1.0e-12:
        return second
    delta_x = clip_first[0] - first[0]
    delta_y = clip_first[1] - first[1]
    factor = (delta_x * clip_dy - delta_y * clip_dx) / denominator
    return first[0] + factor * first_dx, first[1] + factor * first_dy


def polygon_intersection_area(
    subject: list[tuple[float, float]], clip: list[tuple[float, float]]
) -> float:
    output = subject
    for clip_index, clip_first in enumerate(clip):
        clip_second = clip[(clip_index + 1) % len(clip)]
        input_points = output
        output = []
        if not input_points:
            return 0.0
        previous = input_points[-1]
        previous_inside = cross(clip_first, clip_second, previous) >= -1.0e-9
        for current in input_points:
            current_inside = cross(clip_first, clip_second, current) >= -1.0e-9
            if current_inside:
                if not previous_inside:
                    output.append(
                        line_intersection(previous, current, clip_first, clip_second)
                    )
                output.append(current)
            elif previous_inside:
                output.append(
                    line_intersection(previous, current, clip_first, clip_second)
                )
            previous = current
            previous_inside = current_inside
    return polygon_area(output) if output else 0.0


def rotated_iou(first: dict[str, object], second: dict[str, object]) -> float:
    first_corners = box_corners(first)
    second_corners = box_corners(second)
    intersection = polygon_intersection_area(first_corners, second_corners)
    union = polygon_area(first_corners) + polygon_area(second_corners) - intersection
    return intersection / union if union > 0.0 else 0.0


def evaluate_official_geometry(
    predictions: list[dict[str, object]],
    labels: list[dict[str, object]],
    match_iou: float,
) -> dict[str, object]:
    unmatched = set(range(len(labels)))
    matches = []
    for prediction_index, prediction in enumerate(predictions):
        best_label = -1
        best_iou = 0.0
        for label_index in unmatched:
            label = labels[label_index]
            if prediction["class_name"] != label["class_name"]:
                continue
            overlap = rotated_iou(prediction, label["box"])
            if overlap > best_iou:
                best_iou = overlap
                best_label = label_index
        if best_label >= 0 and best_iou >= match_iou:
            unmatched.remove(best_label)
            matches.append(
                {
                    "prediction_index": prediction_index,
                    "label_id": labels[best_label]["id"],
                    "iou": best_iou,
                    "score": float(prediction["score"]),
                    "prediction": prediction,
                }
            )

    best_by_label = {}
    for label in labels:
        compatible = [
            prediction
            for prediction in predictions
            if prediction["class_name"] == label["class_name"]
        ]
        if not compatible:
            best_by_label[str(label["id"])] = {
                "iou": 0.0,
                "score": None,
                "prediction": None,
            }
            continue
        candidates = [
            (rotated_iou(prediction, label["box"]), prediction)
            for prediction in compatible
        ]
        overlap, prediction = max(candidates, key=lambda item: item[0])
        best_by_label[str(label["id"])] = {
            "iou": float(overlap),
            "score": float(prediction["score"]),
            "prediction": prediction,
        }

    true_positives = len(matches)
    false_positives = len(predictions) - true_positives
    false_negatives = len(labels) - true_positives
    return {
        "predictions": len(predictions),
        "labels": len(labels),
        "tp": true_positives,
        "fp": false_positives,
        "fn": false_negatives,
        "precision": (
            true_positives / len(predictions) if predictions else 0.0
        ),
        "recall": true_positives / len(labels) if labels else 0.0,
        "matches": matches,
        "matched_label_ids": [str(row["label_id"]) for row in matches],
        "best_by_label": best_by_label,
    }


def points_in_waymo_box(
    points: np.ndarray,
    box: dict[str, object],
    mirrored_heading: bool = False,
) -> np.ndarray:
    delta_x = points[:, 0] - float(box["x"])
    delta_y = points[:, 1] - float(box["y"])
    heading = float(box["heading"])
    cosine = math.cos(heading)
    sine = math.sin(heading)
    if mirrored_heading:
        local_length = delta_x * cosine - delta_y * sine
        local_width = delta_x * sine + delta_y * cosine
    else:
        local_length = delta_x * cosine + delta_y * sine
        local_width = -delta_x * sine + delta_y * cosine
    horizontal = (
        (np.abs(local_length) <= float(box["length"]) / 2.0 + 1.0e-5)
        & (np.abs(local_width) <= float(box["width"]) / 2.0 + 1.0e-5)
    )
    vertical = (
        np.abs(points[:, 2] - float(box["z"]))
        <= float(box["height"]) / 2.0 + 1.0e-5
    )
    return horizontal & vertical


def points_in_model_range(points: np.ndarray, limits: list[float]) -> np.ndarray:
    return (
        (points[:, 0] >= limits[0])
        & (points[:, 1] >= limits[1])
        & (points[:, 2] >= limits[2])
        & (points[:, 0] < limits[3])
        & (points[:, 1] < limits[4])
        & (points[:, 2] < limits[5])
    )


def analyze_point_counts(
    sources: dict[str, np.ndarray],
    box: dict[str, object],
    model_range: list[float],
    preprocessing: dict[str, object],
) -> dict[str, object]:
    selected_names = {
        source_name(lidar, return_name)
        for lidar in preprocessing["lidars"]
        for return_name in preprocessing["returns"]
    }
    drop_nlz = bool(preprocessing["drop_nlz"])
    all_box_counts = {}
    selected_box_counts = {}
    selected_effective_counts = {}
    all_count = 0
    mirrored_count = 0
    selected_box_count = 0
    selected_model_count = 0
    selected_non_nlz_count = 0
    effective_count = 0
    effective_top_count = 0
    effective_return1_count = 0
    effective_return2_count = 0
    effective_intensity = []

    for name, points in sources.items():
        box_mask = points_in_waymo_box(points, box)
        mirrored_mask = points_in_waymo_box(points, box, mirrored_heading=True)
        box_count = int(np.count_nonzero(box_mask))
        all_box_counts[name] = box_count
        all_count += box_count
        mirrored_count += int(np.count_nonzero(mirrored_mask))
        if name not in selected_names:
            continue
        model_mask = box_mask & points_in_model_range(points, model_range)
        non_nlz_mask = model_mask & (points[:, 5] < 0.0)
        effective_mask = non_nlz_mask if drop_nlz else model_mask
        model_count = int(np.count_nonzero(model_mask))
        non_nlz_count = int(np.count_nonzero(non_nlz_mask))
        current_count = int(np.count_nonzero(effective_mask))
        selected_box_counts[name] = box_count
        selected_effective_counts[name] = current_count
        selected_box_count += box_count
        selected_model_count += model_count
        selected_non_nlz_count += non_nlz_count
        effective_count += current_count
        lidar, return_name = source_parts(name)
        if lidar == "TOP":
            effective_top_count += current_count
        if return_name == "return1":
            effective_return1_count += current_count
        else:
            effective_return2_count += current_count
        if current_count:
            effective_intensity.append(np.tanh(points[effective_mask, 3]))

    intensity = (
        np.concatenate(effective_intensity)
        if effective_intensity
        else np.empty((0,), dtype=np.float32)
    )
    return {
        "all_archive_points_in_box": all_count,
        "mirrored_heading_points_in_box": mirrored_count,
        "selected_points_in_box": selected_box_count,
        "selected_points_in_model_range": selected_model_count,
        "selected_non_nlz_points_in_model_range": selected_non_nlz_count,
        "effective_model_points": effective_count,
        "effective_top_points": effective_top_count,
        "effective_return1_points": effective_return1_count,
        "effective_return2_points": effective_return2_count,
        "range_retention": (
            selected_model_count / selected_box_count if selected_box_count else 0.0
        ),
        "drop_nlz_retention": (
            selected_non_nlz_count / selected_model_count
            if selected_model_count
            else 0.0
        ),
        "mean_tanh_intensity": (
            float(intensity.mean()) if intensity.size else None
        ),
        "max_tanh_intensity": float(intensity.max()) if intensity.size else None,
        "all_source_box_counts": all_box_counts,
        "selected_source_box_counts": selected_box_counts,
        "selected_source_effective_counts": selected_effective_counts,
    }


def load_heatmap(frame_dir: Path) -> np.ndarray:
    metadata = read_json(frame_dir / "07_head" / "center_head_metadata.json")
    heatmap_row = next(
        row for row in metadata["outputs"] if row.get("name") == "hm"
    )
    shape = tuple(int(value) for value in heatmap_row["shape"])
    values = np.fromfile(frame_dir / "07_head" / "hm.bin", dtype="<f4")
    if values.size != int(np.prod(shape)):
        raise ValueError(f"heatmap size mismatch in {frame_dir}")
    return sigmoid(values.reshape(shape)[0])


def heatmap_evidence(
    heatmap: np.ndarray,
    box: dict[str, object],
    class_name: str,
    voxel_metadata: dict[str, object],
    threshold: float,
    radius: int,
) -> dict[str, object]:
    limits = [float(value) for value in voxel_metadata["point_cloud_range"]]
    voxel_size = [float(value) for value in voxel_metadata["voxel_size"]]
    cell_x = math.floor((float(box["x"]) - limits[0]) / voxel_size[0])
    cell_y = math.floor((float(box["y"]) - limits[1]) / voxel_size[1])
    _, height, width = heatmap.shape
    in_range = 0 <= cell_x < width and 0 <= cell_y < height
    result = {
        "cell_x": cell_x,
        "cell_y": cell_y,
        "in_range": in_range,
        "score_threshold": threshold,
        "center_score": None,
        "local_max_score": None,
        "local_max_cell_x": None,
        "local_max_cell_y": None,
    }
    if not in_range:
        return result
    class_scores = heatmap[CLASS_INDEX[class_name]]
    x0 = max(0, cell_x - radius)
    x1 = min(width, cell_x + radius + 1)
    y0 = max(0, cell_y - radius)
    y1 = min(height, cell_y + radius + 1)
    window = class_scores[y0:y1, x0:x1]
    local_index = int(np.argmax(window))
    local_y_offset, local_x_offset = divmod(local_index, window.shape[1])
    local_x = x0 + local_x_offset
    local_y = y0 + local_y_offset
    result.update(
        {
            "center_score": float(class_scores[cell_y, cell_x]),
            "local_max_score": float(class_scores[local_y, local_x]),
            "local_max_cell_x": local_x,
            "local_max_cell_y": local_y,
        }
    )
    return result


def match_false_negative_label(
    false_negative: dict[str, object],
    labels: list[dict[str, object]],
    used_ids: set[str],
) -> dict[str, object]:
    gt_box = false_negative["gt_box"]
    candidates = [
        label
        for label in labels
        if label["class_name"] == false_negative["label"]
        and str(label["id"]) not in used_ids
    ]
    if not candidates:
        raise ValueError(f"no label candidate for {false_negative}")
    label = min(
        candidates,
        key=lambda row: math.hypot(
            float(row["box"]["x"]) - float(gt_box["x"]),
            float(row["box"]["y"]) - float(gt_box["y"]),
        ),
    )
    distance = math.hypot(
        float(label["box"]["x"]) - float(gt_box["x"]),
        float(label["box"]["y"]) - float(gt_box["y"]),
    )
    if distance > 1.0e-4:
        raise ValueError(f"false negative could not be matched to a Waymo label: {distance}")
    used_ids.add(str(label["id"]))
    return label


def center_in_xy_range(box: dict[str, object], limits: list[float]) -> bool:
    return (
        limits[0] <= float(box["x"]) < limits[3]
        and limits[1] <= float(box["y"]) < limits[4]
    )


def box_overlaps_z_range(box: dict[str, object], limits: list[float]) -> bool:
    minimum = float(box["z"]) - float(box["height"]) / 2.0
    maximum = float(box["z"]) + float(box["height"]) / 2.0
    return maximum >= limits[2] and minimum < limits[5]


def classify_false_negative(
    record: dict[str, object], low_point_threshold: int
) -> tuple[str, list[str]]:
    reasons = []
    if bool(record["official_geometry_matched"]):
        return (
            "EVALUATION_GEOMETRY_MISMATCH",
            ["The official Waymo CCW IoU convention recovers this GT."],
        )
    if not bool(record["center_in_model_xy"]) or not bool(
        record["box_overlaps_model_z"]
    ):
        return "OUT_OF_RANGE", ["The GT box is outside the model range."]

    point_counts = record["point_counts"]
    effective = int(point_counts["effective_model_points"])
    selected = int(point_counts["selected_points_in_box"])
    if effective < low_point_threshold:
        if selected >= low_point_threshold:
            return (
                "PREPROCESSING_SENSITIVE",
                ["Range or NLZ filtering reduces the GT below the point threshold."],
            )
        return "LOW_POINT_COUNT", ["Too few selected points fall inside the GT box."]

    official_points = int(record["official_num_lidar_points"])
    counted_points = int(point_counts["all_archive_points_in_box"])
    if official_points > 0 and counted_points / official_points < 0.5:
        reasons.append("Derived archive points recover less than half of the label count.")
    if float(point_counts["range_retention"]) < 0.5:
        reasons.append("The model range removes more than half of the selected GT points.")
    if reasons:
        return "PREPROCESSING_SENSITIVE", reasons

    heatmap = record["heatmap"]
    local_score = heatmap["local_max_score"]
    threshold = float(heatmap["score_threshold"])
    if local_score is not None and float(local_score) >= threshold:
        return (
            "BOX_REGRESSION_ERROR",
            ["Heatmap passes the threshold but no box reaches the match IoU."],
        )
    return "LOW_MODEL_SCORE", ["Point support exists but the heatmap stays below threshold."]


def frame_thresholds(aggregate: dict[str, object]) -> dict[str, float]:
    contract = aggregate["run_contract"]
    decode = contract["decode"]
    class_values = decode.get("class_score_thresholds")
    if isinstance(class_values, list) and len(class_values) == 3:
        return {
            class_name: float(class_values[class_index])
            for class_name, class_index in CLASS_INDEX.items()
        }
    return {
        class_name: float(decode["score_threshold"])
        for class_name in CLASS_INDEX
    }


def validate_frame_manifest(
    eval_dir: Path,
    frame: str,
    run_contract: dict[str, object],
) -> None:
    expected = {
        "schema_version": 1,
        "archive": run_contract["archive"],
        "frame": frame,
        "preprocessing": run_contract["preprocessing"],
        "decode": run_contract["decode"],
        "dependencies": run_contract["dependencies"],
    }
    actual = read_json(eval_dir / frame / "pipeline_cache_manifest.json")
    if actual != expected:
        raise ValueError(f"{frame} pipeline manifest does not match aggregate run_contract")


def summarize_metrics(rows: list[dict[str, object]]) -> dict[str, object]:
    true_positives = sum(int(row["tp"]) for row in rows)
    false_positives = sum(int(row["fp"]) for row in rows)
    false_negatives = sum(int(row["fn"]) for row in rows)
    predictions = sum(int(row["predictions"]) for row in rows)
    labels = sum(int(row["labels"]) for row in rows)
    return {
        "predictions": predictions,
        "labels": labels,
        "tp": true_positives,
        "fp": false_positives,
        "fn": false_negatives,
        "precision": (
            true_positives / (true_positives + false_positives)
            if true_positives + false_positives
            else 0.0
        ),
        "recall": (
            true_positives / (true_positives + false_negatives)
            if true_positives + false_negatives
            else 0.0
        ),
    }


def numeric_summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"minimum": None, "median": None, "mean": None, "maximum": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(array.min()),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "maximum": float(array.max()),
    }


def flatten_record(record: dict[str, object]) -> dict[str, object]:
    box = record["gt_box"]
    points = record["point_counts"]
    heatmap = record["heatmap"]
    row = {
        "fn_index": record["fn_index"],
        "frame": record["frame"],
        "label_id": record["label_id"],
        "class_name": record["class_name"],
        "classification": record["classification"],
        "classification_reasons": "; ".join(record["classification_reasons"]),
        "distance_m": record["distance_m"],
        "gt_x": box["x"],
        "gt_y": box["y"],
        "gt_z": box["z"],
        "gt_length": box["length"],
        "gt_width": box["width"],
        "gt_height": box["height"],
        "gt_heading": box["heading"],
        "official_num_lidar_points": record["official_num_lidar_points"],
        "official_num_top_lidar_points": record["official_num_top_lidar_points"],
        "all_archive_points_in_box": points["all_archive_points_in_box"],
        "mirrored_heading_points_in_box": points[
            "mirrored_heading_points_in_box"
        ],
        "selected_points_in_box": points["selected_points_in_box"],
        "selected_points_in_model_range": points[
            "selected_points_in_model_range"
        ],
        "effective_model_points": points["effective_model_points"],
        "effective_top_points": points["effective_top_points"],
        "effective_return1_points": points["effective_return1_points"],
        "effective_return2_points": points["effective_return2_points"],
        "range_retention": points["range_retention"],
        "drop_nlz_retention": points["drop_nlz_retention"],
        "mean_tanh_intensity": points["mean_tanh_intensity"],
        "heatmap_center_score": heatmap["center_score"],
        "heatmap_local_max_score": heatmap["local_max_score"],
        "score_threshold": heatmap["score_threshold"],
        "baseline_best_iou": record["baseline_best_iou"],
        "baseline_best_score": record["baseline_best_score"],
        "official_geometry_best_iou": record["official_geometry_best_iou"],
        "official_geometry_best_score": record["official_geometry_best_score"],
        "official_geometry_matched": record["official_geometry_matched"],
    }
    for name in sorted(points["selected_source_effective_counts"]):
        row[f"effective_{name}"] = points["selected_source_effective_counts"][name]
    return row


def write_csv(path: Path, records: list[dict[str, object]]) -> None:
    rows = [flatten_record(record) for record in records]
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def format_metric_row(name: str, metrics: dict[str, object]) -> str:
    return (
        f"| {name} | {metrics['tp']} | {metrics['fp']} | {metrics['fn']} | "
        f"{float(metrics['precision']):.4f} | {float(metrics['recall']):.4f} |"
    )


def write_markdown_report(path: Path, report: dict[str, object]) -> None:
    baseline = report["baseline_metrics"]
    official = report["official_geometry_metrics"]
    rotation = report["rotation_point_count_audit"]
    lines = [
        "# Waymo False Negative Analysis",
        "",
        "## Metric Geometry Audit",
        "",
        "| Convention | TP | FP | FN | Precision | Recall |",
        "|---|---:|---:|---:|---:|---:|",
        format_metric_row("Current evaluator", baseline),
        format_metric_row("Official Waymo CCW", official),
        "",
        "## Label Rotation Evidence",
        "",
        f"- Waymo official point count sum: {rotation['official_point_sum']}",
        f"- CCW box counted sum: {rotation['ccw_point_sum']}",
        f"- Mirrored-heading counted sum: {rotation['mirrored_point_sum']}",
        f"- CCW mean absolute error: {float(rotation['ccw_mean_abs_error']):.3f}",
        f"- Mirrored mean absolute error: {float(rotation['mirrored_mean_abs_error']):.3f}",
        "",
        "## False Negative Classes",
        "",
        "| Classification | Count |",
        "|---|---:|",
    ]
    for name, count in report["classification_counts"].items():
        lines.append(f"| `{name}` | {count} |")
    lines.extend(
        [
            "",
            "## Per-GT Evidence",
            "",
            "| Frame | Label | Distance | Official pts | Effective pts | Heatmap | Current IoU | Official IoU | Classification |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for record in report["records"]:
        heatmap_score = record["heatmap"]["local_max_score"]
        heatmap_text = "-" if heatmap_score is None else f"{float(heatmap_score):.4f}"
        lines.append(
            f"| {record['frame']} | {str(record['label_id'])[-6:]} | "
            f"{float(record['distance_m']):.1f} | "
            f"{record['official_num_lidar_points']} | "
            f"{record['point_counts']['effective_model_points']} | {heatmap_text} | "
            f"{float(record['baseline_best_iou']):.3f} | "
            f"{float(record['official_geometry_best_iou']):.3f} | "
            f"`{record['classification']}` |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_frame(
    output_path: Path,
    frame: str,
    frame_dir: Path,
    labels: list[dict[str, object]],
    records: list[dict[str, object]],
    point_sample_limit: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Polygon

    points = np.fromfile(frame_dir / "points.bin", dtype="<f4").reshape(-1, 5)
    step = max(1, math.ceil(points.shape[0] / point_sample_limit))
    sampled = points[::step]
    colors = {
        "EVALUATION_GEOMETRY_MISMATCH": "#d62728",
        "OUT_OF_RANGE": "#9467bd",
        "LOW_POINT_COUNT": "#ff7f0e",
        "PREPROCESSING_SENSITIVE": "#8c564b",
        "BOX_REGRESSION_ERROR": "#e377c2",
        "LOW_MODEL_SCORE": "#bcbd22",
    }
    records_by_id = {str(record["label_id"]): record for record in records}

    figure, axis = plt.subplots(figsize=(12, 8))
    axis.scatter(sampled[:, 0], sampled[:, 1], s=0.25, c="#7f7f7f", alpha=0.3)
    for label in labels:
        record = records_by_id.get(str(label["id"]))
        color = colors.get(str(record["classification"]), "#bdbdbd") if record else "#bdbdbd"
        width = 2.0 if record else 0.7
        corners = np.asarray(box_corners(label["box"]), dtype=np.float32)
        axis.add_patch(
            Polygon(corners, closed=True, fill=False, edgecolor=color, linewidth=width)
        )
        if record:
            axis.text(
                float(label["box"]["x"]),
                float(label["box"]["y"]),
                f"FN{record['fn_index']}\n{record['point_counts']['effective_model_points']}pts",
                color=color,
                fontsize=7,
                ha="center",
                va="bottom",
            )
            prediction = record.get("official_geometry_prediction")
            if prediction is not None:
                prediction_corners = np.asarray(box_corners(prediction), dtype=np.float32)
                axis.add_patch(
                    Polygon(
                        prediction_corners,
                        closed=True,
                        fill=False,
                        edgecolor="#1f77b4",
                        linewidth=1.5,
                        linestyle="--",
                    )
                )

    axis.set_aspect("equal", adjustable="box")
    if sampled.size:
        axis.set_xlim(float(sampled[:, 0].min()) - 4.0, float(sampled[:, 0].max()) + 4.0)
        axis.set_ylim(float(sampled[:, 1].min()) - 4.0, float(sampled[:, 1].max()) + 4.0)
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_title(f"{frame}: false negative evidence")
    handles = [
        Line2D([0], [0], color=color, linewidth=2, label=name)
        for name, color in colors.items()
        if any(str(record["classification"]) == name for record in records)
    ]
    handles.append(
        Line2D([0], [0], color="#1f77b4", linestyle="--", label="Best prediction")
    )
    axis.legend(handles=handles, fontsize=7, loc="best")
    axis.grid(True, linewidth=0.4, alpha=0.35)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main() -> int:
    args = parse_args()
    aggregate = read_json(args.eval_dir / "aggregate_report.json")
    run_contract = aggregate.get("run_contract")
    if not isinstance(run_contract, dict):
        raise ValueError("aggregate report has no run_contract")
    available_frames = [str(frame) for frame in run_contract["frames"]]
    frames = args.frames if args.frames else available_frames
    if not frames or any(frame not in available_frames for frame in frames):
        raise ValueError("requested frames are not part of the evaluation run")
    archive_path = args.archive or Path(str(run_contract["archive"]["path"]))
    if canonical_path(archive_path) != str(run_contract["archive"]["path"]):
        raise ValueError("archive does not match aggregate run_contract")
    for frame in frames:
        validate_frame_manifest(args.eval_dir, frame, run_contract)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = args.output_dir / "figures"
    if not args.no_figures:
        figure_dir.mkdir(parents=True, exist_ok=True)

    thresholds = frame_thresholds(aggregate)
    match_iou = float(run_contract["evaluation"]["match_iou"])
    frame_reports = {
        str(row["frame"]): row
        for row in aggregate["frame_reports"]
        if str(row["frame"]) in frames
    }
    baseline_metrics = summarize_metrics(list(frame_reports.values()))
    false_negatives_by_frame = {
        frame: [
            row
            for row in aggregate["false_negatives"]
            if str(row["frame"]) == frame
        ]
        for frame in frames
    }

    records = []
    rotation_rows = []
    geometry_frame_rows = []
    next_index = 0
    with zipfile.ZipFile(archive_path) as archive:
        for frame in frames:
            frame_dir = args.eval_dir / frame
            labels = load_labels(archive, frame)
            sources = load_source_points(archive, frame)
            predictions = load_predictions(frame_dir)
            geometry = evaluate_official_geometry(predictions, labels, match_iou)
            geometry_frame_rows.append(geometry)
            matched_ids = set(str(value) for value in geometry["matched_label_ids"])
            voxel_metadata = read_json(frame_dir / "02_voxel" / "metadata.json")
            model_range = [
                float(value) for value in voxel_metadata["point_cloud_range"]
            ]
            heatmap = load_heatmap(frame_dir)
            used_ids: set[str] = set()
            frame_records = []

            for label in labels:
                counts = analyze_point_counts(
                    sources, label["box"], model_range, run_contract["preprocessing"]
                )
                rotation_rows.append(
                    {
                        "official": int(label["official_num_lidar_points"]),
                        "ccw": int(counts["all_archive_points_in_box"]),
                        "mirrored": int(counts["mirrored_heading_points_in_box"]),
                    }
                )

            for false_negative in false_negatives_by_frame[frame]:
                label = match_false_negative_label(false_negative, labels, used_ids)
                label_id = str(label["id"])
                point_counts = analyze_point_counts(
                    sources, label["box"], model_range, run_contract["preprocessing"]
                )
                heatmap_row = heatmap_evidence(
                    heatmap,
                    label["box"],
                    str(label["class_name"]),
                    voxel_metadata,
                    thresholds[str(label["class_name"])],
                    args.local_radius,
                )
                official_best = geometry["best_by_label"][label_id]
                record = {
                    "fn_index": next_index,
                    "frame": frame,
                    "label_id": label_id,
                    "class_name": label["class_name"],
                    "gt_box": label["box"],
                    "distance_m": math.hypot(
                        float(label["box"]["x"]), float(label["box"]["y"])
                    ),
                    "official_num_lidar_points": label[
                        "official_num_lidar_points"
                    ],
                    "official_num_top_lidar_points": label[
                        "official_num_top_lidar_points"
                    ],
                    "center_in_model_xy": center_in_xy_range(
                        label["box"], model_range
                    ),
                    "box_overlaps_model_z": box_overlaps_z_range(
                        label["box"], model_range
                    ),
                    "point_counts": point_counts,
                    "heatmap": heatmap_row,
                    "baseline_best_iou": float(
                        false_negative["best_prediction_iou"]
                    ),
                    "baseline_best_score": false_negative[
                        "best_prediction_score"
                    ],
                    "official_geometry_best_iou": float(official_best["iou"]),
                    "official_geometry_best_score": official_best["score"],
                    "official_geometry_matched": label_id in matched_ids,
                    "official_geometry_prediction": official_best["prediction"],
                }
                classification, reasons = classify_false_negative(
                    record, args.low_point_threshold
                )
                record["classification"] = classification
                record["classification_reasons"] = reasons
                records.append(record)
                frame_records.append(record)
                next_index += 1

            if not args.no_figures:
                plot_frame(
                    figure_dir / f"{frame}_fn_analysis.png",
                    frame,
                    frame_dir,
                    labels,
                    frame_records,
                    args.point_sample_limit,
                )

    official_metrics = summarize_metrics(geometry_frame_rows)
    classification_counts = dict(
        sorted(Counter(str(row["classification"]) for row in records).items())
    )
    official_values = np.asarray(
        [row["official"] for row in rotation_rows], dtype=np.float64
    )
    ccw_values = np.asarray([row["ccw"] for row in rotation_rows], dtype=np.float64)
    mirrored_values = np.asarray(
        [row["mirrored"] for row in rotation_rows], dtype=np.float64
    )
    rotation_audit = {
        "labels": len(rotation_rows),
        "official_point_sum": int(official_values.sum()),
        "ccw_point_sum": int(ccw_values.sum()),
        "mirrored_point_sum": int(mirrored_values.sum()),
        "ccw_mean_abs_error": float(np.abs(ccw_values - official_values).mean()),
        "mirrored_mean_abs_error": float(
            np.abs(mirrored_values - official_values).mean()
        ),
        "ccw_exact_matches": int(np.count_nonzero(ccw_values == official_values)),
        "mirrored_exact_matches": int(
            np.count_nonzero(mirrored_values == official_values)
        ),
    }
    effective_values = [
        float(row["point_counts"]["effective_model_points"]) for row in records
    ]
    report = {
        "schema_version": SCHEMA_VERSION,
        "eval_dir": canonical_path(args.eval_dir),
        "archive": canonical_path(archive_path),
        "frames": frames,
        "run_contract": run_contract,
        "low_point_threshold": args.low_point_threshold,
        "baseline_metrics": baseline_metrics,
        "official_geometry_metrics": official_metrics,
        "geometry_metric_delta": {
            key: float(official_metrics[key]) - float(baseline_metrics[key])
            for key in ("tp", "fp", "fn", "precision", "recall")
        },
        "rotation_point_count_audit": rotation_audit,
        "classification_counts": classification_counts,
        "effective_point_summary": numeric_summary(effective_values),
        "records": records,
    }
    (args.output_dir / "fn_analysis.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    write_csv(args.output_dir / "fn_analysis.csv", records)
    write_markdown_report(args.output_dir / "fn_analysis_report.md", report)
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "records"},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
