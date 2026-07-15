#!/usr/bin/env python3
"""Analyze CenterPoint score thresholds and GT recall strata.

The input evaluation must be decoded at the lowest threshold in the requested
sweep. NMS processes boxes in descending score order, so detections from that
run can be filtered for every higher threshold without rerunning CUDA inference.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
import zipfile
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FN_ANALYSIS_PATH = (
    PROJECT_ROOT
    / "12_waymo_fn_analysis_project"
    / "tools"
    / "analyze_waymo_false_negatives.py"
)
DEFAULT_THRESHOLDS = [
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.50,
    0.60,
    0.70,
]
DEFAULT_DISTANCE_EDGES = [0.0, 30.0, 50.0, 75.0]
DEFAULT_POINT_EDGES = [0.0, 5.0, 10.0, 20.0, 50.0]


def load_fn_analysis_module():
    spec = importlib.util.spec_from_file_location(
        "waymo_fn_analysis_for_operating_points", FN_ANALYSIS_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {FN_ANALYSIS_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


FN_ANALYSIS = load_fn_analysis_module()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--thresholds", nargs="+", type=float, default=DEFAULT_THRESHOLDS
    )
    parser.add_argument("--match-iou", type=float, default=None)
    parser.add_argument("--precision-floor", type=float, default=0.8)
    parser.add_argument(
        "--distance-edges",
        nargs="+",
        type=float,
        default=DEFAULT_DISTANCE_EDGES,
    )
    parser.add_argument(
        "--point-edges", nargs="+", type=float, default=DEFAULT_POINT_EDGES
    )
    parser.add_argument("--no-figures", action="store_true")
    return parser.parse_args()


def normalize_thresholds(values: list[float]) -> list[float]:
    thresholds = sorted(set(float(value) for value in values))
    if not thresholds:
        raise ValueError("at least one threshold is required")
    if thresholds[0] < 0.0 or thresholds[-1] > 1.0:
        raise ValueError("thresholds must be in [0, 1]")
    return thresholds


def make_bins(
    edges: list[float], kind: str
) -> list[dict[str, float | str | None]]:
    values = [float(value) for value in edges]
    if not values or values[0] != 0.0:
        raise ValueError(f"{kind} edges must start at 0")
    if any(not math.isfinite(value) for value in values):
        raise ValueError(f"{kind} edges must be finite")
    if any(second <= first for first, second in zip(values, values[1:])):
        raise ValueError(f"{kind} edges must be strictly increasing")

    bins = []
    for index, lower in enumerate(values):
        upper = values[index + 1] if index + 1 < len(values) else None
        if kind == "distance":
            label = (
                f"{lower:g}-{upper:g} m"
                if upper is not None
                else f"{lower:g}+ m"
            )
        elif upper is not None:
            label = f"{int(lower)}-{int(upper) - 1} points"
        else:
            label = f"{int(lower)}+ points"
        bins.append({"lower": lower, "upper": upper, "label": label})
    return bins


def bin_label(
    value: float, bins: list[dict[str, float | str | None]]
) -> str:
    for item in bins:
        upper = item["upper"]
        if float(item["lower"]) <= value and (
            upper is None or value < float(upper)
        ):
            return str(item["label"])
    raise ValueError(f"value {value} does not fit configured bins")


def safe_divide(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def metric_block(
    predictions: int, labels: int, true_positives: int
) -> dict[str, object]:
    false_positives = predictions - true_positives
    false_negatives = labels - true_positives
    precision = safe_divide(true_positives, predictions)
    recall = safe_divide(true_positives, labels)
    return {
        "predictions": predictions,
        "labels": labels,
        "tp": true_positives,
        "fp": false_positives,
        "fn": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": safe_divide(2.0 * precision * recall, precision + recall),
    }


def recall_block(labels: int, true_positives: int) -> dict[str, object]:
    return {
        "labels": labels,
        "tp": true_positives,
        "fn": labels - true_positives,
        "recall": safe_divide(true_positives, labels),
    }


def label_key(frame: str, label_id: object) -> str:
    return f"{frame}:{label_id}"


def prepare_frame_data(
    eval_dir: Path,
    archive_path: Path,
    frames: list[str],
    run_contract: dict[str, object],
) -> list[dict[str, object]]:
    prepared = []
    with zipfile.ZipFile(archive_path) as archive:
        for frame in frames:
            FN_ANALYSIS.validate_frame_manifest(eval_dir, frame, run_contract)
            labels = FN_ANALYSIS.load_labels(archive, frame)
            predictions = FN_ANALYSIS.load_predictions(eval_dir / frame)
            for label in labels:
                box = label["box"]
                label["frame"] = frame
                label["distance_m"] = math.hypot(
                    float(box["x"]), float(box["y"])
                )
                label["point_count"] = int(label["official_num_lidar_points"])
            prepared.append(
                {"frame": frame, "labels": labels, "predictions": predictions}
            )
    return prepared


def summarize_strata(
    label_rows: list[dict[str, object]],
    field: str,
    configured_labels: list[str],
) -> dict[str, dict[str, object]]:
    result = {}
    for name in configured_labels:
        rows = [row for row in label_rows if row[field] == name]
        result[name] = recall_block(
            len(rows), sum(bool(row["matched"]) for row in rows)
        )
    return result


def evaluate_threshold(
    frame_data: list[dict[str, object]],
    threshold: float,
    match_iou: float,
    distance_bins: list[dict[str, float | str | None]],
    point_bins: list[dict[str, float | str | None]],
) -> dict[str, object]:
    prediction_count = 0
    label_count = 0
    true_positives = 0
    prediction_classes: Counter[str] = Counter()
    label_classes: Counter[str] = Counter()
    matched_classes: Counter[str] = Counter()
    label_rows = []
    frame_rows = []

    for frame_item in frame_data:
        frame = str(frame_item["frame"])
        labels = list(frame_item["labels"])
        predictions = [
            prediction
            for prediction in frame_item["predictions"]
            if float(prediction["score"]) + 1.0e-12 >= threshold
        ]
        evaluated = FN_ANALYSIS.evaluate_official_geometry(
            predictions, labels, match_iou
        )
        matched_ids = set(str(value) for value in evaluated["matched_label_ids"])
        label_by_id = {str(label["id"]): label for label in labels}

        prediction_count += len(predictions)
        label_count += len(labels)
        true_positives += int(evaluated["tp"])
        prediction_classes.update(
            str(prediction["class_name"]) for prediction in predictions
        )
        label_classes.update(str(label["class_name"]) for label in labels)
        for matched_id in matched_ids:
            matched_classes[str(label_by_id[matched_id]["class_name"])] += 1

        for label in labels:
            distance = float(label["distance_m"])
            points = int(label["point_count"])
            label_rows.append(
                {
                    "key": label_key(frame, label["id"]),
                    "class_name": str(label["class_name"]),
                    "distance_bin": bin_label(distance, distance_bins),
                    "point_bin": bin_label(float(points), point_bins),
                    "matched": str(label["id"]) in matched_ids,
                }
            )

        frame_metric = metric_block(
            len(predictions), len(labels), int(evaluated["tp"])
        )
        frame_rows.append({"frame": frame, **frame_metric})

    class_names = sorted(set(prediction_classes) | set(label_classes))
    class_metrics = {
        name: metric_block(
            prediction_classes[name], label_classes[name], matched_classes[name]
        )
        for name in class_names
    }
    distance_labels = [str(item["label"]) for item in distance_bins]
    point_labels = [str(item["label"]) for item in point_bins]

    matrix = {}
    for distance_name in distance_labels:
        matrix[distance_name] = {}
        for point_name in point_labels:
            rows = [
                row
                for row in label_rows
                if row["distance_bin"] == distance_name
                and row["point_bin"] == point_name
            ]
            matrix[distance_name][point_name] = recall_block(
                len(rows), sum(bool(row["matched"]) for row in rows)
            )

    return {
        "threshold": threshold,
        "overall": metric_block(prediction_count, label_count, true_positives),
        "classes": class_metrics,
        "distance_bins": summarize_strata(
            label_rows, "distance_bin", distance_labels
        ),
        "point_bins": summarize_strata(label_rows, "point_bin", point_labels),
        "distance_point_matrix": matrix,
        "frames": frame_rows,
    }


def validate_monotonic_results(results: list[dict[str, object]]) -> None:
    for previous, current in zip(results, results[1:]):
        previous_metrics = previous["overall"]
        current_metrics = current["overall"]
        if int(current_metrics["predictions"]) > int(previous_metrics["predictions"]):
            raise ValueError("prediction count increased at a higher threshold")
        if int(current_metrics["tp"]) > int(previous_metrics["tp"]):
            raise ValueError("true-positive count increased at a higher threshold")


def select_operating_points(
    results: list[dict[str, object]], precision_floor: float
) -> dict[str, object]:
    if not 0.0 <= precision_floor <= 1.0:
        raise ValueError("precision floor must be in [0, 1]")

    best_f1 = max(
        results,
        key=lambda row: (
            float(row["overall"]["f1"]),
            float(row["overall"]["precision"]),
            float(row["threshold"]),
        ),
    )
    candidates = [
        row
        for row in results
        if float(row["overall"]["precision"]) >= precision_floor
    ]
    constrained = (
        max(
            candidates,
            key=lambda row: (
                float(row["overall"]["recall"]),
                float(row["overall"]["precision"]),
                float(row["threshold"]),
            ),
        )
        if candidates
        else None
    )
    selected = constrained if constrained is not None else best_f1
    return {
        "precision_floor": precision_floor,
        "best_f1": {
            "threshold": best_f1["threshold"],
            **best_f1["overall"],
        },
        "best_recall_at_precision_floor": (
            {"threshold": constrained["threshold"], **constrained["overall"]}
            if constrained is not None
            else None
        ),
        "selected_threshold": selected["threshold"],
    }


def write_threshold_csv(path: Path, results: list[dict[str, object]]) -> None:
    fields = [
        "threshold",
        "predictions",
        "labels",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow({"threshold": result["threshold"], **result["overall"]})


def write_stratified_csv(path: Path, results: list[dict[str, object]]) -> None:
    fields = [
        "threshold",
        "scope",
        "bin",
        "predictions",
        "labels",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            threshold = result["threshold"]
            for name, metrics in result["classes"].items():
                writer.writerow(
                    {
                        "threshold": threshold,
                        "scope": "class",
                        "bin": name,
                        **metrics,
                    }
                )
            for scope in ["distance_bins", "point_bins"]:
                for name, metrics in result[scope].items():
                    writer.writerow(
                        {
                            "threshold": threshold,
                            "scope": scope,
                            "bin": name,
                            "predictions": "",
                            "fp": "",
                            "precision": "",
                            "f1": "",
                            **metrics,
                        }
                    )


def selected_result(
    results: list[dict[str, object]], selected_threshold: float
) -> dict[str, object]:
    return next(
        row
        for row in results
        if math.isclose(
            float(row["threshold"]), selected_threshold, abs_tol=1.0e-12
        )
    )


def write_recall_table(
    lines: list[str], title: str, values: dict[str, dict[str, object]]
) -> None:
    lines.extend(
        [
            f"## {title}",
            "",
            "| Bin | GT | TP | FN | Recall |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, metrics in values.items():
        lines.append(
            f"| {name} | {metrics['labels']} | {metrics['tp']} | "
            f"{metrics['fn']} | {float(metrics['recall']):.4f} |"
        )
    lines.append("")


def write_markdown_report(path: Path, report: dict[str, object]) -> None:
    lines = [
        "# Waymo Operating Point Study",
        "",
        f"- Frames: {report['frame_count']}",
        f"- Labels: {report['label_count']}",
        f"- Source decode threshold: {report['source_score_threshold']:.3f}",
        f"- Match IoU: {report['match_iou']:.2f}",
        "- Point bins use Waymo official num_lidar_points_in_box.",
        "",
        "## Threshold Sweep",
        "",
        "| Threshold | Pred | TP | FP | FN | Precision | Recall | F1 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in report["results"]:
        metrics = result["overall"]
        lines.append(
            f"| {float(result['threshold']):.2f} | {metrics['predictions']} | "
            f"{metrics['tp']} | {metrics['fp']} | {metrics['fn']} | "
            f"{float(metrics['precision']):.4f} | "
            f"{float(metrics['recall']):.4f} | {float(metrics['f1']):.4f} |"
        )

    operating = report["operating_points"]
    best_f1 = operating["best_f1"]
    constrained = operating["best_recall_at_precision_floor"]
    lines.extend(
        [
            "",
            "## Selected Operating Points",
            "",
            f"- Best F1 threshold: {float(best_f1['threshold']):.2f} "
            f"(F1 {float(best_f1['f1']):.4f})",
        ]
    )
    if constrained is not None:
        lines.append(
            f"- Best recall with precision >= {operating['precision_floor']:.2f}: "
            f"{float(constrained['threshold']):.2f} "
            f"(precision {float(constrained['precision']):.4f}, "
            f"recall {float(constrained['recall']):.4f})"
        )
    else:
        lines.append("- No threshold satisfied the configured precision floor.")
    lines.append("")

    selected = selected_result(report["results"], operating["selected_threshold"])
    lines.extend(
        [
            f"## Class Metrics at Threshold {float(selected['threshold']):.2f}",
            "",
            "| Class | Pred | GT | TP | FP | FN | Precision | Recall |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, metrics in selected["classes"].items():
        lines.append(
            f"| {name} | {metrics['predictions']} | {metrics['labels']} | "
            f"{metrics['tp']} | {metrics['fp']} | {metrics['fn']} | "
            f"{float(metrics['precision']):.4f} | "
            f"{float(metrics['recall']):.4f} |"
        )
    lines.append("")
    write_recall_table(
        lines,
        f"Distance Recall at Threshold {float(selected['threshold']):.2f}",
        selected["distance_bins"],
    )
    write_recall_table(
        lines,
        f"Point-count Recall at Threshold {float(selected['threshold']):.2f}",
        selected["point_bins"],
    )
    lines.extend(
        [
            "## Reuse Contract",
            "",
            "The source run is decoded at the lowest sweep threshold. Rotated NMS "
            "visits boxes in descending score order, so lower-score boxes cannot "
            "suppress a higher-score box. Filtering that output therefore reproduces "
            "all higher thresholds under the same NMS configuration.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_results(path: Path, report: dict[str, object]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = report["results"]
    thresholds = [float(row["threshold"]) for row in results]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)

    axes[0].plot(
        thresholds,
        [float(row["overall"]["precision"]) for row in results],
        marker="o",
        label="Precision",
        color="#1f77b4",
    )
    axes[0].plot(
        thresholds,
        [float(row["overall"]["recall"]) for row in results],
        marker="s",
        label="Recall",
        color="#d62728",
    )
    axes[0].plot(
        thresholds,
        [float(row["overall"]["f1"]) for row in results],
        marker="^",
        label="F1",
        color="#2ca02c",
    )
    axes[0].set_title("Overall operating point")

    distance_names = list(results[0]["distance_bins"])
    distance_colors = ["#9467bd", "#ff7f0e", "#17becf", "#8c564b", "#7f7f7f"]
    for index, name in enumerate(distance_names):
        axes[1].plot(
            thresholds,
            [float(row["distance_bins"][name]["recall"]) for row in results],
            marker="o",
            label=name,
            color=distance_colors[index % len(distance_colors)],
        )
    axes[1].set_title("Recall by distance")

    point_names = list(results[0]["point_bins"])
    point_colors = ["#e377c2", "#bcbd22", "#1f77b4", "#d62728", "#2ca02c"]
    for index, name in enumerate(point_names):
        axes[2].plot(
            thresholds,
            [float(row["point_bins"][name]["recall"]) for row in results],
            marker="o",
            label=name,
            color=point_colors[index % len(point_colors)],
        )
    axes[2].set_title("Recall by GT point count")

    for axis in axes:
        axis.set_xlabel("Score threshold")
        axis.set_ylabel("Metric")
        axis.set_ylim(0.0, 1.03)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle(f"Waymo threshold study ({report['frame_count']} frames)")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    thresholds = normalize_thresholds(args.thresholds)
    distance_bins = make_bins(args.distance_edges, "distance")
    point_bins = make_bins(args.point_edges, "points")

    aggregate = FN_ANALYSIS.read_json(args.eval_dir / "aggregate_report.json")
    run_contract = aggregate["run_contract"]
    decode_contract = run_contract["decode"]
    if decode_contract.get("class_score_thresholds") is not None:
        raise ValueError("global threshold study requires global source thresholds")
    source_threshold = float(decode_contract["score_threshold"])
    if thresholds[0] + 1.0e-12 < source_threshold:
        raise ValueError(
            f"source threshold {source_threshold} is above requested {thresholds[0]}"
        )

    match_iou = (
        float(args.match_iou)
        if args.match_iou is not None
        else float(run_contract["evaluation"]["match_iou"])
    )
    frames = [str(frame) for frame in run_contract["frames"]]
    archive_path = Path(str(run_contract["archive"]["path"]))
    frame_data = prepare_frame_data(
        args.eval_dir, archive_path, frames, run_contract
    )

    results = [
        evaluate_threshold(
            frame_data, threshold, match_iou, distance_bins, point_bins
        )
        for threshold in thresholds
    ]
    validate_monotonic_results(results)
    operating_points = select_operating_points(results, args.precision_floor)
    label_count = sum(len(item["labels"]) for item in frame_data)

    report = {
        "schema_version": 1,
        "eval_dir": str(args.eval_dir.resolve()),
        "archive": str(archive_path),
        "frame_count": len(frames),
        "frames": frames,
        "label_count": label_count,
        "source_score_threshold": source_threshold,
        "source_nms_iou": float(decode_contract["nms_iou"]),
        "source_nms_convention": str(decode_contract["nms_convention"]),
        "match_iou": match_iou,
        "distance_bins": distance_bins,
        "point_bins": point_bins,
        "operating_points": operating_points,
        "run_contract": run_contract,
        "results": results,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "operating_point_analysis.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    write_threshold_csv(args.output_dir / "threshold_summary.csv", results)
    write_stratified_csv(args.output_dir / "stratified_recall.csv", results)
    write_markdown_report(args.output_dir / "operating_point_report.md", report)
    if not args.no_figures:
        plot_results(args.output_dir / "operating_point_study.png", report)

    summary = {
        "frames": len(frames),
        "labels": label_count,
        "source_score_threshold": source_threshold,
        "best_f1": operating_points["best_f1"],
        "best_recall_at_precision_floor": operating_points[
            "best_recall_at_precision_floor"
        ],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
