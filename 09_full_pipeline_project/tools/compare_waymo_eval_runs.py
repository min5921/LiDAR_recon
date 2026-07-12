#!/usr/bin/env python3
"""Compare multiple Waymo evaluation output directories.

Each input directory is expected to contain:
  - aggregate_report.json
  - frame_*/export_summary.json

The output is intentionally compact so preprocessing changes can be checked
without opening every per-frame JSON file by hand.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "runs",
        nargs="+",
        type=Path,
        help="Evaluation output directories to compare.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path for a machine-readable comparison JSON.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_sources(frame_summary: dict) -> dict[str, int]:
    counts: dict[str, int] = {}
    for source in frame_summary.get("sources", []):
        entry = str(source["entry"])
        lidar_return = entry.rsplit("/", 1)[-1].replace(".bin", "")
        counts[lidar_return] = counts.get(lidar_return, 0) + int(
            source["points_after_filter"]
        )
    return dict(sorted(counts.items()))


def summarize_run(run_dir: Path) -> dict:
    aggregate = read_json(run_dir / "aggregate_report.json")
    frame_exports = []
    total_points = 0
    before_filter = 0
    after_filter = 0
    source_counts: dict[str, int] = {}

    for frame_dir in sorted(run_dir.glob("frame_*")):
        export_path = frame_dir / "export_summary.json"
        if not export_path.exists():
            continue
        export = read_json(export_path)
        frame_points = int(export.get("num_points", 0))
        total_points += frame_points
        for source in export.get("sources", []):
            before_filter += int(source.get("points_before_filter", 0))
            after_filter += int(source.get("points_after_filter", 0))
        for name, count in summarize_sources(export).items():
            source_counts[name] = source_counts.get(name, 0) + count
        frame_exports.append(
            {
                "frame": export.get("frame", frame_dir.name),
                "num_points": frame_points,
                "min": export.get("min"),
                "max": export.get("max"),
                "mean": export.get("mean"),
            }
        )

    return {
        "run_dir": str(run_dir),
        "frames": aggregate.get("frames", []),
        "nms_iou": aggregate.get("nms_iou"),
        "score_threshold": aggregate.get("score_threshold"),
        "nms_convention": aggregate.get("nms_convention"),
        "class_score_thresholds": aggregate.get("class_score_thresholds"),
        "total_predictions": aggregate.get("total_predictions"),
        "total_labels": aggregate.get("total_labels"),
        "tp": aggregate.get("tp"),
        "fp": aggregate.get("fp"),
        "fn": aggregate.get("fn"),
        "precision": aggregate.get("precision"),
        "recall": aggregate.get("recall"),
        "total_points": total_points,
        "points_before_filter": before_filter,
        "points_after_filter": after_filter,
        "points_dropped_by_filter": before_filter - after_filter,
        "source_counts": dict(sorted(source_counts.items())),
        "frame_exports": frame_exports,
    }


def print_table(rows: list[dict]) -> None:
    header = (
        "run",
        "points",
        "dropped",
        "pred",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
    )
    print(",".join(header))
    for row in rows:
        print(
            ",".join(
                [
                    Path(row["run_dir"]).name,
                    str(row["total_points"]),
                    str(row["points_dropped_by_filter"]),
                    str(row["total_predictions"]),
                    str(row["tp"]),
                    str(row["fp"]),
                    str(row["fn"]),
                    f"{float(row['precision']):.6f}",
                    f"{float(row['recall']):.6f}",
                ]
            )
        )


def main() -> int:
    args = parse_args()
    rows = [summarize_run(run) for run in args.runs]
    print_table(rows)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
