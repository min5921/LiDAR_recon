#!/usr/bin/env python3
"""Read-only inspection of committed LiDAR_recon artifacts.

The script never changes repository inputs. It may write a JSON summary and a
BEV PNG only when explicit output paths are supplied.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ARTIFACT = "[아티팩트 검산]"
FORMULA = "[수식 검산]"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def native(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def tensor_record(path: Path, dtype: np.dtype, shape: list[int]) -> dict[str, Any]:
    array = np.fromfile(path, dtype=dtype)
    expected_items = math.prod(shape)
    expected_bytes = expected_items * np.dtype(dtype).itemsize
    if array.size != expected_items:
        raise ValueError(f"{path}: expected {expected_items} items, found {array.size}")
    shaped = array.reshape(shape)
    flat = shaped.reshape(-1)
    return {
        "path": path.as_posix(),
        "dtype": np.dtype(dtype).name,
        "shape": shape,
        "actual_bytes": path.stat().st_size,
        "expected_bytes": expected_bytes,
        "size_matches_shape_dtype": path.stat().st_size == expected_bytes,
        "minimum": native(flat.min()) if flat.size else None,
        "maximum": native(flat.max()) if flat.size else None,
        "mean": native(flat.astype(np.float64).mean()) if flat.size else None,
        "first_values": native(flat[:10]),
        "verification_status": ARTIFACT,
    }


def metric_check(metrics: dict[str, Any]) -> dict[str, Any]:
    tp, fp, fn = int(metrics["tp"]), int(metrics["fp"]), int(metrics["fn"])
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision_stored": float(metrics["precision"]),
        "precision_recomputed": precision,
        "recall_stored": float(metrics["recall"]),
        "recall_recomputed": recall,
        "f1_recomputed": f1,
        "precision_matches": math.isclose(precision, float(metrics["precision"]), abs_tol=1e-12),
        "recall_matches": math.isclose(recall, float(metrics["recall"]), abs_tol=1e-12),
        "verification_status": FORMULA,
    }


def inspect(repo: Path) -> dict[str, Any]:
    sample = repo / "00_reference/sample_data/kitti/000000.bin"
    points = np.fromfile(sample, dtype=np.float32)
    if points.size % 4:
        raise ValueError(f"{sample}: float count is not divisible by four")
    points = points.reshape(-1, 4)

    voxel_dir = repo / "02_project/dump/kitti_000000"
    voxel_meta = read_json(voxel_dir / "metadata.json")
    pillars = int(voxel_meta["num_pillars"])
    max_points = int(voxel_meta["max_points_per_pillar"])
    feature_dim = int(voxel_meta["feature_dim"])

    decorated_dir = repo / "03_pillar_feature_project/dump/kitti_000000_decorated"
    decorated_meta = read_json(decorated_dir / "decorated_metadata.json")

    pfn_dir = repo / "04_pfn_project/dump/kitti_000000_pfn"
    pfn_meta = read_json(pfn_dir / "pillar_features_metadata.json")

    comparison = read_json(repo / "11_reference_comparison_project/reference_comparison_5frames.json")
    fn_analysis = read_json(repo / "12_waymo_fn_analysis_project/fn_analysis_5frames.json")
    head_validation = read_json(repo / "10_head_validation_project/head_validation_summary_5frames.json")
    operating = read_json(repo / "13_waymo_operating_point_project/operating_point_analysis_198frames.json")

    threshold_rows: list[dict[str, Any]] = []
    threshold_csv = repo / "13_waymo_operating_point_project/threshold_summary_198frames.csv"
    with threshold_csv.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            parsed = {
                "threshold": float(row["threshold"]),
                "predictions": int(row["predictions"]),
                "labels": int(row["labels"]),
                "tp": int(row["tp"]),
                "fp": int(row["fp"]),
                "fn": int(row["fn"]),
                "precision": float(row["precision"]),
                "recall": float(row["recall"]),
                "f1": float(row["f1"]),
            }
            checked = metric_check(parsed)
            parsed["formula_matches"] = (
                checked["precision_matches"]
                and checked["recall_matches"]
                and math.isclose(parsed["f1"], checked["f1_recomputed"], abs_tol=1e-12)
            )
            threshold_rows.append(parsed)

    best_f1 = max(threshold_rows, key=lambda item: (item["f1"], -item["threshold"]))
    precision_floor = float(operating["operating_points"]["precision_floor"])
    eligible = [row for row in threshold_rows if row["precision"] >= precision_floor]
    best_recall = max(eligible, key=lambda item: (item["recall"], -item["threshold"]))

    result = {
        "schema_version": 1,
        "repository_root": str(repo.resolve()),
        "kitti_sample": {
            "path": sample.relative_to(repo).as_posix(),
            "dtype": "float32",
            "shape": list(points.shape),
            "bytes": sample.stat().st_size,
            "first_point": native(points[0]),
            "minimum_per_feature": native(points.min(axis=0)),
            "maximum_per_feature": native(points.max(axis=0)),
            "mean_per_feature": native(points.astype(np.float64).mean(axis=0)),
            "finite": bool(np.isfinite(points).all()),
            "verification_status": ARTIFACT,
        },
        "committed_stage_artifacts": {
            "voxelization": {
                "metadata": voxel_meta,
                "pillars": tensor_record(
                    voxel_dir / "pillars.bin",
                    np.float32,
                    [pillars, max_points, feature_dim],
                ),
                "coordinates": tensor_record(
                    voxel_dir / "coordinates.bin", np.int32, [pillars, 4]
                ),
                "num_points": tensor_record(
                    voxel_dir / "num_points.bin", np.int32, [pillars]
                ),
            },
            "pillar_decoration": {
                "metadata": decorated_meta,
                "decorated_pillars": tensor_record(
                    decorated_dir / "decorated_pillars.bin",
                    np.float32,
                    [
                        int(decorated_meta["num_pillars"]),
                        int(decorated_meta["max_points_per_pillar"]),
                        int(decorated_meta["decorated_feature_dim"]),
                    ],
                ),
            },
            "pfn_dummy": {
                "metadata": pfn_meta,
                "pillar_features": tensor_record(
                    pfn_dir / "pillar_features.bin",
                    np.float32,
                    [int(pfn_meta["num_pillars"]), int(pfn_meta["out_channels"])],
                ),
                "note": "커밋된 파일은 deterministic dummy PFN 결과이며 checkpoint PFN 결과가 아니다.",
            },
        },
        "reference_comparison_5frames": {
            "raw": metric_check(comparison["raw_metrics"]),
            "tanh_reference": metric_check(comparison["reference_metrics"]),
            "stage_validation": comparison["stage_validation"],
            "source": "11_reference_comparison_project/reference_comparison_5frames.json",
            "verification_status": ARTIFACT,
        },
        "head_validation_5frames": {
            "frames": int(head_validation["frames"]),
            "labels": int(head_validation["labels"]),
            "outcome_counts": head_validation["outcome_counts"],
            "source": "10_head_validation_project/head_validation_summary_5frames.json",
            "verification_status": ARTIFACT,
        },
        "false_negative_analysis_5frames": {
            "baseline": metric_check(fn_analysis["baseline_metrics"]),
            "official_geometry": metric_check(fn_analysis["official_geometry_metrics"]),
            "classification_counts": fn_analysis["classification_counts"],
            "effective_point_summary": fn_analysis["effective_point_summary"],
            "source": "12_waymo_fn_analysis_project/fn_analysis_5frames.json",
            "verification_status": ARTIFACT,
        },
        "operating_point_198frames": {
            "frame_count": int(operating["frame_count"]),
            "label_count": int(operating["label_count"]),
            "precision_floor": precision_floor,
            "best_f1_recomputed": best_f1,
            "best_recall_at_precision_floor_recomputed": best_recall,
            "selected_threshold_stored": float(operating["operating_points"]["selected_threshold"]),
            "all_threshold_formulas_match": all(row["formula_matches"] for row in threshold_rows),
            "threshold_rows": threshold_rows,
            "source": "13_waymo_operating_point_project/operating_point_analysis_198frames.json and threshold_summary_198frames.csv",
            "verification_status": ARTIFACT,
        },
    }
    return result


def make_bev(repo: Path, output: Path, size: int = 1000) -> None:
    from PIL import Image, ImageDraw, ImageFont

    points = np.fromfile(
        repo / "00_reference/sample_data/kitti/000000.bin", dtype=np.float32
    ).reshape(-1, 4)
    x, y, intensity = points[:, 0], points[:, 1], points[:, 3]
    extent = 75.0
    mask = (np.abs(x) <= extent) & (np.abs(y) <= extent) & np.isfinite(points).all(axis=1)
    x, y, intensity = x[mask], y[mask], intensity[mask]

    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    canvas[:] = (5, 18, 25)
    px = np.clip(((x + extent) / (2 * extent) * (size - 1)).astype(np.int32), 0, size - 1)
    py = np.clip(((extent - y) / (2 * extent) * (size - 1)).astype(np.int32), 0, size - 1)
    scaled = np.clip(intensity / max(float(np.percentile(intensity, 99)), 1e-6), 0, 1)
    colors = np.stack(
        [20 + 25 * scaled, 150 + 105 * scaled, 170 + 85 * scaled], axis=1
    ).astype(np.uint8)
    canvas[py, px] = np.maximum(canvas[py, px], colors)

    image = Image.fromarray(canvas, mode="RGB")
    draw = ImageDraw.Draw(image)
    center = size // 2
    grid_color = (31, 78, 88)
    for meters in (-50, -25, 0, 25, 50):
        pos = int((meters + extent) / (2 * extent) * (size - 1))
        draw.line((pos, 0, pos, size), fill=grid_color, width=1)
        draw.line((0, size - 1 - pos, size, size - 1 - pos), fill=grid_color, width=1)
    draw.line((center, 0, center, size), fill=(73, 220, 220), width=2)
    draw.line((0, center, size, center), fill=(73, 220, 220), width=2)
    draw.ellipse((center - 5, center - 5, center + 5, center + 5), fill=(245, 214, 84))
    font = ImageFont.load_default()
    draw.rectangle((16, 16, 480, 72), fill=(5, 18, 25))
    draw.text((28, 26), f"KITTI 000000 | {len(points):,} points | XY BEV", fill=(220, 245, 245), font=font)
    draw.text((28, 47), "cyan: LiDAR returns   yellow: sensor origin", fill=(97, 239, 186), font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--bev-out", type=Path)
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    result = inspect(repo)
    encoded = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(encoded, encoding="utf-8")
    if args.bev_out:
        make_bev(repo, args.bev_out.resolve())
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
