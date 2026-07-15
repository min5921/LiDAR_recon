#!/usr/bin/env python3
"""Combine preprocessing, PFN, Scatter, RPN, Head, and metric comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


PREPROCESSING_EQUALITY_KEYS = (
    "xyz_equal",
    "elongation_equal",
    "reference_intensity_matches_tanh",
    "coordinates_equal",
    "num_points_equal",
)
RUN_CONTRACT_SCHEMA_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-eval-dir", required=True, type=Path)
    parser.add_argument("--reference-eval-dir", required=True, type=Path)
    parser.add_argument("--weights-root", required=True, type=Path)
    parser.add_argument("--raw-head-summary", required=True, type=Path)
    parser.add_argument("--reference-head-summary", required=True, type=Path)
    parser.add_argument("--rpn-probe-validation", required=True, type=Path)
    parser.add_argument("--head-reference-validation", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_path(path: Path) -> str:
    return str(path.resolve())


def path_is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def require_mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} is missing; rerun the multi-frame evaluator")
    return value


def validate_aggregate_contract(
    metrics: dict[str, object], contract: dict[str, object], label: str
) -> list[str]:
    mismatches = []
    archive = require_mapping(contract.get("archive"), f"{label} archive contract")
    decode = require_mapping(contract.get("decode"), f"{label} decode contract")
    evaluation = require_mapping(
        contract.get("evaluation"), f"{label} evaluation contract"
    )
    if list(metrics.get("frames", [])) != list(contract.get("frames", [])):
        mismatches.append(f"{label} aggregate frames do not match its run contract")
    metric_archive = metrics.get("archive")
    if not isinstance(metric_archive, str) or canonical_path(Path(metric_archive)) != archive.get(
        "path"
    ):
        mismatches.append(f"{label} aggregate archive does not match its run contract")
    for metric_key, contract_key in (
        ("nms_iou", "nms_iou"),
        ("score_threshold", "score_threshold"),
        ("nms_convention", "nms_convention"),
        ("class_score_thresholds", "class_score_thresholds"),
    ):
        if metrics.get(metric_key) != decode.get(contract_key):
            mismatches.append(
                f"{label} aggregate {metric_key} does not match its run contract"
            )
    if metrics.get("match_iou") != evaluation.get("match_iou"):
        mismatches.append(f"{label} aggregate match_iou does not match its run contract")
    return mismatches


def validate_experiment_contracts(
    raw_metrics: dict[str, object],
    reference_metrics: dict[str, object],
    weights_root: Path,
) -> dict[str, object]:
    raw = require_mapping(raw_metrics.get("run_contract"), "raw run_contract")
    reference = require_mapping(
        reference_metrics.get("run_contract"), "reference run_contract"
    )
    mismatches = []
    mismatches.extend(validate_aggregate_contract(raw_metrics, raw, "raw"))
    mismatches.extend(
        validate_aggregate_contract(reference_metrics, reference, "reference")
    )
    if raw.get("schema_version") != RUN_CONTRACT_SCHEMA_VERSION:
        mismatches.append("raw run contract uses an unsupported schema version")
    if reference.get("schema_version") != RUN_CONTRACT_SCHEMA_VERSION:
        mismatches.append("reference run contract uses an unsupported schema version")

    raw_preprocessing = require_mapping(
        raw.get("preprocessing"), "raw preprocessing contract"
    )
    reference_preprocessing = require_mapping(
        reference.get("preprocessing"), "reference preprocessing contract"
    )
    if raw_preprocessing.get("intensity_transform") != "none":
        mismatches.append("raw intensity_transform must be 'none'")
    if reference_preprocessing.get("intensity_transform") != "tanh":
        mismatches.append("reference intensity_transform must be 'tanh'")
    raw_preprocessing_without_intensity = {
        key: value
        for key, value in raw_preprocessing.items()
        if key != "intensity_transform"
    }
    reference_preprocessing_without_intensity = {
        key: value
        for key, value in reference_preprocessing.items()
        if key != "intensity_transform"
    }
    if raw_preprocessing_without_intensity != reference_preprocessing_without_intensity:
        mismatches.append("preprocessing settings differ beyond intensity_transform")

    for key in ("schema_version", "archive", "frames", "decode", "evaluation", "dependencies"):
        if raw.get(key) != reference.get(key):
            mismatches.append(f"raw/reference run contracts differ in {key}")

    expected_weights_root = canonical_path(weights_root)
    for label, contract in (("raw", raw), ("reference", reference)):
        dependencies = require_mapping(
            contract.get("dependencies"), f"{label} dependency contract"
        )
        if dependencies.get("weights_root") != expected_weights_root:
            mismatches.append(
                f"{label} run used a different weights_root than this comparison"
            )

    if mismatches:
        details = "\n".join(f"- {message}" for message in mismatches)
        raise ValueError(f"experiment contract mismatch:\n{details}")
    dependencies = require_mapping(raw.get("dependencies"), "dependency contract")
    return {
        "passed": True,
        "raw_intensity_transform": "none",
        "reference_intensity_transform": "tanh",
        "archive": require_mapping(raw.get("archive"), "archive contract")["path"],
        "frames": list(raw["frames"]),
        "project_files_sha256": require_mapping(
            dependencies.get("project_files"), "project file signature"
        )["sha256"],
        "weight_files_sha256": require_mapping(
            dependencies.get("weight_files"), "weight file signature"
        )["sha256"],
    }


def head_summary_frames(summary: dict[str, object]) -> list[str]:
    rows = summary.get("frame_summaries")
    if not isinstance(rows, list):
        return []
    return [str(row.get("frame")) for row in rows if isinstance(row, dict)]


def validate_source_provenance(
    args: argparse.Namespace,
    frames: list[str],
    archive_path: str,
    raw_contract: dict[str, object],
    reference_contract: dict[str, object],
    rpn: dict[str, object],
    head: dict[str, object],
    raw_head: dict[str, object],
    reference_head: dict[str, object],
) -> dict[str, object]:
    mismatches = []
    expected_raw_dir = canonical_path(args.raw_eval_dir)
    expected_reference_dir = canonical_path(args.reference_eval_dir)
    expected_rpn_weights = canonical_path(args.weights_root / "06_rpn")
    expected_head_weights = canonical_path(args.weights_root / "07_head")

    for label, summary, expected_dir, expected_contract in (
        ("raw head summary", raw_head, expected_raw_dir, raw_contract),
        (
            "reference head summary",
            reference_head,
            expected_reference_dir,
            reference_contract,
        ),
    ):
        eval_dir = summary.get("eval_dir")
        if not isinstance(eval_dir, str) or canonical_path(Path(eval_dir)) != expected_dir:
            mismatches.append(f"{label} points to a different eval_dir")
        summary_archive = summary.get("archive")
        if not isinstance(summary_archive, str) or canonical_path(
            Path(summary_archive)
        ) != archive_path:
            mismatches.append(f"{label} points to a different archive")
        if head_summary_frames(summary) != frames:
            mismatches.append(f"{label} frame list differs from the comparison")
        if summary.get("run_contract") != expected_contract:
            mismatches.append(f"{label} run contract is stale or unrelated")

    rpn_weight_dir = rpn.get("weight_dir")
    if not isinstance(rpn_weight_dir, str) or canonical_path(
        Path(rpn_weight_dir)
    ) != expected_rpn_weights:
        mismatches.append("RPN probe validation used a different weight directory")
    probe_json = rpn.get("probe_json")
    expected_probe_root = args.reference_eval_dir / frames[0]
    if not isinstance(probe_json, str) or not path_is_within(
        Path(probe_json), expected_probe_root
    ):
        mismatches.append("RPN probe JSON is not from the reference evaluation frame")
    elif not Path(probe_json).is_file():
        mismatches.append("RPN probe JSON no longer exists")

    expected_manifest_path = expected_probe_root / "pipeline_cache_manifest.json"
    expected_manifest = {
        "schema_version": 1,
        "archive": reference_contract.get("archive"),
        "frame": frames[0],
        "preprocessing": reference_contract.get("preprocessing"),
        "decode": reference_contract.get("decode"),
        "dependencies": reference_contract.get("dependencies"),
    }
    manifest_path = rpn.get("pipeline_manifest_path")
    if not isinstance(manifest_path, str) or canonical_path(
        Path(manifest_path)
    ) != canonical_path(expected_manifest_path):
        mismatches.append("RPN validation points to a different pipeline manifest")
    if rpn.get("pipeline_cache_manifest") != expected_manifest:
        mismatches.append("RPN validation pipeline manifest is stale or unrelated")
    if not expected_manifest_path.is_file():
        mismatches.append("reference pipeline manifest no longer exists")
    elif read_json(expected_manifest_path) != expected_manifest:
        mismatches.append("reference pipeline manifest differs from its run contract")

    head_eval_dir = head.get("eval_dir")
    if not isinstance(head_eval_dir, str) or canonical_path(
        Path(head_eval_dir)
    ) != expected_reference_dir:
        mismatches.append("Head reference validation points to a different eval_dir")
    head_weight_dir = head.get("weight_dir")
    if not isinstance(head_weight_dir, str) or canonical_path(
        Path(head_weight_dir)
    ) != expected_head_weights:
        mismatches.append("Head reference validation used a different weight directory")
    if list(head.get("frame_names", [])) != frames:
        mismatches.append("Head reference validation frame list differs from the comparison")
    if head.get("run_contract") != reference_contract:
        mismatches.append("Head reference validation run contract is stale or unrelated")

    if mismatches:
        details = "\n".join(f"- {message}" for message in mismatches)
        raise ValueError(f"validation source mismatch:\n{details}")
    return {
        "passed": True,
        "raw_eval_dir": expected_raw_dir,
        "reference_eval_dir": expected_reference_dir,
        "weights_root": canonical_path(args.weights_root),
        "rpn_probe_json": canonical_path(Path(str(probe_json))),
        "pipeline_manifest": canonical_path(expected_manifest_path),
    }


def read_layer(weight_dir: Path, prefix: str, in_channels: int, out_channels: int):
    return {
        "linear": np.fromfile(
            weight_dir / f"{prefix}_linear_weight.bin", np.float32
        ).reshape(out_channels, in_channels),
        "weight": np.fromfile(weight_dir / f"{prefix}_bn_weight.bin", np.float32),
        "bias": np.fromfile(weight_dir / f"{prefix}_bn_bias.bin", np.float32),
        "mean": np.fromfile(weight_dir / f"{prefix}_bn_mean.bin", np.float32),
        "var": np.fromfile(weight_dir / f"{prefix}_bn_var.bin", np.float32),
    }


def batch_norm_relu(values: np.ndarray, layer, epsilon: float) -> np.ndarray:
    scale = layer["weight"] / np.sqrt(layer["var"] + np.float32(epsilon))
    values = (values - layer["mean"]) * scale + layer["bias"]
    return np.maximum(values, np.float32(0.0)).astype(np.float32)


def validate_pfn(frame_dir: Path, weight_dir: Path) -> dict[str, object]:
    decorated_meta = read_json(frame_dir / "03_decorated" / "decorated_metadata.json")
    pfn_meta = read_json(frame_dir / "04_pfn" / "pillar_features_metadata.json")
    weight_meta = read_json(weight_dir / "weights_metadata.json")
    shape = (
        int(decorated_meta["num_pillars"]),
        int(decorated_meta["max_points_per_pillar"]),
        int(decorated_meta["decorated_feature_dim"]),
    )
    decorated = np.fromfile(
        frame_dir / "03_decorated" / "decorated_pillars.bin", np.float32
    ).reshape(shape)
    layer0 = read_layer(weight_dir, "layer0", 10, 32)
    layer1 = read_layer(weight_dir, "layer1", 64, 64)
    epsilon = float(weight_meta["batch_norm_eps"])

    local = batch_norm_relu(decorated @ layer0["linear"].T, layer0, epsilon)
    local_max = local.max(axis=1, keepdims=True)
    concatenated = np.concatenate(
        [local, np.repeat(local_max, local.shape[1], axis=1)], axis=2
    )
    final = batch_norm_relu(concatenated @ layer1["linear"].T, layer1, epsilon)
    expected = final.max(axis=1)
    actual = np.fromfile(
        frame_dir / "04_pfn" / "pillar_features.bin", np.float32
    ).reshape(int(pfn_meta["num_pillars"]), int(pfn_meta["out_channels"]))
    difference = np.abs(expected - actual)
    return {
        "pillars": shape[0],
        "passed": bool(np.allclose(expected, actual, rtol=1.0e-5, atol=2.0e-5)),
        "max_abs_diff": float(difference.max()) if difference.size else 0.0,
        "mean_abs_diff": float(difference.mean()) if difference.size else 0.0,
    }


def validate_scatter(frame_dir: Path) -> dict[str, object]:
    pfn_meta = read_json(frame_dir / "04_pfn" / "pillar_features_metadata.json")
    voxel_meta = read_json(frame_dir / "02_voxel" / "metadata.json")
    scatter_meta = read_json(frame_dir / "05_scatter" / "bev_features_metadata.json")
    pillars = int(pfn_meta["num_pillars"])
    channels = int(pfn_meta["out_channels"])
    grid_x, grid_y, _ = [int(value) for value in voxel_meta["grid_size_xyz"]]
    shape = tuple(int(value) for value in scatter_meta["shape"])
    features = np.fromfile(
        frame_dir / "04_pfn" / "pillar_features.bin", np.float32
    ).reshape(pillars, channels)
    coordinates = np.fromfile(
        frame_dir / "02_voxel" / "coordinates.bin", np.int32
    ).reshape(pillars, 4)
    actual = np.memmap(
        frame_dir / "05_scatter" / "bev_features.bin",
        dtype=np.float32,
        mode="r",
        shape=shape,
    )
    expected = np.zeros((1, channels, grid_y, grid_x), dtype=np.float32)
    for pillar, coordinate in enumerate(coordinates):
        batch, _, y, x = [int(value) for value in coordinate]
        expected[batch, :, y, x] = features[pillar]
    difference = np.abs(expected - actual)
    return {
        "shape": list(shape),
        "passed": bool(np.array_equal(expected, actual)),
        "max_abs_diff": float(difference.max()) if difference.size else 0.0,
    }


def compare_preprocessing(raw_frame: Path, reference_frame: Path) -> dict[str, object]:
    raw_points = np.fromfile(raw_frame / "points.bin", np.float32).reshape(-1, 5)
    reference_points = np.fromfile(reference_frame / "points.bin", np.float32).reshape(-1, 5)
    if raw_points.shape != reference_points.shape:
        raise ValueError("raw and reference point shapes differ")
    expected_intensity = np.tanh(raw_points[:, 3])
    return {
        "points": int(raw_points.shape[0]),
        "xyz_equal": bool(np.array_equal(raw_points[:, :3], reference_points[:, :3])),
        "elongation_equal": bool(np.array_equal(raw_points[:, 4], reference_points[:, 4])),
        "reference_intensity_matches_tanh": bool(
            np.array_equal(expected_intensity, reference_points[:, 3])
        ),
        "raw_intensity_min": float(raw_points[:, 3].min()),
        "raw_intensity_max": float(raw_points[:, 3].max()),
        "reference_intensity_min": float(reference_points[:, 3].min()),
        "reference_intensity_max": float(reference_points[:, 3].max()),
        "coordinates_equal": bool(
            np.array_equal(
                np.fromfile(raw_frame / "02_voxel" / "coordinates.bin", np.int32),
                np.fromfile(reference_frame / "02_voxel" / "coordinates.bin", np.int32),
            )
        ),
        "num_points_equal": bool(
            np.array_equal(
                np.fromfile(raw_frame / "02_voxel" / "num_points.bin", np.int32),
                np.fromfile(reference_frame / "02_voxel" / "num_points.bin", np.int32),
            )
        ),
    }


def preprocessing_checks_passed(result: dict[str, object]) -> bool:
    return all(result.get(key) is True for key in PREPROCESSING_EQUALITY_KEYS)


def all_stage_validations_passed(stage_validation: dict[str, object]) -> bool:
    passed_values = [
        bool(value)
        for key, value in stage_validation.items()
        if key.endswith("passed")
    ]
    return bool(passed_values) and all(passed_values)


def metric_summary(report: dict[str, object]) -> dict[str, object]:
    return {
        key: report[key]
        for key in (
            "total_predictions",
            "total_labels",
            "tp",
            "fp",
            "fn",
            "precision",
            "recall",
        )
    }


def main() -> int:
    args = parse_args()
    raw_metrics = read_json(args.raw_eval_dir / "aggregate_report.json")
    reference_metrics = read_json(args.reference_eval_dir / "aggregate_report.json")
    comparison_contract = validate_experiment_contracts(
        raw_metrics, reference_metrics, args.weights_root
    )
    frames = [str(frame) for frame in reference_metrics["frames"]]
    frame_results = []
    for frame in frames:
        raw_frame = args.raw_eval_dir / frame
        reference_frame = args.reference_eval_dir / frame
        frame_results.append(
            {
                "frame": frame,
                "preprocessing": compare_preprocessing(raw_frame, reference_frame),
                "pfn": validate_pfn(reference_frame, args.weights_root / "04_pfn"),
                "scatter": validate_scatter(reference_frame),
            }
        )

    rpn = read_json(args.rpn_probe_validation)
    head = read_json(args.head_reference_validation)
    raw_head = read_json(args.raw_head_summary)
    reference_head = read_json(args.reference_head_summary)
    source_provenance = validate_source_provenance(
        args,
        frames,
        str(comparison_contract["archive"]),
        require_mapping(raw_metrics.get("run_contract"), "raw run_contract"),
        require_mapping(
            reference_metrics.get("run_contract"), "reference run_contract"
        ),
        rpn,
        head,
        raw_head,
        reference_head,
    )
    report = {
        "comparison": "raw intensity vs original CenterPoint tanh intensity",
        "frames": frames,
        "comparison_contract": comparison_contract,
        "source_provenance": source_provenance,
        "raw_metrics": metric_summary(raw_metrics),
        "reference_metrics": metric_summary(reference_metrics),
        "metric_delta": {
            "tp": int(reference_metrics["tp"]) - int(raw_metrics["tp"]),
            "fp": int(reference_metrics["fp"]) - int(raw_metrics["fp"]),
            "fn": int(reference_metrics["fn"]) - int(raw_metrics["fn"]),
            "precision": float(reference_metrics["precision"])
            - float(raw_metrics["precision"]),
            "recall": float(reference_metrics["recall"])
            - float(raw_metrics["recall"]),
        },
        "heatmap_outcomes": {
            "raw": raw_head["outcome_counts"],
            "reference": reference_head["outcome_counts"],
        },
        "stage_validation": {
            "comparison_contract_passed": bool(comparison_contract["passed"]),
            "source_provenance_passed": bool(source_provenance["passed"]),
            "preprocessing_all_frames_passed": all(
                preprocessing_checks_passed(row["preprocessing"])
                for row in frame_results
            ),
            "pfn_all_frames_passed": all(bool(row["pfn"]["passed"]) for row in frame_results),
            "pfn_max_abs_diff": max(float(row["pfn"]["max_abs_diff"]) for row in frame_results),
            "scatter_all_frames_passed": all(
                bool(row["scatter"]["passed"]) for row in frame_results
            ),
            "scatter_max_abs_diff": max(
                float(row["scatter"]["max_abs_diff"]) for row in frame_results
            ),
            "rpn_probes_passed": bool(rpn["passed"]),
            "rpn_probe_count": int(rpn["probes"]),
            "rpn_max_abs_diff": float(rpn["max_abs_diff"]),
            "head_gt_peaks_passed": bool(head["passed"]),
            "head_gt_peak_count": int(head["samples"]),
            "head_max_abs_diff": float(head["max_abs_diff"]),
        },
        "frame_results": frame_results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "frame_results"}, indent=2))
    return 0 if all_stage_validations_passed(report["stage_validation"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
