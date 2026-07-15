from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = (
    PROJECT_ROOT
    / "12_waymo_fn_analysis_project"
    / "tools"
    / "analyze_waymo_false_negatives.py"
)
SPEC = importlib.util.spec_from_file_location("waymo_fn_analysis", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {SCRIPT_PATH}")
analysis = importlib.util.module_from_spec(SPEC)
sys.modules["waymo_fn_analysis"] = analysis
SPEC.loader.exec_module(analysis)

RUNNER_PATH = (
    PROJECT_ROOT
    / "09_full_pipeline_project"
    / "tools"
    / "run_waymo_multiframe_eval.py"
)
RUNNER_SPEC = importlib.util.spec_from_file_location("waymo_multiframe_eval", RUNNER_PATH)
if RUNNER_SPEC is None or RUNNER_SPEC.loader is None:
    raise RuntimeError(f"cannot load {RUNNER_PATH}")
runner = importlib.util.module_from_spec(RUNNER_SPEC)
sys.modules["waymo_multiframe_eval"] = runner
RUNNER_SPEC.loader.exec_module(runner)


def make_box(**overrides):
    box = {
        "x": 0.0,
        "y": 0.0,
        "z": 0.0,
        "length": 4.0,
        "width": 2.0,
        "height": 2.0,
        "heading": 0.0,
    }
    box.update(overrides)
    return box


def make_record(**overrides):
    record = {
        "official_geometry_matched": False,
        "center_in_model_xy": True,
        "box_overlaps_model_z": True,
        "official_num_lidar_points": 20,
        "point_counts": {
            "effective_model_points": 10,
            "selected_points_in_box": 10,
            "all_archive_points_in_box": 20,
            "range_retention": 1.0,
        },
        "heatmap": {"local_max_score": 0.2, "score_threshold": 0.35},
    }
    record.update(overrides)
    return record


class GeometryTests(unittest.TestCase):
    def test_identical_boxes_have_unit_iou(self) -> None:
        box = make_box(heading=0.37)
        self.assertAlmostEqual(analysis.rotated_iou(box, box), 1.0, places=7)

    def test_axis_aligned_half_overlap_has_one_third_iou(self) -> None:
        first = make_box()
        second = make_box(x=2.0)
        self.assertAlmostEqual(
            analysis.rotated_iou(first, second), 1.0 / 3.0, places=7
        )

    def test_waymo_point_mask_uses_ccw_heading(self) -> None:
        heading = math.pi / 4.0
        point = np.asarray(
            [[1.5 * math.cos(heading), 1.5 * math.sin(heading), 0.0, 1.0, 0.0, -1.0]],
            dtype=np.float32,
        )
        box = make_box(width=1.0, heading=heading)
        self.assertTrue(bool(analysis.points_in_waymo_box(point, box)[0]))
        self.assertFalse(
            bool(analysis.points_in_waymo_box(point, box, mirrored_heading=True)[0])
        )

    def test_official_geometry_matches_identical_prediction(self) -> None:
        label = {
            "id": "label",
            "class_name": "VEHICLE",
            "box": make_box(heading=-0.2),
        }
        prediction = {
            **make_box(heading=-0.2),
            "class_name": "VEHICLE",
            "score": 0.9,
        }
        result = analysis.evaluate_official_geometry([prediction], [label], 0.5)
        self.assertEqual(result["tp"], 1)
        self.assertEqual(result["fp"], 0)
        self.assertEqual(result["fn"], 0)

    def test_multiframe_evaluator_uses_waymo_ccw_conversion(self) -> None:
        heading = 0.7
        label = runner.Box(
            x=10.0,
            y=-3.0,
            dx=2.0,
            dy=4.0,
            yaw=heading,
            label="VEHICLE",
            convention="waymo_label",
        )
        prediction = runner.Box(
            x=10.0,
            y=-3.0,
            dx=2.0,
            dy=4.0,
            yaw=-heading - math.pi / 2.0,
            label="VEHICLE",
            convention="prediction",
        )
        self.assertAlmostEqual(runner.rotated_iou(prediction, label), 1.0, places=7)


class PointCountTests(unittest.TestCase):
    def test_effective_count_tracks_range_and_nlz(self) -> None:
        sources = {
            "TOP_return1": np.asarray(
                [
                    [0.0, 0.0, 0.0, 1.0, 0.0, -1.0],
                    [0.0, 0.0, 0.5, 1.0, 0.0, 1.0],
                    [0.0, 0.0, 3.0, 1.0, 0.0, -1.0],
                ],
                dtype=np.float32,
            )
        }
        preprocessing = {
            "lidars": ["TOP"],
            "returns": ["return1"],
            "drop_nlz": False,
        }
        counts = analysis.analyze_point_counts(
            sources, make_box(height=8.0), [-2.0, -2.0, -2.0, 2.0, 2.0, 2.0], preprocessing
        )
        self.assertEqual(counts["selected_points_in_box"], 3)
        self.assertEqual(counts["selected_points_in_model_range"], 2)
        self.assertEqual(counts["selected_non_nlz_points_in_model_range"], 1)
        self.assertEqual(counts["effective_model_points"], 2)

        preprocessing["drop_nlz"] = True
        filtered = analysis.analyze_point_counts(
            sources, make_box(height=8.0), [-2.0, -2.0, -2.0, 2.0, 2.0, 2.0], preprocessing
        )
        self.assertEqual(filtered["effective_model_points"], 1)


class ClassificationTests(unittest.TestCase):
    def test_geometry_mismatch_has_priority(self) -> None:
        classification, _ = analysis.classify_false_negative(
            make_record(official_geometry_matched=True), 5
        )
        self.assertEqual(classification, "EVALUATION_GEOMETRY_MISMATCH")

    def test_low_point_and_preprocessing_classes(self) -> None:
        low_points = make_record(
            point_counts={
                "effective_model_points": 2,
                "selected_points_in_box": 2,
                "all_archive_points_in_box": 2,
                "range_retention": 1.0,
            }
        )
        classification, _ = analysis.classify_false_negative(low_points, 5)
        self.assertEqual(classification, "LOW_POINT_COUNT")

        filtered = make_record(
            point_counts={
                "effective_model_points": 2,
                "selected_points_in_box": 10,
                "all_archive_points_in_box": 10,
                "range_retention": 0.2,
            }
        )
        classification, _ = analysis.classify_false_negative(filtered, 5)
        self.assertEqual(classification, "PREPROCESSING_SENSITIVE")

    def test_score_and_regression_classes(self) -> None:
        classification, _ = analysis.classify_false_negative(make_record(), 5)
        self.assertEqual(classification, "LOW_MODEL_SCORE")

        regression = make_record(
            heatmap={"local_max_score": 0.5, "score_threshold": 0.35}
        )
        classification, _ = analysis.classify_false_negative(regression, 5)
        self.assertEqual(classification, "BOX_REGRESSION_ERROR")


if __name__ == "__main__":
    unittest.main()
