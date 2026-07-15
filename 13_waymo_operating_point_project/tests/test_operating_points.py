from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


study = load_module(
    "waymo_operating_point_study",
    PROJECT_ROOT
    / "13_waymo_operating_point_project"
    / "tools"
    / "analyze_operating_points.py",
)
runner = load_module(
    "waymo_multiframe_eval_for_compact_test",
    PROJECT_ROOT
    / "09_full_pipeline_project"
    / "tools"
    / "run_waymo_multiframe_eval.py",
)


def box(x: float) -> dict[str, float]:
    return {
        "x": x,
        "y": 0.0,
        "z": 0.0,
        "length": 4.0,
        "width": 2.0,
        "height": 2.0,
        "heading": 0.0,
    }


def label(label_id: str, x: float, points: int) -> dict[str, object]:
    return {
        "id": label_id,
        "class_name": "VEHICLE",
        "box": box(x),
        "distance_m": abs(x),
        "point_count": points,
    }


def prediction(x: float, score: float) -> dict[str, object]:
    return {
        **box(x),
        "class_name": "VEHICLE",
        "score": score,
    }


class BinTests(unittest.TestCase):
    def test_bins_include_boundaries_and_open_last_bin(self) -> None:
        bins = study.make_bins([0.0, 5.0, 10.0], "points")
        self.assertEqual(study.bin_label(0.0, bins), "0-4 points")
        self.assertEqual(study.bin_label(5.0, bins), "5-9 points")
        self.assertEqual(study.bin_label(100.0, bins), "10+ points")
        self.assertIsNone(bins[-1]["upper"])

    def test_invalid_edges_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            study.make_bins([1.0, 2.0], "distance")
        with self.assertRaises(ValueError):
            study.make_bins([0.0, 5.0, 5.0], "points")


class EvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.distance_bins = study.make_bins([0.0, 5.0, 20.0], "distance")
        self.point_bins = study.make_bins([0.0, 5.0, 10.0], "points")
        self.frame_data = [
            {
                "frame": "frame_000",
                "labels": [label("near", 0.0, 12), label("far", 10.0, 3)],
                "predictions": [
                    prediction(0.0, 0.8),
                    prediction(30.0, 0.4),
                    prediction(10.0, 0.2),
                ],
            }
        ]

    def test_threshold_filter_changes_tp_and_fp(self) -> None:
        low = study.evaluate_threshold(
            self.frame_data, 0.1, 0.5, self.distance_bins, self.point_bins
        )
        high = study.evaluate_threshold(
            self.frame_data, 0.35, 0.5, self.distance_bins, self.point_bins
        )
        self.assertEqual(low["overall"]["tp"], 2)
        self.assertEqual(low["overall"]["fp"], 1)
        self.assertEqual(high["overall"]["tp"], 1)
        self.assertEqual(high["overall"]["fp"], 1)
        self.assertEqual(high["point_bins"]["0-4 points"]["recall"], 0.0)

    def test_monotonic_validation_rejects_increasing_counts(self) -> None:
        rows = [
            {"overall": {"predictions": 1, "tp": 1}},
            {"overall": {"predictions": 2, "tp": 1}},
        ]
        with self.assertRaises(ValueError):
            study.validate_monotonic_results(rows)

    def test_operating_point_selection(self) -> None:
        rows = [
            {
                "threshold": 0.1,
                "overall": study.metric_block(4, 2, 2),
            },
            {
                "threshold": 0.2,
                "overall": study.metric_block(2, 2, 2),
            },
            {
                "threshold": 0.3,
                "overall": study.metric_block(1, 2, 1),
            },
        ]
        selected = study.select_operating_points(rows, 0.8)
        self.assertEqual(selected["best_f1"]["threshold"], 0.2)
        self.assertEqual(
            selected["best_recall_at_precision_floor"]["threshold"], 0.2
        )


class CompactOutputTests(unittest.TestCase):
    def test_compact_output_keeps_reports_and_detections(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            frame_dir = Path(temporary) / "frame_000"
            for name in [
                "02_voxel",
                "03_decorated",
                "04_pfn",
                "05_scatter",
                "06_rpn",
                "07_head",
                "08_detections",
            ]:
                (frame_dir / name).mkdir(parents=True)
            (frame_dir / "points.bin").write_bytes(b"points")
            (frame_dir / "07_head" / "hm.bin").write_bytes(b"head")
            detections = frame_dir / "08_detections" / "detections.csv"
            detections.write_text("score\n0.5\n", encoding="utf-8")
            report = frame_dir / "match_report.json"
            report.write_text("{}", encoding="utf-8")

            self.assertTrue(runner.full_cache_outputs_present(frame_dir))
            runner.compact_frame_outputs(frame_dir)

            self.assertFalse((frame_dir / "points.bin").exists())
            self.assertFalse((frame_dir / "07_head").exists())
            self.assertTrue(detections.exists())
            self.assertTrue(report.exists())
            self.assertFalse(runner.full_cache_outputs_present(frame_dir))


if __name__ == "__main__":
    unittest.main()
