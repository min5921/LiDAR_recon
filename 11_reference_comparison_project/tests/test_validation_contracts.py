from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_module(name: str, relative_path: str):
    path = PROJECT_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = load_module(
    "centerpoint_multiframe_runner",
    "09_full_pipeline_project/tools/run_waymo_multiframe_eval.py",
)
comparison = load_module(
    "centerpoint_reference_comparison",
    "11_reference_comparison_project/tools/run_reference_comparison.py",
)


class CacheContractTests(unittest.TestCase):
    def test_file_signature_changes_when_contents_change(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dependency = root / "weight.bin"
            dependency.write_bytes(b"one")
            first = runner.file_set_signature(root, [Path("weight.bin")])
            dependency.write_bytes(b"two")
            second = runner.file_set_signature(root, [Path("weight.bin")])
            self.assertNotEqual(first["sha256"], second["sha256"])

    def test_preprocessing_cache_requires_archive_and_frame(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "segment.zip"
            archive.write_bytes(b"archive")
            summary_path = root / "export_summary.json"
            summary = {
                "archive": str(archive),
                "frame": "frame_000",
                "intensity_transform": "tanh",
                "drop_nlz": False,
                "lidars": ["TOP"],
                "returns": ["return1"],
            }
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            args = SimpleNamespace(
                archive=archive,
                intensity_transform="tanh",
                drop_nlz=False,
                lidars=["TOP"],
                returns=["return1"],
            )
            self.assertTrue(
                runner.preprocessing_config_matches(
                    summary_path, args, "frame_000"
                )
            )
            self.assertFalse(
                runner.preprocessing_config_matches(
                    summary_path, args, "frame_001"
                )
            )
            args.archive = root / "different.zip"
            self.assertFalse(
                runner.preprocessing_config_matches(
                    summary_path, args, "frame_000"
                )
            )

    def test_cache_manifest_rejects_changed_decode_settings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = Path(temporary) / "pipeline_cache_manifest.json"
            manifest = {
                "schema_version": 1,
                "decode": {"score_threshold": 0.35},
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertTrue(runner.cache_manifest_matches(manifest_path, manifest))
            changed = {
                "schema_version": 1,
                "decode": {"score_threshold": 0.25},
            }
            self.assertFalse(runner.cache_manifest_matches(manifest_path, changed))


class ComparisonContractTests(unittest.TestCase):
    def make_metrics(
        self,
        archive: Path,
        weights_root: Path,
        intensity_transform: str,
    ) -> dict[str, object]:
        frames = ["frame_000", "frame_001"]
        decode = {
            "nms_iou": 0.5,
            "score_threshold": 0.35,
            "nms_convention": "pcdet",
            "class_score_thresholds": None,
        }
        contract = {
            "schema_version": 1,
            "archive": {
                "path": str(archive.resolve()),
                "size": 123,
                "modified_time_ns": 456,
            },
            "frames": frames,
            "preprocessing": {
                "intensity_transform": intensity_transform,
                "drop_nlz": False,
                "lidars": ["TOP"],
                "returns": ["return1", "return2"],
            },
            "decode": decode,
            "evaluation": {"match_iou": 0.5},
            "dependencies": {
                "weights_root": str(weights_root.resolve()),
                "project_files": {"sha256": "project"},
                "weight_files": {"sha256": "weights"},
            },
        }
        return {
            "archive": str(archive),
            "frames": frames,
            "nms_iou": 0.5,
            "score_threshold": 0.35,
            "nms_convention": "pcdet",
            "class_score_thresholds": None,
            "match_iou": 0.5,
            "run_contract": contract,
        }

    def test_contract_accepts_only_intensity_difference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "segment.zip"
            weights = root / "weights"
            raw = self.make_metrics(archive, weights, "none")
            reference = self.make_metrics(archive, weights, "tanh")
            result = comparison.validate_experiment_contracts(
                raw, reference, weights
            )
            self.assertTrue(result["passed"])

    def test_contract_rejects_decode_difference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "segment.zip"
            weights = root / "weights"
            raw = self.make_metrics(archive, weights, "none")
            reference = self.make_metrics(archive, weights, "tanh")
            reference["score_threshold"] = 0.25
            reference["run_contract"]["decode"]["score_threshold"] = 0.25
            with self.assertRaisesRegex(ValueError, "run contracts differ in decode"):
                comparison.validate_experiment_contracts(raw, reference, weights)

    def test_preprocessing_failure_fails_the_stage(self) -> None:
        preprocessing = {
            key: True for key in comparison.PREPROCESSING_EQUALITY_KEYS
        }
        preprocessing["coordinates_equal"] = False
        self.assertFalse(comparison.preprocessing_checks_passed(preprocessing))
        self.assertFalse(
            comparison.all_stage_validations_passed(
                {
                    "preprocessing_all_frames_passed": False,
                    "pfn_all_frames_passed": True,
                }
            )
        )

    def test_source_provenance_rejects_probe_from_another_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw_dir = root / "raw"
            reference_dir = root / "reference"
            weights = root / "weights"
            probe_json = reference_dir / "frame_000" / "probe" / "rpn_layer_probes.json"
            probe_json.parent.mkdir(parents=True)
            probe_json.write_text("{}", encoding="utf-8")
            archive = str((root / "segment.zip").resolve())
            args = SimpleNamespace(
                raw_eval_dir=raw_dir,
                reference_eval_dir=reference_dir,
                weights_root=weights,
            )
            frames = ["frame_000"]
            shared_contract = {
                "schema_version": 1,
                "archive": {"path": archive, "size": 123, "modified_time_ns": 456},
                "frames": frames,
                "preprocessing": {
                    "intensity_transform": "none",
                    "drop_nlz": False,
                    "lidars": ["TOP"],
                    "returns": ["return1"],
                },
                "decode": {
                    "nms_iou": 0.5,
                    "score_threshold": 0.35,
                    "nms_convention": "pcdet",
                    "class_score_thresholds": None,
                },
                "evaluation": {"match_iou": 0.5},
                "dependencies": {"weights_root": str(weights.resolve())},
            }
            raw_contract = dict(shared_contract)
            reference_contract = dict(shared_contract)
            reference_contract["preprocessing"] = {
                **shared_contract["preprocessing"],
                "intensity_transform": "tanh",
            }
            pipeline_manifest = {
                "schema_version": 1,
                "archive": reference_contract["archive"],
                "frame": "frame_000",
                "preprocessing": reference_contract["preprocessing"],
                "decode": reference_contract["decode"],
                "dependencies": reference_contract["dependencies"],
            }
            manifest_path = reference_dir / "frame_000" / "pipeline_cache_manifest.json"
            manifest_path.write_text(json.dumps(pipeline_manifest), encoding="utf-8")
            raw_head = {
                "eval_dir": str(raw_dir),
                "archive": archive,
                "frame_summaries": [{"frame": "frame_000"}],
                "run_contract": raw_contract,
            }
            reference_head = {
                "eval_dir": str(reference_dir),
                "archive": archive,
                "frame_summaries": [{"frame": "frame_000"}],
                "run_contract": reference_contract,
            }
            rpn = {
                "weight_dir": str(weights / "06_rpn"),
                "probe_json": str(probe_json),
                "pipeline_manifest_path": str(manifest_path),
                "pipeline_cache_manifest": pipeline_manifest,
            }
            head = {
                "eval_dir": str(reference_dir),
                "weight_dir": str(weights / "07_head"),
                "frame_names": frames,
                "run_contract": reference_contract,
            }
            result = comparison.validate_source_provenance(
                args,
                frames,
                archive,
                raw_contract,
                reference_contract,
                rpn,
                head,
                raw_head,
                reference_head,
            )
            self.assertTrue(result["passed"])

            outside_probe = root / "old_run" / "rpn_layer_probes.json"
            outside_probe.parent.mkdir()
            outside_probe.write_text("{}", encoding="utf-8")
            rpn["probe_json"] = str(outside_probe)
            with self.assertRaisesRegex(ValueError, "not from the reference"):
                comparison.validate_source_provenance(
                    args,
                    frames,
                    archive,
                    raw_contract,
                    reference_contract,
                    rpn,
                    head,
                    raw_head,
                    reference_head,
                )

            rpn["probe_json"] = str(probe_json)
            reference_head["run_contract"] = {"stale": True}
            with self.assertRaisesRegex(ValueError, "stale or unrelated"):
                comparison.validate_source_provenance(
                    args,
                    frames,
                    archive,
                    raw_contract,
                    reference_contract,
                    rpn,
                    head,
                    raw_head,
                    reference_head,
                )


if __name__ == "__main__":
    unittest.main()
