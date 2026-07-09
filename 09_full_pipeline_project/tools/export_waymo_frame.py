#!/usr/bin/env python3
"""Export one derived Waymo sensor archive frame to CenterPoint point format.

Input archive lidar bins are float32 columns:
    x, y, z, intensity, elongation, nlz_flag

Output bin is float32 columns used by the current C++ CenterPoint pipeline:
    x, y, z, intensity, elongation
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np


LIDARS = ("TOP", "FRONT", "SIDE_LEFT", "SIDE_RIGHT", "REAR")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path, help="Waymo derived segment .zip")
    parser.add_argument("output_bin", type=Path, help="Output 5-feature float32 .bin")
    parser.add_argument("--frame", default="frame_000", help="Frame directory name")
    parser.add_argument(
        "--lidars",
        nargs="+",
        default=["TOP"],
        choices=LIDARS,
        help="Lidar groups to export",
    )
    parser.add_argument(
        "--returns",
        nargs="+",
        default=["return1"],
        choices=["return1", "return2"],
        help="Lidar returns to export",
    )
    parser.add_argument(
        "--drop-nlz",
        action="store_true",
        help="Drop points whose nlz_flag is not negative.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional JSON summary output path",
    )
    return parser.parse_args()


def read_lidar_entry(zf: zipfile.ZipFile, entry_name: str) -> np.ndarray:
    try:
        raw = zf.read(entry_name)
    except KeyError as exc:
        raise FileNotFoundError(f"missing zip entry: {entry_name}") from exc

    values = np.frombuffer(raw, dtype="<f4")
    if values.size % 6 != 0:
        raise ValueError(f"{entry_name} does not contain Nx6 float32 data")
    return values.reshape(-1, 6).copy()


def main() -> int:
    args = parse_args()
    arrays: list[np.ndarray] = []
    sources: list[dict[str, object]] = []

    with zipfile.ZipFile(args.archive) as zf:
        schema = json.loads(zf.read("schema.json").decode("utf-8"))
        if schema["lidar_bin"]["columns"] != [
            "x",
            "y",
            "z",
            "intensity",
            "elongation",
            "nlz_flag",
        ]:
            raise ValueError("unexpected lidar schema")

        for lidar in args.lidars:
            for ret in args.returns:
                entry = f"{args.frame}/lidar/{lidar}_{ret}.bin"
                points6 = read_lidar_entry(zf, entry)
                before = int(points6.shape[0])
                if args.drop_nlz:
                    points6 = points6[points6[:, 5] < 0.0]
                arrays.append(points6[:, :5].astype("<f4", copy=False))
                sources.append(
                    {
                        "entry": entry,
                        "points_before_filter": before,
                        "points_after_filter": int(points6.shape[0]),
                    }
                )

    points5 = np.concatenate(arrays, axis=0) if arrays else np.empty((0, 5), dtype="<f4")
    args.output_bin.parent.mkdir(parents=True, exist_ok=True)
    points5.astype("<f4", copy=False).tofile(args.output_bin)

    summary = {
        "archive": str(args.archive),
        "frame": args.frame,
        "output_bin": str(args.output_bin),
        "feature_columns": ["x", "y", "z", "intensity", "elongation"],
        "num_points": int(points5.shape[0]),
        "sources": sources,
        "min": points5.min(axis=0).tolist() if points5.size else None,
        "max": points5.max(axis=0).tolist() if points5.size else None,
        "mean": points5.mean(axis=0).tolist() if points5.size else None,
        "samples": points5[:5].tolist(),
    }

    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

