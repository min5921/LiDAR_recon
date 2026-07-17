import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


VOXEL_SIZE = np.asarray([0.32, 0.32, 6.0], dtype=np.float32)
POINT_CLOUD_RANGE = np.asarray(
    [-74.88, -74.88, -2.0, 74.88, 74.88, 4.0], dtype=np.float32
)
GRID_SIZE = np.asarray([468, 468, 1], dtype=np.int32)
MAX_POINTS = 20
MAX_PILLARS = 60000
FEATURE_DIMENSION = 5
PFN_CHANNELS = 64


def read_layer(weight_directory, prefix, in_channels, out_channels):
    return {
        "linear": np.fromfile(
            weight_directory / f"{prefix}_linear_weight.bin", np.float32
        ).reshape(out_channels, in_channels),
        "weight": np.fromfile(
            weight_directory / f"{prefix}_bn_weight.bin", np.float32
        ),
        "bias": np.fromfile(
            weight_directory / f"{prefix}_bn_bias.bin", np.float32
        ),
        "mean": np.fromfile(
            weight_directory / f"{prefix}_bn_mean.bin", np.float32
        ),
        "variance": np.fromfile(
            weight_directory / f"{prefix}_bn_var.bin", np.float32
        ),
    }


def batch_norm_relu(values, layer, epsilon):
    normalized = (values - layer["mean"]) / np.sqrt(
        layer["variance"] + np.float32(epsilon)
    )
    return np.maximum(
        normalized * layer["weight"] + layer["bias"], np.float32(0.0)
    ).astype(np.float32)


def voxelize_reference(points):
    coordinate_to_pillar = -np.ones(
        (int(GRID_SIZE[2]), int(GRID_SIZE[1]), int(GRID_SIZE[0])),
        dtype=np.int32,
    )
    pillars = np.zeros(
        (MAX_PILLARS, MAX_POINTS, FEATURE_DIMENSION), dtype=np.float32
    )
    coordinates = np.zeros((MAX_PILLARS, 4), dtype=np.int32)
    num_points = np.zeros((MAX_PILLARS,), dtype=np.int32)

    pillar_count = 0
    valid_points = 0
    for point in points:
        xyz = np.floor(
            (point[:3] - POINT_CLOUD_RANGE[:3]) / VOXEL_SIZE
        ).astype(np.int32)
        if np.any(xyz < 0) or np.any(xyz >= GRID_SIZE):
            continue
        valid_points += 1
        x, y, z = [int(value) for value in xyz]
        pillar = int(coordinate_to_pillar[z, y, x])
        if pillar < 0:
            if pillar_count >= MAX_PILLARS:
                continue
            pillar = pillar_count
            pillar_count += 1
            coordinate_to_pillar[z, y, x] = pillar
            coordinates[pillar] = [0, z, y, x]

        point_in_pillar = int(num_points[pillar])
        if point_in_pillar < MAX_POINTS:
            pillars[pillar, point_in_pillar] = point
            num_points[pillar] += 1

    return (
        pillars[:pillar_count],
        coordinates[:pillar_count],
        num_points[:pillar_count],
        valid_points,
    )


def build_python_bev(points, weight_directory, chunk_size):
    metadata = json.loads(
        (weight_directory / "weights_metadata.json").read_text(encoding="utf-8")
    )
    layer0 = read_layer(weight_directory, "layer0", 10, 32)
    layer1 = read_layer(weight_directory, "layer1", 64, 64)
    epsilon = np.float32(metadata["batch_norm_eps"])
    pillars, coordinates, num_points, valid_points = voxelize_reference(points)

    bev = np.zeros(
        (1, PFN_CHANNELS, int(GRID_SIZE[1]), int(GRID_SIZE[0])),
        dtype=np.float32,
    )
    point_slots = np.arange(MAX_POINTS, dtype=np.int32)[None, :]
    for start in range(0, pillars.shape[0], chunk_size):
        end = min(start + chunk_size, pillars.shape[0])
        current = pillars[start:end]
        counts = num_points[start:end]
        mask = point_slots < counts[:, None]

        decorated = np.zeros(
            (end - start, MAX_POINTS, FEATURE_DIMENSION + 5),
            dtype=np.float32,
        )
        decorated[:, :, :FEATURE_DIMENSION] = current
        mean_xyz = current[:, :, :3].sum(axis=1) / counts[:, None].astype(
            np.float32
        )
        decorated[:, :, 5:8] = (
            current[:, :, :3] - mean_xyz[:, None, :]
        ) * mask[:, :, None]

        current_coordinates = coordinates[start:end]
        center_x = (
            current_coordinates[:, 3].astype(np.float32) * VOXEL_SIZE[0]
            + VOXEL_SIZE[0] * np.float32(0.5)
            + POINT_CLOUD_RANGE[0]
        )
        center_y = (
            current_coordinates[:, 2].astype(np.float32) * VOXEL_SIZE[1]
            + VOXEL_SIZE[1] * np.float32(0.5)
            + POINT_CLOUD_RANGE[1]
        )
        decorated[:, :, 8] = (
            current[:, :, 0] - center_x[:, None]
        ) * mask
        decorated[:, :, 9] = (
            current[:, :, 1] - center_y[:, None]
        ) * mask

        local = decorated @ layer0["linear"].T
        local = batch_norm_relu(local, layer0, epsilon)
        local_max = local.max(axis=1, keepdims=True)
        concatenated = np.concatenate(
            [local, np.broadcast_to(local_max, local.shape)], axis=2
        )
        final = concatenated @ layer1["linear"].T
        final = batch_norm_relu(final, layer1, epsilon).max(axis=1)

        spatial = (
            current_coordinates[:, 2].astype(np.int64) * int(GRID_SIZE[0])
            + current_coordinates[:, 3].astype(np.int64)
        )
        bev.reshape(PFN_CHANNELS, -1)[:, spatial] = final.T

    return bev, {
        "input_points": int(points.shape[0]),
        "valid_points": int(valid_points),
        "selected_pillars": int(pillars.shape[0]),
    }


def run_gpu(executable, points_path, weight_directory, output_directory):
    output_directory.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            str(executable),
            str(points_path),
            str(weight_directory),
            "--output-dir",
            str(output_directory),
        ],
        check=True,
    )
    summary = json.loads(
        (output_directory / "summary.json").read_text(encoding="utf-8")
    )
    bev = np.fromfile(output_directory / "bev_features.bin", np.float32).reshape(
        1, PFN_CHANNELS, int(GRID_SIZE[1]), int(GRID_SIZE[0])
    )
    return bev, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-exe", required=True, type=Path)
    parser.add_argument("--points", required=True, type=Path)
    parser.add_argument("--weight-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--chunk-size", default=2048, type=int)
    parser.add_argument("--keep-gpu-bev", action="store_true")
    args = parser.parse_args()

    points = np.fromfile(args.points, np.float32)
    if points.size % FEATURE_DIMENSION != 0:
        raise RuntimeError("point file is not divisible by five float features")
    points = points.reshape(-1, FEATURE_DIMENSION)

    python_bev, python_summary = build_python_bev(
        points, args.weight_dir, args.chunk_size
    )
    gpu_bev, gpu_summary = run_gpu(
        args.gpu_exe, args.points, args.weight_dir, args.output_dir / "gpu"
    )

    difference = np.abs(python_bev - gpu_bev)
    max_abs_difference = float(difference.max())
    mean_abs_difference = float(difference.mean())
    strict_allclose = bool(
        np.allclose(python_bev, gpu_bev, rtol=1.0e-5, atol=2.0e-5)
    )
    cuda_allclose = bool(
        np.allclose(python_bev, gpu_bev, rtol=1.0e-5, atol=1.0e-4)
    )
    shape_equal = python_bev.shape == gpu_bev.shape
    count_equal = all(
        python_summary[name] == int(gpu_summary[name])
        for name in ("input_points", "valid_points", "selected_pillars")
    )
    passed = shape_equal and count_equal and cuda_allclose
    report = {
        "passed": passed,
        "shape_equal": shape_equal,
        "count_equal": count_equal,
        "strict_allclose_atol_2e-5": strict_allclose,
        "cuda_allclose_atol_1e-4": cuda_allclose,
        "max_abs_difference": max_abs_difference,
        "mean_abs_difference": mean_abs_difference,
        "python": python_summary,
        "gpu": gpu_summary,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "comparison.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    if not args.keep_gpu_bev:
        (args.output_dir / "gpu" / "bev_features.bin").unlink(missing_ok=True)

    print(f"python BEV shape: {python_bev.shape}")
    print(f"GPU BEV shape:    {gpu_bev.shape}")
    print(f"point/pillar counts equal: {count_equal}")
    print(f"strict allclose (atol=2e-5): {strict_allclose}")
    print(f"CUDA allclose   (atol=1e-4): {cuda_allclose}")
    print(f"max abs diff:  {max_abs_difference:.8f}")
    print(f"mean abs diff: {mean_abs_difference:.10f}")
    print(f"result: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
