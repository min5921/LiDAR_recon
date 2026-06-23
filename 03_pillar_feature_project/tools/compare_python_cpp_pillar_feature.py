import argparse
import json
import sys
from pathlib import Path

import numpy as np


def decorate_reference(pillars, coordinates, num_points, metadata):
    num_pillars = metadata["num_pillars"]
    max_points = metadata["max_points_per_pillar"]
    feature_dim = metadata["feature_dim"]
    decorated_dim = feature_dim + 5
    voxel_size = np.asarray(metadata["voxel_size"], dtype=np.float32)
    pc_range = np.asarray(metadata["point_cloud_range"], dtype=np.float32)
    x_offset = voxel_size[0] / 2.0 + pc_range[0]
    y_offset = voxel_size[1] / 2.0 + pc_range[1]

    decorated = np.zeros((num_pillars, max_points, decorated_dim), dtype=np.float32)
    for pillar_idx in range(num_pillars):
        count = int(num_points[pillar_idx])
        if count == 0:
            continue
        valid = pillars[pillar_idx, :count, :]
        mean_xyz = valid[:, :3].sum(axis=0) / np.float32(count)
        y_coord = coordinates[pillar_idx, 2]
        x_coord = coordinates[pillar_idx, 3]
        center_x = np.float32(x_coord) * voxel_size[0] + x_offset
        center_y = np.float32(y_coord) * voxel_size[1] + y_offset

        decorated[pillar_idx, :count, :feature_dim] = valid
        decorated[pillar_idx, :count, feature_dim : feature_dim + 3] = valid[:, :3] - mean_xyz
        decorated[pillar_idx, :count, feature_dim + 3] = valid[:, 0] - center_x
        decorated[pillar_idx, :count, feature_dim + 4] = valid[:, 1] - center_y
    return decorated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--voxel-dump", required=True, type=Path)
    parser.add_argument("--decorated-dump", required=True, type=Path)
    args = parser.parse_args()

    metadata = json.loads((args.voxel_dump / "metadata.json").read_text(encoding="utf-8"))
    decorated_metadata = json.loads(
        (args.decorated_dump / "decorated_metadata.json").read_text(encoding="utf-8")
    )

    num_pillars = metadata["num_pillars"]
    max_points = metadata["max_points_per_pillar"]
    feature_dim = metadata["feature_dim"]
    decorated_dim = decorated_metadata["decorated_feature_dim"]

    pillars = np.fromfile(args.voxel_dump / "pillars.bin", dtype=np.float32).reshape(
        num_pillars, max_points, feature_dim
    )
    coordinates = np.fromfile(args.voxel_dump / "coordinates.bin", dtype=np.int32).reshape(
        num_pillars, 4
    )
    num_points = np.fromfile(args.voxel_dump / "num_points.bin", dtype=np.int32)

    py_decorated = decorate_reference(pillars, coordinates, num_points, metadata)
    cpp_decorated = np.fromfile(
        args.decorated_dump / "decorated_pillars.bin", dtype=np.float32
    ).reshape(num_pillars, max_points, decorated_dim)

    equal = np.array_equal(py_decorated, cpp_decorated)
    max_abs_diff = float(np.max(np.abs(py_decorated - cpp_decorated))) if py_decorated.size else 0.0

    print(f"python decorated shape: {py_decorated.shape}")
    print(f"cpp decorated shape:    {cpp_decorated.shape}")
    print(f"decorated equal:        {equal}")
    print(f"max abs diff:           {max_abs_diff:.8f}")

    if not equal:
        mismatch = np.argwhere(py_decorated != cpp_decorated)[0]
        i, j, k = [int(v) for v in mismatch]
        print(f"first mismatch: pillar={i}, point={j}, feature={k}")
        print(f"python: {float(py_decorated[i, j, k])}")
        print(f"cpp:    {float(cpp_decorated[i, j, k])}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

