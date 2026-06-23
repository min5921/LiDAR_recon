import argparse
import json
import sys
from pathlib import Path

import numpy as np


def points_to_voxel_reference(points, voxel_size, coors_range, max_points, max_voxels):
    grid_size = np.round((coors_range[3:] - coors_range[:3]) / voxel_size).astype(np.int32)
    coor_to_voxelidx = -np.ones(tuple(grid_size[::-1].tolist()), dtype=np.int32)
    voxels = np.zeros((max_voxels, max_points, points.shape[1]), dtype=np.float32)
    coors = np.zeros((max_voxels, 3), dtype=np.int32)
    num_points_per_voxel = np.zeros((max_voxels,), dtype=np.int32)

    voxel_num = 0
    for point in points:
        coor = np.zeros((3,), dtype=np.int32)
        failed = False
        for axis in range(3):
            c = np.floor((point[axis] - coors_range[axis]) / voxel_size[axis]).astype(np.int32)
            if c < 0 or c >= grid_size[axis]:
                failed = True
                break
            coor[2 - axis] = c

        if failed:
            continue

        voxelidx = coor_to_voxelidx[coor[0], coor[1], coor[2]]
        if voxelidx == -1:
            if voxel_num >= max_voxels:
                continue
            voxelidx = voxel_num
            voxel_num += 1
            coor_to_voxelidx[coor[0], coor[1], coor[2]] = voxelidx
            coors[voxelidx] = coor

        num = num_points_per_voxel[voxelidx]
        if num < max_points:
            voxels[voxelidx, num] = point
            num_points_per_voxel[voxelidx] += 1

    return (
        voxels[:voxel_num],
        coors[:voxel_num],
        num_points_per_voxel[:voxel_num],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", required=True, type=Path)
    parser.add_argument("--cpp-dump", required=True, type=Path)
    parser.add_argument("--feature-dim", default=4, type=int)
    args = parser.parse_args()

    metadata = json.loads((args.cpp_dump / "metadata.json").read_text(encoding="utf-8"))
    points = np.fromfile(args.points, dtype=np.float32)
    if points.size % args.feature_dim != 0:
        raise RuntimeError("point file size is not divisible by feature dim")
    points = points.reshape(-1, args.feature_dim)

    voxel_size = np.asarray(metadata["voxel_size"], dtype=np.float32)
    coors_range = np.asarray(metadata["point_cloud_range"], dtype=np.float32)
    max_points = int(metadata["max_points_per_pillar"])
    max_voxels = 60000

    py_voxels, py_coors_zyx, py_num_points = points_to_voxel_reference(
        points, voxel_size, coors_range, max_points, max_voxels
    )
    py_coors = np.zeros((py_coors_zyx.shape[0], 4), dtype=np.int32)
    py_coors[:, 1:] = py_coors_zyx

    num_pillars = int(metadata["num_pillars"])
    cpp_voxels = np.fromfile(args.cpp_dump / "pillars.bin", dtype=np.float32).reshape(
        num_pillars, max_points, args.feature_dim
    )
    cpp_coors = np.fromfile(args.cpp_dump / "coordinates.bin", dtype=np.int32).reshape(
        num_pillars, 4
    )
    cpp_num_points = np.fromfile(args.cpp_dump / "num_points.bin", dtype=np.int32)

    print(f"python pillars: {py_voxels.shape[0]}")
    print(f"cpp pillars:    {cpp_voxels.shape[0]}")
    print(f"coordinates equal: {np.array_equal(py_coors, cpp_coors)}")
    print(f"num_points equal:  {np.array_equal(py_num_points, cpp_num_points)}")
    print(f"pillars equal:     {np.array_equal(py_voxels, cpp_voxels)}")

    if py_voxels.shape != cpp_voxels.shape:
        print(f"shape mismatch: python={py_voxels.shape}, cpp={cpp_voxels.shape}")
        return 1

    max_abs_diff = float(np.max(np.abs(py_voxels - cpp_voxels))) if py_voxels.size else 0.0
    print(f"pillar max abs diff: {max_abs_diff:.8f}")

    if not np.array_equal(py_coors, cpp_coors):
        mismatch = int(np.where(np.any(py_coors != cpp_coors, axis=1))[0][0])
        print(f"first coordinate mismatch index: {mismatch}")
        print(f"python: {py_coors[mismatch].tolist()}")
        print(f"cpp:    {cpp_coors[mismatch].tolist()}")
        return 1

    if not np.array_equal(py_num_points, cpp_num_points):
        mismatch = int(np.where(py_num_points != cpp_num_points)[0][0])
        print(f"first num_points mismatch index: {mismatch}")
        print(f"python: {int(py_num_points[mismatch])}")
        print(f"cpp:    {int(cpp_num_points[mismatch])}")
        return 1

    if not np.array_equal(py_voxels, cpp_voxels):
        mismatch = np.argwhere(py_voxels != cpp_voxels)[0]
        i, j, k = [int(v) for v in mismatch]
        print(f"first pillar mismatch index: pillar={i}, point={j}, feature={k}")
        print(f"python: {float(py_voxels[i, j, k])}")
        print(f"cpp:    {float(cpp_voxels[i, j, k])}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

