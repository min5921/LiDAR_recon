import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pfn-dump", required=True, type=Path)
    parser.add_argument("--voxel-dump", required=True, type=Path)
    parser.add_argument("--scatter-dump", required=True, type=Path)
    args = parser.parse_args()

    pfn_meta = json.loads((args.pfn_dump / "pillar_features_metadata.json").read_text())
    voxel_meta = json.loads((args.voxel_dump / "metadata.json").read_text())
    bev_meta = json.loads((args.scatter_dump / "bev_features_metadata.json").read_text())

    num_pillars = pfn_meta["num_pillars"]
    channels = pfn_meta["out_channels"]
    grid_x, grid_y, _ = voxel_meta["grid_size_xyz"]
    batch_size = bev_meta["batch_size"]
    pillar_features = np.fromfile(
        args.pfn_dump / "pillar_features.bin", dtype=np.float32
    ).reshape(num_pillars, channels)
    coordinates = np.fromfile(
        args.voxel_dump / "coordinates.bin", dtype=np.int32
    ).reshape(num_pillars, 4)
    cpp = np.fromfile(
        args.scatter_dump / "bev_features.bin", dtype=np.float32
    ).reshape(batch_size, channels, grid_y, grid_x)

    python = np.zeros_like(cpp)
    for pillar, coordinate in enumerate(coordinates):
        batch, _, y, x = [int(value) for value in coordinate]
        python[batch, :, y, x] = pillar_features[pillar]

    exact = np.array_equal(python, cpp)
    max_abs_diff = float(np.max(np.abs(python - cpp))) if python.size else 0.0
    occupied = len({(int(c[0]), int(c[2]), int(c[3])) for c in coordinates})
    print(f"python BEV shape:  {python.shape}")
    print(f"cpp BEV shape:     {cpp.shape}")
    print(f"BEV exactly equal: {exact}")
    print(f"max abs diff:      {max_abs_diff:.8f}")
    print(f"occupied cells:    {occupied}")
    return 0 if exact else 1


if __name__ == "__main__":
    sys.exit(main())
