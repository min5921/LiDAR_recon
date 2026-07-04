import argparse
import sys
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    raw = np.fromfile(args.input, dtype=np.float32)
    if raw.size % 4 != 0:
        raise ValueError("KITTI input does not contain four float32 features per point")
    points4 = raw.reshape(-1, 4)
    points5 = np.zeros((points4.shape[0], 5), dtype=np.float32)
    points5[:, :4] = points4
    args.output.parent.mkdir(parents=True, exist_ok=True)
    points5.tofile(args.output)

    print(f"input shape:  {points4.shape}")
    print(f"output shape: {points5.shape}")
    print("feature order: x,y,z,intensity,elongation_zero")
    print(f"output: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
