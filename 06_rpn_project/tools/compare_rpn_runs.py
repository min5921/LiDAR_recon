import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", required=True, type=Path)
    parser.add_argument("--second", required=True, type=Path)
    args = parser.parse_args()

    first_meta = json.loads(
        (args.first / "rpn_features_metadata.json").read_text(encoding="utf-8")
    )
    second_meta = json.loads(
        (args.second / "rpn_features_metadata.json").read_text(encoding="utf-8")
    )
    first_file = args.first / "rpn_features.bin"
    second_file = args.second / "rpn_features.bin"
    first = np.memmap(first_file, dtype=np.float32, mode="r", shape=tuple(first_meta["shape"]))
    second = np.memmap(second_file, dtype=np.float32, mode="r", shape=tuple(second_meta["shape"]))

    exact = np.array_equal(first, second)
    finite = bool(np.isfinite(first).all())
    maximum_difference = float(np.max(np.abs(first - second)))
    first_hash = sha256(first_file)
    second_hash = sha256(second_file)
    print(f"shape:            {first.shape}")
    print(f"finite:           {finite}")
    print(f"exactly equal:    {exact}")
    print(f"max abs diff:     {maximum_difference:.8f}")
    print(f"first SHA-256:    {first_hash}")
    print(f"second SHA-256:   {second_hash}")
    return 0 if finite and exact and first_hash == second_hash else 1


if __name__ == "__main__":
    sys.exit(main())
