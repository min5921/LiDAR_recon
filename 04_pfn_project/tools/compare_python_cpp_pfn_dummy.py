import argparse
import json
import sys
from pathlib import Path

import numpy as np


def make_dummy_weights(in_channels, out_channels):
    weights = np.zeros((out_channels, in_channels), dtype=np.float32)
    for out in range(out_channels):
        for in_idx in range(in_channels):
            pattern = ((out + 1) * (in_idx + 3)) % 17
            weights[out, in_idx] = (np.float32(pattern) - np.float32(8.0)) * np.float32(0.01)
    return weights


def run_reference(decorated, out_channels, eps=1.0e-3):
    in_channels = decorated.shape[2]
    weights = make_dummy_weights(in_channels, out_channels)
    output = np.zeros((decorated.shape[0], out_channels), dtype=np.float32)
    for pillar in range(decorated.shape[0]):
        valid_mask = np.any(decorated[pillar] != 0.0, axis=1)
        valid = decorated[pillar, valid_mask]
        if valid.shape[0] == 0:
            continue
        linear = valid @ weights.T
        normalized = linear / np.sqrt(np.float32(1.0) + np.float32(eps))
        activated = np.maximum(normalized, np.float32(0.0))
        output[pillar] = activated.max(axis=0)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--decorated-dump", required=True, type=Path)
    parser.add_argument("--pfn-dump", required=True, type=Path)
    args = parser.parse_args()

    decorated_meta = json.loads(
        (args.decorated_dump / "decorated_metadata.json").read_text(encoding="utf-8")
    )
    pfn_meta = json.loads(
        (args.pfn_dump / "pillar_features_metadata.json").read_text(encoding="utf-8")
    )

    num_pillars = decorated_meta["num_pillars"]
    max_points = decorated_meta["max_points_per_pillar"]
    in_channels = decorated_meta["decorated_feature_dim"]
    out_channels = pfn_meta["out_channels"]

    decorated = np.fromfile(
        args.decorated_dump / "decorated_pillars.bin", dtype=np.float32
    ).reshape(num_pillars, max_points, in_channels)
    cpp = np.fromfile(args.pfn_dump / "pillar_features.bin", dtype=np.float32).reshape(
        num_pillars, out_channels
    )
    py = run_reference(decorated, out_channels)

    max_abs_diff = float(np.max(np.abs(py - cpp))) if py.size else 0.0
    close = np.allclose(py, cpp, rtol=1.0e-6, atol=1.0e-6)

    print(f"python PFN shape: {py.shape}")
    print(f"cpp PFN shape:    {cpp.shape}")
    print(f"PFN allclose:     {close}")
    print(f"max abs diff:     {max_abs_diff:.8f}")

    if not close:
        mismatch = np.argwhere(np.abs(py - cpp) > 1.0e-6)[0]
        i, j = [int(v) for v in mismatch]
        print(f"first mismatch: pillar={i}, channel={j}")
        print(f"python: {float(py[i, j])}")
        print(f"cpp:    {float(cpp[i, j])}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

