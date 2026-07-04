import argparse
import json
import sys
from pathlib import Path

import numpy as np


def read_layer(weight_dir, prefix, in_channels, out_channels):
    return {
        "linear": np.fromfile(
            weight_dir / f"{prefix}_linear_weight.bin", np.float32
        ).reshape(out_channels, in_channels),
        "weight": np.fromfile(weight_dir / f"{prefix}_bn_weight.bin", np.float32),
        "bias": np.fromfile(weight_dir / f"{prefix}_bn_bias.bin", np.float32),
        "mean": np.fromfile(weight_dir / f"{prefix}_bn_mean.bin", np.float32),
        "var": np.fromfile(weight_dir / f"{prefix}_bn_var.bin", np.float32),
    }


def batch_norm_relu(values, layer, eps):
    scale = layer["weight"] / np.sqrt(layer["var"] + np.float32(eps))
    values = (values - layer["mean"]) * scale + layer["bias"]
    return np.maximum(values, np.float32(0.0)).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--decorated-dump", required=True, type=Path)
    parser.add_argument("--weight-dir", required=True, type=Path)
    parser.add_argument("--pfn-dump", required=True, type=Path)
    args = parser.parse_args()

    decorated_meta = json.loads(
        (args.decorated_dump / "decorated_metadata.json").read_text(encoding="utf-8")
    )
    weight_meta = json.loads(
        (args.weight_dir / "weights_metadata.json").read_text(encoding="utf-8")
    )
    pfn_meta = json.loads(
        (args.pfn_dump / "pillar_features_metadata.json").read_text(encoding="utf-8")
    )

    shape = (
        decorated_meta["num_pillars"],
        decorated_meta["max_points_per_pillar"],
        decorated_meta["decorated_feature_dim"],
    )
    decorated = np.fromfile(
        args.decorated_dump / "decorated_pillars.bin", np.float32
    ).reshape(shape)
    layer0 = read_layer(args.weight_dir, "layer0", 10, 32)
    layer1 = read_layer(args.weight_dir, "layer1", 64, 64)
    eps = weight_meta["batch_norm_eps"]

    local = decorated @ layer0["linear"].T
    local = batch_norm_relu(local, layer0, eps)
    local_max = local.max(axis=1, keepdims=True)
    concatenated = np.concatenate(
        [local, np.repeat(local_max, local.shape[1], axis=1)], axis=2
    )
    final = concatenated @ layer1["linear"].T
    final = batch_norm_relu(final, layer1, eps)
    python = final.max(axis=1)

    cpp = np.fromfile(
        args.pfn_dump / "pillar_features.bin", np.float32
    ).reshape(pfn_meta["num_pillars"], pfn_meta["out_channels"])

    difference = np.abs(python - cpp)
    max_abs_diff = float(difference.max())
    mean_abs_diff = float(difference.mean())
    close = np.allclose(python, cpp, rtol=1.0e-5, atol=2.0e-5)
    print(f"decorated shape: {decorated.shape}")
    print(f"python PFN:      {python.shape}")
    print(f"C++ PFN:         {cpp.shape}")
    print(f"allclose:        {close}")
    print(f"max abs diff:    {max_abs_diff:.8f}")
    print(f"mean abs diff:   {mean_abs_diff:.8f}")
    print(f"python first 8:  {python[0, :8]}")
    print(f"C++ first 8:     {cpp[0, :8]}")
    return 0 if close else 1


if __name__ == "__main__":
    sys.exit(main())
