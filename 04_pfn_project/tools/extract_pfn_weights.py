import argparse
import collections
import io
import json
import pickle
import sys
import zipfile
from pathlib import Path

import numpy as np


class StorageReference:
    def __init__(self, persistent_id):
        self.persistent_id = persistent_id


def rebuild_tensor(storage, storage_offset, size, stride, *unused):
    return {
        "storage": storage,
        "storage_offset": int(storage_offset),
        "shape": tuple(int(value) for value in size),
        "stride": tuple(int(value) for value in stride),
    }


class CheckpointUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "collections" and name == "OrderedDict":
            return collections.OrderedDict
        if module == "torch._utils" and name.startswith("_rebuild_tensor"):
            return rebuild_tensor
        if module.startswith("torch") and name.endswith("Storage"):
            return type(name, (), {"storage_name": name})
        return super().find_class(module, name)

    def persistent_load(self, persistent_id):
        return StorageReference(persistent_id)


TENSORS = {
    "reader.pfn_layers.0.linear.weight": "layer0_linear_weight.bin",
    "reader.pfn_layers.0.norm.weight": "layer0_bn_weight.bin",
    "reader.pfn_layers.0.norm.bias": "layer0_bn_bias.bin",
    "reader.pfn_layers.0.norm.running_mean": "layer0_bn_mean.bin",
    "reader.pfn_layers.0.norm.running_var": "layer0_bn_var.bin",
    "reader.pfn_layers.1.linear.weight": "layer1_linear_weight.bin",
    "reader.pfn_layers.1.norm.weight": "layer1_bn_weight.bin",
    "reader.pfn_layers.1.norm.bias": "layer1_bn_bias.bin",
    "reader.pfn_layers.1.norm.running_mean": "layer1_bn_mean.bin",
    "reader.pfn_layers.1.norm.running_var": "layer1_bn_var.bin",
}


def load_tensor(archive, archive_root, tensor):
    persistent_id = tensor["storage"].persistent_id
    if persistent_id[0] != "storage":
        raise ValueError(f"unsupported persistent id: {persistent_id[0]}")
    storage_type = persistent_id[1].storage_name
    if storage_type != "FloatStorage":
        raise ValueError(f"unsupported storage type: {storage_type}")

    storage_key = persistent_id[2]
    raw = archive.read(f"{archive_root}/data/{storage_key}")
    item_size = np.dtype("<f4").itemsize
    array = np.ndarray(
        shape=tensor["shape"],
        dtype="<f4",
        buffer=raw,
        offset=tensor["storage_offset"] * item_size,
        strides=tuple(value * item_size for value in tensor["stride"]),
    )
    return np.ascontiguousarray(array, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    if not zipfile.is_zipfile(args.checkpoint):
        raise ValueError("checkpoint is not a PyTorch ZIP archive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    exported = {}
    with zipfile.ZipFile(args.checkpoint) as archive:
        data_name = next(
            name for name in archive.namelist() if name.endswith("data.pkl")
        )
        archive_root = data_name.split("/")[0]
        checkpoint = CheckpointUnpickler(
            io.BytesIO(archive.read(data_name))
        ).load()
        state_dict = checkpoint["state_dict"]

        for tensor_name, file_name in TENSORS.items():
            if tensor_name not in state_dict:
                raise KeyError(f"missing checkpoint tensor: {tensor_name}")
            values = load_tensor(
                archive, archive_root, state_dict[tensor_name]
            )
            values.tofile(args.output_dir / file_name)
            exported[tensor_name] = {
                "file": file_name,
                "shape": list(values.shape),
                "dtype": "float32",
            }

    expected_shapes = {
        "reader.pfn_layers.0.linear.weight": [32, 10],
        "reader.pfn_layers.1.linear.weight": [64, 64],
    }
    for name, shape in expected_shapes.items():
        if exported[name]["shape"] != shape:
            raise ValueError(
                f"unexpected shape for {name}: {exported[name]['shape']}"
            )

    metadata = {
        "source_checkpoint": str(args.checkpoint.resolve()),
        "batch_norm_eps": 0.001,
        "layer0_in_channels": 10,
        "layer0_out_channels": 32,
        "layer1_in_channels": 64,
        "layer1_out_channels": 64,
        "tensors": exported,
    }
    (args.output_dir / "weights_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print(f"exported tensors: {len(exported)}")
    print("layer 0: [32, 10]")
    print("layer 1: [64, 64]")
    print(f"output: {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
