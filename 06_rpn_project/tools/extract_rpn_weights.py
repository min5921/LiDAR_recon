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


def load_tensor(archive, archive_root, tensor):
    persistent_id = tensor["storage"].persistent_id
    storage_type = persistent_id[1].storage_name
    if storage_type != "FloatStorage":
        raise ValueError(f"unsupported storage type: {storage_type}")
    raw = archive.read(f"{archive_root}/data/{persistent_id[2]}")
    item_size = np.dtype("<f4").itemsize
    values = np.ndarray(
        shape=tensor["shape"],
        dtype="<f4",
        buffer=raw,
        offset=tensor["storage_offset"] * item_size,
        strides=tuple(value * item_size for value in tensor["stride"]),
    )
    return np.ascontiguousarray(values, dtype=np.float32)


def layer_specs():
    specs = []
    block_indices = [
        ([1, 4, 7, 10], [2, 5, 8, 11]),
        ([1, 4, 7, 10, 13, 16], [2, 5, 8, 11, 14, 17]),
        ([1, 4, 7, 10, 13, 16], [2, 5, 8, 11, 14, 17]),
    ]
    for block, (conv_indices, bn_indices) in enumerate(block_indices):
        for layer, (conv_index, bn_index) in enumerate(zip(conv_indices, bn_indices)):
            specs.append(
                {
                    "prefix": f"block{block}_conv{layer}",
                    "weight": f"neck.blocks.{block}.{conv_index}.weight",
                    "bn": f"neck.blocks.{block}.{bn_index}",
                    "transposed": False,
                }
            )
    for index in range(3):
        specs.append(
            {
                "prefix": f"deblock{index}",
                "weight": f"neck.deblocks.{index}.0.weight",
                "bn": f"neck.deblocks.{index}.1",
                "transposed": index > 0,
            }
        )
    return specs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    if not zipfile.is_zipfile(args.checkpoint):
        raise ValueError("checkpoint is not a PyTorch ZIP archive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {"source_checkpoint": str(args.checkpoint.resolve()), "layers": []}
    with zipfile.ZipFile(args.checkpoint) as archive:
        data_name = next(name for name in archive.namelist() if name.endswith("data.pkl"))
        archive_root = data_name.split("/")[0]
        checkpoint = CheckpointUnpickler(
            io.BytesIO(archive.read(data_name))
        ).load()
        state = checkpoint["state_dict"]

        for spec in layer_specs():
            prefix = spec["prefix"]
            weight = load_tensor(archive, archive_root, state[spec["weight"]])
            original_shape = list(weight.shape)
            if spec["transposed"]:
                # PyTorch ConvTranspose2d: [in, out, ky, kx]
                # GEMM runtime: [out*ky*kx, in]
                weight = np.transpose(weight, (1, 2, 3, 0)).reshape(-1, weight.shape[0])
                weight_file = f"{prefix}_weight_gemm.bin"
            else:
                weight_file = f"{prefix}_weight.bin"
            np.ascontiguousarray(weight, dtype=np.float32).tofile(
                args.output_dir / weight_file
            )

            bn_files = {}
            for source_suffix, output_suffix in [
                ("weight", "weight"),
                ("bias", "bias"),
                ("running_mean", "mean"),
                ("running_var", "var"),
            ]:
                tensor_name = f"{spec['bn']}.{source_suffix}"
                values = load_tensor(archive, archive_root, state[tensor_name])
                file_name = f"{prefix}_bn_{output_suffix}.bin"
                values.tofile(args.output_dir / file_name)
                bn_files[output_suffix] = file_name

            manifest["layers"].append(
                {
                    "prefix": prefix,
                    "checkpoint_weight": spec["weight"],
                    "original_shape": original_shape,
                    "runtime_weight_shape": list(weight.shape),
                    "weight_file": weight_file,
                    "batch_norm_files": bn_files,
                    "transposed": spec["transposed"],
                }
            )

    (args.output_dir / "rpn_weights_metadata.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"exported layers: {len(manifest['layers'])}")
    print("float tensors: 95")
    print(f"output: {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
