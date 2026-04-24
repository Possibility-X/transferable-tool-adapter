import argparse
import json
import re
from pathlib import Path

import torch
from safetensors.torch import load_file

from train_transfer import (
    DEFAULT_SOURCE_ADAPTER,
    DEFAULT_SOURCE_MODEL,
    build_lora_model,
    ensure_parent,
    get_split_layer,
)


DEFAULT_TARGET_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_OUTPUT = "results/cross_model_compat_qwen.json"


LAYER_RE = re.compile(r"layers\.(\d+)")


def canonical_key(name: str) -> str:
    name = name.replace(".default.", ".")
    if name.startswith("base_model.model."):
        name = name[len("base_model.model."):]
    return name


def layer_id_from_name(name: str):
    match = LAYER_RE.search(name)
    if match is None:
        return None
    return int(match.group(1))


def load_source_lora_shapes(adapter_dir: str):
    path = Path(adapter_dir) / "adapter_model.safetensors"
    if not path.exists():
        raise FileNotFoundError(f"Missing adapter weights: {path}")

    state = load_file(str(path), device="cpu")
    shapes = {}
    for name, tensor in state.items():
        if "lora_" not in name or "layers." not in name:
            continue
        shapes[canonical_key(name)] = tuple(tensor.shape)
    return shapes


def collect_target_lora_shapes(model, split_layer: int):
    shapes = {}
    for name, param in model.named_parameters():
        if "lora_" not in name or "layers." not in name:
            continue
        layer_id = layer_id_from_name(name)
        if layer_id is None or layer_id < split_layer:
            continue
        shapes[canonical_key(name)] = tuple(param.shape)
    return shapes


def compare_shapes(source_shapes, target_shapes):
    copied = []
    missing = []
    shape_mismatch = []

    for key, target_shape in target_shapes.items():
        source_shape = source_shapes.get(key)
        if source_shape is None:
            missing.append(
                {
                    "name": key,
                    "target_shape": list(target_shape),
                }
            )
        elif source_shape != target_shape:
            shape_mismatch.append(
                {
                    "name": key,
                    "source_shape": list(source_shape),
                    "target_shape": list(target_shape),
                }
            )
        else:
            copied.append(
                {
                    "name": key,
                    "shape": list(target_shape),
                }
            )

    return copied, missing, shape_mismatch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-model", type=str, default=DEFAULT_SOURCE_MODEL)
    parser.add_argument("--target-model", type=str, default=DEFAULT_TARGET_MODEL)
    parser.add_argument("--source-adapter", type=str, default=DEFAULT_SOURCE_ADAPTER)
    parser.add_argument("--split-ratio", type=float, default=0.5)
    parser.add_argument("--save", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-examples", type=int, default=10)
    args = parser.parse_args()

    source_shapes = load_source_lora_shapes(args.source_adapter)

    print(f"Loading target model with LoRA: {args.target_model}")
    target_model = build_lora_model(args.target_model)
    split_layer = get_split_layer(target_model, args.split_ratio)
    target_shapes = collect_target_lora_shapes(target_model, split_layer)

    copied, missing, shape_mismatch = compare_shapes(source_shapes, target_shapes)

    target_count = len(target_shapes)
    compatible_ratio = len(copied) / target_count if target_count else 0.0
    report = {
        "source_model": args.source_model,
        "target_model": args.target_model,
        "source_adapter": args.source_adapter,
        "split_ratio": args.split_ratio,
        "split_layer": split_layer,
        "source_late_or_all_lora_params": len(source_shapes),
        "target_late_lora_params": target_count,
        "directly_compatible_params": len(copied),
        "missing_name_params": len(missing),
        "shape_mismatch_params": len(shape_mismatch),
        "compatible_ratio": compatible_ratio,
        "copied_examples": copied[: args.max_examples],
        "missing_examples": missing[: args.max_examples],
        "shape_mismatch_examples": shape_mismatch[: args.max_examples],
    }

    print("\n=== Cross-Model LoRA Compatibility ===")
    print(json.dumps(report, indent=2))

    if args.save:
        ensure_parent(args.save)
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nSaved report to: {args.save}")

    del target_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
