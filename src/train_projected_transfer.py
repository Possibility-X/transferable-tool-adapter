import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import Trainer, TrainingArguments

from train_transfer import (
    DEFAULT_SOURCE_ADAPTER,
    DEFAULT_SOURCE_MODEL,
    DEFAULT_MAX_LEN,
    DEFAULT_SEED,
    DEFAULT_SPLIT_RATIO,
    DEFAULT_TRAIN_SAMPLES,
    build_lora_model,
    build_tokenizer,
    build_training_dataset,
    ensure_dir,
    ensure_parent,
    freeze_tool_layers,
    get_split_layer,
    print_trainable_summary,
    set_seed,
    use_fp16_training,
    verify_late_layers_frozen,
    write_summary,
)


DEFAULT_TARGET_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

DEFAULT_FULL_ADAPTER_DIR = "adapters/adapter_qwen_projected_full"
DEFAULT_FREEZE_ADAPTER_DIR = "adapters/adapter_qwen_projected_freeze"
DEFAULT_AONLY_ADAPTER_DIR = "adapters/adapter_qwen_projected_Aonly"
DEFAULT_ALINEAR_ADAPTER_DIR = "adapters/adapter_qwen_projected_Alinear"
DEFAULT_AMLP_ADAPTER_DIR = "adapters/adapter_qwen_projected_Amlp"

DEFAULT_FULL_TRAIN_OUTPUT = "outputs/qwen_projected_full"
DEFAULT_FREEZE_TRAIN_OUTPUT = "outputs/qwen_projected_freeze"
DEFAULT_AONLY_TRAIN_OUTPUT = "outputs/qwen_projected_Aonly"
DEFAULT_ALINEAR_TRAIN_OUTPUT = "outputs/qwen_projected_Alinear"
DEFAULT_AMLP_TRAIN_OUTPUT = "outputs/qwen_projected_Amlp"

DEFAULT_FULL_SUMMARY = "results/qwen_projected_full_train.json"
DEFAULT_FREEZE_SUMMARY = "results/qwen_projected_freeze_train.json"
DEFAULT_AONLY_SUMMARY = "results/qwen_projected_Aonly_train.json"
DEFAULT_ALINEAR_SUMMARY = "results/qwen_projected_Alinear_train.json"
DEFAULT_AMLP_SUMMARY = "results/qwen_projected_Amlp_train.json"

DEFAULT_PROJECTION_LOG_DIR = "results/analysis"

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


def suffix_from_layer_name(name: str):
    match = LAYER_RE.search(name)
    if match is None:
        return None
    return name[match.end():]


def resize_tensor(src, target_shape):
    if tuple(src.shape) == tuple(target_shape):
        return src

    src = src.float()

    if src.ndim == 2:
        x = src.unsqueeze(0).unsqueeze(0)
        y = F.interpolate(
            x,
            size=target_shape,
            mode="bilinear",
            align_corners=False,
        )
        return y.squeeze(0).squeeze(0)

    raise ValueError(f"Unsupported tensor shape: {src.shape}")


def safe_experiment_id(value: str):
    value = Path(value).stem if value else "projection"
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return value or "projection"


def infer_projection_log_prefix(projection_log_prefix: str | None, summary_path: str, adapter_dir: str):
    if projection_log_prefix:
        return safe_experiment_id(projection_log_prefix)
    if summary_path:
        stem = Path(summary_path).stem
        if stem.endswith("_train"):
            stem = stem[: -len("_train")]
        return safe_experiment_id(stem)
    return safe_experiment_id(Path(adapter_dir).name)


def detach_cpu(tensor: torch.Tensor | None):
    if tensor is None:
        return None
    return tensor.detach().cpu()


def tensor_shape(tensor: torch.Tensor | None):
    if tensor is None:
        return None
    return list(tensor.shape)


def tensor_norm(tensor: torch.Tensor | None):
    if tensor is None:
        return None
    return float(tensor.detach().float().norm().cpu().item())


def cosine_if_shape_compatible(left: torch.Tensor | None, right: torch.Tensor | None):
    if left is None or right is None:
        return None
    if tuple(left.shape) != tuple(right.shape) or left.numel() == 0:
        return None
    try:
        left_flat = left.detach().float().reshape(1, -1)
        right_flat = right.detach().float().reshape(1, -1)
        return float(F.cosine_similarity(left_flat, right_flat).cpu().item())
    except Exception:
        return None


def projection_w_payload(projection: nn.Module | None):
    if projection is None:
        return None, None
    if isinstance(projection, LinearAProjection):
        weight = detach_cpu(projection.proj.weight)
        return tensor_shape(weight), weight
    if isinstance(projection, MLPAProjection):
        weights = {}
        shapes = {}
        for name, tensor in projection.state_dict().items():
            if name.startswith("net.") and name.endswith("weight"):
                weights[name] = detach_cpu(tensor)
                shapes[name] = tensor_shape(tensor)
        return shapes or None, weights or None
    return None, None


class ProjectionInitLogger:
    def __init__(
        self,
        log_dir: str,
        experiment_id: str,
        metadata: dict,
    ):
        self.log_dir = Path(log_dir)
        self.experiment_id = safe_experiment_id(experiment_id)
        self.metadata = dict(metadata)
        self.layer_stats = []
        self.layer_tensors = []
        self.warnings = []

    def add_warning(self, layer_name: str, reason: str):
        self.warnings.append({"layer_name": layer_name, "reason": reason})

    def add_layer(
        self,
        layer_name: str,
        source_A: torch.Tensor | None,
        resized_A: torch.Tensor | None,
        projected_A: torch.Tensor | None,
        source_layer: int | None = None,
        target_layer: int | None = None,
        target_shape=None,
        W_shape=None,
        W=None,
        projection_mode: str | None = None,
        fallback_reason: str | None = None,
    ):
        try:
            stats = {
                "layer_name": layer_name,
                "source_layer": source_layer,
                "target_layer": target_layer,
                "source_A_shape": tensor_shape(source_A),
                "resized_A_shape": tensor_shape(resized_A),
                "projected_A_shape": tensor_shape(projected_A),
                "target_A_shape": target_shape,
                "W_shape": W_shape,
                "source_A_norm": tensor_norm(source_A),
                "resized_A_norm": tensor_norm(resized_A),
                "projected_A_norm": tensor_norm(projected_A),
                "resize_to_project_cosine": cosine_if_shape_compatible(resized_A, projected_A),
                "source_to_resized_cosine": cosine_if_shape_compatible(source_A, resized_A),
                "projection_mode": projection_mode,
                "W_exists": W is not None,
                "fallback_reason": fallback_reason,
            }
            self.layer_stats.append(stats)
            self.layer_tensors.append(
                {
                    "layer_name": layer_name,
                    "source_layer": source_layer,
                    "target_layer": target_layer,
                    "source_A": detach_cpu(source_A),
                    "resized_A": detach_cpu(resized_A),
                    "projected_A": detach_cpu(projected_A),
                    "W": W,
                }
            )
        except Exception as exc:
            self.add_warning(layer_name, f"logging_failed: {type(exc).__name__}: {exc}")

    def save(self):
        self.log_dir.mkdir(parents=True, exist_ok=True)
        json_path = self.log_dir / f"projection_init_{self.experiment_id}.json"
        pt_path = self.log_dir / f"projection_init_{self.experiment_id}.pt"

        metadata = dict(self.metadata)
        metadata.update(
            {
                "experiment_id": self.experiment_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "number_of_logged_layers": len(self.layer_stats),
                "warnings": self.warnings,
            }
        )

        with json_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "metadata": metadata,
                    "layers": self.layer_stats,
                },
                f,
                indent=2,
            )

        torch.save(
            {
                "metadata": metadata,
                "layers": self.layer_tensors,
            },
            pt_path,
        )

        return {
            "projection_init_json": str(json_path),
            "projection_init_pt": str(pt_path),
            "projection_init_logged_layers": len(self.layer_stats),
            "projection_init_warnings": self.warnings,
        }


class LinearAProjection(nn.Module):
    def __init__(self, source_weight: torch.Tensor, target_dim: int):
        super().__init__()
        self.register_buffer("source_weight", source_weight.float())
        self.proj = nn.Linear(source_weight.shape[-1], target_dim, bias=False)

    def forward(self, original):
        source = self.source_weight.to(device=original.device, dtype=self.proj.weight.dtype)
        projected = self.proj(source)
        return projected.to(dtype=original.dtype)


class MLPAProjection(nn.Module):
    def __init__(
        self,
        source_weight: torch.Tensor,
        target_dim: int,
        hidden_dim: int | None = None,
    ):
        super().__init__()
        self.register_buffer("source_weight", source_weight.float())
        hidden_dim = hidden_dim or target_dim
        self.net = nn.Sequential(
            nn.Linear(source_weight.shape[-1], hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim, target_dim, bias=False),
        )

    def forward(self, original):
        first_layer = self.net[0]
        source = self.source_weight.to(
            device=original.device,
            dtype=first_layer.weight.dtype,
        )
        projected = self.net(source)
        return projected.to(dtype=original.dtype)


def count_trainable_params(module: nn.Module):
    return sum(param.numel() for param in module.parameters() if param.requires_grad)


def load_source_lora_by_layer(source_adapter: str):
    path = Path(source_adapter) / "adapter_model.safetensors"
    if not path.exists():
        raise FileNotFoundError(f"Missing adapter weights: {path}")

    state = load_file(str(path), device="cpu")
    by_layer = {}
    source_layers = set()

    for name, tensor in state.items():
        key = canonical_key(name)
        if "lora_" not in key or "layers." not in key:
            continue

        layer_id = layer_id_from_name(key)
        suffix = suffix_from_layer_name(key)
        if layer_id is None or suffix is None:
            continue

        by_layer[(layer_id, suffix)] = tensor
        source_layers.add(layer_id)

    if not by_layer:
        raise ValueError(f"No layerwise LoRA weights found in {path}")

    return by_layer, sorted(source_layers)


def collect_target_lora_layers(model):
    layers = set()
    for name, _ in model.named_parameters():
        key = canonical_key(name)
        if "lora_" not in key or "layers." not in key:
            continue

        layer_id = layer_id_from_name(key)
        if layer_id is not None:
            layers.add(layer_id)

    return sorted(layers)


def build_layer_map(source_layers, target_layers, source_split_layer, target_split_layer):
    source_late = [layer for layer in source_layers if layer >= source_split_layer]
    target_late = [layer for layer in target_layers if layer >= target_split_layer]

    if not source_late:
        raise ValueError("No source late layers available for projection")
    if not target_late:
        raise ValueError("No target late layers available for projection")

    layer_map = {}
    for i, target_layer in enumerate(target_late):
        if len(target_late) == 1 or len(source_late) == 1:
            source_idx = 0
        else:
            source_idx = round(i * (len(source_late) - 1) / (len(target_late) - 1))
        layer_map[target_layer] = source_late[source_idx]

    return layer_map, source_late, target_late


def initialize_projected_late_layers(
    model,
    source_adapter: str,
    split_ratio: float,
    target_split_layer: int,
    projection_scope: str,
    projection_mode: str,
    projection_hidden_dim: int | None = None,
    max_examples: int = 10,
    projection_logger: ProjectionInitLogger | None = None,
):
    learned_projection_modes = {"linear", "mlp"}
    if projection_mode in learned_projection_modes and projection_scope != "lora_A":
        raise ValueError(
            f"--projection-mode {projection_mode} is only supported with --projection-scope lora_A"
        )

    source_by_layer, source_layers = load_source_lora_by_layer(source_adapter)
    target_layers = collect_target_lora_layers(model)

    source_split_layer = int((max(source_layers) + 1) * split_ratio)
    layer_map, source_late, target_late = build_layer_map(
        source_layers=source_layers,
        target_layers=target_layers,
        source_split_layer=source_split_layer,
        target_split_layer=target_split_layer,
    )

    projected = 0
    resized = 0
    direct_shape = 0
    missing = 0
    skipped = 0
    skipped_by_scope = 0
    linear_projected = 0
    linear_projection_params = 0
    linear_projection_init_params = 0
    mlp_projected = 0
    mlp_projection_params = 0
    mlp_projection_init_params = 0
    learned_projected = 0
    learned_projection_params = 0
    learned_projection_init_params = 0
    materialized_projection_layers = 0
    examples = []
    target_named_params = list(model.named_parameters())

    for name, target_param in target_named_params:
        target_key = canonical_key(name)
        if "lora_" not in target_key or "layers." not in target_key:
            continue

        target_layer = layer_id_from_name(target_key)
        suffix = suffix_from_layer_name(target_key)
        if target_layer is None or suffix is None:
            skipped += 1
            continue
        if target_layer < target_split_layer:
            continue

        if projection_scope == "lora_A" and "lora_A" not in target_key:
            skipped_by_scope += 1
            continue

        source_layer = layer_map[target_layer]
        source_tensor = source_by_layer.get((source_layer, suffix))
        if source_tensor is None:
            missing += 1
            if projection_logger is not None and "lora_A" in target_key:
                projection_logger.add_layer(
                    layer_name=target_key,
                    source_A=None,
                    resized_A=None,
                    projected_A=None,
                    source_layer=source_layer,
                    target_layer=target_layer,
                    target_shape=list(target_param.shape),
                    projection_mode=projection_mode,
                    fallback_reason="missing_source_tensor",
                )
            continue

        original_source_tensor = source_tensor
        resized_source_tensor = None
        projected_tensor_for_log = None
        projection_for_log = None
        fallback_reason = None

        if projection_mode in learned_projection_modes and "lora_A" in target_key:
            if source_tensor.ndim != 2 or target_param.ndim != 2:
                raise ValueError(
                    f"{projection_mode} projection only supports 2D LoRA A tensors: {target_key}"
                )

            if source_tensor.shape[0] != target_param.shape[0]:
                source_tensor = resize_tensor(
                    source_tensor,
                    (target_param.shape[0], source_tensor.shape[1]),
                )
                resized_source_tensor = source_tensor
            else:
                resized_source_tensor = source_tensor

            if projection_mode == "linear":
                projection = LinearAProjection(
                    source_weight=source_tensor,
                    target_dim=target_param.shape[1],
                )
                linear_projected += 1
                linear_projection_init_params += count_trainable_params(projection)
            else:
                projection = MLPAProjection(
                    source_weight=source_tensor,
                    target_dim=target_param.shape[1],
                    hidden_dim=projection_hidden_dim,
                )
                mlp_projected += 1
                mlp_projection_init_params += count_trainable_params(projection)

            projection = projection.to(device=target_param.device, dtype=target_param.dtype)
            projection_for_log = projection
            with torch.no_grad():
                projected_tensor = projection(target_param)
                target_param.data.copy_(projected_tensor)
                projected_tensor_for_log = projected_tensor.detach()

            materialized_projection_layers += 1
            learned_projected += 1
            learned_projection_init_params += count_trainable_params(projection)
        else:
            with torch.no_grad():
                projected_tensor = resize_tensor(source_tensor, tuple(target_param.shape))
                target_param.data.copy_(
                    projected_tensor.to(device=target_param.device, dtype=target_param.dtype)
                )
            if "lora_A" in target_key:
                resized_source_tensor = projected_tensor
                projected_tensor_for_log = projected_tensor

        if projection_logger is not None and "lora_A" in target_key:
            try:
                W_shape, W = projection_w_payload(projection_for_log)
                projection_logger.add_layer(
                    layer_name=target_key,
                    source_A=original_source_tensor,
                    resized_A=resized_source_tensor,
                    projected_A=projected_tensor_for_log,
                    source_layer=source_layer,
                    target_layer=target_layer,
                    target_shape=list(target_param.shape),
                    W_shape=W_shape,
                    W=W,
                    projection_mode=projection_mode,
                    fallback_reason=fallback_reason,
                )
            except Exception as exc:
                projection_logger.add_warning(
                    target_key,
                    f"layer_logging_failed: {type(exc).__name__}: {exc}",
                )

        projected += 1
        if projection_mode in learned_projection_modes:
            resized += 1
        elif tuple(source_tensor.shape) == tuple(target_param.shape):
            direct_shape += 1
        else:
            resized += 1

        if len(examples) < max_examples:
            examples.append(
                {
                    "target": target_key,
                    "projection_mode": projection_mode,
                    "source_layer": source_layer,
                    "target_layer": target_layer,
                    "source_shape": list(source_tensor.shape),
                    "target_shape": list(target_param.shape),
                }
            )

    target_late_params = sum(
        1
        for name, _ in target_named_params
        if "lora_" in canonical_key(name)
        and "layers." in canonical_key(name)
        and layer_id_from_name(canonical_key(name)) is not None
        and layer_id_from_name(canonical_key(name)) >= target_split_layer
    )

    report = {
        "projection_method": (
            "linear_lora_A"
            if projection_mode == "linear"
            else
            "mlp_lora_A"
            if projection_mode == "mlp"
            else
            "bilinear_resize_lora_A_only"
            if projection_scope == "lora_A"
            else "bilinear_resize"
        ),
        "projection_scope": projection_scope,
        "projection_mode": projection_mode,
        "split_ratio": split_ratio,
        "source_split_layer": source_split_layer,
        "target_split_layer": target_split_layer,
        "source_late_layers": source_late,
        "target_late_layers": target_late,
        "layer_map": {str(k): v for k, v in layer_map.items()},
        "target_late_params": target_late_params,
        "projected_late_params": projected,
        "resized_late_params": resized,
        "direct_shape_late_params": direct_shape,
        "missing_source_params": missing,
        "skipped_by_projection_scope": skipped_by_scope,
        "skipped_params": skipped,
        "projection_hidden_dim": projection_hidden_dim,
        "learned_projected_lora_A_params": learned_projected,
        "learned_projection_trainable_params": learned_projection_params,
        "learned_projection_init_params": learned_projection_init_params,
        "linear_projected_lora_A_params": linear_projected,
        "linear_projection_trainable_params": linear_projection_params,
        "linear_projection_init_params": linear_projection_init_params,
        "mlp_projected_lora_A_params": mlp_projected,
        "mlp_projection_trainable_params": mlp_projection_params,
        "mlp_projection_init_params": mlp_projection_init_params,
        "materialized_projection_layers": materialized_projection_layers,
        "projection_trainable_params": 0,
        "projection_examples": examples,
    }

    print("\n=== Projection Init Report ===")
    print(json.dumps(report, indent=2))
    if projection_logger is not None:
        log_report = projection_logger.save()
        report.update(log_report)
        print("\n=== Projection Init Log ===")
        print(json.dumps(log_report, indent=2))
    return report


def resolve_outputs(
    mode: str,
    projection_scope: str,
    projection_mode: str,
    adapter_dir: str,
    train_output_dir: str,
    summary_path: str,
):
    is_freeze = mode == "freeze_late"
    is_aonly = mode == "full" and projection_scope == "lora_A" and projection_mode == "resize"
    is_alinear = mode == "full" and projection_scope == "lora_A" and projection_mode == "linear"
    is_amlp = mode == "full" and projection_scope == "lora_A" and projection_mode == "mlp"
    return (
        adapter_dir
        or (
            DEFAULT_AMLP_ADAPTER_DIR
            if is_amlp
            else DEFAULT_ALINEAR_ADAPTER_DIR
            if is_alinear
            else DEFAULT_AONLY_ADAPTER_DIR
            if is_aonly
            else DEFAULT_FREEZE_ADAPTER_DIR
            if is_freeze
            else DEFAULT_FULL_ADAPTER_DIR
        ),
        train_output_dir
        or (
            DEFAULT_AMLP_TRAIN_OUTPUT
            if is_amlp
            else DEFAULT_ALINEAR_TRAIN_OUTPUT
            if is_alinear
            else DEFAULT_AONLY_TRAIN_OUTPUT
            if is_aonly
            else DEFAULT_FREEZE_TRAIN_OUTPUT
            if is_freeze
            else DEFAULT_FULL_TRAIN_OUTPUT
        ),
        summary_path
        or (
            DEFAULT_AMLP_SUMMARY
            if is_amlp
            else DEFAULT_ALINEAR_SUMMARY
            if is_alinear
            else DEFAULT_AONLY_SUMMARY
            if is_aonly
            else DEFAULT_FREEZE_SUMMARY
            if is_freeze
            else DEFAULT_FULL_SUMMARY
        ),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        choices=["full", "freeze_late"],
        default="full",
        help="full: projected init + train all LoRA; freeze_late: freeze projected late layers",
    )
    parser.add_argument("--source-model", type=str, default=DEFAULT_SOURCE_MODEL)
    parser.add_argument("--target-model", type=str, default=DEFAULT_TARGET_MODEL)
    parser.add_argument("--source-adapter", type=str, default=DEFAULT_SOURCE_ADAPTER)
    parser.add_argument("--max-len", type=int, default=DEFAULT_MAX_LEN)
    parser.add_argument("--split-ratio", type=float, default=DEFAULT_SPLIT_RATIO)
    parser.add_argument(
        "--projection-scope",
        type=str,
        choices=["all", "lora_A"],
        default="all",
        help="all: resize/copy LoRA A and B; lora_A: resize/copy only LoRA A and keep B initialized by PEFT",
    )
    parser.add_argument(
        "--projection-mode",
        type=str,
        choices=["resize", "linear", "mlp"],
        default="resize",
        help=(
            "resize: copy resized source weights; "
            "linear: train W so late LoRA A = W(A_source); "
            "mlp: train a two-layer MLP for late LoRA A alignment"
        ),
    )
    parser.add_argument(
        "--projection-hidden-dim",
        type=int,
        default=None,
        help="Hidden size for --projection-mode mlp; defaults to each target LoRA A input dim.",
    )
    parser.add_argument("--train-samples", type=int, default=DEFAULT_TRAIN_SAMPLES)
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Optional JSONL records with instruction and gt/tool/arguments fields.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--adapter-dir", type=str, default=None)
    parser.add_argument(
        "--train-output-dir",
        "--output-dir",
        dest="train_output_dir",
        type=str,
        default=None,
    )
    parser.add_argument("--summary-path", type=str, default=None)
    parser.add_argument(
        "--projection-log-dir",
        type=str,
        default=DEFAULT_PROJECTION_LOG_DIR,
        help="Directory for optional projection initialization logs.",
    )
    parser.add_argument(
        "--projection-log-prefix",
        type=str,
        default=None,
        help="Optional experiment id prefix for projection init logs.",
    )
    parser.add_argument(
        "--save-projection-init",
        action="store_true",
        help="Save projection initialization JSON/PT diagnostics before training.",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    adapter_dir, train_output_dir, summary_path = resolve_outputs(
        mode=args.mode,
        projection_scope=args.projection_scope,
        projection_mode=args.projection_mode,
        adapter_dir=args.adapter_dir,
        train_output_dir=args.train_output_dir,
        summary_path=args.summary_path,
    )

    ensure_dir(adapter_dir)
    ensure_dir(train_output_dir)
    ensure_parent(summary_path)

    tokenizer = build_tokenizer(args.target_model)
    training_dtype = torch.float16 if use_fp16_training(args.target_model) else torch.float32
    training_device_map = {"": 0} if training_dtype == torch.float16 else "auto"
    model = build_lora_model(
        args.target_model,
        dtype=training_dtype,
        device_map=training_device_map,
    )

    target_split_layer = get_split_layer(model, args.split_ratio)
    print("Target split layer =", target_split_layer)

    projection_logger = None
    projection_log_experiment_id = None
    if args.save_projection_init:
        projection_log_experiment_id = infer_projection_log_prefix(
            projection_log_prefix=args.projection_log_prefix,
            summary_path=summary_path,
            adapter_dir=adapter_dir,
        )
        projection_logger = ProjectionInitLogger(
            log_dir=args.projection_log_dir,
            experiment_id=projection_log_experiment_id,
            metadata={
                "projection_mode": args.projection_mode,
                "projection_scope": args.projection_scope,
                "source_model": args.source_model,
                "target_model": args.target_model,
                "source_adapter": args.source_adapter,
                "target_adapter_dir": adapter_dir,
                "dataset_path": args.dataset_path,
                "seed": args.seed,
            },
        )

    projection_report = initialize_projected_late_layers(
        model=model,
        source_adapter=args.source_adapter,
        split_ratio=args.split_ratio,
        target_split_layer=target_split_layer,
        projection_scope=args.projection_scope,
        projection_mode=args.projection_mode,
        projection_hidden_dim=args.projection_hidden_dim,
        projection_logger=projection_logger,
    )

    if args.mode == "freeze_late":
        freeze_tool_layers(model, target_split_layer)
        print("Mode: projected init + freeze late layers")
        verify_late_layers_frozen(model, target_split_layer)
    else:
        print("Mode: projected init + full LoRA training")

    trainable, total, pct = print_trainable_summary(model)

    dataset = build_training_dataset(
        tokenizer=tokenizer,
        train_samples=args.train_samples,
        max_len=args.max_len,
        dataset_path=args.dataset_path,
        seed=args.seed,
    )

    training_args = TrainingArguments(
        output_dir=train_output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=2,
        learning_rate=2e-4,
        logging_steps=20,
        save_steps=200,
        fp16=training_dtype == torch.float16,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    train_result = trainer.train()

    materialized_projection_layers = int(
        projection_report.get("materialized_projection_layers", 0)
    )
    if args.projection_mode in {"linear", "mlp"}:
        print(f"Materialized projected LoRA A layers: {materialized_projection_layers}")

    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"\nSaved adapter to: {adapter_dir}")

    summary = {
        "mode": args.mode,
        "source_model": args.source_model,
        "target_model": args.target_model,
        "source_adapter": args.source_adapter,
        "projection_scope": args.projection_scope,
        "projection_mode": args.projection_mode,
        "projection_hidden_dim": args.projection_hidden_dim,
        "adapter_dir": adapter_dir,
        "train_output_dir": train_output_dir,
        "split_layer": target_split_layer,
        "trainable_params": trainable,
        "total_params": total,
        "trainable_ratio_pct": pct,
        "train_loss": float(train_result.training_loss),
        "global_step": int(train_result.global_step),
        "train_samples": args.train_samples,
        "dataset_path": args.dataset_path,
        "max_len": args.max_len,
        "seed": args.seed,
    }
    if args.save_projection_init:
        summary.update(
            {
                "save_projection_init": True,
                "projection_log_dir": args.projection_log_dir,
                "projection_log_prefix": projection_log_experiment_id,
            }
        )
    summary.update(projection_report)
    summary["materialized_projection_layers"] = materialized_projection_layers
    summary["projection_trainable_params"] = 0

    write_summary(summary_path, summary)


if __name__ == "__main__":
    main()
