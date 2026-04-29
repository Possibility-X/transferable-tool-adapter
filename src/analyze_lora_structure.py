import argparse
import json
import math
import re
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


DEFAULT_SOURCE_ADAPTER = "adapters/adapter_toolbench_source"
DEFAULT_TARGET_ADAPTER = "adapters/adapter_toolbench_qwen_projected_Alinear_1024"
DEFAULT_OUTPUT_JSON = "results/analysis/lora_structure.json"
DEFAULT_FIGURES_DIR = "figures"


def load_adapter_state(path: str):
    target = Path(path)
    if target.is_dir():
        candidates = [
            target / "adapter_model.safetensors",
            target / "adapter_model.bin",
            target / "pytorch_model.bin",
        ]
        for candidate in candidates:
            if candidate.exists():
                target = candidate
                break
        else:
            raise FileNotFoundError(f"No supported adapter weight file found in {path}")

    if target.suffix == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(str(target), device="cpu")
    else:
        state = torch.load(str(target), map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if not isinstance(state, dict):
            raise ValueError(f"Unsupported checkpoint payload: {target}")

    tensor_state = {key: value.detach().cpu().float() for key, value in state.items() if torch.is_tensor(value)}
    return target, tensor_state


def parse_lora_key(key: str):
    if ".lora_A" in key:
        kind = "A"
        module_key = key.split(".lora_A", 1)[0]
    elif ".lora_B" in key:
        kind = "B"
        module_key = key.split(".lora_B", 1)[0]
    else:
        return None

    layer_match = re.search(r"(?:layers|h)\.(\d+)\.", key)
    if not layer_match:
        return None
    layer = int(layer_match.group(1))
    module_match = re.search(r"self_attn\.([^.]+)$", module_key)
    module = module_match.group(1) if module_match else module_key.split(".")[-1]
    return kind, layer, module, module_key


def collect_lora_weights(state: dict):
    weights = {}
    skipped = 0
    for key, tensor in state.items():
        parsed = parse_lora_key(key)
        if not parsed:
            skipped += 1
            continue
        kind, layer, module, module_key = parsed
        weights[(layer, module, kind)] = {
            "key": key,
            "module_key": module_key,
            "tensor": tensor,
            "shape": list(tensor.shape),
        }
    return weights, skipped


def resize_tensor(src: torch.Tensor, target_shape: tuple[int, ...]):
    if tuple(src.shape) == tuple(target_shape):
        return src
    if src.ndim != 2 or len(target_shape) != 2:
        raise ValueError(f"Unsupported tensor resize: {tuple(src.shape)} -> {target_shape}")
    x = src.float().unsqueeze(0).unsqueeze(0)
    y = F.interpolate(x, size=target_shape, mode="bilinear", align_corners=False)
    return y.squeeze(0).squeeze(0)


def cosine_after_resize(src: torch.Tensor, target: torch.Tensor):
    try:
        resized = resize_tensor(src, tuple(target.shape))
    except ValueError:
        return None
    src_flat = resized.flatten()
    target_flat = target.flatten()
    denom = src_flat.norm() * target_flat.norm()
    if denom.item() == 0:
        return None
    return float(torch.dot(src_flat, target_flat) / denom)


def fro_norm(tensor: torch.Tensor):
    return float(torch.linalg.vector_norm(tensor).item())


def aggregate_layer_norms(weights: dict):
    by_layer = defaultdict(lambda: {"A_sq": 0.0, "B_sq": 0.0, "A_count": 0, "B_count": 0})
    for (layer, _module, kind), payload in weights.items():
        norm = fro_norm(payload["tensor"])
        by_layer[layer][f"{kind}_sq"] += norm * norm
        by_layer[layer][f"{kind}_count"] += 1

    result = {}
    for layer, stats in sorted(by_layer.items()):
        a_norm = math.sqrt(stats["A_sq"]) if stats["A_count"] else None
        b_norm = math.sqrt(stats["B_sq"]) if stats["B_count"] else None
        result[layer] = {
            "A_norm": a_norm,
            "B_norm": b_norm,
            "A_count": stats["A_count"],
            "B_count": stats["B_count"],
            "A_B_norm_ratio": a_norm / b_norm if a_norm is not None and b_norm not in (None, 0.0) else None,
        }
    return result


def svd_summary(matrix: torch.Tensor, top_k: int = 8):
    if matrix.numel() == 0:
        return None
    try:
        singular_values = torch.linalg.svdvals(matrix.float())
    except RuntimeError as exc:
        warnings.warn(f"SVD failed for matrix {tuple(matrix.shape)}: {exc}")
        return None

    if singular_values.numel() == 0:
        return None

    total = singular_values.sum()
    sq_total = torch.sum(singular_values * singular_values)
    probs = singular_values / total if total.item() else torch.zeros_like(singular_values)
    entropy = -torch.sum(probs * torch.log(probs.clamp_min(1e-12)))
    stable_rank = sq_total / (singular_values[0] * singular_values[0]).clamp_min(1e-12)

    return {
        "shape": list(matrix.shape),
        "top_singular_values": [float(x) for x in singular_values[:top_k]],
        "stable_rank": float(stable_rank),
        "entropy_effective_rank": float(torch.exp(entropy)),
    }


def concatenate_layer_a(weights: dict, layer: int):
    tensors = [
        payload["tensor"]
        for (item_layer, _module, kind), payload in sorted(weights.items())
        if item_layer == layer and kind == "A" and payload["tensor"].ndim == 2
    ]
    if not tensors:
        return None
    try:
        return torch.cat(tensors, dim=0)
    except RuntimeError:
        flattened = [tensor.reshape(1, -1) for tensor in tensors]
        return torch.cat(flattened, dim=0)


def analyze_layers(source_weights: dict, target_weights: dict):
    source_norms = aggregate_layer_norms(source_weights)
    target_norms = aggregate_layer_norms(target_weights)
    all_layers = sorted(set(source_norms) | set(target_norms))
    layer_rows = []
    warnings_out = []

    for layer in all_layers:
        source = source_norms.get(layer, {})
        target = target_norms.get(layer, {})
        cosine_values = []
        missing_source = []
        missing_target = []

        source_modules = {
            module
            for item_layer, module, kind in source_weights
            if item_layer == layer and kind == "A"
        }
        target_modules = {
            module
            for item_layer, module, kind in target_weights
            if item_layer == layer and kind == "A"
        }

        for module in sorted(source_modules | target_modules):
            source_payload = source_weights.get((layer, module, "A"))
            target_payload = target_weights.get((layer, module, "A"))
            if not source_payload:
                missing_source.append(module)
                continue
            if not target_payload:
                missing_target.append(module)
                continue
            cosine = cosine_after_resize(source_payload["tensor"], target_payload["tensor"])
            if cosine is not None:
                cosine_values.append(cosine)

        if missing_source:
            warnings_out.append(f"Layer {layer}: missing source A modules for {missing_source}")
        if missing_target:
            warnings_out.append(f"Layer {layer}: missing target A modules for {missing_target}")

        target_a = concatenate_layer_a(target_weights, layer)
        spectrum = svd_summary(target_a) if target_a is not None else None

        layer_rows.append(
            {
                "layer": layer,
                "source_A_norm": source.get("A_norm"),
                "source_B_norm": source.get("B_norm"),
                "target_A_norm": target.get("A_norm"),
                "target_B_norm": target.get("B_norm"),
                "source_A_B_norm_ratio": source.get("A_B_norm_ratio"),
                "target_A_B_norm_ratio": target.get("A_B_norm_ratio"),
                "mean_source_resized_target_A_cosine": (
                    sum(cosine_values) / len(cosine_values) if cosine_values else None
                ),
                "matched_A_modules": len(cosine_values),
                "target_A_spectrum": spectrum,
            }
        )

    return layer_rows, warnings_out


def find_projection_matrices(state: dict):
    projection_terms = ("projection", "projector", "alignment", "linear_projection")
    matrices = {}
    for key, tensor in state.items():
        lower = key.lower()
        if any(term in lower for term in projection_terms) and tensor.ndim == 2:
            matrices[key] = tensor
    return matrices


def plot_norms(layer_rows: list[dict], path: Path):
    layers = [row["layer"] for row in layer_rows]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    axes[0].plot(layers, [row["source_A_norm"] for row in layer_rows], marker="o", label="Source A")
    axes[0].plot(layers, [row["target_A_norm"] for row in layer_rows], marker="o", label="Target A")
    axes[0].set_ylabel("Frobenius norm")
    axes[0].set_title("LoRA A/B Layer-wise Norms")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].plot(layers, [row["source_B_norm"] for row in layer_rows], marker="o", label="Source B")
    axes[1].plot(layers, [row["target_B_norm"] for row in layer_rows], marker="o", label="Target B")
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Frobenius norm")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_spectrum(layer_rows: list[dict], projection_spectrum: dict, projection_found: bool, path: Path):
    fig, ax = plt.subplots(figsize=(9, 5))

    if projection_found:
        for key, spectrum in projection_spectrum.items():
            values = spectrum.get("top_singular_values", [])
            ax.plot(range(1, len(values) + 1), values, marker="o", label=key[-48:])
        title = "Projection Matrix Singular Spectrum"
        ax.set_xlabel("Singular value index")
    else:
        selected = [row for row in layer_rows if row.get("target_A_spectrum")]
        if selected:
            first = selected[0]
            mid = selected[len(selected) // 2]
            last = selected[-1]
            for row in (first, mid, last):
                values = row["target_A_spectrum"]["top_singular_values"]
                ax.plot(range(1, len(values) + 1), values, marker="o", label=f"Layer {row['layer']}")
        title = "Projected LoRA A Singular Spectrum"
        ax.set_xlabel("Singular value index")

    ax.set_title(title)
    ax.set_ylabel("Singular value")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def summarize(layer_rows: list[dict]):
    def mean_of(key: str):
        values = [row[key] for row in layer_rows if row.get(key) is not None]
        return sum(values) / len(values) if values else None

    return {
        "layers": len(layer_rows),
        "mean_source_A_norm": mean_of("source_A_norm"),
        "mean_source_B_norm": mean_of("source_B_norm"),
        "mean_target_A_norm": mean_of("target_A_norm"),
        "mean_target_B_norm": mean_of("target_B_norm"),
        "mean_source_resized_target_A_cosine": mean_of("mean_source_resized_target_A_cosine"),
        "mean_target_A_stable_rank": (
            sum(row["target_A_spectrum"]["stable_rank"] for row in layer_rows if row.get("target_A_spectrum"))
            / max(1, sum(1 for row in layer_rows if row.get("target_A_spectrum")))
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-adapter", default=DEFAULT_SOURCE_ADAPTER)
    parser.add_argument("--target-adapter", default=DEFAULT_TARGET_ADAPTER)
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--figures-dir", default=DEFAULT_FIGURES_DIR)
    args = parser.parse_args()

    source_path, source_state = load_adapter_state(args.source_adapter)
    target_path, target_state = load_adapter_state(args.target_adapter)
    source_weights, source_skipped = collect_lora_weights(source_state)
    target_weights, target_skipped = collect_lora_weights(target_state)

    if not source_weights:
        warnings.warn(f"No LoRA A/B weights found in source adapter: {args.source_adapter}")
    if not target_weights:
        warnings.warn(f"No LoRA A/B weights found in target adapter: {args.target_adapter}")

    layer_rows, layer_warnings = analyze_layers(source_weights, target_weights)
    projection_matrices = find_projection_matrices(target_state)
    projection_spectrum = {
        key: svd_summary(tensor)
        for key, tensor in projection_matrices.items()
    }
    projection_spectrum = {key: value for key, value in projection_spectrum.items() if value}
    projection_found = bool(projection_spectrum)

    output = {
        "source_adapter": args.source_adapter,
        "target_adapter": args.target_adapter,
        "source_weight_file": str(source_path),
        "target_weight_file": str(target_path),
        "source_lora_tensors": len(source_weights),
        "target_lora_tensors": len(target_weights),
        "source_non_lora_tensors_skipped": source_skipped,
        "target_non_lora_tensors_skipped": target_skipped,
        "projection_matrix_found": projection_found,
        "projection_spectrum": projection_spectrum,
        "summary": summarize(layer_rows),
        "layers": layer_rows,
        "warnings": layer_warnings,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    figures_dir = Path(args.figures_dir)
    plot_norms(layer_rows, figures_dir / "lora_ab_norms.png")
    plot_spectrum(
        layer_rows,
        projection_spectrum=projection_spectrum,
        projection_found=projection_found,
        path=figures_dir / "projection_spectrum.png",
    )

    for message in layer_warnings:
        warnings.warn(message)

    print(f"Saved analysis JSON to {output_path}")
    print(f"projection_matrix_found={projection_found}")
    print(f"Saved figures to {figures_dir}")


if __name__ == "__main__":
    main()
