import argparse
import json
import math
from pathlib import Path
from statistics import mean, pstdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


LAYER_LIST_KEYS = (
    "layers",
    "layer_stats",
    "layer_records",
    "records",
    "layer_tensors",
)


def as_float(value):
    if value is None:
        return None
    try:
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                return None
            return float(value.detach().cpu().item())
        return float(value)
    except (TypeError, ValueError):
        return None


def finite_values(values):
    return [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]


def mean_or_none(values):
    values = finite_values(values)
    return float(mean(values)) if values else None


def std_or_none(values):
    values = finite_values(values)
    return float(pstdev(values)) if len(values) > 1 else 0.0 if values else None


def tensor_float_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().float()
    return None


def fro_norm(tensor):
    tensor = tensor_float_cpu(tensor)
    if tensor is None:
        return None
    return float(torch.linalg.vector_norm(tensor).item())


def cosine_if_same_shape(left, right):
    left = tensor_float_cpu(left)
    right = tensor_float_cpu(right)
    if left is None or right is None:
        return None
    if tuple(left.shape) != tuple(right.shape) or left.numel() == 0:
        return None
    left_flat = left.reshape(-1)
    right_flat = right.reshape(-1)
    denom = torch.linalg.vector_norm(left_flat) * torch.linalg.vector_norm(right_flat)
    if denom.item() == 0:
        return None
    return float(torch.dot(left_flat, right_flat).div(denom).item())


def singular_values(tensor):
    tensor = tensor_float_cpu(tensor)
    if tensor is None or tensor.ndim != 2:
        return None
    try:
        return torch.linalg.svdvals(tensor)
    except RuntimeError:
        return None


def effective_rank_from_singular_values(values):
    if values is None or values.numel() == 0:
        return None
    values = values.float()
    total = values.sum()
    if total.item() <= 0:
        return None
    probs = values / total
    probs = probs[probs > 0]
    entropy = -(probs * torch.log(probs)).sum()
    return float(torch.exp(entropy).item())


def load_json_log(path):
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    metadata = {}
    layers = []
    if isinstance(payload, dict):
        metadata = payload.get("metadata") or {}
        for key in LAYER_LIST_KEYS:
            value = payload.get(key)
            if isinstance(value, list):
                layers = value
                break
        if not layers:
            for key, value in payload.items():
                if isinstance(value, list) and all(isinstance(item, dict) for item in value):
                    if any("layer" in item or "layer_name" in item for item in value):
                        layers = value
                        break
    elif isinstance(payload, list):
        layers = payload

    if not isinstance(metadata, dict):
        metadata = {}
    layers = [layer for layer in layers if isinstance(layer, dict)]
    return metadata, layers


def first_present(record, keys):
    for key in keys:
        if isinstance(record, dict) and key in record:
            return record[key]
    return None


def layer_name_from_record(record, index):
    value = first_present(
        record,
        ("layer_name", "name", "target_name", "target_layer_name", "key"),
    )
    return str(value) if value is not None else f"layer_{index:03d}"


def tensor_record_from_payload(payload, warnings):
    records = []

    if isinstance(payload, dict):
        for key in LAYER_LIST_KEYS:
            value = payload.get(key)
            if isinstance(value, list):
                records = value
                break
        if not records:
            candidate_records = []
            for key, value in payload.items():
                if key == "metadata":
                    continue
                if isinstance(value, dict):
                    record = dict(value)
                    record.setdefault("layer_name", key)
                    candidate_records.append(record)
            records = candidate_records
    elif isinstance(payload, list):
        records = payload

    tensor_by_layer = {}
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            warnings.append(f"pt record {index} is not a dict; skipped")
            continue
        name = layer_name_from_record(record, index)
        tensor_by_layer[name] = {
            "source_A": first_present(record, ("source_A", "source_a", "source", "A_source")),
            "resized_A": first_present(record, ("resized_A", "resized_a", "resized", "A_resized")),
            "projected_A": first_present(record, ("projected_A", "projected_a", "projected", "A_projected")),
            "W": first_present(record, ("W", "w", "projection_W", "projection_w", "weight")),
        }
    return tensor_by_layer


def load_pt_log(path, warnings):
    try:
        payload = torch.load(path, map_location="cpu")
    except Exception as exc:
        warnings.append(f"failed to load pt file: {type(exc).__name__}: {exc}")
        return {}
    return tensor_record_from_payload(payload, warnings)


def numeric_layer_index(name, fallback):
    digits = []
    for part in str(name).replace(".", "_").split("_"):
        if part.isdigit():
            digits.append(int(part))
    return digits[-1] if digits else fallback


def build_layer_diagnostics(json_layers, tensor_layers, warnings):
    diagnostics = []
    all_names = []
    seen = set()

    for index, layer in enumerate(json_layers):
        name = layer_name_from_record(layer, index)
        all_names.append(name)
        seen.add(name)
    for name in tensor_layers:
        if name not in seen:
            all_names.append(name)

    for index, name in enumerate(all_names):
        json_layer = json_layers[index] if index < len(json_layers) else {}
        if layer_name_from_record(json_layer, index) != name:
            json_layer = next(
                (item for item in json_layers if layer_name_from_record(item, index) == name),
                {},
            )
        tensors = tensor_layers.get(name, {})

        source_A = tensor_float_cpu(tensors.get("source_A"))
        resized_A = tensor_float_cpu(tensors.get("resized_A"))
        projected_A = tensor_float_cpu(tensors.get("projected_A"))
        W = tensors.get("W")
        if isinstance(W, dict):
            warnings.append(f"{name}: W is a dict; singular spectrum skipped")
            W = None
        W = tensor_float_cpu(W)

        source_norm = fro_norm(source_A)
        resized_norm = fro_norm(resized_A)
        projected_norm = fro_norm(projected_A)

        if source_norm is None:
            source_norm = as_float(json_layer.get("source_A_norm"))
        if resized_norm is None:
            resized_norm = as_float(json_layer.get("resized_A_norm"))
        if projected_norm is None:
            projected_norm = as_float(json_layer.get("projected_A_norm"))

        ratio = None
        if resized_norm not in (None, 0.0) and projected_norm is not None:
            ratio = projected_norm / resized_norm

        cosine_resize_project = cosine_if_same_shape(resized_A, projected_A)
        if cosine_resize_project is None:
            cosine_resize_project = as_float(
                first_present(json_layer, ("resize_to_project_cosine", "cosine_resize_project"))
            )

        cosine_source_resized = cosine_if_same_shape(source_A, resized_A)
        if cosine_source_resized is None:
            cosine_source_resized = as_float(
                first_present(json_layer, ("source_to_resized_cosine", "cosine_source_resized"))
            )

        W_singular_values = singular_values(W)
        W_fro = fro_norm(W)
        W_effective_rank = effective_rank_from_singular_values(W_singular_values)

        diagnostics.append(
            {
                "layer_name": name,
                "layer_index": numeric_layer_index(name, index),
                "source_norm": source_norm,
                "resized_norm": resized_norm,
                "projected_norm": projected_norm,
                "norm_ratio_projected_to_resized": ratio,
                "cosine_resize_project": cosine_resize_project,
                "cosine_source_resized": cosine_source_resized,
                "W_fro_norm": W_fro,
                "W_effective_rank": W_effective_rank,
                "W_singular_values": W_singular_values,
                "W_exists": W is not None
                or bool(first_present(json_layer, ("W_exists", "w_exists")))
                or first_present(json_layer, ("W_shape", "w_shape")) is not None,
            }
        )

    return diagnostics


def plot_projection_init_drift(diagnostics, output_path):
    x = list(range(len(diagnostics)))
    source = [item["source_norm"] for item in diagnostics]
    resized = [item["resized_norm"] for item in diagnostics]
    projected = [item["projected_norm"] for item in diagnostics]
    cosines = [item["cosine_resize_project"] for item in diagnostics]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    if finite_values(source):
        ax.plot(x, source, label="source A", linewidth=1.8)
    if finite_values(resized):
        ax.plot(x, resized, label="resized A", linewidth=1.8)
    if finite_values(projected):
        ax.plot(x, projected, label="projected A", linewidth=1.8)
    ax.set_title("Projection Initialization Drift")
    ax.set_xlabel("Logged layer index")
    ax.set_ylabel("Frobenius norm")
    ax.grid(alpha=0.25)

    handles, labels = ax.get_legend_handles_labels()
    if finite_values(cosines):
        ax2 = ax.twinx()
        ax2.plot(x, cosines, label="cos(resized, projected)", color="black", linestyle="--", linewidth=1.4)
        ax2.set_ylabel("Cosine")
        h2, l2 = ax2.get_legend_handles_labels()
        handles += h2
        labels += l2

    if handles:
        ax.legend(handles, labels, loc="best", fontsize=9)
    else:
        ax.text(0.5, 0.5, "No layer diagnostics found", ha="center", va="center", transform=ax.transAxes)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def representative_indices(num_items, max_items=6):
    if num_items <= max_items:
        return list(range(num_items))
    if max_items <= 1:
        return [0]
    return sorted({round(i * (num_items - 1) / (max_items - 1)) for i in range(max_items)})


def plot_projection_w_spectrum(diagnostics, output_path, warnings):
    with_spectrum = [
        item
        for item in diagnostics
        if isinstance(item.get("W_singular_values"), torch.Tensor)
        and item["W_singular_values"].numel() > 0
    ]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    if not with_spectrum:
        ax.axis("off")
        ax.text(0.5, 0.5, "No projection W tensors found", ha="center", va="center", fontsize=14)
        warnings.append("No projection W tensors found")
    else:
        for index in representative_indices(len(with_spectrum), max_items=6):
            item = with_spectrum[index]
            values = item["W_singular_values"].float()
            first = values[0].item()
            if first <= 0:
                continue
            normalized = (values / first).cpu().tolist()
            ax.plot(
                range(1, len(normalized) + 1),
                normalized,
                linewidth=1.6,
                label=item["layer_name"],
            )
        ax.set_title("Projection W Singular Spectrum")
        ax.set_xlabel("Singular value rank")
        ax.set_ylabel("Singular value / top singular value")
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=7)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_effective_rank_by_layer(diagnostics, output_path, warnings):
    ranks = [item["W_effective_rank"] for item in diagnostics]
    values = finite_values(ranks)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    if not values:
        ax.axis("off")
        ax.text(0.5, 0.5, "No projection W tensors found", ha="center", va="center", fontsize=14)
        warnings.append("No W effective ranks found for layer-wise plot")
    else:
        x = list(range(len(diagnostics)))
        ax.plot(x, ranks, marker="o", linewidth=1.5, markersize=3.5)
        ax.set_title("Projection W Effective Rank by Layer")
        ax.set_xlabel("Logged layer index")
        ax.set_ylabel("W effective rank")
        ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_norm_ratio_by_layer(diagnostics, output_path, warnings):
    ratios = [item["norm_ratio_projected_to_resized"] for item in diagnostics]
    values = finite_values(ratios)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    if not values:
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "No projected/resized norm ratios found",
            ha="center",
            va="center",
            fontsize=14,
        )
        warnings.append("No projected/resized norm ratios found for layer-wise plot")
    else:
        x = list(range(len(diagnostics)))
        ratio_mean = mean(values)
        ax.plot(x, ratios, marker="o", linewidth=1.5, markersize=3.5)
        ax.axhline(
            ratio_mean,
            color="black",
            linestyle="--",
            linewidth=1.2,
            label=f"mean={ratio_mean:.4f}",
        )
        ax.set_title("Projected-to-Resized A Norm Ratio by Layer")
        ax.set_xlabel("Logged layer index")
        ax.set_ylabel("projected_A_norm / resized_A_norm")
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def layer_value_records(diagnostics, key):
    records = []
    for item in diagnostics:
        value = item.get(key)
        if isinstance(value, (int, float)) and math.isfinite(value):
            records.append({"layer_name": item["layer_name"], "value": float(value)})
    return records


def write_summary(args, json_layers, tensor_layers, diagnostics, representative_layers, warnings):
    norm_ratios = [item["norm_ratio_projected_to_resized"] for item in diagnostics]
    effective_ranks = [item["W_effective_rank"] for item in diagnostics]
    summary = {
        "json_path": str(args.json_path),
        "pt_path": str(args.pt_path),
        "num_json_layers": len(json_layers),
        "num_tensor_layers": len(tensor_layers),
        "num_layers_with_W": sum(1 for item in diagnostics if item.get("W_exists")),
        "avg_cosine_resize_project": mean_or_none(
            [item["cosine_resize_project"] for item in diagnostics]
        ),
        "avg_norm_ratio_projected_to_resized": mean_or_none(
            [item["norm_ratio_projected_to_resized"] for item in diagnostics]
        ),
        "W_fro_norm_mean": mean_or_none([item["W_fro_norm"] for item in diagnostics]),
        "W_fro_norm_std": std_or_none([item["W_fro_norm"] for item in diagnostics]),
        "W_effective_rank_mean": mean_or_none(
            [item["W_effective_rank"] for item in diagnostics]
        ),
        "W_effective_rank_std": std_or_none(
            effective_ranks
        ),
        "norm_ratio_min": min(finite_values(norm_ratios), default=None),
        "norm_ratio_max": max(finite_values(norm_ratios), default=None),
        "W_effective_rank_min": min(finite_values(effective_ranks), default=None),
        "W_effective_rank_max": max(finite_values(effective_ranks), default=None),
        "W_effective_rank_by_layer": layer_value_records(diagnostics, "W_effective_rank"),
        "norm_ratio_by_layer": layer_value_records(
            diagnostics,
            "norm_ratio_projected_to_resized",
        ),
        "representative_spectrum_layers": representative_layers,
        "warnings": warnings,
    }
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot projection initialization mechanism diagnostics from JSON/PT logs."
    )
    parser.add_argument("--json-path", type=Path, required=True)
    parser.add_argument("--pt-path", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("figures"))
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=Path("results/analysis/projection_mechanism_summary.json"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    warnings = []

    if not args.json_path.exists():
        raise FileNotFoundError(f"Missing JSON log: {args.json_path}")
    if not args.pt_path.exists():
        raise FileNotFoundError(f"Missing PT log: {args.pt_path}")

    metadata, json_layers = load_json_log(args.json_path)
    tensor_layers = load_pt_log(args.pt_path, warnings)
    diagnostics = build_layer_diagnostics(json_layers, tensor_layers, warnings)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    drift_path = args.out_dir / "projection_init_drift.png"
    spectrum_path = args.out_dir / "projection_w_spectrum.png"
    effective_rank_path = args.out_dir / "projection_effective_rank_by_layer.png"
    norm_ratio_path = args.out_dir / "projection_norm_ratio_by_layer.png"

    plot_projection_init_drift(diagnostics, drift_path)

    spectrum_items = [
        item
        for item in diagnostics
        if isinstance(item.get("W_singular_values"), torch.Tensor)
        and item["W_singular_values"].numel() > 0
    ]
    representative_layers = [
        spectrum_items[index]["layer_name"]
        for index in representative_indices(len(spectrum_items), max_items=6)
    ]
    plot_projection_w_spectrum(diagnostics, spectrum_path, warnings)
    plot_effective_rank_by_layer(diagnostics, effective_rank_path, warnings)
    plot_norm_ratio_by_layer(diagnostics, norm_ratio_path, warnings)

    summary = write_summary(
        args=args,
        json_layers=json_layers,
        tensor_layers=tensor_layers,
        diagnostics=diagnostics,
        representative_layers=representative_layers,
        warnings=warnings,
    )

    print(f"Loaded JSON layers: {len(json_layers)}")
    print(f"Loaded tensor layers: {len(tensor_layers)}")
    print(f"Metadata keys: {sorted(metadata.keys())}")
    print(f"W layers: {summary['num_layers_with_W']}")
    print(f"Wrote: {drift_path}")
    print(f"Wrote: {spectrum_path}")
    print(f"Wrote: {effective_rank_path}")
    print(f"Wrote: {norm_ratio_path}")
    print(f"Wrote: {args.summary_path}")


if __name__ == "__main__":
    main()
