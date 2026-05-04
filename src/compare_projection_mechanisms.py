import argparse
import csv
import json
from pathlib import Path


INTERPRETATION = (
    "Projection geometry is target-dependent: Mistral preserves more of resized A norm "
    "and uses higher effective-rank W maps than Llama, while both rely on high-rank "
    "target-space alignment."
)

EXPERIMENT_FIELDS = [
    "name",
    "summary_path",
    "num_json_layers",
    "num_tensor_layers",
    "num_layers_with_W",
    "avg_cosine_resize_project",
    "avg_norm_ratio_projected_to_resized",
    "norm_ratio_min",
    "norm_ratio_max",
    "W_fro_norm_mean",
    "W_fro_norm_std",
    "W_effective_rank_mean",
    "W_effective_rank_std",
    "W_effective_rank_min",
    "W_effective_rank_max",
    "warnings",
]


def parse_summary_arg(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("--summary must use name=path format")
    name, path = value.split("=", 1)
    name = name.strip()
    path = path.strip()
    if not name or not path:
        raise argparse.ArgumentTypeError("--summary must use non-empty name=path format")
    return name, Path(path)


def maybe_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def maybe_int(value):
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def round_or_null(value, digits):
    value = maybe_float(value)
    if value is None:
        return None
    return round(value, digits)


def load_summary(name, path):
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    return {
        "name": name,
        "summary_path": str(path),
        "num_json_layers": maybe_int(payload.get("num_json_layers")),
        "num_tensor_layers": maybe_int(payload.get("num_tensor_layers")),
        "num_layers_with_W": maybe_int(payload.get("num_layers_with_W")),
        "avg_cosine_resize_project": round_or_null(
            payload.get("avg_cosine_resize_project"),
            4,
        ),
        "avg_norm_ratio_projected_to_resized": round_or_null(
            payload.get("avg_norm_ratio_projected_to_resized"),
            4,
        ),
        "norm_ratio_min": round_or_null(payload.get("norm_ratio_min"), 4),
        "norm_ratio_max": round_or_null(payload.get("norm_ratio_max"), 4),
        "W_fro_norm_mean": round_or_null(payload.get("W_fro_norm_mean"), 4),
        "W_fro_norm_std": round_or_null(payload.get("W_fro_norm_std"), 4),
        "W_effective_rank_mean": round_or_null(
            payload.get("W_effective_rank_mean"),
            1,
        ),
        "W_effective_rank_std": round_or_null(
            payload.get("W_effective_rank_std"),
            1,
        ),
        "W_effective_rank_min": round_or_null(
            payload.get("W_effective_rank_min"),
            1,
        ),
        "W_effective_rank_max": round_or_null(
            payload.get("W_effective_rank_max"),
            1,
        ),
        "warnings": payload.get("warnings") if payload.get("warnings") is not None else [],
    }


def compute_derived(experiments):
    by_name = {item["name"]: item for item in experiments}
    llama = by_name.get("llama_Alinear")
    mistral = by_name.get("mistral_Alinear")
    if not llama or not mistral:
        return {}

    llama_norm = maybe_float(llama.get("avg_norm_ratio_projected_to_resized"))
    mistral_norm = maybe_float(mistral.get("avg_norm_ratio_projected_to_resized"))
    llama_rank = maybe_float(llama.get("W_effective_rank_mean"))
    mistral_rank = maybe_float(mistral.get("W_effective_rank_mean"))
    llama_layers = maybe_float(llama.get("num_layers_with_W"))
    mistral_layers = maybe_float(mistral.get("num_layers_with_W"))

    return {
        "mistral_minus_llama_norm_ratio": (
            round(mistral_norm - llama_norm, 4)
            if mistral_norm is not None and llama_norm is not None
            else None
        ),
        "mistral_minus_llama_effective_rank": (
            round(mistral_rank - llama_rank, 1)
            if mistral_rank is not None and llama_rank is not None
            else None
        ),
        "layer_count_ratio_mistral_to_llama": (
            round(mistral_layers / llama_layers, 4)
            if mistral_layers is not None and llama_layers not in (None, 0)
            else None
        ),
    }


def markdown_value(value, digits=None):
    if value is None:
        return "null"
    if isinstance(value, float) and digits is not None:
        return f"{value:.{digits}f}"
    return str(value)


def write_json(path, source_files, experiments, derived):
    payload = {
        "source_files": source_files,
        "experiments": experiments,
        "derived_comparison": derived,
        "interpretation": INTERPRETATION,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_markdown(path, experiments):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Projection Mechanism Comparison",
        "",
        "| name | W layers | cosine resize/project | norm ratio | rank mean | norm ratio range | rank range |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for item in experiments:
        norm_range = (
            f"{markdown_value(item.get('norm_ratio_min'), 4)}-{markdown_value(item.get('norm_ratio_max'), 4)}"
            if item.get("norm_ratio_min") is not None or item.get("norm_ratio_max") is not None
            else "null"
        )
        rank_range = (
            f"{markdown_value(item.get('W_effective_rank_min'), 1)}-{markdown_value(item.get('W_effective_rank_max'), 1)}"
            if item.get("W_effective_rank_min") is not None
            or item.get("W_effective_rank_max") is not None
            else "null"
        )
        lines.append(
            "| {name} | {layers} | {cosine} | {norm_ratio} | {rank_mean} | {norm_range} | {rank_range} |".format(
                name=item["name"],
                layers=markdown_value(item.get("num_layers_with_W")),
                cosine=markdown_value(item.get("avg_cosine_resize_project"), 4),
                norm_ratio=markdown_value(
                    item.get("avg_norm_ratio_projected_to_resized"),
                    4,
                ),
                rank_mean=markdown_value(item.get("W_effective_rank_mean"), 1),
                norm_range=norm_range,
                rank_range=rank_range,
            )
        )
    lines.extend(["", INTERPRETATION, ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_csv(path, experiments):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EXPERIMENT_FIELDS)
        writer.writeheader()
        for item in experiments:
            row = dict(item)
            row["warnings"] = json.dumps(row.get("warnings", []))
            writer.writerow(row)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare multiple projection mechanism summary JSON files."
    )
    parser.add_argument(
        "--summary",
        action="append",
        type=parse_summary_arg,
        required=True,
        help="Summary in name=path format. Can be provided multiple times.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("results/analysis/projection_mechanism_comparison.json"),
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("results/analysis/projection_mechanism_comparison.md"),
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("results/analysis/projection_mechanism_comparison.csv"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    source_files = {name: str(path) for name, path in args.summary}
    experiments = [load_summary(name, path) for name, path in args.summary]
    derived = compute_derived(experiments)

    write_json(args.out_json, source_files, experiments, derived)
    write_markdown(args.out_md, experiments)
    write_csv(args.out_csv, experiments)

    print(f"Wrote: {args.out_json}")
    print(f"Wrote: {args.out_md}")
    print(f"Wrote: {args.out_csv}")


if __name__ == "__main__":
    main()
