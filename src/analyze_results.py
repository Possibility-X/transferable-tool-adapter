import argparse
import json
from pathlib import Path


DEFAULT_RESULTS_DIR = "results"
DEFAULT_OUTPUT = "results/summary.md"


ROWS = [
    {
        "setting": "Source",
        "route": "Source adapter",
        "model": "TinyLlama",
        "ood": "ood_source.json",
    },
    {
        "setting": "Full",
        "route": "Full target training",
        "model": "TinyLlama",
        "ood": "ood_full.json",
    },
    {
        "setting": "Distill",
        "route": "Route A",
        "model": "TinyLlama",
        "ood": "ood_distill.json",
        "summary": "distill_train_summary.json",
    },
    {
        "setting": "Qwen Full",
        "route": "Route C baseline",
        "model": "Qwen2.5-0.5B",
        "ood": "ood_qwen_full.json",
        "summary": "qwen_full_train.json",
    },
    {
        "setting": "Qwen Distill",
        "route": "Route C baseline",
        "model": "Qwen2.5-0.5B",
        "ood": "ood_qwen_distill.json",
        "summary": "qwen_distill_train.json",
    },
    {
        "setting": "Qwen Projected Full",
        "route": "Route C projection v0",
        "model": "Qwen2.5-0.5B",
        "ood": "ood_qwen_projected_full.json",
        "summary": "qwen_projected_full_train.json",
    },
    {
        "setting": "Qwen Projected Freeze",
        "route": "Route C projection v0",
        "model": "Qwen2.5-0.5B",
        "ood": "ood_qwen_projected_freeze.json",
        "summary": "qwen_projected_freeze_train.json",
    },
    {
        "setting": "Qwen Projected A-only",
        "route": "Route C projection v1",
        "model": "Qwen2.5-0.5B",
        "ood": "ood_qwen_projected_Aonly.json",
        "summary": "qwen_projected_Aonly_train.json",
    },
    {
        "setting": "Qwen Projected A-linear",
        "route": "Route C projection v2",
        "model": "Qwen2.5-0.5B",
        "ood": "ood_qwen_projected_Alinear.json",
        "summary": "qwen_projected_Alinear_train.json",
    },
    {
        "setting": "Transfer",
        "route": "Route B",
        "model": "TinyLlama",
        "split_ratio": "0.50",
        "ood": "ood_transfer.json",
    },
    {
        "setting": "Split 0.25",
        "route": "Route B ablation",
        "model": "TinyLlama",
        "split_ratio": "0.25",
        "ood": "ood_split025.json",
        "summary": "train_split025_summary.json",
    },
    {
        "setting": "Split 0.50",
        "route": "Route B ablation",
        "model": "TinyLlama",
        "split_ratio": "0.50",
        "ood": "ood_split050.json",
        "summary": "train_split050_summary.json",
    },
    {
        "setting": "Split 0.75",
        "route": "Route B ablation",
        "model": "TinyLlama",
        "split_ratio": "0.75",
        "ood": "ood_split075.json",
        "summary": "train_split075_summary.json",
    },
]


def load_json(path: Path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def fmt(value):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def build_rows(results_dir: Path):
    rows = []
    for spec in ROWS:
        ood = load_json(results_dir / spec["ood"])
        summary = load_json(results_dir / spec["summary"]) if "summary" in spec else None

        rows.append(
            {
                "Setting": spec["setting"],
                "Route": spec["route"],
                "Model": spec["model"],
                "Split Ratio": spec.get("split_ratio", "-"),
                "Split Layer": summary.get("split_layer") if summary else None,
                "Parsed": ood.get("parsed") if ood else None,
                "Tool Acc": ood.get("tool_acc") if ood else None,
                "Arg Acc": ood.get("arg_acc") if ood else None,
            }
        )
    return rows


def to_markdown(rows):
    headers = [
        "Setting",
        "Route",
        "Model",
        "Split Ratio",
        "Split Layer",
        "Parsed",
        "Tool Acc",
        "Arg Acc",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row[h]) for h in headers) + " |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Print the Markdown table without writing an output file.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    rows = build_rows(results_dir)
    markdown = to_markdown(rows)
    print(markdown, end="")

    if not args.no_save:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown, encoding="utf-8")
        print(f"\nSaved summary to: {output_path}")


if __name__ == "__main__":
    main()
