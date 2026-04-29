import argparse
import csv
import json
from pathlib import Path

import yaml


DEFAULT_REGISTRY = "experiments/registry.yaml"
DEFAULT_MARKDOWN = "results/merged/summary.md"
DEFAULT_CSV = "results/merged/summary.csv"


def load_registry(path: str):
    with open(path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    experiments = payload.get("experiments", [])
    if not isinstance(experiments, list):
        raise ValueError("registry.yaml must contain an experiments list")
    return experiments


def load_json(path: str | None):
    if not path:
        return None
    target = Path(path)
    if not target.exists():
        return None
    with target.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_nested(mapping: dict, path: list[str]):
    value = mapping
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def fmt(value):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def build_rows(experiments):
    rows = []
    for exp in experiments:
        train_path = get_nested(exp, ["train", "args", "summary-path"])
        eval_path = get_nested(exp, ["eval", "args", "save"])
        summary = load_json(train_path)
        result = load_json(eval_path)

        rows.append(
            {
                "id": exp.get("id"),
                "dataset": exp.get("dataset"),
                "method": exp.get("method"),
                "model": exp.get("model"),
                "source_model": exp.get("source_model") or exp.get("teacher"),
                "machine": exp.get("machine"),
                "priority": exp.get("priority"),
                "status": exp.get("status", "done" if result else "missing"),
                "samples": result.get("samples") if result else None,
                "parsed": result.get("parsed") if result else None,
                "tool_acc": result.get("tool_acc") if result else None,
                "arg_acc": result.get("arg_acc") if result else None,
                "router_acc": result.get("router_acc") if result else None,
                "max_input_tokens": result.get("max_input_tokens") if result else None,
                "train_samples": summary.get("train_samples") if summary else None,
                "max_len": summary.get("max_len") if summary else None,
                "train_loss": summary.get("train_loss") if summary else None,
                "global_step": summary.get("global_step") if summary else None,
                "eval_path": eval_path,
                "summary_path": train_path,
            }
        )
    return rows


def write_csv(path: str, rows, headers):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved CSV summary to: {output}")


def to_markdown(rows, headers):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(header)) for header in headers) + " |")
    return "\n".join(lines) + "\n"


def write_markdown(path: str, rows, headers):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    markdown = to_markdown(rows, headers)
    output.write_text(markdown, encoding="utf-8")
    print(markdown, end="")
    print(f"\nSaved Markdown summary to: {output}")


def write_plot(path: str, rows):
    import matplotlib.pyplot as plt

    plotted = [row for row in rows if isinstance(row.get("arg_acc"), float)]
    if not plotted:
        print("No arg_acc values found; skipping plot.")
        return

    labels = [row["id"] for row in plotted]
    values = [row["arg_acc"] for row in plotted]

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.8), 4))
    ax.bar(labels, values)
    ax.set_ylabel("Argument Accuracy")
    ax.set_ylim(0, max(values) * 1.15 if values else 1)
    ax.tick_params(axis="x", rotation=35, labelsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"Saved plot to: {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=str, default=DEFAULT_REGISTRY)
    parser.add_argument("--markdown", type=str, default=DEFAULT_MARKDOWN)
    parser.add_argument("--csv", type=str, default=DEFAULT_CSV)
    parser.add_argument("--plot", type=str, default=None)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    experiments = load_registry(args.registry)
    rows = build_rows(experiments)
    headers = [
        "id",
        "dataset",
        "method",
        "model",
        "source_model",
        "machine",
        "priority",
        "status",
        "samples",
        "parsed",
        "tool_acc",
        "arg_acc",
        "router_acc",
        "max_input_tokens",
        "train_samples",
        "max_len",
        "train_loss",
        "global_step",
        "eval_path",
        "summary_path",
    ]

    if args.no_save:
        print(to_markdown(rows, headers), end="")
        return

    write_markdown(args.markdown, rows, headers)
    write_csv(args.csv, rows, headers)
    if args.plot:
        write_plot(args.plot, rows)


if __name__ == "__main__":
    main()
