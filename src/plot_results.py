import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


DEFAULT_SUMMARY = "results/merged/summary.csv"
DEFAULT_FIGURES_DIR = "figures"
METRICS = {
    "parsed": "Parsed",
    "tool_acc": "Tool Acc",
    "arg_acc": "Arg Acc",
}


def load_summary(path: str):
    rows = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[row["id"]] = row
    return rows


def as_float(row: dict, key: str):
    value = row.get(key)
    if value in (None, ""):
        return None
    return float(value)


def require_row(rows: dict, exp_id: str):
    if exp_id not in rows:
        raise KeyError(f"Missing experiment id in summary: {exp_id}")
    return rows[exp_id]


def collect_panel(rows: dict, mapping: list[tuple[str, str]], metrics: list[str]):
    panel = []
    for label, exp_id in mapping:
        row = require_row(rows, exp_id)
        values = {metric: as_float(row, metric) for metric in metrics}
        panel.append({"label": label, "id": exp_id, "values": values})
    return panel


def add_bar_labels(ax, bars):
    for bar in bars:
        height = bar.get_height()
        if height <= 0:
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.015,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=90,
        )


def grouped_metric_panel(ax, panel, metrics: list[str], title: str):
    labels = [item["label"] for item in panel]
    x_positions = list(range(len(labels)))
    width = 0.22 if len(metrics) == 3 else 0.32
    offsets = [width * (idx - (len(metrics) - 1) / 2) for idx in range(len(metrics))]
    colors = ["#3f6fb5", "#d9822b", "#3f8f5f"]

    for idx, metric in enumerate(metrics):
        values = [item["values"][metric] or 0.0 for item in panel]
        bars = ax.bar(
            [x + offsets[idx] for x in x_positions],
            values,
            width=width,
            label=METRICS[metric],
            color=colors[idx % len(colors)],
        )
        add_bar_labels(ax, bars)

    ax.set_title(title, fontsize=11)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.grid(axis="y", alpha=0.25)


def save_figure(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_toolbench_api_tradeoff(rows: dict, out_dir: Path, plot_data: dict):
    metrics = ["parsed", "tool_acc", "arg_acc"]
    toolbench_mapping = [
        ("Full", "toolbench_qwen_full_1024"),
        ("Projection", "toolbench_qwen_projected_Alinear_1024"),
        ("Non-param", "toolbench_qwen_nonparam_1024"),
        ("Hybrid", "toolbench_qwen_hybrid_Alinear_1024"),
        ("MCP-style", "toolbench_qwen_mcpstyle_1024"),
        ("MCP+TTP", "toolbench_qwen_mcpstyle_hybrid_Alinear_1024"),
    ]
    apibank_mapping = [
        ("Full", "apibank_qwen_full"),
        ("Projection", "apibank_qwen_projected_Alinear"),
        ("Non-param", "apibank_qwen_nonparam"),
        ("Hybrid", "apibank_qwen_hybrid_Alinear"),
        ("MCP-style", "apibank_qwen_mcpstyle"),
        ("MCP+TTP", "apibank_qwen_mcpstyle_hybrid_Alinear"),
    ]

    panels = {
        "ToolBench 1024": collect_panel(rows, toolbench_mapping, metrics),
        "API-Bank": collect_panel(rows, apibank_mapping, metrics),
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, (title, panel) in zip(axes, panels.items()):
        grouped_metric_panel(ax, panel, metrics, title)
    axes[1].legend(loc="upper center", bbox_to_anchor=(-0.1, 1.18), ncol=3, frameon=False)
    fig.suptitle("Parse, Tool, and Argument Tradeoff", fontsize=13)
    save_figure(fig, out_dir / "toolbench_api_tradeoff.png")
    plot_data["toolbench_api_tradeoff"] = panels


def plot_context_scaling(rows: dict, out_dir: Path, plot_data: dict):
    mapping_512 = [
        ("Full", "toolbench_qwen_full"),
        ("Projection", "toolbench_qwen_projected_Alinear"),
        ("Non-param", "toolbench_qwen_nonparam"),
        ("Hybrid", "toolbench_qwen_hybrid_Alinear"),
    ]
    mapping_1024 = [
        ("Full", "toolbench_qwen_full_1024"),
        ("Projection", "toolbench_qwen_projected_Alinear_1024"),
        ("Non-param", "toolbench_qwen_nonparam_1024"),
        ("Hybrid", "toolbench_qwen_hybrid_Alinear_1024"),
    ]

    labels = [label for label, _ in mapping_512]
    values_512 = [as_float(require_row(rows, exp_id), "tool_acc") for _, exp_id in mapping_512]
    values_1024 = [as_float(require_row(rows, exp_id), "tool_acc") for _, exp_id in mapping_1024]

    x_positions = list(range(len(labels)))
    width = 0.34
    fig, ax = plt.subplots(figsize=(8, 5))
    bars_512 = ax.bar([x - width / 2 for x in x_positions], values_512, width, label="512", color="#6f7f8f")
    bars_1024 = ax.bar([x + width / 2 for x in x_positions], values_1024, width, label="1024", color="#2f8f7f")
    add_bar_labels(ax, bars_512)
    add_bar_labels(ax, bars_1024)

    ax.set_title("ToolBench Context Scaling", fontsize=12)
    ax.set_ylabel("Tool Acc")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 0.52)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Max input tokens", frameon=False)

    save_figure(fig, out_dir / "context_scaling.png")
    plot_data["context_scaling"] = {
        "metric": "tool_acc",
        "512": collect_panel(rows, mapping_512, ["tool_acc"]),
        "1024": collect_panel(rows, mapping_1024, ["tool_acc"]),
    }


def plot_hybrid_composition(rows: dict, out_dir: Path, plot_data: dict):
    metrics = ["parsed", "tool_acc"]
    toolbench_mapping = [
        ("Naive Hybrid", "toolbench_qwen_hybrid_Alinear_1024"),
        ("Router-prior", "toolbench_qwen_hybrid_router_prior_1024"),
        ("MCP+TTP", "toolbench_qwen_mcpstyle_hybrid_Alinear_1024"),
    ]
    apibank_mapping = [
        ("Naive Hybrid", "apibank_qwen_hybrid_Alinear"),
        ("Router-prior", "apibank_qwen_hybrid_router_prior"),
        ("MCP+TTP", "apibank_qwen_mcpstyle_hybrid_Alinear"),
    ]

    panels = {
        "ToolBench 1024": collect_panel(rows, toolbench_mapping, metrics),
        "API-Bank": collect_panel(rows, apibank_mapping, metrics),
    }

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for ax, (title, panel) in zip(axes, panels.items()):
        grouped_metric_panel(ax, panel, metrics, title)
    axes[1].legend(loc="upper center", bbox_to_anchor=(-0.1, 1.16), ncol=2, frameon=False)
    fig.suptitle("Hybrid Composition Difficulty", fontsize=13)
    save_figure(fig, out_dir / "hybrid_composition.png")
    plot_data["hybrid_composition"] = panels


def write_plot_data(path: Path, plot_data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(plot_data, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default=DEFAULT_SUMMARY)
    parser.add_argument("--figures-dir", default=DEFAULT_FIGURES_DIR)
    args = parser.parse_args()

    rows = load_summary(args.summary)
    out_dir = Path(args.figures_dir)
    plot_data = {
        "summary_path": args.summary,
        "figures_dir": str(out_dir),
    }

    plot_toolbench_api_tradeoff(rows, out_dir, plot_data)
    plot_context_scaling(rows, out_dir, plot_data)
    plot_hybrid_composition(rows, out_dir, plot_data)
    write_plot_data(out_dir / "plot_data.json", plot_data)

    print(f"Saved figures to {out_dir}")
    print(f"Saved plot data to {out_dir / 'plot_data.json'}")


if __name__ == "__main__":
    main()
