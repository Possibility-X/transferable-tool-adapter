from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

try:
    from .adapters.apibank import load_examples
    from .canonical_schema import canonical_to_training_record, load_jsonl, write_jsonl
    from .evaluate_canonical import evaluate_predictions, parse_prediction, score_args
except ImportError:  # pragma: no cover - supports direct script execution
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from tool_prior_workflow.adapters.apibank import load_examples
    from tool_prior_workflow.canonical_schema import (
        canonical_to_training_record,
        load_jsonl,
        write_jsonl,
    )
    from tool_prior_workflow.evaluate_canonical import (
        evaluate_predictions,
        parse_prediction,
        score_args,
    )


DEFAULT_LIMIT = 20


def gold_predictions(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    for example in examples:
        gold = example["gold_call"]
        predictions.append(
            {
                "tool": gold["name"],
                "arguments": gold["arguments"],
            }
        )
    return predictions


def prediction_rows(
    examples: list[dict[str, Any]], predictions: list[Any]
) -> list[dict[str, Any]]:
    return [
        {
            "id": example["id"],
            "prediction": prediction,
            "gold_call": example["gold_call"],
        }
        for example, prediction in zip(examples, predictions)
    ]


def _duplicate_count(values: list[str]) -> int:
    counts = Counter(values)
    return sum(count - 1 for count in counts.values() if count > 1)


def _prediction_id_diagnostics(
    rows: list[dict[str, Any]], example_ids: list[str]
) -> dict[str, Any]:
    rows_have_ids = bool(rows) and all(isinstance(row, dict) and "id" in row for row in rows)
    prediction_ids = [str(row["id"]) for row in rows] if rows_have_ids else []
    missing_ids = sorted(set(example_ids) - set(prediction_ids)) if rows_have_ids else []
    return {
        "rows_have_ids": rows_have_ids,
        "duplicate_example_id_count": _duplicate_count(example_ids),
        "duplicate_prediction_id_count": _duplicate_count(prediction_ids) if rows_have_ids else None,
        "missing_prediction_id_count": len(missing_ids) if rows_have_ids else None,
    }


def load_prediction_file(
    path: str | Path, examples: list[dict[str, Any]]
) -> tuple[list[Any], str, dict[str, Any]]:
    rows = load_jsonl(path)
    example_ids = [example["id"] for example in examples]
    diagnostics = _prediction_id_diagnostics(rows, example_ids)

    if diagnostics["rows_have_ids"]:
        if diagnostics["duplicate_example_id_count"] > 0:
            return rows[: len(examples)], "order_duplicate_example_ids", diagnostics
        if diagnostics["duplicate_prediction_id_count"] > 0:
            return rows[: len(examples)], "order_duplicate_prediction_ids", diagnostics
        if diagnostics["missing_prediction_id_count"] > 0:
            return rows[: len(examples)], "order_missing_prediction_ids", diagnostics

        by_id = {str(row["id"]): row for row in rows}
        if all(example_id in by_id for example_id in example_ids):
            return [by_id[example_id] for example_id in example_ids], "id", diagnostics

    return rows[: len(examples)], "order", diagnostics


def evaluate_legacy_records(
    records: list[dict[str, Any]], predictions: list[Any]
) -> dict[str, Any]:
    if len(records) != len(predictions):
        raise ValueError(
            f"Record/prediction length mismatch: {len(records)} vs {len(predictions)}"
        )

    parsed = 0
    correct_tool = 0
    correct_args = 0.0

    for record, prediction in zip(records, predictions):
        pred_call = parse_prediction(prediction)
        if pred_call is None:
            continue

        gt = record["gt"]
        parsed += 1
        if pred_call["tool"] == gt["tool"]:
            correct_tool += 1
        correct_args += score_args(pred_call["arguments"], gt["arguments"])

    total = len(records)
    return {
        "samples": total,
        "parsed": parsed / total if total else 0.0,
        "tool_acc": correct_tool / parsed if parsed else 0.0,
        "arg_acc": correct_args / parsed if parsed else 0.0,
    }


def metric_deltas(
    legacy_metrics: dict[str, Any], canonical_metrics: dict[str, Any]
) -> dict[str, float]:
    return {
        key: float(canonical_metrics[key]) - float(legacy_metrics[key])
        for key in ("parsed", "tool_acc", "arg_acc")
    }


def tool_coverage_summary(examples: list[dict[str, Any]]) -> dict[str, Any]:
    counts = [len(example["tools"]) for example in examples]
    gold_flags = [
        example["metadata"].get("gold_tool_in_extracted_tools")
        for example in examples
    ]
    gold_found = sum(flag is True for flag in gold_flags)
    placeholder_count = sum(flag is False for flag in gold_flags)

    return {
        "tool_count_min": min(counts) if counts else 0,
        "tool_count_max": max(counts) if counts else 0,
        "tool_count_avg": sum(counts) / len(counts) if counts else 0.0,
        "gold_tool_in_extracted_tools": gold_found,
        "placeholder_tool_count": placeholder_count,
    }


def run_alignment(
    dataset_path: str | Path,
    split: str | None,
    limit: int | None,
    prediction_mode: str,
    predictions_path: str | Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    examples = load_examples(dataset_path, split=split, limit=limit)
    if prediction_mode == "gold":
        predictions = gold_predictions(examples)
        alignment = "gold"
        prediction_id_diagnostics = None
    elif prediction_mode == "file":
        if predictions_path is None:
            raise ValueError("--predictions-path is required for file prediction mode")
        predictions, alignment, prediction_id_diagnostics = load_prediction_file(
            predictions_path, examples
        )
    else:
        raise ValueError(f"Unsupported prediction mode: {prediction_mode}")

    if len(examples) != len(predictions):
        raise ValueError(
            f"Example/prediction length mismatch: {len(examples)} vs {len(predictions)}"
        )

    legacy_records = [canonical_to_training_record(example) for example in examples]
    legacy_metrics = evaluate_legacy_records(legacy_records, predictions)
    canonical_metrics = evaluate_predictions(examples, predictions)
    deltas = metric_deltas(legacy_metrics, canonical_metrics)

    summary = {
        "dataset_path": str(dataset_path),
        "split": split,
        "limit": limit,
        "prediction_mode": prediction_mode,
        "prediction_alignment": alignment,
        "samples": len(examples),
        "legacy_metrics": legacy_metrics,
        "canonical_metrics": canonical_metrics,
        "metric_deltas": deltas,
        "metrics_match": all(abs(value) <= 1e-12 for value in deltas.values()),
        "coverage": tool_coverage_summary(examples),
    }
    if prediction_id_diagnostics is not None:
        summary["prediction_id_diagnostics"] = prediction_id_diagnostics
    return summary, prediction_rows(examples, predictions)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare API-Bank legacy scoring with canonical schema-call scoring "
            "on the same saved or synthetic predictions."
        )
    )
    parser.add_argument("--dataset-path", default="data/apibank_eval.jsonl")
    parser.add_argument("--split", default=None)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument(
        "--all-examples",
        action="store_true",
        help="Use every example instead of the limit.",
    )
    parser.add_argument("--prediction-mode", choices=("gold", "file"), default="gold")
    parser.add_argument("--predictions-path", default=None)
    parser.add_argument("--save-summary", default=None)
    parser.add_argument("--save-predictions", default=None)
    args = parser.parse_args()

    limit = None if args.all_examples else args.limit
    summary, rows = run_alignment(
        dataset_path=args.dataset_path,
        split=args.split,
        limit=limit,
        prediction_mode=args.prediction_mode,
        predictions_path=args.predictions_path,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.save_summary:
        out_path = Path(args.save_summary)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    if args.save_predictions:
        write_jsonl(args.save_predictions, rows)


if __name__ == "__main__":
    main()
