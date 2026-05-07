from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from .canonical_schema import load_jsonl, validate_canonical
    from .evaluate_canonical import extract_first_balanced_json, is_valid_call, parse_prediction
except ImportError:  # pragma: no cover - supports direct script execution
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from tool_prior_workflow.canonical_schema import load_jsonl, validate_canonical
    from tool_prior_workflow.evaluate_canonical import (
        extract_first_balanced_json,
        is_valid_call,
        parse_prediction,
    )


def _prediction_text(prediction: Any) -> str | None:
    if isinstance(prediction, str):
        return prediction
    if isinstance(prediction, dict):
        for key in ("prediction", "prediction_text", "pred", "raw", "text", "output"):
            value = prediction.get(key)
            if isinstance(value, str):
                return value
    return None


def _has_extra_text(raw_text: str) -> bool:
    start = raw_text.find("{")
    if start == -1:
        return False
    depth = 0
    in_string = False
    escape = False
    end = -1
    for index in range(start, len(raw_text)):
        char = raw_text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                end = index
                break
    if end == -1:
        return False
    return bool(raw_text[:start].strip() or raw_text[end + 1 :].strip())


def classify_error(example: dict[str, Any], prediction: Any) -> dict[str, Any]:
    canonical = validate_canonical(example)
    gold = canonical["gold_call"]
    raw_text = _prediction_text(prediction)
    parsed_obj = extract_first_balanced_json(raw_text) if raw_text is not None else prediction

    if parsed_obj is None:
        return {"error_type": "parse_failure"}
    if not is_valid_call(parsed_obj):
        return {"error_type": "schema_mismatch", "parsed": parsed_obj}

    pred_call = parse_prediction(parsed_obj)
    if pred_call is None:
        return {"error_type": "schema_mismatch", "parsed": parsed_obj}
    if pred_call["tool"] != gold["name"]:
        return {
            "error_type": "wrong_tool",
            "gold_tool": gold["name"],
            "pred_tool": pred_call["tool"],
        }

    pred_args = pred_call["arguments"]
    gold_args = gold["arguments"]
    missing = [key for key in gold_args if key not in pred_args]
    if missing:
        return {"error_type": "missing_argument", "missing_arguments": missing}

    wrong_values = [
        key for key, value in gold_args.items() if pred_args.get(key) != value
    ]
    if wrong_values:
        return {"error_type": "wrong_argument_value", "wrong_arguments": wrong_values}

    if raw_text is not None and _has_extra_text(raw_text):
        return {"error_type": "over_generation"}
    return {"error_type": "ok"}


def summarize_errors(examples: list[dict[str, Any]], predictions: list[Any]) -> dict[str, Any]:
    if len(examples) != len(predictions):
        raise ValueError(
            f"Example/prediction length mismatch: {len(examples)} vs {len(predictions)}"
        )

    counter: Counter[str] = Counter()
    for example, prediction in zip(examples, predictions):
        result = classify_error(example, prediction)
        counter[result["error_type"]] += 1

    total = len(examples)
    return {
        "samples": total,
        "counts": dict(counter),
        "rates": {
            key: value / total if total else 0.0
            for key, value in sorted(counter.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--save", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    examples = load_jsonl(args.examples, limit=args.limit)
    predictions = load_jsonl(args.predictions, limit=args.limit)
    summary = summarize_errors(examples, predictions)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
