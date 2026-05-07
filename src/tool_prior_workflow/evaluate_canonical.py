from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from .canonical_schema import load_jsonl, validate_canonical
except ImportError:  # pragma: no cover - supports direct script execution
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from tool_prior_workflow.canonical_schema import load_jsonl, validate_canonical


def extract_first_balanced_json(text: str) -> Any | None:
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
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
                try:
                    return json.loads(text[start : index + 1])
                except json.JSONDecodeError:
                    return None
    return None


def is_valid_call(obj: Any) -> bool:
    return (
        isinstance(obj, dict)
        and set(obj.keys()) == {"tool", "arguments"}
        and isinstance(obj["tool"], str)
        and isinstance(obj["arguments"], dict)
    )


def parse_prediction(prediction: Any) -> dict[str, Any] | None:
    if is_valid_call(prediction):
        return prediction
    if isinstance(prediction, dict):
        for key in ("prediction", "prediction_text", "pred", "raw", "text", "output"):
            if key in prediction:
                return parse_prediction(prediction[key])
        return None
    if isinstance(prediction, str):
        parsed = extract_first_balanced_json(prediction)
        return parsed if is_valid_call(parsed) else None
    return None


def score_args(pred_args: dict[str, Any], gold_args: dict[str, Any]) -> float:
    if not gold_args:
        return 1.0
    matches = sum(pred_args.get(key) == value for key, value in gold_args.items())
    return matches / len(gold_args)


def evaluate_predictions(
    examples: list[dict[str, Any]],
    predictions: list[Any],
) -> dict[str, Any]:
    if len(examples) != len(predictions):
        raise ValueError(
            f"Example/prediction length mismatch: {len(examples)} vs {len(predictions)}"
        )

    parsed = 0
    correct_tool = 0
    correct_args = 0.0

    for raw_example, prediction in zip(examples, predictions):
        example = validate_canonical(raw_example)
        pred_call = parse_prediction(prediction)
        if pred_call is None:
            continue

        parsed += 1
        gold_call = example["gold_call"]
        if pred_call["tool"] == gold_call["name"]:
            correct_tool += 1
        correct_args += score_args(pred_call["arguments"], gold_call["arguments"])

    total = len(examples)
    return {
        "samples": total,
        "parsed": parsed / total if total else 0.0,
        "tool_acc": correct_tool / parsed if parsed else 0.0,
        "arg_acc": correct_args / parsed if parsed else 0.0,
    }


def load_predictions(path: str | Path, limit: int | None = None) -> list[Any]:
    return load_jsonl(path, limit=limit)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--save", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    examples = load_jsonl(args.examples, limit=args.limit)
    predictions = load_predictions(args.predictions, limit=args.limit)
    result = evaluate_predictions(examples, predictions)
    print(json.dumps(result, indent=2, ensure_ascii=False))

    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
