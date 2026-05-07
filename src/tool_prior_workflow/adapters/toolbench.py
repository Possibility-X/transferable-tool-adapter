from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

try:
    from ..canonical_schema import (
        infer_split_from_path,
        legacy_record_to_canonical,
        load_jsonl,
        validate_canonical,
        write_jsonl,
    )
except ImportError:  # pragma: no cover - supports direct script execution
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from tool_prior_workflow.canonical_schema import (
        infer_split_from_path,
        legacy_record_to_canonical,
        load_jsonl,
        validate_canonical,
        write_jsonl,
    )


API_MARKER = "Specifically, you have access to the following APIs:"


def _balanced_segment(text: str, start: int, open_char: str, close_char: str) -> str | None:
    depth = 0
    in_string = False
    quote = ""
    escape = False
    begin = -1

    for index in range(start, len(text)):
        char = text[index]
        if begin == -1:
            if char == open_char:
                begin = index
                depth = 1
            continue

        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == quote:
                in_string = False
            continue

        if char in {"'", '"'}:
            in_string = True
            quote = char
        elif char == open_char:
            depth += 1
        elif char == close_char:
            depth -= 1
            if depth == 0:
                return text[begin : index + 1]
    return None


def _load_tool_list(raw: str) -> list[dict[str, Any]]:
    for parser in (ast.literal_eval, json.loads):
        try:
            parsed = parser(raw)
        except Exception:
            continue
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
    return []


def _normalize_parameters(parameters: Any) -> dict[str, Any]:
    if not isinstance(parameters, dict):
        return {}
    if parameters.get("type") == "object" and isinstance(parameters.get("properties"), dict):
        schema = dict(parameters)
        schema.setdefault("required", parameters.get("required", []))
        return schema
    properties = parameters.get("properties") if isinstance(parameters.get("properties"), dict) else parameters
    return {
        "type": "object",
        "properties": properties if isinstance(properties, dict) else {},
        "required": parameters.get("required", []) if isinstance(parameters.get("required"), list) else [],
    }


def extract_tools_from_instruction(instruction: str) -> list[dict[str, Any]]:
    marker_index = instruction.find(API_MARKER)
    if marker_index == -1:
        return []

    segment = _balanced_segment(
        instruction,
        start=marker_index + len(API_MARKER),
        open_char="[",
        close_char="]",
    )
    if segment is None:
        return []

    tools = []
    for item in _load_tool_list(segment):
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        tools.append(
            {
                "name": name,
                "description": str(item.get("description") or ""),
                "arguments_schema": _normalize_parameters(item.get("parameters", {})),
            }
        )
    return tools


def to_canonical(record: dict[str, Any], split: str, index: int) -> dict[str, Any]:
    instruction = str(record.get("instruction") or "")
    tools = extract_tools_from_instruction(instruction)
    metadata = {
        "raw_action_input": record.get("raw_action_input"),
        "schema_extraction_method": "toolbench_instruction_api_list",
        "extracted_tool_count": len(tools),
    }
    return legacy_record_to_canonical(
        record=record,
        dataset="toolbench",
        split=split,
        index=index,
        tools=tools,
        metadata=metadata,
    )


def load_examples(path: str | Path, split: str | None = None, limit: int | None = None) -> list[dict[str, Any]]:
    split = split or infer_split_from_path(path)
    records = load_jsonl(path, limit=limit)
    return [to_canonical(record, split=split, index=index) for index, record in enumerate(records)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    examples = load_examples(args.input, split=args.split, limit=args.limit)
    for example in examples:
        validate_canonical(example)
    write_jsonl(args.output, examples)
    print(f"Saved {len(examples)} canonical ToolBench examples to {args.output}")


if __name__ == "__main__":
    main()
