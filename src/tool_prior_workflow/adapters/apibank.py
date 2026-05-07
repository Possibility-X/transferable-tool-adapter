from __future__ import annotations

import argparse
import ast
import json
import re
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


TOOL_SEARCHER_API_RE = re.compile(
    r"API:\s*(?P<name>[^|]+?)\s*\|\s*"
    r"Description:\s*(?P<description>.*?)\s*\|\s*"
    r"Input parameters:\s*(?P<parameters>\{.*?\})\s*\|\s*Output",
    re.DOTALL,
)


def _json_type(value: Any) -> str:
    if not isinstance(value, str):
        return "string"
    normalized = value.lower()
    return {
        "str": "string",
        "string": "string",
        "int": "integer",
        "integer": "integer",
        "float": "number",
        "double": "number",
        "bool": "boolean",
        "boolean": "boolean",
        "list": "array",
        "dict": "object",
    }.get(normalized, normalized)


def normalize_api_parameters(parameters: Any) -> dict[str, Any]:
    if not isinstance(parameters, dict):
        return {"type": "object", "properties": {}, "required": []}

    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, spec in parameters.items():
        if not isinstance(name, str):
            continue
        if isinstance(spec, dict):
            properties[name] = {
                "type": _json_type(spec.get("type", "string")),
                "description": str(spec.get("description") or ""),
            }
            if spec.get("required") is True:
                required.append(name)
            if "format" in spec:
                properties[name]["format"] = spec["format"]
        else:
            properties[name] = {"type": _json_type(spec), "description": ""}

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


def _tool_from_api_description(raw: dict[str, Any]) -> dict[str, Any] | None:
    name = raw.get("apiCode", raw.get("name"))
    if not isinstance(name, str) or not name.strip():
        return None
    return {
        "name": name,
        "description": str(raw.get("description") or ""),
        "arguments_schema": normalize_api_parameters(raw.get("parameters", {})),
    }


def extract_json_api_descriptions(instruction: str) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    in_api_block = False
    for line in instruction.splitlines():
        stripped = line.strip()
        if stripped.startswith("API descriptions:"):
            in_api_block = True
            continue
        if not in_api_block:
            continue
        if not stripped:
            continue
        if not stripped.startswith("{"):
            if tools:
                break
            continue
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        tool = _tool_from_api_description(parsed)
        if tool is not None:
            tools.append(tool)
    return tools


def extract_tool_searcher_descriptions(instruction: str) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    for match in TOOL_SEARCHER_API_RE.finditer(instruction):
        parameters_text = match.group("parameters")
        try:
            parameters = ast.literal_eval(parameters_text)
        except (SyntaxError, ValueError):
            parameters = {}
        tools.append(
            {
                "name": match.group("name").strip(),
                "description": match.group("description").strip(),
                "arguments_schema": normalize_api_parameters(parameters),
            }
        )
    return tools


def extract_tools_from_instruction(instruction: str) -> list[dict[str, Any]]:
    tools_by_name: dict[str, dict[str, Any]] = {}
    for tool in extract_json_api_descriptions(instruction) + extract_tool_searcher_descriptions(instruction):
        tools_by_name.setdefault(tool["name"], tool)
    return list(tools_by_name.values())


def _placeholder_tool(name: str) -> dict[str, Any]:
    return {
        "name": name,
        "description": "",
        "arguments_schema": {"type": "object", "properties": {}, "required": []},
    }


def to_canonical(record: dict[str, Any], split: str, index: int) -> dict[str, Any]:
    instruction = str(record.get("instruction") or "")
    tools = extract_tools_from_instruction(instruction)
    gt = record.get("gt") if isinstance(record.get("gt"), dict) else {}
    gold_name = gt.get("tool")
    gold_tool_in_tools = isinstance(gold_name, str) and any(tool["name"] == gold_name for tool in tools)
    if isinstance(gold_name, str) and not gold_tool_in_tools:
        tools.append(_placeholder_tool(gold_name))

    metadata = {
        "raw_api_request": record.get("raw_api_request"),
        "schema_extraction_method": "apibank_instruction_json_blocks_and_toolsearcher_text",
        "extracted_tool_count": len(tools),
        "gold_tool_in_extracted_tools": gold_tool_in_tools,
    }
    return legacy_record_to_canonical(
        record=record,
        dataset="apibank",
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
    print(f"Saved {len(examples)} canonical API-Bank examples to {args.output}")


if __name__ == "__main__":
    main()
