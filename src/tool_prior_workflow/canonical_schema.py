from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class CanonicalValidationError(ValueError):
    """Raised when a canonical schema-call example is invalid."""


def _require_mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CanonicalValidationError(f"{field} must be an object")
    return value


def _require_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CanonicalValidationError(f"{field} must be a non-empty string")
    return value


def normalize_tool_call(call: dict[str, Any], field: str = "gold_call") -> dict[str, Any]:
    call = _require_mapping(call, field)
    name = call.get("name", call.get("tool"))
    arguments = call.get("arguments")
    return {
        "name": _require_string(name, f"{field}.name"),
        "arguments": _require_mapping(arguments, f"{field}.arguments"),
    }


def normalize_tool(tool: dict[str, Any], index: int) -> dict[str, Any]:
    tool = _require_mapping(tool, f"tools[{index}]")
    return {
        "name": _require_string(tool.get("name"), f"tools[{index}].name"),
        "description": str(tool.get("description") or ""),
        "arguments_schema": _require_mapping(
            tool.get("arguments_schema", {}),
            f"tools[{index}].arguments_schema",
        ),
    }


def validate_canonical(example: dict[str, Any]) -> dict[str, Any]:
    """Validate and return a normalized canonical schema-call example."""

    example = _require_mapping(example, "example")
    tools = example.get("tools")
    if not isinstance(tools, list):
        raise CanonicalValidationError("tools must be a list")

    normalized = {
        "id": _require_string(example.get("id"), "id"),
        "dataset": _require_string(example.get("dataset"), "dataset"),
        "split": _require_string(example.get("split"), "split"),
        "instruction": _require_string(example.get("instruction"), "instruction"),
        "tools": [normalize_tool(tool, index) for index, tool in enumerate(tools)],
        "gold_call": normalize_tool_call(example.get("gold_call"), "gold_call"),
        "metadata": _require_mapping(example.get("metadata", {}), "metadata"),
    }
    return normalized


def legacy_record_to_canonical(
    record: dict[str, Any],
    dataset: str,
    split: str,
    index: int,
    tools: list[dict[str, Any]] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert an existing `{instruction, gt}` record into canonical form."""

    record = _require_mapping(record, "record")
    gt = record.get("gt")
    if gt is None:
        gt = {
            "tool": record.get("tool"),
            "arguments": record.get("arguments", {}),
        }

    source_id = record.get("source_id")
    example_id = str(source_id) if source_id else f"{dataset}:{split}:{index}"
    merged_metadata = {
        "source_id": source_id,
    }
    if metadata:
        merged_metadata.update(metadata)

    return validate_canonical(
        {
            "id": example_id,
            "dataset": dataset,
            "split": split,
            "instruction": record.get("instruction"),
            "tools": tools or [],
            "gold_call": normalize_tool_call(gt, "gt"),
            "metadata": merged_metadata,
        }
    )


def canonical_to_training_record(example: dict[str, Any]) -> dict[str, Any]:
    """Return the legacy record shape consumed by current training scripts."""

    canonical = validate_canonical(example)
    metadata = canonical["metadata"]
    return {
        "source_id": metadata.get("source_id") or canonical["id"],
        "instruction": canonical["instruction"],
        "gt": {
            "tool": canonical["gold_call"]["name"],
            "arguments": canonical["gold_call"]["arguments"],
        },
    }


def load_jsonl(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if limit is not None and len(records) >= limit:
                break
    return records


def write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def infer_split_from_path(path: str | Path) -> str:
    name = Path(path).name.lower()
    if "train" in name:
        return "train"
    if "eval" in name or "test" in name:
        return "eval"
    return "unknown"
