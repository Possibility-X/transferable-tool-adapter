import argparse
import ast
import json
import random
import re
from pathlib import Path

from tool_data import write_jsonl_records


DEFAULT_RAW_TRAIN = [
    "data/apibank/raw/lv1-api-train.json",
    "data/apibank/raw/lv2-api-train.json",
]
DEFAULT_RAW_EVAL = [
    "data/apibank/raw/level-1-api.json",
    "data/apibank/raw/level-2-api.json",
]
DEFAULT_TRAIN_OUT = "data/apibank_train.jsonl"
DEFAULT_EVAL_OUT = "data/apibank_eval.jsonl"

OFFICIAL_REPO = "https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/api-bank"
OFFICIAL_DATASET = "https://huggingface.co/datasets/liminghao1630/API-Bank"

API_REQUEST_RE = re.compile(r"API-Request\s*:\s*\[(.+?)\]\s*$", re.DOTALL)


def load_rows(path: str):
    raw_path = Path(path)
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Missing official API-Bank file: {raw_path}\n"
            "Download the official API-Bank files from Hugging Face:\n"
            f"- Repository: {OFFICIAL_REPO}\n"
            f"- Dataset: {OFFICIAL_DATASET}"
        )
    with raw_path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"Expected list JSON in {raw_path}")
    return rows


def literal_or_source(node):
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError):
        return ast.unparse(node)


def parse_api_request(text: str):
    match = API_REQUEST_RE.search(text.strip())
    if match is None:
        return None

    call_text = match.group(1).strip()
    try:
        parsed = ast.parse(call_text, mode="eval")
    except SyntaxError:
        return None

    call = parsed.body
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
        return None

    arguments = {}
    for index, arg in enumerate(call.args):
        arguments[f"arg{index}"] = literal_or_source(arg)
    for keyword in call.keywords:
        if keyword.arg is None:
            continue
        arguments[keyword.arg] = literal_or_source(keyword.value)

    return {
        "tool": call.func.id,
        "arguments": arguments,
    }


def build_instruction(row: dict, max_chars: int):
    instruction = str(row.get("instruction") or "").strip()
    user_input = str(row.get("input") or "").strip()
    combined = f"{instruction}\n{user_input}".strip()
    if not combined:
        return None
    return combined[:max_chars]


def convert_row(row: dict, source_path: str, max_instruction_chars: int):
    expected = row.get("expected_output") or row.get("output")
    if not isinstance(expected, str):
        return None

    gt = parse_api_request(expected)
    if gt is None:
        return None

    instruction = build_instruction(row, max_instruction_chars)
    if instruction is None:
        return None

    return {
        "source_id": f"{source_path}:{row.get('id', '')}",
        "instruction": instruction,
        "gt": gt,
        "raw_api_request": expected,
    }


def convert_files(paths, max_instruction_chars: int):
    records = []
    skipped = 0
    for path in paths:
        rows = load_rows(path)
        for row in rows:
            record = convert_row(row, path, max_instruction_chars)
            if record is None:
                skipped += 1
                continue
            records.append(record)
        print(f"Loaded {len(rows)} rows from {path}")
    return records, skipped


def dedupe(records):
    seen = set()
    unique = []
    skipped = 0
    for record in records:
        key = (
            record["instruction"],
            record["gt"]["tool"],
            json.dumps(record["gt"]["arguments"], sort_keys=True, ensure_ascii=False),
        )
        if key in seen:
            skipped += 1
            continue
        seen.add(key)
        unique.append(record)
    return unique, skipped


def limit_records(records, limit: int | None, seed: int):
    if limit is None or len(records) <= limit:
        return records
    rng = random.Random(seed)
    selected = records.copy()
    rng.shuffle(selected)
    return selected[:limit]


def write_split(name: str, paths, out_path: str, limit: int | None, max_instruction_chars: int, seed: int):
    records, skipped_parse = convert_files(paths, max_instruction_chars)
    records, skipped_dupe = dedupe(records)
    records = limit_records(records, limit, seed)
    write_jsonl_records(out_path, records)

    tools = sorted({record["gt"]["tool"] for record in records})
    print(f"\n{name}")
    print(f"Saved records: {len(records)} -> {out_path}")
    print(f"Unique tools: {len(tools)}")
    print(f"Skipped parse: {skipped_parse}")
    print(f"Skipped duplicates: {skipped_dupe}")
    if not records:
        raise RuntimeError(f"No API-Bank records extracted for {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-train", nargs="+", default=DEFAULT_RAW_TRAIN)
    parser.add_argument("--raw-eval", nargs="+", default=DEFAULT_RAW_EVAL)
    parser.add_argument("--train-out", type=str, default=DEFAULT_TRAIN_OUT)
    parser.add_argument("--eval-out", type=str, default=DEFAULT_EVAL_OUT)
    parser.add_argument("--max-train", type=int, default=3000)
    parser.add_argument("--max-eval", type=int, default=None)
    parser.add_argument("--max-instruction-chars", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    write_split(
        name="API-Bank train",
        paths=args.raw_train,
        out_path=args.train_out,
        limit=args.max_train,
        max_instruction_chars=args.max_instruction_chars,
        seed=args.seed,
    )
    write_split(
        name="API-Bank eval",
        paths=args.raw_eval,
        out_path=args.eval_out,
        limit=args.max_eval,
        max_instruction_chars=args.max_instruction_chars,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
