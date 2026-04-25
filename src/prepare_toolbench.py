import argparse
import ast
import json
import re
from pathlib import Path

from tool_data import write_jsonl_records


DEFAULT_RAW_TRAIN = "outputs/toolbench_raw/data/toolllama_G123_dfs_train.json"
DEFAULT_RAW_EVAL = "outputs/toolbench_raw/data/toolllama_G123_dfs_eval.json"
DEFAULT_TRAIN_OUT = "data/toolbench_train.jsonl"
DEFAULT_EVAL_OUT = "data/toolbench_eval.jsonl"

OFFICIAL_REPO = "https://github.com/OpenBMB/ToolBench"
OFFICIAL_BROWSER_URL = (
    "https://drive.google.com/uc?id=1XFjDxVZdUY7TXYF2yvzx3pJlS2fy78jk"
)
OFFICIAL_TSINGHUA_URL = "https://cloud.tsinghua.edu.cn/f/c9e50625743b40bfbe10/"
OFFICIAL_DOWNLOAD = (
    "https://drive.google.com/uc?export=download&id="
    "1XFjDxVZdUY7TXYF2yvzx3pJlS2fy78jk&confirm=yes"
)

ACTION_RE = re.compile(r"Action\s*:\s*(.+?)(?:\n|$)", re.IGNORECASE)
ACTION_INPUT_RE = re.compile(
    r"Action Input\s*:\s*(.+?)(?:\nObservation\s*:|\nThought\s*:|\Z)",
    re.IGNORECASE | re.DOTALL,
)


def load_raw_json(path: str):
    raw_path = Path(path)
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Missing official ToolBench file: {raw_path}\n"
            "Download the official OpenBMB ToolBench data.zip, unzip it, and "
            "pass --raw-train/--raw-eval if your extracted paths differ.\n"
            f"- Repository: {OFFICIAL_REPO}\n"
            f"- Google Drive browser URL: {OFFICIAL_BROWSER_URL}\n"
            f"- Google Drive direct URL: {OFFICIAL_DOWNLOAD}\n"
            f"- Tsinghua Cloud: {OFFICIAL_TSINGHUA_URL}"
        )

    with raw_path.open("r", encoding="utf-8") as f:
        text = f.read().strip()

    if text.startswith("["):
        return json.loads(text)

    records = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def normalize_conversations(value):
    if isinstance(value, list):
        return value
    if isinstance(value, dict) and "from" in value and "value" in value:
        return [
            {"from": role, "value": text}
            for role, text in zip(value["from"], value["value"])
        ]
    raise ValueError("Unsupported conversations format")


def role_of(message):
    role = message.get("from") or message.get("role")
    return str(role).lower() if role is not None else ""


def text_of(message):
    value = message.get("value") or message.get("content") or ""
    return str(value)


def parse_arguments(raw_args: str):
    cleaned = raw_args.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()

    candidates = [cleaned]
    json_match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if json_match is not None:
        candidates.append(json_match.group(0))

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(candidate)
            except (SyntaxError, ValueError):
                continue
        if isinstance(parsed, dict):
            return parsed
        return {"value": parsed}

    if cleaned:
        return {"value": cleaned}
    return None


def parse_action(text: str):
    action_match = ACTION_RE.search(text)
    input_match = ACTION_INPUT_RE.search(text)
    if action_match is None or input_match is None:
        return None

    tool = action_match.group(1).strip().strip("`")
    if not tool or tool.lower() in {"finish", "final answer", "final_answer"}:
        return None

    raw_action_input = input_match.group(1).strip()
    arguments = parse_arguments(raw_action_input)
    if arguments is None:
        return None

    return {
        "tool": tool,
        "arguments": arguments,
        "raw_action_input": raw_action_input,
    }


def build_instruction(conversations, assistant_idx: int, max_chars: int):
    parts = []
    for message in conversations[:assistant_idx]:
        role = role_of(message)
        if role not in {"human", "user", "system"}:
            continue
        text = text_of(message).strip()
        if text:
            parts.append(text)

    instruction = "\n\n".join(parts).strip()
    if not instruction:
        return None
    return instruction[:max_chars]


def convert_row(row: dict, max_instruction_chars: int):
    if "conversations" not in row:
        return None

    conversations = normalize_conversations(row["conversations"])
    for idx, message in enumerate(conversations):
        if role_of(message) not in {"gpt", "assistant"}:
            continue

        action = parse_action(text_of(message))
        if action is None:
            continue

        instruction = build_instruction(conversations, idx, max_instruction_chars)
        if instruction is None:
            continue

        return {
            "source_id": row.get("id"),
            "instruction": instruction,
            "gt": {
                "tool": action["tool"],
                "arguments": action["arguments"],
            },
            "raw_action_input": action["raw_action_input"],
        }
    return None


def convert_rows(rows, limit: int, max_instruction_chars: int):
    records = []
    seen = set()
    skipped = 0

    for row in rows:
        record = convert_row(row, max_instruction_chars)
        if record is None:
            skipped += 1
            continue

        key = (
            record["instruction"],
            record["gt"]["tool"],
            json.dumps(record["gt"]["arguments"], sort_keys=True, ensure_ascii=False),
        )
        if key in seen:
            skipped += 1
            continue

        seen.add(key)
        records.append(record)
        if len(records) >= limit:
            break

    return records, skipped


def convert_file(raw_path: str, out_path: str, limit: int, max_instruction_chars: int):
    rows = load_raw_json(raw_path)
    records, skipped = convert_rows(rows, limit, max_instruction_chars)
    write_jsonl_records(out_path, records)
    print(f"Loaded raw rows: {len(rows)}")
    print(f"Saved converted records: {len(records)} -> {out_path}")
    print(f"Skipped rows: {skipped}")
    if not records:
        raise RuntimeError(f"No ToolBench tool-call records extracted from {raw_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-train", type=str, default=DEFAULT_RAW_TRAIN)
    parser.add_argument("--raw-eval", type=str, default=DEFAULT_RAW_EVAL)
    parser.add_argument("--train-out", type=str, default=DEFAULT_TRAIN_OUT)
    parser.add_argument("--eval-out", type=str, default=DEFAULT_EVAL_OUT)
    parser.add_argument("--max-train", type=int, default=3000)
    parser.add_argument("--max-eval", type=int, default=500)
    parser.add_argument("--max-instruction-chars", type=int, default=6000)
    args = parser.parse_args()

    print("Converting official ToolBench train file...")
    convert_file(
        raw_path=args.raw_train,
        out_path=args.train_out,
        limit=args.max_train,
        max_instruction_chars=args.max_instruction_chars,
    )

    print("\nConverting official ToolBench eval file...")
    convert_file(
        raw_path=args.raw_eval,
        out_path=args.eval_out,
        limit=args.max_eval,
        max_instruction_chars=args.max_instruction_chars,
    )


if __name__ == "__main__":
    main()
