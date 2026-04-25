import json
import random
from pathlib import Path

from datasets import Dataset


def normalize_gt(record: dict):
    if "gt" in record:
        gt = record["gt"]
    else:
        gt = {
            "tool": record["tool"],
            "arguments": record.get("arguments", {}),
        }

    if not isinstance(gt, dict):
        raise ValueError("gt must be a dict")
    if "tool" not in gt or "arguments" not in gt:
        raise ValueError("gt must contain tool and arguments")
    if not isinstance(gt["tool"], str):
        raise ValueError("gt.tool must be a string")
    if not isinstance(gt["arguments"], dict):
        raise ValueError("gt.arguments must be a dict")
    return gt


def load_jsonl_records(path: str, limit: int | None = None, seed: int | None = None):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            gt = normalize_gt(payload)
            instruction = payload.get("instruction")
            if not isinstance(instruction, str) or not instruction.strip():
                raise ValueError(f"Missing instruction in {path}")
            records.append(
                {
                    "instruction": instruction,
                    "gt": gt,
                    "source_id": payload.get("source_id"),
                }
            )

    if seed is not None:
        rng = random.Random(seed)
        rng.shuffle(records)

    if limit is not None:
        records = records[:limit]

    if not records:
        raise ValueError(f"No records loaded from {path}")
    return records


def write_jsonl_records(path: str, records):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def format_training_text(instruction: str, gt: dict, fewshot: str):
    return (
        f"{fewshot}\n\n"
        f"User: {instruction}\n"
        "Assistant:\n"
        "TOOL_CALL:\n"
        f"{json.dumps(gt, ensure_ascii=False)}"
    )


def format_prompt_text(instruction: str, fewshot: str):
    return (
        f"{fewshot}\n\n"
        f"User: {instruction}\n"
        "Assistant:\n"
        "TOOL_CALL:\n"
    )


def tokenize_prompt_completion(prompt_text: str, completion_text: str, tokenizer, max_len: int):
    prompt_ids = tokenizer(prompt_text, add_special_tokens=True)["input_ids"]
    completion_ids = tokenizer(completion_text, add_special_tokens=False)["input_ids"]

    eos_id = tokenizer.eos_token_id
    if eos_id is not None:
        completion_ids = completion_ids + [eos_id]

    if len(completion_ids) >= max_len:
        completion_ids = completion_ids[:max_len]
        prompt_ids = []
    else:
        prompt_budget = max_len - len(completion_ids)
        prompt_ids = prompt_ids[-prompt_budget:]

    input_ids = prompt_ids + completion_ids
    labels = [-100] * len(prompt_ids) + completion_ids.copy()
    attention_mask = [1] * len(input_ids)

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        pad_id = 0

    padding = max_len - len(input_ids)
    if padding > 0:
        input_ids = input_ids + [pad_id] * padding
        attention_mask = attention_mask + [0] * padding
        labels = labels + [-100] * padding

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def build_training_dataset_from_records(records, tokenizer, max_len: int, fewshot: str):
    examples = [
        {
            "prompt": format_prompt_text(record["instruction"], fewshot),
            "completion": json.dumps(record["gt"], ensure_ascii=False),
        }
        for record in records
    ]
    dataset = Dataset.from_list(examples)

    def tokenize(example):
        return tokenize_prompt_completion(
            example["prompt"],
            example["completion"],
            tokenizer=tokenizer,
            max_len=max_len,
        )

    dataset = dataset.map(tokenize, remove_columns=["prompt", "completion"])
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return dataset
