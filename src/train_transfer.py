import json
import random
import argparse
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, PeftModel
from tool_data import (
    build_training_dataset_from_records,
    load_jsonl_records,
)


# =========================
# Defaults
# =========================

DEFAULT_SOURCE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_TARGET_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

DEFAULT_SOURCE_ADAPTER = "adapters/adapter_source_full"

DEFAULT_MAX_LEN = 256
DEFAULT_SPLIT_RATIO = 0.5
DEFAULT_TRAIN_SAMPLES = 3000
DEFAULT_SEED = 42

DEFAULT_TRANSFER_OUTPUT = "adapters/adapter_target_transfer"
DEFAULT_FULL_OUTPUT = "adapters/adapter_target_full"

DEFAULT_TRANSFER_TRAIN_OUTPUT = "outputs/target_transfer_train"
DEFAULT_FULL_TRAIN_OUTPUT = "outputs/target_full_train"

DEFAULT_TRANSFER_SUMMARY = "results/target_transfer_train_summary.json"
DEFAULT_FULL_SUMMARY = "results/target_full_train_summary.json"


# =========================
# Data config
# =========================

LOCATIONS = ["Tokyo", "New York", "Paris", "London"]
TIMES = ["now", "today", "tomorrow"]

SYSTEM_HINT = (
    "You are a tool caller. "
    "You must output exactly one JSON object. "
    "The JSON must have exactly two top-level keys: tool, arguments. "
    "Do not add explanations. Do not add timestamps. "
    "Do not paraphrase argument values."
)

FEWSHOT = """System: You are a tool caller. You must output exactly one JSON object. The JSON must have exactly two top-level keys: tool, arguments. Do not add explanations. Do not add timestamps. Do not paraphrase argument values.
User: What's the weather in Tokyo today?
Assistant:
TOOL_CALL:
{"tool": "weather_api", "arguments": {"location": "Tokyo", "time": "today"}}

User: Calculate 3 * 4.
Assistant:
TOOL_CALL:
{"tool": "calculator", "arguments": {"expression": "3 * 4"}}"""


# =========================
# Helpers
# =========================

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)


def ensure_parent(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def gen_expr():
    a, b = random.randint(1, 50), random.randint(1, 50)
    op = random.choice(["+", "-", "*"])
    return f"{a} {op} {b}"


def build_instruction_and_gt():
    if random.random() < 0.5:
        loc = random.choice(LOCATIONS)
        t = random.choice(TIMES)
        instr = f"What's the weather in {loc} {t}?"
        gt = {
            "tool": "weather_api",
            "arguments": {
                "location": loc,
                "time": t,
            },
        }
    else:
        expr = gen_expr()
        instr = f"Calculate {expr}."
        gt = {
            "tool": "calculator",
            "arguments": {
                "expression": expr,
            },
        }
    return instr, gt


def format_training_text(instr: str, gt: dict) -> str:
    return (
        f"{FEWSHOT}\n\n"
        f"User: {instr}\n"
        f"Assistant:\n"
        f"TOOL_CALL:\n"
        f"{json.dumps(gt, ensure_ascii=False)}"
    )


def generate_sample():
    instr, gt = build_instruction_and_gt()
    return {"text": format_training_text(instr, gt)}


def build_dataset(n: int):
    return Dataset.from_list([generate_sample() for _ in range(n)])


def get_split_layer(model, split_ratio: float):
    max_layer = 0
    for name, _ in model.named_parameters():
        if "layers." in name:
            layer_id = int(name.split("layers.")[1].split(".")[0])
            max_layer = max(max_layer, layer_id)
    return int((max_layer + 1) * split_ratio)


def freeze_tool_layers(model, split_layer: int):
    for name, param in model.named_parameters():
        if "lora" in name and "layers." in name:
            layer_id = int(name.split("layers.")[1].split(".")[0])
            if layer_id >= split_layer:
                param.requires_grad = False


def print_trainable_summary(model):
    trainable = 0
    total = 0
    for _, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    pct = 100 * trainable / total if total else 0.0
    print(f"Trainable params: {trainable:,} / {total:,} ({pct:.4f}%)")
    return trainable, total, pct


def verify_late_layers_frozen(model, split_layer: int, max_show: int = 10):
    shown = 0
    print("\n[Freeze Check]")
    for name, param in model.named_parameters():
        if "lora" in name and "layers." in name:
            layer_id = int(name.split("layers.")[1].split(".")[0])
            if layer_id >= split_layer:
                print(f"{name}: requires_grad={param.requires_grad}")
                shown += 1
                if shown >= max_show:
                    break


def load_source_late_lora_weights(
    target_model,
    source_model_name: str,
    source_adapter_path: str,
    split_layer: int,
):
    """
    Load late-layer LoRA params from a trained source adapter into target model.
    Assumes source and target are at least layer-name compatible.
    """
    print(f"Loading source adapter from: {source_adapter_path}")

    source_base = AutoModelForCausalLM.from_pretrained(
        source_model_name,
        dtype=torch.float32,
        device_map="cpu",
    )
    source_model = PeftModel.from_pretrained(source_base, source_adapter_path)

    source_named = dict(source_model.named_parameters())
    target_named = dict(target_model.named_parameters())

    copied = 0
    skipped = 0

    for name, target_param in target_named.items():
        if "lora" not in name or "layers." not in name:
            continue

        layer_id = int(name.split("layers.")[1].split(".")[0])
        if layer_id < split_layer:
            continue

        if name not in source_named:
            skipped += 1
            continue

        source_param = source_named[name]
        if source_param.shape != target_param.shape:
            skipped += 1
            continue

        target_param.data.copy_(source_param.data.to(target_param.device))
        copied += 1

    print(f"Copied late-layer LoRA params: {copied}")
    print(f"Skipped params: {skipped}")

    return copied, skipped


def build_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_lora_model(model_name: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,
        device_map="auto",
    )

    model.gradient_checkpointing_enable()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    return model


def build_training_dataset(tokenizer, train_samples: int, max_len: int, dataset_path: str | None = None, seed: int | None = None):
    if dataset_path:
        records = load_jsonl_records(dataset_path, limit=train_samples, seed=seed)
        return build_training_dataset_from_records(
            records=records,
            tokenizer=tokenizer,
            max_len=max_len,
            fewshot=FEWSHOT,
        )

    dataset = build_dataset(train_samples)

    def tokenize(example):
        full_text = example["text"]

        split_marker = "Assistant:\nTOOL_CALL:\n"
        split_idx = full_text.rfind(split_marker)
        if split_idx == -1:
            raise ValueError("split marker not found")

        prompt_text = full_text[: split_idx + len(split_marker)]

        full_enc = tokenizer(
            full_text,
            truncation=True,
            padding="max_length",
            max_length=max_len,
        )
        prompt_enc = tokenizer(
            prompt_text,
            truncation=True,
            padding=False,
            max_length=max_len,
        )

        input_ids = full_enc["input_ids"]
        attention_mask = full_enc["attention_mask"]
        labels = input_ids.copy()

        prompt_len = len(prompt_enc["input_ids"])

        # prompt part does not contribute to loss
        for i in range(min(prompt_len, len(labels))):
            labels[i] = -100

        # padding does not contribute to loss
        for i, m in enumerate(attention_mask):
            if m == 0:
                labels[i] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    dataset = dataset.map(tokenize, remove_columns=["text"])
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return dataset


def write_summary(path: str, payload: dict):
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Saved summary to: {path}")


# =========================
# Main
# =========================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        type=str,
        choices=["transfer", "full"],
        default="transfer",
        help="transfer: load source late layers + freeze later layers; full: train full target LoRA",
    )

    parser.add_argument("--source-model", type=str, default=DEFAULT_SOURCE_MODEL)
    parser.add_argument("--target-model", type=str, default=DEFAULT_TARGET_MODEL)
    parser.add_argument("--source-adapter", type=str, default=DEFAULT_SOURCE_ADAPTER)

    parser.add_argument("--max-len", type=int, default=DEFAULT_MAX_LEN)
    parser.add_argument("--split-ratio", type=float, default=DEFAULT_SPLIT_RATIO)
    parser.add_argument("--train-samples", type=int, default=DEFAULT_TRAIN_SAMPLES)
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Optional JSONL records with instruction and gt/tool/arguments fields.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)

    parser.add_argument("--adapter-dir", type=str, default=None)
    parser.add_argument("--train-output-dir", type=str, default=None)
    parser.add_argument("--summary-path", type=str, default=None)

    args = parser.parse_args()

    set_seed(args.seed)

    mode = args.mode
    is_transfer = mode == "transfer"

    adapter_dir = args.adapter_dir or (
        DEFAULT_TRANSFER_OUTPUT if is_transfer else DEFAULT_FULL_OUTPUT
    )
    train_output_dir = args.train_output_dir or (
        DEFAULT_TRANSFER_TRAIN_OUTPUT if is_transfer else DEFAULT_FULL_TRAIN_OUTPUT
    )
    summary_path = args.summary_path or (
        DEFAULT_TRANSFER_SUMMARY if is_transfer else DEFAULT_FULL_SUMMARY
    )

    ensure_dir(adapter_dir)
    ensure_dir(train_output_dir)
    ensure_parent(summary_path)

    tokenizer = build_tokenizer(args.target_model)
    model = build_lora_model(args.target_model)

    split_layer = get_split_layer(model, args.split_ratio)
    print("Split layer =", split_layer)

    copied = 0
    skipped = 0

    if is_transfer:
        copied, skipped = load_source_late_lora_weights(
            target_model=model,
            source_model_name=args.source_model,
            source_adapter_path=args.source_adapter,
            split_layer=split_layer,
        )
        freeze_tool_layers(model, split_layer)
        print("Mode: transfer (load source late layers + freeze later layers)")
        verify_late_layers_frozen(model, split_layer)
    else:
        print("Mode: full target training")

    trainable, total, pct = print_trainable_summary(model)

    dataset = build_training_dataset(
        tokenizer=tokenizer,
        train_samples=args.train_samples,
        max_len=args.max_len,
        dataset_path=args.dataset_path,
        seed=args.seed,
    )

    training_args = TrainingArguments(
        output_dir=train_output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=2,
        learning_rate=2e-4,
        logging_steps=20,
        save_steps=200,
        fp16=False,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    train_result = trainer.train()

    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"\nSaved adapter to: {adapter_dir}")

    summary = {
        "mode": mode,
        "source_model": args.source_model,
        "target_model": args.target_model,
        "source_adapter": args.source_adapter if is_transfer else None,
        "adapter_dir": adapter_dir,
        "train_output_dir": train_output_dir,
        "split_layer": split_layer,
        "trainable_params": trainable,
        "total_params": total,
        "trainable_ratio_pct": pct,
        "copied_late_params": copied if is_transfer else None,
        "skipped_late_params": skipped if is_transfer else None,
        "train_loss": float(train_result.training_loss),
        "global_step": int(train_result.global_step),
        "train_samples": args.train_samples,
        "dataset_path": args.dataset_path,
        "max_len": args.max_len,
        "seed": args.seed,
    }

    write_summary(summary_path, summary)


if __name__ == "__main__":
    main()
