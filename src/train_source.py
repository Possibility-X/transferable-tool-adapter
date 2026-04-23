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
from peft import LoraConfig, get_peft_model


# =========================
# Defaults
# =========================

DEFAULT_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_MAX_LEN = 256
DEFAULT_TRAIN_SAMPLES = 3000
DEFAULT_EVAL_SAMPLES = 100
DEFAULT_SEED = 42

DEFAULT_OUTPUT_DIR = "outputs/source_train"
DEFAULT_ADAPTER_DIR = "adapters/adapter_source_full"
DEFAULT_RESULTS_PATH = "results/source_in_domain.json"


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


def format_inference_prompt(instr: str) -> str:
    return (
        f"{FEWSHOT}\n\n"
        f"User: {instr}\n"
        f"Assistant:\n"
        f"TOOL_CALL:\n"
    )


def generate_sample():
    instr, gt = build_instruction_and_gt()
    return {"text": format_training_text(instr, gt)}


def build_dataset(n: int):
    return Dataset.from_list([generate_sample() for _ in range(n)])


def is_valid_schema(obj):
    return (
        isinstance(obj, dict)
        and set(obj.keys()) == {"tool", "arguments"}
        and isinstance(obj["tool"], str)
        and isinstance(obj["arguments"], dict)
    )


def extract_first_balanced_json(text: str):
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):
        ch = text[i]

        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start:i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    return None

    return None


@torch.no_grad()
def generate(model, tokenizer, prompt: str):
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    outputs = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )

    new_tokens = outputs[0][input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def evaluate(model, tokenizer, n=100, verbose_examples=5):
    correct_tool = 0
    correct_args = 0.0
    total = 0
    shown = 0

    for _ in range(n):
        instr, gt = build_instruction_and_gt()
        prompt = format_inference_prompt(instr)

        pred_text = generate(model, tokenizer, prompt)
        pred_json = extract_first_balanced_json(pred_text)

        if pred_json is None or not is_valid_schema(pred_json):
            if shown < verbose_examples:
                print("\n[Parse Failed]")
                print("Instruction:", instr)
                print("Raw output:", pred_text)
                print("Parsed:", pred_json)
                shown += 1
            continue

        total += 1

        if pred_json["tool"] == gt["tool"]:
            correct_tool += 1

        pred_args = pred_json["arguments"]
        arg_match = 0
        for k, v in gt["arguments"].items():
            if pred_args.get(k) == v:
                arg_match += 1

        correct_args += arg_match / len(gt["arguments"])

        if shown < verbose_examples:
            print("\n[Example]")
            print("Instruction:", instr)
            print("GT:", gt)
            print("Pred raw:", pred_text)
            print("Pred parsed:", pred_json)
            shown += 1

    result = {
        "parsed": total / n,
        "tool_acc": correct_tool / total if total else 0.0,
        "arg_acc": correct_args / total if total else 0.0,
    }

    print("\n=== In-domain Evaluation ===")
    print(result)
    return result


# =========================
# Main
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--max-len", type=int, default=DEFAULT_MAX_LEN)
    parser.add_argument("--train-samples", type=int, default=DEFAULT_TRAIN_SAMPLES)
    parser.add_argument("--eval-samples", type=int, default=DEFAULT_EVAL_SAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--adapter-dir", type=str, default=DEFAULT_ADAPTER_DIR)
    parser.add_argument("--results-path", type=str, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)

    ensure_parent(args.results_path)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.adapter_dir).mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
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

    dataset = build_dataset(args.train_samples)

    def tokenize(example):
        full_text = example["text"]

        split_marker = "Assistant:\nTOOL_CALL:\n"
        split_idx = full_text.rfind(split_marker)
        if split_idx == -1:
            raise ValueError("split marker not found in training text")

        prompt_text = full_text[: split_idx + len(split_marker)]

        full_enc = tokenizer(
            full_text,
            truncation=True,
            padding="max_length",
            max_length=args.max_len,
        )
        prompt_enc = tokenizer(
            prompt_text,
            truncation=True,
            padding=False,
            max_length=args.max_len,
        )

        input_ids = full_enc["input_ids"]
        attention_mask = full_enc["attention_mask"]
        labels = input_ids.copy()

        prompt_len = len(prompt_enc["input_ids"])

        # prompt 不参与 loss
        for i in range(min(prompt_len, len(labels))):
            labels[i] = -100

        # padding 不参与 loss
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

    training_args = TrainingArguments(
        output_dir=args.output_dir,
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

    trainer.train()

    model.save_pretrained(args.adapter_dir)
    tokenizer.save_pretrained(args.adapter_dir)
    print(f"\nSaved adapter to: {args.adapter_dir}")

    if not args.skip_eval:
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = True

        result = evaluate(model, tokenizer, n=args.eval_samples)

        with open(args.results_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Saved results to: {args.results_path}")


if __name__ == "__main__":
    main()