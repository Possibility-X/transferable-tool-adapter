import argparse
import json

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_ood import extract_first_balanced_json, is_valid_schema
from tool_data import load_jsonl_records
from train_transfer import FEWSHOT, ensure_parent, use_fp16_training


DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_DATASET_PATH = "data/toolbench_eval.jsonl"


def format_inference_prompt(instruction: str):
    return (
        f"{FEWSHOT}\n\n"
        f"User: {instruction}\n"
        "Assistant:\n"
        "TOOL_CALL:\n"
    )


def get_input_device(model):
    try:
        return model.get_input_embeddings().weight.device
    except AttributeError:
        return next(model.parameters()).device


@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new_tokens: int, max_input_tokens: int):
    model.eval()
    input_device = get_input_device(model)
    old_truncation_side = tokenizer.truncation_side
    tokenizer.truncation_side = "left"
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens,
    ).to(input_device)
    tokenizer.truncation_side = old_truncation_side
    input_len = inputs["input_ids"].shape[1]

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    new_tokens = outputs[0][input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def score_args(pred_args: dict, gt_args: dict):
    if not gt_args:
        return 1.0
    matches = sum(pred_args.get(key) == value for key, value in gt_args.items())
    return matches / len(gt_args)


def evaluate(
    model,
    tokenizer,
    records,
    max_new_tokens: int,
    max_input_tokens: int,
    verbose_examples: int,
):
    parsed = 0
    correct_tool = 0
    correct_args = 0.0
    shown = 0

    for record in records:
        prompt = format_inference_prompt(record["instruction"])
        pred_text = generate(model, tokenizer, prompt, max_new_tokens, max_input_tokens)
        pred_json = extract_first_balanced_json(pred_text)
        gt = record["gt"]

        if pred_json is None or not is_valid_schema(pred_json):
            if shown < verbose_examples:
                print("\n[Parse Failed]")
                print("Instruction:", record["instruction"][:500])
                print("GT:", gt)
                print("Raw:", pred_text)
                shown += 1
            continue

        parsed += 1
        if pred_json["tool"] == gt["tool"]:
            correct_tool += 1
        correct_args += score_args(pred_json["arguments"], gt["arguments"])

        if shown < verbose_examples:
            print("\n[Example]")
            print("Instruction:", record["instruction"][:500])
            print("GT:", gt)
            print("Pred:", pred_json)
            shown += 1

    total = len(records)
    return {
        "samples": total,
        "max_input_tokens": max_input_tokens,
        "parsed": parsed / total if total else 0.0,
        "tool_acc": correct_tool / parsed if parsed else 0.0,
        "arg_acc": correct_args / parsed if parsed else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--adapter", type=str, default=None)
    parser.add_argument("--dataset-path", type=str, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-input-tokens", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--verbose-examples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    records = load_jsonl_records(args.dataset_path, limit=args.max_samples)

    tokenizer_path = args.adapter or args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    eval_dtype = torch.float16 if use_fp16_training(args.model) else torch.float32
    eval_device_map = {"": 0} if eval_dtype == torch.float16 else "auto"
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=eval_dtype,
        device_map=eval_device_map,
    )
    model = PeftModel.from_pretrained(base_model, args.adapter) if args.adapter else base_model

    result = evaluate(
        model=model,
        tokenizer=tokenizer,
        records=records,
        max_new_tokens=args.max_new_tokens,
        max_input_tokens=args.max_input_tokens,
        verbose_examples=args.verbose_examples,
    )
    result.update(
        {
            "model": args.model,
            "adapter": args.adapter,
            "dataset_path": args.dataset_path,
        }
    )

    print("\n=== ToolBench Evaluation ===")
    print(result)

    if args.save:
        ensure_parent(args.save)
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nSaved to {args.save}")


if __name__ == "__main__":
    main()
