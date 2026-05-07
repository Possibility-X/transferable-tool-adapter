import argparse
import json

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_ood import extract_first_balanced_json, is_valid_schema
from eval_toolbench import generate, score_args
from tool_data import load_jsonl_records
from train_transfer import FEWSHOT, ensure_parent


DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_DATASET_PATH = "data/apibank_eval.jsonl"


def format_inference_prompt(instruction: str):
    return (
        f"{FEWSHOT}\n\n"
        f"User: {instruction}\n"
        "Assistant:\n"
        "TOOL_CALL:\n"
    )


def infer_split_from_path(path: str):
    name = path.lower()
    if "train" in name:
        return "train"
    if "eval" in name or "test" in name:
        return "eval"
    return "eval"


def write_prediction_row(handle, record: dict, index: int, split: str, pred_text: str, parsed_pred):
    gt = record["gt"]
    row = {
        "id": record.get("source_id") or f"apibank:{split}:{index}",
        "index": index,
        "prediction": pred_text,
        "parsed_prediction": parsed_pred,
        "gold_call": {
            "name": gt["tool"],
            "arguments": gt["arguments"],
        },
        "legacy_gt": gt,
    }
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def evaluate(
    model,
    tokenizer,
    records,
    max_new_tokens: int,
    max_input_tokens: int,
    verbose_examples: int,
    save_predictions: str | None = None,
    prediction_split: str = "eval",
):
    parsed = 0
    correct_tool = 0
    correct_args = 0.0
    shown = 0
    prediction_handle = None
    if save_predictions:
        ensure_parent(save_predictions)
        prediction_handle = open(save_predictions, "w", encoding="utf-8", newline="\n")

    try:
        for index, record in enumerate(records):
            prompt = format_inference_prompt(record["instruction"])
            pred_text = generate(model, tokenizer, prompt, max_new_tokens, max_input_tokens)
            pred_json = extract_first_balanced_json(pred_text)
            gt = record["gt"]
            parsed_pred = pred_json if pred_json is not None and is_valid_schema(pred_json) else None

            if prediction_handle is not None:
                write_prediction_row(
                    prediction_handle,
                    record=record,
                    index=index,
                    split=prediction_split,
                    pred_text=pred_text,
                    parsed_pred=parsed_pred,
                )

            if parsed_pred is None:
                if shown < verbose_examples:
                    print("\n[Parse Failed]")
                    print("Instruction:", record["instruction"][:500])
                    print("GT:", gt)
                    print("Raw:", pred_text)
                    shown += 1
                continue

            parsed += 1
            if parsed_pred["tool"] == gt["tool"]:
                correct_tool += 1
            correct_args += score_args(parsed_pred["arguments"], gt["arguments"])

            if shown < verbose_examples:
                print("\n[Example]")
                print("Instruction:", record["instruction"][:500])
                print("GT:", gt)
                print("Pred:", parsed_pred)
                shown += 1
    finally:
        if prediction_handle is not None:
            prediction_handle.close()

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
    parser.add_argument("--max-input-tokens", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--save-predictions", type=str, default=None)
    parser.add_argument("--verbose-examples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    records = load_jsonl_records(args.dataset_path, limit=args.max_samples)

    tokenizer_path = args.adapter or args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float32,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, args.adapter) if args.adapter else base_model

    result = evaluate(
        model=model,
        tokenizer=tokenizer,
        records=records,
        max_new_tokens=args.max_new_tokens,
        max_input_tokens=args.max_input_tokens,
        verbose_examples=args.verbose_examples,
        save_predictions=args.save_predictions,
        prediction_split=infer_split_from_path(args.dataset_path),
    )
    result.update(
        {
            "model": args.model,
            "adapter": args.adapter,
            "dataset_path": args.dataset_path,
        }
    )

    print("\n=== API-Bank Evaluation ===")
    print(result)

    if args.save:
        ensure_parent(args.save)
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nSaved to {args.save}")


if __name__ == "__main__":
    main()
