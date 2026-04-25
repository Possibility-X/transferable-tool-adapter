import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments
from transformers import logging as hf_logging

from eval_ood import extract_first_balanced_json, is_valid_schema
from tool_data import build_training_dataset_from_records, load_jsonl_records
from train_transfer import (
    FEWSHOT,
    build_instruction_and_gt,
    build_lora_model,
    build_tokenizer,
    ensure_dir,
    ensure_parent,
    print_trainable_summary,
    set_seed,
    write_summary,
)


DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_TEACHER_ADAPTER = "adapters/adapter_source_full"
DEFAULT_ADAPTER_DIR = "adapters/adapter_distill"
DEFAULT_TRAIN_OUTPUT_DIR = "outputs/target_distill_train"
DEFAULT_PSEUDO_PATH = "outputs/target_distill_pseudo_labels.jsonl"
DEFAULT_SUMMARY_PATH = "results/distill_train_summary.json"

DEFAULT_MAX_LEN = 256
DEFAULT_TRAIN_SAMPLES = 3000
DEFAULT_TEACHER_BATCH_SIZE = 4
DEFAULT_SEED = 42


def format_inference_prompt(instr: str) -> str:
    return (
        f"{FEWSHOT}\n\n"
        f"User: {instr}\n"
        "Assistant:\n"
        "TOOL_CALL:\n"
    )


def get_input_device(model):
    try:
        return model.get_input_embeddings().weight.device
    except AttributeError:
        return next(model.parameters()).device


@torch.no_grad()
def generate_batch(model, tokenizer, prompts, max_len: int):
    tokenizer.padding_side = "left"
    old_truncation_side = tokenizer.truncation_side
    tokenizer.truncation_side = "left"
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len,
    )
    tokenizer.truncation_side = old_truncation_side
    input_device = get_input_device(model)
    inputs = {key: value.to(input_device) for key, value in inputs.items()}
    input_width = inputs["input_ids"].shape[1]

    outputs = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )

    new_tokens = outputs[:, input_width:]
    return tokenizer.batch_decode(new_tokens, skip_special_tokens=True)


def score_teacher(pred_json: dict, gt: dict):
    tool_ok = int(pred_json.get("tool") == gt["tool"])
    if not gt["arguments"]:
        return tool_ok, 1.0
    arg_match = sum(
        pred_json.get("arguments", {}).get(k) == v
        for k, v in gt["arguments"].items()
    )
    arg_score = arg_match / len(gt["arguments"])
    return tool_ok, arg_score


def write_pseudo_records(path: str, records):
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            payload = {
                "instruction": record["instruction"],
                "gt": record["gt"],
                "teacher": record["teacher"],
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    print(f"Saved pseudo labels to: {path}")


def load_pseudo_records(path: str):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            payload = json.loads(line)
            teacher = payload["teacher"]
            records.append(
                {
                    "instruction": payload["instruction"],
                    "gt": payload["gt"],
                    "teacher": teacher,
                }
            )
    print(f"Loaded pseudo labels from: {path}")
    return records


def build_teacher(model_name: str, adapter_path: str):
    tokenizer = build_tokenizer(adapter_path)
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True

    return model, tokenizer


def build_pseudo_dataset(args):
    pseudo_file = Path(args.pseudo_path)
    if args.reuse_pseudo and pseudo_file.exists():
        records = load_pseudo_records(args.pseudo_path)
        return records[: args.train_samples], {
            "teacher_attempted": None,
            "teacher_parsed": None,
            "teacher_parse_rate": None,
            "teacher_tool_acc_against_gt": None,
            "teacher_arg_acc_against_gt": None,
            "filter_invalid_teacher_outputs": True,
            "teacher_invalid_filtered": None,
        }

    teacher, tokenizer = build_teacher(args.teacher_model, args.teacher_adapter)
    print(f"Teacher input device: {get_input_device(teacher)}")

    records = []
    attempted = 0
    parsed = 0
    correct_tool = 0
    correct_args = 0.0
    max_attempts = args.train_samples * args.max_attempt_factor
    source_records = None
    source_index = 0

    if args.dataset_path:
        source_records = load_jsonl_records(
            args.dataset_path,
            limit=max_attempts,
            seed=args.seed,
        )
        print(f"Loaded source records for distillation: {len(source_records)}")

    while len(records) < args.train_samples and attempted < max_attempts:
        batch_items = []
        if source_records is None:
            for _ in range(args.teacher_batch_size):
                instr, gt = build_instruction_and_gt()
                batch_items.append((instr, gt))
        else:
            while (
                len(batch_items) < args.teacher_batch_size
                and source_index < len(source_records)
            ):
                record = source_records[source_index]
                source_index += 1
                batch_items.append((record["instruction"], record["gt"]))

        if not batch_items:
            break

        prompts = [format_inference_prompt(instr) for instr, _ in batch_items]
        outputs = generate_batch(teacher, tokenizer, prompts, args.max_len)

        for (instr, gt), pred_text in zip(batch_items, outputs):
            attempted += 1
            pred_json = extract_first_balanced_json(pred_text)

            if pred_json is None or not is_valid_schema(pred_json):
                continue

            parsed += 1
            tool_ok, arg_score = score_teacher(pred_json, gt)
            correct_tool += tool_ok
            correct_args += arg_score

            records.append(
                {
                    "instruction": instr,
                    "gt": gt,
                    "teacher": pred_json,
                }
            )

            if len(records) % 100 == 0:
                print(f"Accepted pseudo labels: {len(records)} / {args.train_samples}")

            if len(records) >= args.train_samples:
                break

    del teacher
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if len(records) < args.train_samples and source_records is None:
        raise RuntimeError(
            f"Only collected {len(records)} pseudo labels after {attempted} attempts"
        )
    if len(records) < args.train_samples:
        print(
            f"Using {len(records)} valid pseudo labels after filtering "
            f"{attempted - parsed} invalid teacher outputs."
        )

    write_pseudo_records(args.pseudo_path, records)

    stats = {
        "teacher_attempted": attempted,
        "teacher_parsed": parsed,
        "teacher_parse_rate": parsed / attempted if attempted else 0.0,
        "teacher_tool_acc_against_gt": correct_tool / parsed if parsed else 0.0,
        "teacher_arg_acc_against_gt": correct_args / parsed if parsed else 0.0,
        "filter_invalid_teacher_outputs": True,
        "teacher_invalid_filtered": attempted - parsed,
    }
    print("\n=== Teacher Pseudo-Label Stats ===")
    print(stats)
    return records, stats


def build_training_dataset(records, tokenizer, max_len: int):
    training_records = [
        {
            "instruction": record["instruction"],
            "gt": record["teacher"],
        }
        for record in records
    ]
    return build_training_dataset_from_records(
        training_records,
        tokenizer=tokenizer,
        max_len=max_len,
        fewshot=FEWSHOT,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--target-model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--teacher-adapter", type=str, default=DEFAULT_TEACHER_ADAPTER)
    parser.add_argument("--adapter-dir", type=str, default=DEFAULT_ADAPTER_DIR)
    parser.add_argument(
        "--train-output-dir",
        "--output-dir",
        dest="train_output_dir",
        type=str,
        default=DEFAULT_TRAIN_OUTPUT_DIR,
    )
    parser.add_argument("--pseudo-path", type=str, default=DEFAULT_PSEUDO_PATH)
    parser.add_argument("--summary-path", type=str, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--max-len", type=int, default=DEFAULT_MAX_LEN)
    parser.add_argument("--train-samples", type=int, default=DEFAULT_TRAIN_SAMPLES)
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Optional JSONL records with instruction and gt/tool/arguments fields.",
    )
    parser.add_argument("--teacher-batch-size", type=int, default=DEFAULT_TEACHER_BATCH_SIZE)
    parser.add_argument("--max-attempt-factor", type=int, default=3)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--reuse-pseudo", action="store_true")
    args = parser.parse_args()

    hf_logging.set_verbosity_error()
    set_seed(args.seed)

    ensure_dir(args.adapter_dir)
    ensure_dir(args.train_output_dir)
    ensure_parent(args.summary_path)
    ensure_parent(args.pseudo_path)

    records, teacher_stats = build_pseudo_dataset(args)

    tokenizer = build_tokenizer(args.target_model)
    model = build_lora_model(args.target_model)
    trainable, total, pct = print_trainable_summary(model)

    dataset = build_training_dataset(records, tokenizer, args.max_len)

    training_args = TrainingArguments(
        output_dir=args.train_output_dir,
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

    model.save_pretrained(args.adapter_dir)
    tokenizer.save_pretrained(args.adapter_dir)
    print(f"\nSaved adapter to: {args.adapter_dir}")

    summary = {
        "mode": "distill",
        "teacher_model": args.teacher_model,
        "target_model": args.target_model,
        "teacher_adapter": args.teacher_adapter,
        "adapter_dir": args.adapter_dir,
        "train_output_dir": args.train_output_dir,
        "pseudo_path": args.pseudo_path,
        "pseudo_samples": len(records),
        "dataset_path": args.dataset_path,
        "trainable_params": trainable,
        "total_params": total,
        "trainable_ratio_pct": pct,
        "train_loss": float(train_result.training_loss),
        "global_step": int(train_result.global_step),
        "train_samples": args.train_samples,
        "max_len": args.max_len,
        "teacher_batch_size": args.teacher_batch_size,
        "seed": args.seed,
    }
    summary.update(teacher_stats)
    write_summary(args.summary_path, summary)


if __name__ == "__main__":
    main()
