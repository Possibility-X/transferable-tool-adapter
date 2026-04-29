import argparse
import json
import re

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_ood import extract_first_balanced_json, is_valid_schema
from eval_toolbench import generate, score_args
from tool_data import load_jsonl_records
from train_transfer import ensure_parent


DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_DATASET_PATH = "data/apibank_eval.jsonl"

ROUTER_INSTRUCTION = """Tool routing task:
The task above includes available tools or API descriptions.
Choose exactly one tool/API name to call next.
Output only the tool/API name, with no arguments, JSON, markdown, or extra text."""

PRIOR_INSTRUCTION = """Tool-call generation task:
The router selected this tool/API:
{tool_name}

Generate exactly one JSON object with exactly two top-level keys: tool, arguments.
The value of "tool" must be exactly "{tool_name}".
Fill "arguments" using the dialogue context and the selected tool schema.
Use an empty arguments object if no arguments are required.
Do not output Thought, Action, Action Input, FUNCTION_CALL, markdown, explanations, or extra text.
The first character of your answer must be {{ and the last character must be }}."""


def chat_or_plain_prompt(content: str, tokenizer, use_chat_template: bool):
    if use_chat_template and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"{content}\n\nAssistant:\n"


def format_router_prompt(instruction: str, tokenizer, use_chat_template: bool):
    content = (
        "Task and available tools:\n"
        f"{instruction}\n\n"
        f"{ROUTER_INSTRUCTION}"
    )
    return chat_or_plain_prompt(content, tokenizer, use_chat_template)


def format_prior_prompt(instruction: str, tool_name: str, tokenizer, use_chat_template: bool):
    content = (
        "Task and available tools:\n"
        f"{instruction}\n\n"
        f"{PRIOR_INSTRUCTION.format(tool_name=tool_name)}"
    )
    return chat_or_plain_prompt(content, tokenizer, use_chat_template)


def normalize_tool_name(raw_text: str):
    text = raw_text.strip()
    json_obj = extract_first_balanced_json(text)
    if isinstance(json_obj, dict):
        for key in ("tool", "name", "api", "api_name", "tool_name"):
            value = json_obj.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip().strip("`'\"")

    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    first_line = re.sub(r"^(tool|api|name|selected_tool)\s*[:=]\s*", "", first_line, flags=re.I)
    first_line = first_line.strip().strip("`'\"")
    match = re.search(r"[A-Za-z_][A-Za-z0-9_]*", first_line)
    return match.group(0) if match else first_line


def get_input_device(model):
    try:
        return model.get_input_embeddings().weight.device
    except AttributeError:
        return next(model.parameters()).device


def build_tokenizer(path: str):
    tokenizer = AutoTokenizer.from_pretrained(path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_base_model(model_name: str):
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,
        device_map="auto",
    )


def unload_model(model):
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def route_tools(
    model_name: str,
    records,
    max_input_tokens: int,
    max_router_tokens: int,
    use_chat_template: bool,
):
    tokenizer = build_tokenizer(model_name)
    model = build_base_model(model_name)
    routes = []

    for record in records:
        prompt = format_router_prompt(
            record["instruction"],
            tokenizer=tokenizer,
            use_chat_template=use_chat_template,
        )
        raw_text = generate(model, tokenizer, prompt, max_router_tokens, max_input_tokens)
        routes.append(
            {
                "tool": normalize_tool_name(raw_text),
                "raw": raw_text,
            }
        )

    unload_model(model)
    return routes


def evaluate_with_prior(
    model_name: str,
    adapter_path: str,
    records,
    routes,
    max_input_tokens: int,
    max_new_tokens: int,
    use_chat_template: bool,
    verbose_examples: int,
):
    tokenizer = build_tokenizer(adapter_path)
    base_model = build_base_model(model_name)
    model = PeftModel.from_pretrained(base_model, adapter_path)

    parsed = 0
    correct_tool = 0
    correct_args = 0.0
    router_correct = 0
    shown = 0

    for record, route in zip(records, routes):
        gt = record["gt"]
        routed_tool = route["tool"]
        if routed_tool == gt["tool"]:
            router_correct += 1

        prompt = format_prior_prompt(
            record["instruction"],
            tool_name=routed_tool,
            tokenizer=tokenizer,
            use_chat_template=use_chat_template,
        )
        pred_text = generate(model, tokenizer, prompt, max_new_tokens, max_input_tokens)
        pred_json = extract_first_balanced_json(pred_text)

        if pred_json is None or not is_valid_schema(pred_json):
            if shown < verbose_examples:
                print("\n[Parse Failed]")
                print("Instruction:", record["instruction"][:500])
                print("GT:", gt)
                print("Routed:", routed_tool)
                print("Router raw:", route["raw"])
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
            print("Routed:", routed_tool)
            print("Pred:", pred_json)
            shown += 1

    total = len(records)
    unload_model(model)
    return {
        "samples": total,
        "max_input_tokens": max_input_tokens,
        "router_acc": router_correct / total if total else 0.0,
        "parsed": parsed / total if total else 0.0,
        "tool_acc": correct_tool / parsed if parsed else 0.0,
        "arg_acc": correct_args / parsed if parsed else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--adapter", type=str, required=True)
    parser.add_argument("--dataset-path", type=str, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-input-tokens", type=int, default=1024)
    parser.add_argument("--max-router-tokens", type=int, default=24)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--verbose-examples", type=int, default=5)
    parser.add_argument("--use-chat-template", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    records = load_jsonl_records(args.dataset_path, limit=args.max_samples)

    routes = route_tools(
        model_name=args.model,
        records=records,
        max_input_tokens=args.max_input_tokens,
        max_router_tokens=args.max_router_tokens,
        use_chat_template=args.use_chat_template,
    )
    result = evaluate_with_prior(
        model_name=args.model,
        adapter_path=args.adapter,
        records=records,
        routes=routes,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
        use_chat_template=args.use_chat_template,
        verbose_examples=args.verbose_examples,
    )
    result.update(
        {
            "model": args.model,
            "adapter": args.adapter,
            "method": "hybrid_router_prior",
            "prompt_style": (
                "router_then_prior_chat"
                if args.use_chat_template
                else "router_then_prior_plain"
            ),
            "dataset_path": args.dataset_path,
        }
    )

    print("\n=== Hybrid Router-Prior Evaluation ===")
    print(result)

    if args.save:
        ensure_parent(args.save)
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nSaved to {args.save}")


if __name__ == "__main__":
    main()
