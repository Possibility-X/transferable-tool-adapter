import argparse
import ast
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
DEFAULT_DATASET_PATH = "data/toolbench_eval.jsonl"

MCP_STYLE_INSTRUCTION = """MCP-style tool registry:
You are given a task context and a structured registry of available tools/APIs.
Choose exactly one registry entry and return a strict JSON tool call.
Output exactly one JSON object with exactly two top-level keys: tool, arguments.
The "tool" value must exactly match one registry name.
The "arguments" value must be a JSON object containing only arguments supported by that registry entry.
Use an empty arguments object when no arguments are required.
Do not output Thought, Action, Action Input, FUNCTION_CALL, markdown, explanations, or extra text.
The first character of your answer must be { and the last character must be }."""


def compact_text(value: str, max_chars: int = 360):
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def normalize_schema_tool(tool: dict):
    name = tool.get("name") or tool.get("tool") or tool.get("api_name")
    if not isinstance(name, str) or not name.strip():
        return None

    parameters = (
        tool.get("parameters")
        or tool.get("input_parameters")
        or tool.get("inputs")
        or {}
    )
    if not isinstance(parameters, dict):
        parameters = {}

    properties = parameters.get("properties") if isinstance(parameters.get("properties"), dict) else parameters
    if not isinstance(properties, dict):
        properties = {}

    required = parameters.get("required", tool.get("required", []))
    optional = parameters.get("optional", tool.get("optional", []))
    if not isinstance(required, list):
        required = []
    if not isinstance(optional, list):
        optional = []

    args = {}
    for arg_name, spec in properties.items():
        if not isinstance(spec, dict):
            args[arg_name] = {"type": compact_text(spec, 80)}
            continue
        args[arg_name] = {
            "type": compact_text(spec.get("type", ""), 80),
            "description": compact_text(spec.get("description", ""), 180),
        }
        if "enum" in spec:
            args[arg_name]["enum"] = spec["enum"]
        if "example_value" in spec:
            args[arg_name]["example"] = spec["example_value"]

    return {
        "name": name.strip(),
        "description": compact_text(tool.get("description", ""), 300),
        "arguments": args,
        "required": required,
        "optional": optional,
    }


def extract_json_objects_after_marker(text: str, marker: str):
    marker_idx = text.find(marker)
    if marker_idx < 0:
        return None, text

    prefix = text[:marker_idx].rstrip()
    tail = text[marker_idx + len(marker) :].lstrip()
    tools = []
    consumed = 0

    for match in re.finditer(r"\{", tail):
        start = match.start()
        if tail[consumed:start].strip():
            break
        obj = extract_first_balanced_json(tail[start:])
        if not isinstance(obj, dict):
            break
        tools.append(obj)
        raw = tail[start:]
        depth = 0
        end = None
        in_str = False
        escape = False
        for idx, char in enumerate(raw):
            if in_str:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_str = False
                continue
            if char == '"':
                in_str = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    end = idx + 1
                    break
        if end is None:
            break
        consumed = start + end

    context_tail = tail[consumed:].lstrip()
    context = f"{prefix}\n{context_tail}".strip()
    return tools, context


def extract_toolbench_tools(text: str):
    marker = "Specifically, you have access to the following APIs:"
    marker_idx = text.find(marker)
    if marker_idx < 0:
        return None, text

    list_start = text.find("[", marker_idx)
    if list_start < 0:
        return None, text

    depth = 0
    in_str = False
    quote = ""
    escape = False
    list_end = None
    for idx in range(list_start, len(text)):
        char = text[idx]
        if in_str:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == quote:
                in_str = False
            continue
        if char in ("'", '"'):
            in_str = True
            quote = char
        elif char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                list_end = idx + 1
                break

    if list_end is None:
        return None, text

    raw_list = text[list_start:list_end]
    try:
        tools = ast.literal_eval(raw_list)
    except (SyntaxError, ValueError):
        return None, text
    if not isinstance(tools, list):
        return None, text

    before = text[:marker_idx].rstrip()
    after = text[list_end:].lstrip()
    context = sanitize_toolbench_context(f"{before}\n{after}".strip())
    return tools, context


def extract_apibank_tools(text: str):
    tools, context = extract_json_objects_after_marker(text, "API descriptions:")
    return tools, sanitize_apibank_context(context)


def sanitize_apibank_context(context: str):
    context = re.sub(
        r"Generate an API request in the format of \[ApiName\(key1='value1', key2='value2', \.\.\.\)\] based on the previous dialogue context\.\n?",
        "Use the dialogue context to choose and call one API.\n",
        context,
    )
    context = re.sub(
        r"\nExpected output:\nAPI-Request: \[ApiName\(key1='value1', key2='value2', \.\.\.\)\]\n",
        "\n",
        context,
    )
    context = context.replace(
        "Generate API Request:",
        "Choose one API and return the JSON tool call:",
    )
    return re.sub(r"\n{3,}", "\n\n", context).strip()


def sanitize_toolbench_context(context: str):
    context = context.replace(
        "At each step, you need to give your thought to analyze the status now and what to do next, with a function call to actually excute your step. Your output should follow this format:\nThought:\nAction\nAction Input:\n",
        "Choose the next function call needed for the task.\n",
    )
    context = context.replace(
        "Do not use origin tool names, use only subfunctions' names.",
        "Use only registry function names.",
    )
    return re.sub(r"\n{3,}", "\n\n", context).strip()


def build_registry_and_context(instruction: str):
    tools, context = extract_apibank_tools(instruction)
    if not tools:
        tools, context = extract_toolbench_tools(instruction)

    if not tools:
        return [], instruction

    registry = []
    for tool in tools:
        if isinstance(tool, dict):
            normalized = normalize_schema_tool(tool)
            if normalized:
                registry.append(normalized)

    if not registry:
        return [], instruction
    return registry, context


def format_registry_block(registry):
    return json.dumps({"tools": registry}, ensure_ascii=False, separators=(",", ":"))


def chat_or_plain_prompt(content: str, tokenizer, use_chat_template: bool):
    if use_chat_template and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"{content}\n\nAssistant:\n"


def format_mcp_prompt(instruction: str, tokenizer, use_chat_template: bool):
    registry, context = build_registry_and_context(instruction)
    if registry:
        content = (
            f"{MCP_STYLE_INSTRUCTION}\n\n"
            f"Task context:\n{context}\n\n"
            f"Tool registry JSON:\n{format_registry_block(registry)}"
        )
    else:
        content = (
            f"{MCP_STYLE_INSTRUCTION}\n\n"
            "Task context and available tools:\n"
            f"{instruction}"
        )
    return chat_or_plain_prompt(content, tokenizer, use_chat_template)


def build_tokenizer(path: str):
    tokenizer = AutoTokenizer.from_pretrained(path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_model(model_name: str, adapter_path: str | None):
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,
        device_map="auto",
    )
    if adapter_path:
        return PeftModel.from_pretrained(base_model, adapter_path)
    return base_model


def evaluate(
    model,
    tokenizer,
    records,
    max_new_tokens: int,
    max_input_tokens: int,
    verbose_examples: int,
    use_chat_template: bool,
):
    parsed = 0
    correct_tool = 0
    correct_args = 0.0
    registry_extracted = 0
    shown = 0

    for record in records:
        registry, _ = build_registry_and_context(record["instruction"])
        if registry:
            registry_extracted += 1

        prompt = format_mcp_prompt(
            record["instruction"],
            tokenizer=tokenizer,
            use_chat_template=use_chat_template,
        )
        pred_text = generate(model, tokenizer, prompt, max_new_tokens, max_input_tokens)
        pred_json = extract_first_balanced_json(pred_text)
        gt = record["gt"]

        if pred_json is None or not is_valid_schema(pred_json):
            if shown < verbose_examples:
                print("\n[Parse Failed]")
                print("Instruction:", record["instruction"][:500])
                print("Registry extracted:", bool(registry))
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
            print("Registry extracted:", bool(registry))
            print("GT:", gt)
            print("Pred:", pred_json)
            shown += 1

    total = len(records)
    return {
        "samples": total,
        "max_input_tokens": max_input_tokens,
        "registry_extracted": registry_extracted / total if total else 0.0,
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
    parser.add_argument("--verbose-examples", type=int, default=5)
    parser.add_argument("--use-chat-template", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    records = load_jsonl_records(args.dataset_path, limit=args.max_samples)

    tokenizer_path = args.adapter or args.model
    tokenizer = build_tokenizer(tokenizer_path)
    model = build_model(args.model, args.adapter)

    result = evaluate(
        model=model,
        tokenizer=tokenizer,
        records=records,
        max_new_tokens=args.max_new_tokens,
        max_input_tokens=args.max_input_tokens,
        verbose_examples=args.verbose_examples,
        use_chat_template=args.use_chat_template,
    )
    result.update(
        {
            "model": args.model,
            "adapter": args.adapter,
            "method": "mcpstyle_hybrid_adapter" if args.adapter else "mcpstyle_prompt",
            "prompt_style": "mcpstyle_registry_chat" if args.use_chat_template else "mcpstyle_registry_plain",
            "use_chat_template": args.use_chat_template,
            "dataset_path": args.dataset_path,
        }
    )

    title = "MCP-style Hybrid Evaluation" if args.adapter else "MCP-style Prompt Evaluation"
    print(f"\n=== {title} ===")
    print(result)

    if args.save:
        ensure_parent(args.save)
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nSaved to {args.save}")


if __name__ == "__main__":
    main()
