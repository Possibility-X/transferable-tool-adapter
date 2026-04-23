import json
import random
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# =========================
# Config (default)
# =========================

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_ADAPTER_PATH = "adapters/adapter_target_full"
EVAL_SAMPLES = 200
SEED = 42

# =========================
# Prompt
# =========================

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
# OOD generation
# =========================

OOD_LOCATIONS = ["Berlin", "Sydney", "Moscow", "Beijing"]
OOD_TIMES = ["yesterday", "next week", "tonight"]

OOD_WEATHER_TEMPLATES = [
    "What's the weather in {location} {time}?",
    "Give me the weather for {location} {time}.",
    "I need the weather in {location} {time}.",
]

OOD_CALC_TEMPLATES = [
    "Calculate {expr}.",
    "Compute {expr}.",
    "What is {expr}?",
]


def format_inference_prompt(instr: str) -> str:
    return (
        f"{FEWSHOT}\n\n"
        f"User: {instr}\n"
        f"Assistant:\n"
        f"TOOL_CALL:\n"
    )


def gen_ood_expr():
    kind = random.choice(["plain", "paren"])
    if kind == "plain":
        a, b = random.randint(60, 120), random.randint(60, 120)
        op = random.choice(["+", "-", "*"])
        return f"{a} {op} {b}"
    else:
        a, b, c = random.randint(10, 30), random.randint(10, 30), random.randint(2, 9)
        op = random.choice(["+", "-"])
        return f"({a} {op} {b}) * {c}"


def build_ood_instruction_and_gt():
    if random.random() < 0.5:
        loc = random.choice(OOD_LOCATIONS)
        t = random.choice(OOD_TIMES)
        template = random.choice(OOD_WEATHER_TEMPLATES)
        instr = template.format(location=loc, time=t)
        gt = {
            "tool": "weather_api",
            "arguments": {"location": loc, "time": t},
        }
    else:
        expr = gen_ood_expr()
        template = random.choice(OOD_CALC_TEMPLATES)
        instr = template.format(expr=expr)
        gt = {
            "tool": "calculator",
            "arguments": {"expression": expr},
        }
    return instr, gt


# =========================
# JSON extraction
# =========================

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


def is_valid_schema(obj):
    return (
        isinstance(obj, dict)
        and set(obj.keys()) == {"tool", "arguments"}
        and isinstance(obj["tool"], str)
        and isinstance(obj["arguments"], dict)
    )


# =========================
# Generation
# =========================

@torch.no_grad()
def generate(model, tokenizer, prompt, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    outputs = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    new_tokens = outputs[0][input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# =========================
# Main
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapter",
        type=str,
        default=DEFAULT_ADAPTER_PATH,
        help="Path to adapter",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Save results to JSON file",
    )

    args = parser.parse_args()

    random.seed(SEED)
    torch.manual_seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map="auto",
    )

    model = PeftModel.from_pretrained(base_model, args.adapter)
    model.eval()

    parsed = 0
    correct_tool = 0
    correct_args = 0.0

    for i in range(EVAL_SAMPLES):
        instr, gt = build_ood_instruction_and_gt()
        prompt = format_inference_prompt(instr)

        pred_text = generate(model, tokenizer, prompt, device)
        pred_json = extract_first_balanced_json(pred_text)

        if pred_json is None or not is_valid_schema(pred_json):
            if i < 5:
                print("\n[Parse Failed]")
                print("Instruction:", instr)
                print("Raw:", pred_text)
            continue

        parsed += 1

        if pred_json["tool"] == gt["tool"]:
            correct_tool += 1

        match = sum(
            pred_json["arguments"].get(k) == v
            for k, v in gt["arguments"].items()
        )
        correct_args += match / len(gt["arguments"])

        if i < 5:
            print("\n[Example]")
            print("Instruction:", instr)
            print("GT:", gt)
            print("Pred:", pred_json)

    result = {
        "parsed": parsed / EVAL_SAMPLES,
        "tool_acc": correct_tool / parsed if parsed else 0,
        "arg_acc": correct_args / parsed if parsed else 0,
    }

    print("\n=== OOD Evaluation ===")
    print(result)

    if args.save:
        with open(args.save, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved to {args.save}")


if __name__ == "__main__":
    main()