# Transferable Tool-Use Adapter

This repository studies whether **tool-use capability in LLMs can be modularized and transferred**.

We propose a **layerwise decomposition of LoRA adapters** into:

- Early layers → input alignment  
- Late layers → tool-use knowledge  

and demonstrate that:

> Tool knowledge can be **reused and transferred** by freezing late-layer adapters.

---

## Key Idea

We decompose tool-use into two components:

Input → Early Layers → Late Layers → JSON Tool Call

- Early layers adapt to input distribution  
- Late layers encode tool semantics and structured output  

---

## Method Overview

![Architecture](figures/architecture.svg)

---

## Main Contributions

1. Layerwise decomposition of tool-use  
2. Transfer via adapter reuse (freeze late layers)  
3. Empirical evidence that:
   - Tool selection is preserved  
   - JSON structure is stable  
   - Transfer matches or exceeds full retraining  

---

## Project Structure

transferable-tool-adapter/

- src/  
  - train_source.py  
  - train_transfer.py  
  - eval_ood.py  

- data/  
- adapters/        (ignored) trained LoRA adapters  
- outputs/         (ignored) training outputs  
- results/         evaluation results  

- pyproject.toml  
- uv.lock  
- README.md  

---

## Environment Setup (uv)

Install dependencies:

uv sync

Verify CUDA:

uv run python -c "import torch; print(torch.cuda.is_available())"

---

## Experiments

### Step 1 — Train Source Adapter

uv run python src/train_source.py

Output:

adapters/adapter_source_full

---

### Step 2 — Transfer Training

#### Transfer (freeze late layers)

uv run python src/train_transfer.py --mode transfer

#### Full Target Training (baseline)

uv run python src/train_transfer.py --mode full

---

### Step 3 — OOD Evaluation

uv run python src/eval_ood.py --adapter adapters/adapter_target_transfer --save results/transfer.json

---

## Results

| Setting   | Tool Acc | Arg Acc |
|----------|----------|---------|
| Source   | 1.00     | 0.71    |
| Transfer | 1.00     | 0.75    |
| Full     | 1.00     | 0.71    |

---

## Observations

- Tool selection is **perfectly preserved**  
- JSON output is **stable**  
- Transfer achieves **equal or better performance**  
- Errors mainly come from **argument normalization**  

---

## Analysis

We observe clear layer specialization:

- Early layers → input alignment  
- Late layers → tool knowledge  

This explains why freezing late layers still works.

---

## Transfer Mechanism

Source Model  
↓  
Late-layer LoRA (tool knowledge)  
↓ copy  
Target Model  
↓ freeze  
Train Early Layers  

---

## Notes

- Adapters and outputs are **not included** (see `.gitignore`)  
- Results are provided in `results/`  
- All experiments are reproducible via `uv`  

---

## Future Work

- Cross-model transfer (TinyLlama → Qwen)  
- Adapter projection across architectures  
- Multi-tool and real API tasks  

---

## License

MIT (or your choice)

---

## TODO (Research Backlog)

> Ongoing experiments and future directions.

### Experiments

- [ ] Split ratio ablation (vary early/late boundary)
- [ ] Multi-tool extension (more tools, more schemas)
- [ ] Prompt robustness (template / phrasing variation)
- [ ] Cross-model transfer (e.g., TinyLlama → other LLMs)

### Analysis

- [ ] Error breakdown (tool vs argument vs format)
- [ ] Layer-wise behavior analysis

### Figures (to be added later)

- [ ] Figure: Method overview (clean version)
- [ ] Figure: Source vs Transfer vs Full comparison
- [ ] Figure: Layer-wise ablation (split ratio)
- [ ] Figure: Error analysis
- [ ] Figure: Cross-model transfer (future)