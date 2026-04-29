# Experiment Workflow Guide

This document describes how to reproduce and manage experiments in this repository without committing local checkpoints or machine-specific artifacts.

## Milestone Tags

Use tags to recover major experiment states:

| Tag | Milestone |
| --- | --- |
| `v0.1-route-b` | Route B baseline |
| `v0.2-route-b-ablation` | split-ratio ablation |
| `v0.3-route-a-distill` | Route A distillation |
| `v0.4-route-c-baseline` | Route C Qwen baselines |
| `v0.5-route-c-projection-v0` | naive projected transfer |
| `v0.6-route-c-projection-v1` | A-only projection |
| `v0.7-route-c-projection-v2` | A-linear projection |
| `v0.8-exp-orchestration` | registry-based experiment runner |
| `v0.9-toolbench-subset` | ToolBench subset pipeline |
| `v1.0-route-c-projection-v3` | A-MLP projection |
| `v1.2-toolbench-fullcontext-core` | ToolBench 1024 core results |
| `v1.3-apibank-pipeline` | API-Bank data pipeline |
| `v1.4-apibank-results` | API-Bank benchmark results |
| `v1.6-mcp-style-baselines` | MCP-style baselines |
| `v1.7-result-visualization` | result figures |
| `v1.8-lora-structure-analysis` | LoRA structure diagnostics |

## Running Experiments

Experiments are registered in `experiments/registry.yaml`. Prefer the registry runner over ad hoc commands.

Dry-run first:

```powershell
uv --cache-dir .uv-cache run python src/run_experiment.py --id apibank_qwen_full --stage all --dry-run
```

Run a full experiment:

```powershell
uv --cache-dir .uv-cache run python src/run_experiment.py --id apibank_qwen_full --stage all
```

Useful stages:

```powershell
uv --cache-dir .uv-cache run python src/run_experiment.py --id <experiment_id> --stage train
uv --cache-dir .uv-cache run python src/run_experiment.py --id <experiment_id> --stage eval
uv --cache-dir .uv-cache run python src/run_experiment.py --id <experiment_id> --stage all
```

Logs are written under `runs/logs/` and should remain local.

## Aggregating Results

After adding result JSON files, regenerate the merged summary:

```powershell
uv --cache-dir .uv-cache run python src/aggregate_results.py
```

Outputs:

- `results/merged/summary.md`
- `results/merged/summary.csv`

Commit result JSON files and the merged summaries together when possible.

## Regenerating Figures

Figures are generated from `results/merged/summary.csv`; no experiment is rerun.

```powershell
uv --cache-dir .uv-cache run python src/plot_results.py
```

Outputs:

- `figures/toolbench_api_tradeoff.png`
- `figures/context_scaling.png`
- `figures/hybrid_composition.png`
- `figures/plot_data.json`

`figures/plot_data.json` records the experiment IDs and metric values used in each plot.

## LoRA Structure Diagnostics

Run adapter-level diagnostics after the relevant source and projected adapters exist locally:

```powershell
uv --cache-dir .uv-cache run python src/analyze_lora_structure.py
```

Outputs:

- `results/analysis/lora_structure.json`
- `figures/lora_ab_norms.png`
- `figures/projection_spectrum.png`

These diagnostics support the LoRA A/B transferability analysis. They inspect local PEFT adapters in `adapters/`, but the adapter weights themselves are not committed.

## Multi-Machine Workflow

Use the registry to split experiments across machines:

- RTX 3060: baseline, evaluation, non-parametric, hybrid, and analysis jobs.
- RTX 5060: long-running projection or full fine-tuning jobs.

Recommended workflow:

1. Commit code and registry changes on `main`.
2. Pull the same commit on each machine.
3. Run disjoint experiment IDs.
4. Commit only result JSON and merged summaries back to `main`.

When importing results from another branch or machine, check out only the required result files:

```powershell
git fetch origin exp-5060-sm120
git checkout origin/exp-5060-sm120 -- `
  results/apibank_qwen_full_train.json `
  results/ood_apibank_qwen_full.json
```

Then regenerate summaries:

```powershell
uv --cache-dir .uv-cache run python src/aggregate_results.py
```

## RTX 5060 `sm_120` Notes

Keep RTX 5060 or CUDA `sm_120` compatibility work isolated from core experiment logic. If a workaround branch is needed, use a dedicated branch such as `exp-5060-sm120` and merge or check out only result artifacts into `main`.

Do not commit local CUDA caches, virtual environments, model checkpoints, or machine-specific configuration.

## Ignored Files Policy

Commit:

- `src/`
- `experiments/`
- `results/*.json` for finalized metrics
- `results/merged/*`
- `results/analysis/*`
- `figures/*.png`
- `figures/plot_data.json`
- public docs such as `README.md` and `docs/EXPERIMENTS.md`

Do not commit:

- `adapters/`
- `outputs/`
- `runs/`
- `.uv-cache/`
- `.venv/`
- raw downloaded archives such as `data/toolbench/data.zip`
- raw API-Bank files under `data/apibank/`
- smoke outputs such as `results/distill_smoke_summary.json`
- private paper drafts under `paper/`

Before committing, verify staged files:

```powershell
git diff --cached --name-status
```
