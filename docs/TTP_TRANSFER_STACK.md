# TTP Transfer Stack

## Purpose

The TTP Transfer Stack is a thin interface layer around the existing data,
training, projection, evaluation, and diagnostics scripts. It makes the
schema-call data contract explicit without replacing the current experiment
runner.

TTP means Tool-Transfer Priors: adapter priors learned for strict tool calls
and reused through LoRA factor projection.

## Canonical Record Format

Canonical examples use this minimum shape:

```json
{
  "id": "...",
  "dataset": "...",
  "split": "...",
  "instruction": "...",
  "tools": [
    {
      "name": "...",
      "description": "...",
      "arguments_schema": {}
    }
  ],
  "gold_call": {
    "name": "...",
    "arguments": {}
  },
  "metadata": {}
}
```

Model predictions keep the existing strict JSON contract:

```json
{"tool": "...", "arguments": {}}
```

## Current Modules

- `src/tool_prior_workflow/canonical_schema.py`: validation, JSONL helpers, and
  conversion back to the legacy `{instruction, gt}` record shape.
- `src/tool_prior_workflow/adapters/toolbench.py`: ToolBench-to-canonical
  adapter.
- `src/tool_prior_workflow/adapters/apibank.py`: API-Bank-to-canonical adapter.
- `src/tool_prior_workflow/evaluate_canonical.py`: prediction-only Parsed,
  Tool Acc, and Arg Acc scoring.
- `src/tool_prior_workflow/diagnostics.py`: small error taxonomy for saved
  predictions.
- `src/tool_prior_workflow/apibank_canonical_alignment.py`: API-Bank legacy
  scorer versus canonical scorer alignment on the same predictions.

## Dataset Adapters

ToolBench is the primary evidence path. Its adapter extracts available tool
schemas from the tool list embedded in the instruction and preserves provenance
in `metadata`.

API-Bank canonicalization is best-effort. The adapter extracts API description
JSON blocks and ToolSearcher text when present, but API-Bank remains secondary
adaptation evidence until normalized schema coverage is reviewed.

No training is launched by these adapters.

## Evaluation Metrics

The canonical evaluator keeps the same paper-facing metric semantics as the
existing ToolBench and API-Bank evaluators:

- `Parsed`: fraction of examples with a valid strict JSON call.
- `Tool Acc`: correct tool name among parsed examples.
- `Arg Acc`: average gold-argument exact match among parsed examples.

Tool Acc and Arg Acc are conditional on parsed outputs, so Parsed should be
read with the semantic metrics.

## Current Status / Caveats

- This is not full workflow validation yet.
- Existing training, projection, and registry scripts remain the source of
  record for experiments.
- API-Bank schema extraction is best-effort and should be reviewed before
  stronger claims.
- The current evaluator scores saved predictions; it does not generate model
  outputs.

## API-Bank Alignment Check

The alignment helper verifies that API-Bank records converted through the
canonical adapter receive the same Parsed, Tool Acc, and Arg Acc values as the
legacy API-Bank metric code when both paths score the same predictions. It does
not load a model or launch training.

By default it uses 20 examples and gold calls as synthetic perfect predictions:

```powershell
python src/tool_prior_workflow/apibank_canonical_alignment.py `
  --dataset-path data/apibank_eval.jsonl `
  --prediction-mode gold `
  --limit 20
```

Saved prediction JSONL can be checked by order or by `id` when every row has an
`id` field:

```powershell
python src/tool_prior_workflow/apibank_canonical_alignment.py `
  --dataset-path data/apibank_eval.jsonl `
  --prediction-mode file `
  --predictions-path outputs/canonical/apibank_predictions.jsonl `
  --limit 100
```

Use explicit save paths only for local analysis artifacts:

```powershell
python src/tool_prior_workflow/apibank_canonical_alignment.py `
  --dataset-path data/apibank_eval.jsonl `
  --prediction-mode gold `
  --limit 20 `
  --save-summary outputs/canonical/apibank_alignment_summary.json `
  --save-predictions outputs/canonical/apibank_gold_predictions.jsonl
```

## Example Commands

Convert a few ToolBench records:

```powershell
python src/tool_prior_workflow/adapters/toolbench.py `
  --input data/toolbench_train.jsonl `
  --output tmp/toolbench_canonical.jsonl `
  --limit 5
```

Convert a few API-Bank records:

```powershell
python src/tool_prior_workflow/adapters/apibank.py `
  --input data/apibank_train.jsonl `
  --output tmp/apibank_canonical.jsonl `
  --limit 5
```

Score saved predictions:

```powershell
python src/tool_prior_workflow/evaluate_canonical.py `
  --examples tmp/toolbench_canonical.jsonl `
  --predictions tmp/toolbench_predictions.jsonl
```
