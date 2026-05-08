# API-Bank Canonical Alignment Summary

## Summary

API-Bank selected runs were checked with saved prediction JSONL and the canonical alignment helper. The legacy scorer and canonical evaluator match exactly on saved predictions for the selected runs. The per-example prediction JSONL files remain local ignored artifacts under `outputs/canonical/`.

| Run | Samples | Parsed | Tool Acc | Arg Acc | Alignment | Metrics match |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| Qwen non-param local eval_apibank smoke | 100 | 0.98 | 0.15306122448979592 | 0.4991496598639456 | order_duplicate_example_ids | true |
| Qwen Full | 100 | 1.0 | 0.81 | 0.6916666666666665 | order_duplicate_example_ids | true |
| Qwen Projected A-linear | 100 | 0.99 | 0.7878787878787878 | 0.6658249158249159 | order_duplicate_example_ids | true |

## Interpretation

- API-Bank selected runs can be re-scored through the same canonical evaluator.
- Legacy scorer and canonical evaluator match exactly on saved predictions for the selected runs.
- API-Bank remains secondary adaptation evidence because schemas are extracted from instruction text rather than supplied through a normalized registry.

## Caveats

- The non-param local smoke used `eval_apibank.py` without adapter and is not exact registry reproduction, because registry non-param uses `eval_nonparam.py`.
- API-Bank source ids repeat; order fallback is expected for the current converted eval split.
- Per-example prediction JSONL remains ignored and is not committed.
