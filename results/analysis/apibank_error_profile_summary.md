# API-Bank Small Error Profile Summary

## Summary

This is a small diagnostic API-Bank error profile. It helps interpret selected saved predictions under canonical re-scoring and remains secondary evidence. ToolBench error profiling is deferred until per-example ToolBench prediction logging exists. Per-example prediction JSONL remains ignored and is not committed.

| Run | Samples | Parse failures | Wrong tool | Missing arg | Wrong arg value | Extra arg | Over-generation | Exact calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen non-param local eval_apibank smoke | 100 | 2 | 83 | 2 | 4 | 1 | 3 | 5 |
| Qwen Full | 100 | 0 | 19 | 3 | 16 | 2 | 0 | 60 |
| Qwen Projected A-linear | 100 | 1 | 21 | 5 | 17 | 2 | 0 | 54 |

## Interpretation

- This profile provides a compact view of error types in selected API-Bank saved predictions.
- Qwen Full and Qwen Projected A-linear both reduce wrong-tool errors relative to the local non-param smoke, while wrong argument values remain visible in the selected samples.
- The profile should be read with the API-Bank caveats: schemas are best-effort extracted from instruction text, and the non-param row is not exact registry reproduction.

## Source Artifacts

- `outputs/canonical/apibank_error_profile_100_summary.json`
- Per-example prediction JSONL remains local under `outputs/canonical/` and is not committed.
