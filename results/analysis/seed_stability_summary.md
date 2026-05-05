# Seed Stability Summary

Source files:

- `results/ood_toolbench_llama_projected_Alinear_1024.json`
- `results/ood_toolbench_llama_projected_Alinear_1024_seed2.json`
- `results/ood_toolbench_mistral_projected_AB_1024.json`
- `results/ood_toolbench_mistral_projected_AB_1024_seed2.json`

| run | seed42 parsed | seed2 parsed | delta parsed | seed42 tool_acc | seed2 tool_acc | delta tool_acc | seed42 arg_acc | seed2 arg_acc | delta arg_acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llama_Alinear | 0.9968 | 0.9968 | 0.0000 | 0.4569 | 0.4473 | -0.0096 | 0.6214 | 0.6200 | -0.0013 |
| mistral_AB | 0.9936 | 0.8822 | -0.1115 | 0.5064 | 0.5415 | 0.0351 | 0.6459 | 0.6612 | 0.0152 |

## Interpretation

Llama A-linear appears relatively stable across the two seeds, while Mistral A+B is more seed-sensitive: seed2 improves conditional semantic accuracy but lowers parse reliability.

- Llama A-linear: Seed2 closely matches seed42, with a small Tool Acc decrease and nearly unchanged parse/argument accuracy.
- Mistral A+B: Seed2 improves conditional Tool Acc and Arg Acc but substantially lowers Parsed, indicating seed-sensitive structure/semantics tradeoff.
