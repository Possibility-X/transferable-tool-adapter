# Seed Stability Summary

Source files:

- `results/ood_toolbench_llama_projected_Alinear_1024.json`
- `results/ood_toolbench_llama_projected_Alinear_1024_seed2.json`
- `results/ood_toolbench_mistral_projected_AB_1024.json`
- `results/ood_toolbench_mistral_projected_AB_1024_seed2.json`
- `results/ood_toolbench_mistral_projected_AB_1024_seed3.json`

| Run | Seed | Parsed | Tool Acc | Arg Acc | Delta Parsed vs 42 | Delta Tool vs 42 | Delta Arg vs 42 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Llama A-linear | 42 | 0.9968 | 0.4569 | 0.6214 | 0.0000 | 0.0000 | 0.0000 |
| Llama A-linear | 2 | 0.9968 | 0.4473 | 0.6200 | 0.0000 | -0.0096 | -0.0013 |
| Mistral A+B | 42 | 0.9936 | 0.5064 | 0.6459 | 0.0000 | 0.0000 | 0.0000 |
| Mistral A+B | 2 | 0.8822 | 0.5415 | 0.6612 | -0.1115 | 0.0351 | 0.0152 |
| Mistral A+B | 3 | 0.9936 | 0.5160 | 0.6438 | 0.0000 | 0.0096 | -0.0022 |

## Interpretation

- Llama A-linear remains relatively stable across the two checked seeds.
- Mistral A+B is seed-sensitive: seed2 trades parse reliability for higher conditional Tool/Arg accuracy.
- Mistral A+B seed3 returns to seed42-level parse reliability, with modest Tool Acc improvement and nearly unchanged Arg Acc.
