| Setting | Route | Model | Split Ratio | Split Layer | Parsed | Tool Acc | Arg Acc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Source | Source adapter | TinyLlama | - | - | 1.0000 | 1.0000 | 0.7075 |
| Full | Full target training | TinyLlama | - | - | 1.0000 | 1.0000 | 0.7075 |
| Distill | Route A | TinyLlama | - | - | 1.0000 | 1.0000 | 0.7075 |
| Qwen Full | Route C baseline | Qwen2.5-0.5B | - | 12 | 0.9850 | 1.0000 | 0.8173 |
| Qwen Distill | Route C baseline | Qwen2.5-0.5B | - | - | 0.9700 | 1.0000 | 0.8428 |
| Qwen Projected Full | Route C projection v0 | Qwen2.5-0.5B | - | 12 | 0.9800 | 1.0000 | 0.8138 |
| Qwen Projected Freeze | Route C projection v0 | Qwen2.5-0.5B | - | 12 | 1.0000 | 1.0000 | 0.7325 |
| Transfer | Route B | TinyLlama | 0.50 | - | 1.0000 | 1.0000 | 0.7525 |
| Split 0.25 | Route B ablation | TinyLlama | 0.25 | 5 | 1.0000 | 1.0000 | 0.7200 |
| Split 0.50 | Route B ablation | TinyLlama | 0.50 | 11 | 1.0000 | 1.0000 | 0.7525 |
| Split 0.75 | Route B ablation | TinyLlama | 0.75 | 16 | 1.0000 | 1.0000 | 0.7750 |
