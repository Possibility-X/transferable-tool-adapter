| Setting | Route | Split Ratio | Split Layer | Parsed | Tool Acc | Arg Acc |
| --- | --- | --- | --- | --- | --- | --- |
| Source | Source adapter | - | - | 1.0000 | 1.0000 | 0.7075 |
| Full | Full target training | - | - | 1.0000 | 1.0000 | 0.7075 |
| Distill | Route A | - | - | 1.0000 | 1.0000 | 0.7075 |
| Transfer | Route B | 0.50 | - | 1.0000 | 1.0000 | 0.7525 |
| Split 0.25 | Route B ablation | 0.25 | 5 | 1.0000 | 1.0000 | 0.7200 |
| Split 0.50 | Route B ablation | 0.50 | 11 | 1.0000 | 1.0000 | 0.7525 |
| Split 0.75 | Route B ablation | 0.75 | 16 | 1.0000 | 1.0000 | 0.7750 |
