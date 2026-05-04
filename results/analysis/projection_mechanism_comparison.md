# Projection Mechanism Comparison

| name | W layers | cosine resize/project | norm ratio | rank mean | norm ratio range | rank range |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| llama_Alinear | 24 | -0.0010 | 0.5777 | 1648.6 | 0.5730-0.5839 | 1648.2-1649.3 |
| mistral_Alinear | 48 | null | 0.8167 | 1896.5 | 0.8124-0.8226 | 1896.3-1896.8 |

Projection geometry is target-dependent: Mistral preserves more of resized A norm and uses higher effective-rank W maps than Llama, while both rely on high-rank target-space alignment.
