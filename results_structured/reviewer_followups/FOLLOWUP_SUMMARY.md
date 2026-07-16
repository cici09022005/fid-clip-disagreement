# Reviewer Follow-up Summary

This file records the clean final numbers used for the revised manuscript after deduplicating resumed runs.

## High-Sample Re-Evaluation

Seed-42 checkpoints were reevaluated with 50 generated images per validation prompt.

| Dataset | Regime | Random FID | HAC FID | Random CLIP | HAC CLIP | Random per-prompt CLIP | HAC per-prompt CLIP |
|---|---:|---:|---:|---:|---:|---:|---:|
| Anime-Student | D-Medium | 222.2274 | 236.9708 | 0.2983 | 0.3054 | 0.3322 | 0.3343 |
| WikiArt-Mixed | D-High | 213.8770 | 222.4956 | 0.1744 | 0.1730 | 0.3362 | 0.3347 |
| DreamBooth-Mixed | D-High | 225.1836 | 230.7928 | 0.1687 | 0.1649 | 0.3397 | 0.3390 |
| DreamBooth-Single | D-High | 106.8510 | 180.7535 | 0.2680 | 0.2786 | 0.2878 | 0.3373 |

## Sample-Count Sensitivity

Prompt-balanced reevaluation from the smallest generated subset to the largest:

| Dataset | Regime | Random FID | HAC FID | Random per-prompt CLIP | HAC per-prompt CLIP |
|---|---:|---:|---:|---:|---:|
| Anime-Student | D-Medium | 308.3188 -> 222.2274 | 320.8058 -> 236.9708 | 0.3307 -> 0.3322 | 0.3318 -> 0.3343 |
| WikiArt-Mixed | D-High | 270.6119 -> 213.8770 | 284.8547 -> 222.4956 | 0.3326 -> 0.3362 | 0.3329 -> 0.3347 |
| DreamBooth-Mixed | D-High | 234.6814 -> 225.1836 | 243.7926 -> 230.7928 | 0.3367 -> 0.3397 | 0.3317 -> 0.3390 |
| DreamBooth-Single | D-High | 110.1772 -> 106.8510 | 205.0347 -> 180.7535 | 0.2887 -> 0.2878 | 0.3366 -> 0.3373 |

## Targeted Three-Seed Random-vs-HAC Follow-up

Mean +- standard deviation after deduplicating resumed runs by strategy and seed:

| Dataset | Regime | Random FID | HAC FID | Delta FID (HAC - Random) | Random CLIP | HAC CLIP | Delta CLIP (HAC - Random) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Anime-Student | D-Medium | 223.5718 +- 14.7578 | 239.1319 +- 16.4870 | +15.5600 | 0.2999 +- 0.0095 | 0.3033 +- 0.0036 | +0.0034 |
| WikiArt-Mixed | D-High | 218.6050 +- 7.5782 | 217.9751 +- 1.8592 | -0.6299 | 0.1744 +- 0.0030 | 0.1737 +- 0.0012 | -0.0007 |
| DreamBooth-Mixed | D-High | 222.1049 +- 1.8660 | 229.5031 +- 2.7809 | +7.3982 | 0.1723 +- 0.0057 | 0.1688 +- 0.0028 | -0.0035 |
| DreamBooth-Single | D-High | 108.8960 +- 13.0147 | 125.8541 +- 4.3392 | +16.9581 | 0.2745 +- 0.0026 | 0.2770 +- 0.0045 | +0.0025 |

## Interpretation Notes

- Anime-Student is the clearest example of close-call instability: the original single-run HAC FID win at D-Medium does not persist under higher-sample reevaluation or the targeted three-seed rerun.
- WikiArt-Mixed remains effectively tied under repeated evaluation.
- DreamBooth-Mixed and DreamBooth-Single preserve the qualitative conclusion that Random attains lower FID, with DreamBooth-Single showing the largest practical gap.
