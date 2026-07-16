# FID–CLIP Disagreement in Low-Data Diffusion Model Fine-Tuning

Code and result data for the paper:

> **Practical Consequences of FID–CLIP Disagreement in Low-Data Diffusion Model Fine-Tuning**
> Weijun Zhou, Cuilian Zhang, Lixiang Song.

This repository contains the **evaluation pipeline** and the **result data** needed to reproduce all tables and figures in the paper. It does **not** redistribute the third-party training images or the trained model weights (see *Data and weights* below).

The structured strategy studied in the paper is the **KMC (KMeans-Curriculum)** ordering. In the code, configuration labels use the prefix `KMC_` for this strategy and `Random_` / `Anti_Curriculum` for the baselines.

---

## Repository structure

```
scripts/                     Python pipeline (feature extraction, clustering,
                             curriculum, training, evaluation, figure generation)
configs/                     YAML configuration files (per dataset / regime)
results_structured/          Per-configuration evaluation results (CSV) for SD 1.5
results_structured/robustness/   Multi-seed run logs
results_structured/reviewer_followups/   Targeted three-seed follow-up and
                             high-sample reevaluation (run logs, eval JSONs, summary)
results_sdxl/                SDXL cross-architecture results (CSV)
figures/advanced_stats/      Correlation and bootstrap CSVs
figures/robustness/          Multi-seed summary CSVs
artifacts/                   Cluster assignments, quality scores, curriculum orderings
```

Only text files (`.py`, `.yaml`, `.csv`, `.txt`) are included. Generated images,
model checkpoints (`.safetensors`), and feature caches (`.npy`) are excluded for
size and licensing reasons.

## Key result files (mapping to the paper)

| Paper item | File |
|---|---|
| FID/CLIP for all 68 configurations (Tables III–V) | `results_structured/all_eval_results.csv` |
| Spearman/Pearson correlation + CI (Table III) | `figures/advanced_stats/spearman_correlation_ci.csv`, `pearson_correlation_ci.csv` |
| Five-seed D-High robustness (Tables V–VI) | `figures/robustness/fig1_multiseed_dhigh_followup_summary.csv` |
| Per-seed robustness run logs | `results_structured/robustness/<dataset>/run_log.csv` |
| Targeted three-seed follow-up (Table VII) | `results_structured/reviewer_followups/multiseed_targeted_followup.csv` (per-seed) and `FOLLOWUP_SUMMARY.md` (aggregates) |
| High-sample reevaluation and sample-count sensitivity | `results_structured/reviewer_followups/highsample/<dataset>/<strategy>/generated_npp50_seed42/` |
| SDXL cross-architecture validation (Table IX) | `results_sdxl/sdxl_summary.csv` |
| KMC curriculum (Algorithm 1) | `scripts/cluster_and_quality.py`, `scripts/build_curriculum.py` |
| Training / evaluation driver | `scripts/run_experiments.py` |
| Figure generation | `scripts/generate_figures.py` and other `scripts/generate_*.py` |

## Method pipeline (Algorithm 1, KMC)

1. Extract CLIP ViT-L/14 features (`scripts/compute_features.py`).
2. KMeans clustering with silhouette-based `k` selection, and a composite
   difficulty score from quality, typicality, and heterogeneity
   (`scripts/cluster_and_quality.py`, `scripts/build_curriculum.py`).
3. Fine-tune Stable Diffusion 1.5 with LoRA under the phased schedule
   (`scripts/run_experiments.py`).
4. Compute FID (torchmetrics, 2048-dim Inception-v3 features) and a CLIP-based
   text–image similarity score; aggregate into the result CSVs.
5. Reproduce tables and figures from the CSVs (`scripts/generate_*.py`).

The provided CSVs already contain the aggregated metrics, so the tables and
figures can be regenerated without re-running training or image generation.

## Data and weights (not included)

The **Anime-Student** dataset was collected by the authors and is **not
redistributed here**; it is available from the corresponding author upon
reasonable request. The **WikiArt-Mixed** dataset is derived from the WikiArt collection
(https://www.wikiart.org), and the **DreamBooth-Mixed** and
**DreamBooth-Single** datasets are derived from the official DreamBooth
dataset (https://github.com/google/dreambooth); all three should be obtained
under their original licenses. Trained LoRA adapters and Stable
Diffusion / SDXL weights are not included; the base models are available from
their official releases.

## Requirements

Python 3.9+, with `torch`, `diffusers`, `peft`, `open_clip_torch`,
`torchmetrics`, `scikit-learn`, `pandas`, `numpy`, and `matplotlib`. A CUDA GPU
is required for training and image generation; reproducing the tables and
figures from the provided CSVs requires only `pandas`/`numpy`/`matplotlib`.

## License

Code is released under the MIT License (see `LICENSE`). The result data files
are provided for academic reproducibility.
