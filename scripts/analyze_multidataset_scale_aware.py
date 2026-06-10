"""
Analyze completed multidataset scale-aware experiments.

Outputs are written under:
  <out_root>/analysis/
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

MPL_CACHE_DIR = Path("kmc_lora") / ".mpl-cache"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATASET_ORDER = ["wikiart_mixed", "dreambooth_mixed"]
SPLIT_ORDER = ["D-High", "D-Medium", "D-Low", "D-Sub-50", "D-Sub-25"]
METHOD_ORDER = ["p2-only", "p3-only-long", "p2-p3-replay", "scale-aware-v1"]
METHOD_LABELS = {
    "p2-only": "P2 Only",
    "p3-only-long": "P3 Only Long",
    "p2-p3-replay": "P2+P3 Replay",
    "scale-aware-v1": "Scale-Aware v1",
}
DATASET_LABELS = {
    "wikiart_mixed": "WikiArt Mixed",
    "dreambooth_mixed": "DreamBooth Mixed",
}
SPLIT_LABELS = {
    "D-High": "High",
    "D-Medium": "Medium",
    "D-Low": "Low",
    "D-Sub-50": "Sub-50",
    "D-Sub-25": "Sub-25",
}
COLORS = {
    "p2-only": "#1f77b4",
    "p3-only-long": "#ff7f0e",
    "p2-p3-replay": "#2ca02c",
    "scale-aware-v1": "#d62728",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default="kmc_lora/results/scale_aware_multidataset_v1")
    return ap.parse_args()


def load_records(out_root: Path) -> list[dict]:
    records: list[dict] = []
    for dataset_dir in out_root.iterdir():
        if not dataset_dir.is_dir():
            continue
        dataset_name = dataset_dir.name
        for eval_path in dataset_dir.glob("*_seed*/eval.csv"):
            exp_name = eval_path.parent.name
            if not eval_path.exists():
                continue
            parts = exp_name.rsplit("_", 2)
            if len(parts) != 3:
                continue
            method, split_name, seed_text = parts
            df = pd.read_csv(eval_path)
            if df.empty or "fid" not in df.columns:
                continue
            records.append(
                {
                    "dataset": dataset_name,
                    "experiment": exp_name,
                    "method": method,
                    "split": split_name,
                    "seed": int(seed_text.replace("seed", "")),
                    "fid": float(df.iloc[0]["fid"]),
                }
            )
    if not records:
        raise FileNotFoundError(f"No eval.csv files found under {out_root}")
    return records


def plot_dataset_panels(summary_df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, len(DATASET_ORDER), figsize=(12, 4.4), sharey=True)
    if len(DATASET_ORDER) == 1:
        axes = [axes]

    for ax, dataset in zip(axes, DATASET_ORDER):
        ds_df = summary_df[summary_df["dataset"] == dataset].copy()
        for method in METHOD_ORDER:
            sub = ds_df[ds_df["method"] == method].copy()
            if sub.empty:
                continue
            sub["split"] = pd.Categorical(sub["split"], categories=SPLIT_ORDER, ordered=True)
            sub = sub.sort_values("split")
            x = np.arange(len(sub))
            y = sub["mean_fid"].to_numpy()
            err = sub["std_fid"].fillna(0.0).to_numpy()
            ax.plot(x, y, marker="o", linewidth=2, color=COLORS[method], label=METHOD_LABELS[method])
            ax.fill_between(x, y - err, y + err, color=COLORS[method], alpha=0.12)
        ax.set_title(DATASET_LABELS.get(dataset, dataset))
        ax.set_xticks(np.arange(len(SPLIT_ORDER)))
        ax.set_xticklabels([SPLIT_LABELS[s] for s in SPLIT_ORDER], rotation=20)
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("Split")
    axes[0].set_ylabel("FID")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout()
    fig.savefig(out_dir / "fid_by_split_multidataset.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "fid_by_split_multidataset.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_best_method_heatmap(best_df: pd.DataFrame, out_dir: Path) -> None:
    method_to_idx = {m: i for i, m in enumerate(METHOD_ORDER)}
    matrix = np.full((len(DATASET_ORDER), len(SPLIT_ORDER)), np.nan)
    for i, dataset in enumerate(DATASET_ORDER):
        for j, split_name in enumerate(SPLIT_ORDER):
            sub = best_df[(best_df["dataset"] == dataset) & (best_df["split"] == split_name)]
            if not sub.empty:
                matrix[i, j] = method_to_idx[sub.iloc[0]["best_method"]]

    fig, ax = plt.subplots(figsize=(8.6, 2.8))
    cmap = matplotlib.colors.ListedColormap([COLORS[m] for m in METHOD_ORDER])
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=-0.5, vmax=len(METHOD_ORDER) - 0.5)
    ax.set_xticks(np.arange(len(SPLIT_ORDER)))
    ax.set_xticklabels([SPLIT_LABELS[s] for s in SPLIT_ORDER])
    ax.set_yticks(np.arange(len(DATASET_ORDER)))
    ax.set_yticklabels([DATASET_LABELS[d] for d in DATASET_ORDER])
    ax.set_title("Best method by dataset and split")
    for i, dataset in enumerate(DATASET_ORDER):
        for j, split_name in enumerate(SPLIT_ORDER):
            sub = best_df[(best_df["dataset"] == dataset) & (best_df["split"] == split_name)]
            if not sub.empty:
                label = METHOD_LABELS[sub.iloc[0]["best_method"]].replace(" ", "\n")
                ax.text(j, i, label, ha="center", va="center", fontsize=8, color="white")
    cbar = fig.colorbar(im, ax=ax, ticks=np.arange(len(METHOD_ORDER)))
    cbar.ax.set_yticklabels([METHOD_LABELS[m] for m in METHOD_ORDER])
    fig.tight_layout()
    fig.savefig(out_dir / "best_method_heatmap.png", dpi=300)
    fig.savefig(out_dir / "best_method_heatmap.pdf")
    plt.close(fig)


def plot_scaleaware_delta(summary_df: pd.DataFrame, out_dir: Path) -> None:
    rows = []
    for dataset in DATASET_ORDER:
        for split_name in SPLIT_ORDER:
            sub = summary_df[(summary_df["dataset"] == dataset) & (summary_df["split"] == split_name)]
            if sub.empty or "scale-aware-v1" not in set(sub["method"]):
                continue
            baseline_sub = sub[sub["method"].isin(["p2-only", "p3-only-long", "p2-p3-replay"])]
            if baseline_sub.empty:
                continue
            selected = float(sub[sub["method"] == "scale-aware-v1"].iloc[0]["mean_fid"])
            best_baseline = float(baseline_sub["mean_fid"].min())
            rows.append(
                {
                    "dataset": dataset,
                    "split": split_name,
                    "selected_mean_fid": selected,
                    "best_baseline_mean_fid": best_baseline,
                    "delta": selected - best_baseline,
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    labels = [f"{DATASET_LABELS[r.dataset]}\n{SPLIT_LABELS[r.split]}" for r in df.itertuples()]
    x = np.arange(len(df))
    colors = ["#2ca02c" if d <= 0 else "#d62728" for d in df["delta"]]
    ax.bar(x, df["delta"], color=colors)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Scale-aware mean FID - best explicit baseline")
    ax.set_title("Where scale-aware helps or hurts")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "scaleaware_vs_best_baseline_delta.png", dpi=300)
    fig.savefig(out_dir / "scaleaware_vs_best_baseline_delta.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root)
    analysis_dir = out_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    raw_df = pd.DataFrame(load_records(out_root))
    raw_df["dataset"] = pd.Categorical(raw_df["dataset"], categories=DATASET_ORDER, ordered=True)
    raw_df["split"] = pd.Categorical(raw_df["split"], categories=SPLIT_ORDER, ordered=True)
    raw_df["method"] = pd.Categorical(raw_df["method"], categories=METHOD_ORDER, ordered=True)
    raw_df = raw_df.sort_values(["dataset", "split", "method", "seed"]).reset_index(drop=True)
    raw_df.to_csv(analysis_dir / "multidataset_raw_seed_metrics.csv", index=False)

    summary_df = (
        raw_df.groupby(["dataset", "split", "method"], observed=True)
        .agg(mean_fid=("fid", "mean"), std_fid=("fid", "std"), num_seeds=("fid", "count"))
        .reset_index()
    )
    summary_df["std_fid"] = summary_df["std_fid"].fillna(0.0)
    summary_df.to_csv(analysis_dir / "multidataset_method_summary.csv", index=False)

    best_rows = []
    for dataset in DATASET_ORDER:
        for split_name in SPLIT_ORDER:
            sub = summary_df[
                (summary_df["dataset"] == dataset)
                & (summary_df["split"] == split_name)
                & (summary_df["method"].isin(["p2-only", "p3-only-long", "p2-p3-replay"]))
            ]
            if sub.empty:
                continue
            best = sub.sort_values("mean_fid").iloc[0]
            best_rows.append(
                {
                    "dataset": dataset,
                    "split": split_name,
                    "best_method": best["method"],
                    "best_mean_fid": round(float(best["mean_fid"]), 4),
                }
            )
    best_df = pd.DataFrame(best_rows)
    best_df.to_csv(analysis_dir / "best_method_by_dataset_split.csv", index=False)

    overall_rows = []
    for dataset in DATASET_ORDER:
        ds_baseline = summary_df[
            (summary_df["dataset"] == dataset)
            & (summary_df["method"].isin(["p2-only", "p3-only-long", "p2-p3-replay"]))
        ]
        ds_scale = summary_df[(summary_df["dataset"] == dataset) & (summary_df["method"] == "scale-aware-v1")]
        if not ds_baseline.empty:
            overall_rows.append(
                {
                    "dataset": dataset,
                    "avg_best_explicit_baseline_fid": round(
                        float(ds_baseline.groupby("split")["mean_fid"].min().mean()), 4
                    ),
                    "avg_scaleaware_fid": round(float(ds_scale["mean_fid"].mean()), 4) if not ds_scale.empty else np.nan,
                }
            )
    overall_df = pd.DataFrame(overall_rows)
    if not overall_df.empty:
        overall_df["delta_scaleaware_minus_best_explicit"] = (
            overall_df["avg_scaleaware_fid"] - overall_df["avg_best_explicit_baseline_fid"]
        ).round(4)
    overall_df.to_csv(analysis_dir / "multidataset_overall_summary.csv", index=False)

    plot_dataset_panels(summary_df, analysis_dir)
    plot_best_method_heatmap(best_df, analysis_dir)
    plot_scaleaware_delta(summary_df, analysis_dir)

    readme = analysis_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Multidataset Analysis",
                "",
                "Generated files:",
                "- `multidataset_raw_seed_metrics.csv`: one row per completed seed/run",
                "- `multidataset_method_summary.csv`: mean/std FID by dataset/split/method",
                "- `best_method_by_dataset_split.csv`: best explicit method per dataset/split",
                "- `multidataset_overall_summary.csv`: dataset-level averages",
                "- `fid_by_split_multidataset.png`: line plots per dataset",
                "- `best_method_heatmap.png`: best method heatmap",
                "- `scaleaware_vs_best_baseline_delta.png`: scale-aware vs best-explicit delta",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
