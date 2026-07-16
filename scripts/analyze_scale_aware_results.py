"""
Analyze scale-aware publication results and generate summary tables/figures.

Outputs are written under:
  <out_root>/analysis/

Example:
  python kmc_lora/scripts/analyze_scale_aware_results.py ^
      --out-root kmc_lora/results/scale_aware_paper_v1
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
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


METHOD_ORDER = ["p2-only", "p3-only-long", "p2-p3-replay"]
SPLIT_ORDER = ["D-High", "D-Medium", "D-Low", "D-Sub-50", "D-Sub-25"]
METHOD_LABELS = {
    "p2-only": "P2 Only",
    "p3-only-long": "P3 Only Long",
    "p2-p3-replay": "P2+P3 Replay",
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
    "selected": "#d62728",
    "baseline": "#7f7f7f",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default="kmc_lora/results/scale_aware_paper_v1")
    return ap.parse_args()


def load_mapping(out_root: Path) -> dict[str, str]:
    path = out_root / "scale_aware_mapping.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_records(out_root: Path) -> list[dict]:
    records: list[dict] = []
    for eval_path in out_root.glob("*_seed*/generated/eval_metrics.json"):
        payload = json.loads(eval_path.read_text(encoding="utf-8"))
        exp_name = payload["experiment"]
        method, split_name, seed_text = exp_name.rsplit("_", 2)
        seed = int(seed_text.replace("seed", ""))
        records.append({
            "experiment": exp_name,
            "method": method,
            "split": split_name,
            "seed": seed,
            "fid": float(payload["fid"]),
            "num_real": int(payload["num_real"]),
            "num_gen": int(payload["num_gen"]),
        })
    if not records:
        raise FileNotFoundError(f"No eval_metrics.json found under {out_root}")
    return records


def sign_flip_pvalue(diffs: list[float]) -> float:
    if not diffs:
        return float("nan")
    observed = abs(sum(diffs))
    total = 0
    extreme = 0
    for signs in itertools.product([-1.0, 1.0], repeat=len(diffs)):
        total += 1
        signed_sum = abs(sum(sign * diff for sign, diff in zip(signs, diffs)))
        if signed_sum >= observed - 1e-12:
            extreme += 1
    return extreme / total


def cohens_d_paired(diffs: list[float]) -> float:
    if len(diffs) < 2:
        return float("nan")
    arr = np.asarray(diffs, dtype=float)
    std = arr.std(ddof=1)
    if std == 0:
        return 0.0
    return arr.mean() / std


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_method_curves(summary_df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for method in METHOD_ORDER:
        subset = summary_df[summary_df["method"] == method].copy()
        if subset.empty:
            continue
        subset["split"] = pd.Categorical(subset["split"], categories=SPLIT_ORDER, ordered=True)
        subset = subset.sort_values("split")
        x = np.arange(len(subset))
        y = subset["mean_fid"].to_numpy()
        err = subset["std_fid"].to_numpy()
        ax.plot(x, y, marker="o", linewidth=2, color=COLORS[method], label=METHOD_LABELS[method])
        ax.fill_between(x, y - err, y + err, color=COLORS[method], alpha=0.15)

    ax.set_xticks(np.arange(len(SPLIT_ORDER)))
    ax.set_xticklabels([SPLIT_LABELS[s] for s in SPLIT_ORDER])
    ax.set_ylabel("FID")
    ax.set_xlabel("Split")
    ax.set_title("Scale-aware methods across data diversity splits")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "fid_by_split.png", dpi=300)
    fig.savefig(out_dir / "fid_by_split.pdf")
    plt.close(fig)


def plot_selected_vs_baseline(compare_df: pd.DataFrame, out_dir: Path) -> None:
    ordered = compare_df.copy()
    ordered["split"] = pd.Categorical(ordered["split"], categories=SPLIT_ORDER, ordered=True)
    ordered = ordered.sort_values("split").reset_index(drop=True)
    x = np.arange(len(ordered))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(x - width / 2, ordered["selected_mean_fid"], width=width,
           color=COLORS["selected"], label="Selected")
    ax.bar(x + width / 2, ordered["best_single_mean_fid"], width=width,
           color=COLORS["baseline"], label="Best single-stage baseline")

    for idx, row in ordered.iterrows():
        ax.text(
            idx,
            max(row["selected_mean_fid"], row["best_single_mean_fid"]) + 4,
            f"{row['delta_vs_best_single']:+.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([SPLIT_LABELS[s] for s in ordered["split"]])
    ax.set_ylabel("Mean FID")
    ax.set_xlabel("Split")
    ax.set_title("Selected method vs best single-stage baseline")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "selected_vs_best_single.png", dpi=300)
    fig.savefig(out_dir / "selected_vs_best_single.pdf")
    plt.close(fig)


def plot_delta_heatmap(pairwise_df: pd.DataFrame, out_dir: Path) -> None:
    matrix = np.full((3, len(SPLIT_ORDER)), np.nan, dtype=float)
    pairs = [
        ("p2-only", "p3-only-long"),
        ("p2-only", "p2-p3-replay"),
        ("p3-only-long", "p2-p3-replay"),
    ]
    for row_idx, (method_a, method_b) in enumerate(pairs):
        for split_idx, split_name in enumerate(SPLIT_ORDER):
            subset = pairwise_df[
                (pairwise_df["split"] == split_name)
                & (pairwise_df["method_a"] == method_a)
                & (pairwise_df["method_b"] == method_b)
            ]
            if not subset.empty:
                matrix[row_idx, split_idx] = subset.iloc[0]["mean_delta_a_minus_b"]

    fig, ax = plt.subplots(figsize=(8.2, 3.2))
    vmax = np.nanmax(np.abs(matrix)) if not np.isnan(matrix).all() else 1.0
    im = ax.imshow(matrix, cmap="coolwarm_r", aspect="auto", vmin=-vmax, vmax=vmax)

    ax.set_xticks(np.arange(len(SPLIT_ORDER)))
    ax.set_xticklabels([SPLIT_LABELS[s] for s in SPLIT_ORDER])
    ax.set_yticks(np.arange(3))
    ax.set_yticklabels(["P2 - P3", "P2 - Replay", "P3 - Replay"])
    ax.set_title("Pairwise mean FID deltas")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if not math.isnan(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:+.1f}", ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, label="Mean FID delta")
    fig.tight_layout()
    fig.savefig(out_dir / "pairwise_delta_heatmap.png", dpi=300)
    fig.savefig(out_dir / "pairwise_delta_heatmap.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root)
    analysis_dir = out_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    mapping = load_mapping(out_root)
    raw_df = pd.DataFrame(load_records(out_root))
    raw_df["split"] = pd.Categorical(raw_df["split"], categories=SPLIT_ORDER, ordered=True)
    raw_df["method"] = pd.Categorical(raw_df["method"], categories=METHOD_ORDER, ordered=True)
    raw_df = raw_df.sort_values(["split", "method", "seed"]).reset_index(drop=True)
    raw_df.to_csv(analysis_dir / "raw_seed_metrics.csv", index=False)

    summary_df = (
        raw_df.groupby(["split", "method"], observed=True)
        .agg(
            mean_fid=("fid", "mean"),
            std_fid=("fid", "std"),
            min_fid=("fid", "min"),
            max_fid=("fid", "max"),
            num_seeds=("fid", "count"),
        )
        .reset_index()
    )
    summary_df["std_fid"] = summary_df["std_fid"].fillna(0.0)
    summary_df.to_csv(analysis_dir / "method_summary_analysis.csv", index=False)

    pairwise_rows: list[dict] = []
    for split_name in SPLIT_ORDER:
        split_df = raw_df[raw_df["split"] == split_name]
        methods = [m for m in METHOD_ORDER if m in set(split_df["method"].astype(str))]
        for method_a, method_b in itertools.combinations(methods, 2):
            pivot = (
                split_df[split_df["method"].astype(str).isin([method_a, method_b])]
                .pivot(index="seed", columns="method", values="fid")
                .dropna()
            )
            if pivot.empty:
                continue
            diffs = (pivot[method_a] - pivot[method_b]).tolist()
            pairwise_rows.append({
                "split": split_name,
                "method_a": method_a,
                "method_b": method_b,
                "mean_delta_a_minus_b": float(np.mean(diffs)),
                "median_delta_a_minus_b": float(np.median(diffs)),
                "wins_a": int(sum(diff < 0 for diff in diffs)),
                "wins_b": int(sum(diff > 0 for diff in diffs)),
                "ties": int(sum(diff == 0 for diff in diffs)),
                "pvalue_sign_flip": sign_flip_pvalue(diffs),
                "cohens_d_paired": cohens_d_paired(diffs),
            })

    pairwise_df = pd.DataFrame(pairwise_rows)
    if not pairwise_df.empty:
        pairwise_df.to_csv(analysis_dir / "pairwise_tests.csv", index=False)

    compare_rows: list[dict] = []
    for split_name in SPLIT_ORDER:
        selected_method = mapping.get(split_name)
        if not selected_method:
            continue
        split_summary = summary_df[summary_df["split"] == split_name].copy()
        selected_row = split_summary[split_summary["method"] == selected_method].iloc[0]

        single_stage = split_summary[split_summary["method"].isin(["p2-only", "p3-only-long"])].copy()
        best_single = single_stage.sort_values("mean_fid").iloc[0]
        compare_rows.append({
            "split": split_name,
            "selected_method": selected_method,
            "selected_mean_fid": float(selected_row["mean_fid"]),
            "selected_std_fid": float(selected_row["std_fid"]),
            "best_single_method": str(best_single["method"]),
            "best_single_mean_fid": float(best_single["mean_fid"]),
            "best_single_std_fid": float(best_single["std_fid"]),
            "delta_vs_best_single": float(selected_row["mean_fid"] - best_single["mean_fid"]),
        })

    compare_df = pd.DataFrame(compare_rows)
    if not compare_df.empty:
        compare_df.to_csv(analysis_dir / "selected_vs_best_single.csv", index=False)

    selected_mean = compare_df["selected_mean_fid"].mean() if not compare_df.empty else float("nan")
    best_single_mean = compare_df["best_single_mean_fid"].mean() if not compare_df.empty else float("nan")
    overall_rows = [
        {"metric": "overall_avg_selected_mean_fid", "value": selected_mean},
        {"metric": "overall_avg_best_single_mean_fid", "value": best_single_mean},
        {
            "metric": "overall_delta_selected_minus_best_single",
            "value": selected_mean - best_single_mean if not math.isnan(selected_mean) and not math.isnan(best_single_mean) else float("nan"),
        },
    ]
    write_csv(analysis_dir / "overall_summary.csv", overall_rows, ["metric", "value"])

    summary_lines = [
        "# Scale-aware analysis",
        "",
        f"- Result root: `{out_root}`",
        f"- Total evaluated runs: `{len(raw_df)}`",
        f"- Splits: `{', '.join(SPLIT_ORDER)}`",
        f"- Methods: `{', '.join(METHOD_ORDER)}`",
        "",
        "## Selected mapping",
    ]
    for split_name in SPLIT_ORDER:
        if split_name in mapping:
            summary_lines.append(f"- {split_name}: `{mapping[split_name]}`")
    summary_lines += [
        "",
        "## Overall comparison",
        f"- Selected scale-aware average mean FID: `{selected_mean:.4f}`",
        f"- Best single-stage average mean FID: `{best_single_mean:.4f}`",
        f"- Delta (selected - best single-stage): `{selected_mean - best_single_mean:+.4f}`",
        "",
        "## Notes",
        "- Pairwise tests use an exact sign-flip test on per-seed paired FID differences.",
        "- Negative delta means the first method has lower FID and is better.",
    ]
    (analysis_dir / "README.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    plot_method_curves(summary_df, analysis_dir)
    if not compare_df.empty:
        plot_selected_vs_baseline(compare_df, analysis_dir)
    if not pairwise_df.empty:
        plot_delta_heatmap(pairwise_df, analysis_dir)

    print(f"[OK] Analysis written to: {analysis_dir}")


if __name__ == "__main__":
    main()
