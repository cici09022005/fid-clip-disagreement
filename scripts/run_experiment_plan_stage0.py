import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(r"<project_root>")
RESULTS_CSV = ROOT / "kmc_lora" / "results" / "all_eval_results.csv"
OUT_DIR = ROOT / "kmc_lora" / "figures" / "experiment_plan_m0"
TRACKER = ROOT / "refine-logs" / "EXPERIMENT_TRACKER.md"


REGIMES = ["D-High", "D-Medium", "D-Low", "D-Sub-50", "D-Sub-25"]


def extract_regime(exp_name: str) -> str:
    for regime in REGIMES:
        if regime in exp_name:
            return regime
    return "Other"


def update_tracker(run_ids, status, note_suffix=""):
    if not TRACKER.exists():
        return
    text = TRACKER.read_text(encoding="utf-8")
    for run_id in run_ids:
        text = text.replace(f"| {run_id} |", f"| {run_id} |")
        lines = []
        for line in text.splitlines():
            if line.startswith(f"| {run_id} |"):
                parts = line.split("|")
                if len(parts) >= 10:
                    parts[8] = f" {status} "
                    if note_suffix:
                        parts[9] = f" {note_suffix} "
                    line = "|".join(parts)
            lines.append(line)
        text = "\n".join(lines)
    TRACKER.write_text(text, encoding="utf-8")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    update_tracker(["R001", "R002", "R003"], "IN_PROGRESS", "auto-started after robustness")

    df = pd.read_csv(RESULTS_CSV)
    df = df[df["fid"].notna() & df["clip_score"].notna()].copy()
    df["regime"] = df["experiment"].apply(extract_regime)

    pair_rows = []
    for (dataset, regime), group in df.groupby(["dataset", "regime"], dropna=False):
        if regime == "Other" or len(group) < 2:
            continue
        fid_best = group.loc[group["fid"].idxmin()]
        clip_best = group.loc[group["clip_score"].idxmax()]
        pair_rows.append(
            {
                "dataset": dataset,
                "regime": regime,
                "fid_winner": fid_best["experiment"],
                "clip_winner": clip_best["experiment"],
                "disagree": fid_best["experiment"] != clip_best["experiment"],
                "num_candidates": int(len(group)),
            }
        )

    pair_df = pd.DataFrame(pair_rows).sort_values(["dataset", "regime"])
    observed_disagreement = int(pair_df["disagree"].sum())
    total_pairs = int(len(pair_df))

    rng = np.random.default_rng(42)
    n_perm = 10000
    null_counts = []
    for _ in range(n_perm):
        count = 0
        for _, row in pair_df.iterrows():
            n = int(row["num_candidates"])
            fid_pick = int(rng.integers(0, n))
            clip_pick = int(rng.integers(0, n))
            if fid_pick != clip_pick:
                count += 1
        null_counts.append(count)
    null_counts = np.array(null_counts)

    p_value = float((np.sum(null_counts >= observed_disagreement) + 1) / (len(null_counts) + 1))
    expected_null = float(null_counts.mean())
    ci_low, ci_high = np.percentile(null_counts, [2.5, 97.5])

    boot_counts = []
    for _ in range(n_perm):
        sample = pair_df.sample(n=total_pairs, replace=True, random_state=int(rng.integers(0, 1_000_000)))
        boot_counts.append(int(sample["disagree"].sum()))
    boot_counts = np.array(boot_counts)
    boot_low, boot_high = np.percentile(boot_counts, [2.5, 97.5])

    pair_df.to_csv(OUT_DIR / "pairwise_winner_disagreement.csv", index=False)
    pd.DataFrame(
        {
            "null_disagreement_count": null_counts,
            "bootstrap_disagreement_count": boot_counts,
        }
    ).to_csv(OUT_DIR / "null_and_bootstrap_samples.csv", index=False)

    summary = {
        "observed_disagreement": observed_disagreement,
        "total_pairs": total_pairs,
        "observed_rate": observed_disagreement / total_pairs if total_pairs else None,
        "null_expected_count": expected_null,
        "null_ci_low": float(ci_low),
        "null_ci_high": float(ci_high),
        "empirical_p_value": p_value,
        "bootstrap_ci_low": float(boot_low),
        "bootstrap_ci_high": float(boot_high),
        "n_permutations": n_perm,
    }
    (OUT_DIR / "stage0_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plt.figure(figsize=(7, 4.5))
    plt.hist(null_counts, bins=np.arange(null_counts.min(), null_counts.max() + 2) - 0.5,
             alpha=0.8, color="#90CAF9", edgecolor="white")
    plt.axvline(observed_disagreement, color="crimson", linestyle="--", linewidth=2,
                label=f"Observed = {observed_disagreement}/{total_pairs}")
    plt.xlabel("Disagreement Count Under Null")
    plt.ylabel("Frequency")
    plt.title("Winner Disagreement: Observed vs Null")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "observed_vs_null_histogram.png", dpi=220)
    plt.close()

    update_tracker(["R001", "R002", "R003"], "DONE", "auto-finished after robustness")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
