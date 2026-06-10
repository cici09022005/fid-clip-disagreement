from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
ROBUSTNESS_DIR = ROOT / "kmc_lora" / "results_structured" / "robustness"
OUT_DIR = ROOT / "kmc_lora" / "figures" / "robustness"

DATASETS = [
    ("anime_student", "Anime\nStudent"),
    ("wikiart_mixed", "WikiArt\nMixed"),
    ("dreambooth_mixed", "DB\nMixed"),
    ("dreambooth_single", "DB\nSingle"),
]

STRATEGIES = [
    ("Random_D-High", "Random_D-High", "#1f77b4"),
    ("KMC_D-High", "KMC_D-High", "#d62728"),
    ("Anti_Curriculum", "Anti_Curriculum", "#2ca02c"),
]

STATUS_PRIORITY = {"ok": 2, "cached": 1}


def read_final_rows(dataset: str) -> dict[tuple[str, str], dict[str, str]]:
    path = ROBUSTNESS_DIR / dataset / "run_log.csv"
    final_rows: dict[tuple[str, str], dict[str, str]] = {}
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            status = row["status"]
            fid = row["fid"]
            if status not in STATUS_PRIORITY or fid in {"", "nan", "NaN"}:
                continue
            key = (row["strategy"], row["seed"])
            prev = final_rows.get(key)
            # Keep the most recent valid record for each strategy/seed pair.
            if prev is None or STATUS_PRIORITY[status] >= STATUS_PRIORITY[prev["status"]]:
                final_rows[key] = row
    return final_rows


def summarize() -> dict[str, dict[str, tuple[float, float]]]:
    summary: dict[str, dict[str, tuple[float, float]]] = {}
    for dataset_key, dataset_label in DATASETS:
        final_rows = read_final_rows(dataset_key)
        by_strategy: dict[str, list[float]] = defaultdict(list)
        for row in final_rows.values():
            by_strategy[row["strategy"]].append(float(row["fid"]))
        dataset_summary: dict[str, tuple[float, float]] = {}
        for strategy_key, _, _ in STRATEGIES:
            vals = sorted(by_strategy[strategy_key])
            dataset_summary[strategy_key] = (statistics.mean(vals), statistics.stdev(vals))
        summary[dataset_label] = dataset_summary
    return summary


def write_summary_csv(summary: dict[str, dict[str, tuple[float, float]]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / "fig1_multiseed_dhigh_followup_summary.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "strategy", "fid_mean", "fid_std"])
        for _, dataset_label in DATASETS:
            for strategy_key, _, _ in STRATEGIES:
                fid_mean, fid_std = summary[dataset_label][strategy_key]
                writer.writerow([dataset_label.replace("\n", " "), strategy_key, f"{fid_mean:.4f}", f"{fid_std:.4f}"])


def plot(summary: dict[str, dict[str, tuple[float, float]]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8.6, 4.8))

    x_positions = list(range(len(DATASETS)))
    offsets = [-0.24, 0.0, 0.24]

    for offset, (strategy_key, strategy_label, color) in zip(offsets, STRATEGIES):
        xs = [x + offset for x in x_positions]
        means = [summary[label][strategy_key][0] for _, label in DATASETS]
        stds = [summary[label][strategy_key][1] for _, label in DATASETS]
        ax.errorbar(
            xs,
            means,
            yerr=stds,
            fmt="o-",
            color=color,
            ecolor=color,
            elinewidth=1.5,
            capsize=4,
            capthick=1.5,
            markersize=6,
            linewidth=2.0,
            label=strategy_label,
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels([label for _, label in DATASETS], fontsize=10)
    ax.set_ylabel("FID (mean +/- std over 5 seeds)")
    ax.set_title("Multi-seed D-High follow-up across four datasets")
    ax.legend(frameon=True, fontsize=9, loc="upper center", ncol=3)
    ax.set_axisbelow(True)
    ax.set_ylim(bottom=85)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig1_multiseed_dhigh_followup.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig1_multiseed_dhigh_followup.pdf", bbox_inches="tight")


if __name__ == "__main__":
    summary = summarize()
    write_summary_csv(summary)
    plot(summary)
