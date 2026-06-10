import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(r"e:\1 zhclian\AIGC 02")
PYTHON = Path(r"C:\Users\Administrator\AppData\Local\Programs\Python\Python311\python.exe")
ROBUSTNESS_ROOT = ROOT / "kmc_lora" / "results_structured" / "robustness"
OUT_DIR = ROOT / "kmc_lora" / "figures" / "experiment_plan_m1"
TRACKER = ROOT / "refine-logs" / "EXPERIMENT_TRACKER.md"
DATASETS = ["anime_student", "wikiart_mixed", "dreambooth_mixed", "dreambooth_single"]


def update_tracker(run_ids, status, note_suffix=""):
    if not TRACKER.exists():
        return
    text = TRACKER.read_text(encoding="utf-8")
    for run_id in run_ids:
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
    update_tracker(["R004", "R005", "R006", "R007"], "IN_PROGRESS", "auto-stage1 summarizing robustness")

    rows = []
    for dataset in DATASETS:
        run_log = ROBUSTNESS_ROOT / dataset / "run_log.csv"
        if run_log.exists():
            subprocess.run(
                [
                    str(PYTHON),
                    str(ROOT / "kmc_lora" / "scripts" / "run_robustness_suite.py"),
                    "--dataset", dataset,
                    "--mode", "analyze-only",
                    "--out-root", str(ROBUSTNESS_ROOT),
                ],
                cwd=ROOT,
                check=False,
            )
            df = pd.read_csv(run_log)
            ok = int(df["status"].isin(["ok", "cached"]).sum())
            fail = int((df["status"] == "fail").sum())
            rows.append(
                {
                    "dataset": dataset,
                    "num_rows": int(len(df)),
                    "ok_or_cached": ok,
                    "failed": fail,
                    "has_analysis": int((ROBUSTNESS_ROOT / dataset / "analysis").exists()),
                }
            )
        else:
            rows.append(
                {
                    "dataset": dataset,
                    "num_rows": 0,
                    "ok_or_cached": 0,
                    "failed": 0,
                    "has_analysis": 0,
                }
            )

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(OUT_DIR / "robustness_stage1_summary.csv", index=False)
    (OUT_DIR / "robustness_stage1_summary.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )

    update_tracker(["R004", "R005", "R006", "R007"], "DONE", "auto-stage1 summary completed")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
