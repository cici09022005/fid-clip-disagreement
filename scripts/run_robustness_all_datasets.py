import argparse
import subprocess
import sys
import time
from pathlib import Path


DATASETS = ["anime_student", "wikiart_mixed", "dreambooth_mixed"]


def run(cmd, log_file=None):
    print(" ".join(cmd), flush=True)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "w", encoding="utf-8") as f:
            result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
        return result.returncode
    result = subprocess.run(cmd, text=True)
    return result.returncode


def main():
    ap = argparse.ArgumentParser(description="Run robustness suite sequentially across datasets")
    ap.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    ap.add_argument("--out-root", default="kmc_lora/results/robustness")
    ap.add_argument("--mode", choices=["all", "multi-seed", "hyperparam"], default="all")
    ap.add_argument("--num-per-prompt", type=int, default=50)
    ap.add_argument("--max-real", type=int, default=500)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = Path(args.out_root)
    root.mkdir(parents=True, exist_ok=True)
    suite_log = root / "all_datasets_runner.log"

    with open(suite_log, "a", encoding="utf-8") as log:
        for dataset in args.datasets:
            started = time.time()
            cmd = [
                sys.executable,
                "kmc_lora/scripts/run_robustness_suite.py",
                "--dataset",
                dataset,
                "--mode",
                args.mode,
                "--out-root",
                args.out_root,
                "--num-per-prompt",
                str(args.num_per_prompt),
                "--max-real",
                str(args.max_real),
            ]
            if args.resume:
                cmd.append("--resume")
            if args.dry_run:
                cmd.append("--dry-run")

            log.write(f"\n[START] dataset={dataset} epoch={started:.3f}\n")
            log.flush()
            rc = run(cmd)
            ended = time.time()
            log.write(f"[END] dataset={dataset} rc={rc} elapsed_sec={ended-started:.1f}\n")
            log.flush()
            if rc != 0:
                raise SystemExit(rc)


if __name__ == "__main__":
    main()
