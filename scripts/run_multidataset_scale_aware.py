import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path


DATASET_CONFIGS = {
    "wikiart_mixed": "kmc_lora/configs/wikiart_mixed_scale_aware_local.yaml",
    "dreambooth_mixed": "kmc_lora/configs/dreambooth_mixed_scale_aware_local.yaml",
    "dreambooth_single": "kmc_lora/configs/dreambooth_single_scale_aware_local.yaml",
}

DEFAULT_SPLITS = ["D-High", "D-Medium", "D-Low", "D-Sub-50", "D-Sub-25"]
DEFAULT_BASELINES = ["p2-only", "p3-only-long", "p2-p3-replay"]
DEFAULT_SEEDS = [42, 52, 62, 72, 82]


def run(cmd):
    print(" ".join(cmd), flush=True)
    result = subprocess.run(cmd, text=True)
    return result.returncode


def main():
    ap = argparse.ArgumentParser(
        description="Run scale-aware refinement sequentially across multiple datasets"
    )
    ap.add_argument(
        "--datasets",
        nargs="+",
        default=["wikiart_mixed", "dreambooth_mixed", "dreambooth_single"],
        choices=sorted(DATASET_CONFIGS.keys()),
    )
    ap.add_argument("--out-root", default="kmc_lora/results/scale_aware_multidataset")
    ap.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS)
    ap.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--baselines", nargs="*", default=DEFAULT_BASELINES)
    ap.add_argument("--num-per-prompt", type=int, default=50)
    ap.add_argument("--max-real", type=int, default=500)
    ap.add_argument("--total-steps-override", type=int, default=None)
    ap.add_argument("--disable-train-validation", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-generate", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    args = ap.parse_args()

    root = Path(args.out_root)
    root.mkdir(parents=True, exist_ok=True)
    log_csv = root / "multidataset_suite_log.csv"

    log_exists = log_csv.exists()
    with open(log_csv, "a" if log_exists else "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not log_exists:
            writer.writerow(
                [
                    "dataset",
                    "config",
                    "status",
                    "exit_code",
                    "elapsed_sec",
                    "started_at_epoch",
                    "finished_at_epoch",
                    "out_root",
                ]
            )

        for dataset_name in args.datasets:
            config_path = DATASET_CONFIGS[dataset_name]
            dataset_out = root / dataset_name
            dataset_out.mkdir(parents=True, exist_ok=True)

            cmd = [
                sys.executable,
                "kmc_lora/scripts/run_scale_aware_suite.py",
                "--config",
                config_path,
                "--out-root",
                str(dataset_out),
                "--splits",
                *args.splits,
                "--seeds",
                *[str(seed) for seed in args.seeds],
                "--num-per-prompt",
                str(args.num_per_prompt),
                "--max-real",
                str(args.max_real),
                "--baselines",
                *args.baselines,
            ]
            if args.total_steps_override is not None:
                cmd += ["--total-steps-override", str(args.total_steps_override)]
            if args.disable_train_validation:
                cmd.append("--disable-train-validation")
            if args.resume:
                cmd.append("--resume")
            if args.skip_train:
                cmd.append("--skip-train")
            if args.skip_generate:
                cmd.append("--skip-generate")
            if args.skip_eval:
                cmd.append("--skip-eval")

            started = time.time()
            rc = run(cmd)
            finished = time.time()
            writer.writerow(
                [
                    dataset_name,
                    config_path,
                    "ok" if rc == 0 else "fail",
                    rc,
                    round(finished - started, 1),
                    round(started, 3),
                    round(finished, 3),
                    str(dataset_out),
                ]
            )
            f.flush()

            if rc != 0:
                raise SystemExit(rc)


if __name__ == "__main__":
    main()
