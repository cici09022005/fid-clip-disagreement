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
DEFAULT_SPLIT_METHODS = {
    "D-Sub-25": "scale-aware-v1",
    "D-Sub-50": "scale-aware-v1",
    "D-Medium": "scale-aware-v1",
}


def parse_split_method_overrides(items):
    mapping = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"invalid split method override: {item}")
        split_name, method_name = item.split("=", 1)
        mapping[split_name] = method_name
    return mapping


def ordered_methods_for_split(split_name, baselines, split_methods, scale_aware_only):
    methods = []
    scale_method = split_methods.get(split_name)
    if scale_method:
        methods.append(scale_method)
    if not scale_aware_only:
        methods.extend(baselines)

    seen = set()
    ordered = []
    for method in methods:
        if method not in seen:
            ordered.append(method)
            seen.add(method)
    return ordered


def run_one(cmd):
    print(" ".join(cmd), flush=True)
    return subprocess.run(cmd, text=True).returncode


def main():
    ap = argparse.ArgumentParser(
        description="Robust multidataset scale-aware runner with per-experiment retries"
    )
    ap.add_argument(
        "--datasets",
        nargs="+",
        default=["wikiart_mixed", "dreambooth_mixed", "dreambooth_single"],
        choices=sorted(DATASET_CONFIGS.keys()),
    )
    ap.add_argument("--out-root", default="kmc_lora/results/scale_aware_multidataset_v1")
    ap.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS)
    ap.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--baselines", nargs="*", default=DEFAULT_BASELINES)
    ap.add_argument("--split-method", nargs="*", default=[])
    ap.add_argument("--num-per-prompt", type=int, default=50)
    ap.add_argument("--max-real", type=int, default=500)
    ap.add_argument("--total-steps-override", type=int, default=None)
    ap.add_argument("--disable-train-validation", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-generate", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--scale-aware-only", action="store_true")
    ap.add_argument("--retries", type=int, default=5)
    ap.add_argument("--retry-wait-sec", type=int, default=15)
    ap.add_argument("--continue-on-error", action="store_true")
    args = ap.parse_args()

    split_methods = dict(DEFAULT_SPLIT_METHODS)
    split_methods.update(parse_split_method_overrides(args.split_method))

    root = Path(args.out_root)
    root.mkdir(parents=True, exist_ok=True)
    log_csv = root / "multidataset_robust_job_log.csv"

    log_exists = log_csv.exists()
    with open(log_csv, "a" if log_exists else "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not log_exists:
            writer.writerow(
                [
                    "dataset",
                    "split",
                    "seed",
                    "method",
                    "attempt",
                    "status",
                    "exit_code",
                    "elapsed_sec",
                    "exp_dir",
                ]
            )

        for dataset_name in args.datasets:
            config_path = DATASET_CONFIGS[dataset_name]
            dataset_out = root / dataset_name
            dataset_out.mkdir(parents=True, exist_ok=True)

            for split_name in args.splits:
                for seed in args.seeds:
                    for method_name in ordered_methods_for_split(
                        split_name, args.baselines, split_methods, args.scale_aware_only
                    ):
                        exp_dir = dataset_out / f"{method_name}_{split_name}_seed{seed}"
                        completion_path = exp_dir / "completion_summary.json"
                        if completion_path.exists():
                            writer.writerow(
                                [
                                    dataset_name,
                                    split_name,
                                    seed,
                                    method_name,
                                    0,
                                    "skip_completed",
                                    0,
                                    0.0,
                                    str(exp_dir),
                                ]
                            )
                            f.flush()
                            continue

                        cmd = [
                            sys.executable,
                            "kmc_lora/scripts/run_refinement_experiment.py",
                            "--config",
                            config_path,
                            "--method",
                            method_name,
                            "--split",
                            split_name,
                            "--seed",
                            str(seed),
                            "--out-root",
                            str(dataset_out),
                            "--num-per-prompt",
                            str(args.num_per_prompt),
                            "--max-real",
                            str(args.max_real),
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

                        success = False
                        last_rc = 1
                        for attempt in range(1, args.retries + 1):
                            t0 = time.time()
                            rc = run_one(cmd)
                            elapsed = round(time.time() - t0, 1)
                            success = rc == 0 and completion_path.exists()
                            last_rc = rc
                            writer.writerow(
                                [
                                    dataset_name,
                                    split_name,
                                    seed,
                                    method_name,
                                    attempt,
                                    "ok" if success else "fail",
                                    rc,
                                    elapsed,
                                    str(exp_dir),
                                ]
                            )
                            f.flush()
                            if success:
                                break
                            if attempt < args.retries:
                                time.sleep(args.retry_wait_sec)

                        if not success and not args.continue_on_error:
                            raise SystemExit(last_rc)


if __name__ == "__main__":
    main()
