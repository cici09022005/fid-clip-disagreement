#!/usr/bin/env python3
"""
Multi-seed validation experiment — lightweight validation of top-3 strategies
with multiple random seeds to quantify strategy stability vs random variation.
"""
import os
# Force offline mode for HuggingFace to prevent network timeouts
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['DIFFUSERS_OFFLINE'] = '1'

import argparse, csv, json, subprocess, sys, time
from datetime import datetime
from pathlib import Path
import numpy as np

# ──────────────────── Configuration ────────────────────

# Dataset configs (matching original experiments)
DATASETS = {
    "anime_student": {
        "data_dir": "<local_datasets>/anime_student",
        "artifacts": "kmc_lora/artifacts/anime_student",
        "instance_prompt": "a photo of student",
        "validation_prompts": ["a photo of student"],
    },
    "wikiart_mixed": {
        "data_dir": "<local_datasets>/wikiart_5k",
        "artifacts": "kmc_lora/artifacts/wikiart_mixed",
        "instance_prompt": "a painting in the style of the artist",
        "validation_prompts": ["a painting in the style of the artist"],
    },
}

# Training config (must match original for comparability)
BASE_MODEL = "models/sd15"  # Local model (not HuggingFace)
RESOLUTION = 512
BATCH_SIZE = 4
GRAD_ACCUM = 1
LORA_RANK = 16
LR = 1e-4
TOTAL_STEPS = 1200

# Base directory
BASE_DIR = Path("<project_root>")
OUTPUT_ROOT = BASE_DIR / "kmc_lora" / "output_multiseed"
RESULTS_FILE = BASE_DIR / "kmc_lora" / "results" / "multiseed_results.csv"

# Experiments to run: (dataset, relative_path, strategy_name)
# Note: For quality filter, use lists/ subdirectory
EXPERIMENTS = [
    ("anime_student", "splits/D-Sub-25.txt", "Random_D-Sub-25"),
    ("anime_student", "splits/D-High.txt", "Random_D-High"),
    ("wikiart_mixed", "lists/Quality-Top.txt", "Quality_Filter"),
]

SEEDS = [0, 1, 2, 3, 4]


# ──────────────────── Helpers ────────────────────

def run(cmd, log_file=None, dry_run=False):
    """Run subprocess with optional logging."""
    print(f"[CMD] {' '.join(cmd)}")
    if dry_run:
        return 0
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, 'w', encoding='utf-8') as f:
            result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
        return result.returncode
    else:
        return subprocess.call(cmd)


def train_single(dataset_name, image_list, output_dir, steps, seed, dry_run=False):
    """Train single LoRA with specific seed."""
    ds = DATASETS[dataset_name]

    cmd = [
        sys.executable,
        str(BASE_DIR / "kmc_lora" / "scripts" / "train_lora.py"),
        "--base-model", BASE_MODEL,
        "--image-list", str(image_list),
        "--output-dir", str(output_dir),
        "--instance-prompt", ds["instance_prompt"],
        "--resolution", str(RESOLUTION),
        "--train-batch-size", str(BATCH_SIZE),
        "--gradient-accumulation-steps", str(GRAD_ACCUM),
        "--max-train-steps", str(steps),
        "--lr", str(LR),
        "--lora-rank", str(LORA_RANK),
        "--lora-alpha", str(LORA_RANK),
        "--seed", str(seed),
        "--save-steps", "0",  # Only final
        "--mixed-precision", "fp16",
    ]

    log_file = Path(output_dir) / "train.log"

    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"Training: {dataset_name} / {output_dir.name}")
    print(f"  steps={steps}, lr={LR}, seed={seed}")
    print(f"{'='*60}")

    if not dry_run:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    rc = run(cmd, log_file=log_file, dry_run=dry_run)
    t1 = time.time()

    status = "OK" if rc == 0 else "FAIL"
    print(f"[{status}] Duration: {(t1-t0)/60:.1f} min")

    return rc == 0


def evaluate_single(dataset_name, output_dir, dry_run=False):
    """Run FID and CLIP evaluation."""
    final_adapter = Path(output_dir) / "final"
    if not final_adapter.exists():
        print(f"[SKIP] No adapter found: {final_adapter}")
        return None

    eval_file = Path(output_dir) / "eval" / "fid_clip.json"
    if eval_file.exists():
        print(f"[SKIP] Evaluation exists: {eval_file}")
        with open(eval_file) as f:
            return json.load(f)

    print(f"[EVAL] Evaluating {output_dir.name}")

    # Generate samples
    gen_dir = Path(output_dir) / "generated"
    cmd_gen = [
        sys.executable,
        str(BASE_DIR / "kmc_lora" / "scripts" / "generate_samples.py"),
        "--lora-path", str(final_adapter),
        "--output-dir", str(gen_dir),
        "--num-samples", "100",
        "--batch-size", "4",
    ]
    log_gen = Path(output_dir) / "gen.log"
    rc = run(cmd_gen, log_file=log_gen, dry_run=dry_run)
    if rc != 0 or dry_run:
        return None

    # Compute FID/CLIP
    eval_dir = Path(output_dir) / "eval"
    ds = DATASETS[dataset_name]
    cmd_eval = [
        sys.executable,
        str(BASE_DIR / "kmc_lora" / "scripts" / "run_evaluation.py"),
        "--generated-dir", str(gen_dir),
        "--reference-dir", ds["data_dir"],
        "--output-dir", str(eval_dir),
    ]
    log_eval = Path(output_dir) / "eval.log"
    rc = run(cmd_eval, log_file=log_eval, dry_run=dry_run)
    if rc != 0:
        return None

    # Read results
    if eval_file.exists():
        with open(eval_file) as f:
            return json.load(f)
    return None


def save_results(results):
    """Save results to CSV."""
    Path(RESULTS_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset', 'strategy', 'seed', 'fid', 'clip_score', 'output_dir'])
        for r in results:
            writer.writerow([
                r['dataset'], r['strategy'], r['seed'],
                r.get('fid'), r.get('clip_score'), r.get('output_dir')
            ])
    print(f"\n[OK] Results saved: {RESULTS_FILE}")


def print_summary(results):
    """Print statistical summary."""
    print("\n" + "="*60)
    print("MULTI-SEED STATISTICS SUMMARY")
    print("="*60)

    from collections import defaultdict
    by_strategy = defaultdict(list)
    for r in results:
        if r.get('fid') is not None:
            key = (r['dataset'], r['strategy'])
            by_strategy[key].append(r)

    for (dataset, strategy), runs in sorted(by_strategy.items()):
        fids = [r['fid'] for r in runs]
        clips = [r['clip_score'] for r in runs]

        fid_mean = np.mean(fids)
        fid_std = np.std(fids, ddof=1)
        fid_cv = fid_std / fid_mean * 100 if fid_mean != 0 else 0

        print(f"\n{dataset} / {strategy} (N={len(runs)}):")
        print(f"  FID:  {fid_mean:.2f} ± {fid_std:.2f} (CV={fid_cv:.1f}%)")
        print(f"  CLIP: {np.mean(clips):.4f} ± {np.std(clips):.4f}")


# ──────────────────── Main ────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true', help='Print commands only')
    parser.add_argument('--skip-eval', action='store_true', help='Skip evaluation')
    parser.add_argument('--strategy', type=int, choices=[0, 1, 2], help='Run single strategy')
    parser.add_argument('--seed', type=int, help='Run single seed')
    args = parser.parse_args()

    print("="*60)
    print("MULTI-SEED VALIDATION EXPERIMENT")
    print("="*60)
    print(f"Strategies: {len(EXPERIMENTS)}")
    print(f"Seeds: {len(SEEDS)}")
    print(f"Total: {len(EXPERIMENTS) * len(SEEDS)} experiments")
    print("="*60)

    if args.dry_run:
        print("\n[DRY RUN - commands only]\n")

    # Filter experiments if requested
    exps = EXPERIMENTS
    if args.strategy is not None:
        exps = [EXPERIMENTS[args.strategy]]

    seeds = SEEDS
    if args.seed is not None:
        seeds = [args.seed]

    all_results = []

    for dataset, split_file, strategy in exps:
        ds = DATASETS[dataset]
        image_list = BASE_DIR / ds["artifacts"] / split_file

        for seed in seeds:
            exp_name = f"{strategy}_seed{seed}"
            output_dir = OUTPUT_ROOT / dataset / exp_name

            # Training
            success = train_single(
                dataset, str(image_list), output_dir, TOTAL_STEPS, seed,
                dry_run=args.dry_run
            )

            # Evaluation
            if success and not args.dry_run and not args.skip_eval:
                results = evaluate_single(dataset, output_dir, dry_run=args.dry_run)
                if results:
                    all_results.append({
                        'dataset': dataset,
                        'strategy': strategy,
                        'seed': seed,
                        'fid': results.get('fid'),
                        'clip_score': results.get('clip_score'),
                        'output_dir': str(output_dir),
                    })
                    save_results(all_results)

    if not args.dry_run and all_results:
        print(f"\n[OK] Completed {len(all_results)}/{len(exps) * len(seeds)} experiments")
        print_summary(all_results)
    elif args.dry_run:
        print("\n[DONE] Dry run complete")


if __name__ == '__main__':
    main()
