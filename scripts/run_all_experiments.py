"""
Master experiment runner for KMC-LoRA across all datasets.

Runs the full pipeline (scan → features → cluster → curriculum → train) for
each dataset config, then generates samples, evaluates, and collects results.

Usage:
    python kmc_lora/scripts/run_all_experiments.py [--dry-run]
    python kmc_lora/scripts/run_all_experiments.py --datasets anime wikiart
    python kmc_lora/scripts/run_all_experiments.py --eval-only
"""
import argparse, json, os, subprocess, sys, time
from datetime import datetime
from pathlib import Path
import yaml

CONFIGS = {
    "anime":             "kmc_lora/configs/base.yaml",
    "wikiart":           "kmc_lora/configs/wikiart_mixed.yaml",
    "dreambooth_mixed":  "kmc_lora/configs/dreambooth_mixed.yaml",
    "dreambooth_single": "kmc_lora/configs/dreambooth_single.yaml",
}


def run_cmd(cmd, dry_run=False, log_file=None):
    cmd_str = " ".join(cmd)
    print(f"\n{'[DRY-RUN] ' if dry_run else '>>> '}{cmd_str}")
    if dry_run:
        return 0
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, 'w', encoding='utf-8') as f:
            r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
        return r.returncode
    return subprocess.call(cmd)


def find_final_loras(results_dir):
    """Find all experiments that have a final/ adapter."""
    loras = []
    for p in sorted(Path(results_dir).rglob('final/adapter_model.safetensors')):
        exp_dir = p.parent.parent
        rel = exp_dir.relative_to(results_dir)
        loras.append((str(rel), str(p.parent)))
    return loras


def generate_and_evaluate(cfg, results_dir, dry_run=False):
    """Generate samples from all final LoRAs and compute metrics."""
    base_model = cfg['model']['base_model']
    prompts = cfg['prompts']['validation_prompts']
    instance_prompt = cfg['prompts']['instance_prompt']
    all_prompts = [instance_prompt] + prompts

    # Real image list for FID
    artifacts_dir = Path(cfg['paths']['artifacts_dir'])
    splits_dir = artifacts_dir / 'splits'
    real_list = str(splits_dir / 'D-High.txt')
    if not Path(real_list).exists():
        # fallback to curriculum.csv
        cur = artifacts_dir / 'curriculum.csv'
        if cur.exists():
            import pandas as pd
            df = pd.read_csv(cur)
            fallback = artifacts_dir / 'all_images.txt'
            df['path'].to_csv(fallback, index=False, header=False)
            real_list = str(fallback)

    eval_csv = str(Path(results_dir) / 'eval_summary.csv')

    loras = find_final_loras(results_dir)
    print(f"\nFound {len(loras)} trained models to evaluate")

    for name, lora_path in loras:
        gen_dir = str(Path(results_dir) / name / 'generated')

        # Skip if already evaluated
        if Path(gen_dir).exists() and \
           (Path(gen_dir) / 'eval_metrics.json').exists():
            print(f"[SKIP-EVAL] {name} already evaluated")
            continue

        # Generate
        rc = run_cmd([
            sys.executable, 'kmc_lora/scripts/generate_samples.py',
            '--base-model', base_model,
            '--lora-path', lora_path,
            '--prompts'] + all_prompts + [
            '--out-dir', gen_dir,
            '--num-per-prompt', '10',
        ], dry_run=dry_run)

        if rc != 0 and not dry_run:
            print(f"[FAIL] Generation failed for {name}")
            continue

        # Evaluate
        run_cmd([
            sys.executable, 'kmc_lora/scripts/evaluate_fid.py',
            '--real-list', real_list,
            '--gen-dir', gen_dir,
            '--prompts'] + all_prompts + [
            '--out-csv', eval_csv,
            '--experiment-name', name,
        ], dry_run=dry_run)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=list(CONFIGS.keys()),
                    choices=list(CONFIGS.keys()))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--quality-ratio", type=float, default=0.3)
    ap.add_argument("--skip-training", action="store_true",
                    help="Skip training, only generate/evaluate/collect")
    ap.add_argument("--eval-only", action="store_true",
                    help="Only evaluate existing models and collect results")
    args = ap.parse_args()

    master_log = {
        'start': datetime.now().isoformat(),
        'datasets': args.datasets,
        'results': {},
    }
    overall_t0 = time.time()

    for ds_name in args.datasets:
        config_path = CONFIGS[ds_name]
        cfg = yaml.safe_load(open(config_path, 'r', encoding='utf-8'))
        results_dir = cfg['paths']['results_dir']

        print(f"\n{'='*60}")
        print(f"  DATASET: {ds_name}")
        print(f"  CONFIG:  {config_path}")
        print(f"  RESULTS: {results_dir}")
        print(f"{'='*60}")

        ds_t0 = time.time()

        # ── Training ──
        if not args.skip_training and not args.eval_only:
            cmd = [
                sys.executable, "kmc_lora/scripts/run_experiments.py",
                "--config", config_path,
                "--quality-ratio", str(args.quality_ratio),
                "--skip-preprocessing",  # already done
            ]
            if args.dry_run:
                cmd.append("--dry-run")
            rc = run_cmd(cmd, dry_run=False)
        else:
            rc = 0

        # ── Generate + Evaluate ──
        if not args.dry_run:
            generate_and_evaluate(cfg, results_dir, dry_run=args.dry_run)

        # ── Collect results ──
        run_cmd([
            sys.executable, 'kmc_lora/scripts/collect_results.py',
            '--results-dir', results_dir,
        ], dry_run=args.dry_run)

        ds_time = (time.time() - ds_t0) / 60
        master_log['results'][ds_name] = {
            'config': config_path,
            'results_dir': results_dir,
            'duration_min': round(ds_time, 2),
            'exit_code': rc,
        }
        print(f"\n[{ds_name}] Completed in {ds_time:.1f} min")

    # ── Save master log ──
    total_time = (time.time() - overall_t0) / 60
    master_log['end'] = datetime.now().isoformat()
    master_log['total_time_min'] = round(total_time, 2)

    log_path = Path('kmc_lora/results/master_log.json')
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, 'w') as f:
        json.dump(master_log, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  ALL EXPERIMENTS COMPLETE  ({total_time:.1f} min total)")
    print(f"  Master log: {log_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
