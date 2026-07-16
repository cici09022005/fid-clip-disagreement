"""
Collect all experiment results into paper-ready summary tables.
Scans results directories for training_summary.json, loss_log.csv,
eval_metrics.json, and produces:
  - results_summary.csv    (one row per experiment)
  - loss_curves.csv        (all loss curves merged for plotting)
  - eval_summary.csv       (FID/CLIP scores)
"""
import argparse, csv, json
from pathlib import Path
import pandas as pd


def find_training_summaries(results_dir):
    """Walk results tree and find all training_summary.json files."""
    summaries = []
    for p in sorted(Path(results_dir).rglob('training_summary.json')):
        exp_dir = p.parent
        # Determine experiment name from relative path
        rel = exp_dir.relative_to(results_dir)
        parts = list(rel.parts)
        name = '/'.join(parts)
        summaries.append((name, exp_dir, p))
    return summaries


def find_loss_logs(results_dir):
    """Find all loss_log.csv files."""
    logs = []
    for p in sorted(Path(results_dir).rglob('loss_log.csv')):
        exp_dir = p.parent
        rel = exp_dir.relative_to(results_dir)
        name = '/'.join(rel.parts)
        logs.append((name, p))
    return logs


def find_eval_metrics(results_dir):
    """Find all eval_metrics.json files."""
    metrics = []
    for p in sorted(Path(results_dir).rglob('eval_metrics.json')):
        gen_dir = p.parent
        # find parent experiment
        rel = gen_dir.relative_to(results_dir)
        name = '/'.join(rel.parts)
        metrics.append((name, p))
    return metrics


def collect_training_configs(results_dir):
    """Find all training_config.json files."""
    configs = []
    for p in sorted(Path(results_dir).rglob('training_config.json')):
        exp_dir = p.parent
        rel = exp_dir.relative_to(results_dir)
        name = '/'.join(rel.parts)
        with open(p, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
        cfg['experiment'] = name
        configs.append(cfg)
    return configs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results-dir', required=True,
                    help='Root results directory to scan')
    ap.add_argument('--out-dir', default=None,
                    help='Output directory (default: same as results-dir)')
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir) if args.out_dir else results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Training summaries → results_summary.csv ──
    summaries = find_training_summaries(results_dir)
    if summaries:
        rows = []
        for name, exp_dir, p in summaries:
            with open(p, 'r') as f:
                data = json.load(f)
            # also read config if available
            cfg_path = exp_dir / 'training_config.json'
            extra = {}
            if cfg_path.exists():
                with open(cfg_path, 'r', encoding='utf-8') as f:
                    cfg = json.load(f)
                extra = {
                    'num_images': cfg.get('num_images'),
                    'lora_rank': cfg.get('lora_rank'),
                    'learning_rate': cfg.get('learning_rate'),
                    'effective_batch_size': cfg.get('effective_batch_size'),
                    'lora_path': cfg.get('lora_path', ''),
                }
            rows.append({
                'experiment': name,
                **data,
                **extra,
            })
        df = pd.DataFrame(rows)
        out_path = out_dir / 'results_summary.csv'
        df.to_csv(out_path, index=False)
        print(f"[OK] {len(rows)} training summaries → {out_path}")
    else:
        print("[WARN] No training_summary.json found")

    # ── 2. Loss curves → loss_curves.csv ──
    logs = find_loss_logs(results_dir)
    if logs:
        frames = []
        for name, p in logs:
            df_log = pd.read_csv(p)
            df_log.insert(0, 'experiment', name)
            frames.append(df_log)
        df_all = pd.concat(frames, ignore_index=True)
        out_path = out_dir / 'loss_curves.csv'
        df_all.to_csv(out_path, index=False)
        print(f"[OK] {len(frames)} loss logs → {out_path}  "
              f"({len(df_all)} rows)")
    else:
        print("[WARN] No loss_log.csv found")

    # ── 3. Eval metrics → eval_summary.csv ──
    evals = find_eval_metrics(results_dir)
    if evals:
        rows = []
        for name, p in evals:
            with open(p, 'r') as f:
                data = json.load(f)
            data['experiment'] = name
            rows.append(data)
        df = pd.DataFrame(rows)
        out_path = out_dir / 'eval_summary.csv'
        df.to_csv(out_path, index=False)
        print(f"[OK] {len(rows)} eval results → {out_path}")
    else:
        print("[WARN] No eval_metrics.json found (run evaluation first)")

    # ── 4. All training configs → configs_summary.csv ──
    configs = collect_training_configs(results_dir)
    if configs:
        df = pd.DataFrame(configs)
        out_path = out_dir / 'configs_summary.csv'
        df.to_csv(out_path, index=False)
        print(f"[OK] {len(configs)} configs → {out_path}")

    # ── 5. Quick console summary ──
    print(f"\n{'='*60}")
    print(f"RESULTS COLLECTION COMPLETE")
    print(f"  Directory: {results_dir}")
    print(f"  Training runs found: {len(summaries)}")
    print(f"  Loss logs found:     {len(logs)}")
    print(f"  Eval metrics found:  {len(evals)}")
    print(f"  Configs found:       {len(configs)}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
