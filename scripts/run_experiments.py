"""
Master experiment runner — orchestrates preprocessing + training for all
baselines, ablations, and the KMC-LoRA curriculum.
Saves a master experiment log (CSV) and per-run logs for paper reproducibility.
"""
import argparse, csv, json, os, subprocess, sys, time
from datetime import datetime
from pathlib import Path
import yaml
import pandas as pd
import numpy as np


# ──────────────────── Helpers ────────────────────
def run(cmd, dry_run=False, log_file=None):
    """Run a subprocess, optionally logging its output."""
    print(' '.join(cmd))
    if dry_run:
        return 0
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, 'w', encoding='utf-8') as f:
            result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
        return result.returncode
    else:
        return subprocess.call(cmd)


def is_experiment_done(out_dir):
    """Check if experiment already completed (has final/ adapter)."""
    final = Path(out_dir) / 'final' / 'adapter_model.safetensors'
    if final.exists():
        return True
    final_bin = Path(out_dir) / 'final' / 'adapter_model.bin'
    return final_bin.exists()


def build_lists(artifacts_dir, quality_ratio=0.3, dry_run=False):
    cur_path = Path(artifacts_dir) / 'curriculum.csv'
    quality_list = Path(artifacts_dir) / 'lists' / 'Quality-Top.txt'
    anti_list = Path(artifacts_dir) / 'lists' / 'AntiCurriculum.txt'

    if dry_run or not cur_path.exists():
        return str(quality_list), str(anti_list)

    df = pd.read_csv(cur_path)
    quality_list.parent.mkdir(parents=True, exist_ok=True)

    df_sorted = df.sort_values('quality', ascending=False)
    n = max(1, int(len(df_sorted) * quality_ratio))
    df_sorted.head(n)['path'].to_csv(quality_list, index=False, header=False)

    df.sort_values('difficulty', ascending=False)['path'].to_csv(
        anti_list, index=False, header=False)

    return str(quality_list), str(anti_list)


def build_split_phase_lists(curriculum_csv, split_file, out_dir):
    """Create split-specific phase lists from curriculum annotations."""
    df = pd.read_csv(curriculum_csv)
    split_paths = {
        p.strip()
        for p in Path(split_file).read_text(encoding='utf-8').splitlines()
        if p.strip()
    }
    split_df = df[df['path'].isin(split_paths)].copy()
    if split_df.empty:
        raise ValueError(f"No overlap between curriculum and split: {split_file}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    phase_files = {}
    for phase_name in ['phase1', 'phase2', 'phase3']:
        phase_path = out_dir / f'{phase_name}.txt'
        split_df.loc[split_df[phase_name].astype(bool), 'path'].to_csv(
            phase_path, index=False, header=False
        )
        phase_files[phase_name] = str(phase_path)
    return phase_files


# ──────────────────── Experiment logger ────────────────────
class ExperimentLog:
    """Master CSV tracking every experiment run."""
    def __init__(self, results_dir):
        self.path = Path(results_dir) / 'experiment_log.csv'
        self.path.parent.mkdir(parents=True, exist_ok=True)
        exists = self.path.exists()
        self._file = open(self.path, 'a', newline='', encoding='utf-8')
        self._writer = csv.writer(self._file)
        if not exists:
            self._writer.writerow([
                'experiment', 'status', 'exit_code',
                'start_time', 'end_time', 'duration_min',
                'output_dir', 'image_list', 'steps', 'lr',
                'lora_path', 'notes'
            ])
            self._file.flush()

    def record(self, name, status, exit_code, start, end, out_dir,
               image_list, steps, lr, lora_path=None, notes=''):
        duration = (end - start) / 60
        self._writer.writerow([
            name, status, exit_code,
            datetime.fromtimestamp(start).isoformat(),
            datetime.fromtimestamp(end).isoformat(),
            f'{duration:.2f}',
            str(out_dir), str(image_list), steps, lr,
            str(lora_path or ''), notes
        ])
        self._file.flush()

    def close(self):
        self._file.close()


# ──────────────────── Training dispatch ────────────────────
def train_phase(name, base_args, image_list, out_dir, steps, lr,
                lora_path=None, dry_run=False, exp_log=None):
    """Launch a single training phase; returns True on success."""

    # Skip completed
    if not dry_run and is_experiment_done(out_dir):
        print(f"[SKIP] {name} already complete → {out_dir}")
        if exp_log:
            now = time.time()
            exp_log.record(name, 'skipped', 0, now, now,
                           out_dir, image_list, steps, lr, lora_path,
                           'already completed')
        return True

    cmd = [sys.executable, str(Path('kmc_lora') / 'scripts' / 'train_lora.py')]
    cmd += ['--base-model', base_args['base_model']]
    cmd += ['--image-list', image_list]
    cmd += ['--output-dir', out_dir]
    cmd += ['--instance-prompt', base_args['instance_prompt']]
    cmd += ['--resolution', str(base_args['resolution'])]
    cmd += ['--train-batch-size', str(base_args['train_batch_size'])]
    cmd += ['--gradient-accumulation-steps',
            str(base_args['gradient_accumulation_steps'])]
    cmd += ['--max-train-steps', str(steps)]
    cmd += ['--save-steps', str(base_args['save_steps'])]
    cmd += ['--lr', str(lr)]
    cmd += ['--lora-rank', str(base_args['lora_rank'])]
    cmd += ['--seed', str(base_args['seed'])]
    if base_args.get('max_gpu_memory_fraction', 1.0) < 1.0:
        cmd += ['--max-gpu-memory-fraction',
                str(base_args['max_gpu_memory_fraction'])]
    if base_args.get('validation_prompts'):
        cmd += ['--validation-prompts'] + base_args['validation_prompts']
    if lora_path:
        cmd += ['--lora-path', lora_path]

    if dry_run:
        print(' '.join(cmd))
        return True

    log_file = str(Path(out_dir) / 'stdout.log')
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"[RUN] {name}  →  {out_dir}")
    print(f"  steps={steps}  lr={lr}  images={image_list}")
    print(f"{'='*60}")

    rc = run(cmd, log_file=log_file)
    t1 = time.time()

    status = 'ok' if rc == 0 else 'FAIL'
    duration = (t1 - t0) / 60
    print(f"[{status}] {name}  ({duration:.1f} min, exit={rc})")

    if exp_log:
        exp_log.record(name, status, rc, t0, t1,
                       out_dir, image_list, steps, lr, lora_path)
    return rc == 0


# ──────────────────── Main ────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='kmc_lora/configs/base.yaml')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--quality-ratio', type=float, default=0.3)
    ap.add_argument('--skip-preprocessing', action='store_true',
                    help='Skip scan/features/cluster/curriculum/splits')
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, 'r', encoding='utf-8'))

    artifacts_dir = Path(cfg['paths']['artifacts_dir'])
    results_dir = Path(cfg['paths']['results_dir'])
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save a copy of the config file alongside results
    import shutil
    shutil.copy2(args.config, results_dir / 'config_used.yaml')

    metadata = artifacts_dir / 'metadata.csv'
    features = artifacts_dir / 'features.npy'
    paths_json = artifacts_dir / 'paths.json'
    cq_csv = artifacts_dir / 'cluster_quality.csv'
    curriculum_csv = artifacts_dir / 'curriculum.csv'
    splits_dir = artifacts_dir / 'splits'
    phases_dir = artifacts_dir / 'phases'

    # ── Preprocessing (idempotent) ──
    if not args.skip_preprocessing:
        if not metadata.exists():
            run([sys.executable, 'kmc_lora/scripts/scan_dataset.py',
                 '--dataset-dir', cfg['paths']['dataset_dir'],
                 '--out', str(metadata),
                 '--convert-gif', '--converted-dir',
                 str(artifacts_dir / 'gif_frames')],
                dry_run=args.dry_run)

        if not features.exists():
            run([sys.executable, 'kmc_lora/scripts/compute_features.py',
                 '--metadata', str(metadata),
                 '--out-features', str(features),
                 '--out-paths', str(paths_json),
                 '--batch-size', '8'],
                dry_run=args.dry_run)

        if not cq_csv.exists():
            aesthetics_weights = None
            if (cfg['quality'].get('use_aesthetics') and
                    cfg['quality'].get('aesthetics_model_url')):
                aesthetics_weights = artifacts_dir / 'aesthetics_linear.pth'
                if not aesthetics_weights.exists():
                    run([sys.executable, 'kmc_lora/scripts/download_file.py',
                         '--url', cfg['quality']['aesthetics_model_url'],
                         '--out', str(aesthetics_weights)],
                        dry_run=args.dry_run)
            cmd = [sys.executable, 'kmc_lora/scripts/cluster_and_quality.py',
                   '--metadata', str(metadata),
                   '--features', str(features),
                   '--out', str(cq_csv),
                   '--k-min', str(cfg['curriculum']['k_range'][0]),
                   '--k-max', str(cfg['curriculum']['k_range'][1])]
            if cfg['quality'].get('use_aesthetics'):
                cmd += ['--use-aesthetics', '--aesthetics-weights',
                        str(aesthetics_weights)]
            run(cmd, dry_run=args.dry_run)

        if not curriculum_csv.exists():
            run([sys.executable, 'kmc_lora/scripts/build_curriculum.py',
                 '--in-csv', str(cq_csv),
                 '--out-csv', str(curriculum_csv),
                 '--phase1', str(cfg['curriculum']['phase1_ratio']),
                 '--phase2', str(cfg['curriculum']['phase2_ratio']),
                 '--phase3', str(cfg['curriculum']['phase3_ratio']),
                 '--w-quality', str(cfg['curriculum']['quality_weight']),
                 '--w-typicality', str(cfg['curriculum']['typicality_weight']),
                 '--w-heterogeneity',
                     str(cfg['curriculum']['heterogeneity_weight']),
                 '--out-dir', str(phases_dir)],
                dry_run=args.dry_run)

        if not splits_dir.exists():
            run([sys.executable, 'kmc_lora/scripts/make_splits.py',
                 '--in-csv', str(curriculum_csv),
                 '--out-dir', str(splits_dir),
                 '--medium-percentile',
                     str(cfg['splits']['d_medium_keep_percentile']),
                 '--sub-50', str(cfg['splits']['sub_50']),
                 '--sub-25', str(cfg['splits']['sub_25'])],
                dry_run=args.dry_run)

    quality_list, anti_list = build_lists(
        artifacts_dir, quality_ratio=args.quality_ratio, dry_run=args.dry_run)

    base_args = {
        'base_model': cfg['model']['base_model'],
        'instance_prompt': cfg['prompts']['instance_prompt'],
        'validation_prompts': cfg['prompts']['validation_prompts'],
        'resolution': cfg['model']['resolution'],
        'train_batch_size': cfg['model']['train_batch_size'],
        'gradient_accumulation_steps':
            cfg['model']['gradient_accumulation_steps'],
        'save_steps': cfg['model']['save_steps'],
        'lora_rank': cfg['model']['lora_rank'],
        'max_gpu_memory_fraction': cfg['model'].get('max_gpu_memory_fraction',
                                                    1.0),
        'seed': cfg['experiment']['seed'],
    }

    total_steps = cfg['model']['max_train_steps']
    p1 = int(total_steps * cfg['curriculum']['phase1_ratio'])
    p2 = int(total_steps * cfg['curriculum']['phase2_ratio'])
    p3 = total_steps - p1 - p2

    exp_log = ExperimentLog(results_dir)
    ok_count, fail_count, skip_count = 0, 0, 0

    def do(name, img_list, out, steps, lr, lora=None):
        nonlocal ok_count, fail_count, skip_count
        if is_experiment_done(out) and not args.dry_run:
            skip_count += 1
        ok = train_phase(name, base_args, img_list, out, steps, lr,
                         lora, dry_run=args.dry_run, exp_log=exp_log)
        if ok:
            ok_count += 1
        else:
            fail_count += 1
        return ok

    # ───── Experiment A: KMC vs Random across splits ─────
    split_files = ['D-High.txt', 'D-Medium.txt', 'D-Low.txt',
                   'D-Sub-50.txt', 'D-Sub-25.txt']
    for split in split_files:
        sp = str(Path(splits_dir) / split)
        tag = split.replace('.txt', '')
        split_phase_lists = build_split_phase_lists(
            curriculum_csv,
            sp,
            artifacts_dir / 'split_phases' / tag,
        )

        # Random baseline
        do(f'Random_{tag}', sp,
           str(results_dir / f'Random_{tag}'),
           total_steps, 1e-4)

        # KMC 3-phase
        kmc = results_dir / f'KMC_{tag}'
        do(f'KMC_{tag}_P1', split_phase_lists['phase1'],
           str(kmc / 'phase1'), p1, 5e-5)
        do(f'KMC_{tag}_P2', split_phase_lists['phase2'],
           str(kmc / 'phase2'), p2, 1e-4,
           str(kmc / 'phase1' / 'final'))
        do(f'KMC_{tag}_P3', split_phase_lists['phase3'],
           str(kmc / 'phase3'), p3, 2e-5,
           str(kmc / 'phase2' / 'final'))

    # ───── Experiment B: Ablations on D-High ─────
    phase1_list = str(Path(phases_dir) / 'phase1.txt')
    phase2_list = str(Path(phases_dir) / 'phase2.txt')
    phase3_list = str(Path(phases_dir) / 'phase3.txt')

    # No Phase1
    np1 = results_dir / 'Ablation_NoPhase1'
    do('Ablation_NoP1_P2', phase2_list, str(np1 / 'phase2'), p2, 1e-4)
    do('Ablation_NoP1_P3', phase3_list, str(np1 / 'phase3'), p3, 2e-5,
       str(np1 / 'phase2' / 'final'))

    # No Phase2
    np2 = results_dir / 'Ablation_NoPhase2'
    do('Ablation_NoP2_P1', phase1_list, str(np2 / 'phase1'), p1, 5e-5)
    do('Ablation_NoP2_P3', phase3_list, str(np2 / 'phase3'), p3, 2e-5,
       str(np2 / 'phase1' / 'final'))

    # No Phase3
    np3 = results_dir / 'Ablation_NoPhase3'
    do('Ablation_NoP3_P1', phase1_list, str(np3 / 'phase1'), p1, 5e-5)
    do('Ablation_NoP3_P2', phase2_list, str(np3 / 'phase2'), p2, 1e-4,
       str(np3 / 'phase1' / 'final'))

    # Phase1 only
    do('Ablation_P1Only', phase1_list,
       str(results_dir / 'Ablation_Phase1Only' / 'phase1'), p1, 5e-5)

    # Phase3 only
    do('Ablation_P3Only', phase3_list,
       str(results_dir / 'Ablation_Phase3Only' / 'phase3'), p3, 2e-5)

    # ───── Experiment C: Additional baselines ─────
    do('Quality_Filter', quality_list,
       str(results_dir / 'Quality_Filter'), total_steps, 1e-4)
    do('Anti_Curriculum', anti_list,
       str(results_dir / 'Anti_Curriculum'), total_steps, 1e-4)

    exp_log.close()

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"ALL EXPERIMENTS DONE")
    print(f"  OK: {ok_count}  |  FAIL: {fail_count}  |  SKIPPED: {skip_count}")
    print(f"  Log: {exp_log.path}")
    print(f"{'='*60}")

    # Save overall summary JSON
    summary = {
        'completed_at': datetime.now().isoformat(),
        'config': args.config,
        'ok': ok_count, 'fail': fail_count, 'skipped': skip_count,
        'results_dir': str(results_dir),
    }
    with open(results_dir / 'run_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == '__main__':
    main()
