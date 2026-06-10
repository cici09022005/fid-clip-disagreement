"""
快速 FID 评估 — 使用 pytorch-fid (scipy.linalg.sqrtm)，比 torchmetrics 快很多。
GPU 加速 InceptionV3 特征提取 + scipy 计算 FID 距离。
"""
import csv, gc, json, sys, time
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from pytorch_fid.fid_score import calculate_fid_given_paths
from pytorch_fid.inception import InceptionV3

RESULTS = Path('kmc_lora/results')
REAL_LIST = Path('kmc_lora/artifacts/splits/D-High.txt')
OUT_CSV = RESULTS / 'eval_all_fixed.csv'

EXPERIMENTS = [
    'KMC_D-High', 'KMC_D-Medium', 'KMC_D-Low', 'KMC_D-Sub-50', 'KMC_D-Sub-25',
    'Ablation_NoPhase1', 'Ablation_NoPhase2', 'Ablation_NoPhase3',
    'Ablation_Phase1Only', 'Ablation_Phase3Only',
    'Random_D-High', 'Random_D-Medium', 'Random_D-Low',
    'Random_D-Sub-50', 'Random_D-Sub-25',
    'Quality_Filter', 'Anti_Curriculum',
]


def prepare_real_dir():
    """Create a temp directory with resized real images for pytorch-fid."""
    real_paths = [p.strip() for p in REAL_LIST.read_text('utf-8').splitlines() if p.strip()]
    real_dir = RESULTS / '_real_for_fid'
    if real_dir.exists() and len(list(real_dir.glob('*.png'))) >= min(len(real_paths), 500):
        return str(real_dir), len(list(real_dir.glob('*.png')))

    real_dir.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    for i, p in enumerate(real_paths[:500]):
        src = Path(p)
        if src.exists():
            dst = real_dir / f'{i:04d}.png'
            if not dst.exists():
                img = Image.open(src).convert('RGB').resize((512, 512), Image.LANCZOS)
                img.save(dst)
    n = len(list(real_dir.glob('*.png')))
    return str(real_dir), n


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Start: {datetime.now().isoformat()}')
    print('=' * 60)

    # Prepare real images directory
    print('Preparing real images ...')
    real_dir, n_real = prepare_real_dir()
    print(f'  {n_real} real images in {real_dir}')

    # Remove old CSV
    if OUT_CSV.exists():
        OUT_CSV.unlink()

    results = []
    for idx, exp in enumerate(EXPERIMENTS, 1):
        gen_dir = RESULTS / exp / 'generated_fixed'
        if not gen_dir.exists():
            print(f'[{idx}/{len(EXPERIMENTS)}] SKIP {exp}: no generated_fixed')
            continue

        n_gen = len(list(gen_dir.glob('*.png')))
        print(f'[{idx}/{len(EXPERIMENTS)}] {exp} ({n_gen} gen) ...', end=' ', flush=True)

        t0 = time.time()
        try:
            fid_score = calculate_fid_given_paths(
                [real_dir, str(gen_dir)],
                batch_size=32,
                device=device,
                dims=2048,
                num_workers=0,
            )
            dt = time.time() - t0
            print(f'FID = {fid_score:.2f} ({dt:.1f}s)')
        except Exception as e:
            dt = time.time() - t0
            fid_score = -1
            print(f'FAIL ({dt:.1f}s): {e}')

        row = {
            'experiment': exp,
            'fid': round(fid_score, 4),
            'clip_score': '',
            'num_real': n_real,
            'num_gen': n_gen,
            'gen_dir': str(gen_dir),
        }
        results.append(row)

        # Save per-experiment
        with open(gen_dir / 'eval_metrics.json', 'w') as f:
            json.dump(row, f, indent=2)

        gc.collect()
        torch.cuda.empty_cache()

    # Write CSV
    if results:
        with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader()
            w.writerows(results)
        print(f'\n结果保存至: {OUT_CSV}')

    print('=' * 60)
    print(f'完成: {datetime.now().isoformat()}')

    # Summary table
    print(f'\n{"Experiment":<25s} {"FID":>10s}')
    print('-' * 37)
    for r in results:
        fval = r["fid"]
        fstr = f'{fval:.2f}' if fval >= 0 else 'FAIL'
        print(f'{r["experiment"]:<25s} {fstr:>10s}')


if __name__ == '__main__':
    main()
