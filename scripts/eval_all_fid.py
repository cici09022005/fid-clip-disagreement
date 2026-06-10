"""
一体化 FID 评估：只加载一次 InceptionV3，批量评估所有实验。
用 GPU 加速 + 批处理，大幅减少总评估时间。
"""
import csv, gc, json, time
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from PIL import Image
from torchmetrics.image.fid import FrechetInceptionDistance

RESULTS = Path('kmc_lora/results')
REAL_LIST = Path('kmc_lora/artifacts/splits/D-High.txt')
OUT_CSV = RESULTS / 'eval_all_fixed.csv'
BATCH_SIZE = 16
MAX_REAL = 500
MAX_GEN = 500

EXPERIMENTS = [
    'KMC_D-High', 'KMC_D-Medium', 'KMC_D-Low', 'KMC_D-Sub-50', 'KMC_D-Sub-25',
    'Ablation_NoPhase1', 'Ablation_NoPhase2', 'Ablation_NoPhase3',
    'Ablation_Phase1Only', 'Ablation_Phase3Only',
    'Random_D-High', 'Random_D-Medium', 'Random_D-Low',
    'Random_D-Sub-50', 'Random_D-Sub-25',
    'Quality_Filter', 'Anti_Curriculum',
]


def load_image_batch(paths):
    """Load a batch of images as uint8 tensors [N, 3, 299, 299]."""
    tensors = []
    for p in paths:
        try:
            img = Image.open(p).convert('RGB').resize((299, 299))
            arr = np.array(img, dtype=np.uint8)
            t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
            tensors.append(t)
        except Exception as e:
            print(f'  WARN: skip {p}: {e}')
    if not tensors:
        return torch.empty(0, 3, 299, 299, dtype=torch.uint8)
    return torch.stack(tensors)


def compute_fid_gpu(real_paths, gen_paths, device):
    """Compute FID on GPU with batch processing."""
    fid = FrechetInceptionDistance(feature=2048).to(device)

    # Feed real images in batches
    for i in range(0, len(real_paths), BATCH_SIZE):
        batch = load_image_batch(real_paths[i:i + BATCH_SIZE])
        if batch.shape[0] > 0:
            fid.update(batch.to(device), real=True)

    # Feed generated images in batches
    for i in range(0, len(gen_paths), BATCH_SIZE):
        batch = load_image_batch(gen_paths[i:i + BATCH_SIZE])
        if batch.shape[0] > 0:
            fid.update(batch.to(device), real=False)

    score = fid.compute().item()
    del fid
    torch.cuda.empty_cache()
    return score


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    print(f'Start: {datetime.now().isoformat()}')
    print('=' * 60)

    # Load real image paths once
    real_paths = [p.strip() for p in REAL_LIST.read_text('utf-8').splitlines() if p.strip()]
    real_paths = real_paths[:MAX_REAL]
    print(f'Real images: {len(real_paths)}')

    # Pre-load real images into memory to avoid re-reading for each experiment
    print('Pre-loading real images ...')
    real_batches = []
    for i in range(0, len(real_paths), BATCH_SIZE):
        batch = load_image_batch(real_paths[i:i + BATCH_SIZE])
        real_batches.append(batch)
    print(f'  {sum(b.shape[0] for b in real_batches)} real images loaded')

    # Remove old CSV
    if OUT_CSV.exists():
        OUT_CSV.unlink()

    results = []
    for idx, exp in enumerate(EXPERIMENTS, 1):
        gen_dir = RESULTS / exp / 'generated_fixed'
        if not gen_dir.exists():
            print(f'[{idx}/{len(EXPERIMENTS)}] SKIP {exp}: no generated_fixed')
            continue

        gen_paths = sorted([str(p) for p in gen_dir.glob('*.png')])[:MAX_GEN]
        print(f'[{idx}/{len(EXPERIMENTS)}] {exp} ({len(gen_paths)} gen) ...', end=' ', flush=True)

        t0 = time.time()

        # Create fresh FID metric
        fid = FrechetInceptionDistance(feature=2048).to(device)

        # Feed pre-loaded real batches
        for batch in real_batches:
            if batch.shape[0] > 0:
                fid.update(batch.to(device), real=True)

        # Feed generated images
        for i in range(0, len(gen_paths), BATCH_SIZE):
            batch = load_image_batch(gen_paths[i:i + BATCH_SIZE])
            if batch.shape[0] > 0:
                fid.update(batch.to(device), real=False)

        fid_score = fid.compute().item()
        dt = time.time() - t0

        print(f'FID = {fid_score:.2f} ({dt:.1f}s)')

        row = {
            'experiment': exp,
            'fid': round(fid_score, 4),
            'clip_score': '',
            'num_real': len(real_paths),
            'num_gen': len(gen_paths),
            'gen_dir': str(gen_dir),
        }
        results.append(row)

        # Save eval_metrics.json
        with open(gen_dir / 'eval_metrics.json', 'w') as f:
            json.dump(row, f, indent=2)

        del fid
        torch.cuda.empty_cache()
        gc.collect()

    # Write CSV
    if results:
        with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader()
            w.writerows(results)
        print(f'\n结果保存至: {OUT_CSV}')

    print('=' * 60)
    print(f'完成: {datetime.now().isoformat()}')

    # Print summary table
    print(f'\n{"Experiment":<25s} {"FID":>10s}')
    print('-' * 37)
    for r in results:
        print(f'{r["experiment"]:<25s} {r["fid"]:10.2f}')


if __name__ == '__main__':
    main()
