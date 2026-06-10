"""批量重跑 FID + CLIP 评估 (图片已生成好)"""
import subprocess, sys, time
from pathlib import Path
from datetime import datetime

RESULTS = Path('kmc_lora/results')
REAL_LIST = 'kmc_lora/artifacts/splits/D-High.txt'
OUT_CSV = str(RESULTS / 'eval_all_fixed.csv')
PROMPTS = [
    'a hand-drawn anime storyboard of a brave knight',
    'a hand-drawn anime storyboard of a city at night',
    'a hand-drawn anime storyboard of a forest spirit',
]

experiments = [
    'KMC_D-High', 'KMC_D-Medium', 'KMC_D-Low', 'KMC_D-Sub-50', 'KMC_D-Sub-25',
    'Ablation_NoPhase1', 'Ablation_NoPhase2', 'Ablation_NoPhase3',
    'Ablation_Phase1Only', 'Ablation_Phase3Only',
    'Random_D-High', 'Random_D-Medium', 'Random_D-Low',
    'Random_D-Sub-50', 'Random_D-Sub-25',
    'Quality_Filter', 'Anti_Curriculum',
]

# Remove old eval CSV
if Path(OUT_CSV).exists():
    Path(OUT_CSV).unlink()

print(f'开始评估 {len(experiments)} 个实验 | {datetime.now().isoformat()}')
print('='*60)

for i, exp in enumerate(experiments, 1):
    gen_dir = RESULTS / exp / 'generated_fixed'
    if not gen_dir.exists():
        print(f'[{i}/{len(experiments)}] SKIP {exp}: no generated_fixed')
        continue

    n_images = len(list(gen_dir.glob('*.png')))
    print(f'[{i}/{len(experiments)}] {exp} ({n_images} images) ...', end=' ', flush=True)

    t0 = time.time()
    cmd = [
        sys.executable, 'kmc_lora/scripts/evaluate_fid.py',
        '--real-list', REAL_LIST,
        '--gen-dir', str(gen_dir),
        '--experiment-name', exp,
        '--max-real', '500',
        '--max-gen', '500',
        '--out-csv', OUT_CSV,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    dt = time.time() - t0

    if result.returncode == 0:
        # Extract FID and CLIP from output
        for line in result.stdout.split('\n'):
            if 'FID' in line and ':' in line:
                print(line.strip(), end=' | ')
            elif 'CLIP' in line and ':' in line:
                print(line.strip(), end=' ')
        print(f'({dt:.0f}s)')
    else:
        print(f'FAIL ({dt:.0f}s)')
        err = result.stderr[:200] if result.stderr else result.stdout[:200]
        print(f'  Error: {err}')

print('='*60)
print(f'完成 | {datetime.now().isoformat()}')
print(f'结果保存至: {OUT_CSV}')
