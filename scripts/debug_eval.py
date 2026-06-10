"""Quick debug script to check eval results."""
import json
from pathlib import Path

# Check D-High.txt
lines = open('kmc_lora/artifacts/splits/D-High.txt', encoding='utf-8').readlines()
print(f"D-High lines: {len(lines)}")
first = lines[0].strip()
print(f"First path: {first}")
p = Path(first)
print(f"Exists: {p.exists()}")
if not p.exists():
    # Try relative to workspace
    p2 = Path("g:/LaTex Paper/AIGC 02") / first
    print(f"Try absolute: {p2} -> {p2.exists()}")

# Check gen image counts and sizes
for exp in ['Random_D-High', 'KMC_D-High', 'Random_D-Medium', 'KMC_D-Medium']:
    gd = Path(f'kmc_lora/results/{exp}/generated')
    pngs = sorted(gd.glob('*.png'))
    sizes = [pp.stat().st_size for pp in pngs[:3]]
    print(f"{exp}: {len(pngs)} pngs, first sizes: {sizes}")

# Check eval_metrics.json
print("\n--- eval_metrics.json ---")
for exp in ['Random_D-High', 'KMC_D-High', 'Random_D-Medium', 'KMC_D-Medium']:
    ej = Path(f'kmc_lora/results/{exp}/generated/eval_metrics.json')
    if ej.exists():
        d = json.load(open(ej))
        print(f"  {exp}: fid={d['fid']}, clip={d['clip_score']}, num_real={d.get('num_real')}, num_gen={d.get('num_gen')}")
    else:
        print(f"  {exp}: no eval_metrics.json")

# Check if all gen dirs are actually different or symlinks
print("\n--- Gen dir identity ---")
import os
for exp in ['Random_D-High', 'KMC_D-High', 'Random_D-Medium']:
    gd = Path(f'kmc_lora/results/{exp}/generated')
    pngs = sorted(gd.glob('*.png'))
    if pngs:
        # hash first image to see if they differ
        import hashlib
        h = hashlib.md5(open(pngs[0], 'rb').read()).hexdigest()
        print(f"  {exp}: first_img={pngs[0].name}, md5={h}")
