"""
Run FID + CLIP evaluation ONLY (assumes images already generated).
Scans for experiment/generated/ dirs and computes metrics.
"""
import argparse, json, subprocess, sys, gc, os
from pathlib import Path
from datetime import datetime

DATASETS = {
    'anime_student': {
        'results_dir': 'kmc_lora/results',
        'real_list': 'kmc_lora/artifacts/splits/D-High.txt',
        'prompts': [
            "a hand-drawn anime storyboard of a brave knight",
            "a hand-drawn anime storyboard of a city at night",
            "a hand-drawn anime storyboard of a forest spirit",
        ],
    },
    'wikiart_mixed': {
        'results_dir': 'kmc_lora/results/wikiart_mixed',
        'real_list': 'kmc_lora/artifacts/wikiart_mixed/splits/D-High.txt',
        'prompts': [
            "a painting of a sunset over mountains",
            "a painting of a woman reading a book",
            "a painting of a bustling city street",
            "a painting of flowers in a vase",
        ],
    },
    'dreambooth_mixed': {
        'results_dir': 'kmc_lora/results/dreambooth_mixed',
        'real_list': 'kmc_lora/artifacts/dreambooth_mixed/splits/D-High.txt',
        'prompts': [
            "a photo of a sks dog sitting on a beach",
            "a photo of a sks cat on a sofa",
            "a photo of a sks backpack in a forest",
        ],
    },
    'dreambooth_single': {
        'results_dir': 'kmc_lora/results/dreambooth_single',
        'real_list': 'kmc_lora/artifacts/dreambooth_single/splits/D-High.txt',
        'prompts': [
            "a photo of a sks dog on the beach",
            "a photo of a sks dog in a garden",
            "a photo of a sks dog wearing a hat",
        ],
    },
}

EXPERIMENTS = [
    'Random_D-High', 'KMC_D-High',
    'Random_D-Medium', 'KMC_D-Medium',
    'Random_D-Low', 'KMC_D-Low',
    'Random_D-Sub-50', 'KMC_D-Sub-50',
    'Random_D-Sub-25', 'KMC_D-Sub-25',
    'Ablation_NoPhase1', 'Ablation_NoPhase2', 'Ablation_NoPhase3',
    'Ablation_Phase1Only', 'Ablation_Phase3Only',
    'Quality_Filter', 'Anti_Curriculum',
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='*', default=list(DATASETS.keys()))
    args = ap.parse_args()

    print(f"={'='*50}")
    print(f"KMC-LoRA Evaluation Only (FID + CLIP)")
    print(f"Started: {datetime.now().isoformat()}")
    print(f"={'='*50}\n")

    for ds_key in args.datasets:
        cfg = DATASETS[ds_key]
        rd = Path(cfg['results_dir'])
        real_list = cfg['real_list']
        prompts = cfg['prompts']
        out_csv = str(rd / 'eval_all.csv')

        # Remove old eval_all.csv to avoid duplicates
        if Path(out_csv).exists():
            Path(out_csv).unlink()

        print(f"\n{'='*40}")
        print(f"Dataset: {ds_key}")
        print(f"{'='*40}")

        ok = 0
        fail = 0
        for exp in EXPERIMENTS:
            gen_dir = rd / exp / 'generated'
            if not gen_dir.exists() or len(list(gen_dir.glob('*.png'))) < 5:
                print(f"  [SKIP] {exp}: no generated images")
                fail += 1
                continue

            eval_json = gen_dir / 'eval_metrics.json'
            if eval_json.exists():
                print(f"  [SKIP] {exp}: already evaluated")
                ok += 1
                continue

            cmd = [
                sys.executable, 'kmc_lora/scripts/evaluate_fid.py',
                '--real-list', real_list,
                '--gen-dir', str(gen_dir),
                '--experiment-name', exp,
                '--max-real', '500',
                '--max-gen', '500',
                '--out-csv', out_csv,
                '--prompts',
            ] + prompts

            print(f"  [EVAL] {exp} ...", end='', flush=True)
            try:
                result = subprocess.run(cmd, capture_output=True, text=True,
                                       timeout=300)
                if result.returncode == 0:
                    # Parse FID/CLIP from output
                    for line in result.stdout.split('\n'):
                        if 'FID' in line or 'CLIP' in line:
                            print(f"  {line.strip()}", end='')
                    print(f"  OK")
                    ok += 1
                else:
                    print(f"  FAIL")
                    print(f"    {result.stderr[:200]}")
                    fail += 1
            except subprocess.TimeoutExpired:
                print(f"  TIMEOUT")
                fail += 1

        print(f"\n  {ds_key}: ok={ok}, fail={fail}")

    print(f"\n{'='*50}")
    print(f"ALL EVALUATIONS COMPLETE: {datetime.now().isoformat()}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
