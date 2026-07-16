"""
Full evaluation pipeline for KMC-LoRA experiments.
For each dataset × experiment:
  1. Generate sample images from the final LoRA
  2. Compute FID and CLIP Score
  3. Save eval_metrics.json
"""
import argparse, json, subprocess, sys, csv, os, gc, time
from pathlib import Path
from datetime import datetime


# ── Experiment definitions ──
RANDOM_SPLITS = ['Random_D-High', 'Random_D-Medium', 'Random_D-Low',
                 'Random_D-Sub-50', 'Random_D-Sub-25']
KMC_SPLITS = ['KMC_D-High', 'KMC_D-Medium', 'KMC_D-Low',
              'KMC_D-Sub-50', 'KMC_D-Sub-25']
ABLATIONS = ['Ablation_NoPhase1', 'Ablation_NoPhase2', 'Ablation_NoPhase3',
             'Ablation_Phase1Only', 'Ablation_Phase3Only']
BASELINES = ['Quality_Filter', 'Anti_Curriculum']

# Final phase for each experiment type
FINAL_PHASE = {
    'Ablation_NoPhase1': 'phase3',
    'Ablation_NoPhase2': 'phase3',
    'Ablation_NoPhase3': 'phase2',
    'Ablation_Phase1Only': 'phase1',
    'Ablation_Phase3Only': 'phase3',
}


def get_lora_path(results_dir, exp_name):
    """Return the path to the final LoRA for a given experiment."""
    rd = Path(results_dir)
    # Random / Baselines: single directory
    if exp_name in RANDOM_SPLITS + BASELINES:
        final = rd / exp_name / 'final'
        if final.exists():
            return str(final)
        return None
    # KMC splits: always phase3
    if exp_name in KMC_SPLITS:
        final = rd / exp_name / 'phase3' / 'final'
        if final.exists():
            return str(final)
        return None
    # Ablations: specific final phase
    if exp_name in ABLATIONS:
        phase = FINAL_PHASE.get(exp_name, 'phase3')
        final = rd / exp_name / phase / 'final'
        if final.exists():
            return str(final)
        return None
    return None


def get_real_image_list(dataset_key, artifacts_base):
    """Return the real image list path (D-High split = all images)."""
    ab = Path(artifacts_base)
    if dataset_key == 'anime_student':
        p = ab / 'splits' / 'D-High.txt'
    else:
        p = ab / dataset_key / 'splits' / 'D-High.txt'
    if p.exists():
        return str(p)
    return None


def run_generate(base_model, lora_path, prompts, out_dir,
                 num_per_prompt=10, seed=42, steps=30):
    """Generate images using generate_samples.py."""
    cmd = [
        sys.executable, 'kmc_lora/scripts/generate_samples.py',
        '--base-model', base_model,
        '--lora-path', lora_path,
        '--prompts'] + prompts + [
        '--out-dir', out_dir,
        '--num-per-prompt', str(num_per_prompt),
        '--seed', str(seed),
        '--steps', str(steps),
    ]
    print(f"  [GEN] {out_dir}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        print(f"  [FAIL] generate: {result.stderr[:500]}")
        return False
    return True


def run_evaluate(real_list, gen_dir, prompts, out_csv, exp_name):
    """Evaluate FID and CLIP Score."""
    cmd = [
        sys.executable, 'kmc_lora/scripts/evaluate_fid.py',
        '--real-list', real_list,
        '--gen-dir', gen_dir,
        '--experiment-name', exp_name,
        '--max-real', '500',
        '--max-gen', '500',
    ]
    if out_csv:
        cmd += ['--out-csv', out_csv]
    if prompts:
        cmd += ['--prompts'] + prompts
    print(f"  [EVAL] {exp_name}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        print(f"  [FAIL] evaluate: {result.stderr[:500]}")
        return False
    # Print key results
    for line in result.stdout.strip().split('\n'):
        if 'FID' in line or 'CLIP' in line:
            print(f"    {line}")
    return True


def run_pipeline(dataset_key, results_dir, artifacts_base, base_model,
                 prompts, num_per_prompt=10, seed=42):
    """Run full generate+evaluate pipeline for one dataset."""
    all_exps = RANDOM_SPLITS + KMC_SPLITS + ABLATIONS + BASELINES
    real_list = get_real_image_list(dataset_key, artifacts_base)
    if not real_list:
        print(f"[SKIP] No real image list for {dataset_key}")
        return

    out_csv = str(Path(results_dir) / 'eval_all.csv')
    done = 0
    fail = 0

    for exp in all_exps:
        lora_path = get_lora_path(results_dir, exp)
        if not lora_path:
            print(f"  [SKIP] {exp}: no final LoRA")
            fail += 1
            continue

        gen_dir = str(Path(results_dir) / exp / 'generated')
        eval_json = Path(gen_dir) / 'eval_metrics.json'

        # Skip if already evaluated
        if eval_json.exists():
            print(f"  [SKIP] {exp}: already evaluated")
            done += 1
            continue

        # Generate
        ok = run_generate(base_model, lora_path, prompts, gen_dir,
                          num_per_prompt=num_per_prompt, seed=seed)
        if not ok:
            fail += 1
            continue

        # Evaluate
        ok = run_evaluate(real_list, gen_dir, prompts, out_csv, exp)
        if ok:
            done += 1
        else:
            fail += 1

        # Free GPU memory between experiments
        gc.collect()
        if hasattr(__builtins__, '__import__'):
            try:
                import torch
                torch.cuda.empty_cache()
            except:
                pass

    print(f"\n  {dataset_key}: done={done}, fail={fail}")


# ── Dataset configs ──
DATASET_CONFIGS = {
    'anime_student': {
        'results_dir': 'kmc_lora/results',
        'artifacts_base': 'kmc_lora/artifacts',
        'prompts': [
            "a hand-drawn anime storyboard of a brave knight",
            "a hand-drawn anime storyboard of a city at night",
            "a hand-drawn anime storyboard of a forest spirit",
        ],
    },
    'wikiart_mixed': {
        'results_dir': 'kmc_lora/results/wikiart_mixed',
        'artifacts_base': 'kmc_lora/artifacts',
        'prompts': [
            "a painting of a sunset over mountains",
            "a painting of a woman reading a book",
            "a painting of a bustling city street",
            "a painting of flowers in a vase",
        ],
    },
    'dreambooth_mixed': {
        'results_dir': 'kmc_lora/results/dreambooth_mixed',
        'artifacts_base': 'kmc_lora/artifacts',
        'prompts': [
            "a photo of a sks dog sitting on a beach",
            "a photo of a sks cat on a sofa",
            "a photo of a sks backpack in a forest",
        ],
    },
    'dreambooth_single': {
        'results_dir': 'kmc_lora/results/dreambooth_single',
        'artifacts_base': 'kmc_lora/artifacts',
        'prompts': [
            "a photo of a sks dog on the beach",
            "a photo of a sks dog in a garden",
            "a photo of a sks dog wearing a hat",
        ],
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='*',
                    default=list(DATASET_CONFIGS.keys()),
                    help='Which datasets to evaluate')
    ap.add_argument('--base-model', default='runwayml/stable-diffusion-v1-5')
    ap.add_argument('--num-per-prompt', type=int, default=10)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    print(f"{'='*60}")
    print(f"KMC-LoRA Full Evaluation Pipeline")
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Datasets: {args.datasets}")
    print(f"{'='*60}\n")

    for ds in args.datasets:
        cfg = DATASET_CONFIGS[ds]
        print(f"\n{'='*40}")
        print(f"Dataset: {ds}")
        print(f"{'='*40}")
        run_pipeline(
            dataset_key=ds,
            results_dir=cfg['results_dir'],
            artifacts_base=cfg['artifacts_base'],
            base_model=args.base_model,
            prompts=cfg['prompts'],
            num_per_prompt=args.num_per_prompt,
            seed=args.seed,
        )

    print(f"\n{'='*60}")
    print(f"ALL EVALUATIONS COMPLETE: {datetime.now().isoformat()}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
