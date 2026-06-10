import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import yaml


METHODS = {
    'p2-only': [('phase2', 'phase2', None, 1.0, 1e-4)],
    'p3-only-long': [('phase3', 'phase3', None, 1.0, 2e-5)],
    'p2-p3-hard': [
        ('phase2', 'phase2', None, 0.75, 1e-4),
        ('phase3', 'phase3', 'phase2', 0.25, 2e-5),
    ],
    'p2-p3-replay': [
        ('phase2', 'phase2', None, 0.75, 1e-4),
        ('phase3_replay', 'phase3', 'phase2', 0.25, 2e-5),
    ],
    'p2-p3-replay-long': [
        ('phase2', 'phase2', None, 2 / 3, 1e-4),
        ('phase3_replay', 'phase3', 'phase2', 1 / 3, 2e-5),
    ],
}

AUTO_STRATEGIES = {
    'scale-aware-v1': {
        'D-Sub-25': 'p2-p3-replay',
        'D-Sub-50': 'p2-only',
        'D-Medium': 'p3-only-long',
    },
}

METHOD_CHOICES = sorted(list(METHODS.keys()) + list(AUTO_STRATEGIES.keys()))


def resolve_method(method_name, split_name):
    if method_name in METHODS:
        return method_name
    if method_name in AUTO_STRATEGIES:
        split_map = AUTO_STRATEGIES[method_name]
        if split_name not in split_map:
            raise ValueError(
                f'auto strategy {method_name} has no verified rule for split {split_name}; '
                'use an explicit method or extend AUTO_STRATEGIES'
            )
        return split_map[split_name]
    raise ValueError(f'unknown method: {method_name}')


def run(cmd):
    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        raise RuntimeError(f'command failed with exit code {result.returncode}')
    return ''


def train_phase(base_model, instance_prompt, validation_prompts, resolution,
                batch_size, grad_accum, save_steps, lora_rank, seed,
                image_list, out_dir, steps, lr, lora_path=None):
    cmd = [
        sys.executable, 'kmc_lora/scripts/train_lora.py',
        '--base-model', base_model,
        '--image-list', image_list,
        '--output-dir', str(out_dir),
        '--instance-prompt', instance_prompt,
        '--max-train-steps', str(steps),
        '--save-steps', str(save_steps),
        '--lr', str(lr),
        '--seed', str(seed),
        '--resolution', str(resolution),
        '--train-batch-size', str(batch_size),
        '--gradient-accumulation-steps', str(grad_accum),
        '--lora-rank', str(lora_rank),
    ]
    if validation_prompts:
        cmd += ['--validation-prompts'] + validation_prompts
    if lora_path:
        cmd += ['--lora-path', str(lora_path)]
    return run(cmd)


def generate_images(base_model, lora_path, prompts, out_dir, num_per_prompt, seed):
    cmd = [
        sys.executable, 'kmc_lora/scripts/generate_samples.py',
        '--base-model', base_model,
        '--lora-path', str(lora_path),
        '--prompts', *prompts,
        '--out-dir', str(out_dir),
        '--num-per-prompt', str(num_per_prompt),
        '--seed', str(seed),
    ]
    return run(cmd)


def evaluate(real_list, gen_dir, experiment_name, max_real, max_gen, out_csv):
    cmd = [
        sys.executable, 'kmc_lora/scripts/evaluate_fid.py',
        '--real-list', str(real_list),
        '--gen-dir', str(gen_dir),
        '--experiment-name', experiment_name,
        '--max-real', str(max_real),
        '--max-gen', str(max_gen),
        '--out-csv', str(out_csv),
    ]
    return run(cmd)


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding='utf-8')


def ensure_split_phase_lists(artifacts_dir: Path, split_name: str, cfg: dict) -> Path:
    split_dir = artifacts_dir / 'split_phases' / split_name
    required = [split_dir / 'phase1.txt', split_dir / 'phase2.txt', split_dir / 'phase3.txt']
    if all(path.exists() for path in required):
        return split_dir

    curriculum_csv = artifacts_dir / 'curriculum.csv'
    split_file = artifacts_dir / 'splits' / f'{split_name}.txt'
    if not curriculum_csv.exists():
        raise FileNotFoundError(f'missing curriculum file: {curriculum_csv}')
    if not split_file.exists():
        raise FileNotFoundError(f'missing split file: {split_file}')

    df = pd.read_csv(curriculum_csv)
    split_paths = {
        p.strip()
        for p in split_file.read_text(encoding='utf-8').splitlines()
        if p.strip()
    }
    split_df = df[df['path'].isin(split_paths)].copy()
    if split_df.empty:
        raise ValueError(f'No overlap between curriculum and split: {split_file}')

    split_dir.mkdir(parents=True, exist_ok=True)
    if all(col in split_df.columns for col in ['phase1', 'phase2', 'phase3']):
        phase_to_paths = {
            phase_name: split_df.loc[split_df[phase_name].astype(bool), 'path']
            for phase_name in ['phase1', 'phase2', 'phase3']
        }
    else:
        phase1_ratio = cfg['curriculum']['phase1_ratio']
        phase3_ratio = cfg['curriculum']['phase3_ratio']
        ordered = split_df.sort_values('difficulty', ascending=True).reset_index(drop=True)
        n = len(ordered)
        n1 = int(n * phase1_ratio)
        n3 = max(1, int(round(n * phase3_ratio)))

        phase1_paths = ordered.loc[:max(0, n1 - 1), 'path'] if n1 > 0 else ordered.iloc[0:0]['path']
        phase2_paths = ordered['path']
        typical_top_idx = ordered.sort_values('typicality', ascending=False).head(n3).index
        phase3_paths = ordered.loc[typical_top_idx, 'path']

        phase_to_paths = {
            'phase1': phase1_paths,
            'phase2': phase2_paths,
            'phase3': phase3_paths,
        }

    for phase_name, phase_paths in phase_to_paths.items():
        phase_path = split_dir / f'{phase_name}.txt'
        phase_paths.to_csv(phase_path, index=False, header=False)
    return split_dir


def main():
    ap = argparse.ArgumentParser(description='Run refinement-method experiments with smoke-first support')
    ap.add_argument('--config', default='kmc_lora/configs/base.yaml')
    ap.add_argument('--method', required=True, choices=METHOD_CHOICES)
    ap.add_argument('--split', default='D-Sub-25')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out-root', default='kmc_lora/results/refinement_search')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--num-per-prompt', type=int, default=None)
    ap.add_argument('--max-real', type=int, default=None)
    ap.add_argument('--total-steps-override', type=int, default=None)
    ap.add_argument('--replay-primary-ratio', type=float, default=0.8)
    ap.add_argument('--replay-ratio', type=float, default=0.2)
    ap.add_argument('--disable-train-validation', action='store_true')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--skip-train', action='store_true')
    ap.add_argument('--skip-generate', action='store_true')
    ap.add_argument('--skip-eval', action='store_true')
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, 'r', encoding='utf-8'))
    artifacts_dir = Path(cfg['paths']['artifacts_dir'])
    results_root = Path(args.out_root)
    prompts = cfg['prompts']['validation_prompts']
    instance_prompt = cfg['prompts']['instance_prompt']
    split_dir = ensure_split_phase_lists(artifacts_dir, args.split, cfg)
    real_list = artifacts_dir / 'splits' / f'{args.split}.txt'
    total_steps = args.total_steps_override or cfg['model']['max_train_steps']
    resolved_method = resolve_method(args.method, args.split)

    if args.smoke:
        phase_steps = {'phase2': 12, 'phase3': 8, 'phase3_replay': 8}
        num_per_prompt = args.num_per_prompt or 2
        max_real = args.max_real or 50
    else:
        phase_steps = {
            name: max(1, int(round(total_steps * ratio)))
            for name, _, _, ratio, _ in METHODS[resolved_method]
        }
        num_per_prompt = args.num_per_prompt or 50
        max_real = args.max_real or 500

    exp_name = f"{args.method}_{args.split}_seed{args.seed}"
    if args.smoke:
        exp_name = f"smoke_{exp_name}"
    exp_dir = results_root / exp_name
    if exp_dir.exists() and not args.resume:
        shutil.rmtree(exp_dir, ignore_errors=True)
    exp_dir.mkdir(parents=True, exist_ok=True)

    phase_plan = []
    for phase_name, list_key, load_from, ratio, lr in METHODS[resolved_method]:
        planned_image_list = str(split_dir / f'{list_key}.txt')
        if phase_name == 'phase3_replay':
            planned_image_list = str(exp_dir / 'phase3_replay.txt')
        phase_plan.append({
            'phase_name': phase_name,
            'list_key': list_key,
            'load_from': load_from,
            'ratio': ratio,
            'lr': lr,
            'planned_steps': phase_steps[phase_name],
            'image_list': planned_image_list,
        })

    write_json(exp_dir / 'run_manifest.json', {
        'experiment_name': exp_name,
        'created_at_epoch': time.time(),
        'config_path': args.config,
        'requested_method': args.method,
        'resolved_method': resolved_method,
        'split': args.split,
        'seed': args.seed,
        'smoke': args.smoke,
        'out_root': str(results_root),
        'exp_dir': str(exp_dir),
        'real_list': str(real_list),
        'split_dir': str(split_dir),
        'base_model': cfg['model']['base_model'],
        'instance_prompt': instance_prompt,
        'validation_prompts': prompts,
        'num_per_prompt': num_per_prompt,
        'max_real': max_real,
        'total_steps': total_steps,
        'disable_train_validation': args.disable_train_validation,
        'resume': args.resume,
        'skip_train': args.skip_train,
        'skip_generate': args.skip_generate,
        'skip_eval': args.skip_eval,
        'replay_primary_ratio': args.replay_primary_ratio,
        'replay_ratio': args.replay_ratio,
        'phase_plan': phase_plan,
    })

    log_csv = exp_dir / 'phase_log.csv'
    log_exists = log_csv.exists()
    with open(log_csv, 'a' if log_exists else 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not log_exists:
            writer.writerow(['phase', 'steps', 'lr', 'image_list', 'elapsed_sec'])

        if not args.skip_train:
            for phase_name, list_key, load_from, ratio, lr in METHODS[resolved_method]:
                image_list = split_dir / f'{list_key}.txt'
                if phase_name == 'phase3_replay':
                    replay_list = split_dir / 'phase2.txt'
                    image_list = exp_dir / 'phase3_replay.txt'
                    if not image_list.exists() or not args.resume:
                        run([
                            sys.executable, 'kmc_lora/scripts/build_replay_list.py',
                            '--primary-list', str(split_dir / 'phase3.txt'),
                            '--replay-list', str(replay_list),
                            '--primary-ratio', str(args.replay_primary_ratio),
                            '--replay-ratio', str(args.replay_ratio),
                            '--seed', str(args.seed),
                            '--out-list', str(image_list),
                        ])

                steps = phase_steps[phase_name]
                out_dir = exp_dir / phase_name
                final_adapter = out_dir / 'final' / 'adapter_model.safetensors'
                if args.resume and final_adapter.exists():
                    continue

                lora_path = exp_dir / load_from / 'final' if load_from else None
                t0 = time.time()
                train_phase(
                    base_model=cfg['model']['base_model'],
                    instance_prompt=instance_prompt,
                    validation_prompts=[] if args.disable_train_validation else prompts,
                    resolution=cfg['model']['resolution'],
                    batch_size=cfg['model']['train_batch_size'],
                    grad_accum=cfg['model']['gradient_accumulation_steps'],
                    save_steps=max(1, min(cfg['model']['save_steps'], steps)),
                    lora_rank=cfg['model']['lora_rank'],
                    seed=args.seed,
                    image_list=str(image_list),
                    out_dir=out_dir,
                    steps=steps,
                    lr=lr,
                    lora_path=lora_path,
                )
                elapsed = time.time() - t0
                writer.writerow([phase_name, steps, lr, str(image_list), round(elapsed, 1)])

    final_phase = METHODS[resolved_method][-1][0]
    final_lora = exp_dir / final_phase / 'final'
    gen_dir = exp_dir / 'generated'
    expected_gen = num_per_prompt * len(prompts)
    if not args.skip_generate:
        current_gen = len(list(gen_dir.glob('*.png'))) if gen_dir.exists() else 0
        if not (args.resume and current_gen >= expected_gen):
            generate_images(cfg['model']['base_model'], final_lora, prompts, gen_dir, num_per_prompt, args.seed)

    eval_csv = exp_dir / 'eval.csv'
    if not args.skip_eval:
        if not (args.resume and eval_csv.exists()):
            evaluate(real_list, gen_dir, exp_name, max_real, expected_gen, eval_csv)

    generated_count = len(list(gen_dir.glob('*.png'))) if gen_dir.exists() else 0
    write_json(exp_dir / 'completion_summary.json', {
        'experiment_name': exp_name,
        'completed_at_epoch': time.time(),
        'requested_method': args.method,
        'resolved_method': resolved_method,
        'split': args.split,
        'seed': args.seed,
        'final_phase': final_phase,
        'final_lora_dir': str(final_lora),
        'generated_dir': str(gen_dir),
        'generated_count': generated_count,
        'expected_generated_count': expected_gen,
        'eval_csv': str(eval_csv),
        'real_list': str(real_list),
        'phase_log_csv': str(log_csv),
        'run_manifest': str(exp_dir / 'run_manifest.json'),
    })

    print(f'completed={exp_name}')
    print(f'out_dir={exp_dir}')
    print(f'resolved_method={resolved_method}')


if __name__ == '__main__':
    main()
