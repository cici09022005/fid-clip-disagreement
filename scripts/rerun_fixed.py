"""
KMC-LoRA 修复重跑脚本
修复问题：
  1. 旧 curriculum.csv 用 phase 单列(互斥分区) → 改为 phase1/phase2/phase3 布尔列(累积)
  2. KMC 实验全部用了全局 phase 文件 → 改为 split-specific phase 文件
  3. Ablation 实验用了互斥 phase → 用修正后的累积 phase
  4. FID 生成样本数 30 → 增加到 200

只重跑需要修复的实验（KMC × 5 + Ablation × 6），
保留有效的 Random/Quality_Filter/Anti_Curriculum 结果。
"""
import argparse, csv, gc, json, os, shutil, subprocess, sys, time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def rebuild_curriculum(artifacts_dir, cfg):
    """Regenerate curriculum.csv with boolean phase1/phase2/phase3 columns."""
    cq_csv = Path(artifacts_dir) / 'cluster_quality.csv'
    curriculum_csv = Path(artifacts_dir) / 'curriculum.csv'

    if not cq_csv.exists():
        raise FileNotFoundError(f"cluster_quality.csv not found: {cq_csv}")

    # Backup old curriculum
    if curriculum_csv.exists():
        backup = curriculum_csv.with_suffix('.csv.bak_old_exclusive')
        if not backup.exists():
            shutil.copy2(curriculum_csv, backup)
            print(f"[BACKUP] Old curriculum → {backup.name}")

    # Rebuild using build_curriculum.py
    phases_dir = Path(artifacts_dir) / 'phases'
    cmd = [
        sys.executable, 'kmc_lora/scripts/build_curriculum.py',
        '--in-csv', str(cq_csv),
        '--out-csv', str(curriculum_csv),
        '--phase1', str(cfg['curriculum']['phase1_ratio']),
        '--phase2', str(cfg['curriculum']['phase2_ratio']),
        '--phase3', str(cfg['curriculum']['phase3_ratio']),
        '--w-quality', str(cfg['curriculum']['quality_weight']),
        '--w-typicality', str(cfg['curriculum']['typicality_weight']),
        '--w-heterogeneity', str(cfg['curriculum']['heterogeneity_weight']),
        '--out-dir', str(phases_dir),
    ]
    print(f"[REBUILD] curriculum.csv ...")
    rc = subprocess.call(cmd)
    if rc != 0:
        raise RuntimeError("build_curriculum.py failed")

    # Verify columns
    df = pd.read_csv(curriculum_csv)
    for col in ['phase1', 'phase2', 'phase3']:
        if col not in df.columns:
            raise RuntimeError(f"Missing column {col} in rebuilt curriculum")
    print(f"  Phase1: {df['phase1'].sum()}, Phase2: {df['phase2'].sum()}, "
          f"Phase3: {df['phase3'].sum()}, Total: {len(df)}")


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
        raise ValueError(f"No overlap: {split_file}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    phase_files = {}
    for phase_name in ['phase1', 'phase2', 'phase3']:
        phase_path = out_dir / f'{phase_name}.txt'
        phase_data = split_df.loc[split_df[phase_name].astype(bool), 'path']
        phase_data.to_csv(phase_path, index=False, header=False)
        phase_files[phase_name] = str(phase_path)
        print(f"    {phase_name}: {len(phase_data)} images → {phase_path.name}")
    return phase_files


def clean_experiment(results_dir, exp_name):
    """Remove old experiment results to force rerun."""
    exp_dir = Path(results_dir) / exp_name
    if exp_dir.exists():
        backup_name = f"{exp_name}_bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        backup_dir = Path(results_dir) / '_old_results' / backup_name
        backup_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(exp_dir), str(backup_dir))
        print(f"  [MOVE] {exp_name} → _old_results/{backup_name}")


def train_phase(name, base_model, image_list, out_dir, steps, lr, seed,
                resolution, batch_size, grad_accum, save_steps,
                lora_rank, validation_prompts, lora_path=None,
                max_gpu_mem_frac=0.8):
    """Run a single training phase."""
    # Skip if already done
    final_st = Path(out_dir) / 'final' / 'adapter_model.safetensors'
    final_bin = Path(out_dir) / 'final' / 'adapter_model.bin'
    if final_st.exists() or final_bin.exists():
        print(f"  [SKIP] {name}: already complete")
        return True

    cmd = [
        sys.executable, 'kmc_lora/scripts/train_lora.py',
        '--base-model', base_model,
        '--image-list', image_list,
        '--output-dir', out_dir,
        '--instance-prompt', 'a hand-drawn anime storyboard, clean line art',
        '--resolution', str(resolution),
        '--train-batch-size', str(batch_size),
        '--gradient-accumulation-steps', str(grad_accum),
        '--max-train-steps', str(steps),
        '--save-steps', str(save_steps),
        '--lr', str(lr),
        '--lora-rank', str(lora_rank),
        '--seed', str(seed),
        '--max-gpu-memory-fraction', str(max_gpu_mem_frac),
    ]
    if validation_prompts:
        cmd += ['--validation-prompts'] + validation_prompts
    if lora_path:
        cmd += ['--lora-path', lora_path]

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(out_dir) / 'stdout.log'

    print(f"\n{'='*60}")
    print(f"[TRAIN] {name} → {out_dir}")
    print(f"  steps={steps}  lr={lr}  images={image_list}")
    print(f"{'='*60}")

    t0 = time.time()
    with open(log_file, 'w', encoding='utf-8') as f:
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
    t1 = time.time()
    duration = (t1 - t0) / 60

    status = 'ok' if rc == 0 else 'FAIL'
    print(f"  [{status}] {name} ({duration:.1f} min)")
    return rc == 0


def generate_and_evaluate(base_model, lora_path, gen_dir, real_list,
                          prompts, num_per_prompt=50, exp_name=''):
    """Generate images and evaluate FID + CLIP."""
    gen_dir = Path(gen_dir)

    # Generate
    if not gen_dir.exists() or len(list(gen_dir.glob('*.png'))) < 10:
        cmd = [
            sys.executable, 'kmc_lora/scripts/generate_samples.py',
            '--base-model', base_model,
            '--lora-path', lora_path,
            '--prompts'] + prompts + [
            '--out-dir', str(gen_dir),
            '--num-per-prompt', str(num_per_prompt),
            '--seed', '42',
        ]
        print(f"  [GEN] {exp_name}: {num_per_prompt} × {len(prompts)} prompts ...")
        rc = subprocess.call(cmd)
        if rc != 0:
            print(f"  [FAIL] Generation failed for {exp_name}")
            return False
    else:
        print(f"  [SKIP] {exp_name}: images already exist")

    # Evaluate
    eval_csv = str(gen_dir.parent.parent / 'eval_all_fixed.csv')
    cmd = [
        sys.executable, 'kmc_lora/scripts/evaluate_fid.py',
        '--real-list', real_list,
        '--gen-dir', str(gen_dir),
        '--experiment-name', exp_name,
        '--max-real', '500',
        '--max-gen', '500',
    ]
    if prompts:
        cmd += ['--prompts'] + prompts
    cmd += ['--out-csv', eval_csv]

    print(f"  [EVAL] {exp_name} ...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        print(f"  [FAIL] Eval failed: {result.stderr[:300]}")
        return False
    for line in result.stdout.strip().split('\n'):
        if 'FID' in line or 'CLIP' in line:
            print(f"    {line}")
    return True


def main():
    ap = argparse.ArgumentParser(description='KMC-LoRA fixed rerun')
    ap.add_argument('--config', default='kmc_lora/configs/base.yaml')
    ap.add_argument('--base-model', default='models/sd15',
                    help='Local SD1.5 model path')
    ap.add_argument('--num-per-prompt', type=int, default=50,
                    help='Images per prompt for FID (default: 50 → 150 total)')
    ap.add_argument('--skip-train', action='store_true',
                    help='Skip training, only regenerate + evaluate')
    ap.add_argument('--skip-eval', action='store_true',
                    help='Skip evaluation, only train')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, 'r', encoding='utf-8'))
    artifacts_dir = Path(cfg['paths']['artifacts_dir'])
    results_dir = Path(cfg['paths']['results_dir'])
    curriculum_csv = artifacts_dir / 'curriculum.csv'
    splits_dir = artifacts_dir / 'splits'
    phases_dir = artifacts_dir / 'phases'

    total_steps = cfg['model']['max_train_steps']
    p1_steps = int(total_steps * cfg['curriculum']['phase1_ratio'])
    p2_steps = int(total_steps * cfg['curriculum']['phase2_ratio'])
    p3_steps = total_steps - p1_steps - p2_steps

    prompts = cfg['prompts']['validation_prompts']
    real_list = str(splits_dir / 'D-High.txt')

    train_args = dict(
        base_model=args.base_model,
        seed=cfg['experiment']['seed'],
        resolution=cfg['model']['resolution'],
        batch_size=cfg['model']['train_batch_size'],
        grad_accum=cfg['model']['gradient_accumulation_steps'],
        save_steps=cfg['model']['save_steps'],
        lora_rank=cfg['model']['lora_rank'],
        validation_prompts=prompts,
        max_gpu_mem_frac=cfg['model'].get('max_gpu_memory_fraction', 0.8),
    )

    print(f"{'='*60}")
    print(f"KMC-LoRA 修复重跑")
    print(f"  开始时间: {datetime.now().isoformat()}")
    print(f"  Base model: {args.base_model}")
    print(f"  Total steps: {total_steps} (P1={p1_steps}, P2={p2_steps}, P3={p3_steps})")
    print(f"  FID samples: {args.num_per_prompt} × {len(prompts)} = {args.num_per_prompt * len(prompts)}")
    print(f"{'='*60}\n")

    # ── Step 1: Rebuild curriculum ──
    print("[STEP 1] 重建 curriculum.csv (累积布尔列) ...")
    rebuild_curriculum(artifacts_dir, cfg)

    # ── Step 2: Build split-specific phase lists ──
    print("\n[STEP 2] 生成 split-specific phase 文件 ...")
    split_files = ['D-High.txt', 'D-Medium.txt', 'D-Low.txt',
                   'D-Sub-50.txt', 'D-Sub-25.txt']
    split_phases = {}
    for split in split_files:
        tag = split.replace('.txt', '')
        sp = str(splits_dir / split)
        out = artifacts_dir / 'split_phases' / tag
        print(f"  [{tag}]")
        split_phases[tag] = build_split_phase_lists(curriculum_csv, sp, out)

    # ── Step 3: Clean old KMC + Ablation results ──
    print("\n[STEP 3] 移除旧的 KMC 和 Ablation 结果 ...")
    exps_to_clean = []
    for split in split_files:
        tag = split.replace('.txt', '')
        exps_to_clean.append(f'KMC_{tag}')
    exps_to_clean += [
        'Ablation_NoPhase1', 'Ablation_NoPhase2', 'Ablation_NoPhase3',
        'Ablation_Phase1Only', 'Ablation_Phase3Only',
    ]
    for exp in exps_to_clean:
        clean_experiment(results_dir, exp)

    if args.skip_train:
        print("\n[SKIP] 跳过训练阶段")
    else:
        # ── Step 4: Run KMC experiments ──
        print(f"\n[STEP 4] 运行 KMC 三阶段训练 (5 splits × 3 phases) ...")
        exp_log = open(results_dir / 'experiment_log_fixed.csv', 'w',
                       newline='', encoding='utf-8')
        log_writer = csv.writer(exp_log)
        log_writer.writerow(['experiment', 'status', 'start_time', 'end_time',
                             'duration_min', 'image_list', 'steps', 'lr'])

        for split in split_files:
            tag = split.replace('.txt', '')
            sp = split_phases[tag]
            kmc_dir = str(results_dir / f'KMC_{tag}')

            for phase_name, phase_steps, phase_lr in [
                ('phase1', p1_steps, 5e-5),
                ('phase2', p2_steps, 1e-4),
                ('phase3', p3_steps, 2e-5),
            ]:
                exp_name = f'KMC_{tag}_{phase_name}'
                out_dir = str(Path(kmc_dir) / phase_name)
                img_list = sp[phase_name]

                # Previous phase LoRA path
                lora_path = None
                if phase_name == 'phase2':
                    lora_path = str(Path(kmc_dir) / 'phase1' / 'final')
                elif phase_name == 'phase3':
                    lora_path = str(Path(kmc_dir) / 'phase2' / 'final')

                t0 = time.time()
                ok = train_phase(
                    exp_name, image_list=img_list, out_dir=out_dir,
                    steps=phase_steps, lr=phase_lr, lora_path=lora_path,
                    **train_args
                )
                t1 = time.time()
                log_writer.writerow([
                    exp_name, 'ok' if ok else 'FAIL',
                    datetime.fromtimestamp(t0).isoformat(),
                    datetime.fromtimestamp(t1).isoformat(),
                    f'{(t1-t0)/60:.2f}', img_list, phase_steps, phase_lr
                ])
                exp_log.flush()

                if not ok:
                    print(f"  [ABORT] {tag} failed at {phase_name}")
                    break

        # ── Step 5: Run Ablation experiments (using corrected global phases) ──
        print(f"\n[STEP 5] 运行 Ablation 实验 (6 组) ...")
        phase1_list = str(phases_dir / 'phase1.txt')
        phase2_list = str(phases_dir / 'phase2.txt')
        phase3_list = str(phases_dir / 'phase3.txt')

        # No Phase1: P2 → P3
        np1_dir = str(results_dir / 'Ablation_NoPhase1')
        train_phase('Abl_NoP1_P2', image_list=phase2_list,
                    out_dir=f'{np1_dir}/phase2', steps=p2_steps, lr=1e-4,
                    **train_args)
        train_phase('Abl_NoP1_P3', image_list=phase3_list,
                    out_dir=f'{np1_dir}/phase3', steps=p3_steps, lr=2e-5,
                    lora_path=f'{np1_dir}/phase2/final', **train_args)

        # No Phase2: P1 → P3
        np2_dir = str(results_dir / 'Ablation_NoPhase2')
        train_phase('Abl_NoP2_P1', image_list=phase1_list,
                    out_dir=f'{np2_dir}/phase1', steps=p1_steps, lr=5e-5,
                    **train_args)
        train_phase('Abl_NoP2_P3', image_list=phase3_list,
                    out_dir=f'{np2_dir}/phase3', steps=p3_steps, lr=2e-5,
                    lora_path=f'{np2_dir}/phase1/final', **train_args)

        # No Phase3: P1 → P2
        np3_dir = str(results_dir / 'Ablation_NoPhase3')
        train_phase('Abl_NoP3_P1', image_list=phase1_list,
                    out_dir=f'{np3_dir}/phase1', steps=p1_steps, lr=5e-5,
                    **train_args)
        train_phase('Abl_NoP3_P2', image_list=phase2_list,
                    out_dir=f'{np3_dir}/phase2', steps=p2_steps, lr=1e-4,
                    lora_path=f'{np3_dir}/phase1/final', **train_args)

        # Phase1 Only
        train_phase('Abl_P1Only', image_list=phase1_list,
                    out_dir=str(results_dir / 'Ablation_Phase1Only' / 'phase1'),
                    steps=p1_steps, lr=5e-5, **train_args)

        # Phase3 Only
        train_phase('Abl_P3Only', image_list=phase3_list,
                    out_dir=str(results_dir / 'Ablation_Phase3Only' / 'phase3'),
                    steps=p3_steps, lr=2e-5, **train_args)

        exp_log.close()

    if args.skip_eval:
        print("\n[SKIP] 跳过评估阶段")
    else:
        # ── Step 6: Generate + Evaluate ──
        print(f"\n[STEP 6] 生成样本 + FID/CLIP 评估 ...")

        all_exps = [f'KMC_{s.replace(".txt","")}' for s in split_files]
        all_exps += ['Ablation_NoPhase1', 'Ablation_NoPhase2',
                     'Ablation_NoPhase3', 'Ablation_Phase1Only',
                     'Ablation_Phase3Only']
        # Also re-evaluate Random baselines with more samples
        all_exps += [f'Random_{s.replace(".txt","")}' for s in split_files]
        all_exps += ['Quality_Filter', 'Anti_Curriculum']

        FINAL_PHASE = {
            'Ablation_NoPhase1': 'phase3',
            'Ablation_NoPhase2': 'phase3',
            'Ablation_NoPhase3': 'phase2',
            'Ablation_Phase1Only': 'phase1',
            'Ablation_Phase3Only': 'phase3',
        }

        for exp in all_exps:
            exp_dir = results_dir / exp
            if exp.startswith('KMC_'):
                lora = str(exp_dir / 'phase3' / 'final')
            elif exp in FINAL_PHASE:
                lora = str(exp_dir / FINAL_PHASE[exp] / 'final')
            else:
                lora = str(exp_dir / 'final')

            if not (Path(lora) / 'adapter_model.safetensors').exists() and \
               not (Path(lora) / 'adapter_model.bin').exists():
                print(f"  [SKIP] {exp}: no final LoRA at {lora}")
                continue

            gen_dir = str(exp_dir / 'generated_fixed')
            generate_and_evaluate(
                args.base_model, lora, gen_dir, real_list,
                prompts, num_per_prompt=args.num_per_prompt, exp_name=exp
            )
            gc.collect()

    print(f"\n{'='*60}")
    print(f"全部完成: {datetime.now().isoformat()}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
