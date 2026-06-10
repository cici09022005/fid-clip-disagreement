"""
KMC-LoRA 完整重跑 V2
──────────────────────────────────────────────
修复的关键 Bug:
 1. PEFT LoRA key 格式不匹配 — phase2/3 训练时未能继承上一阶段的权重
 2. generate_samples.py 无法加载 PEFT 格式 LoRA — 生成图全部相同
 3. NSFW safety checker 导致黑图

流程:
 Step 1: 清理旧 KMC/Ablation 结果
 Step 2: 重新训练 5×KMC (3-phase) + 5×Ablation
 Step 3: 重新生成图片 (17 experiments × 150 images)
 Step 4: FID 评估 (pytorch-fid)
"""
import argparse, csv, gc, json, os, shutil, subprocess, sys, time
from datetime import datetime
from pathlib import Path
import yaml


def clean_experiment(results_dir, exp_name):
    """Remove training + generated results for an experiment."""
    exp_dir = results_dir / exp_name
    if exp_dir.exists():
        shutil.rmtree(exp_dir, ignore_errors=True)
        print(f"    Cleaned {exp_name}")


def train_phase(exp_name, image_list, out_dir, steps, lr,
                base_model, seed, resolution, batch_size, grad_accum,
                save_steps, lora_rank, instance_prompt, validation_prompts,
                max_gpu_mem_frac=0.8, lora_path=None):
    """Run a single training phase via train_lora.py subprocess."""
    cmd = [
        sys.executable, 'kmc_lora/scripts/train_lora.py',
        '--base-model', base_model,
        '--image-list', image_list,
        '--output-dir', out_dir,
        '--instance-prompt', instance_prompt,
        '--max-train-steps', str(steps),
        '--lr', str(lr),
        '--seed', str(seed),
        '--resolution', str(resolution),
        '--train-batch-size', str(batch_size),
        '--gradient-accumulation-steps', str(grad_accum),
        '--save-steps', str(save_steps),
        '--lora-rank', str(lora_rank),
    ]
    if validation_prompts:
        cmd += ['--validation-prompts'] + validation_prompts
    if lora_path:
        cmd += ['--lora-path', lora_path]

    print(f"    [{exp_name}] steps={steps} lr={lr} images={image_list}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t0

    if result.returncode != 0:
        print(f"    [FAIL] {exp_name} ({dt:.0f}s)")
        # Save error log
        err_dir = Path(out_dir)
        err_dir.mkdir(parents=True, exist_ok=True)
        (err_dir / 'train_error.txt').write_text(
            result.stderr[-2000:] if result.stderr else 'no stderr', encoding='utf-8')
        return False

    print(f"    [OK] {exp_name} ({dt:.0f}s)")
    return True


def generate_images(base_model, lora_path, gen_dir, prompts,
                    num_per_prompt=50, seed=42):
    """Generate images using fixed generate_samples.py."""
    gen_dir = Path(gen_dir)
    # Always regenerate — delete old
    if gen_dir.exists():
        shutil.rmtree(gen_dir, ignore_errors=True)

    cmd = [
        sys.executable, 'kmc_lora/scripts/generate_samples.py',
        '--base-model', base_model,
        '--lora-path', lora_path,
        '--prompts'] + prompts + [
        '--out-dir', str(gen_dir),
        '--num-per-prompt', str(num_per_prompt),
        '--seed', str(seed),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"      [FAIL-GEN] {result.stderr[-500:]}")
        return False

    n = len(list(gen_dir.glob('*.png')))
    print(f"      Generated {n} images")
    return n > 0


def main():
    ap = argparse.ArgumentParser(description='KMC-LoRA V2 完整重跑')
    ap.add_argument('--config', default='kmc_lora/configs/base.yaml')
    ap.add_argument('--base-model', default='models/sd15')
    ap.add_argument('--num-per-prompt', type=int, default=50)
    ap.add_argument('--skip-train', action='store_true')
    ap.add_argument('--skip-gen', action='store_true')
    ap.add_argument('--skip-eval', action='store_true')
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, 'r', encoding='utf-8'))
    artifacts_dir = Path(cfg['paths']['artifacts_dir'])
    results_dir = Path(cfg['paths']['results_dir'])
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
        instance_prompt=cfg['prompts']['instance_prompt'],
        validation_prompts=prompts,
        max_gpu_mem_frac=cfg['model'].get('max_gpu_memory_fraction', 0.8),
    )

    split_files = ['D-High.txt', 'D-Medium.txt', 'D-Low.txt',
                   'D-Sub-50.txt', 'D-Sub-25.txt']

    # Log file
    log_path = results_dir / 'rerun_v2_log.csv'

    print('=' * 60)
    print('KMC-LoRA V2 完整重跑 (修复 LoRA key + 生成 + NSFW)')
    print(f'  Time: {datetime.now().isoformat()}')
    print(f'  Steps: {total_steps} (P1={p1_steps}, P2={p2_steps}, P3={p3_steps})')
    print(f'  Images/prompt: {args.num_per_prompt} × {len(prompts)} = '
          f'{args.num_per_prompt * len(prompts)}')
    print('=' * 60)

    log_rows = []

    # ════════════════════════════════════════════════════
    #  TRAINING
    # ════════════════════════════════════════════════════
    if not args.skip_train:
        # Read split-specific phase lists
        split_phase_dir = artifacts_dir / 'split_phases'

        # ── Clean old KMC + Ablation results ──
        print('\n[STEP 1] 清理旧 KMC/Ablation 结果 ...')
        for split in split_files:
            tag = split.replace('.txt', '')
            clean_experiment(results_dir, f'KMC_{tag}')
        for abl in ['Ablation_NoPhase1', 'Ablation_NoPhase2',
                     'Ablation_NoPhase3', 'Ablation_Phase1Only',
                     'Ablation_Phase3Only']:
            clean_experiment(results_dir, abl)

        # ── KMC 3-phase training ──
        print(f'\n[STEP 2] 训练 KMC 3 阶段 (5 splits × 3 phases = 15 runs) ...')
        for split in split_files:
            tag = split.replace('.txt', '')
            kmc_dir = results_dir / f'KMC_{tag}'

            phase_lists = {
                'phase1': str(split_phase_dir / tag / 'phase1.txt'),
                'phase2': str(split_phase_dir / tag / 'phase2.txt'),
                'phase3': str(split_phase_dir / tag / 'phase3.txt'),
            }

            for phase_name, phase_steps, phase_lr in [
                ('phase1', p1_steps, 5e-5),
                ('phase2', p2_steps, 1e-4),
                ('phase3', p3_steps, 2e-5),
            ]:
                exp_name = f'KMC_{tag}_{phase_name}'
                out_dir = str(kmc_dir / phase_name)
                img_list = phase_lists[phase_name]

                # Previous phase LoRA path for continuation
                lora_path = None
                if phase_name == 'phase2':
                    lora_path = str(kmc_dir / 'phase1' / 'final')
                elif phase_name == 'phase3':
                    lora_path = str(kmc_dir / 'phase2' / 'final')

                t0 = time.time()
                ok = train_phase(
                    exp_name, image_list=img_list, out_dir=out_dir,
                    steps=phase_steps, lr=phase_lr, lora_path=lora_path,
                    **train_args
                )
                dt = time.time() - t0

                log_rows.append({
                    'experiment': exp_name,
                    'status': 'ok' if ok else 'FAIL',
                    'time_sec': round(dt, 1),
                    'steps': phase_steps,
                    'lr': phase_lr,
                    'image_list': img_list,
                })

                if not ok:
                    print(f'    [ABORT] {tag} failed at {phase_name}')
                    break

        # ── Ablation experiments ──
        print(f'\n[STEP 3] 训练 Ablation 实验 ...')
        phase1_list = str(phases_dir / 'phase1.txt')
        phase2_list = str(phases_dir / 'phase2.txt')
        phase3_list = str(phases_dir / 'phase3.txt')

        ablations = [
            # (name, phases: [(img_list, out_subdir, steps, lr, lora_from)])
            ('Ablation_NoPhase1', [
                (phase2_list, 'phase2', p2_steps, 1e-4, None),
                (phase3_list, 'phase3', p3_steps, 2e-5, 'phase2'),
            ]),
            ('Ablation_NoPhase2', [
                (phase1_list, 'phase1', p1_steps, 5e-5, None),
                (phase3_list, 'phase3', p3_steps, 2e-5, 'phase1'),
            ]),
            ('Ablation_NoPhase3', [
                (phase1_list, 'phase1', p1_steps, 5e-5, None),
                (phase2_list, 'phase2', p2_steps, 1e-4, 'phase1'),
            ]),
            ('Ablation_Phase1Only', [
                (phase1_list, 'phase1', p1_steps, 5e-5, None),
            ]),
            ('Ablation_Phase3Only', [
                (phase3_list, 'phase3', p3_steps, 2e-5, None),
            ]),
        ]

        for abl_name, phases in ablations:
            abl_dir = results_dir / abl_name
            for img_list, phase_sub, steps, lr, lora_from in phases:
                exp_name = f'{abl_name}_{phase_sub}'
                out_dir = str(abl_dir / phase_sub)
                lora_path = str(abl_dir / lora_from / 'final') if lora_from else None

                t0 = time.time()
                ok = train_phase(
                    exp_name, image_list=img_list, out_dir=out_dir,
                    steps=steps, lr=lr, lora_path=lora_path,
                    **train_args
                )
                dt = time.time() - t0
                log_rows.append({
                    'experiment': exp_name,
                    'status': 'ok' if ok else 'FAIL',
                    'time_sec': round(dt, 1),
                    'steps': steps,
                    'lr': lr,
                    'image_list': img_list,
                })
                if not ok:
                    break

        # Save training log
        if log_rows:
            with open(log_path, 'w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
                w.writeheader()
                w.writerows(log_rows)
            print(f'\n  训练日志: {log_path}')
    else:
        print('\n[SKIP] 跳过训练')

    # ════════════════════════════════════════════════════
    #  IMAGE GENERATION
    # ════════════════════════════════════════════════════
    if not args.skip_gen:
        print(f'\n[STEP 4] 生成样本图片 ...')

        # All experiments to generate
        FINAL_PHASE = {
            'Ablation_NoPhase1': 'phase3',
            'Ablation_NoPhase2': 'phase3',
            'Ablation_NoPhase3': 'phase2',
            'Ablation_Phase1Only': 'phase1',
            'Ablation_Phase3Only': 'phase3',
        }

        all_exps = [f'KMC_{s.replace(".txt","")}' for s in split_files]
        all_exps += list(FINAL_PHASE.keys())
        all_exps += [f'Random_{s.replace(".txt","")}' for s in split_files]
        all_exps += ['Quality_Filter', 'Anti_Curriculum']

        for i, exp in enumerate(all_exps, 1):
            exp_dir = results_dir / exp

            if exp.startswith('KMC_'):
                lora = str(exp_dir / 'phase3' / 'final')
            elif exp in FINAL_PHASE:
                lora = str(exp_dir / FINAL_PHASE[exp] / 'final')
            else:
                lora = str(exp_dir / 'final')

            # Check LoRA exists
            has_safetensors = (Path(lora) / 'adapter_model.safetensors').exists()
            has_bin = (Path(lora) / 'adapter_model.bin').exists()
            if not has_safetensors and not has_bin:
                print(f'  [{i}/{len(all_exps)}] SKIP {exp}: no LoRA')
                continue

            gen_dir = str(exp_dir / 'generated_v2')
            print(f'  [{i}/{len(all_exps)}] {exp} ...', end=' ', flush=True)

            t0 = time.time()
            ok = generate_images(
                args.base_model, lora, gen_dir, prompts,
                num_per_prompt=args.num_per_prompt, seed=42
            )
            dt = time.time() - t0
            if ok:
                print(f'({dt:.0f}s)')
            else:
                print(f'FAIL ({dt:.0f}s)')

            gc.collect()
    else:
        print('\n[SKIP] 跳过图片生成')

    # ════════════════════════════════════════════════════
    #  FID EVALUATION
    # ════════════════════════════════════════════════════
    if not args.skip_eval:
        print(f'\n[STEP 5] FID 评估 (pytorch-fid) ...')
        import torch
        import numpy as np
        from pytorch_fid.fid_score import calculate_fid_given_paths

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Prepare real images (resized to 512x512)
        real_paths = [p.strip() for p in
                      open(real_list, 'r', encoding='utf-8').readlines()
                      if p.strip()]
        real_dir = results_dir / '_real_for_fid'
        if not real_dir.exists() or len(list(real_dir.glob('*.png'))) < 100:
            real_dir.mkdir(parents=True, exist_ok=True)
            from PIL import Image
            for i, p in enumerate(real_paths[:500]):
                src = Path(p)
                if src.exists():
                    dst = real_dir / f'{i:04d}.png'
                    if not dst.exists():
                        img = Image.open(src).convert('RGB').resize(
                            (512, 512), Image.LANCZOS)
                        img.save(dst)
            print(f'  Real images: {len(list(real_dir.glob("*.png")))}')

        # Evaluate each experiment
        FINAL_PHASE = {
            'Ablation_NoPhase1': 'phase3',
            'Ablation_NoPhase2': 'phase3',
            'Ablation_NoPhase3': 'phase2',
            'Ablation_Phase1Only': 'phase1',
            'Ablation_Phase3Only': 'phase3',
        }

        all_exps = [f'KMC_{s.replace(".txt","")}' for s in split_files]
        all_exps += list(FINAL_PHASE.keys())
        all_exps += [f'Random_{s.replace(".txt","")}' for s in split_files]
        all_exps += ['Quality_Filter', 'Anti_Curriculum']

        eval_results = []
        out_csv = results_dir / 'eval_v2.csv'

        for i, exp in enumerate(all_exps, 1):
            gen_dir = results_dir / exp / 'generated_v2'
            if not gen_dir.exists():
                # Fall back to generated_fixed for Random/Quality/Anti
                gen_dir = results_dir / exp / 'generated_fixed'
            if not gen_dir.exists() or len(list(gen_dir.glob('*.png'))) < 10:
                print(f'  [{i}/{len(all_exps)}] SKIP {exp}')
                continue

            n_gen = len(list(gen_dir.glob('*.png')))
            print(f'  [{i}/{len(all_exps)}] {exp} ({n_gen} gen) ...', end=' ',
                  flush=True)

            t0 = time.time()
            try:
                fid = calculate_fid_given_paths(
                    [str(real_dir), str(gen_dir)],
                    batch_size=32, device=device, dims=2048, num_workers=0)
                dt = time.time() - t0
                print(f'FID = {fid:.2f} ({dt:.1f}s)')
            except Exception as e:
                dt = time.time() - t0
                fid = -1
                print(f'FAIL ({dt:.1f}s): {e}')

            eval_results.append({
                'experiment': exp,
                'fid': round(fid, 4),
                'num_gen': n_gen,
                'gen_dir': str(gen_dir),
            })

            gc.collect()
            torch.cuda.empty_cache()

        # Write CSV
        if eval_results:
            with open(out_csv, 'w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=list(eval_results[0].keys()))
                w.writeheader()
                w.writerows(eval_results)
            print(f'\n  结果: {out_csv}')

            # Summary
            print(f'\n{"Experiment":<25s} {"FID":>10s}')
            print('-' * 37)
            for r in eval_results:
                print(f'{r["experiment"]:<25s} {r["fid"]:>10.2f}')
    else:
        print('\n[SKIP] 跳过评估')

    print(f'\n{"="*60}')
    print(f'全部完成: {datetime.now().isoformat()}')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
