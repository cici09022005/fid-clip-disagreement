"""
KMC-LoRA 实验进度监控脚本
每隔 30 分钟自动记录一次实验进度，输出到 progress_log.csv 和终端。
可也手动运行一次查看当前状态。

用法:
  python kmc_lora/scripts/monitor_progress.py                # 单次查看
  python kmc_lora/scripts/monitor_progress.py --loop          # 每30分钟自动记录
  python kmc_lora/scripts/monitor_progress.py --loop --interval 10  # 每10分钟
"""
import argparse
import csv
import os
import re
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path

RESULTS_DIR = Path('kmc_lora/results')
ARTIFACTS_DIR = Path('kmc_lora/artifacts')
PROGRESS_LOG = RESULTS_DIR / 'progress_log.csv'

# 所有实验列表及其结构
KMC_SPLITS = ['D-High', 'D-Medium', 'D-Low', 'D-Sub-50', 'D-Sub-25']
ABLATION_EXPS = {
    'Ablation_NoPhase1': ['phase2', 'phase3'],
    'Ablation_NoPhase2': ['phase1', 'phase3'],
    'Ablation_NoPhase3': ['phase1', 'phase2'],
    'Ablation_Phase1Only': ['phase1'],
    'Ablation_Phase3Only': ['phase3'],
}
OTHER_EXPS = ['Random_D-High', 'Random_D-Medium', 'Random_D-Low',
              'Random_D-Sub-50', 'Random_D-Sub-25',
              'Quality_Filter', 'Anti_Curriculum']

PHASE_STEPS = {'phase1': 480, 'phase2': 420, 'phase3': 300}


def get_gpu_info():
    """获取 GPU 状态。"""
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,'
             'temperature.gpu,power.draw', '--format=csv,noheader,nounits'],
            text=True, timeout=10
        ).strip()
        parts = [x.strip() for x in out.split(',')]
        return {
            'gpu_util': f'{parts[0]}%',
            'mem_used': f'{parts[1]}MB',
            'mem_total': f'{parts[2]}MB',
            'temp': f'{parts[3]}°C',
            'power': f'{parts[4]}W',
        }
    except Exception:
        return {'gpu_util': 'N/A', 'mem_used': 'N/A', 'mem_total': 'N/A',
                'temp': 'N/A', 'power': 'N/A'}


def parse_phase_status(phase_dir):
    """分析某个 phase 目录的训练状态。"""
    phase_dir = Path(phase_dir)
    result = {
        'status': 'not_started',
        'steps_done': 0,
        'steps_total': 0,
        'progress_pct': 0.0,
        'loss': None,
        'speed_it_s': None,
        'eta_sec': None,
    }

    stdout_log = phase_dir / 'stdout.log'
    loss_log = phase_dir / 'loss_log.csv'

    if not phase_dir.exists():
        return result

    # 检查是否已完成
    final_safetensors = phase_dir / 'final' / 'adapter_model.safetensors'
    final_bin = phase_dir / 'final' / 'adapter_model.bin'
    if final_safetensors.exists() or final_bin.exists():
        result['status'] = 'completed'
        result['progress_pct'] = 100.0
        # 从 stdout 读取完成信息
        if stdout_log.exists():
            lines = stdout_log.read_text(encoding='utf-8', errors='ignore').splitlines()
            for line in reversed(lines):
                if '[DONE]' in line:
                    m = re.search(r'(\d+) steps in ([\d.]+) min', line)
                    if m:
                        result['steps_done'] = int(m.group(1))
                        result['steps_total'] = result['steps_done']
                    break
        return result

    # 从 loss_log 获取步数
    if loss_log.exists():
        try:
            lines = loss_log.read_text(encoding='utf-8', errors='ignore').splitlines()
            if len(lines) > 1:
                result['steps_done'] = len(lines) - 1  # 减去 header
                last = lines[-1].split(',')
                if len(last) >= 3:
                    result['loss'] = float(last[2])
        except Exception:
            pass

    # 从 stdout 获取进度条信息
    if stdout_log.exists():
        try:
            content = stdout_log.read_text(encoding='utf-8', errors='ignore')
            # 匹配最后一个 tqdm 进度行: XX%|...| 123/456 [01:23<00:45, 3.20it/s
            matches = list(re.finditer(
                r'(\d+)%\|[^|]*\|\s*(\d+)/(\d+)\s*\[[\d:]+<([\d:]+),\s*([\d.]+)it/s',
                content
            ))
            if matches:
                last_match = matches[-1]
                result['steps_done'] = int(last_match.group(2))
                result['steps_total'] = int(last_match.group(3))
                result['progress_pct'] = float(last_match.group(1))
                result['speed_it_s'] = float(last_match.group(5))
                # Parse ETA
                eta_str = last_match.group(4)
                parts = eta_str.split(':')
                if len(parts) == 2:
                    result['eta_sec'] = int(parts[0]) * 60 + int(parts[1])
                elif len(parts) == 3:
                    result['eta_sec'] = int(parts[0])*3600 + int(parts[1])*60 + int(parts[2])
                result['status'] = 'training'
        except Exception:
            pass

    if result['status'] == 'not_started' and result['steps_done'] > 0:
        result['status'] = 'training'

    return result


def get_experiment_status(exp_name, phases):
    """获取一个实验的完整状态。"""
    exp_dir = RESULTS_DIR / exp_name
    if not exp_dir.exists():
        return 'not_started', None, 0.0

    completed_phases = 0
    current_phase = None
    current_progress = None

    for phase in phases:
        phase_dir = exp_dir / phase
        status = parse_phase_status(phase_dir)

        if status['status'] == 'completed':
            completed_phases += 1
        elif status['status'] == 'training':
            current_phase = phase
            current_progress = status
            break
        else:
            break

    if completed_phases == len(phases):
        # 检查是否有 generated/evaluated
        has_gen = (exp_dir / 'generated_fixed').exists()
        if has_gen:
            return 'evaluated', None, 100.0
        return 'trained', None, 100.0

    if current_phase:
        total_steps_all = sum(PHASE_STEPS[p] for p in phases)
        done_steps = sum(PHASE_STEPS[phases[i]] for i in range(completed_phases))
        done_steps += current_progress['steps_done']
        overall_pct = done_steps / total_steps_all * 100
        return 'training', current_progress, overall_pct

    if completed_phases > 0:
        total_steps_all = sum(PHASE_STEPS[p] for p in phases)
        done_steps = sum(PHASE_STEPS[phases[i]] for i in range(completed_phases))
        return 'partial', None, done_steps / total_steps_all * 100

    return 'not_started', None, 0.0


def format_time(seconds):
    """格式化秒数为 H:MM:SS。"""
    if seconds is None or seconds < 0:
        return 'N/A'
    h = int(seconds) // 3600
    m = (int(seconds) % 3600) // 60
    s = int(seconds) % 60
    if h > 0:
        return f'{h}:{m:02d}:{s:02d}'
    return f'{m}:{s:02d}'


def estimate_remaining():
    """估算剩余总时间。"""
    # 统计各实验状态
    experiments = []

    for split in KMC_SPLITS:
        name = f'KMC_{split}'
        phases = ['phase1', 'phase2', 'phase3']
        status, progress, pct = get_experiment_status(name, phases)
        experiments.append((name, phases, status, progress, pct))

    for abl_name, phases in ABLATION_EXPS.items():
        status, progress, pct = get_experiment_status(abl_name, phases)
        experiments.append((abl_name, phases, status, progress, pct))

    total_remaining_steps = 0
    remaining_phases = 0
    avg_speed = 3.2  # default it/s

    for name, phases, status, progress, pct in experiments:
        if status in ('trained', 'evaluated'):
            continue
        total_steps = sum(PHASE_STEPS[p] for p in phases)
        done_steps = int(total_steps * pct / 100)
        total_remaining_steps += (total_steps - done_steps)

        # Count remaining phases for overhead
        exp_dir = RESULTS_DIR / name
        for phase in phases:
            phase_dir = exp_dir / phase
            final_st = phase_dir / 'final' / 'adapter_model.safetensors'
            final_bin = phase_dir / 'final' / 'adapter_model.bin'
            if not final_st.exists() and not final_bin.exists():
                remaining_phases += 1

        if progress and progress.get('speed_it_s'):
            avg_speed = progress['speed_it_s']

    # Training time estimate
    train_sec = total_remaining_steps / avg_speed if avg_speed > 0 else 0
    overhead_sec = remaining_phases * 90  # ~1.5 min overhead per phase
    train_total = train_sec + overhead_sec

    # Evaluation time estimate (not started experiments count)
    eval_exps = 0
    for name, phases, status, progress, pct in experiments:
        if status != 'evaluated':
            eval_exps += 1
    # Also count already-trained experiments that need re-eval
    for other in OTHER_EXPS:
        exp_dir = RESULTS_DIR / other
        if exp_dir.exists() and not (exp_dir / 'generated_fixed').exists():
            eval_exps += 1

    # ~5 min per experiment for gen+eval
    eval_sec = eval_exps * 300

    return train_total, eval_sec, total_remaining_steps, remaining_phases


def print_status():
    """打印当前完整状态。"""
    now = datetime.now()
    gpu = get_gpu_info()

    print(f'\n{"="*70}')
    print(f'  KMC-LoRA 实验进度报告  |  {now.strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'{"="*70}')
    print(f'  GPU: {gpu["gpu_util"]} | 显存: {gpu["mem_used"]}/{gpu["mem_total"]} | '
          f'温度: {gpu["temp"]} | 功耗: {gpu["power"]}')
    print(f'{"="*70}')

    all_rows = []

    # KMC experiments
    print(f'\n  ── KMC 实验 (5 splits × 3 phases) ──')
    for split in KMC_SPLITS:
        name = f'KMC_{split}'
        status, progress, pct = get_experiment_status(
            name, ['phase1', 'phase2', 'phase3'])
        icon = {'not_started': '⬜', 'training': '🔄', 'partial': '⏸️',
                'trained': '✅', 'evaluated': '🏁'}.get(status, '❓')
        extra = ''
        if progress:
            extra = f' | step {progress["steps_done"]}/{progress["steps_total"]}'
            if progress.get('loss'):
                extra += f' loss={progress["loss"]:.4f}'
            if progress.get('speed_it_s'):
                extra += f' {progress["speed_it_s"]:.1f}it/s'
        print(f'    {icon} {name:20s} {pct:5.1f}%  [{status:12s}]{extra}')
        all_rows.append([now.isoformat(), name, status, f'{pct:.1f}',
                         progress.get('loss', '') if progress else '',
                         progress.get('speed_it_s', '') if progress else ''])

    # Ablation experiments
    print(f'\n  ── Ablation 实验 (5 组) ──')
    for abl_name, phases in ABLATION_EXPS.items():
        status, progress, pct = get_experiment_status(abl_name, phases)
        icon = {'not_started': '⬜', 'training': '🔄', 'partial': '⏸️',
                'trained': '✅', 'evaluated': '🏁'}.get(status, '❓')
        extra = ''
        if progress:
            extra = f' | step {progress["steps_done"]}/{progress["steps_total"]}'
            if progress.get('loss'):
                extra += f' loss={progress["loss"]:.4f}'
        print(f'    {icon} {abl_name:20s} {pct:5.1f}%  [{status:12s}]{extra}')
        all_rows.append([now.isoformat(), abl_name, status, f'{pct:.1f}',
                         progress.get('loss', '') if progress else '',
                         progress.get('speed_it_s', '') if progress else ''])

    # Other (should already be done)
    print(f'\n  ── 其他实验 (已有结果) ──')
    for other in OTHER_EXPS:
        exp_dir = RESULTS_DIR / other
        if exp_dir.exists():
            has_gen = (exp_dir / 'generated_fixed').exists()
            icon = '🏁' if has_gen else '✅'
            status = 'evaluated' if has_gen else 'trained'
        else:
            icon = '⬜'
            status = 'missing'
        print(f'    {icon} {other:20s} [{status}]')
        all_rows.append([now.isoformat(), other, status, '100' if status != 'missing' else '0', '', ''])

    # Time estimate
    train_sec, eval_sec, remain_steps, remain_phases = estimate_remaining()
    total_sec = train_sec + eval_sec

    print(f'\n  ── 时间估算 ──')
    print(f'    训练剩余: {format_time(train_sec)} ({remain_steps} steps, {remain_phases} phases)')
    print(f'    评估剩余: {format_time(eval_sec)}')
    print(f'    总计剩余: {format_time(total_sec)}')
    if total_sec > 0:
        eta = now + timedelta(seconds=total_sec)
        print(f'    预计完成: {eta.strftime("%Y-%m-%d %H:%M")}')
    print(f'{"="*70}\n')

    # Write to progress_log.csv
    write_header = not PROGRESS_LOG.exists()
    with open(PROGRESS_LOG, 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(['timestamp', 'experiment', 'status', 'progress_pct',
                        'loss', 'speed_it_s', 'gpu_util', 'mem_used', 'temp',
                        'power', 'remaining_train_sec', 'remaining_eval_sec'])
        for row in all_rows:
            w.writerow(row + [gpu['gpu_util'], gpu['mem_used'], gpu['temp'],
                              gpu['power'], f'{train_sec:.0f}', f'{eval_sec:.0f}'])

    return total_sec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--loop', action='store_true',
                    help='持续每隔 interval 分钟记录一次')
    ap.add_argument('--interval', type=int, default=30,
                    help='记录间隔(分钟), 默认 30')
    args = ap.parse_args()

    if args.loop:
        print(f'[MONITOR] 每 {args.interval} 分钟记录一次进度 → {PROGRESS_LOG}')
        print(f'[MONITOR] 按 Ctrl+C 停止')
        while True:
            remaining = print_status()
            if remaining <= 0:
                print('[MONITOR] 所有实验已完成！')
                break
            time.sleep(args.interval * 60)
    else:
        print_status()


if __name__ == '__main__':
    main()
