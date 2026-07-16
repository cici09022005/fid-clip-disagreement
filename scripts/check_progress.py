"""
实验进度监控脚本 — 输出可读的进度报告到 TXT 文件。
用法:
    python kmc_lora/scripts/check_progress.py                 # 默认每30分钟刷新
    python kmc_lora/scripts/check_progress.py --results-dir kmc_lora/results
    python kmc_lora/scripts/check_progress.py --watch 60      # 每60秒自动刷新
    python kmc_lora/scripts/check_progress.py --once          # 只运行一次
"""
import argparse, json, os, time
from datetime import datetime
from pathlib import Path

try:
    import subprocess
    def gpu_info():
        r = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu',
             '--format=csv,noheader'],
            capture_output=True, text=True, timeout=5)
        return r.stdout.strip() if r.returncode == 0 else 'N/A'
except Exception:
    def gpu_info():
        return 'N/A'


def scan_experiments(results_dir):
    """扫描所有实验目录，返回进度信息列表。"""
    results = []
    results_dir = Path(results_dir)
    if not results_dir.exists():
        return results

    for d in sorted(results_dir.iterdir()):
        if not d.is_dir():
            continue
        # 跳过非实验目录
        if d.name.startswith('.') or d.name in ('__pycache__',):
            continue

        info = {
            'name': d.name,
            'status': '未开始',
            'steps': 0,
            'max_steps': '?',
            'final_loss': '',
            'duration': '',
            'gpu_peak': '',
            'has_checkpoints': [],
            'has_validation': False,
            'sub_phases': [],
        }

        # 检查是否有子阶段 (KMC 实验有 phase1/phase2/phase3)
        phase_dirs = [d / f'phase{i}' for i in (1, 2, 3)]
        has_phases = any(pd.exists() for pd in phase_dirs)

        if has_phases:
            for pd in phase_dirs:
                if not pd.exists():
                    continue
                phase_info = _scan_single(pd)
                phase_info['name'] = pd.name
                info['sub_phases'].append(phase_info)
            # 整体状态取决于子阶段
            statuses = [p['status'] for p in info['sub_phases']]
            if all(s == '已完成' for s in statuses):
                info['status'] = '已完成'
            elif any(s == '训练中' for s in statuses):
                info['status'] = '训练中'
            elif any(s == '已完成' for s in statuses):
                info['status'] = '部分完成'
            else:
                info['status'] = '未开始'
        else:
            single = _scan_single(d)
            info.update(single)

        results.append(info)

    return results


def _scan_single(d):
    """扫描单个实验/阶段目录。"""
    info = {
        'status': '未开始',
        'steps': 0,
        'max_steps': '?',
        'final_loss': '',
        'duration': '',
        'gpu_peak': '',
        'has_checkpoints': [],
        'has_validation': False,
    }

    # 检查 final adapter
    final_st = d / 'final' / 'adapter_model.safetensors'
    final_bin = d / 'final' / 'adapter_model.bin'
    has_final = final_st.exists() or final_bin.exists()

    # 读 loss_log.csv
    loss_log = d / 'loss_log.csv'
    if loss_log.exists():
        lines = loss_log.read_text(encoding='utf-8').strip().split('\n')
        info['steps'] = max(0, len(lines) - 1)  # 减去 header
        if len(lines) > 1:
            last_line = lines[-1].split(',')
            if len(last_line) >= 3:
                info['final_loss'] = last_line[2]  # loss column

    # 读 training_summary.json
    summary = d / 'training_summary.json'
    if summary.exists():
        try:
            data = json.loads(summary.read_text())
            info['max_steps'] = data.get('total_steps', '?')
            info['duration'] = f"{data.get('training_time_min', 0):.1f} min"
            info['gpu_peak'] = f"{data.get('peak_gpu_memory_MB', 0):.0f} MB"
            info['final_loss'] = f"{data.get('final_loss', 0):.6f}"
        except Exception:
            pass

    # 读 training_config.json 获取 max_steps
    config = d / 'training_config.json'
    if config.exists() and info['max_steps'] == '?':
        try:
            data = json.loads(config.read_text())
            info['max_steps'] = data.get('max_train_steps', '?')
        except Exception:
            pass

    # 检查 checkpoints
    for ckpt in sorted(d.glob('checkpoint-*')):
        if ckpt.is_dir():
            info['has_checkpoints'].append(ckpt.name)
            # 检查是否有 validation samples
            if (ckpt / 'samples').exists():
                info['has_validation'] = True

    # 状态
    if has_final:
        info['status'] = '已完成'
    elif info['steps'] > 0:
        info['status'] = '训练中'
    else:
        info['status'] = '未开始'

    return info


def read_experiment_log(results_dir):
    """读取 experiment_log.csv"""
    log_path = Path(results_dir) / 'experiment_log.csv'
    if not log_path.exists():
        return []
    rows = []
    lines = log_path.read_text(encoding='utf-8').strip().split('\n')
    if len(lines) < 2:
        return []
    headers = lines[0].split(',')
    for line in lines[1:]:
        vals = line.split(',')
        row = dict(zip(headers, vals))
        rows.append(row)
    return rows


def generate_report(results_dir):
    """生成可读的进度报告文本。"""
    results_dir = Path(results_dir)
    lines = []

    lines.append('=' * 70)
    lines.append(f'  KMC-LoRA 实验进度报告')
    lines.append(f'  生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    lines.append(f'  结果目录: {results_dir.resolve()}')
    lines.append('=' * 70)

    # GPU 状态
    gpu = gpu_info()
    lines.append(f'\n[GPU 状态] {gpu}')

    # 扫描实验
    experiments = scan_experiments(results_dir)
    if not experiments:
        lines.append('\n(未找到任何实验)')
        return '\n'.join(lines)

    # 统计
    done = sum(1 for e in experiments if e['status'] == '已完成')
    running = sum(1 for e in experiments if e['status'] == '训练中')
    pending = sum(1 for e in experiments if e['status'] == '未开始')
    partial = sum(1 for e in experiments if e['status'] == '部分完成')

    lines.append(f'\n[总体进度] 已完成: {done}  训练中: {running}  '
                 f'部分完成: {partial}  未开始: {pending}  总计: {len(experiments)}')
    pct = done / len(experiments) * 100 if experiments else 0
    bar_len = 40
    filled = int(bar_len * pct / 100)
    bar = '█' * filled + '░' * (bar_len - filled)
    lines.append(f'  [{bar}] {pct:.0f}%')

    # 详细列表
    lines.append(f'\n{"─" * 70}')
    lines.append(f'{"实验名":<30} {"状态":<8} {"步数":<12} {"Loss":<12} {"耗时":<12}')
    lines.append(f'{"─" * 70}')

    for e in experiments:
        if e.get('sub_phases'):
            # KMC 多阶段实验
            lines.append(f'{e["name"]:<30} {e["status"]:<8}')
            for p in e['sub_phases']:
                steps_str = f'{p["steps"]}/{p["max_steps"]}'
                loss_str = p.get('final_loss', '')[:10]
                dur_str = p.get('duration', '')
                lines.append(f'  └─ {p["name"]:<26} {p["status"]:<8} '
                             f'{steps_str:<12} {loss_str:<12} {dur_str:<12}')
        else:
            steps_str = f'{e["steps"]}/{e["max_steps"]}'
            loss_str = e.get('final_loss', '')[:10]
            dur_str = e.get('duration', '')
            lines.append(f'{e["name"]:<30} {e["status"]:<8} '
                         f'{steps_str:<12} {loss_str:<12} {dur_str:<12}')

    # Experiment Log
    log_rows = read_experiment_log(results_dir)
    if log_rows:
        lines.append(f'\n{"─" * 70}')
        lines.append(f'[实验日志] (来自 experiment_log.csv)')
        lines.append(f'{"─" * 70}')
        lines.append(f'{"实验":<28} {"状态":<6} {"耗时(min)":<10} {"退出码":<6}')
        for r in log_rows:
            lines.append(f'{r.get("experiment",""):<28} '
                         f'{r.get("status",""):<6} '
                         f'{r.get("duration_min",""):<10} '
                         f'{r.get("exit_code",""):<6}')

    lines.append(f'\n{"=" * 70}')
    lines.append(f'报告结束')
    lines.append(f'{"=" * 70}')

    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description='实验进度监控')
    ap.add_argument('--results-dir', default='kmc_lora/results',
                    help='实验结果目录')
    ap.add_argument('--out', default='kmc_lora/results/progress.txt',
                    help='输出报告文件路径')
    ap.add_argument('--watch', type=int, default=1800,
                    help='自动刷新间隔(秒), 默认1800秒(30分钟), 0=只运行一次')
    ap.add_argument('--once', action='store_true',
                    help='只运行一次并退出(等价于 --watch 0)')
    args = ap.parse_args()

    if args.once:
        args.watch = 0

    while True:
        report = generate_report(args.results_dir)

        # 写文件
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as f:
            f.write(report)

        # 同时打印到终端
        print(report)

        if args.watch <= 0:
            break
        next_minutes = args.watch / 60
        print(f'\n(下次刷新: {args.watch}秒后, 约 {next_minutes:.1f} 分钟后...)')
        time.sleep(args.watch)


if __name__ == '__main__':
    main()
