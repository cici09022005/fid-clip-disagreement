import argparse
import csv
import json
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


STAGE_A_SPLITS = ['D-Sub-25', 'D-Sub-50', 'D-Medium']
STAGE_B_SPLITS = ['D-High', 'D-Low']
STAGE_A_METHODS = ['p2-only', 'p3-only-long', 'p2-p3-replay']
STAGE_B_METHODS = ['p2-only', 'p3-only-long']


def timestamp():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


class PipelineLogger:
    def __init__(self, out_root):
        self.out_root = Path(out_root)
        self.out_root.mkdir(parents=True, exist_ok=True)
        self.log_path = self.out_root / 'pipeline.log'

    def log(self, message):
        line = f'[{timestamp()}] {message}'
        print(line, flush=True)
        with open(self.log_path, 'a', encoding='utf-8') as handle:
            handle.write(line + '\n')


def update_stage_status(out_root, stage_name, status, extra=None):
    status_path = Path(out_root) / 'stage_status.json'
    data = {}
    if status_path.exists():
        data = json.loads(status_path.read_text(encoding='utf-8'))
    data[stage_name] = {
        'status': status,
        'updated_at': datetime.now().isoformat(),
    }
    if extra:
        data[stage_name].update(extra)
    status_path.write_text(json.dumps(data, indent=2, ensure_ascii=True), encoding='utf-8')


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding='utf-8')


def run_streaming(cmd, logger, max_retries, retry_delay_sec):
    for attempt in range(1, max_retries + 1):
        cmd_text = ' '.join(cmd)
        logger.log(f'RUN attempt={attempt}/{max_retries}: {cmd_text}')
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            logger.log(line.rstrip())
        rc = process.wait()
        if rc == 0:
            return
        logger.log(f'FAIL exit_code={rc}')
        if attempt == max_retries:
            raise RuntimeError(f'command failed after {max_retries} attempts: {cmd_text}')
        logger.log(f'RETRY sleeping {retry_delay_sec} seconds before retry')
        time.sleep(retry_delay_sec)


def experiment_dir(out_root, method, split_name, seed):
    return Path(out_root) / f'{method}_{split_name}_seed{seed}'


def eval_csv_path(out_root, method, split_name, seed):
    return experiment_dir(out_root, method, split_name, seed) / 'eval.csv'


def run_refinement(method, split_name, seed, args, logger):
    eval_csv = eval_csv_path(args.out_root, method, split_name, seed)
    if eval_csv.exists():
        logger.log(f'SKIP existing eval: method={method} split={split_name} seed={seed}')
        return

    cmd = [
        sys.executable,
        'kmc_lora/scripts/run_refinement_experiment.py',
        '--method', method,
        '--split', split_name,
        '--seed', str(seed),
        '--out-root', args.out_root,
        '--num-per-prompt', str(args.num_per_prompt),
        '--max-real', str(args.max_real),
        '--resume',
        '--disable-train-validation',
    ]
    if args.total_steps_override is not None:
        cmd += ['--total-steps-override', str(args.total_steps_override)]

    run_streaming(cmd, logger, args.max_retries, args.retry_delay_sec)


def read_fid(csv_path):
    with open(csv_path, 'r', encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        row = next(reader)
    return float(row['fid'])


def collect_stats(out_root, split_methods, seeds):
    stats = {}
    for split_name, methods in split_methods.items():
        for method in methods:
            fids = []
            for seed in seeds:
                csv_path = eval_csv_path(out_root, method, split_name, seed)
                if not csv_path.exists():
                    raise FileNotFoundError(f'missing eval file: {csv_path}')
                fids.append(read_fid(csv_path))
            stats[(split_name, method)] = {
                'mean_fid': statistics.mean(fids),
                'std_fid': statistics.stdev(fids) if len(fids) > 1 else 0.0,
                'num_seeds': len(fids),
                'fids': fids,
            }
    return stats


def choose_best_methods(stats, splits, methods):
    mapping = {}
    for split_name in splits:
        best_method = None
        best_mean = None
        for method in methods:
            mean_fid = stats[(split_name, method)]['mean_fid']
            if best_mean is None or mean_fid < best_mean:
                best_mean = mean_fid
                best_method = method
        mapping[split_name] = best_method
    return mapping


def write_method_summary(out_root, stats):
    path = Path(out_root) / 'method_summary.csv'
    with open(path, 'w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['split', 'method', 'mean_fid', 'std_fid', 'num_seeds', 'seed_fids'])
        for (split_name, method), info in sorted(stats.items()):
            writer.writerow([
                split_name,
                method,
                f"{info['mean_fid']:.4f}",
                f"{info['std_fid']:.4f}",
                info['num_seeds'],
                ';'.join(f'{fid:.4f}' for fid in info['fids']),
            ])
    return path


def write_mapping(out_root, mapping, stats):
    json_path = Path(out_root) / 'scale_aware_mapping.json'
    json_path.write_text(json.dumps(mapping, indent=2, ensure_ascii=True), encoding='utf-8')

    csv_path = Path(out_root) / 'scale_aware_mapping.csv'
    with open(csv_path, 'w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['split', 'selected_method', 'mean_fid', 'std_fid'])
        for split_name, method in mapping.items():
            info = stats[(split_name, method)]
            writer.writerow([split_name, method, f"{info['mean_fid']:.4f}", f"{info['std_fid']:.4f}"])
    return json_path, csv_path


def write_scale_aware_summary(out_root, mapping, stats):
    path = Path(out_root) / 'scale_aware_summary.csv'
    selected_means = []
    with open(path, 'w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['split', 'scale_aware_method', 'mean_fid', 'std_fid'])
        for split_name in sorted(mapping):
            method = mapping[split_name]
            info = stats[(split_name, method)]
            selected_means.append(info['mean_fid'])
            writer.writerow([split_name, method, f"{info['mean_fid']:.4f}", f"{info['std_fid']:.4f}"])

        writer.writerow([])
        writer.writerow(['overall_avg_mean_fid', f'{statistics.mean(selected_means):.4f}'])
    return path


def main():
    ap = argparse.ArgumentParser(description='Run unattended publication-oriented refinement pipeline')
    ap.add_argument('--out-root', default='kmc_lora/results/scale_aware_paper')
    ap.add_argument('--seeds', nargs='+', type=int, default=[42, 52, 62])
    ap.add_argument('--num-per-prompt', type=int, default=100)
    ap.add_argument('--max-real', type=int, default=500)
    ap.add_argument('--total-steps-override', type=int, default=None)
    ap.add_argument('--max-retries', type=int, default=3)
    ap.add_argument('--retry-delay-sec', type=int, default=120)
    args = ap.parse_args()

    logger = PipelineLogger(args.out_root)
    logger.log('publication pipeline started')

    stage_a_splits = list(STAGE_A_SPLITS)
    stage_b_splits = list(STAGE_B_SPLITS)
    stage_a_methods = list(STAGE_A_METHODS)
    stage_b_methods = list(STAGE_B_METHODS)

    write_json(Path(args.out_root) / 'pipeline_manifest.json', {
        'created_at': datetime.now().isoformat(),
        'out_root': args.out_root,
        'seeds': args.seeds,
        'num_per_prompt': args.num_per_prompt,
        'max_real': args.max_real,
        'total_steps_override': args.total_steps_override,
        'max_retries': args.max_retries,
        'retry_delay_sec': args.retry_delay_sec,
        'stage_a_splits': stage_a_splits,
        'stage_b_splits': stage_b_splits,
        'stage_a_methods': stage_a_methods,
        'stage_b_methods': stage_b_methods,
    })

    update_stage_status(args.out_root, 'stage_a', 'running', {
        'splits': stage_a_splits,
        'methods': stage_a_methods,
        'seeds': args.seeds,
    })
    for split_name in stage_a_splits:
        for method in stage_a_methods:
            for seed in args.seeds:
                run_refinement(method, split_name, seed, args, logger)
    update_stage_status(args.out_root, 'stage_a', 'completed')

    update_stage_status(args.out_root, 'stage_b', 'running', {
        'splits': stage_b_splits,
        'methods': stage_b_methods,
        'seeds': args.seeds,
    })
    for split_name in stage_b_splits:
        for method in stage_b_methods:
            for seed in args.seeds:
                run_refinement(method, split_name, seed, args, logger)
    update_stage_status(args.out_root, 'stage_b', 'completed')

    update_stage_status(args.out_root, 'aggregation', 'running')
    all_splits = stage_a_splits + stage_b_splits
    split_methods = {split_name: list(stage_a_methods) for split_name in stage_a_splits}
    split_methods.update({split_name: list(stage_b_methods) for split_name in stage_b_splits})
    stats = collect_stats(args.out_root, split_methods, args.seeds)
    method_summary = write_method_summary(args.out_root, stats)

    mapping = {}
    mapping.update(choose_best_methods(stats, stage_a_splits, stage_a_methods))
    mapping.update(choose_best_methods(stats, stage_b_splits, stage_b_methods))
    mapping_json, mapping_csv = write_mapping(args.out_root, mapping, stats)
    scale_aware_summary = write_scale_aware_summary(args.out_root, mapping, stats)

    update_stage_status(args.out_root, 'aggregation', 'completed', {
        'method_summary': str(method_summary),
        'mapping_json': str(mapping_json),
        'mapping_csv': str(mapping_csv),
        'scale_aware_summary': str(scale_aware_summary),
    })
    update_stage_status(args.out_root, 'pipeline', 'completed', {
        'selected_mapping': mapping,
    })

    logger.log(f'final_mapping={json.dumps(mapping, ensure_ascii=True)}')
    logger.log(f'method_summary={method_summary}')
    logger.log(f'scale_aware_summary={scale_aware_summary}')
    # --- Postprocess: analysis and figure generation ---
    update_stage_status(args.out_root, 'postprocess', 'running')
    logger.log('postprocess: running collect_all_results.py')
    try:
        subprocess.run([
            sys.executable,
            'kmc_lora/scripts/collect_all_results.py'
        ], check=True)
        logger.log('postprocess: collect_all_results.py completed')
    except Exception as e:
        logger.log(f'postprocess: collect_all_results.py failed: {e}')
        update_stage_status(args.out_root, 'postprocess', 'failed', {'error': str(e)})
        return

    logger.log('postprocess: running generate_figures.py')
    figures_dir = Path(args.out_root) / 'figures'
    figures_dir.mkdir(exist_ok=True)
    try:
        subprocess.run([
            sys.executable,
            'kmc_lora/scripts/generate_figures.py',
            '--out-dir', str(figures_dir)
        ], check=True)
        logger.log('postprocess: generate_figures.py completed')
        update_stage_status(args.out_root, 'postprocess', 'completed', {'figures_dir': str(figures_dir)})
    except Exception as e:
        logger.log(f'postprocess: generate_figures.py failed: {e}')
        update_stage_status(args.out_root, 'postprocess', 'failed', {'error': str(e)})
        return

    logger.log('postprocess: running analyze_scale_aware_results.py')
    try:
        subprocess.run([
            sys.executable,
            'kmc_lora/scripts/analyze_scale_aware_results.py',
            '--out-root', args.out_root,
        ], check=True)
        logger.log('postprocess: analyze_scale_aware_results.py completed')
    except Exception as e:
        logger.log(f'postprocess: analyze_scale_aware_results.py failed: {e}')
        update_stage_status(args.out_root, 'postprocess', 'failed', {'error': str(e)})
        return

    logger.log('publication pipeline completed')


if __name__ == '__main__':
    main()
