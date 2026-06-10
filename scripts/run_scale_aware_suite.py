import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_SPLIT_METHODS = {
    'D-Sub-25': 'scale-aware-v1',
    'D-Sub-50': 'scale-aware-v1',
    'D-Medium': 'scale-aware-v1',
}

DEFAULT_BASELINES = ['p2-only', 'p3-only-long', 'p2-p3-replay']


def run(cmd):
    print(' '.join(cmd))
    result = subprocess.run(cmd, text=True)
    return result.returncode


def parse_split_method_overrides(items):
    mapping = {}
    for item in items or []:
        if '=' not in item:
            raise ValueError(f'invalid split method override: {item}')
        split_name, method_name = item.split('=', 1)
        mapping[split_name] = method_name
    return mapping


def main():
    ap = argparse.ArgumentParser(description='Run publication-oriented scale-aware experiment suites')
    ap.add_argument('--config', default='kmc_lora/configs/base.yaml')
    ap.add_argument('--out-root', default='kmc_lora/results/scale_aware_formal')
    ap.add_argument('--splits', nargs='+', default=['D-Sub-25', 'D-Sub-50', 'D-Medium'])
    ap.add_argument('--seeds', nargs='+', type=int, default=[42, 52, 62])
    ap.add_argument('--num-per-prompt', type=int, default=100)
    ap.add_argument('--max-real', type=int, default=500)
    ap.add_argument('--total-steps-override', type=int, default=None)
    ap.add_argument('--disable-train-validation', action='store_true')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--skip-train', action='store_true')
    ap.add_argument('--skip-generate', action='store_true')
    ap.add_argument('--skip-eval', action='store_true')
    ap.add_argument('--scale-aware-only', action='store_true')
    ap.add_argument('--baselines', nargs='*', default=DEFAULT_BASELINES)
    ap.add_argument('--split-method', nargs='*', default=[],
                    help='Override scale-aware mapping for specific splits, e.g. D-High=p2-only')
    args = ap.parse_args()

    split_methods = dict(DEFAULT_SPLIT_METHODS)
    split_methods.update(parse_split_method_overrides(args.split_method))

    suite_dir = Path(args.out_root)
    suite_dir.mkdir(parents=True, exist_ok=True)
    log_csv = suite_dir / 'suite_log.csv'

    log_exists = log_csv.exists()
    with open(log_csv, 'a' if log_exists else 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not log_exists:
            writer.writerow([
                'experiment_group', 'split', 'seed', 'method', 'status',
                'exit_code', 'elapsed_sec'
            ])

        for split_name in args.splits:
            methods = []
            scale_method = split_methods.get(split_name)
            if scale_method:
                methods.append(scale_method)
            if not args.scale_aware_only:
                methods.extend(args.baselines)

            seen = set()
            ordered_methods = []
            for method_name in methods:
                if method_name not in seen:
                    ordered_methods.append(method_name)
                    seen.add(method_name)

            for seed in args.seeds:
                for method_name in ordered_methods:
                    cmd = [
                        sys.executable,
                        'kmc_lora/scripts/run_refinement_experiment.py',
                        '--config', args.config,
                        '--method', method_name,
                        '--split', split_name,
                        '--seed', str(seed),
                        '--out-root', str(suite_dir),
                        '--num-per-prompt', str(args.num_per_prompt),
                        '--max-real', str(args.max_real),
                    ]
                    if args.total_steps_override is not None:
                        cmd += ['--total-steps-override', str(args.total_steps_override)]
                    if args.disable_train_validation:
                        cmd.append('--disable-train-validation')
                    if args.resume:
                        cmd.append('--resume')
                    if args.skip_train:
                        cmd.append('--skip-train')
                    if args.skip_generate:
                        cmd.append('--skip-generate')
                    if args.skip_eval:
                        cmd.append('--skip-eval')

                    t0 = time.time()
                    rc = run(cmd)
                    elapsed = round(time.time() - t0, 1)
                    writer.writerow([
                        'scale-aware-formal', split_name, seed, method_name,
                        'ok' if rc == 0 else 'fail', rc, elapsed,
                    ])
                    f.flush()

                    if rc != 0:
                        raise SystemExit(rc)


if __name__ == '__main__':
    main()
