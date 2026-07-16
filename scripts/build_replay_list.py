import argparse
import random
from pathlib import Path


def read_list(path):
    return [line.strip() for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--primary-list', required=True,
                    help='Main list to prioritize, e.g. phase3 typical samples')
    ap.add_argument('--replay-list', required=True,
                    help='Replay list to mix in, e.g. phase2 full split')
    ap.add_argument('--primary-ratio', type=float, default=0.8)
    ap.add_argument('--replay-ratio', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out-list', required=True)
    args = ap.parse_args()

    primary = read_list(args.primary_list)
    replay = read_list(args.replay_list)
    if not primary:
        raise ValueError('primary list is empty')
    if not replay:
        raise ValueError('replay list is empty')
    if args.primary_ratio <= 0 or args.replay_ratio < 0:
        raise ValueError('ratios must be non-negative and primary_ratio > 0')

    rng = random.Random(args.seed)
    replay_count = int(round(len(primary) * args.replay_ratio / args.primary_ratio))
    if replay_count <= 0 and args.replay_ratio > 0:
        replay_count = 1

    if replay_count <= len(replay):
        replay_subset = rng.sample(replay, replay_count)
    else:
        replay_subset = [rng.choice(replay) for _ in range(replay_count)]

    mixed = list(primary) + replay_subset
    rng.shuffle(mixed)

    out_path = Path(args.out_list)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text('\n'.join(mixed) + '\n', encoding='utf-8')

    realized_primary = len(primary) / len(mixed)
    realized_replay = len(replay_subset) / len(mixed)
    print(f'primary={len(primary)} replay={len(replay_subset)} total={len(mixed)}')
    print(f'ratios: primary={realized_primary:.4f} replay={realized_replay:.4f}')
    print(f'out={out_path}')


if __name__ == '__main__':
    main()