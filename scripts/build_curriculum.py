import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in-csv', required=True)
    ap.add_argument('--out-csv', required=True)
    ap.add_argument('--phase1', type=float, default=0.4)
    ap.add_argument('--phase2', type=float, default=0.35)
    ap.add_argument('--phase3', type=float, default=0.25)
    ap.add_argument('--w-quality', type=float, default=0.5)
    ap.add_argument('--w-typicality', type=float, default=0.3)
    ap.add_argument('--w-heterogeneity', type=float, default=0.2)
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)

    difficulty = (
        args.w_quality * (1.0 - df['quality'].values) +
        args.w_typicality * (1.0 - df['typicality'].values) +
        args.w_heterogeneity * df['heterogeneity'].values
    )
    df['difficulty'] = difficulty

    df = df.sort_values('difficulty', ascending=True).reset_index(drop=True)
    n = len(df)
    n1 = int(n * args.phase1)
    n3 = max(1, int(round(n * args.phase3)))

    # Use cumulative curriculum stages instead of mutually-exclusive bins.
    df['phase1'] = False
    if n1 > 0:
        df.loc[:n1 - 1, 'phase1'] = True

    # Phase 2 expands to the full split.
    df['phase2'] = True

    # Phase 3 refines on the most typical samples.
    df['phase3'] = False
    typical_top_idx = df.sort_values('typicality', ascending=False).head(n3).index
    df.loc[typical_top_idx, 'phase3'] = True

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df[df['phase1']]['path'].to_csv(out_dir / 'phase1.txt', index=False, header=False)
    df[df['phase2']]['path'].to_csv(out_dir / 'phase2.txt', index=False, header=False)
    df[df['phase3']]['path'].to_csv(out_dir / 'phase3.txt', index=False, header=False)

    print(
        "Phase1: "
        f"{int(df['phase1'].sum())}, "
        f"Phase2: {int(df['phase2'].sum())}, "
        f"Phase3: {int(df['phase3'].sum())}"
    )


if __name__ == '__main__':
    main()
