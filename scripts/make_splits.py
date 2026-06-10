import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in-csv', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--medium-percentile', type=float, default=90)
    ap.add_argument('--sub-50', type=int, default=50)
    ap.add_argument('--sub-25', type=int, default=25)
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # D-High = full
    df['path'].to_csv(out_dir / 'D-High.txt', index=False, header=False)

    # D-Medium = remove top (100 - percentile) difficulty
    thresh = np.percentile(df['difficulty'], args.medium_percentile)
    df[df['difficulty'] <= thresh]['path'].to_csv(out_dir / 'D-Medium.txt', index=False, header=False)

    # D-Low = largest cluster only
    counts = df['cluster'].value_counts()
    largest = counts.index[0]
    df[df['cluster'] == largest]['path'].to_csv(out_dir / 'D-Low.txt', index=False, header=False)

    # Subsets
    df.sample(n=min(args.sub_50, len(df)), random_state=42)['path'].to_csv(out_dir / 'D-Sub-50.txt', index=False, header=False)
    df.sample(n=min(args.sub_25, len(df)), random_state=43)['path'].to_csv(out_dir / 'D-Sub-25.txt', index=False, header=False)


if __name__ == '__main__':
    main()
