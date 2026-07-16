"""
Analysis and visualization for multi-seed validation results.
Generates statistics table and comparison figure for paper.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_multiseed_results(csv_path, output_dir):
    """Analyze multi-seed results and generate figure for paper."""
    df = pd.read_csv(csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("MULTI-SEED VALIDATION ANALYSIS")
    print("="*60)

    # Compute statistics per strategy
    stats = []
    for (dataset, strategy), group in df.groupby(['dataset', 'strategy']):
        fids = group['fid'].dropna()
        clips = group['clip_score'].dropna()

        if len(fids) < 2:
            continue

        stat = {
            'dataset': dataset,
            'strategy': strategy,
            'n': len(fids),
            'fid_mean': fids.mean(),
            'fid_std': fids.std(ddof=1),
            'fid_cv': fids.std(ddof=1) / fids.mean() * 100,
            'fid_min': fids.min(),
            'fid_max': fids.max(),
            'clip_mean': clips.mean(),
            'clip_std': clips.std(ddof=1),
            'clip_cv': clips.std(ddof=1) / clips.mean() * 100,
            'clip_min': clips.min(),
            'clip_max': clips.max(),
        }
        stats.append(stat)

        print(f"\n{dataset} / {strategy}:")
        print(f"  N={stat['n']}, FID={stat['fid_mean']:.2f}±{stat['fid_std']:.2f} (CV={stat['fid_cv']:.1f}%)")
        print(f"  CLIP={stat['clip_mean']:.4f}±{stat['clip_std']:.4f} (CV={stat['clip_cv']:.1f}%)")

    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(output_dir / 'multiseed_statistics.csv', index=False)

    # Generate visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Plot 1: FID by strategy with error bars
    ax = axes[0]
    x = np.arange(len(stats_df))
    means = stats_df['fid_mean'].values
    stds = stats_df['fid_std'].values
    labels = [f"{s['dataset'][:3]}\n{s['strategy'][:10]}" for _, s in stats_df.iterrows()]

    bars = ax.bar(x, means, yerr=stds, capsize=5, color=['#2196F3', '#4CAF50', '#FF9800'])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('FID (lower is better)', fontsize=11)
    ax.set_title('FID Stability Across Random Seeds\n(5 seeds per strategy)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Add CV annotation
    for i, (mean, std, cv) in enumerate(zip(means, stds, stats_df['fid_cv'])):
        ax.annotate(f'CV={cv:.1f}%', (i, mean + std + 5), ha='center', fontsize=8)

    # Plot 2: Distribution of runs
    ax = axes[1]
    for i, ((dataset, strategy), group) in enumerate(df.groupby(['dataset', 'strategy'])):
        fids = group['fid'].dropna().values
        if len(fids) > 0:
            # Scatter with jitter
            x_pos = np.random.normal(i, 0.1, len(fids))
            ax.scatter(x_pos, fids, alpha=0.6, s=100, color=['#2196F3', '#4CAF50', '#FF9800'][i % 3])
            # Mean line
            ax.hlines(fids.mean(), i-0.3, i+0.3, colors='red', linestyles='--', linewidth=2)

    ax.set_xticks(range(len(stats_df)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('FID', fontsize=11)
    ax.set_title('Individual Runs Distribution\n(Red line = mean)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'fig_multiseed_validation.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig_multiseed_validation.png', dpi=300, bbox_inches='tight')
    print(f"\n[OK] Figure saved to {output_dir / 'fig_multiseed_validation.png'}")

    # Generate discussion text
    print("\n" + "="*60)
    print("DISCUSSION POINTS FOR PAPER")
    print("="*60)

    max_cv = stats_df['fid_cv'].max()
    min_cv = stats_df['fid_cv'].min()

    print(f"\n1. Coefficient of Variation (CV) ranges from {min_cv:.1f}% to {max_cv:.1f}%")
    print(f"   - CV < 5%: Very stable")
    print(f"   - CV 5-10%: Moderately stable")
    print(f"   - CV > 10%: High variance")

    print(f"\n2. Strategy comparison:")
    for _, s in stats_df.iterrows():
        stability = "stable" if s['fid_cv'] < 5 else "moderate" if s['fid_cv'] < 10 else "variable"
        print(f"   - {s['strategy']}: FID varies {s['fid_std']:.2f} ({stability})")

    # Compare to single-run baseline
    print(f"\n3. Implications for previous analysis:")
    print(f"   - Original N=17 experiments assumed negligible seed variance")
    print(f"   - Multi-seed validation shows actual run-to-run variation")
    print(f"   - Bootstrap CI in main analysis likely UNDERESTIMATES uncertainty")
    print(f"     (it accounts for sampling variation but not seed variance)")

    return stats_df

if __name__ == '__main__':
    import sys

    results_file = Path('<project_root>/kmc_lora/results/multiseed_results.csv')
    output_dir = Path('<project_root>/kmc_lora/figures/multiseed')

    if not results_file.exists():
        print(f"[WARN] Results file not found: {results_file}")
        print("Run the multi-seed experiment first:")
        print("  python kmc_lora/scripts/run_multiseed_validation.py")
        sys.exit(1)

    analyze_multiseed_results(results_file, output_dir)
