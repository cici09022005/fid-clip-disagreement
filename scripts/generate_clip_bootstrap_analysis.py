"""
Generate CLIP Score analysis and FID-CLIP joint Pareto analysis
Also includes bootstrap confidence intervals
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# Read data
df = pd.read_csv('kmc_lora/results/all_eval_results.csv')

# Create output directory
out_dir = Path('kmc_lora/figures/pareto_analysis_v2')
out_dir.mkdir(parents=True, exist_ok=True)

# Define method categories
df['method_type'] = df['experiment'].apply(lambda x:
    'Random' if 'Random' in x else
    'KMC' if 'KMC' in x else
    'Ablation' if 'Ablation' in x else
    'Quality' if 'Quality' in x else
    'Anti' if 'Anti' in x else 'Other'
)

# Extract regime
def extract_regime(exp_name):
    if 'D-High' in exp_name:
        return 'D-High'
    elif 'D-Medium' in exp_name:
        return 'D-Medium'
    elif 'D-Low' in exp_name:
        return 'D-Low'
    elif 'D-Sub-50' in exp_name:
        return 'D-Sub-50'
    elif 'D-Sub-25' in exp_name:
        return 'D-Sub-25'
    return 'Other'

df['regime'] = df['experiment'].apply(extract_regime)

# Training time map
training_time_map = {
    'anime_student': {
        'Random_D-High': 12.0, 'KMC_D-High': 17.2,
        'Random_D-Medium': 25.0, 'KMC_D-Medium': 19.7,
        'Random_D-Low': 22.5, 'KMC_D-Low': 29.4,
        'Random_D-Sub-50': 25.2, 'KMC_D-Sub-50': 11.9,
        'Random_D-Sub-25': 21.6, 'KMC_D-Sub-25': 12.2,
        'Ablation_NoPhase1': 15.0, 'Ablation_NoPhase2': 15.0, 'Ablation_NoPhase3': 15.0,
        'Ablation_Phase1Only': 20.0, 'Ablation_Phase3Only': 15.0,
        'Quality_Filter': 13.6, 'Anti_Curriculum': 11.2
    },
    'wikiart_mixed': {
        'Random_D-High': 70.8, 'KMC_D-High': 8.9,
        'Random_D-Medium': 8.1, 'KMC_D-Medium': 8.7,
        'Random_D-Low': 8.7, 'KMC_D-Low': 8.8,
        'Random_D-Sub-50': 11.2, 'KMC_D-Sub-50': 9.1,
        'Random_D-Sub-25': 15.0, 'KMC_D-Sub-25': 8.7,
        'Ablation_NoPhase1': 10.0, 'Ablation_NoPhase2': 10.0, 'Ablation_NoPhase3': 10.0,
        'Ablation_Phase1Only': 18.0, 'Ablation_Phase3Only': 10.0,
        'Quality_Filter': 9.2, 'Anti_Curriculum': 8.0
    },
    'dreambooth_mixed': {
        'Random_D-High': 6.1, 'KMC_D-High': 11.0,
        'Random_D-Medium': 6.4, 'KMC_D-Medium': 11.0,
        'Random_D-Low': 16.4, 'KMC_D-Low': 11.0,
        'Random_D-Sub-50': 6.1, 'KMC_D-Sub-50': 11.0,
        'Random_D-Sub-25': 8.8, 'KMC_D-Sub-25': 11.1,
        'Ablation_NoPhase1': 11.0, 'Ablation_NoPhase2': 11.0, 'Ablation_NoPhase3': 11.0,
        'Ablation_Phase1Only': 12.0, 'Ablation_Phase3Only': 11.0,
        'Quality_Filter': 12.5, 'Anti_Curriculum': 6.1
    },
    'dreambooth_single': {
        'Random_D-High': 14.4, 'KMC_D-High': 42.3,
        'Random_D-Medium': 17.5, 'KMC_D-Medium': 43.2,
        'Random_D-Low': 22.5, 'KMC_D-Low': 43.0,
        'Random_D-Sub-50': 14.6, 'KMC_D-Sub-50': 43.5,
        'Random_D-Sub-25': 14.8, 'KMC_D-Sub-25': 43.7,
        'Ablation_NoPhase1': 43.0, 'Ablation_NoPhase2': 43.0, 'Ablation_NoPhase3': 43.0,
        'Ablation_Phase1Only': 25.0, 'Ablation_Phase3Only': 43.0,
        'Quality_Filter': 64.7, 'Anti_Curriculum': 15.3
    }
}

df['training_time'] = df.apply(
    lambda row: training_time_map.get(row['dataset'], {}).get(row['experiment'], np.nan),
    axis=1
)

df = df[df['training_time'].notna()]

# ==================== FIGURE 1: FID vs CLIP Trade-off ====================
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

datasets = ['anime_student', 'wikiart_mixed', 'dreambooth_mixed', 'dreambooth_single']
dataset_labels = ['Anime-Student', 'WikiArt-Mixed', 'DreamBooth-Mixed', 'DreamBooth-Single']
colors = {'Random': '#FF9800', 'KMC': '#2196F3', 'Ablation': '#9C27B0',
          'Quality': '#00BCD4', 'Anti': '#E91E63'}

fid_clip_correlation = []

for idx, (dataset, label) in enumerate(zip(datasets, dataset_labels)):
    ax = axes[idx]
    ds_data = df[df['dataset'] == dataset]

    # Scatter plot by method type
    for mtype, color in colors.items():
        subset = ds_data[ds_data['method_type'] == mtype]
        if len(subset) > 0:
            ax.scatter(subset['fid'], subset['clip_score'],
                      c=color, label=mtype, alpha=0.6, s=100,
                      edgecolors='white', linewidth=1)

    # Compute and report correlation
    valid_data = ds_data[ds_data['fid'].notna() & ds_data['clip_score'].notna()]
    if len(valid_data) > 3:
        corr, p_value = stats.pearsonr(valid_data['fid'], valid_data['clip_score'])
        fid_clip_correlation.append({
            'dataset': dataset,
            'correlation': corr,
            'p_value': p_value
        })

        # Add correlation text
        ax.text(0.05, 0.95, f'Correlation: {corr:.3f}\np-value: {p_value:.3f}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel('FID (lower is better)', fontsize=11)
    ax.set_ylabel('CLIP Score (higher is better)', fontsize=11)
    ax.set_title(label, fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    if idx == 0:
        ax.legend(loc='upper right')

plt.tight_layout()
plt.savefig(out_dir / 'fig1_fid_vs_clip_tradeoff.pdf', dpi=300, bbox_inches='tight')
plt.savefig(out_dir / 'fig1_fid_vs_clip_tradeoff.png', dpi=300, bbox_inches='tight')
plt.close()

print("[OK] Generated Figure 1: FID vs CLIP Trade-off")

# ==================== FIGURE 2: 2D Pareto Frontiers (FID vs CLIP) ====================
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

regimes = ['D-High', 'D-Medium', 'D-Low', 'D-Sub-50', 'D-Sub-25']
regime_labels = ['D-High\n(Full)', 'D-Medium\n(90%)', 'D-Low\n(Largest Cluster)',
                 'D-Sub-50', 'D-Sub-25']

pareto_points_by_regime = {}

for idx, (regime, label) in enumerate(zip(regimes, regime_labels)):
    ax = axes[idx]
    regime_data = df[df['regime'] == regime]

    # Find Pareto optimal points in FID-CLIP space
    # Minimize FID, Maximize CLIP -> convert CLIP to negative for minimization
    points = regime_data[['fid', 'clip_score']].values
    valid_mask = ~np.isnan(points).any(axis=1)
    points = points[valid_mask]
    experiment_names = regime_data['experiment'].values[valid_mask]
    method_types = regime_data['method_type'].values[valid_mask]

    # Convert to minimization problem: minimize FID, minimize (-CLIP)
    pareto_points = []
    pareto_names = []
    pareto_types = []

    for i, (fid, clip) in enumerate(points):
        # Check if dominated
        is_dominated = False
        for j, (other_fid, other_clip) in enumerate(points):
            if i != j:
                if other_fid <= fid and other_clip >= clip and (other_fid < fid or other_clip > clip):
                    is_dominated = True
                    break
        if not is_dominated:
            pareto_points.append([fid, clip])
            pareto_names.append(experiment_names[i])
            pareto_types.append(method_types[i])

    pareto_points_by_regime[regime] = {
        'points': pareto_points,
        'names': pareto_names,
        'types': pareto_types
    }

    # Plot all points
    for mtype, color in colors.items():
        mask = np.array([t == mtype for t in method_types])
        if mask.any():
            ax.scatter(points[mask, 0], points[mask, 1],
                      c=color, label=mtype, alpha=0.5, s=80,
                      edgecolors='white', linewidth=0.5)

    # Plot Pareto frontier
    if pareto_points:
        pareto_array = np.array(pareto_points)
        # Sort by FID for line plotting
        sort_idx = np.argsort(pareto_array[:, 0])
        pareto_sorted = pareto_array[sort_idx]
        ax.plot(pareto_sorted[:, 0], pareto_sorted[:, 1],
                'k--', alpha=0.7, linewidth=2, label='Pareto frontier')

        # Highlight Pareto optimal points
        ax.scatter(pareto_array[:, 0], pareto_array[:, 1],
                  c='red', s=150, marker='*', edgecolors='black',
                  linewidth=1.5, label='Pareto optimal', zorder=5)

        # Annotate Pareto names
        for i, (pt, name) in enumerate(zip(pareto_points, pareto_names)):
            short_name = name.replace('Ablation_', '').replace('KMC_', 'KMC_').replace('_', ' ')
            ax.annotate(short_name, (pt[0], pt[1]),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=7, alpha=0.8)

    ax.set_xlabel('FID (↓)', fontsize=10)
    ax.set_ylabel('CLIP Score (↑)', fontsize=10)
    ax.set_title(f'{label}\n({len(pareto_points)} Pareto optimal)',
                fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    if idx == 0:
        ax.legend(loc='best', fontsize=8)

axes[-1].axis('off')
plt.tight_layout()
plt.savefig(out_dir / 'fig2_2d_pareto_frontiers.pdf', dpi=300, bbox_inches='tight')
plt.savefig(out_dir / 'fig2_2d_pareto_frontiers.png', dpi=300, bbox_inches='tight')
plt.close()

print("[OK] Generated Figure 2: 2D Pareto Frontiers (FID vs CLIP)")

# ==================== FIGURE 3: Bootstrap Confidence Intervals ====================
def bootstrap_pareto_ci(data_df, n_bootstrap=1000, ci=0.95):
    """Compute bootstrap confidence intervals for Pareto frontier"""
    rng = np.random.default_rng(42)
    indices = rng.integers(0, len(data_df), size=(n_bootstrap, len(data_df)))

    pareto_curves = []
    for idx_sample in indices:
        sample = data_df.iloc[idx_sample]
        points = sample[['fid', 'clip_score']].dropna().values

        # Find Pareto optimal
        pareto = []
        for i, (fid, clip) in enumerate(points):
            is_dominated = False
            for j, (ofid, oclip) in enumerate(points):
                if i != j and ofid <= fid and oclip >= clip:
                    if ofid < fid or oclip > clip:
                        is_dominated = True
                        break
            if not is_dominated:
                pareto.append([fid, clip])

        if pareto:
            pareto_curves.append(np.array(pareto))

    return pareto_curves

# Figure 3: Bootstrap CI for a representative regime
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

selected_regimes = [
    ('anime_student', 'D-High'),
    ('anime_student', 'D-Medium'),
    ('dreambooth_mixed', 'D-High'),
    ('wikiart_mixed', 'D-Low')
]

for idx, (dataset, regime) in enumerate(selected_regimes):
    ax = axes[idx]

    subset = df[(df['dataset'] == dataset) & (df['regime'] == regime)]

    # Bootstrap
    bootstrap_curves = bootstrap_pareto_ci(subset, n_bootstrap=500)

    # Plot individual bootstrap curves (translucent)
    for curve in bootstrap_curves[:50]:  # Plot first 50 for clarity
        sort_idx = np.argsort(curve[:, 0])
        ax.plot(curve[sort_idx, 0], curve[sort_idx, 1],
               'b-', alpha=0.05, linewidth=1)

    # Compute mean and CI envelope
    # Interpolate to common FID grid
    fid_grid = np.linspace(subset['fid'].min(), subset['fid'].max(), 50)
    clip_interpolated = []

    for curve in bootstrap_curves:
        if len(curve) > 1:
            sort_idx = np.argsort(curve[:, 0])
            curve_sorted = curve[sort_idx]
            # Interpolate clip values
            clips = np.interp(fid_grid, curve_sorted[:, 0], curve_sorted[:, 1],
                             left=np.nan, right=np.nan)
            clip_interpolated.append(clips)

    if clip_interpolated:
        clip_interpolated = np.array(clip_interpolated)
        mean_clip = np.nanmean(clip_interpolated, axis=0)
        ci_lower = np.nanpercentile(clip_interpolated, 2.5, axis=0)
        ci_upper = np.nanpercentile(clip_interpolated, 97.5, axis=0)

        valid_mask = ~np.isnan(mean_clip)
        ax.plot(fid_grid[valid_mask], mean_clip[valid_mask],
               'b-', linewidth=3, label='Mean Pareto frontier')
        ax.fill_between(fid_grid[valid_mask],
                        ci_lower[valid_mask], ci_upper[valid_mask],
                        alpha=0.3, color='blue', label='95% CI')

    # Plot actual points
    for mtype, color in colors.items():
        mask = subset['method_type'] == mtype
        if mask.any():
            ax.scatter(subset[mask]['fid'], subset[mask]['clip_score'],
                      c=color, label=mtype, alpha=0.7, s=80,
                      edgecolors='white', linewidth=1)

    ax.set_xlabel('FID (lower is better)', fontsize=10)
    ax.set_ylabel('CLIP Score (higher is better)', fontsize=10)
    ax.set_title(f'{dataset.replace("_", " ").title()} - {regime}\n'
                f'Bootstrap: 500 samples', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    if idx == 0:
        ax.legend(loc='best', fontsize=8)

plt.tight_layout()
plt.savefig(out_dir / 'fig3_bootstrap_ci.pdf', dpi=300, bbox_inches='tight')
plt.savefig(out_dir / 'fig3_bootstrap_ci.png', dpi=300, bbox_inches='tight')
plt.close()

print("[OK] Generated Figure 3: Bootstrap Confidence Intervals")

# ==================== FIGURE 4: Metric Disagreement Analysis ====================
# Show where FID and CLIP disagree on "best strategy"

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

disagreement_stats = []

for idx, (dataset, label) in enumerate(zip(datasets, dataset_labels)):
    ax = axes[idx]
    ds_data = df[df['dataset'] == dataset]

    # For each regime, find best by FID and best by CLIP
    for regime in ['D-High', 'D-Medium', 'D-Low', 'D-Sub-50', 'D-Sub-25']:
        regime_data = ds_data[ds_data['regime'] == regime]

        if len(regime_data) > 1:
            best_fid = regime_data.loc[regime_data['fid'].idxmin()]
            best_clip = regime_data.loc[regime_data['clip_score'].idxmax()]

            # Check if they disagree
            if best_fid['experiment'] != best_clip['experiment']:
                disagreement_stats.append({
                    'dataset': dataset,
                    'regime': regime,
                    'best_fid_method': best_fid['experiment'],
                    'best_clip_method': best_clip['experiment'],
                    'fid_winner_fid': best_fid['fid'],
                    'fid_winner_clip': best_fid['clip_score'],
                    'clip_winner_fid': best_clip['fid'],
                    'clip_winner_clip': best_clip['clip_score']
                })

    # Plot to visualize disagreement
    for _, row in ds_data.iterrows():
        is_best_fid = (row['fid'] == ds_data['fid'].min()) or (row['fid'] < ds_data['fid'].quantile(0.1))
        is_best_clip = (row['clip_score'] == ds_data['clip_score'].max()) or (row['clip_score'] > ds_data['clip_score'].quantile(0.9))

        if is_best_fid and is_best_clip:
            marker, size, alpha = 'o', 200, 1.0  # Agree on top
            color = 'green'
        elif is_best_fid:
            marker, size, alpha = 'v', 150, 0.8  # Best FID only
            color = 'blue'
        elif is_best_clip:
            marker, size, alpha = '^', 150, 0.8  # Best CLIP only
            color = 'red'
        else:
            marker, size, alpha = '.', 50, 0.3
            color = 'gray'

        ax.scatter(row['fid'], row['clip_score'], c=color, marker=marker,
                  s=size, alpha=alpha, edgecolors='black' if size > 100 else None)

    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green',
               markersize=10, label='Top by both metrics'),
        Line2D([0], [0], marker='v', color='w', markerfacecolor='blue',
               markersize=10, label='Top by FID only'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='red',
               markersize=10, label='Top by CLIP only'),
        Line2D([0], [0], marker='.', color='gray', markersize=5,
               label='Other methods', linestyle='None')
    ]

    ax.set_xlabel('FID (lower is better)', fontsize=11)
    ax.set_ylabel('CLIP Score (higher is better)', fontsize=11)
    ax.set_title(label, fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(handles=legend_elements, loc='best', fontsize=9)

plt.tight_layout()
plt.savefig(out_dir / 'fig4_metric_disagreement.pdf', dpi=300, bbox_inches='tight')
plt.savefig(out_dir / 'fig4_metric_disagreement.png', dpi=300, bbox_inches='tight')
plt.close()

print("[OK] Generated Figure 4: Metric Disagreement Analysis")

# ==================== Summary Statistics ====================
# Generate comprehensive summary with std and correlation info
summary = []

for dataset in datasets:
    for regime in regimes:
        subset = df[(df['dataset'] == dataset) & (df['regime'] == regime)]
        if len(subset) > 0:
            # Find statistics
            summary.append({
                'dataset': dataset,
                'regime': regime,
                'n_methods': len(subset),
                'fid_range': f"[{subset['fid'].min():.1f}, {subset['fid'].max():.1f}]",
                'clip_range': f"[{subset['clip_score'].min():.3f}, {subset['clip_score'].max():.3f}]",
                'best_fid': subset.loc[subset['fid'].idxmin(), 'experiment'],
                'best_clip': subset.loc[subset['clip_score'].idxmax(), 'experiment'],
                'correlation': subset[['fid', 'clip_score']].corr().iloc[0,1]
            })

summary_df = pd.DataFrame(summary)
summary_df.to_csv(out_dir / 'analysis_summary.csv', index=False)

# Disagreement summary
disagreement_df = pd.DataFrame(disagreement_stats) if disagreement_stats else pd.DataFrame()
if not disagreement_df.empty:
    disagreement_df.to_csv(out_dir / 'metric_disagreement.csv', index=False)

print("\n=== Summary ===")
print(f"Generated {out_dir}/ with:")
print(f"  - FID-CLIP correlation by dataset: {len(fid_clip_correlation)} correlations")
print(f"  - Metric disagreement cases: {len(disagreement_stats)} cases")
print(f"  - 2D Pareto frontiers for {len(regimes)} regimes")
print(f"  - Bootstrap CI for 4 representative scenarios")

# Print key insights
print("\n=== Key Findings ===")
print("\nFID-CLIP Correlations:")
for corr in fid_clip_correlation:
    print(f"  {corr['dataset']}: r={corr['correlation']:.3f}, p={corr['p_value']:.3f}")

print(f"\nMetric Disagreement: {len(disagreement_stats)} out of {len(summary)} regime-dataset pairs")
print("(FID and CLIP select different 'best' methods)")
