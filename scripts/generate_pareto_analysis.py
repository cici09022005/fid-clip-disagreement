"""
Generate Pareto frontier analysis for efficiency-performance trade-off.

This version reads training time from experiment outputs instead of
using a hand-written lookup table.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Read the comprehensive results
df = pd.read_csv('kmc_lora/results/all_eval_results.csv')

# Define method categories
df['is_random'] = df['experiment'].str.startswith('Random')
df['is_kmc'] = df['experiment'].str.startswith('KMC')
df['is_ablation'] = df['experiment'].str.startswith('Ablation')
df['is_quality'] = df['experiment'].str.startswith('Quality')
df['is_anti'] = df['experiment'].str.startswith('Anti')

# Extract regime info
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

def experiment_root(dataset):
    if dataset == 'anime_student':
        return Path('kmc_lora/results')
    return Path('kmc_lora/results') / dataset


def load_json(path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return None


def read_training_time_min(dataset, experiment):
    exp_dir = experiment_root(dataset) / experiment
    if not exp_dir.exists():
        return np.nan

    direct_summary = exp_dir / 'training_summary.json'
    if direct_summary.exists():
        data = load_json(direct_summary)
        if data and data.get('training_time_min') is not None:
            return float(data['training_time_min'])

    phase_summaries = sorted(exp_dir.glob('phase*/training_summary.json'))
    if phase_summaries:
        total = 0.0
        found = False
        for summary_path in phase_summaries:
            data = load_json(summary_path)
            if data and data.get('training_time_min') is not None:
                total += float(data['training_time_min'])
                found = True
        if found:
            return total

    return np.nan


# Map training times from real summaries
df['training_time'] = df.apply(
    lambda row: read_training_time_min(row['dataset'], row['experiment']),
    axis=1
)

# Create output directory
out_dir = Path('kmc_lora/figures/pareto_analysis')
out_dir.mkdir(parents=True, exist_ok=True)

# Define colors and markers
colors = {
    'Random': '#FF9800',
    'KMC': '#2196F3',
    'Ablation': '#9C27B0',
    'Quality': '#00BCD4',
    'Anti': '#E91E63'
}

# Map internal experiment prefixes to display labels
DISPLAY_LABELS = {
    'Random': 'Random',
    'KMC': 'KMC',
    'Ablation': 'Ablation',
    'Quality': 'Quality',
    'Anti': 'Anti',
}

# Figure 1: Overall Pareto Frontier (all datasets combined)
fig, ax = plt.subplots(figsize=(10, 7))

# Filter valid data
valid_df = df[df['training_time'].notna() & df['fid'].notna()].copy()

# Plot each method type
EXP_PREFIX_TO_COLOR = {'Random': colors['Random'], 'KMC': colors['KMC'],
                       'Ablation': colors['Ablation'], 'Quality': colors['Quality'],
                       'Anti': colors['Anti']}
for method_type, color in EXP_PREFIX_TO_COLOR.items():
    mask = valid_df['experiment'].str.startswith(method_type)
    if method_type == 'KMC':
        mask = valid_df['is_kmc']
    elif method_type == 'Random':
        mask = valid_df['is_random']

    subset = valid_df[mask]
    if len(subset) > 0:
        display_label = DISPLAY_LABELS.get(method_type, method_type)
        ax.scatter(subset['training_time'], subset['fid'],
                  c=color, label=display_label, alpha=0.6, s=80, edgecolors='white', linewidth=0.5)

ax.set_xlabel('Training Time (minutes)', fontsize=12)
ax.set_ylabel('FID (lower is better)', fontsize=12)
ax.set_title('Efficiency-Performance Trade-off: All Methods', fontsize=14, fontweight='bold')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
ax.invert_yaxis()  # Lower FID is better

plt.tight_layout()
plt.savefig(out_dir / 'fig1_overall_pareto.pdf', dpi=300, bbox_inches='tight')
plt.savefig(out_dir / 'fig1_overall_pareto.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"[OK] Generated {out_dir / 'fig1_overall_pareto.pdf'}")

# Figure 2: Per-regime Pareto Frontiers
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

regimes = ['D-High', 'D-Medium', 'D-Low', 'D-Sub-50', 'D-Sub-25']
regime_labels = ['High Diversity\n(Full Data)', 'Medium Diversity\n(90%)', 'Low Diversity\n(Largest Cluster)',
                 'Sub-sampled 50%', 'Sub-sampled 25%']

for idx, (regime, label) in enumerate(zip(regimes, regime_labels)):
    ax = axes[idx]
    regime_data = valid_df[valid_df['regime'] == regime]

    # Plot different method types
    method_masks = {
        'Random': regime_data['is_random'],
        'KMC': regime_data['is_kmc'],
        'Ablation': regime_data['is_ablation'],
        'Baseline': regime_data['is_quality'] | regime_data['is_anti']
    }

    for method_name, mask in method_masks.items():
        subset = regime_data[mask]
        if len(subset) > 0:
            color = colors.get(method_name, '#757575')
            if method_name == 'Baseline':
                color = '#757575'
            ax.scatter(subset['training_time'], subset['fid'],
                      c=color, label=method_name, alpha=0.7, s=100, edgecolors='white', linewidth=1)

    # Compute and plot Pareto frontier
    regime_points = regime_data[['training_time', 'fid']].values
    if len(regime_points) > 0:
        # Find Pareto optimal points (minimize both time and FID)
        pareto_points = []
        for i, (t1, f1) in enumerate(regime_points):
            is_dominated = False
            for j, (t2, f2) in enumerate(regime_points):
                if i != j and t2 <= t1 and f2 <= f1 and (t2 < t1 or f2 < f1):
                    is_dominated = True
                    break
            if not is_dominated:
                pareto_points.append((t1, f1))

        if pareto_points:
            pareto_points = sorted(pareto_points, key=lambda x: x[0])
            pareto_x, pareto_y = zip(*pareto_points)
            ax.plot(pareto_x, pareto_y, 'k--', alpha=0.5, linewidth=2, label='Pareto frontier')

    ax.set_xlabel('Time (min)', fontsize=10)
    ax.set_ylabel('FID', fontsize=10)
    ax.set_title(label, fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    if idx == 0:
        ax.legend(loc='upper right', fontsize=8)

# Remove empty subplot
axes[-1].axis('off')

plt.tight_layout()
plt.savefig(out_dir / 'fig2_per_regime_pareto.pdf', dpi=300, bbox_inches='tight')
plt.savefig(out_dir / 'fig2_per_regime_pareto.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"[OK] Generated {out_dir / 'fig2_per_regime_pareto.pdf'}")

# Figure 3: Efficiency Score Analysis
# Define efficiency score: FID improvement per minute
def compute_efficiency_score(row):
    """Compute relative efficiency compared to random baseline in same regime"""
    dataset = row['dataset']
    regime = row['regime']

    # Get random baseline for this dataset/regime
    random_baseline = valid_df[(valid_df['dataset'] == dataset) &
                                (valid_df['regime'] == regime) &
                                (valid_df['is_random'])]

    if len(random_baseline) == 0:
        return np.nan

    random_fid = random_baseline['fid'].mean()
    random_time = random_baseline['training_time'].mean()

    # Efficiency = (FID improvement) / (time ratio)
    fid_improvement = random_fid - row['fid']
    time_ratio = row['training_time'] / random_time if random_time > 0 else 1

    # Normalized efficiency score
    if time_ratio > 0:
        return fid_improvement / time_ratio
    return np.nan

valid_df['efficiency_score'] = valid_df.apply(compute_efficiency_score, axis=1)

fig, ax = plt.subplots(figsize=(12, 6))

# Group by method type and regime
method_names = []
regimes_list = []
scores = []

for method_type in ['KMC', 'Ablation_NoPhase1', 'Ablation_NoPhase2', 'Ablation_NoPhase3', 'Quality_Filter', 'Anti_Curriculum']:  # KMC is experiment prefix, display as KMC
    for regime in regimes:
        subset = valid_df[(valid_df['experiment'].str.startswith(method_type)) &
                          (valid_df['regime'] == regime)]
        if len(subset) > 0 and subset['efficiency_score'].notna().any():
            display_name = DISPLAY_LABELS.get(method_type, method_type).replace('Ablation_', '')
            method_names.append(display_name)
            regimes_list.append(regime)
            scores.append(subset['efficiency_score'].mean())

# Create bar plot
x_labels = [f"{m}\n{r}" for m, r in zip(method_names, regimes_list)]
x_pos = np.arange(len(x_labels))

colors_bar = ['#4CAF50' if s > 0 else '#F44336' for s in scores]
bars = ax.bar(x_pos, scores, color=colors_bar, alpha=0.7, edgecolor='white')

ax.set_xlabel('Method & Regime', fontsize=11)
ax.set_ylabel('Efficiency Score (ΔFID / Time Ratio)', fontsize=11)
ax.set_title('Training Efficiency Across Methods and Regimes\n(Positive = Better than Random Baseline)',
             fontsize=13, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(out_dir / 'fig3_efficiency_scores.pdf', dpi=300, bbox_inches='tight')
plt.savefig(out_dir / 'fig3_efficiency_scores.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"[OK] Generated {out_dir / 'fig3_efficiency_scores.pdf'}")

# Figure 4: Dataset-Specific Analysis
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

datasets = ['anime_student', 'wikiart_mixed', 'dreambooth_mixed', 'dreambooth_single']
dataset_labels = ['Anime-Student', 'WikiArt-Mixed', 'DreamBooth-Mixed', 'DreamBooth-Single']

for idx, (dataset, label) in enumerate(zip(datasets, dataset_labels)):
    ax = axes[idx]
    ds_data = valid_df[valid_df['dataset'] == dataset]

    # Color by regime
    regime_colors = {'D-High': '#2196F3', 'D-Medium': '#4CAF50', 'D-Low': '#FF9800',
                     'D-Sub-50': '#9C27B0', 'D-Sub-25': '#F44336'}

    for regime in regimes:
        regime_data = ds_data[ds_data['regime'] == regime]
        if len(regime_data) > 0:
            ax.scatter(regime_data['training_time'], regime_data['fid'],
                      c=regime_colors.get(regime, '#757575'),
                      label=regime, alpha=0.7, s=100, edgecolors='white', linewidth=1)

    ax.set_xlabel('Training Time (min)', fontsize=10)
    ax.set_ylabel('FID', fontsize=10)
    ax.set_title(label, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    if idx == 0:
        ax.legend(loc='upper right', fontsize=8)

plt.tight_layout()
plt.savefig(out_dir / 'fig4_dataset_comparison.pdf', dpi=300, bbox_inches='tight')
plt.savefig(out_dir / 'fig4_dataset_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"[OK] Generated {out_dir / 'fig4_dataset_comparison.pdf'}")

# Figure 5: True 3D trade-off view (Time, FID, CLIP)
fig = plt.figure(figsize=(11, 8))
ax = fig.add_subplot(111, projection='3d')

for method_type, color in EXP_PREFIX_TO_COLOR.items():
    mask = valid_df['experiment'].str.startswith(method_type)
    if method_type == 'KMC':
        mask = valid_df['is_kmc']
    elif method_type == 'Random':
        mask = valid_df['is_random']

    subset = valid_df[mask & valid_df['clip_score'].notna()]
    if len(subset) > 0:
        display_label = DISPLAY_LABELS.get(method_type, method_type)
        ax.scatter(
            subset['training_time'],
            subset['fid'],
            subset['clip_score'],
            c=color,
            label=display_label,
            alpha=0.75,
            s=55,
            edgecolors='white',
            linewidth=0.4
        )

ax.set_xlabel('Training Time (min)', fontsize=11, labelpad=10)
ax.set_ylabel('FID', fontsize=11, labelpad=10)
ax.set_zlabel('CLIP Score', fontsize=11, labelpad=10)
ax.set_title('3D Trade-off: Time vs FID vs CLIP', fontsize=14, fontweight='bold', pad=18)
ax.view_init(elev=22, azim=-55)
ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98))

plt.tight_layout()
plt.savefig(out_dir / 'fig5_3d_tradeoff.pdf', dpi=300, bbox_inches='tight')
plt.savefig(out_dir / 'fig5_3d_tradeoff.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"[OK] Generated {out_dir / 'fig5_3d_tradeoff.pdf'}")

# Generate summary statistics table
summary_stats = []

for dataset in datasets:
    for regime in regimes:
        subset = valid_df[(valid_df['dataset'] == dataset) & (valid_df['regime'] == regime)]
        if len(subset) > 0:
            # Find Pareto optimal points
            best_fid = subset.loc[subset['fid'].idxmin()]
            fastest = subset.loc[subset['training_time'].idxmin()]

            # Find best efficiency
            valid_eff = subset[subset['efficiency_score'].notna()]
            if len(valid_eff) > 0:
                best_eff = valid_eff.loc[valid_eff['efficiency_score'].idxmax()]
            else:
                best_eff = None

            summary_stats.append({
                'Dataset': dataset,
                'Regime': regime,
                'Best FID': f"{best_fid['fid']:.2f} ({best_fid['experiment']})",
                'Fastest': f"{fastest['training_time']:.1f}min ({fastest['experiment']})",
                'Best Efficiency': f"{best_eff['efficiency_score']:.2f} ({best_eff['experiment']})" if best_eff is not None else 'N/A'
            })

summary_df = pd.DataFrame(summary_stats)
summary_df.to_csv(out_dir / 'pareto_summary.csv', index=False)
print(f"[OK] Generated summary table: {out_dir / 'pareto_summary.csv'}")

print("\n=== Pareto Analysis Complete ===")
print(f"Output directory: {out_dir}")
print(f"Total experiments analyzed: {len(valid_df)}")
