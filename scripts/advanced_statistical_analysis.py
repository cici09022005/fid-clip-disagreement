"""
Advanced statistical analysis with Spearman correlation, 95% CI, and Simpson's paradox detection
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import spearmanr, bootstrap
from pathlib import Path

# Read data
df = pd.read_csv('kmc_lora/results/all_eval_results.csv')

# Define categories
df['method_type'] = df['experiment'].apply(lambda x:
    'Random' if 'Random' in x else
    'KMC' if 'KMC' in x else
    'Ablation' if 'Ablation' in x else
    'Quality' if 'Quality' in x else
    'Anti' if 'Anti' in x else 'Other'
)

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

# Output directory
out_dir = Path('kmc_lora/figures/advanced_stats')
out_dir.mkdir(parents=True, exist_ok=True)

# ==================== Spearman Correlation with Confidence Intervals ====================

def bootstrap_correlation_ci(x, y, n_bootstrap=5000, method='spearman'):
    """Calculate bootstrap 95% CI for correlation coefficient"""
    if len(x) < 3:
        return np.nan, np.nan, np.nan, np.nan

    def corr_func(x, y):
        if method == 'spearman':
            return spearmanr(x, y)[0]
        else:
            return stats.pearsonr(x, y)[0]

    # Bootstrap
    rng = np.random.default_rng(42)
    indices = rng.integers(0, len(x), size=(n_bootstrap, len(x)))
    corrs = []

    for idx in indices:
        x_boot = x.iloc[idx] if hasattr(x, 'iloc') else x[idx]
        y_boot = y.iloc[idx] if hasattr(y, 'iloc') else y[idx]
        try:
            corr = corr_func(x_boot, y_boot)
            if not np.isnan(corr):
                corrs.append(corr)
        except:
            pass

    if len(corrs) < 100:
        return np.nan, np.nan, np.nan, np.nan

    corrs = np.array(corrs)
    point_est = corr_func(x, y)
    ci_low = np.percentile(corrs, 2.5)
    ci_high = np.percentile(corrs, 97.5)
    std = np.std(corrs)

    return point_est, ci_low, ci_high, std

# Calculate correlations for each dataset
datasets = ['anime_student', 'wikiart_mixed', 'dreambooth_mixed', 'dreambooth_single']
results_spearman = []
results_pearson = []

print("=" * 80)
print("CORRELATION ANALYSIS: SPEARMAN vs PEARSON with 95% CI")
print("=" * 80)

for dataset in datasets:
    ds_data = df[(df['dataset'] == dataset) & (df['fid'].notna()) & (df['clip_score'].notna())]

    if len(ds_data) < 3:
        continue

    # Spearman
    rho, rho_low, rho_high, rho_std = bootstrap_correlation_ci(
        ds_data['fid'], ds_data['clip_score'], method='spearman'
    )

    # Pearson
    r, r_low, r_high, r_std = bootstrap_correlation_ci(
        ds_data['fid'], ds_data['clip_score'], method='pearson'
    )

    # Traditional test
    spearman_stat, spearman_p = spearmanr(ds_data['fid'], ds_data['clip_score'])
    pearson_stat, pearson_p = stats.pearsonr(ds_data['fid'], ds_data['clip_score'])

    results_spearman.append({
        'dataset': dataset,
        'rho': rho,
        'rho_ci_low': rho_low,
        'rho_ci_high': rho_high,
        'rho_std': rho_std,
        'p_value': spearman_p,
        'n': len(ds_data)
    })

    results_pearson.append({
        'dataset': dataset,
        'r': r,
        'r_ci_low': r_low,
        'r_ci_high': r_high,
        'r_std': r_std,
        'p_value': pearson_p,
        'n': len(ds_data)
    })

    print(f"\n{dataset.upper()}:")
    print(f"  Sample size: N={len(ds_data)}")
    print(f"  Spearman: ρ={rho:.3f}, 95% CI=[{rho_low:.3f}, {rho_high:.3f}], p={spearman_p:.4f}")
    print(f"  Pearson:  r={r:.3f}, 95% CI=[{r_low:.3f}, {r_high:.3f}], p={pearson_p:.4f}")
    print(f"  CI width: Spearman={rho_high-rho_low:.3f}, Pearson={r_high-r_low:.3f}")

# Save results
spearman_df = pd.DataFrame(results_spearman)
pearson_df = pd.DataFrame(results_pearson)
spearman_df.to_csv(out_dir / 'spearman_correlation_ci.csv', index=False)
pearson_df.to_csv(out_dir / 'pearson_correlation_ci.csv', index=False)

# ==================== SIMPSON'S PARADOX DETECTION ====================

print("\n" + "=" * 80)
print("SIMPSON'S PARADOX ANALYSIS: WITHIN-REGIME vs CROSS-REGIME")
print("=" * 80)

simpson_results = []

for dataset in datasets:
    print(f"\n{dataset.upper()}:")
    ds_data = df[(df['dataset'] == dataset) & (df['fid'].notna()) & (df['clip_score'].notna())]

    # Overall correlation (what we calculated above)
    overall_rho, _ = spearmanr(ds_data['fid'], ds_data['clip_score'])

    # Within-regime correlations
    regime_corrs = []
    for regime in ['D-High', 'D-Medium', 'D-Low', 'D-Sub-50', 'D-Sub-25']:
        reg_data = ds_data[ds_data['regime'] == regime]
        if len(reg_data) >= 3:  # Need at least 3 points
            rho, p = spearmanr(reg_data['fid'], reg_data['clip_score'])
            regime_corrs.append({
                'regime': regime,
                'rho': rho,
                'p': p,
                'n': len(reg_data)
            })
            print(f"  {regime}: ρ={rho:.3f}, p={p:.3f}, N={len(reg_data)}")

    # Check for Simpson's paradox
    if regime_corrs:
        avg_within_rho = np.mean([r['rho'] for r in regime_corrs])
    else:
        avg_within_rho = np.nan

    simpson_detected = (overall_rho > 0 and avg_within_rho < 0) or (overall_rho < 0 and avg_within_rho > 0)

    simpson_results.append({
        'dataset': dataset,
        'overall_rho': overall_rho,
        'avg_within_regime_rho': avg_within_rho,
        'simpson_detected': simpson_detected,
        'n_regimes': len(regime_corrs)
    })

    print(f"  Overall: ρ={overall_rho:.3f}")
    print(f"  Avg within-regime: ρ={avg_within_rho:.3f}")
    print(f"  Simpson's paradox: {'DETECTED' if simpson_detected else 'NOT DETECTED'}")

simpson_df = pd.DataFrame(simpson_results)
simpson_df.to_csv(out_dir / 'simpson_paradox_analysis.csv', index=False)

# ==================== VISUALIZATION: Correlation Comparison ====================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Spearman vs Pearson
ax = axes[0]
x = np.arange(len(datasets))
width = 0.35

spearman_rhos = [r['rho'] for r in results_spearman]
pearson_rs = [r['r'] for r in results_pearson]
spearman_cis = [(r['rho']-r['rho_ci_low'], r['rho_ci_high']-r['rho']) for r in results_spearman]
pearson_cis = [(r['r']-r['r_ci_low'], r['r_ci_high']-r['r']) for r in results_pearson]

bars1 = ax.bar(x - width/2, spearman_rhos, width, yerr=list(zip(*spearman_cis)),
               label='Spearman (robust)', capsize=5, color='#2196F3', alpha=0.8)
bars2 = ax.bar(x + width/2, pearson_rs, width, yerr=list(zip(*pearson_cis)),
               label='Pearson (original)', capsize=5, color='#FF9800', alpha=0.8)

ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.set_ylabel('Correlation Coefficient', fontsize=11)
ax.set_xlabel('Dataset', fontsize=11)
ax.set_title('Spearman vs Pearson Correlation\n(with 95% Bootstrap CI)', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([d.replace('_', '\n') for d in datasets], fontsize=9)
ax.legend(loc='best')
ax.grid(True, alpha=0.3, axis='y')

# Add CI width annotation
for i, (s_r, p_r) in enumerate(zip(results_spearman, results_pearson)):
    s_width = s_r['rho_ci_high'] - s_r['rho_ci_low']
    p_width = p_r['r_ci_high'] - p_r['r_ci_low']
    ax.annotate(f"CI:{s_width:.2f}", (i-width/2, s_r['rho_ci_high']+0.05), ha='center', fontsize=7, color='blue')
    ax.annotate(f"CI:{p_width:.2f}", (i+width/2, p_r['r_ci_high']+0.05), ha='center', fontsize=7, color='orange')

# Plot 2: Simpson's Paradox Visualization
ax = axes[1]

for i, dataset in enumerate(datasets):
    sr = simpson_results[i]
    x_vals = [0, 1]
    y_vals = [sr['avg_within_regime_rho'], sr['overall_rho']]
    color = 'red' if sr['simpson_detected'] else 'gray'
    ax.plot(x_vals, y_vals, 'o-', color=color, linewidth=2, markersize=8,
            label=dataset if sr['simpson_detected'] else None)

ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.set_xticks([0, 1])
ax.set_xticklabels(['Avg Within-Regime', 'Overall (Cross-Regime)'])
ax.set_ylabel('Spearman ρ', fontsize=11)
ax.set_title("Simpson's Paradox Detection\n(Red lines indicate paradox)", fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
if any(s['simpson_detected'] for s in simpson_results):
    ax.legend(loc='best', fontsize=9, title='Simpson Detected')

plt.tight_layout()
plt.savefig(out_dir / 'fig1_correlation_comparison.pdf', dpi=300, bbox_inches='tight')
plt.savefig(out_dir / 'fig1_correlation_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"\n[OK] Generated correlation comparison figure")

# ==================== DETAILED CORRELATION BY REGIME ====================

print("\n" + "=" * 80)
print("DETAILED WITHIN-REGIME CORRELATION MATRIX")
print("=" * 80)

# Create matrix for visualization
regimes = ['D-High', 'D-Medium', 'D-Low', 'D-Sub-50', 'D-Sub-25']
corr_matrix = np.full((len(datasets), len(regimes)), np.nan)
p_matrix = np.full((len(datasets), len(regimes)), np.nan)
n_matrix = np.full((len(datasets), len(regimes)), 0)

for i, dataset in enumerate(datasets):
    for j, regime in enumerate(regimes):
        reg_data = df[(df['dataset'] == dataset) & (df['regime'] == regime) &
                      (df['fid'].notna()) & (df['clip_score'].notna())]
        if len(reg_data) >= 3:
            rho, p = spearmanr(reg_data['fid'], reg_data['clip_score'])
            corr_matrix[i, j] = rho
            p_matrix[i, j] = p
            n_matrix[i, j] = len(reg_data)

# Visualize heatmap
fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)

ax.set_xticks(np.arange(len(regimes)))
ax.set_yticks(np.arange(len(datasets)))
ax.set_xticklabels(regimes)
ax.set_yticklabels([d.replace('_', ' ') for d in datasets])

# Annotate with values
for i in range(len(datasets)):
    for j in range(len(regimes)):
        if not np.isnan(corr_matrix[i, j]):
            text = f"ρ={corr_matrix[i, j]:.2f}\nn={int(n_matrix[i, j])}"
            color = 'white' if abs(corr_matrix[i, j]) > 0.5 else 'black'
            ax.text(j, i, text, ha='center', va='center', fontsize=9, color=color)

ax.set_xlabel('Regime', fontsize=11)
ax.set_ylabel('Dataset', fontsize=11)
ax.set_title('Spearman ρ (FID vs CLIP) by Dataset and Regime\n(with sample size)', fontsize=12, fontweight='bold')
plt.colorbar(im, ax=ax, label='Spearman ρ')

plt.tight_layout()
plt.savefig(out_dir / 'fig2_regime_correlation_heatmap.pdf', dpi=300, bbox_inches='tight')
plt.savefig(out_dir / 'fig2_regime_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"[OK] Generated regime correlation heatmap")

# Save detailed results
detailed_results = []
for i, dataset in enumerate(datasets):
    for j, regime in enumerate(regimes):
        if not np.isnan(corr_matrix[i, j]):
            detailed_results.append({
                'dataset': dataset,
                'regime': regime,
                'spearman_rho': corr_matrix[i, j],
                'p_value': p_matrix[i, j],
                'n': int(n_matrix[i, j])
            })

detailed_df = pd.DataFrame(detailed_results)
detailed_df.to_csv(out_dir / 'within_regime_correlations.csv', index=False)

print("\n" + "=" * 80)
print("SUMMARY OF KEY FINDINGS")
print("=" * 80)
print(f"\n1. Correlation Method Comparison:")
for ds, sp, pe in zip(datasets, results_spearman, results_pearson):
    print(f"   {ds}: Spearman ρ={sp['rho']:.3f} (CI: {sp['rho_ci_low']:.3f} to {sp['rho_ci_high']:.3f})")
    print(f"           Pearson r={pe['r']:.3f} (CI: {pe['r_ci_low']:.3f} to {pe['r_ci_high']:.3f})")
    print(f"           CI width: Spearman {sp['rho_ci_high']-sp['rho_ci_low']:.3f}, Pearson {pe['r_ci_high']-pe['r_ci_low']:.3f}")

print(f"\n2. Simpson's Paradox Detection:")
for s in simpson_results:
    status = "⚠️  DETECTED" if s['simpson_detected'] else "✓ Not detected"
    print(f"   {s['dataset']}: {status}")
    print(f"      Overall ρ={s['overall_rho']:.3f}, Avg within-regime ρ={s['avg_within_regime_rho']:.3f}")

print(f"\n3. Within-Regime Correlation Valid Cases:")
for ds, row in zip(datasets, corr_matrix):
    valid = np.sum(~np.isnan(row))
    print(f"   {ds}: {valid}/5 regimes have sufficient data (n≥3)")

print(f"\n[OK] All analysis saved to {out_dir}/")
