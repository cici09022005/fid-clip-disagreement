"""
Generate all paper-ready figures for KMC-LoRA.
Reads training results and evaluation metrics, produces:
  1. fig1_loss_curves.pdf       - Training loss curves (KMC vs Random)
  2. fig2_fid_comparison.pdf    - FID bar chart across methods
  3. fig3_clip_comparison.pdf   - CLIP Score bar chart
  4. fig4_ablation.pdf          - Ablation study bar chart
  5. fig5_cross_dataset.pdf     - Cross-dataset comparison
  6. fig6_convergence.pdf       - Loss convergence comparison (all datasets)
  7. fig7_phase_loss.pdf        - Phase-wise loss decomposition
  8. fig8_diversity_vs_fid.pdf  - Diversity level vs FID scatter

Usage:
  python kmc_lora/scripts/generate_figures.py --out-dir kmc_lora/figures
"""
import argparse, json, os, sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Global style ──
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# ── Color scheme (colorblind-friendly) ──
COLORS = {
    'KMC': '#2196F3',       # Blue
    'Random': '#FF9800',    # Orange
    'NoPhase1': '#4CAF50',  # Green
    'NoPhase2': '#9C27B0',  # Purple
    'NoPhase3': '#F44336',  # Red
    'Phase1Only': '#795548', # Brown
    'Phase3Only': '#607D8B', # Blue-grey
    'Quality': '#00BCD4',   # Cyan
    'Anti': '#E91E63',      # Pink
    'phase1': '#66BB6A',
    'phase2': '#42A5F5',
    'phase3': '#EF5350',
}

DATASETS_DISPLAY = {
    'anime_student': 'Anime-Student',
    'wikiart_mixed': 'WikiArt-Mixed',
    'dreambooth_mixed': 'DreamBooth-Mixed',
    'dreambooth_single': 'DreamBooth-Single',
}

SPLITS_SHORT = {
    'D-High': '100%',
    'D-Medium': '90%',
    'D-Low': 'Largest',
    'D-Sub-50': '50%',
    'D-Sub-25': '25%',
}


# ─────────────────────────────────────────────────────────────────
# Data loading helpers
# ─────────────────────────────────────────────────────────────────

def load_loss_logs(results_dir, experiments):
    """Load loss_log.csv for given list of experiments."""
    data = {}
    rd = Path(results_dir)
    for exp in experiments:
        # Single-phase experiments
        p = rd / exp / 'loss_log.csv'
        if p.exists():
            df = pd.read_csv(p)
            data[exp] = df
            continue
        # Multi-phase: concatenate
        frames = []
        cumulative_steps = 0
        for phase in ['phase1', 'phase2', 'phase3']:
            pp = rd / exp / phase / 'loss_log.csv'
            if pp.exists():
                df = pd.read_csv(pp)
                df['step'] = df['step'] + cumulative_steps
                df['phase'] = phase
                cumulative_steps = df['step'].max()
                frames.append(df)
        if frames:
            data[exp] = pd.concat(frames, ignore_index=True)
    return data


def load_training_summaries(results_dir, experiments):
    """Load training_summary.json for experiments."""
    data = {}
    rd = Path(results_dir)
    for exp in experiments:
        p = rd / exp / 'training_summary.json'
        if p.exists():
            with open(p) as f:
                data[exp] = json.load(f)
            continue
        # Multi-phase: load last phase
        for phase in ['phase3', 'phase2', 'phase1']:
            p = rd / exp / phase / 'training_summary.json'
            if p.exists():
                with open(p) as f:
                    data[exp] = json.load(f)
                break
    return data


def load_eval_metrics(results_dir, experiments):
    """Load eval_metrics.json for experiments."""
    data = {}
    rd = Path(results_dir)
    for exp in experiments:
        # Check generated/ subdir first
        p = rd / exp / 'generated' / 'eval_metrics.json'
        if p.exists():
            with open(p) as f:
                data[exp] = json.load(f)
            continue
        # Also check direct
        p = rd / exp / 'eval_metrics.json'
        if p.exists():
            with open(p) as f:
                data[exp] = json.load(f)
    return data


def load_phase_summaries(results_dir, exp_name):
    """Load per-phase training summary for an KMC experiment."""
    rd = Path(results_dir)
    phases = {}
    for phase in ['phase1', 'phase2', 'phase3']:
        p = rd / exp_name / phase / 'training_summary.json'
        if p.exists():
            with open(p) as f:
                phases[phase] = json.load(f)
    return phases


# ─────────────────────────────────────────────────────────────────
# Dataset result paths
# ─────────────────────────────────────────────────────────────────

DATASET_RESULTS = {
    'anime_student': 'kmc_lora/results',
    'wikiart_mixed': 'kmc_lora/results/wikiart_mixed',
    'dreambooth_mixed': 'kmc_lora/results/dreambooth_mixed',
    'dreambooth_single': 'kmc_lora/results/dreambooth_single',
}


# ─────────────────────────────────────────────────────────────────
# Figure 1: Training Loss Curves (KMC vs Random) per dataset
# ─────────────────────────────────────────────────────────────────

def fig1_loss_curves(out_dir):
    """2×2 grid: loss curves for D-High split, KMC vs Random, 4 datasets."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.flatten()

    for idx, (ds_key, ds_name) in enumerate(DATASETS_DISPLAY.items()):
        ax = axes[idx]
        rd = DATASET_RESULTS[ds_key]

        # Load Random_D-High and KMC_D-High
        loss_data = load_loss_logs(rd, ['Random_D-High', 'KMC_D-High'])

        if 'Random_D-High' in loss_data:
            df = loss_data['Random_D-High']
            # Smooth with rolling average
            smoothed = df['loss'].rolling(window=20, min_periods=1).mean()
            ax.plot(df['step'], smoothed, color=COLORS['Random'],
                    label='Random', alpha=0.9, linewidth=1.5)

        if 'KMC_D-High' in loss_data:
            df = loss_data['KMC_D-High']
            smoothed = df['loss'].rolling(window=20, min_periods=1).mean()
            ax.plot(df['step'], smoothed, color=COLORS['KMC'],
                    label='KMC', alpha=0.9, linewidth=1.5)

            # Mark phase boundaries
            if 'phase' in df.columns:
                for phase in ['phase1', 'phase2']:
                    phase_end = df[df['phase'] == phase]['step'].max()
                    ax.axvline(x=phase_end, color='gray', linestyle='--',
                              alpha=0.5, linewidth=0.8)

        ax.set_title(ds_name, fontweight='bold')
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Loss')
        ax.legend(loc='upper right', framealpha=0.8)
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    path = Path(out_dir) / 'fig1_loss_curves.pdf'
    fig.savefig(path)
    fig.savefig(path.with_suffix('.png'))
    plt.close(fig)
    print(f"[OK] {path}")


# ─────────────────────────────────────────────────────────────────
# Figure 2: Final Loss Comparison (bar chart)
# ─────────────────────────────────────────────────────────────────

def fig2_final_loss_comparison(out_dir):
    """Grouped bar chart: Final loss, KMC vs Random, per diversity split."""
    splits = ['D-High', 'D-Medium', 'D-Low', 'D-Sub-50', 'D-Sub-25']
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 5.2), sharey=False)
    axes = axes.flatten()

    for idx, (ds_key, ds_name) in enumerate(DATASETS_DISPLAY.items()):
        ax = axes[idx]
        rd = DATASET_RESULTS[ds_key]

        random_losses = []
        kmc_losses = []
        for sp in splits:
            r_summ = load_training_summaries(rd, [f'Random_{sp}'])
            h_summ = load_training_summaries(rd, [f'KMC_{sp}'])
            r_loss = r_summ.get(f'Random_{sp}', {}).get('final_loss', np.nan)
            h_loss = h_summ.get(f'KMC_{sp}', {}).get('final_loss', np.nan)
            random_losses.append(r_loss)
            kmc_losses.append(h_loss)

        x = np.arange(len(splits))
        w = 0.35
        ax.bar(x - w/2, random_losses, w, label='Random',
               color=COLORS['Random'], alpha=0.85, edgecolor='white')
        ax.bar(x + w/2, kmc_losses, w, label='KMC',
               color=COLORS['KMC'], alpha=0.85, edgecolor='white')

        ax.set_title(ds_name, fontweight='bold', fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([SPLITS_SHORT[s] for s in splits], rotation=0)
        ax.set_xlabel('Data Diversity')
        if idx == 0:
            ax.set_ylabel('Final Loss')
        ax.legend(loc='upper right', fontsize=7)

    plt.tight_layout(pad=0.6, w_pad=0.8, h_pad=1.0)
    path = Path(out_dir) / 'fig2_final_loss.pdf'
    fig.savefig(path)
    fig.savefig(path.with_suffix('.png'))
    plt.close(fig)
    print(f"[OK] {path}")


# ─────────────────────────────────────────────────────────────────
# Figure 3: FID Comparison
# ─────────────────────────────────────────────────────────────────

def fig3_fid_comparison(out_dir):
    """Grouped bar chart: FID scores, KMC vs Random vs Baselines."""
    methods = ['Random_D-High', 'KMC_D-High', 'Quality_Filter', 'Anti_Curriculum']
    method_labels = ['Random', 'KMC', 'Quality\nFilter', 'Anti-\nCurriculum']
    method_colors = [COLORS['Random'], COLORS['KMC'], COLORS['Quality'], COLORS['Anti']]

    fig, axes = plt.subplots(1, 4, figsize=(12, 3.5), sharey=False)

    has_data = False
    for idx, (ds_key, ds_name) in enumerate(DATASETS_DISPLAY.items()):
        ax = axes[idx]
        rd = DATASET_RESULTS[ds_key]
        evals = load_eval_metrics(rd, methods)

        fid_vals = []
        for m in methods:
            if m in evals and 'fid' in evals[m]:
                fid_vals.append(evals[m]['fid'])
                has_data = True
            else:
                fid_vals.append(0)

        x = np.arange(len(methods))
        bars = ax.bar(x, fid_vals, color=method_colors, alpha=0.85,
                      edgecolor='white', width=0.6)

        # Add value labels
        for bar, val in zip(bars, fid_vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                        f'{val:.1f}', ha='center', va='bottom', fontsize=7)

        ax.set_title(ds_name, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(method_labels, fontsize=7)
        if idx == 0:
            ax.set_ylabel('FID ↓')

    if not has_data:
        print("[SKIP] fig3_fid_comparison: no eval_metrics.json data yet")
        plt.close(fig)
        return

    plt.tight_layout()
    path = Path(out_dir) / 'fig3_fid_comparison.pdf'
    fig.savefig(path)
    fig.savefig(path.with_suffix('.png'))
    plt.close(fig)
    print(f"[OK] {path}")


# ─────────────────────────────────────────────────────────────────
# Figure 4: CLIP Score Comparison
# ─────────────────────────────────────────────────────────────────

def fig4_clip_comparison(out_dir):
    """Grouped bar chart: CLIP scores, KMC vs Random vs Baselines."""
    methods = ['Random_D-High', 'KMC_D-High', 'Quality_Filter', 'Anti_Curriculum']
    method_labels = ['Random', 'KMC', 'Quality\nFilter', 'Anti-\nCurriculum']
    method_colors = [COLORS['Random'], COLORS['KMC'], COLORS['Quality'], COLORS['Anti']]

    fig, axes = plt.subplots(1, 4, figsize=(12, 3.5), sharey=False)

    has_data = False
    for idx, (ds_key, ds_name) in enumerate(DATASETS_DISPLAY.items()):
        ax = axes[idx]
        rd = DATASET_RESULTS[ds_key]
        evals = load_eval_metrics(rd, methods)

        clip_vals = []
        for m in methods:
            if m in evals and 'clip_score' in evals[m] and evals[m]['clip_score']:
                clip_vals.append(evals[m]['clip_score'])
                has_data = True
            else:
                clip_vals.append(0)

        x = np.arange(len(methods))
        bars = ax.bar(x, clip_vals, color=method_colors, alpha=0.85,
                      edgecolor='white', width=0.6)

        for bar, val in zip(bars, clip_vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.002,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=7)

        ax.set_title(ds_name, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(method_labels, fontsize=7)
        if idx == 0:
            ax.set_ylabel('CLIP Score ↑')

    if not has_data:
        print("[SKIP] fig4_clip_comparison: no eval_metrics.json data yet")
        plt.close(fig)
        return

    plt.tight_layout()
    path = Path(out_dir) / 'fig4_clip_comparison.pdf'
    fig.savefig(path)
    fig.savefig(path.with_suffix('.png'))
    plt.close(fig)
    print(f"[OK] {path}")


# ─────────────────────────────────────────────────────────────────
# Figure 5: Ablation Study
# ─────────────────────────────────────────────────────────────────

def fig5_ablation(out_dir):
    """Bar chart: Ablation study - final loss for different phase combos."""
    ablation_exps = ['KMC_D-High',
                     'Ablation_NoPhase1', 'Ablation_NoPhase2', 'Ablation_NoPhase3',
                     'Ablation_Phase1Only', 'Ablation_Phase3Only']
    labels = ['Full\nKMC', 'w/o\nPhase1', 'w/o\nPhase2', 'w/o\nPhase3',
              'Phase1\nOnly', 'Phase3\nOnly']
    colors = [COLORS['KMC'], COLORS['NoPhase1'], COLORS['NoPhase2'],
              COLORS['NoPhase3'], COLORS['Phase1Only'], COLORS['Phase3Only']]

    fig, axes = plt.subplots(2, 2, figsize=(6.8, 5.2), sharey=False)
    axes = axes.flatten()

    for idx, (ds_key, ds_name) in enumerate(DATASETS_DISPLAY.items()):
        ax = axes[idx]
        rd = DATASET_RESULTS[ds_key]
        summaries = load_training_summaries(rd, ablation_exps)

        losses = []
        for exp in ablation_exps:
            if exp in summaries:
                losses.append(summaries[exp].get('final_loss', np.nan))
            else:
                losses.append(np.nan)

        x = np.arange(len(ablation_exps))
        bars = ax.bar(x, losses, color=colors, alpha=0.85,
                      edgecolor='white', width=0.65)

        # Value labels
        for bar, val in zip(bars, losses):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                        f'{val:.4f}', ha='center', va='bottom', fontsize=6,
                        rotation=45)

        ax.set_title(ds_name, fontweight='bold', fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7)
        if idx == 0:
            ax.set_ylabel('Final Loss ↓')

    plt.tight_layout(pad=0.6, w_pad=0.8, h_pad=1.0)
    path = Path(out_dir) / 'fig5_ablation.pdf'
    fig.savefig(path)
    fig.savefig(path.with_suffix('.png'))
    plt.close(fig)
    print(f"[OK] {path}")


# ─────────────────────────────────────────────────────────────────
# Figure 6: Cross-Dataset Summary (radar / grouped bar)
# ─────────────────────────────────────────────────────────────────

def fig6_cross_dataset(out_dir):
    """Grouped bar: KMC improvement over Random across datasets and splits."""
    splits = ['D-High', 'D-Medium', 'D-Low', 'D-Sub-50', 'D-Sub-25']

    fig, ax = plt.subplots(figsize=(10, 4.5))

    ds_keys = list(DATASETS_DISPLAY.keys())
    n_ds = len(ds_keys)
    n_splits = len(splits)
    w = 0.15
    x = np.arange(n_splits)

    for i, ds_key in enumerate(ds_keys):
        rd = DATASET_RESULTS[ds_key]
        improvements = []

        for sp in splits:
            r_summ = load_training_summaries(rd, [f'Random_{sp}'])
            h_summ = load_training_summaries(rd, [f'KMC_{sp}'])
            r_loss = r_summ.get(f'Random_{sp}', {}).get('final_loss', np.nan)
            h_loss = h_summ.get(f'KMC_{sp}', {}).get('final_loss', np.nan)
            if not np.isnan(r_loss) and r_loss > 0:
                improv = (r_loss - h_loss) / r_loss * 100
            else:
                improv = 0
            improvements.append(improv)

        offset = (i - n_ds/2 + 0.5) * w
        bars = ax.bar(x + offset, improvements, w,
                      label=DATASETS_DISPLAY[ds_key], alpha=0.85,
                      edgecolor='white')

    ax.axhline(y=0, color='gray', linewidth=0.5)
    ax.set_xlabel('Data Diversity Level')
    ax.set_ylabel('Loss Improvement (%) ↑')
    ax.set_title('KMC Improvement over Random Baseline', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([SPLITS_SHORT[s] for s in splits])
    ax.legend(loc='best', fontsize=8)

    plt.tight_layout()
    path = Path(out_dir) / 'fig6_cross_dataset.pdf'
    fig.savefig(path)
    fig.savefig(path.with_suffix('.png'))
    plt.close(fig)
    print(f"[OK] {path}")


# ─────────────────────────────────────────────────────────────────
# Figure 7: Phase-wise Loss Decomposition
# ─────────────────────────────────────────────────────────────────

def fig7_phase_loss(out_dir):
    """Stacked / grouped bar: per-phase final loss for KMC_D-High."""
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.5))

    for idx, (ds_key, ds_name) in enumerate(DATASETS_DISPLAY.items()):
        ax = axes[idx]
        rd = DATASET_RESULTS[ds_key]
        phases = load_phase_summaries(rd, 'KMC_D-High')

        phase_names = ['phase1', 'phase2', 'phase3']
        phase_labels = ['Phase 1\n(Easy)', 'Phase 2\n(Medium)', 'Phase 3\n(Hard)']
        phase_colors = [COLORS['phase1'], COLORS['phase2'], COLORS['phase3']]

        losses = [phases.get(p, {}).get('final_loss', 0) for p in phase_names]
        times = [phases.get(p, {}).get('training_time_min', 0) for p in phase_names]

        x = np.arange(3)

        # Loss bars
        bars = ax.bar(x, losses, color=phase_colors, alpha=0.85,
                      edgecolor='white', width=0.5)

        # Add time annotation
        for bar, t in zip(bars, times):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{t:.1f}m', ha='center', va='bottom', fontsize=7)

        ax.set_title(ds_name, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(phase_labels, fontsize=7)
        if idx == 0:
            ax.set_ylabel('Final Loss')

    plt.tight_layout()
    path = Path(out_dir) / 'fig7_phase_loss.pdf'
    fig.savefig(path)
    fig.savefig(path.with_suffix('.png'))
    plt.close(fig)
    print(f"[OK] {path}")


# ─────────────────────────────────────────────────────────────────
# Figure 8: Training Time Comparison
# ─────────────────────────────────────────────────────────────────

def fig8_training_time(out_dir):
    """Bar chart: Total training time KMC vs Random vs Baselines."""
    methods = ['Random_D-High', 'KMC_D-High', 'Quality_Filter', 'Anti_Curriculum']
    method_labels = ['Random', 'KMC', 'Quality\nFilter', 'Anti-\nCurriculum']
    method_colors = [COLORS['Random'], COLORS['KMC'], COLORS['Quality'], COLORS['Anti']]

    fig, axes = plt.subplots(2, 2, figsize=(6.8, 5.2), sharey=False)
    axes = axes.flatten()

    for idx, (ds_key, ds_name) in enumerate(DATASETS_DISPLAY.items()):
        ax = axes[idx]
        rd = DATASET_RESULTS[ds_key]

        times = []
        for method in methods:
            if method.startswith('KMC_'):
                # Sum all phases
                phases = load_phase_summaries(rd, method)
                total = sum(p.get('training_time_min', 0) for p in phases.values())
                times.append(total)
            else:
                summ = load_training_summaries(rd, [method])
                t = summ.get(method, {}).get('training_time_min', 0)
                times.append(t)

        x = np.arange(len(methods))
        bars = ax.bar(x, times, color=method_colors, alpha=0.85,
                      edgecolor='white', width=0.6)

        for bar, val in zip(bars, times):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                        f'{val:.1f}m', ha='center', va='bottom', fontsize=6)

        ax.set_title(ds_name, fontweight='bold', fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(method_labels, fontsize=7, rotation=12)
        if idx % 2 == 0:
            ax.set_ylabel('Time (min)')
        ax.tick_params(axis='y', labelsize=8)

    plt.tight_layout(pad=0.6, w_pad=0.9, h_pad=1.0)
    path = Path(out_dir) / 'fig8_training_time.pdf'
    fig.savefig(path)
    fig.savefig(path.with_suffix('.png'))
    plt.close(fig)
    print(f"[OK] {path}")


# ─────────────────────────────────────────────────────────────────
# Figure 9: Comprehensive Results Table (as figure)
# ─────────────────────────────────────────────────────────────────

def fig9_results_table(out_dir):
    """Generate a formatted results table as an image for the paper."""
    all_exps = (['Random_D-High', 'KMC_D-High'] +
                ['Random_D-Medium', 'KMC_D-Medium'] +
                ['Random_D-Low', 'KMC_D-Low'] +
                ['Random_D-Sub-50', 'KMC_D-Sub-50'] +
                ['Random_D-Sub-25', 'KMC_D-Sub-25'] +
                ['Quality_Filter', 'Anti_Curriculum'])

    rows = []
    for ds_key in DATASETS_DISPLAY:
        rd = DATASET_RESULTS[ds_key]
        for exp in all_exps:
            summaries = load_training_summaries(rd, [exp])
            evals = load_eval_metrics(rd, [exp])
            if exp in summaries:
                s = summaries[exp]
                e = evals.get(exp, {})
                # Compute total time for KMC
                if exp.startswith('KMC_'):
                    phases = load_phase_summaries(rd, exp)
                    time_min = sum(p.get('training_time_min', 0) for p in phases.values())
                else:
                    time_min = s.get('training_time_min', 0)

                rows.append({
                    'Dataset': DATASETS_DISPLAY[ds_key],
                    'Method': exp.replace('_', ' '),
                    'Final Loss': f"{s.get('final_loss', 0):.4f}",
                    'Time (min)': f"{time_min:.1f}",
                    'FID': f"{e.get('fid', '-')}",
                    'CLIP': f"{e.get('clip_score', '-')}",
                })

    # Save as CSV for easy inclusion
    df = pd.DataFrame(rows)
    csv_path = Path(out_dir) / 'results_table.csv'
    df.to_csv(csv_path, index=False)
    print(f"[OK] {csv_path}")

    # Also create a figure-table
    fig, ax = plt.subplots(figsize=(14, len(rows) * 0.25 + 1.5))
    ax.axis('off')

    table = ax.table(cellText=df.values,
                     colLabels=df.columns,
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.2)

    # Style header
    for j in range(len(df.columns)):
        table[0, j].set_facecolor('#4472C4')
        table[0, j].set_text_props(color='white', fontweight='bold')

    # Alternate row colors
    for i in range(1, len(df) + 1):
        color = '#F2F2F2' if i % 2 == 0 else 'white'
        for j in range(len(df.columns)):
            table[i, j].set_facecolor(color)

    path = Path(out_dir) / 'fig9_results_table.pdf'
    fig.savefig(path)
    fig.savefig(path.with_suffix('.png'))
    plt.close(fig)
    print(f"[OK] {path}")


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default='kmc_lora/figures',
                    help='Output directory for figures')
    ap.add_argument('--figures', nargs='*', default=None,
                    help='Specific figures to generate (1-9), default: all')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    figure_funcs = {
        '1': ('Loss Curves (KMC vs Random)', fig1_loss_curves),
        '2': ('Final Loss Comparison', fig2_final_loss_comparison),
        '3': ('FID Comparison', fig3_fid_comparison),
        '4': ('CLIP Score Comparison', fig4_clip_comparison),
        '5': ('Ablation Study', fig5_ablation),
        '6': ('Cross-Dataset Improvement', fig6_cross_dataset),
        '7': ('Phase-wise Loss', fig7_phase_loss),
        '8': ('Training Time', fig8_training_time),
        '9': ('Results Table', fig9_results_table),
    }

    figs_to_gen = args.figures or list(figure_funcs.keys())

    print(f"{'='*50}")
    print(f"KMC-LoRA Paper Figure Generation")
    print(f"Output: {out_dir}")
    print(f"Figures: {figs_to_gen}")
    print(f"{'='*50}\n")

    for fig_id in figs_to_gen:
        if fig_id in figure_funcs:
            name, func = figure_funcs[fig_id]
            print(f"\n[Fig {fig_id}] {name}")
            try:
                func(str(out_dir))
            except Exception as e:
                print(f"  [ERROR] {e}")
                import traceback
                traceback.print_exc()

    print(f"\n{'='*50}")
    print(f"Figure generation complete.")
    print(f"Files in: {out_dir}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
