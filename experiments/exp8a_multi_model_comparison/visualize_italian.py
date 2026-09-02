#!/usr/bin/env python3
"""
Visualization for Experiment 8a: Multi-Model Comparison with Stylometry (ITALIAN)
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = Path(__file__).parent.parent.parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_results():
    """Load Italian results from JSON file."""
    with open(RESULTS_DIR / "multi_model_results_italian.json") as f:
        return json.load(f)


def plot_combined_comparison():
    """Plot combined accuracy comparison across all Italian models."""
    results = load_results()

    # Prepare data
    models = []
    emb_only = []
    combined = []
    types = []

    stylometry = results['stylometry']['full']['acc']

    # Sort by embedding-only accuracy
    sorted_models = sorted(
        results['models'].items(),
        key=lambda x: x[1]['emb_only']['acc'],
        reverse=True
    )

    for name, data in sorted_models:
        models.append(name)
        emb_only.append(data['emb_only']['acc'] * 100)
        combined.append(data['combined']['acc'] * 100)
        types.append(data['type'])

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(models))
    width = 0.35

    # Colors
    emb_colors = ['#e74c3c' if t == 'API' else '#3498db' for t in types]
    comb_colors = ['#c0392b' if t == 'API' else '#2980b9' for t in types]

    # Bars
    bars1 = ax.bar(x - width/2, emb_only, width, label='Embeddings Only',
                   color=emb_colors, edgecolor='black', linewidth=1, alpha=0.7)
    bars2 = ax.bar(x + width/2, combined, width, label='Combined (Emb + Stylometry)',
                   color=comb_colors, edgecolor='black', linewidth=1.5)

    # Stylometry baseline
    ax.axhline(y=stylometry * 100, color='green', linestyle='--', linewidth=2,
               label=f'Stylometry Only ({stylometry*100:.1f}%)')

    # Labels
    ax.set_ylabel('Classification Accuracy (%)', fontsize=12)
    ax.set_title('Experiment 8a: Embedding Models + Stylometry Combination\n(52 Italian poets, 200 poems each)',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylim(0, 100)
    ax.legend(loc='upper right', fontsize=10)

    # Add value labels on combined bars
    for bar, val in zip(bars2, combined):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Add gain labels
    for i, (bar, comb_val) in enumerate(zip(bars2, combined)):
        gain = comb_val - stylometry * 100
        sign = '+' if gain >= 0 else ''
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 3,
                f'{sign}{gain:.1f}%', ha='center', va='top', fontsize=9, color='white', fontweight='bold')

    # Add API/Local labels
    for i, (model, t) in enumerate(zip(models, types)):
        ax.text(i, 2, t, ha='center', fontsize=9, color='gray')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'combined_comparison_italian.png', dpi=150, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'combined_comparison_italian.pdf', bbox_inches='tight')
    print(f"Saved: {FIGURES_DIR / 'combined_comparison_italian.png'}")


def plot_unique_signal():
    """Plot unique signal (residual accuracy) for each Italian model."""
    results = load_results()

    # Prepare data
    models = []
    unique_emb = []
    var_explained = []
    types = []

    sorted_models = sorted(
        results['models'].items(),
        key=lambda x: x[1]['residual_emb']['acc'],
        reverse=True
    )

    for name, data in sorted_models:
        models.append(name)
        unique_emb.append(data['residual_emb']['acc'] * 100)
        var_explained.append(data['var_explained_by_styl'] * 100)
        types.append(data['type'])

    chance = results['chance'] * 100

    # Create figure with two panels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: Unique signal (residual accuracy)
    colors = ['#e74c3c' if t == 'API' else '#3498db' for t in types]
    bars = ax1.bar(models, unique_emb, color=colors, edgecolor='black', linewidth=1.5)
    ax1.axhline(y=chance, color='gray', linestyle='--', linewidth=2, label=f'Chance ({chance:.1f}%)')
    ax1.set_ylabel('Residual Accuracy (%)', fontsize=12)
    ax1.set_title('A. Unique Signal in Embeddings\n(after removing stylometry)', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 15)
    ax1.legend(loc='upper right')

    for bar, val in zip(bars, unique_emb):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Panel 2: Variance explained by stylometry
    bars2 = ax2.bar(models, var_explained, color=colors, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Variance Explained (%)', fontsize=12)
    ax2.set_title('B. Embedding Variance Explained by Stylometry\n(overlap with stylometry)', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 100)

    for bar, val in zip(bars2, var_explained):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Add API/Local legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', edgecolor='black', label='API'),
        Patch(facecolor='#3498db', edgecolor='black', label='Local'),
    ]
    ax2.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'unique_signal_italian.png', dpi=150, bbox_inches='tight')
    plt.savefig(RESULTS_DIR / 'unique_signal_italian.pdf', bbox_inches='tight')
    print(f"Saved: {RESULTS_DIR / 'unique_signal_italian.png'}")


def plot_summary_table():
    """Create summary visualization for Italian."""
    results = load_results()

    # Prepare data
    stylometry = results['stylometry']['full']['acc'] * 100
    chance = results['chance'] * 100

    data_rows = []
    sorted_models = sorted(
        results['models'].items(),
        key=lambda x: x[1]['combined']['acc'],
        reverse=True
    )

    for name, data in sorted_models:
        data_rows.append({
            'model': name,
            'type': data['type'],
            'dims': data['dims'],
            'emb_only': data['emb_only']['acc'] * 100,
            'combined': data['combined']['acc'] * 100,
            'gain': (data['combined']['acc'] - results['stylometry']['full']['acc']) * 100,
            'unique': data['residual_emb']['acc'] * 100,
            'var_exp': data['var_explained_by_styl'] * 100
        })

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('off')

    # Table data
    columns = ['Model', 'Type', 'Dims', 'Emb Only', 'Combined', 'Gain', 'Unique Signal', 'Var Explained']
    cell_text = []

    for row in data_rows:
        cell_text.append([
            row['model'],
            row['type'],
            str(row['dims']),
            f"{row['emb_only']:.1f}%",
            f"{row['combined']:.1f}%",
            f"{row['gain']:+.1f}%",
            f"{row['unique']:.1f}%",
            f"{row['var_exp']:.1f}%"
        ])

    # Add stylometry row
    cell_text.insert(0, ['Stylometry', '-', '4222', '-', f'{stylometry:.1f}%', '-', '-', '-'])

    table = ax.table(
        cellText=cell_text,
        colLabels=columns,
        loc='center',
        cellLoc='center',
        colColours=['#f0f0f0'] * len(columns)
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # Style header
    for i in range(len(columns)):
        table[(0, i)].set_text_props(fontweight='bold')

    # Highlight best combined
    table[(1, 4)].set_facecolor('#90EE90')

    ax.set_title('Experiment 8a: Multi-Model Comparison Summary (ITALIAN)\n52 authors, 10,400 poems',
                 fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'summary_table_italian.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {RESULTS_DIR / 'summary_table_italian.png'}")


if __name__ == "__main__":
    plot_combined_comparison()
    plot_unique_signal()
    plot_summary_table()
    print("\nAll Italian visualizations created!")
