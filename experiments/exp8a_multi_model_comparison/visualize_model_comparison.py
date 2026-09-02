#!/usr/bin/env python3
"""
Visualize embedding model comparison for author attribution.

Reads the measured accuracies from results/multi_model_results.json (written by
run.py) rather than hardcoding them, so the figure always matches the pipeline.

Usage:
    python experiments/exp8a_multi_model_comparison/run.py                  # produce results
    python experiments/exp8a_multi_model_comparison/visualize_model_comparison.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = Path(__file__).parent.parent.parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def main():
    results_file = RESULTS_DIR / "multi_model_results.json"
    if not results_file.exists():
        print(f"ERROR: {results_file} not found.")
        print("Run experiments/exp8a_multi_model_comparison/run.py first.")
        return 1

    with open(results_file, encoding="utf-8") as f:
        data = json.load(f)

    chance = data["chance"] * 100

    # Sort by embeddings-only accuracy, highest first.
    entries = sorted(
        data["models"].items(),
        key=lambda kv: kv[1]["emb_only"]["acc"],
        reverse=True,
    )
    names = [name for name, _ in entries]
    accuracy = [m["emb_only"]["acc"] * 100 for _, m in entries]
    dims = [m["dims"] for _, m in entries]
    types = [m["type"] for _, m in entries]

    print(f"Models found: {len(names)}")
    for name, acc, dim in zip(names, accuracy, dims):
        print(f"  {name:12s} {acc:5.1f}%  ({dim}d)")

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    colors = ['#e74c3c' if t == 'API' else '#3498db' for t in types]
    bars = ax.bar(names, accuracy, color=colors, edgecolor='black',
                  linewidth=1.5, width=0.65)

    ax.axhline(y=chance, color='gray', linestyle='--', linewidth=2)

    legend_elements = [
        Patch(facecolor='#e74c3c', edgecolor='black', label='API'),
        Patch(facecolor='#3498db', edgecolor='black', label='Local'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11)

    ax.set_ylabel('Classification Accuracy (%)', fontsize=12)
    ax.set_title(
        f'Author Attribution: {len(names)} Embedding Models Compared\n'
        f'({data["n_authors"]} Russian poets, '
        f'{data["n_samples"] // data["n_authors"]} poems each, 5-fold CV)',
        fontsize=13, fontweight='bold')
    ax.set_ylim(0, max(accuracy) * 1.18)

    for bar, val, dim in zip(bars, accuracy, dims):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f'{val:.1f}%\n({dim}d)', ha='center', va='bottom',
                fontsize=9, fontweight='bold')

    for bar, val in zip(bars, accuracy):
        ax.text(bar.get_x() + bar.get_width() / 2, 5,
                f'{val / chance:.0f}×', ha='center', va='bottom',
                fontsize=9, color='white', fontweight='bold')

    # Divider between API and local models (they are contiguous once sorted
    # only if the ranking happens to group them; draw it where the split falls).
    n_api = sum(1 for t in types if t == 'API')
    if 0 < n_api < len(types) and types[:n_api] == ['API'] * n_api:
        ax.axvline(x=n_api - 0.5, color='darkgreen', linestyle=':',
                   linewidth=2, alpha=0.7)
        ax.text(n_api - 0.5, max(accuracy) * 1.10, 'API | Local', ha='center',
                fontsize=10, color='darkgreen', fontweight='bold')

    plt.tight_layout()
    png = FIGURES_DIR / 'embedding_model_comparison.png'
    plt.savefig(png, dpi=150, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'embedding_model_comparison.pdf', bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
