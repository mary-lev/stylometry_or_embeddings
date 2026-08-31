#!/usr/bin/env python3
"""
EXPERIMENT 2 (Italian): All-Pairs Binary Classification with Embeddings

Purpose: Compare embedding-based pairwise classification with stylometry for Italian poetry.
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from itertools import combinations
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold

DATA_DIR = Path(__file__).parent.parent.parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_italian_dataset(n_per_author=200, seed=42):
    """Load the published Italian dataset with Gemini embeddings.

    Data comes from data/italian/ through the shared loader; see
    data/README.md. The n_per_author and seed arguments are kept for
    call-site compatibility -- the published dataset is fixed at 200
    poems per author, seed 42.
    """
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
    from data_loader import load_dataset

    data = load_dataset('italian', embedding_model='gemini')

    return {
        'embeddings': data['embeddings'],
        'labels': data['labels'],
        'author_list': data['author_list'],
        'n_per_author': data['metadata']['n_per_author'],
    }


def classify_pair(embeddings, labels, cv_folds=5):
    """Run cross-validated classification on a pair."""
    clf = LogisticRegression(max_iter=2000, solver='lbfgs')
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scores = cross_val_score(clf, embeddings, labels, cv=cv)
    return float(np.mean(scores)), float(np.std(scores))


def main():
    parser = argparse.ArgumentParser(description="EXP2 (Italian): Pairwise Classification with Embeddings")
    parser.add_argument('--n-per-author', type=int, default=200)
    parser.add_argument('--cv-folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    print("=" * 70)
    print("EXPERIMENT 2 (Italian): Pairwise Classification with Gemini Embeddings")
    print("=" * 70)

    # Load data
    print("\n[Loading Italian dataset with Gemini embeddings...]")
    data = load_italian_dataset(n_per_author=args.n_per_author, seed=args.seed)

    authors = data['author_list']
    embeddings = data['embeddings']
    labels = data['labels']
    n_per_author = data['n_per_author']

    n_authors = len(authors)
    n_pairs = n_authors * (n_authors - 1) // 2

    print(f"  Authors: {n_authors}")
    print(f"  Samples: {len(labels)}")
    print(f"  Embedding dims: {embeddings.shape[1]}")
    print(f"  Pairs to process: {n_pairs}")

    # Process all pairs
    print(f"\n[Processing {n_pairs} pairs...]")
    results = []

    author_to_idx = {a: i for i, a in enumerate(authors)}

    for i, (a1, a2) in enumerate(combinations(authors, 2)):
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{n_pairs} pairs...")

        idx1 = author_to_idx[a1]
        idx2 = author_to_idx[a2]

        # Get embeddings for this pair
        mask1 = labels == idx1
        mask2 = labels == idx2

        emb_pair = np.vstack([embeddings[mask1], embeddings[mask2]])
        labels_pair = np.array([0] * mask1.sum() + [1] * mask2.sum())

        acc_mean, acc_std = classify_pair(emb_pair, labels_pair, cv_folds=args.cv_folds)

        results.append({
            'author1': a1,
            'author2': a2,
            'accuracy': acc_mean,
            'accuracy_std': acc_std,
            'n_per_author': n_per_author,
            'distance': 2 * (acc_mean - 0.5)
        })

    print(f"  Completed: {len(results)} pairs")

    # Build distance matrix
    distance_matrix = np.zeros((n_authors, n_authors))
    for r in results:
        i = author_to_idx[r['author1']]
        j = author_to_idx[r['author2']]
        distance_matrix[i, j] = r['distance']
        distance_matrix[j, i] = r['distance']

    # Sort by accuracy
    sorted_results = sorted(results, key=lambda x: x['accuracy'])
    top_similar = sorted_results[:10]
    top_different = sorted_results[-10:][::-1]

    # Statistics
    accuracies = [r['accuracy'] for r in results]

    print(f"\n[Results]")
    print(f"\n  Top-10 Most Similar Pairs (lowest accuracy):")
    for r in top_similar:
        print(f"    {r['accuracy']:.1%}: {r['author1'][:25]} vs {r['author2'][:25]}")

    print(f"\n  Top-10 Most Different Pairs (highest accuracy):")
    for r in top_different:
        print(f"    {r['accuracy']:.1%}: {r['author1'][:25]} vs {r['author2'][:25]}")

    print(f"\n  Accuracy Statistics:")
    print(f"    Mean: {np.mean(accuracies):.1%}")
    print(f"    Std: {np.std(accuracies):.1%}")
    print(f"    Min: {np.min(accuracies):.1%}")
    print(f"    Max: {np.max(accuracies):.1%}")

    # Compare with stylometry
    print(f"\n  Comparison with Stylometry:")
    print(f"    Embedding mean: {np.mean(accuracies):.1%}")
    print(f"    Stylometry mean: 98.0% (from exp0)")
    diff = 0.980 - np.mean(accuracies)
    print(f"    Difference: {diff:+.1%} (stylometry {'wins' if diff > 0 else 'loses'})")

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'settings': {
            'mode': 'clean_dataset',
            'language': 'italian',
            'embedding_model': 'gemini',
            'n_per_author': n_per_author,
            'cv_folds': args.cv_folds,
            'seed': args.seed,
            'n_authors': n_authors,
            'n_pairs_processed': len(results)
        },
        'author_list': authors,
        'pairwise_results': results,
        'statistics': {
            'mean_accuracy': float(np.mean(accuracies)),
            'std_accuracy': float(np.std(accuracies)),
            'min_accuracy': float(np.min(accuracies)),
            'max_accuracy': float(np.max(accuracies))
        },
        'top_similar': top_similar,
        'top_different': top_different
    }

    results_file = RESULTS_DIR / f"pairwise_results_italian_n{n_per_author}_gemini.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {results_file}")

    matrix_file = RESULTS_DIR / f"distance_matrix_italian_n{n_per_author}_gemini.npy"
    np.save(matrix_file, distance_matrix)
    print(f"Distance matrix saved to: {matrix_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
