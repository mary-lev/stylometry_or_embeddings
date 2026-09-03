#!/usr/bin/env python3
"""
EXPERIMENT 2: All-Pairs Binary Classification

Purpose: RQ1 - Can embeddings distinguish authors? Create complete stylistic distance matrix.

Method:
1. For each author pair (406 pairs from 29 Russian authors):
   - Use balanced data (200 poems per author from the clean dataset)
   - Run 5-fold CV classification on embeddings
   - Record accuracy
2. Convert accuracy to distance: distance = 2 * (accuracy - 0.5)
3. Build distance matrix
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from itertools import combinations
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Constants
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# The published dataset is fixed at 200 poems per author (see data/README.md);
# every result in the paper is computed at this size.
N_PER_AUTHOR = 200



def classify_pair(embeddings, labels, cv_folds=5):
    """Run cross-validated classification on a pair."""
    clf = LogisticRegression(max_iter=2000, solver='lbfgs')
    scores = cross_val_score(clf, embeddings, labels, cv=cv_folds)
    return float(np.mean(scores)), float(np.std(scores))


def run_with_clean_dataset(n_per_author=100, cv_folds=5, seed=42, embedding_model='openai'):
    """Run experiment using the pre-created clean dataset."""
    from clean_dataset import load_clean_dataset, get_pair_data

    print("\n[Loading clean dataset...]")
    data = load_clean_dataset(n_per_author=n_per_author, seed=seed, embedding_model=embedding_model)
    print(f"  Embedding model: {embedding_model}")
    print(f"  Embedding dimensions: {data['embeddings'].shape[1]}")

    authors = data['author_list']
    n_authors = len(authors)
    n_per_author = data['metadata']['n_per_author']
    n_pairs = n_authors * (n_authors - 1) // 2

    print(f"\n[Settings]")
    print(f"  Dataset: clean_dataset_n{n_per_author}_seed{seed}")
    print(f"  Authors: {n_authors}")
    print(f"  Poems per author: {n_per_author}")
    print(f"  Pairs to process: {n_pairs}")
    print(f"  CV folds: {cv_folds}")

    # Process all pairs
    print(f"\n[Processing pairs...]")
    results = []

    for i, (a1, a2) in enumerate(combinations(authors, 2)):
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{n_pairs} pairs...")

        emb, labels = get_pair_data(data, a1, a2)
        acc_mean, acc_std = classify_pair(emb, labels, cv_folds=cv_folds)

        results.append({
            'author1': a1,
            'author2': a2,
            'accuracy': acc_mean,
            'accuracy_std': acc_std,
            'n_per_author': n_per_author,
            'distance': 2 * (acc_mean - 0.5)
        })

    return results, authors, n_per_author, seed


def run_with_corpus(min_poems=100, min_pair_poems=50, fixed_n=None, cv_folds=5, seed=42):
    """Run experiment using corpus directly (legacy mode).

    Not available in the public release: this path needs the raw unbalanced
    corpus, which is not distributed. Only the balanced clean dataset is
    published to Zenodo. See data/README.md.
    """
    raise SystemExit(
        "Legacy corpus mode is not available in the public release: it needs "
        "the raw unbalanced corpus, which is not distributed. The published "
        "balanced dataset is used by default."
    )


def main():
    parser = argparse.ArgumentParser(description="EXP2: All-Pairs Binary Classification")
    parser.add_argument('--use-clean-dataset', action='store_true', default=True,
                        help='Use the published clean dataset (default; the legacy '
                             'corpus mode is not available in the public release)')
    parser.add_argument('--min-poems', type=int, default=100,
                        help='Minimum poems per author to include (corpus mode)')
    parser.add_argument('--min-pair-poems', type=int, default=50,
                        help='Minimum poems per author for pairwise comparison (corpus variable mode)')
    parser.add_argument('--fixed-n', type=int, default=None,
                        help='Fixed number of poems per author (corpus fixed mode)')
    parser.add_argument('--cv-folds', type=int, default=5,
                        help='CV folds')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--embedding-model', type=str, default='gemini',
                        choices=['openai', 'gemini', 'voyage', 'qwen8b', 'e5-large', 'bge-m3', 'qwen06b'],
                        help='Embedding model to use')
    args = parser.parse_args()

    print("=" * 60)
    print("EXPERIMENT 2: All-Pairs Binary Classification")
    print("=" * 60)

    if args.use_clean_dataset:
        results, authors, n_per_author, seed = run_with_clean_dataset(
            n_per_author=N_PER_AUTHOR,
            cv_folds=args.cv_folds,
            seed=args.seed,
            embedding_model=args.embedding_model
        )
        # Include embedding model in suffix if not openai
        model_suffix = f"_{args.embedding_model}" if args.embedding_model != 'openai' else ""
        # Name the output after the poems-per-author actually loaded.
        suffix = f"_clean_n{n_per_author}{model_suffix}"
    else:
        results, authors, n_per_author, seed = run_with_corpus(
            min_poems=args.min_poems,
            min_pair_poems=args.min_pair_poems,
            fixed_n=args.fixed_n,
            cv_folds=args.cv_folds,
            seed=args.seed
        )
        suffix = f"_fixed{args.fixed_n}" if args.fixed_n else "_variable"

    print(f"  Completed: {len(results)} pairs")

    # Build distance matrix
    n_authors = len(authors)
    author_to_idx = {a: i for i, a in enumerate(authors)}
    distance_matrix = np.zeros((n_authors, n_authors))

    for r in results:
        i = author_to_idx[r['author1']]
        j = author_to_idx[r['author2']]
        distance_matrix[i, j] = r['distance']
        distance_matrix[j, i] = r['distance']

    # Sort by accuracy to find top similar/different
    sorted_results = sorted(results, key=lambda x: x['accuracy'])
    top_similar = sorted_results[:10]
    top_different = sorted_results[-10:][::-1]

    # Print summary
    print(f"\n[Results]")
    print(f"\n  Top-10 Most Similar Pairs (lowest accuracy):")
    for r in top_similar:
        print(f"    {r['accuracy']:.1%}: {r['author1']} vs {r['author2']}")

    print(f"\n  Top-10 Most Different Pairs (highest accuracy):")
    for r in top_different:
        print(f"    {r['accuracy']:.1%}: {r['author1']} vs {r['author2']}")

    # Statistics
    accuracies = [r['accuracy'] for r in results]
    print(f"\n  Accuracy Statistics:")
    print(f"    Mean: {np.mean(accuracies):.1%}")
    print(f"    Std: {np.std(accuracies):.1%}")
    print(f"    Min: {np.min(accuracies):.1%}")
    print(f"    Max: {np.max(accuracies):.1%}")

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'settings': {
            'mode': 'clean_dataset' if args.use_clean_dataset else ('fixed' if args.fixed_n else 'variable'),
            'embedding_model': args.embedding_model if args.use_clean_dataset else 'openai',
            'n_per_author': n_per_author,
            'cv_folds': args.cv_folds,
            'seed': seed,
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

    # Save JSON results
    results_file = RESULTS_DIR / f"pairwise_results{suffix}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults saved to: {results_file}")

    # Save distance matrix
    matrix_file = RESULTS_DIR / f"distance_matrix{suffix}.npy"
    np.save(matrix_file, distance_matrix)
    print(f"Distance matrix saved to: {matrix_file}")

    # Save author list
    authors_file = RESULTS_DIR / "author_list.json"
    with open(authors_file, 'w', encoding='utf-8') as f:
        json.dump(authors, f, indent=2, ensure_ascii=False)
    print(f"Author list saved to: {authors_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
