#!/usr/bin/env python3
"""
EXPERIMENT 4: Multiclass Author Attribution

Purpose: RQ1 extended - How well do embeddings distinguish among many authors simultaneously?

Method:
1. Use clean balanced dataset (N poems per author with complete data)
2. Run 5-fold stratified CV
3. Report accuracy, confusion matrix, per-class F1
"""

import sys
import json
import argparse
import warnings
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Constants
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="EXP4: Multiclass Author Attribution")
    parser.add_argument('--use-clean-dataset', action='store_true',
                        help='Use pre-created clean dataset (recommended)')
    parser.add_argument('--n-per-author', type=int, default=200,
                        help='Poems per author (100 or 200 for clean dataset)')
    parser.add_argument('--min-poems', type=int, default=100,
                        help='Minimum poems per author (legacy mode)')
    parser.add_argument('--cv-folds', type=int, default=5,
                        help='CV folds')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--embedding-model', type=str, default='openai',
                        choices=['openai', 'gemini'],
                        help='Embedding model to use')
    args = parser.parse_args()

    print("=" * 60)
    print(f"EXPERIMENT 4: Multiclass Author Attribution ({args.embedding_model.upper()})")
    print("=" * 60)

    if args.use_clean_dataset:
        # Load clean dataset
        print(f"\n[Loading clean dataset with {args.embedding_model} embeddings...]")
        from clean_dataset import load_clean_dataset

        data = load_clean_dataset(
            n_per_author=args.n_per_author,
            seed=args.seed,
            embedding_model=args.embedding_model
        )

        embeddings = data['embeddings']
        y = data['labels']
        unique_authors = data['author_list']
        author_to_idx = data['author_to_idx']
        idx_to_author = data['idx_to_author']

        n_authors = len(unique_authors)
        n_samples = len(y)
        model_suffix = f"_{args.embedding_model}" if args.embedding_model != 'openai' else ""
        suffix = f"_clean_n{args.n_per_author}{model_suffix}"
    else:
        # Legacy: Load from corpus
        print("\n[Loading data from corpus...]")
        from data_loader import load_corpus, balance_by_author

        corpus = load_corpus(min_poems_per_author=args.min_poems, include_prosody=True, include_embeddings=True)

        print("[Balancing dataset...]")
        poems, embeddings, labels = balance_by_author(
            corpus['poems'],
            corpus['embeddings'],
            n_per_author=args.n_per_author,
            random_state=args.seed
        )

        unique_authors = sorted(set(labels))
        author_to_idx = {a: i for i, a in enumerate(unique_authors)}
        idx_to_author = {i: a for a, i in author_to_idx.items()}
        y = np.array([author_to_idx[a] for a in labels])

        n_authors = len(unique_authors)
        n_samples = len(y)
        suffix = "_legacy"

    chance_level = 1 / n_authors

    print(f"\n[Settings]")
    print(f"  Dataset: {'clean' if args.use_clean_dataset else 'legacy'}")
    print(f"  Embedding model: {args.embedding_model}")
    print(f"  Embedding dims: {embeddings.shape[1]}")
    print(f"  Authors: {n_authors}")
    print(f"  Poems per author: {args.n_per_author}")
    print(f"  Total samples: {n_samples}")
    print(f"  CV folds: {args.cv_folds}")
    print(f"  Chance level: {chance_level:.1%}")

    # Scale embeddings
    print("\n[Scaling embeddings...]")
    scaler = StandardScaler()
    X = scaler.fit_transform(embeddings)

    # Run cross-validated prediction
    print("[Running cross-validated classification...]")
    clf = LogisticRegression(max_iter=2000, solver='lbfgs')

    # Get predictions for confusion matrix
    y_pred = cross_val_predict(clf, X, y, cv=args.cv_folds)

    # Calculate metrics
    accuracy = accuracy_score(y, y_pred)
    conf_matrix = confusion_matrix(y, y_pred)

    # Per-class metrics
    report = classification_report(y, y_pred, target_names=unique_authors, output_dict=True)

    # Also run CV for accuracy with std
    kfold = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
    cv_scores = []
    for train_idx, test_idx in kfold.split(X, y):
        clf_fold = LogisticRegression(max_iter=2000, solver='lbfgs')
        clf_fold.fit(X[train_idx], y[train_idx])
        cv_scores.append(clf_fold.score(X[test_idx], y[test_idx]))

    acc_mean = float(np.mean(cv_scores))
    acc_std = float(np.std(cv_scores))

    # Extract per-author F1 scores
    per_author_f1 = {}
    for author in unique_authors:
        per_author_f1[author] = {
            'f1': report[author]['f1-score'],
            'precision': report[author]['precision'],
            'recall': report[author]['recall'],
            'support': report[author]['support']
        }

    # Sort by F1
    sorted_f1 = sorted(per_author_f1.items(), key=lambda x: x[1]['f1'], reverse=True)

    # Find most confused pairs
    conf_off_diag = conf_matrix.copy()
    np.fill_diagonal(conf_off_diag, 0)

    # Get top confused pairs
    top_confused = []
    flat_indices = np.argsort(conf_off_diag.flatten())[::-1][:10]
    for flat_idx in flat_indices:
        i, j = np.unravel_index(flat_idx, conf_matrix.shape)
        if conf_off_diag[i, j] > 0:
            top_confused.append({
                'true_author': idx_to_author[i],
                'predicted_as': idx_to_author[j],
                'count': int(conf_off_diag[i, j])
            })

    # Print results
    print(f"\n[Results]")
    print(f"  Overall Accuracy: {acc_mean:.1%} ± {acc_std:.1%}")
    print(f"  Chance Level: {chance_level:.1%}")
    print(f"  Lift over chance: {acc_mean / chance_level:.1f}x")

    print(f"\n  Top-5 Best Classified Authors (by F1):")
    for author, metrics in sorted_f1[:5]:
        short_name = author.split()[-1]
        print(f"    {metrics['f1']:.2f}: {short_name}")

    print(f"\n  Top-5 Worst Classified Authors (by F1):")
    for author, metrics in sorted_f1[-5:]:
        short_name = author.split()[-1]
        print(f"    {metrics['f1']:.2f}: {short_name}")

    print(f"\n  Top-5 Most Confused Pairs:")
    for pair in top_confused[:5]:
        true_short = pair['true_author'].split()[-1]
        pred_short = pair['predicted_as'].split()[-1]
        print(f"    {pair['count']}: {true_short} → {pred_short}")

    # Prepare results
    results = {
        'timestamp': datetime.now().isoformat(),
        'settings': {
            'dataset': 'clean' if args.use_clean_dataset else 'legacy',
            'embedding_model': args.embedding_model,
            'embedding_dim': embeddings.shape[1],
            'n_authors': n_authors,
            'n_per_author': args.n_per_author,
            'total_samples': n_samples,
            'cv_folds': args.cv_folds,
            'seed': args.seed,
            'chance_level': chance_level
        },
        'metrics': {
            'accuracy_mean': acc_mean,
            'accuracy_std': acc_std,
            'macro_f1': report['macro avg']['f1-score'],
            'weighted_f1': report['weighted avg']['f1-score'],
            'lift_over_chance': acc_mean / chance_level
        },
        'per_author_metrics': per_author_f1,
        'top_confused_pairs': top_confused,
        'author_list': unique_authors
    }

    # Save results
    results_file = RESULTS_DIR / f"multiclass_results{suffix}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults saved to: {results_file}")

    # Save confusion matrix
    matrix_file = RESULTS_DIR / f"confusion_matrix{suffix}.npy"
    np.save(matrix_file, conf_matrix)
    print(f"Confusion matrix saved to: {matrix_file}")

    # Save author list for matrix interpretation
    authors_file = RESULTS_DIR / "author_list.json"
    with open(authors_file, 'w', encoding='utf-8') as f:
        json.dump(unique_authors, f, indent=2, ensure_ascii=False)
    print(f"Author list saved to: {authors_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
