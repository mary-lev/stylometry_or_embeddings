#!/usr/bin/env python3
"""
EXPERIMENT 8: Combined Features (Embeddings + Stylometry)

Purpose: Establish ceiling performance by combining LLM embeddings with classical stylometry.

Methods:
1. Early fusion: Concatenate scaled embeddings + stylometry features
2. Compare: embeddings only, stylometry only, combined

Both multiclass (29 authors) and pairwise binary classification.
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from itertools import combinations
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Constants
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# The published dataset is fixed at 200 poems per author (see data/README.md);
# every result in the paper is computed at this size.
N_PER_AUTHOR = 200


# Russian function words
RUSSIAN_FUNCTION_WORDS = list(set([
    'я', 'ты', 'он', 'она', 'оно', 'мы', 'вы', 'они',
    'мне', 'тебе', 'ему', 'ей', 'нам', 'вам', 'им',
    'меня', 'тебя', 'его', 'её', 'нас', 'вас', 'их',
    'мой', 'твой', 'наш', 'ваш',
    'этот', 'тот', 'такой', 'какой', 'который', 'чей',
    'кто', 'что', 'весь', 'всё', 'сам', 'самый',
    'себя', 'себе', 'собой',
    'в', 'на', 'с', 'к', 'у', 'о', 'об', 'по', 'из', 'за',
    'от', 'до', 'для', 'при', 'над', 'под', 'перед', 'между',
    'через', 'без', 'про', 'ради', 'вместо', 'кроме',
    'и', 'а', 'но', 'да', 'или', 'либо', 'то', 'ни',
    'чтобы', 'как', 'когда', 'если', 'хотя', 'пока',
    'потому', 'поэтому', 'так', 'будто', 'словно', 'точно',
    'не', 'бы', 'же', 'ли', 'ведь', 'вот', 'вон',
    'даже', 'лишь', 'только', 'уже', 'ещё', 'еще', 'именно',
    'быть', 'есть', 'был', 'была', 'было', 'были', 'будет',
    'стать', 'стал', 'стала', 'мочь', 'может', 'могу',
    'где', 'куда', 'откуда', 'там', 'тут', 'здесь',
    'очень', 'тоже', 'также',
    'всегда', 'никогда', 'иногда', 'теперь', 'потом', 'опять',
]))


def extract_stylometry_features(texts):
    """Extract stylometry features: char 3-grams + function words."""
    # Character 3-grams (TF-IDF)
    char_vec = TfidfVectorizer(
        analyzer='char',
        ngram_range=(3, 3),
        max_features=2000,
        lowercase=True
    )
    X_char = char_vec.fit_transform(texts)

    # Function words (normalized counts)
    func_vec = CountVectorizer(
        analyzer='word',
        vocabulary=RUSSIAN_FUNCTION_WORDS,
        lowercase=True
    )
    X_func = func_vec.fit_transform(texts)
    row_sums = X_func.sum(axis=1).A1
    row_sums[row_sums == 0] = 1
    X_func = X_func.multiply(1 / row_sums[:, np.newaxis])

    # Combine stylometry features
    X_style = hstack([X_char, X_func])

    return X_style, char_vec, func_vec


def run_classification(X, y, cv_folds=5, seed=42):
    """Run cross-validated classification."""
    # Handle sparse matrices
    if hasattr(X, 'toarray'):
        scaler = StandardScaler(with_mean=False)
    else:
        scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(max_iter=2000, C=1.0)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring='accuracy')

    return float(np.mean(scores)), float(np.std(scores))


def run_pairwise(X, y, authors, cv_folds=5, seed=42):
    """Run pairwise binary classification."""
    author_to_idx = {a: i for i, a in enumerate(authors)}
    results = []

    # Convert to CSR for efficient row slicing
    if hasattr(X, 'tocsr'):
        X = X.tocsr()

    for a1, a2 in combinations(authors, 2):
        idx1 = author_to_idx[a1]
        idx2 = author_to_idx[a2]

        mask = (y == idx1) | (y == idx2)
        X_pair = X[mask]
        y_pair = (y[mask] == idx2).astype(int)

        acc, std = run_classification(X_pair, y_pair, cv_folds=cv_folds, seed=seed)
        results.append({
            'author1': a1,
            'author2': a2,
            'accuracy': acc,
            'accuracy_std': std
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="EXP8: Combined Features")
    parser.add_argument('--cv-folds', type=int, default=5,
                        help='CV folds')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--embedding-model', type=str, default='gemini',
                        choices=['openai', 'gemini'],
                        help='Embedding model to use')
    parser.add_argument('--pairwise', action='store_true',
                        help='Also run pairwise classification (slow)')
    args = parser.parse_args()

    np.random.seed(args.seed)

    print("=" * 60)
    print("EXPERIMENT 8: Combined Features (Embeddings + Stylometry)")
    print("=" * 60)

    # Load clean dataset
    print(f"\n[Loading clean dataset with {args.embedding_model} embeddings...]")
    from clean_dataset import load_clean_dataset

    data = load_clean_dataset(
        n_per_author=N_PER_AUTHOR,
        seed=args.seed,
        include_poems=True,
        embedding_model=args.embedding_model
    )

    texts = [p['text'] for p in data['poems']]
    embeddings = data['embeddings']
    labels = data['labels']
    authors = data['author_list']

    n_authors = len(authors)
    n_samples = len(texts)
    chance = 1 / n_authors

    print(f"  Authors: {n_authors}")
    print(f"  Samples: {n_samples}")
    print(f"  Embedding model: {args.embedding_model}")
    print(f"  Embedding dims: {embeddings.shape[1]}")
    print(f"  Chance level: {chance:.1%}")

    # Extract stylometry features
    print("\n[Extracting stylometry features...]")
    X_style, char_vec, func_vec = extract_stylometry_features(texts)
    print(f"  Stylometry features: {X_style.shape[1]}")

    # Scale embeddings
    print("[Scaling embeddings...]")
    emb_scaler = StandardScaler()
    X_emb = emb_scaler.fit_transform(embeddings)
    print(f"  Embedding features: {X_emb.shape[1]}")

    # Combine: embeddings + stylometry (early fusion)
    print("[Combining features (early fusion)...]")
    # Convert embeddings to sparse for efficient hstack
    X_emb_sparse = csr_matrix(X_emb)
    X_combined = hstack([X_emb_sparse, X_style]).tocsr()
    X_style = X_style.tocsr()  # Convert for pairwise indexing
    print(f"  Combined features: {X_combined.shape[1]}")

    # Results storage
    results = {
        'timestamp': datetime.now().isoformat(),
        'language': 'russian',
        'settings': {
            'n_authors': n_authors,
            'n_samples': n_samples,
            'n_per_author': N_PER_AUTHOR,
            'cv_folds': args.cv_folds,
            'seed': args.seed,
            'chance_level': chance,
            'embedding_model': args.embedding_model,
            'stylometry_features': X_style.shape[1],
            'embedding_features': X_emb.shape[1],
            'combined_features': X_combined.shape[1]
        },
        'multiclass': {},
        'pairwise': {}
    }

    # ==================== MULTICLASS ====================
    print("\n" + "=" * 60)
    print("MULTICLASS CLASSIFICATION (29 authors)")
    print("=" * 60)

    # Method 1: Embeddings only
    print("\n[1. Embeddings only...]")
    acc_emb, std_emb = run_classification(X_emb, labels, cv_folds=args.cv_folds, seed=args.seed)
    print(f"  Accuracy: {acc_emb:.1%} ± {std_emb:.1%}")
    results['multiclass']['embeddings'] = {'accuracy': acc_emb, 'std': std_emb}

    # Method 2: Stylometry only
    print("\n[2. Stylometry only...]")
    acc_style, std_style = run_classification(X_style, labels, cv_folds=args.cv_folds, seed=args.seed)
    print(f"  Accuracy: {acc_style:.1%} ± {std_style:.1%}")
    results['multiclass']['stylometry'] = {'accuracy': acc_style, 'std': std_style}

    # Method 3: Combined (early fusion)
    print("\n[3. Combined (embeddings + stylometry)...]")
    acc_comb, std_comb = run_classification(X_combined, labels, cv_folds=args.cv_folds, seed=args.seed)
    print(f"  Accuracy: {acc_comb:.1%} ± {std_comb:.1%}")
    results['multiclass']['combined'] = {'accuracy': acc_comb, 'std': std_comb}

    # Summary
    print("\n" + "-" * 60)
    print("MULTICLASS SUMMARY")
    print("-" * 60)
    print(f"{'Method':<25} {'Accuracy':>12} {'Lift':>8}")
    print("-" * 60)
    print(f"{'Embeddings only':<25} {acc_emb:>11.1%} {acc_emb/chance:>7.1f}x")
    print(f"{'Stylometry only':<25} {acc_style:>11.1%} {acc_style/chance:>7.1f}x")
    print(f"{'Combined (fusion)':<25} {acc_comb:>11.1%} {acc_comb/chance:>7.1f}x")
    print("-" * 60)
    print(f"{'Chance':<25} {chance:>11.1%} {1.0:>7.1f}x")

    improvement_over_best = acc_comb - max(acc_emb, acc_style)
    print(f"\nCombined improvement over best single: {improvement_over_best:+.1%}")

    # ==================== PAIRWISE ====================
    if args.pairwise:
        print("\n" + "=" * 60)
        print("PAIRWISE BINARY CLASSIFICATION (406 pairs)")
        print("=" * 60)

        # Method 1: Embeddings only
        print("\n[1. Embeddings pairwise...]")
        pw_emb = run_pairwise(X_emb, labels, authors, cv_folds=args.cv_folds, seed=args.seed)
        emb_accs = [p['accuracy'] for p in pw_emb]
        print(f"  Mean: {np.mean(emb_accs):.1%}, Min: {np.min(emb_accs):.1%}, Max: {np.max(emb_accs):.1%}")
        results['pairwise']['embeddings'] = {
            'mean': float(np.mean(emb_accs)),
            'min': float(np.min(emb_accs)),
            'max': float(np.max(emb_accs)),
            'results': pw_emb
        }

        # Method 2: Stylometry only
        print("\n[2. Stylometry pairwise...]")
        pw_style = run_pairwise(X_style, labels, authors, cv_folds=args.cv_folds, seed=args.seed)
        style_accs = [p['accuracy'] for p in pw_style]
        print(f"  Mean: {np.mean(style_accs):.1%}, Min: {np.min(style_accs):.1%}, Max: {np.max(style_accs):.1%}")
        results['pairwise']['stylometry'] = {
            'mean': float(np.mean(style_accs)),
            'min': float(np.min(style_accs)),
            'max': float(np.max(style_accs)),
            'results': pw_style
        }

        # Method 3: Combined
        print("\n[3. Combined pairwise...]")
        pw_comb = run_pairwise(X_combined, labels, authors, cv_folds=args.cv_folds, seed=args.seed)
        comb_accs = [p['accuracy'] for p in pw_comb]
        print(f"  Mean: {np.mean(comb_accs):.1%}, Min: {np.min(comb_accs):.1%}, Max: {np.max(comb_accs):.1%}")
        results['pairwise']['combined'] = {
            'mean': float(np.mean(comb_accs)),
            'min': float(np.min(comb_accs)),
            'max': float(np.max(comb_accs)),
            'results': pw_comb
        }

        # Find most similar pairs for combined
        sorted_comb = sorted(pw_comb, key=lambda x: x['accuracy'])
        print("\n  Top-5 Most Similar Pairs (Combined):")
        for p in sorted_comb[:5]:
            a1 = p['author1'].split()[-1]
            a2 = p['author2'].split()[-1]
            print(f"    {p['accuracy']:.1%}: {a1} vs {a2}")

        # Summary
        print("\n" + "-" * 60)
        print("PAIRWISE SUMMARY")
        print("-" * 60)
        print(f"{'Method':<25} {'Mean':>10} {'Min':>10} {'Max':>10}")
        print("-" * 60)
        print(f"{'Embeddings only':<25} {np.mean(emb_accs):>9.1%} {np.min(emb_accs):>9.1%} {np.max(emb_accs):>9.1%}")
        print(f"{'Stylometry only':<25} {np.mean(style_accs):>9.1%} {np.min(style_accs):>9.1%} {np.max(style_accs):>9.1%}")
        print(f"{'Combined (fusion)':<25} {np.mean(comb_accs):>9.1%} {np.min(comb_accs):>9.1%} {np.max(comb_accs):>9.1%}")

    # Save results
    model_suffix = f"_{args.embedding_model}" if args.embedding_model != 'openai' else ""
    output_file = RESULTS_DIR / f"combined_results_n{N_PER_AUTHOR}{model_suffix}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
