#!/usr/bin/env python3
"""
EXPERIMENT 8 (Italian): Combined Features (Embeddings + Stylometry)

Purpose: Establish ceiling performance by combining Gemini embeddings with classical stylometry.

Methods:
1. Early fusion: Concatenate scaled embeddings + stylometry features
2. Compare: embeddings only, stylometry only, combined

Both multiclass (52 authors) and pairwise binary classification.
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

# Constants
DATA_DIR = Path(__file__).parent.parent.parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Italian function words (from exp0_italian_stylometry_baseline)
ITALIAN_FUNCTION_WORDS = list(set([
    # Articles
    'il', 'lo', 'la', 'i', 'gli', 'le', 'un', 'uno', 'una',
    # Prepositions
    'di', 'a', 'da', 'in', 'con', 'su', 'per', 'tra', 'fra',
    'del', 'dello', 'della', 'dei', 'degli', 'delle',
    'al', 'allo', 'alla', 'ai', 'agli', 'alle',
    'dal', 'dallo', 'dalla', 'dai', 'dagli', 'dalle',
    'nel', 'nello', 'nella', 'nei', 'negli', 'nelle',
    'sul', 'sullo', 'sulla', 'sui', 'sugli', 'sulle',
    # Pronouns
    'io', 'tu', 'lui', 'lei', 'noi', 'voi', 'loro', 'egli', 'ella', 'esso', 'essa',
    'mi', 'ti', 'ci', 'vi', 'si', 'lo', 'la', 'li', 'le', 'ne',
    'me', 'te', 'sé', 'ce', 've',
    'mio', 'tuo', 'suo', 'nostro', 'vostro', 'loro',
    'mia', 'tua', 'sua', 'nostra', 'vostra',
    'miei', 'tuoi', 'suoi', 'nostri', 'vostri',
    'mie', 'tue', 'sue', 'nostre', 'vostre',
    'questo', 'questa', 'questi', 'queste', 'quello', 'quella', 'quelli', 'quelle',
    'chi', 'che', 'cui', 'quale', 'quali', 'quanto', 'quanta', 'quanti', 'quante',
    # Conjunctions
    'e', 'ed', 'o', 'od', 'ma', 'però', 'anzi', 'né', 'neanche', 'neppure',
    'se', 'quando', 'mentre', 'perché', 'poiché', 'giacché', 'affinché',
    'come', 'siccome', 'benché', 'sebbene', 'quantunque',
    'dunque', 'quindi', 'pertanto', 'perciò', 'onde', 'ché',
    # Particles and adverbs
    'non', 'più', 'mai', 'sempre', 'già', 'ancora', 'ora', 'adesso',
    'poi', 'dopo', 'prima', 'dove', 'qui', 'qua', 'là', 'lì',
    'su', 'giù', 'fuori', 'dentro', 'sopra', 'sotto',
    'molto', 'poco', 'troppo', 'tanto', 'così', 'pure', 'anche', 'solo',
    'sì', 'no', 'forse', 'certo', 'bene', 'male',
    # Auxiliary verbs
    'essere', 'è', 'era', 'sono', 'sei', 'siamo', 'siete', 'erano', 'fu', 'furono',
    'avere', 'ha', 'ho', 'hai', 'hanno', 'abbiamo', 'avete', 'aveva', 'avevano', 'ebbe',
    'fare', 'fa', 'fece', 'fanno',
    'potere', 'può', 'poteva', 'volere', 'vuole', 'voleva',
    'dovere', 'deve', 'doveva',
    # Common verbs in poetry
    'dire', 'dice', 'disse', 'vedere', 'vede', 'vide', 'venire', 'viene', 'venne',
]))


def load_italian_dataset(n_per_author=200, seed=42):
    """Load Italian clean dataset with Gemini embeddings."""
    # Add src to path for data_loader
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
    from data_loader import load_dataset

    data = load_dataset('italian', embedding_model='gemini', include_poems=True)

    embeddings = data['embeddings']
    labels = data['labels']
    texts = [p['text'] for p in data['poems']]
    author_list = data['author_list']

    return {
        'embeddings': embeddings,
        'labels': labels,
        'texts': texts,
        'author_list': author_list,
        'n_per_author': n_per_author
    }


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
        vocabulary=ITALIAN_FUNCTION_WORDS,
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


def run_pairwise(X, y, authors, cv_folds=5, seed=42, max_pairs=None):
    """Run pairwise binary classification."""
    author_to_idx = {a: i for i, a in enumerate(authors)}
    results = []

    # Convert to CSR for efficient row slicing
    if hasattr(X, 'tocsr'):
        X = X.tocsr()

    pairs = list(combinations(authors, 2))
    if max_pairs:
        pairs = pairs[:max_pairs]

    for i, (a1, a2) in enumerate(pairs):
        if (i + 1) % 200 == 0:
            print(f"    Progress: {i+1}/{len(pairs)} pairs...")

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
    parser = argparse.ArgumentParser(description="EXP8 (Italian): Combined Features")
    parser.add_argument('--n-per-author', type=int, default=200,
                        help='Poems per author')
    parser.add_argument('--cv-folds', type=int, default=5,
                        help='CV folds')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--pairwise', action='store_true',
                        help='Also run pairwise classification (slow - 1326 pairs)')
    args = parser.parse_args()

    np.random.seed(args.seed)

    print("=" * 70)
    print("EXPERIMENT 8 (Italian): Combined Features (Embeddings + Stylometry)")
    print("=" * 70)

    # Load Italian dataset
    print("\n[Loading Italian dataset...]")
    data = load_italian_dataset(n_per_author=args.n_per_author, seed=args.seed)

    texts = data['texts']
    embeddings = data['embeddings']
    labels = data['labels']
    authors = data['author_list']

    n_authors = len(authors)
    n_samples = len(texts)
    chance = 1 / n_authors

    print(f"  Authors: {n_authors}")
    print(f"  Samples: {n_samples}")
    print(f"  Embedding model: Gemini")
    print(f"  Embedding dims: {embeddings.shape[1]}")
    print(f"  Chance level: {chance:.1%}")

    # Extract stylometry features
    print("\n[Extracting stylometry features...]")
    X_style, char_vec, func_vec = extract_stylometry_features(texts)
    print(f"  Char 3-grams: 2000 features")
    print(f"  Function words: {len(ITALIAN_FUNCTION_WORDS)} features")
    print(f"  Total stylometry: {X_style.shape[1]} features")

    # Scale embeddings
    print("[Scaling embeddings...]")
    emb_scaler = StandardScaler()
    X_emb = emb_scaler.fit_transform(embeddings)
    print(f"  Embedding features: {X_emb.shape[1]}")

    # Combine: embeddings + stylometry (early fusion)
    print("[Combining features (early fusion)...]")
    X_emb_sparse = csr_matrix(X_emb)
    X_combined = hstack([X_emb_sparse, X_style]).tocsr()
    X_style = X_style.tocsr()
    print(f"  Combined features: {X_combined.shape[1]}")

    # Results storage
    results = {
        'timestamp': datetime.now().isoformat(),
        'language': 'italian',
        'settings': {
            'n_authors': n_authors,
            'n_samples': n_samples,
            'n_per_author': args.n_per_author,
            'cv_folds': args.cv_folds,
            'seed': args.seed,
            'chance_level': chance,
            'embedding_model': 'gemini',
            'stylometry_features': X_style.shape[1],
            'embedding_features': X_emb.shape[1],
            'combined_features': X_combined.shape[1]
        },
        'multiclass': {},
        'pairwise': {}
    }

    # ==================== MULTICLASS ====================
    print("\n" + "=" * 70)
    print(f"MULTICLASS CLASSIFICATION ({n_authors} authors)")
    print("=" * 70)

    # Method 1: Embeddings only
    print("\n[1. Gemini embeddings only...]")
    acc_emb, std_emb = run_classification(X_emb, labels, cv_folds=args.cv_folds, seed=args.seed)
    print(f"  Accuracy: {acc_emb:.1%} ± {std_emb:.1%}")
    results['multiclass']['embeddings'] = {'accuracy': acc_emb, 'std': std_emb}

    # Method 2: Stylometry only (char 3-grams + function words)
    print("\n[2. Stylometry only (char 3-grams + function words)...]")
    acc_style, std_style = run_classification(X_style, labels, cv_folds=args.cv_folds, seed=args.seed)
    print(f"  Accuracy: {acc_style:.1%} ± {std_style:.1%}")
    results['multiclass']['stylometry'] = {'accuracy': acc_style, 'std': std_style}

    # Method 3: Combined (early fusion)
    print("\n[3. Combined (embeddings + stylometry)...]")
    acc_comb, std_comb = run_classification(X_combined, labels, cv_folds=args.cv_folds, seed=args.seed)
    print(f"  Accuracy: {acc_comb:.1%} ± {std_comb:.1%}")
    results['multiclass']['combined'] = {'accuracy': acc_comb, 'std': std_comb}

    # Summary
    print("\n" + "-" * 70)
    print("MULTICLASS SUMMARY (Italian)")
    print("-" * 70)
    print(f"{'Method':<35} {'Accuracy':>12} {'Lift':>8}")
    print("-" * 70)
    print(f"{'Gemini embeddings only':<35} {acc_emb:>11.1%} {acc_emb/chance:>7.1f}x")
    print(f"{'Stylometry (char3 + func)':<35} {acc_style:>11.1%} {acc_style/chance:>7.1f}x")
    print(f"{'Combined (fusion)':<35} {acc_comb:>11.1%} {acc_comb/chance:>7.1f}x")
    print("-" * 70)
    print(f"{'Chance':<35} {chance:>11.1%} {1.0:>7.1f}x")

    improvement_over_emb = acc_comb - acc_emb
    improvement_over_style = acc_comb - acc_style
    print(f"\nCombined improvement over embeddings: {improvement_over_emb:+.1%}")
    print(f"Combined improvement over stylometry: {improvement_over_style:+.1%}")

    # ==================== PAIRWISE ====================
    if args.pairwise:
        n_pairs = n_authors * (n_authors - 1) // 2
        print("\n" + "=" * 70)
        print(f"PAIRWISE BINARY CLASSIFICATION ({n_pairs} pairs)")
        print("=" * 70)

        # Method 1: Embeddings only
        print("\n[1. Embeddings pairwise...]")
        pw_emb = run_pairwise(X_emb, labels, authors, cv_folds=args.cv_folds, seed=args.seed)
        emb_accs = [p['accuracy'] for p in pw_emb]
        print(f"  Mean: {np.mean(emb_accs):.1%}, Min: {np.min(emb_accs):.1%}, Max: {np.max(emb_accs):.1%}")
        results['pairwise']['embeddings'] = {
            'mean': float(np.mean(emb_accs)),
            'std': float(np.std(emb_accs)),
            'min': float(np.min(emb_accs)),
            'max': float(np.max(emb_accs)),
        }

        # Method 2: Stylometry only
        print("\n[2. Stylometry pairwise...]")
        pw_style = run_pairwise(X_style, labels, authors, cv_folds=args.cv_folds, seed=args.seed)
        style_accs = [p['accuracy'] for p in pw_style]
        print(f"  Mean: {np.mean(style_accs):.1%}, Min: {np.min(style_accs):.1%}, Max: {np.max(style_accs):.1%}")
        results['pairwise']['stylometry'] = {
            'mean': float(np.mean(style_accs)),
            'std': float(np.std(style_accs)),
            'min': float(np.min(style_accs)),
            'max': float(np.max(style_accs)),
        }

        # Method 3: Combined
        print("\n[3. Combined pairwise...]")
        pw_comb = run_pairwise(X_combined, labels, authors, cv_folds=args.cv_folds, seed=args.seed)
        comb_accs = [p['accuracy'] for p in pw_comb]
        print(f"  Mean: {np.mean(comb_accs):.1%}, Min: {np.min(comb_accs):.1%}, Max: {np.max(comb_accs):.1%}")
        results['pairwise']['combined'] = {
            'mean': float(np.mean(comb_accs)),
            'std': float(np.std(comb_accs)),
            'min': float(np.min(comb_accs)),
            'max': float(np.max(comb_accs)),
        }

        # Find most similar pairs for combined
        sorted_comb = sorted(pw_comb, key=lambda x: x['accuracy'])
        print("\n  Top-5 Most Similar Pairs (Combined):")
        for p in sorted_comb[:5]:
            print(f"    {p['accuracy']:.1%}: {p['author1'][:30]} vs {p['author2'][:30]}")

        # Summary
        print("\n" + "-" * 70)
        print("PAIRWISE SUMMARY (Italian)")
        print("-" * 70)
        print(f"{'Method':<35} {'Mean':>10} {'Min':>10} {'Max':>10}")
        print("-" * 70)
        print(f"{'Embeddings only':<35} {np.mean(emb_accs):>9.1%} {np.min(emb_accs):>9.1%} {np.max(emb_accs):>9.1%}")
        print(f"{'Stylometry only':<35} {np.mean(style_accs):>9.1%} {np.min(style_accs):>9.1%} {np.max(style_accs):>9.1%}")
        print(f"{'Combined (fusion)':<35} {np.mean(comb_accs):>9.1%} {np.min(comb_accs):>9.1%} {np.max(comb_accs):>9.1%}")

    # Save results
    output_file = RESULTS_DIR / f"combined_results_italian_n{args.n_per_author}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
