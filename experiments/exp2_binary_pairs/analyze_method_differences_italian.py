#!/usr/bin/env python3
"""
Analyze Method Differences: Embeddings vs Stylometry (ITALIAN)

Same analysis as Russian but for Italian corpus.
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from itertools import combinations
from collections import defaultdict, Counter
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.stats as stats

# Constants
DATA_DIR = Path(__file__).parent.parent.parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_italian_dataset(n_per_author=200, seed=42):
    """Load the published Italian dataset.

    Data comes from data/italian/ through the shared loader; see
    data/README.md. The n_per_author and seed arguments are kept for
    call-site compatibility -- the published dataset is fixed at 200
    poems per author, seed 42.
    """
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
    from data_loader import load_dataset

    data = load_dataset('italian', embedding_model='gemini', include_poems=True)
    poems = data['poems']

    return {
        'embeddings': data['embeddings'],
        'labels': data['labels'],
        'texts': [p['text'] for p in poems],
        'poems': poems,
        'author_list': data['author_list'],
        'n_per_author': data['metadata']['n_per_author'],
    }


class StyleometryExtractor:
    """Extract stylometry features."""

    def __init__(self, char_ngram_range=(2, 4), word_ngram_range=(1, 2),
                 max_features=2000):
        self.char_vectorizer = TfidfVectorizer(
            analyzer='char',
            ngram_range=char_ngram_range,
            max_features=max_features,
            lowercase=True
        )
        self.word_vectorizer = TfidfVectorizer(
            analyzer='word',
            ngram_range=word_ngram_range,
            max_features=max_features,
            lowercase=True
        )

    def fit_transform(self, texts):
        char_features = self.char_vectorizer.fit_transform(texts).toarray()
        word_features = self.word_vectorizer.fit_transform(texts).toarray()
        return np.hstack([char_features, word_features])

    def transform(self, texts):
        char_features = self.char_vectorizer.transform(texts).toarray()
        word_features = self.word_vectorizer.transform(texts).toarray()
        return np.hstack([char_features, word_features])


def get_predictions_for_pair(embeddings, labels, cv_folds=5, seed=42):
    """Get cross-validated predictions for a pair."""
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    scaler = StandardScaler()
    emb_scaled = scaler.fit_transform(embeddings)
    clf = LogisticRegression(max_iter=2000, solver='lbfgs')
    predictions = cross_val_predict(clf, emb_scaled, labels, cv=cv)
    return predictions


def get_stylometry_predictions_for_pair(texts, labels, cv_folds=5, seed=42):
    """Get cross-validated stylometry predictions for a pair."""
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    predictions = np.zeros(len(labels), dtype=int)

    for train_idx, test_idx in cv.split(texts, labels):
        train_texts = [texts[i] for i in train_idx]
        test_texts = [texts[i] for i in test_idx]

        extractor = StyleometryExtractor()
        X_train = extractor.fit_transform(train_texts)
        X_test = extractor.transform(test_texts)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        clf = LogisticRegression(max_iter=2000, solver='lbfgs')
        clf.fit(X_train_scaled, labels[train_idx])
        predictions[test_idx] = clf.predict(X_test_scaled)

    return predictions


def main():
    parser = argparse.ArgumentParser(description="Analyze Method Differences (Italian)")
    parser.add_argument('--n-per-author', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cv-folds', type=int, default=5)
    parser.add_argument('--top-n', type=int, default=30)
    parser.add_argument('--max-pairs', type=int, default=None,
                        help='Max pairs to process (for testing)')
    args = parser.parse_args()

    print("=" * 70)
    print("ANALYZE METHOD DIFFERENCES: EMBEDDINGS vs STYLOMETRY (ITALIAN)")
    print("=" * 70)

    # Load dataset
    print(f"\n[Loading Italian dataset...]")
    data = load_italian_dataset(n_per_author=args.n_per_author, seed=args.seed)

    poems = data['poems']
    texts = data['texts']
    embeddings = data['embeddings']
    authors = data['author_list']

    n_authors = len(authors)
    n_poems = len(poems)

    print(f"  Authors: {n_authors}")
    print(f"  Total poems: {n_poems}")

    author_to_idx = {a: i for i, a in enumerate(authors)}

    # Track per-poem statistics
    poem_stats = []
    for i, poem in enumerate(poems):
        author_idx = i // args.n_per_author
        poem_stats.append({
            'global_idx': i,
            'author': authors[author_idx],
            'word_count': len(texts[i].split()),
            'text': texts[i],
            'emb_correct': 0,
            'emb_wrong': 0,
            'stylo_correct': 0,
            'stylo_wrong': 0,
            'emb_only_correct': 0,
            'stylo_only_correct': 0,
            'both_correct': 0,
            'both_wrong': 0
        })

    # Process pairs
    pairs = list(combinations(authors, 2))
    if args.max_pairs:
        pairs = pairs[:args.max_pairs]

    n_pairs = len(pairs)
    print(f"\n[Processing {n_pairs} pairs...]")

    for pair_idx, (a1, a2) in enumerate(pairs):
        if (pair_idx + 1) % 200 == 0:
            print(f"  Progress: {pair_idx + 1}/{n_pairs} pairs...")

        idx1 = author_to_idx[a1]
        idx2 = author_to_idx[a2]

        start1 = idx1 * args.n_per_author
        end1 = start1 + args.n_per_author
        start2 = idx2 * args.n_per_author
        end2 = start2 + args.n_per_author

        pair_indices = list(range(start1, end1)) + list(range(start2, end2))
        pair_embeddings = embeddings[pair_indices]
        pair_labels = np.array([0] * args.n_per_author + [1] * args.n_per_author)
        pair_texts = [texts[i] for i in pair_indices]

        # Get predictions
        emb_preds = get_predictions_for_pair(
            pair_embeddings, pair_labels,
            cv_folds=args.cv_folds, seed=args.seed
        )
        stylo_preds = get_stylometry_predictions_for_pair(
            pair_texts, pair_labels,
            cv_folds=args.cv_folds, seed=args.seed
        )

        # Record results
        for local_idx, global_idx in enumerate(pair_indices):
            true_label = pair_labels[local_idx]
            emb_correct = (emb_preds[local_idx] == true_label)
            stylo_correct = (stylo_preds[local_idx] == true_label)

            if emb_correct:
                poem_stats[global_idx]['emb_correct'] += 1
            else:
                poem_stats[global_idx]['emb_wrong'] += 1

            if stylo_correct:
                poem_stats[global_idx]['stylo_correct'] += 1
            else:
                poem_stats[global_idx]['stylo_wrong'] += 1

            if emb_correct and not stylo_correct:
                poem_stats[global_idx]['emb_only_correct'] += 1
            elif stylo_correct and not emb_correct:
                poem_stats[global_idx]['stylo_only_correct'] += 1
            elif emb_correct and stylo_correct:
                poem_stats[global_idx]['both_correct'] += 1
            else:
                poem_stats[global_idx]['both_wrong'] += 1

    # Calculate rates
    n_comparisons = n_authors - 1

    for p in poem_stats:
        p['emb_accuracy'] = p['emb_correct'] / n_comparisons
        p['stylo_accuracy'] = p['stylo_correct'] / n_comparisons
        p['emb_advantage'] = p['emb_only_correct'] - p['stylo_only_correct']

    print(f"\n[Analyzing results...]")

    # ========================================================================
    # ANALYSIS 1: Method-specific distinctive poems
    # ========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 1: METHOD-SPECIFIC DISTINCTIVE POEMS")
    print("=" * 70)

    sorted_by_emb_advantage = sorted(poem_stats, key=lambda x: x['emb_advantage'], reverse=True)

    print(f"\n### Poems DISTINCTIVE for EMBEDDINGS")
    print(f"\n| Rank | Author | Words | Emb Acc | Stylo Acc | Emb Advantage |")
    print(f"|------|--------|-------|---------|-----------|---------------|")

    emb_distinctive = [p for p in sorted_by_emb_advantage if p['emb_advantage'] > 0][:args.top_n]
    for i, p in enumerate(emb_distinctive[:15]):
        author_short = p['author'][:20]
        print(f"| {i+1} | {author_short} | {p['word_count']} | {p['emb_accuracy']:.0%} | {p['stylo_accuracy']:.0%} | +{p['emb_advantage']} |")

    sorted_by_stylo_advantage = sorted(poem_stats, key=lambda x: -x['emb_advantage'], reverse=True)

    print(f"\n### Poems DISTINCTIVE for STYLOMETRY")
    print(f"\n| Rank | Author | Words | Emb Acc | Stylo Acc | Stylo Advantage |")
    print(f"|------|--------|-------|---------|-----------|-----------------|")

    stylo_distinctive = [p for p in sorted_by_stylo_advantage if p['emb_advantage'] < 0][:args.top_n]
    for i, p in enumerate(stylo_distinctive[:15]):
        author_short = p['author'][:20]
        print(f"| {i+1} | {author_short} | {p['word_count']} | {p['emb_accuracy']:.0%} | {p['stylo_accuracy']:.0%} | +{-p['emb_advantage']} |")

    # ========================================================================
    # ANALYSIS 2: Short poem analysis
    # ========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 2: SHORT POEM DIFFICULTY")
    print("=" * 70)

    bins = [
        (0, 30, "Very short (≤30)"),
        (31, 50, "Short (31-50)"),
        (51, 80, "Medium-short (51-80)"),
        (81, 120, "Medium (81-120)"),
        (121, 200, "Long (121-200)"),
        (201, 500, "Very long (201-500)"),
        (501, float('inf'), "Epic (>500)")
    ]

    print(f"\n### Error Rates by Poem Length")
    print(f"\n| Length Bin | N Poems | Emb Error | Stylo Error | Both Err | Diff (E-S) |")
    print(f"|------------|---------|-----------|-------------|----------|------------|")

    length_analysis = []
    for min_w, max_w, label in bins:
        bin_poems = [p for p in poem_stats if min_w <= p['word_count'] <= max_w]
        if not bin_poems:
            continue

        n_poems = len(bin_poems)
        avg_emb_err = 1 - np.mean([p['emb_accuracy'] for p in bin_poems])
        avg_stylo_err = 1 - np.mean([p['stylo_accuracy'] for p in bin_poems])
        avg_both_err = np.mean([p['both_wrong'] / n_comparisons for p in bin_poems])

        length_analysis.append({
            'label': label,
            'n_poems': n_poems,
            'emb_error': avg_emb_err,
            'stylo_error': avg_stylo_err,
            'both_error': avg_both_err,
            'diff': avg_emb_err - avg_stylo_err
        })

        print(f"| {label} | {n_poems} | {avg_emb_err:.1%} | {avg_stylo_err:.1%} | {avg_both_err:.1%} | {avg_emb_err - avg_stylo_err:+.1%} |")

    # Statistical test
    word_counts = [p['word_count'] for p in poem_stats]
    emb_accs = [p['emb_accuracy'] for p in poem_stats]
    stylo_accs = [p['stylo_accuracy'] for p in poem_stats]

    emb_corr, emb_p = stats.spearmanr(word_counts, emb_accs)
    stylo_corr, stylo_p = stats.spearmanr(word_counts, stylo_accs)

    print(f"\n### Correlation: Length vs Accuracy")
    print(f"  Embeddings:  r = {emb_corr:.3f} (p = {emb_p:.2e})")
    print(f"  Stylometry:  r = {stylo_corr:.3f} (p = {stylo_p:.2e})")

    if emb_corr > stylo_corr:
        print(f"\n  → Embeddings are MORE sensitive to short texts")
    else:
        print(f"\n  → Stylometry is MORE sensitive to short texts")

    # ========================================================================
    # ANALYSIS 3: Which short poems are hard?
    # ========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 3: WHICH SHORT POEMS ARE HARD?")
    print("=" * 70)

    short_poems = [p for p in poem_stats if p['word_count'] <= 50]
    short_poems_sorted = sorted(short_poems, key=lambda x: x['emb_accuracy'] + x['stylo_accuracy'], reverse=True)

    print(f"\n### Short Poems (≤50 words) that are EASY to classify")
    print(f"\n| Author | Words | Emb Acc | Stylo Acc | Sample |")
    print(f"|--------|-------|---------|-----------|--------|")

    for p in short_poems_sorted[:10]:
        sample = p['text'][:40].replace('\n', ' ') + '...'
        print(f"| {p['author'][:20]} | {p['word_count']} | {p['emb_accuracy']:.0%} | {p['stylo_accuracy']:.0%} | {sample} |")

    print(f"\n### Short Poems (≤50 words) that are HARD to classify")
    print(f"\n| Author | Words | Emb Acc | Stylo Acc | Sample |")
    print(f"|--------|-------|---------|-----------|--------|")

    for p in short_poems_sorted[-10:]:
        sample = p['text'][:40].replace('\n', ' ') + '...'
        print(f"| {p['author'][:20]} | {p['word_count']} | {p['emb_accuracy']:.0%} | {p['stylo_accuracy']:.0%} | {sample} |")

    # Easy vs hard short poems
    easy_short = [p for p in short_poems if p['emb_accuracy'] >= 0.8 and p['stylo_accuracy'] >= 0.8]
    hard_short = [p for p in short_poems if p['emb_accuracy'] <= 0.6 and p['stylo_accuracy'] <= 0.6]

    print(f"\n### Comparing Easy vs Hard Short Poems")
    print(f"  Easy short poems (≥80% both): {len(easy_short)}")
    print(f"  Hard short poems (≤60% both): {len(hard_short)}")

    if easy_short and hard_short:
        easy_authors = [p['author'] for p in easy_short]
        hard_authors = [p['author'] for p in hard_short]

        easy_author_counts = Counter(easy_authors)
        hard_author_counts = Counter(hard_authors)

        print(f"\n  Authors with most EASY short poems:")
        for author, count in easy_author_counts.most_common(5):
            print(f"    {author[:30]}: {count}")

        print(f"\n  Authors with most HARD short poems:")
        for author, count in hard_author_counts.most_common(5):
            print(f"    {author[:30]}: {count}")

    # ========================================================================
    # Save results
    # ========================================================================
    output = {
        'timestamp': datetime.now().isoformat(),
        'language': 'italian',
        'settings': {
            'n_per_author': args.n_per_author,
            'n_authors': n_authors,
            'n_pairs': n_pairs
        },
        'correlations': {
            'embedding_length_corr': emb_corr,
            'embedding_length_pvalue': emb_p,
            'stylometry_length_corr': stylo_corr,
            'stylometry_length_pvalue': stylo_p
        },
        'length_analysis': length_analysis,
        'emb_distinctive_poems': [{
            'author': p['author'],
            'word_count': p['word_count'],
            'emb_accuracy': p['emb_accuracy'],
            'stylo_accuracy': p['stylo_accuracy'],
            'emb_advantage': p['emb_advantage'],
            'text': p['text']
        } for p in emb_distinctive],
        'stylo_distinctive_poems': [{
            'author': p['author'],
            'word_count': p['word_count'],
            'emb_accuracy': p['emb_accuracy'],
            'stylo_accuracy': p['stylo_accuracy'],
            'stylo_advantage': -p['emb_advantage'],
            'text': p['text']
        } for p in stylo_distinctive]
    }

    output_file = RESULTS_DIR / f"method_differences_italian.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_file}")

    # Save distinctive poems to readable files
    emb_file = RESULTS_DIR / f"distinctive_for_embeddings_italian.txt"
    with open(emb_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("POEMS DISTINCTIVE FOR EMBEDDINGS (ITALIAN)\n")
        f.write("=" * 70 + "\n\n")

        for i, p in enumerate(emb_distinctive):
            f.write("-" * 70 + "\n")
            f.write(f"RANK #{i+1}\n")
            f.write(f"Author: {p['author']}\n")
            f.write(f"Word count: {p['word_count']}\n")
            f.write(f"Embedding accuracy: {p['emb_accuracy']:.0%}\n")
            f.write(f"Stylometry accuracy: {p['stylo_accuracy']:.0%}\n")
            f.write(f"Embedding advantage: +{p['emb_advantage']} comparisons\n")
            f.write("\nTEXT:\n")
            f.write(p['text'][:2000])  # Limit for very long poems
            if len(p['text']) > 2000:
                f.write("\n... [truncated]")
            f.write("\n\n")

    print(f"Embedding-distinctive poems saved to: {emb_file}")

    stylo_file = RESULTS_DIR / f"distinctive_for_stylometry_italian.txt"
    with open(stylo_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("POEMS DISTINCTIVE FOR STYLOMETRY (ITALIAN)\n")
        f.write("=" * 70 + "\n\n")

        for i, p in enumerate(stylo_distinctive):
            f.write("-" * 70 + "\n")
            f.write(f"RANK #{i+1}\n")
            f.write(f"Author: {p['author']}\n")
            f.write(f"Word count: {p['word_count']}\n")
            f.write(f"Embedding accuracy: {p['emb_accuracy']:.0%}\n")
            f.write(f"Stylometry accuracy: {p['stylo_accuracy']:.0%}\n")
            f.write(f"Stylometry advantage: +{-p['emb_advantage']} comparisons\n")
            f.write("\nTEXT:\n")
            f.write(p['text'][:2000])
            if len(p['text']) > 2000:
                f.write("\n... [truncated]")
            f.write("\n\n")

    print(f"Stylometry-distinctive poems saved to: {stylo_file}")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
