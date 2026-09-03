#!/usr/bin/env python3
"""
Analyze Method Differences: Embeddings vs Stylometry

Purpose:
1. Find poems distinctive for EMBEDDINGS (correctly classified by embeddings,
   but misclassified by stylometry)
2. Find poems distinctive for STYLOMETRY (correctly classified by stylometry,
   but misclassified by embeddings)
3. Analyze if short poems are harder for both methods, one method, or neither

This reveals what each method captures that the other doesn't.
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from itertools import combinations
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.stats as stats

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from clean_dataset import load_clean_dataset

# Constants
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# The published dataset is fixed at 200 poems per author (see data/README.md);
# every result in the paper is computed at this size.
N_PER_AUTHOR = 200



class StyleometryExtractor:
    """Extract stylometry features (char n-grams + word n-grams)."""

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
    parser = argparse.ArgumentParser(description="Analyze Method Differences")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cv-folds', type=int, default=5)
    parser.add_argument('--embedding-model', type=str, default='gemini',
                        choices=['openai', 'gemini'])
    parser.add_argument('--top-n', type=int, default=30,
                        help='Number of top poems to show')
    args = parser.parse_args()

    print("=" * 70)
    print("ANALYZE METHOD DIFFERENCES: EMBEDDINGS vs STYLOMETRY")
    print("=" * 70)

    # Load dataset
    print(f"\n[Loading dataset with {args.embedding_model} embeddings...]")
    data = load_clean_dataset(
        n_per_author=N_PER_AUTHOR,
        seed=args.seed,
        include_poems=True,
        embedding_model=args.embedding_model
    )

    poems = data['poems']
    embeddings = data['embeddings']
    labels = data['labels']
    authors = data['author_list']

    n_authors = len(authors)
    n_poems = len(poems)

    print(f"  Authors: {n_authors}")
    print(f"  Total poems: {n_poems}")

    author_to_idx = {a: i for i, a in enumerate(authors)}

    # Track per-poem statistics
    poem_stats = []
    for i, poem in enumerate(poems):
        author_idx = int(labels[i])  # FIX: poems are in selected_indices order, not author blocks
        poem_stats.append({
            'global_idx': i,
            'author': authors[author_idx],
            'word_count': len(poem['text'].split()),
            'text': poem['text'],
            'emb_correct': 0,
            'emb_wrong': 0,
            'stylo_correct': 0,
            'stylo_wrong': 0,
            'emb_only_correct': 0,  # embedding correct, stylometry wrong
            'stylo_only_correct': 0,  # stylometry correct, embedding wrong
            'both_correct': 0,
            'both_wrong': 0
        })

    # Process all pairs
    pairs = list(combinations(authors, 2))
    n_pairs = len(pairs)
    print(f"\n[Processing {n_pairs} pairs...]")

    for pair_idx, (a1, a2) in enumerate(pairs):
        if (pair_idx + 1) % 100 == 0:
            print(f"  Progress: {pair_idx + 1}/{n_pairs} pairs...")

        idx1 = author_to_idx[a1]
        idx2 = author_to_idx[a2]

        # FIX: select by label mask, not positional blocks (data is in selected_indices order)
        indices1 = list(np.where(labels == idx1)[0])
        indices2 = list(np.where(labels == idx2)[0])

        pair_indices = indices1 + indices2
        pair_embeddings = embeddings[pair_indices]
        pair_labels = np.array([0] * len(indices1) + [1] * len(indices2))
        pair_texts = [poems[i]['text'] for i in pair_indices]

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
    n_comparisons = n_authors - 1  # Each poem participates in 28 comparisons

    for p in poem_stats:
        p['emb_accuracy'] = p['emb_correct'] / n_comparisons
        p['stylo_accuracy'] = p['stylo_correct'] / n_comparisons
        p['emb_advantage'] = p['emb_only_correct'] - p['stylo_only_correct']
        # Positive = embedding better, Negative = stylometry better

    print(f"\n[Analyzing results...]")

    # ========================================================================
    # ANALYSIS 1: Poems where methods diverge
    # ========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 1: METHOD-SPECIFIC DISTINCTIVE POEMS")
    print("=" * 70)

    # Sort by embedding advantage (embedding better than stylometry)
    sorted_by_emb_advantage = sorted(poem_stats, key=lambda x: x['emb_advantage'], reverse=True)

    print(f"\n### Poems DISTINCTIVE for EMBEDDINGS")
    print("(Correctly classified by embeddings, misclassified by stylometry)")
    print(f"\n| Rank | Author | Words | Emb Acc | Stylo Acc | Emb Advantage |")
    print(f"|------|--------|-------|---------|-----------|---------------|")

    emb_distinctive = [p for p in sorted_by_emb_advantage if p['emb_advantage'] > 0][:args.top_n]
    for i, p in enumerate(emb_distinctive[:15]):
        print(f"| {i+1} | {p['author'][:15]} | {p['word_count']} | {p['emb_accuracy']:.0%} | {p['stylo_accuracy']:.0%} | +{p['emb_advantage']} |")

    # Sort by stylometry advantage
    sorted_by_stylo_advantage = sorted(poem_stats, key=lambda x: -x['emb_advantage'], reverse=True)

    print(f"\n### Poems DISTINCTIVE for STYLOMETRY")
    print("(Correctly classified by stylometry, misclassified by embeddings)")
    print(f"\n| Rank | Author | Words | Emb Acc | Stylo Acc | Stylo Advantage |")
    print(f"|------|--------|-------|---------|-----------|-----------------|")

    stylo_distinctive = [p for p in sorted_by_stylo_advantage if p['emb_advantage'] < 0][:args.top_n]
    for i, p in enumerate(stylo_distinctive[:15]):
        print(f"| {i+1} | {p['author'][:15]} | {p['word_count']} | {p['emb_accuracy']:.0%} | {p['stylo_accuracy']:.0%} | +{-p['emb_advantage']} |")

    # ========================================================================
    # ANALYSIS 2: Short poem analysis
    # ========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 2: SHORT POEM DIFFICULTY")
    print("=" * 70)

    # Define word count bins
    bins = [
        (0, 30, "Very short (≤30)"),
        (31, 50, "Short (31-50)"),
        (51, 80, "Medium-short (51-80)"),
        (81, 120, "Medium (81-120)"),
        (121, 200, "Long (121-200)"),
        (201, float('inf'), "Very long (>200)")
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

    # Statistical test: correlation between length and accuracy
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
    # ANALYSIS 3: Are ALL short poems hard, or only SOME?
    # ========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS 3: WHICH SHORT POEMS ARE HARD?")
    print("=" * 70)

    short_poems = [p for p in poem_stats if p['word_count'] <= 50]
    short_poems_sorted = sorted(short_poems, key=lambda x: x['emb_accuracy'] + x['stylo_accuracy'], reverse=True)

    # Best short poems (high accuracy despite being short)
    print(f"\n### Short Poems (≤50 words) that are EASY to classify")
    print(f"\n| Author | Words | Emb Acc | Stylo Acc | Sample |")
    print(f"|--------|-------|---------|-----------|--------|")

    for p in short_poems_sorted[:10]:
        sample = p['text'][:40].replace('\n', ' ') + '...'
        print(f"| {p['author'][:15]} | {p['word_count']} | {p['emb_accuracy']:.0%} | {p['stylo_accuracy']:.0%} | {sample} |")

    # Worst short poems
    print(f"\n### Short Poems (≤50 words) that are HARD to classify")
    print(f"\n| Author | Words | Emb Acc | Stylo Acc | Sample |")
    print(f"|--------|-------|---------|-----------|--------|")

    for p in short_poems_sorted[-10:]:
        sample = p['text'][:40].replace('\n', ' ') + '...'
        print(f"| {p['author'][:15]} | {p['word_count']} | {p['emb_accuracy']:.0%} | {p['stylo_accuracy']:.0%} | {sample} |")

    # What distinguishes easy vs hard short poems?
    easy_short = [p for p in short_poems if p['emb_accuracy'] >= 0.8 and p['stylo_accuracy'] >= 0.8]
    hard_short = [p for p in short_poems if p['emb_accuracy'] <= 0.6 and p['stylo_accuracy'] <= 0.6]

    print(f"\n### Comparing Easy vs Hard Short Poems")
    print(f"  Easy short poems (≥80% both): {len(easy_short)}")
    print(f"  Hard short poems (≤60% both): {len(hard_short)}")

    if easy_short and hard_short:
        easy_authors = [p['author'] for p in easy_short]
        hard_authors = [p['author'] for p in hard_short]

        from collections import Counter
        easy_author_counts = Counter(easy_authors)
        hard_author_counts = Counter(hard_authors)

        print(f"\n  Authors with most EASY short poems:")
        for author, count in easy_author_counts.most_common(5):
            print(f"    {author}: {count}")

        print(f"\n  Authors with most HARD short poems:")
        for author, count in hard_author_counts.most_common(5):
            print(f"    {author}: {count}")

    # ========================================================================
    # Save results
    # ========================================================================
    output = {
        'timestamp': datetime.now().isoformat(),
        'settings': {
            'n_per_author': N_PER_AUTHOR,
            'n_authors': n_authors,
            'n_pairs': n_pairs,
            'embedding_model': args.embedding_model
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
        } for p in stylo_distinctive],
        'short_poem_analysis': {
            'easy_short_count': len(easy_short) if 'easy_short' in dir() else 0,
            'hard_short_count': len(hard_short) if 'hard_short' in dir() else 0,
        }
    }

    output_file = RESULTS_DIR / f"method_differences_{args.embedding_model}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_file}")

    # Save distinctive poems to readable files
    emb_file = RESULTS_DIR / f"distinctive_for_embeddings_{args.embedding_model}.txt"
    with open(emb_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("POEMS DISTINCTIVE FOR EMBEDDINGS\n")
        f.write("(Correctly classified by embeddings, misclassified by stylometry)\n")
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
            f.write(p['text'])
            f.write("\n\n")

    print(f"Embedding-distinctive poems saved to: {emb_file}")

    stylo_file = RESULTS_DIR / f"distinctive_for_stylometry_{args.embedding_model}.txt"
    with open(stylo_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("POEMS DISTINCTIVE FOR STYLOMETRY\n")
        f.write("(Correctly classified by stylometry, misclassified by embeddings)\n")
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
            f.write(p['text'])
            f.write("\n\n")

    print(f"Stylometry-distinctive poems saved to: {stylo_file}")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
