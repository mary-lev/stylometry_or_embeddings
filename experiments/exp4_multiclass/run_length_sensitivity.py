#!/usr/bin/env python3
"""
Poem Length Sensitivity Analysis

Purpose: Test whether short poem lengths affect classification accuracy.
Eder (2017) suggests minimum 3,000 words for stable stylometric features in poetry.
Our median poem is only 81 words.

Method:
1. Report poem length statistics
2. Compare single-poem vs concatenated-poem classification
3. Test different concatenation sizes (1, 5, 10, 20 poems)
4. For embeddings: average embeddings of concatenated poems
5. For stylometry: concatenate texts, then extract features

Reference: Eder, M. (2017). Short samples in authorship attribution: A new approach.
"""

import sys
import json
import argparse
import warnings
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from clean_dataset import load_clean_dataset
from statistical_tests import bootstrap_ci

# Suppress warnings
warnings.filterwarnings('ignore')

# Constants
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# The published dataset is fixed at 200 poems per author (see data/README.md);
# every result in the paper is computed at this size.
N_PER_AUTHOR = 200



def get_poem_lengths(poems):
    """Calculate word counts for all poems."""
    return np.array([len(p['text'].split()) for p in poems])


def create_concatenated_samples(poems, embeddings, labels, concat_size, seed=42):
    """
    Create samples by concatenating multiple poems per author.

    For embeddings: average the embeddings
    For text: concatenate the texts

    Returns samples with approximately equal representation per author.
    """
    np.random.seed(seed)

    # Group poems by author
    author_poems = defaultdict(list)
    author_embeddings = defaultdict(list)

    for i, (poem, emb, label) in enumerate(zip(poems, embeddings, labels)):
        author_poems[label].append(poem)
        author_embeddings[label].append(emb)

    # Create concatenated samples
    new_texts = []
    new_embeddings = []
    new_labels = []
    new_word_counts = []

    for label in sorted(author_poems.keys()):
        poems_list = author_poems[label]
        embs_list = author_embeddings[label]

        # Shuffle poems for this author
        indices = np.random.permutation(len(poems_list))

        # Create groups of concat_size
        for i in range(0, len(indices) - concat_size + 1, concat_size):
            group_indices = indices[i:i + concat_size]

            # Concatenate texts
            concat_text = '\n\n'.join([poems_list[j]['text'] for j in group_indices])

            # Average embeddings
            avg_emb = np.mean([embs_list[j] for j in group_indices], axis=0)

            new_texts.append(concat_text)
            new_embeddings.append(avg_emb)
            new_labels.append(label)
            new_word_counts.append(len(concat_text.split()))

    return (new_texts, np.array(new_embeddings), np.array(new_labels),
            np.array(new_word_counts))


def run_embedding_classification(embeddings, labels, n_folds=5, seed=42):
    """Run classification using embeddings."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(embeddings)

    clf = LogisticRegression(max_iter=2000, solver='lbfgs')
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scores = cross_val_score(clf, X_scaled, labels, cv=cv, scoring='accuracy')

    return float(np.mean(scores)), float(np.std(scores)), list(scores)


def run_stylometry_classification(texts, labels, n_folds=5, seed=42):
    """Run classification using char 3-grams (traditional stylometry)."""
    # TF-IDF char 3-grams
    vectorizer = TfidfVectorizer(
        analyzer='char',
        ngram_range=(3, 3),
        max_features=2000,
        lowercase=True
    )

    X = vectorizer.fit_transform(texts)

    clf = LogisticRegression(max_iter=2000, solver='lbfgs')
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    # Need to handle sparse matrix
    from sklearn.pipeline import Pipeline
    pipe = Pipeline([
        ('scaler', StandardScaler(with_mean=False)),
        ('clf', LogisticRegression(max_iter=2000, solver='lbfgs'))
    ])

    scores = cross_val_score(pipe, X, labels, cv=cv, scoring='accuracy')

    return float(np.mean(scores)), float(np.std(scores)), list(scores)


def main():
    parser = argparse.ArgumentParser(description="Poem Length Sensitivity Analysis")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--embedding-model', type=str, default='gemini',
                        choices=['openai', 'gemini'])
    parser.add_argument('--n-folds', type=int, default=5)
    args = parser.parse_args()

    print("=" * 70)
    print("POEM LENGTH SENSITIVITY ANALYSIS")
    print("=" * 70)
    print("\nEder (2017): Minimum 3,000 words recommended for poetry attribution")
    print("Question: Does concatenating poems improve accuracy?")

    # Load data
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
    author_list = data['author_list']

    n_samples = len(poems)
    n_authors = len(author_list)
    chance = 1 / n_authors

    # Poem length statistics
    word_counts = get_poem_lengths(poems)

    print(f"\n[Poem Length Statistics]")
    print(f"  Authors: {n_authors}")
    print(f"  Poems: {n_samples}")
    print(f"  Mean length: {np.mean(word_counts):.1f} words")
    print(f"  Median length: {np.median(word_counts):.1f} words")
    print(f"  Std: {np.std(word_counts):.1f} words")
    print(f"  Range: {np.min(word_counts)} - {np.max(word_counts)} words")
    print(f"  Eder minimum (3000): {sum(word_counts >= 3000)} poems ({sum(word_counts >= 3000)/len(word_counts)*100:.1f}%)")
    print(f"  Poems needed for 3000 words: ~{int(3000 / np.median(word_counts))}")

    results = {
        'timestamp': datetime.now().isoformat(),
        'settings': {
            'n_authors': n_authors,
            'n_per_author': N_PER_AUTHOR,
            'embedding_model': args.embedding_model,
            'n_folds': args.n_folds,
            'seed': args.seed,
            'chance_level': chance
        },
        'poem_length_stats': {
            'mean': float(np.mean(word_counts)),
            'median': float(np.median(word_counts)),
            'std': float(np.std(word_counts)),
            'min': int(np.min(word_counts)),
            'max': int(np.max(word_counts)),
            'q25': float(np.percentile(word_counts, 25)),
            'q75': float(np.percentile(word_counts, 75)),
            'eder_minimum': 3000,
            'poems_meeting_eder': int(sum(word_counts >= 3000))
        },
        'experiments': {}
    }

    # Concatenation sizes to test
    concat_sizes = [1, 5, 10, 20, 40]

    print("\n" + "=" * 70)
    print("SENSITIVITY ANALYSIS: Embeddings")
    print("=" * 70)

    embedding_results = []

    for concat_size in concat_sizes:
        print(f"\n[Concatenation size: {concat_size} poems]")

        if concat_size == 1:
            # Single poems
            texts = [p['text'] for p in poems]
            embs = embeddings
            lbls = labels
            wc = word_counts
        else:
            # Concatenated poems
            texts, embs, lbls, wc = create_concatenated_samples(
                poems, embeddings, labels, concat_size, seed=args.seed)

        n_samples_concat = len(lbls)
        mean_wc = np.mean(wc)

        print(f"  Samples: {n_samples_concat}")
        print(f"  Mean words/sample: {mean_wc:.1f}")
        print(f"  Samples per author: ~{n_samples_concat // n_authors}")

        # Check if we have enough samples
        if n_samples_concat < n_authors * 2:
            print(f"  Skipped: insufficient samples")
            continue

        # Run classification
        acc, std, scores = run_embedding_classification(embs, lbls,
                                                        n_folds=args.n_folds,
                                                        seed=args.seed)

        print(f"  Accuracy: {acc:.1%} ± {std:.1%} (lift: {acc/chance:.1f}×)")

        embedding_results.append({
            'concat_size': concat_size,
            'n_samples': n_samples_concat,
            'samples_per_author': n_samples_concat // n_authors,
            'mean_words': float(mean_wc),
            'accuracy': acc,
            'accuracy_std': std,
            'lift': acc / chance,
            'fold_scores': scores
        })

    results['experiments']['embeddings'] = embedding_results

    print("\n" + "=" * 70)
    print("SENSITIVITY ANALYSIS: Stylometry (Char 3-grams)")
    print("=" * 70)

    stylometry_results = []

    for concat_size in concat_sizes:
        print(f"\n[Concatenation size: {concat_size} poems]")

        if concat_size == 1:
            texts = [p['text'] for p in poems]
            lbls = labels
            wc = word_counts
        else:
            texts, _, lbls, wc = create_concatenated_samples(
                poems, embeddings, labels, concat_size, seed=args.seed)

        n_samples_concat = len(lbls)
        mean_wc = np.mean(wc)

        print(f"  Samples: {n_samples_concat}")
        print(f"  Mean words/sample: {mean_wc:.1f}")

        if n_samples_concat < n_authors * 2:
            print(f"  Skipped: insufficient samples")
            continue

        # Run classification
        acc, std, scores = run_stylometry_classification(texts, lbls,
                                                         n_folds=args.n_folds,
                                                         seed=args.seed)

        print(f"  Accuracy: {acc:.1%} ± {std:.1%} (lift: {acc/chance:.1f}×)")

        stylometry_results.append({
            'concat_size': concat_size,
            'n_samples': n_samples_concat,
            'samples_per_author': n_samples_concat // n_authors,
            'mean_words': float(mean_wc),
            'accuracy': acc,
            'accuracy_std': std,
            'lift': acc / chance,
            'fold_scores': scores
        })

    results['experiments']['stylometry'] = stylometry_results

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n  {'Concat':<8} {'Words':<10} {'Embeddings':<15} {'Stylometry':<15}")
    print(f"  {'-' * 55}")

    for i, concat_size in enumerate(concat_sizes):
        emb_res = next((r for r in embedding_results if r['concat_size'] == concat_size), None)
        sty_res = next((r for r in stylometry_results if r['concat_size'] == concat_size), None)

        if emb_res and sty_res:
            print(f"  {concat_size:<8} {emb_res['mean_words']:<10.0f} "
                  f"{emb_res['accuracy']:.1%} ({emb_res['lift']:.1f}×)   "
                  f"{sty_res['accuracy']:.1%} ({sty_res['lift']:.1f}×)")

    # Interpretation
    print("\n[Interpretation]")

    if len(embedding_results) >= 2:
        single = next(r for r in embedding_results if r['concat_size'] == 1)
        best = max(embedding_results, key=lambda x: x['accuracy'])

        if best['concat_size'] > 1:
            improvement = best['accuracy'] - single['accuracy']
            print(f"  Embeddings: +{improvement:.1%} when concatenating {best['concat_size']} poems")
            print(f"    Single poem: {single['accuracy']:.1%}")
            print(f"    Best ({best['concat_size']} poems, ~{best['mean_words']:.0f} words): {best['accuracy']:.1%}")
        else:
            print(f"  Embeddings: No improvement from concatenation")

    if len(stylometry_results) >= 2:
        single = next(r for r in stylometry_results if r['concat_size'] == 1)
        best = max(stylometry_results, key=lambda x: x['accuracy'])

        if best['concat_size'] > 1:
            improvement = best['accuracy'] - single['accuracy']
            print(f"  Stylometry: +{improvement:.1%} when concatenating {best['concat_size']} poems")
            print(f"    Single poem: {single['accuracy']:.1%}")
            print(f"    Best ({best['concat_size']} poems, ~{best['mean_words']:.0f} words): {best['accuracy']:.1%}")
        else:
            print(f"  Stylometry: No improvement from concatenation")

    # Eder threshold analysis
    print(f"\n[Eder Threshold Analysis]")
    eder_concat = int(np.ceil(3000 / np.median(word_counts)))
    print(f"  To reach 3000 words: concatenate ~{eder_concat} poems")

    eder_result = next((r for r in embedding_results if r['concat_size'] >= eder_concat), None)
    if eder_result:
        print(f"  At {eder_result['concat_size']} poems ({eder_result['mean_words']:.0f} words):")
        print(f"    Embeddings: {eder_result['accuracy']:.1%}")
    else:
        print(f"  Not tested (would need concat_size >= {eder_concat})")

    # Save results
    output_file = RESULTS_DIR / f"length_sensitivity_{args.embedding_model}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults saved to: {output_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
