#!/usr/bin/env python3
"""
Frequency Band Analysis (Rybicki & Eder "Deeper Delta")

Purpose: Test whether different word frequency bands capture different information:
- Top 100 MFW: Grammar, syntax (function words)
- 100-500 MFW: Style, register
- 500-2000 MFW: Topic, vocabulary (content words)

Reference: Rybicki & Eder (2011) "Deeper Delta across genres and languages"
"""

import sys
import json
import argparse
import warnings
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict
from itertools import combinations

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from clean_dataset import load_clean_dataset

# Suppress warnings
warnings.filterwarnings('ignore')

# Constants
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Frequency bands as per Rybicki & Eder
BANDS = {
    'band_1_100': (0, 100),      # Grammar, syntax
    'band_100_500': (100, 500),   # Style, register
    'band_500_2000': (500, 2000), # Topic, vocabulary
    'band_2000_5000': (2000, 5000) # Extended vocabulary
}


def build_vocabulary_by_frequency(texts):
    """Build vocabulary sorted by frequency."""
    word_counts = Counter()
    for text in texts:
        words = text.lower().split()
        word_counts.update(words)

    # Sort by frequency (most common first)
    sorted_vocab = [word for word, count in word_counts.most_common()]
    return sorted_vocab, word_counts


def extract_band_features(texts, vocabulary, band_start, band_end):
    """Extract TF-IDF features for a specific frequency band."""
    # Get words in this band
    band_vocab = vocabulary[band_start:band_end]

    if len(band_vocab) == 0:
        return None, None

    # Create vectorizer with this vocabulary
    vectorizer = TfidfVectorizer(
        vocabulary={word: i for i, word in enumerate(band_vocab)},
        lowercase=True
    )

    X = vectorizer.fit_transform(texts)
    return X, band_vocab


def analyze_band(X, y, band_name):
    """Analyze classification performance for a band."""
    if X is None:
        return {'accuracy': 0, 'accuracy_std': 0}

    clf = LogisticRegression(
        penalty='l2',
        solver='lbfgs',
        C=1.0,
        max_iter=500,
        random_state=42
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')

    return {
        'accuracy': float(np.mean(scores)),
        'accuracy_std': float(np.std(scores)),
        'n_features': X.shape[1]
    }


def analyze_band_pairwise(texts1, texts2, vocabulary, band_start, band_end):
    """Analyze a single band for a pair of authors."""
    all_texts = texts1 + texts2
    y = np.array([0] * len(texts1) + [1] * len(texts2))

    X, band_vocab = extract_band_features(all_texts, vocabulary, band_start, band_end)

    if X is None or X.shape[1] < 10:
        return None, None

    clf = LogisticRegression(
        penalty='l2',
        solver='lbfgs',
        C=1.0,
        max_iter=500,
        random_state=42
    )

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    try:
        scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
        accuracy = np.mean(scores)
    except:
        return None, None

    # Get top discriminative words
    clf.fit(X, y)
    coefficients = clf.coef_[0]

    sorted_idx = np.argsort(coefficients)
    top_words = {
        'author1': [(band_vocab[i], float(coefficients[i])) for i in sorted_idx[:10]],
        'author2': [(band_vocab[i], float(coefficients[i])) for i in sorted_idx[-10:][::-1]]
    }

    return accuracy, top_words


def categorize_word(word, word_counts, total_vocab_size):
    """Categorize a word by its likely linguistic function."""
    # Russian function words (approximate)
    function_words = {
        'и', 'в', 'не', 'на', 'я', 'что', 'он', 'с', 'как', 'а', 'то', 'все',
        'она', 'так', 'его', 'но', 'да', 'ты', 'к', 'у', 'же', 'вы', 'за',
        'бы', 'по', 'только', 'её', 'мне', 'было', 'вот', 'от', 'меня', 'ещё',
        'нет', 'о', 'из', 'ему', 'теперь', 'когда', 'уже', 'для', 'вас', 'ни',
        'себя', 'ничего', 'им', 'тогда', 'кто', 'этот', 'того', 'потому', 'этого',
        'этой', 'какой', 'где', 'есть', 'мы', 'они', 'их', 'без', 'при', 'над',
        'под', 'через', 'после', 'между', 'или', 'если', 'чтобы', 'хотя', 'тоже',
        'даже', 'ли', 'ведь', 'почти', 'будто', 'там', 'здесь', 'потом', 'опять',
        'раз', 'где', 'куда', 'сюда', 'туда', 'тут', 'оттуда', 'откуда'
    }

    if word in function_words:
        return 'function'

    # Check frequency rank
    rank = list(word_counts.keys()).index(word) if word in word_counts else total_vocab_size

    if rank < 100:
        return 'high_freq'
    elif rank < 500:
        return 'mid_freq'
    else:
        return 'low_freq'


def main():
    parser = argparse.ArgumentParser(description="Frequency Band Analysis")
    parser.add_argument('--n-per-author', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    print("=" * 70)
    print("FREQUENCY BAND ANALYSIS (Rybicki & Eder 'Deeper Delta')")
    print("=" * 70)

    # Load data
    print(f"\n[Loading dataset...]")
    data = load_clean_dataset(
        n_per_author=args.n_per_author,
        seed=args.seed,
        include_poems=True,
        embedding_model='gemini'
    )

    poems = data['poems']
    labels = data['labels']
    idx_to_author = data['idx_to_author']
    author_names = [idx_to_author[i] for i in range(len(idx_to_author))]

    # Extract texts
    texts = [p['text'] for p in poems]

    print(f"  Authors: {len(author_names)}")
    print(f"  Poems: {len(texts)}")

    # Build global vocabulary
    print(f"\n[Building vocabulary by frequency...]")
    vocabulary, word_counts = build_vocabulary_by_frequency(texts)
    print(f"  Total vocabulary: {len(vocabulary)}")

    # Show examples from each band
    print(f"\n[Sample words from each band:]")
    for band_name, (start, end) in BANDS.items():
        band_words = vocabulary[start:min(end, len(vocabulary))][:10]
        print(f"  {band_name}: {', '.join(band_words)}")

    # Group texts by author
    author_texts = defaultdict(list)
    for poem, label in zip(poems, labels):
        author_texts[idx_to_author[label]].append(poem['text'])

    # Analyze all pairs for each band
    print(f"\n[Analyzing frequency bands across all author pairs...]")

    all_pairs = list(combinations(author_names, 2))
    band_results = {band: [] for band in BANDS.keys()}
    band_top_words = {band: defaultdict(list) for band in BANDS.keys()}

    for i, (author1, author2) in enumerate(all_pairs):
        if (i + 1) % 100 == 0:
            print(f"    Progress: {i+1}/{len(all_pairs)}")

        texts1 = author_texts[author1]
        texts2 = author_texts[author2]

        for band_name, (start, end) in BANDS.items():
            accuracy, top_words = analyze_band_pairwise(
                texts1, texts2, vocabulary, start, end
            )
            if accuracy is not None:
                band_results[band_name].append(accuracy)

                # Collect top words
                if top_words:
                    for word, coef in top_words['author1']:
                        band_top_words[band_name][author1].append((word, coef))
                    for word, coef in top_words['author2']:
                        band_top_words[band_name][author2].append((word, coef))

    # Report results
    print("\n" + "=" * 70)
    print("RESULTS BY FREQUENCY BAND")
    print("=" * 70)

    print("\n### Classification Accuracy by Band")
    print("\n| Band | Range | Accuracy | Expected Content |")
    print("|------|-------|----------|------------------|")

    band_summaries = {}
    for band_name, (start, end) in BANDS.items():
        accs = band_results[band_name]
        if accs:
            mean_acc = np.mean(accs)
            std_acc = np.std(accs)

            expected = {
                'band_1_100': 'Grammar, syntax',
                'band_100_500': 'Style, register',
                'band_500_2000': 'Topic, vocabulary',
                'band_2000_5000': 'Extended vocabulary'
            }

            print(f"| {band_name} | {start}-{end} | {mean_acc:.1%} ± {std_acc:.1%} | {expected[band_name]} |")

            band_summaries[band_name] = {
                'range': f"{start}-{end}",
                'accuracy_mean': float(mean_acc),
                'accuracy_std': float(std_acc),
                'n_pairs': len(accs)
            }

    # Find most stable discriminative words per band
    print("\n" + "=" * 70)
    print("MOST DISCRIMINATIVE WORDS BY BAND")
    print("=" * 70)

    for band_name, (start, end) in BANDS.items():
        print(f"\n### {band_name} ({start}-{end})")

        # Count word appearances across authors
        word_author_counts = Counter()
        for author, words in band_top_words[band_name].items():
            for word, _ in words:
                word_author_counts[word] += 1

        # Most common discriminative words
        print(f"  Most discriminative words (appear for multiple authors):")
        for word, count in word_author_counts.most_common(10):
            freq_rank = vocabulary.index(word) if word in vocabulary else -1
            print(f"    '{word}': {count} authors (rank {freq_rank})")

    # Compare bands
    print("\n" + "=" * 70)
    print("BAND COMPARISON")
    print("=" * 70)

    sorted_bands = sorted(band_summaries.items(),
                          key=lambda x: x[1]['accuracy_mean'],
                          reverse=True)

    print("\n  Bands ranked by discriminative power:")
    for i, (band_name, summary) in enumerate(sorted_bands, 1):
        print(f"    {i}. {band_name}: {summary['accuracy_mean']:.1%}")

    # Analysis
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    if sorted_bands:
        best_band = sorted_bands[0][0]
        print(f"\n  Most discriminative band: {best_band}")

        if 'band_1_100' == best_band:
            print("  → Function words (grammar/syntax) most distinguish authors")
            print("  → Consistent with Rybicki & Eder's findings for prose")
        elif 'band_100_500' == best_band:
            print("  → Style/register words most distinguish authors")
        elif 'band_500_2000' == best_band:
            print("  → Topic/vocabulary words most distinguish authors")
            print("  → Poetry may differ from prose (more thematic variation)")
        else:
            print("  → Extended vocabulary most discriminative")
            print("  → Suggests lexical richness matters for poetry")

    # Rybicki & Eder comparison
    print("\n  Rybicki & Eder (2011) findings for prose:")
    print("    - Band 1-100: Grammar, syntax (function words)")
    print("    - Band 100-500: Style, register")
    print("    - Band 500-2000: Topic, vocabulary")
    print("    - They found TOP 100 most stable for authorship")

    print("\n  Our findings for Russian poetry:")
    for band_name, summary in sorted_bands[:3]:
        print(f"    - {band_name}: {summary['accuracy_mean']:.1%} accuracy")

    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'settings': {
            'n_authors': len(author_names),
            'n_pairs': len(all_pairs),
            'n_poems': len(texts),
            'vocabulary_size': len(vocabulary)
        },
        'bands': BANDS,
        'band_summaries': band_summaries,
        'sample_words': {
            band: vocabulary[start:min(end, len(vocabulary))][:20]
            for band, (start, end) in BANDS.items()
        },
        'interpretation': {
            'best_band': sorted_bands[0][0] if sorted_bands else None,
            'ranking': [b[0] for b in sorted_bands]
        }
    }

    output_file = RESULTS_DIR / 'frequency_band_analysis.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_file}")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
