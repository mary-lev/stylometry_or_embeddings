#!/usr/bin/env python3
"""
Probe Embeddings for Discriminative Features

Purpose: Test whether embeddings encode the specific patterns we discovered
as most discriminative for authorship:
1. Char n-gram patterns (em-dash, ellipsis, hyphen rates)
2. Stylometry features (question_rate, exclam_rate, etc.)
3. Frequency band words (topic vocabulary)

This closes the loop: if embeddings encode these patterns with high R²,
then embeddings and char n-grams capture the SAME information.
"""

import sys
import json
import argparse
import warnings
import re
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter

from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from scipy import stats

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from clean_dataset import load_clean_dataset
from features import extract_tier1_surface

# Suppress warnings
warnings.filterwarnings('ignore')

# Constants
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ============================================================================
# Feature Extraction: Char N-gram Patterns
# ============================================================================

def extract_punctuation_patterns(text):
    """Extract the punctuation patterns we found most discriminative."""
    word_count = max(len(text.split()), 1)
    char_count = max(len(text), 1)

    features = {}

    # Em-dash patterns (Tsvetaeva's signature)
    features['em_dash_rate'] = text.count('—') / word_count
    features['em_dash_space_rate'] = text.count('— ') / word_count
    features['space_em_dash_rate'] = text.count(' —') / word_count

    # En-dash patterns (Merezhkovsky, Lokhvitskaya)
    features['en_dash_rate'] = text.count('–') / word_count

    # Regular hyphen patterns (Blok, Voloshin)
    features['hyphen_rate'] = text.count('-') / word_count
    features['space_hyphen_rate'] = text.count(' -') / word_count

    # Ellipsis patterns (Annensky, Baltrušaitis)
    features['ellipsis_unicode_rate'] = text.count('…') / word_count
    features['ellipsis_dots_rate'] = text.count('...') / word_count
    features['double_dot_rate'] = text.count('..') / word_count

    # Line-end punctuation (era markers)
    lines = text.split('\n')
    n_lines = max(len(lines), 1)

    line_end_semicolon = sum(1 for l in lines if l.rstrip().endswith(';'))
    line_end_exclam = sum(1 for l in lines if l.rstrip().endswith('!'))
    line_end_period = sum(1 for l in lines if l.rstrip().endswith('.'))

    features['line_end_semicolon_rate'] = line_end_semicolon / n_lines
    features['line_end_exclam_rate'] = line_end_exclam / n_lines
    features['line_end_period_rate'] = line_end_period / n_lines

    # Sentence-internal punctuation (prose-like)
    features['period_space_rate'] = text.count('. ') / word_count
    features['comma_space_rate'] = text.count(', ') / word_count

    return features


def extract_stylometry_features(poem):
    """Extract the stylometry features we found most discriminative."""
    text = poem['text']
    word_count = max(len(text.split()), 1)
    char_count = max(len(text), 1)

    features = {}

    # Punctuation rates (top discriminative)
    features['question_rate'] = text.count('?') / word_count
    features['exclam_rate'] = text.count('!') / word_count
    features['comma_rate'] = text.count(',') / word_count
    features['colon_rate'] = text.count(':') / word_count
    features['semicolon_rate'] = text.count(';') / word_count
    features['paren_rate'] = (text.count('(') + text.count(')')) / word_count
    features['quote_rate'] = (text.count('«') + text.count('»') +
                              text.count('"') + text.count("'")) / word_count

    # Word-level features
    words = text.split()
    unique_words = set(w.lower() for w in words)
    features['type_token_ratio'] = len(unique_words) / word_count if word_count > 0 else 0

    word_lengths = [len(w) for w in words]
    features['avg_word_length'] = np.mean(word_lengths) if word_lengths else 0

    # Line-level features
    lines = [l for l in text.split('\n') if l.strip()]
    n_lines = max(len(lines), 1)
    features['avg_line_length_words'] = word_count / n_lines
    features['avg_line_length_chars'] = char_count / n_lines

    # Case features
    features['upper_rate'] = sum(1 for c in text if c.isupper()) / char_count

    return features


def extract_frequency_band_features(text, band_vocab):
    """Extract features based on frequency band vocabulary."""
    words = text.lower().split()
    word_count = max(len(words), 1)

    features = {}

    for band_name, vocab_set in band_vocab.items():
        band_words = [w for w in words if w in vocab_set]
        features[f'{band_name}_coverage'] = len(band_words) / word_count
        features[f'{band_name}_unique'] = len(set(band_words)) / max(len(vocab_set), 1)

    return features


def build_frequency_band_vocab(texts):
    """Build vocabulary for frequency bands."""
    word_counts = Counter()
    for text in texts:
        words = text.lower().split()
        word_counts.update(words)

    sorted_words = [w for w, _ in word_counts.most_common()]

    return {
        'band_1_100': set(sorted_words[0:100]),
        'band_100_500': set(sorted_words[100:500]),
        'band_500_2000': set(sorted_words[500:2000]),
    }


# ============================================================================
# Probing
# ============================================================================

def probe_feature(embeddings, feature_values, feature_name, n_folds=5):
    """Probe embeddings for a single feature using Ridge regression."""
    # Handle NaN/Inf
    feature_values = np.nan_to_num(feature_values, nan=0, posinf=0, neginf=0)

    # Skip if no variance
    if np.std(feature_values) < 1e-10:
        return {
            'r2_mean': 0,
            'r2_std': 0,
            'pearson_r': 0,
            'spearman_r': 0,
            'feature_std': 0,
            'skipped': True
        }

    # Standardize
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    # Cross-validated R²
    cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    ridge = Ridge(alpha=1.0)

    r2_scores = cross_val_score(ridge, embeddings_scaled, feature_values,
                                cv=cv, scoring='r2')

    # Correlation (full data)
    ridge.fit(embeddings_scaled, feature_values)
    predictions = ridge.predict(embeddings_scaled)

    pearson_r, _ = stats.pearsonr(feature_values, predictions)
    spearman_r, _ = stats.spearmanr(feature_values, predictions)

    return {
        'r2_mean': float(np.mean(r2_scores)),
        'r2_std': float(np.std(r2_scores)),
        'pearson_r': float(pearson_r),
        'spearman_r': float(spearman_r),
        'feature_std': float(np.std(feature_values)),
        'feature_mean': float(np.mean(feature_values)),
        'skipped': False
    }


def main():
    parser = argparse.ArgumentParser(description="Probe Embeddings for Discriminative Features")
    parser.add_argument('--n-per-author', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--embedding-model', type=str, default='gemini',
                        choices=['openai', 'gemini'])
    args = parser.parse_args()

    print("=" * 70)
    print("PROBING EMBEDDINGS FOR DISCRIMINATIVE FEATURES")
    print("=" * 70)

    # Load data
    print(f"\n[Loading dataset with {args.embedding_model} embeddings...]")
    data = load_clean_dataset(
        n_per_author=args.n_per_author,
        seed=args.seed,
        include_poems=True,
        embedding_model=args.embedding_model
    )

    poems = data['poems']
    embeddings = data['embeddings']

    n_samples = len(poems)
    print(f"  Samples: {n_samples}")
    print(f"  Embedding dim: {embeddings.shape[1]}")

    # Extract all discriminative features
    print(f"\n[Extracting discriminative features...]")

    # 1. Punctuation patterns (from char n-gram analysis)
    print("  Extracting punctuation patterns...")
    punct_features = [extract_punctuation_patterns(p['text']) for p in poems]
    punct_names = list(punct_features[0].keys())

    # 2. Stylometry features
    print("  Extracting stylometry features...")
    stylo_features = [extract_stylometry_features(p) for p in poems]
    stylo_names = list(stylo_features[0].keys())

    # 3. Frequency band features
    print("  Building frequency band vocabulary...")
    texts = [p['text'] for p in poems]
    band_vocab = build_frequency_band_vocab(texts)

    print("  Extracting frequency band features...")
    band_features = [extract_frequency_band_features(p['text'], band_vocab) for p in poems]
    band_names = list(band_features[0].keys())

    # Combine all features
    all_feature_names = punct_names + stylo_names + band_names
    print(f"\n  Total features to probe: {len(all_feature_names)}")

    # Probe each feature
    print(f"\n[Probing embeddings for each feature...]")

    results = {}

    # Punctuation patterns
    print("\n### Punctuation Patterns (Char N-gram)")
    for fname in punct_names:
        values = np.array([f[fname] for f in punct_features])
        result = probe_feature(embeddings, values, fname)
        results[fname] = result
        if not result['skipped']:
            r2 = result['r2_mean']
            symbol = '***' if r2 > 0.3 else '**' if r2 > 0.1 else '*' if r2 > 0.05 else ''
            print(f"  {fname}: R² = {r2:.3f} {symbol}")

    # Stylometry features
    print("\n### Stylometry Features")
    for fname in stylo_names:
        values = np.array([f[fname] for f in stylo_features])
        result = probe_feature(embeddings, values, fname)
        results[fname] = result
        if not result['skipped']:
            r2 = result['r2_mean']
            symbol = '***' if r2 > 0.3 else '**' if r2 > 0.1 else '*' if r2 > 0.05 else ''
            print(f"  {fname}: R² = {r2:.3f} {symbol}")

    # Frequency band features
    print("\n### Frequency Band Features")
    for fname in band_names:
        values = np.array([f[fname] for f in band_features])
        result = probe_feature(embeddings, values, fname)
        results[fname] = result
        if not result['skipped']:
            r2 = result['r2_mean']
            symbol = '***' if r2 > 0.3 else '**' if r2 > 0.1 else '*' if r2 > 0.05 else ''
            print(f"  {fname}: R² = {r2:.3f} {symbol}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: What Do Embeddings Encode?")
    print("=" * 70)

    # Sort by R²
    sorted_results = sorted(
        [(name, r['r2_mean']) for name, r in results.items() if not r.get('skipped', False)],
        key=lambda x: x[1],
        reverse=True
    )

    print("\n### Top 15 Features Encoded by Embeddings")
    print("\n| Feature | R² | Category |")
    print("|---------|-----|----------|")

    for fname, r2 in sorted_results[:15]:
        if fname in punct_names:
            cat = 'Punctuation'
        elif fname in stylo_names:
            cat = 'Stylometry'
        else:
            cat = 'Freq Band'
        print(f"| {fname} | {r2:.3f} | {cat} |")

    # Category summary
    print("\n### Average R² by Category")

    punct_r2 = [results[f]['r2_mean'] for f in punct_names if not results[f].get('skipped')]
    stylo_r2 = [results[f]['r2_mean'] for f in stylo_names if not results[f].get('skipped')]
    band_r2 = [results[f]['r2_mean'] for f in band_names if not results[f].get('skipped')]

    print(f"\n  Punctuation patterns: R² = {np.mean(punct_r2):.3f} (n={len(punct_r2)})")
    print(f"  Stylometry features:  R² = {np.mean(stylo_r2):.3f} (n={len(stylo_r2)})")
    print(f"  Frequency bands:      R² = {np.mean(band_r2):.3f} (n={len(band_r2)})")

    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    high_r2 = [f for f, r2 in sorted_results if r2 > 0.1]
    low_r2 = [f for f, r2 in sorted_results if r2 < 0.05]

    print(f"\n  Features with R² > 0.1 (well encoded): {len(high_r2)}")
    for f in high_r2[:5]:
        print(f"    - {f}")

    print(f"\n  Features with R² < 0.05 (poorly encoded): {len(low_r2)}")
    for f in low_r2[:5]:
        print(f"    - {f}")

    # Key question
    print("\n### Key Question: Do embeddings encode discriminative char n-grams?")

    key_punct = ['em_dash_rate', 'en_dash_rate', 'hyphen_rate', 'ellipsis_unicode_rate']
    for f in key_punct:
        if f in results and not results[f].get('skipped'):
            r2 = results[f]['r2_mean']
            encoded = 'YES' if r2 > 0.1 else 'PARTIALLY' if r2 > 0.05 else 'NO'
            print(f"  {f}: R² = {r2:.3f} → {encoded}")

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'settings': {
            'embedding_model': args.embedding_model,
            'n_samples': n_samples,
            'embedding_dim': embeddings.shape[1]
        },
        'feature_results': results,
        'category_summary': {
            'punctuation': {
                'mean_r2': float(np.mean(punct_r2)),
                'n_features': len(punct_r2)
            },
            'stylometry': {
                'mean_r2': float(np.mean(stylo_r2)),
                'n_features': len(stylo_r2)
            },
            'frequency_bands': {
                'mean_r2': float(np.mean(band_r2)),
                'n_features': len(band_r2)
            }
        },
        'top_encoded': sorted_results[:20]
    }

    output_file = RESULTS_DIR / f'probe_discriminative_{args.embedding_model}.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_file}")

    print("\n" + "=" * 70)
    print("PROBING COMPLETE")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
