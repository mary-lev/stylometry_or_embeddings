#!/usr/bin/env python3
"""
EXPERIMENT 7 EXTENDED: Six-Tier Residualization Waterfall with Stylometry

Purpose: Address the critical gap where char n-grams explain 73.5% of embedding
variance (EXP8) but were missing from the original waterfall (18.5% R²).

New Tiers:
- Tier 5: Character n-grams (2-4) - captures orthographic patterns
- Tier 6: Word bigrams - captures collocational patterns

Expected outcome: Residual should drop from ~11.6× chance to ~3.6× chance,
revealing that embeddings primarily encode character-level patterns.

Method:
1. Baseline: Raw embeddings -> classify
2. Tier 1: Remove Surface (20 features) -> classify
3. Tier 2: Remove + Content (40 features) -> classify
4. Tier 3: Remove + Grammar (100 features) -> classify
5. Tier 4: Remove + Prosody (15 features) -> classify
6. Tier 5: Remove + Char n-grams (~2000 features) -> classify  [NEW]
7. Tier 6: Remove + Word bigrams (~2000 features) -> classify  [NEW]
8. Kernel: Remove non-linear interactions -> classify
"""

import sys
import json
import argparse
import warnings
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import r2_score

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from clean_dataset import load_clean_dataset
from features import (
    extract_tier1_surface, extract_tier3_grammar, extract_tier4_prosody,
    Tier2Extractor, TIER_DEFINITIONS
)
from statistical_tests import (
    bootstrap_ci, bootstrap_ci_difference, permutation_test_accuracy,
    lift_over_chance, format_results_table
)

# Suppress warnings
warnings.filterwarnings('ignore')

# Constants
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# The published dataset is fixed at 200 poems per author (see data/README.md);
# every result in the paper is computed at this size.
N_PER_AUTHOR = 200



class CharNgramExtractor:
    """
    Character n-gram feature extractor for Tier 5.
    Fits TF-IDF vectorizer on train data only.
    """

    def __init__(self, n_range=(2, 4), max_features=2000):
        self.n_range = n_range
        self.max_features = max_features
        self.vectorizer = None

    def fit(self, poems):
        """Fit on training data only."""
        texts = [p['text'] for p in poems]
        self.vectorizer = TfidfVectorizer(
            analyzer='char',
            ngram_range=self.n_range,
            max_features=self.max_features,
            lowercase=True
        )
        self.vectorizer.fit(texts)
        return self

    def transform(self, poems):
        """Transform poems to char n-gram features."""
        texts = [p['text'] for p in poems]
        return self.vectorizer.transform(texts).toarray()


class WordBigramExtractor:
    """
    Word bigram feature extractor for Tier 6.
    Fits TF-IDF vectorizer on train data only.

    Literature shows word bigrams (including punctuation) outperform MFW for Latin.
    """

    def __init__(self, max_features=2000, include_unigrams=False):
        self.max_features = max_features
        self.include_unigrams = include_unigrams
        self.vectorizer = None

    def fit(self, poems):
        """Fit on training data only."""
        texts = [p['text'] for p in poems]
        n_range = (1, 2) if self.include_unigrams else (2, 2)
        self.vectorizer = TfidfVectorizer(
            analyzer='word',
            ngram_range=n_range,
            max_features=self.max_features,
            lowercase=True
        )
        self.vectorizer.fit(texts)
        return self

    def transform(self, poems):
        """Transform poems to word bigram features."""
        texts = [p['text'] for p in poems]
        return self.vectorizer.transform(texts).toarray()


def run_waterfall_cv(poems, embeddings, y, tier_order, n_authors, n_folds=5, seed=42,
                     char_ngram_features=2000, word_bigram_features=2000):
    """
    Run the extended waterfall residualization experiment with PROPER CV.

    CRITICAL: All fitting (scaler, residualizer, extractors) is done
    on train data only within each fold.

    tier_order: list of tier names, e.g., ['tier1', 'tier2', 'tier3', 'tier4', 'tier5', 'tier6']
    y: numeric labels (already converted)
    n_authors: number of unique authors
    """
    chance_level = 1 / n_authors

    # Tiers that need fitting within CV
    fitted_tiers = {'tier2', 'tier5', 'tier6'}

    # Deterministic tiers (can be pre-extracted)
    det_tiers = [t for t in tier_order if t not in fitted_tiers]

    # Pre-extract deterministic features (doesn't require fitting)
    all_det_features = {}
    for tier in det_tiers:
        if tier == 'tier1':
            all_det_features[tier] = np.array([extract_tier1_surface(p) for p in poems])
        elif tier == 'tier3':
            all_det_features[tier] = np.array([extract_tier3_grammar(p) for p in poems])
        elif tier == 'tier4':
            all_det_features[tier] = np.array([extract_tier4_prosody(p) for p in poems])

    # Initialize results storage
    stages = ['baseline'] + [f'after_{t}' for t in tier_order] + ['after_kernel']
    fold_scores = {stage: [] for stage in stages}
    fold_r2 = {stage: [] for stage in stages}
    tier_feature_counts = {t: [] for t in tier_order}

    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(embeddings, y)):
        print(f"    Fold {fold_idx + 1}/{n_folds}...")

        # Split data
        emb_train, emb_test = embeddings[train_idx], embeddings[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        train_poems = [poems[i] for i in train_idx]
        test_poems = [poems[i] for i in test_idx]

        # === BASELINE: Raw embeddings ===
        scaler = StandardScaler()
        emb_train_scaled = scaler.fit_transform(emb_train)
        emb_test_scaled = scaler.transform(emb_test)

        clf = LogisticRegression(max_iter=2000, solver='lbfgs')
        clf.fit(emb_train_scaled, y_train)
        fold_scores['baseline'].append(clf.score(emb_test_scaled, y_test))
        fold_r2['baseline'].append(0.0)

        # === PROGRESSIVE RESIDUALIZATION ===
        current_emb_train = emb_train.copy()
        current_emb_test = emb_test.copy()
        cumul_feat_train = None
        cumul_feat_test = None

        # Fit extractors on TRAIN ONLY
        extractors = {}

        if 'tier2' in tier_order:
            extractors['tier2'] = Tier2Extractor(n_tfidf=20, n_topics=20)
            extractors['tier2'].fit(train_poems)

        if 'tier5' in tier_order:
            extractors['tier5'] = CharNgramExtractor(
                n_range=(2, 4), max_features=char_ngram_features)
            extractors['tier5'].fit(train_poems)

        if 'tier6' in tier_order:
            extractors['tier6'] = WordBigramExtractor(
                max_features=word_bigram_features, include_unigrams=False)
            extractors['tier6'].fit(train_poems)

        for tier in tier_order:
            # Extract features for this tier
            if tier in fitted_tiers:
                if tier not in extractors:
                    continue
                tier_feat_train = extractors[tier].transform(train_poems)
                tier_feat_test = extractors[tier].transform(test_poems)
            else:
                tier_feat_train = all_det_features[tier][train_idx]
                tier_feat_test = all_det_features[tier][test_idx]

            # Track feature counts
            tier_feature_counts[tier].append(tier_feat_train.shape[1])

            # Accumulate features
            if cumul_feat_train is None:
                cumul_feat_train = tier_feat_train
                cumul_feat_test = tier_feat_test
            else:
                cumul_feat_train = np.hstack([cumul_feat_train, tier_feat_train])
                cumul_feat_test = np.hstack([cumul_feat_test, tier_feat_test])

            # Scale features (fit on train only)
            feat_scaler = StandardScaler()
            feat_train_scaled = feat_scaler.fit_transform(
                np.nan_to_num(cumul_feat_train, nan=0.0))
            feat_test_scaled = feat_scaler.transform(
                np.nan_to_num(cumul_feat_test, nan=0.0))

            # Fit residualizer on TRAIN ONLY
            ridge = Ridge(alpha=1.0)
            ridge.fit(feat_train_scaled, current_emb_train)

            # Compute residuals for both train and test
            pred_train = ridge.predict(feat_train_scaled)
            pred_test = ridge.predict(feat_test_scaled)

            resid_train = current_emb_train - pred_train
            resid_test = current_emb_test - pred_test

            # R² on train (diagnostic)
            r2 = r2_score(current_emb_train, pred_train, multioutput='variance_weighted')

            # Scale residuals for classification
            resid_scaler = StandardScaler()
            resid_train_scaled = resid_scaler.fit_transform(resid_train)
            resid_test_scaled = resid_scaler.transform(resid_test)

            # Classify
            clf = LogisticRegression(max_iter=2000, solver='lbfgs')
            clf.fit(resid_train_scaled, y_train)
            score = clf.score(resid_test_scaled, y_test)

            fold_scores[f'after_{tier}'].append(score)
            fold_r2[f'after_{tier}'].append(r2)

            # Update current embeddings for next tier
            current_emb_train = resid_train
            current_emb_test = resid_test

        # === KERNEL RESIDUALIZATION ===
        feat_scaler = StandardScaler()
        feat_train_scaled = feat_scaler.fit_transform(
            np.nan_to_num(cumul_feat_train, nan=0.0))
        feat_test_scaled = feat_scaler.transform(
            np.nan_to_num(cumul_feat_test, nan=0.0))

        # Nystroem + Ridge pipeline (approximates Kernel Ridge)
        n_components = min(500, feat_train_scaled.shape[0] // 2)
        kernel_pipe = Pipeline([
            ('nystroem', Nystroem(kernel='rbf', gamma=0.1,
                                  n_components=n_components, random_state=seed)),
            ('ridge', Ridge(alpha=1.0))
        ])

        # Fit on RESIDUALIZED embeddings from linear tiers
        kernel_pipe.fit(feat_train_scaled, current_emb_train)

        pred_train = kernel_pipe.predict(feat_train_scaled)
        pred_test = kernel_pipe.predict(feat_test_scaled)

        resid_train = current_emb_train - pred_train
        resid_test = current_emb_test - pred_test

        r2_kernel = r2_score(current_emb_train, pred_train, multioutput='variance_weighted')

        # Scale and classify
        resid_scaler = StandardScaler()
        resid_train_scaled = resid_scaler.fit_transform(resid_train)
        resid_test_scaled = resid_scaler.transform(resid_test)

        clf = LogisticRegression(max_iter=2000, solver='lbfgs')
        clf.fit(resid_train_scaled, y_train)
        score = clf.score(resid_test_scaled, y_test)

        fold_scores['after_kernel'].append(score)
        fold_r2['after_kernel'].append(r2_kernel)

    # Aggregate results with bootstrap CIs
    results = {
        'n_authors': n_authors,
        'chance_level': chance_level,
        'tier_order': tier_order,
        'n_folds': n_folds,
        'stages': {},
        'fold_scores': {},
        'tier_feature_counts': {t: int(np.mean(counts)) for t, counts in tier_feature_counts.items()}
    }

    for stage in stages:
        if fold_scores[stage]:
            scores = fold_scores[stage]
            mean_acc, ci_low, ci_high = bootstrap_ci(scores, confidence=0.95)
            lift = lift_over_chance(mean_acc, n_authors)

            results['stages'][stage] = {
                'accuracy': mean_acc,
                'accuracy_std': float(np.std(scores)),
                'ci_95_low': ci_low,
                'ci_95_high': ci_high,
                'lift_over_chance': lift,
                'r2': float(np.mean(fold_r2[stage])),
                'r2_std': float(np.std(fold_r2[stage]))
            }
            results['fold_scores'][stage] = scores

    return results


def main():
    parser = argparse.ArgumentParser(description="EXP7 EXTENDED: Waterfall with Stylometry Tiers")
    parser.add_argument('--n-folds', type=int, default=5,
                        help='Number of CV folds (default: 5)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--embedding-model', type=str, default='gemini',
                        choices=['openai', 'gemini', 'qwen8b'],
                        help='Embedding model to use (default: gemini)')
    parser.add_argument('--char-ngram-features', type=int, default=2000,
                        help='Number of char n-gram features (default: 2000)')
    parser.add_argument('--word-bigram-features', type=int, default=2000,
                        help='Number of word bigram features (default: 2000)')
    parser.add_argument('--skip-tier2', action='store_true',
                        help='Skip Tier 2 (Content/TF-IDF)')
    args = parser.parse_args()

    print("=" * 70)
    print("EXPERIMENT 7 EXTENDED: Six-Tier Waterfall with Stylometry")
    print("=" * 70)
    print("\nPurpose: Add char n-grams and word bigrams to measure true residual")
    print("Expected: Residual should drop from ~11.6× chance to ~3.6× chance")

    # Load clean dataset
    print(f"\n[Loading clean dataset with {args.embedding_model} embeddings...]")
    data = load_clean_dataset(
        n_per_author=N_PER_AUTHOR,
        seed=args.seed,
        include_poems=True,
        embedding_model=args.embedding_model
    )

    poems = data['poems']
    embeddings = data['embeddings']
    labels = data['labels']
    unique_authors = data['author_list']

    n_samples = len(poems)
    n_authors = len(unique_authors)

    print(f"\n[Settings]")
    print(f"  Authors: {n_authors}")
    print(f"  Poems per author: {N_PER_AUTHOR}")
    print(f"  Total samples: {n_samples}")
    print(f"  CV folds: {args.n_folds}")
    print(f"  Embedding model: {args.embedding_model}")
    print(f"  Embedding dims: {embeddings.shape[1]}")
    print(f"  Chance level: {1/n_authors:.1%}")
    print(f"  Char n-gram features: {args.char_ngram_features}")
    print(f"  Word bigram features: {args.word_bigram_features}")

    results = {
        'timestamp': datetime.now().isoformat(),
        'version': 'extended_stylometry',
        'language': 'russian',
        'settings': {
            'n_authors': n_authors,
            'n_per_author': N_PER_AUTHOR,
            'total_samples': n_samples,
            'n_folds': args.n_folds,
            'seed': args.seed,
            'embedding_model': args.embedding_model,
            'embedding_dim': embeddings.shape[1],
            'char_ngram_features': args.char_ngram_features,
            'word_bigram_features': args.word_bigram_features
        },
        'experiments': {}
    }

    # ========================================================================
    # EXPERIMENT 1: Original 4-tier waterfall (for comparison)
    # ========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Original 4-Tier Waterfall (baseline comparison)")
    print("=" * 70)

    original_order = ['tier1', 'tier2', 'tier3', 'tier4']
    print(f"\n[Order]: {' -> '.join(original_order)}")

    results['experiments']['original_4tier'] = run_waterfall_cv(
        poems, embeddings, labels, original_order, n_authors,
        n_folds=args.n_folds, seed=args.seed)

    print("\n  Results:")
    for stage, data_stage in results['experiments']['original_4tier']['stages'].items():
        ci_str = f"[{data_stage['ci_95_low']:.1%}, {data_stage['ci_95_high']:.1%}]"
        print(f"    {stage}: {data_stage['accuracy']:.1%} {ci_str} "
              f"(lift={data_stage['lift_over_chance']:.1f}x, R²={data_stage['r2']:.3f})")

    # ========================================================================
    # EXPERIMENT 2: Extended 6-tier waterfall (with stylometry)
    # ========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Extended 6-Tier Waterfall (with stylometry)")
    print("=" * 70)

    extended_order = ['tier1', 'tier2', 'tier3', 'tier4', 'tier5', 'tier6']
    print(f"\n[Order]: {' -> '.join(extended_order)}")
    print("  Tier 5 = Char n-grams (2-4)")
    print("  Tier 6 = Word bigrams")

    results['experiments']['extended_6tier'] = run_waterfall_cv(
        poems, embeddings, labels, extended_order, n_authors,
        n_folds=args.n_folds, seed=args.seed,
        char_ngram_features=args.char_ngram_features,
        word_bigram_features=args.word_bigram_features)

    print("\n  Results:")
    for stage, data_stage in results['experiments']['extended_6tier']['stages'].items():
        ci_str = f"[{data_stage['ci_95_low']:.1%}, {data_stage['ci_95_high']:.1%}]"
        print(f"    {stage}: {data_stage['accuracy']:.1%} {ci_str} "
              f"(lift={data_stage['lift_over_chance']:.1f}x, R²={data_stage['r2']:.3f})")

    print(f"\n  Feature counts: {results['experiments']['extended_6tier']['tier_feature_counts']}")

    # ========================================================================
    # EXPERIMENT 3: Only stylometry tiers (char n-grams + word bigrams)
    # ========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Stylometry-Only Waterfall (char n-grams + word bigrams)")
    print("=" * 70)

    stylometry_order = ['tier5', 'tier6']
    print(f"\n[Order]: {' -> '.join(stylometry_order)}")

    results['experiments']['stylometry_only'] = run_waterfall_cv(
        poems, embeddings, labels, stylometry_order, n_authors,
        n_folds=args.n_folds, seed=args.seed,
        char_ngram_features=args.char_ngram_features,
        word_bigram_features=args.word_bigram_features)

    print("\n  Results:")
    for stage, data_stage in results['experiments']['stylometry_only']['stages'].items():
        ci_str = f"[{data_stage['ci_95_low']:.1%}, {data_stage['ci_95_high']:.1%}]"
        print(f"    {stage}: {data_stage['accuracy']:.1%} {ci_str} "
              f"(lift={data_stage['lift_over_chance']:.1f}x, R²={data_stage['r2']:.3f})")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY: Comparison of Waterfall Experiments")
    print("=" * 70)

    chance = 1 / n_authors
    print(f"\n  Chance level: {chance:.1%}")

    # Compare final accuracies
    final_original = results['experiments']['original_4tier']['stages']['after_kernel']['accuracy']
    final_extended = results['experiments']['extended_6tier']['stages']['after_kernel']['accuracy']
    final_stylometry = results['experiments']['stylometry_only']['stages']['after_kernel']['accuracy']

    r2_original = results['experiments']['original_4tier']['stages']['after_kernel']['r2']
    r2_extended = results['experiments']['extended_6tier']['stages']['after_kernel']['r2']
    r2_stylometry = results['experiments']['stylometry_only']['stages']['after_kernel']['r2']

    print(f"\n  {'Experiment':<30} {'Final Acc':>10} {'Lift':>8} {'R²':>8}")
    print(f"  {'-' * 60}")
    print(f"  {'Original (4 tiers: S+C+G+P)':<30} {final_original:>9.1%} {final_original/chance:>7.1f}× {r2_original:>7.3f}")
    print(f"  {'Extended (6 tiers: +Char+Word)':<30} {final_extended:>9.1%} {final_extended/chance:>7.1f}× {r2_extended:>7.3f}")
    print(f"  {'Stylometry only (Char+Word)':<30} {final_stylometry:>9.1%} {final_stylometry/chance:>7.1f}× {r2_stylometry:>7.3f}")
    print(f"  {'-' * 60}")
    print(f"  {'Chance':<30} {chance:>9.1%}")

    # Key insight
    print(f"\n[Key Finding]")
    print(f"  Original residual:  {final_original:.1%} ({final_original/chance:.1f}× chance)")
    print(f"  Extended residual:  {final_extended:.1%} ({final_extended/chance:.1f}× chance)")
    print(f"  Reduction:          {final_original - final_extended:.1%} ({(final_original - final_extended)/final_original*100:.0f}% drop)")

    # R² increase
    after_tier4_r2 = results['experiments']['extended_6tier']['stages']['after_tier4']['r2']
    after_tier5_r2 = results['experiments']['extended_6tier']['stages']['after_tier5']['r2']
    after_tier6_r2 = results['experiments']['extended_6tier']['stages']['after_tier6']['r2']

    print(f"\n[Variance Explained by Tier]")
    print(f"  After Tier 4 (Prosody):    R² = {after_tier4_r2:.1%}")
    print(f"  After Tier 5 (Char n-grams): R² = {after_tier5_r2:.1%} (+{after_tier5_r2 - after_tier4_r2:.1%})")
    print(f"  After Tier 6 (Word bigrams): R² = {after_tier6_r2:.1%} (+{after_tier6_r2 - after_tier5_r2:.1%})")
    print(f"  After Kernel:              R² = {r2_extended:.1%} (+{r2_extended - after_tier6_r2:.1%})")

    # Store comparison
    results['comparison'] = {
        'original_4tier': {
            'final_accuracy': final_original,
            'lift_over_chance': final_original / chance,
            'r2': r2_original
        },
        'extended_6tier': {
            'final_accuracy': final_extended,
            'lift_over_chance': final_extended / chance,
            'r2': r2_extended
        },
        'stylometry_only': {
            'final_accuracy': final_stylometry,
            'lift_over_chance': final_stylometry / chance,
            'r2': r2_stylometry
        },
        'reduction': {
            'absolute': final_original - final_extended,
            'relative': (final_original - final_extended) / final_original
        }
    }

    # Save results
    results_file = RESULTS_DIR / f"waterfall_extended_{args.embedding_model}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults saved to: {results_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
