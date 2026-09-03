#!/usr/bin/env python3
"""
Test: Does applying kernel (nonlinear) residualization at each tier change results?

Purpose: Address reviewer concern that linear Ridge may underestimate what features explain
if embeddings encode features nonlinearly.

Method:
1. Run waterfall with LINEAR residualization at each tier (current approach)
2. Run waterfall with KERNEL residualization at each tier (test approach)
3. Compare accuracy drops and final residuals

Expected outcomes:
- If embeddings encode features linearly: No difference
- If embeddings encode features nonlinearly: Kernel version shows larger drops
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from clean_dataset import load_clean_dataset
from features import extract_tier1_surface, extract_tier3_grammar, extract_tier4_prosody

RESULTS_DIR = Path(__file__).parent / "results"


class Tier2Extractor:
    def __init__(self, n_tfidf=20, n_topics=20):
        self.n_tfidf = n_tfidf
        self.n_topics = n_topics
        self.tfidf = None
        self.lda = None

    def fit(self, poems):
        from sklearn.decomposition import LatentDirichletAllocation
        texts = [p['text'] for p in poems]
        self.tfidf = TfidfVectorizer(max_features=500, min_df=2, max_df=0.95)
        tfidf_matrix = self.tfidf.fit_transform(texts)
        self.lda = LatentDirichletAllocation(n_components=self.n_topics, random_state=42, max_iter=10)
        self.lda.fit(tfidf_matrix)
        return self

    def transform(self, poems):
        texts = [p['text'] for p in poems]
        tfidf_matrix = self.tfidf.transform(texts)
        topics = self.lda.transform(tfidf_matrix)
        tfidf_top = tfidf_matrix[:, :self.n_tfidf].toarray()
        return np.hstack([tfidf_top, topics])


class CharNgramExtractor:
    def __init__(self, max_features=2000):
        self.vectorizer = None
        self.max_features = max_features

    def fit(self, poems):
        texts = [p['text'] for p in poems]
        self.vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4),
                                          max_features=self.max_features)
        self.vectorizer.fit(texts)
        return self

    def transform(self, poems):
        texts = [p['text'] for p in poems]
        return self.vectorizer.transform(texts).toarray()


class WordBigramExtractor:
    def __init__(self, max_features=2000):
        self.vectorizer = None
        self.max_features = max_features

    def fit(self, poems):
        texts = [p['text'] for p in poems]
        self.vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(2, 2),
                                          max_features=self.max_features)
        self.vectorizer.fit(texts)
        return self

    def transform(self, poems):
        texts = [p['text'] for p in poems]
        return self.vectorizer.transform(texts).toarray()


def run_comparison(poems, embeddings, labels, seed=42, n_folds=3):
    """Compare linear vs kernel residualization at each tier."""

    tier_order = ['tier1', 'tier2', 'tier3', 'tier4', 'tier5', 'tier6']
    fitted_tiers = {'tier2', 'tier5', 'tier6'}

    # Pre-extract deterministic features
    print("  Extracting features...")
    det_features = {
        'tier1': np.array([extract_tier1_surface(p) for p in poems]),
        'tier3': np.array([extract_tier3_grammar(p) for p in poems]),
        'tier4': np.array([extract_tier4_prosody(p) for p in poems])
    }

    results = {'linear': {}, 'kernel': {}}

    for use_kernel in [False, True]:
        method = 'kernel' if use_kernel else 'linear'
        print(f"\n  [{method.upper()} residualization]")

        kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        stage_scores = {stage: [] for stage in ['baseline'] + [f'after_{t}' for t in tier_order]}

        for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(embeddings, labels)):
            emb_train, emb_test = embeddings[train_idx].copy(), embeddings[test_idx].copy()
            y_train, y_test = labels[train_idx], labels[test_idx]
            train_poems = [poems[i] for i in train_idx]
            test_poems = [poems[i] for i in test_idx]

            # Baseline
            scaler = StandardScaler()
            X_train = scaler.fit_transform(emb_train)
            X_test = scaler.transform(emb_test)
            clf = LogisticRegression(max_iter=1000)
            clf.fit(X_train, y_train)
            stage_scores['baseline'].append(clf.score(X_test, y_test))

            # Fit extractors
            extractors = {
                'tier2': Tier2Extractor().fit(train_poems),
                'tier5': CharNgramExtractor().fit(train_poems),
                'tier6': WordBigramExtractor().fit(train_poems)
            }

            cumul_feat_train = None
            cumul_feat_test = None

            for tier in tier_order:
                # Get features
                if tier in fitted_tiers:
                    feat_train = extractors[tier].transform(train_poems)
                    feat_test = extractors[tier].transform(test_poems)
                else:
                    feat_train = det_features[tier][train_idx]
                    feat_test = det_features[tier][test_idx]

                # Accumulate
                if cumul_feat_train is None:
                    cumul_feat_train = feat_train
                    cumul_feat_test = feat_test
                else:
                    cumul_feat_train = np.hstack([cumul_feat_train, feat_train])
                    cumul_feat_test = np.hstack([cumul_feat_test, feat_test])

                # Scale features
                feat_scaler = StandardScaler()
                feat_train_scaled = feat_scaler.fit_transform(np.nan_to_num(cumul_feat_train, nan=0.0))
                feat_test_scaled = feat_scaler.transform(np.nan_to_num(cumul_feat_test, nan=0.0))

                # Residualize (linear or kernel)
                if use_kernel:
                    n_comp = min(300, feat_train_scaled.shape[0] // 3, feat_train_scaled.shape[1])
                    pipe = Pipeline([
                        ('nystroem', Nystroem(kernel='rbf', gamma=0.1, n_components=n_comp, random_state=seed)),
                        ('ridge', Ridge(alpha=1.0))
                    ])
                    pipe.fit(feat_train_scaled, emb_train)
                    pred_train = pipe.predict(feat_train_scaled)
                    pred_test = pipe.predict(feat_test_scaled)
                else:
                    ridge = Ridge(alpha=1.0)
                    ridge.fit(feat_train_scaled, emb_train)
                    pred_train = ridge.predict(feat_train_scaled)
                    pred_test = ridge.predict(feat_test_scaled)

                emb_train = emb_train - pred_train
                emb_test = emb_test - pred_test

                # Classify residual
                scaler = StandardScaler()
                X_train = scaler.fit_transform(emb_train)
                X_test = scaler.transform(emb_test)
                clf = LogisticRegression(max_iter=1000)
                clf.fit(X_train, y_train)
                stage_scores[f'after_{tier}'].append(clf.score(X_test, y_test))

        # Average across folds
        for stage, scores in stage_scores.items():
            results[method][stage] = {
                'accuracy': float(np.mean(scores)),
                'std': float(np.std(scores))
            }
            if stage != 'baseline':
                print(f"    {stage}: {np.mean(scores):.2%} ± {np.std(scores):.2%}")

    return results


def main():
    print("=" * 60)
    print("TEST: Linear vs Kernel Residualization at Each Tier")
    print("=" * 60)

    print("\n[Loading data...]")
    data = load_clean_dataset(n_per_author=200, seed=42, include_poems=True, embedding_model='gemini')
    poems = data['poems']
    embeddings = data['embeddings']
    labels = data['labels']

    n_authors = len(data['author_list'])
    chance = 1 / n_authors

    print(f"  Authors: {n_authors}, Samples: {len(poems)}, Chance: {chance:.2%}")

    print("\n[Running comparison (3-fold CV for speed)...]")
    results = run_comparison(poems, embeddings, labels, seed=42, n_folds=3)

    # Summary
    print("\n" + "=" * 60)
    print("COMPARISON: Linear vs Kernel Residualization")
    print("=" * 60)
    print(f"\n{'Stage':<20} {'Linear':<15} {'Kernel':<15} {'Δ':<10}")
    print("-" * 60)

    for stage in ['baseline', 'after_tier1', 'after_tier2', 'after_tier3',
                  'after_tier4', 'after_tier5', 'after_tier6']:
        lin = results['linear'][stage]['accuracy']
        ker = results['kernel'][stage]['accuracy']
        diff = ker - lin
        print(f"{stage:<20} {lin:>6.2%}         {ker:>6.2%}         {diff:>+.2%}")

    # Interpretation
    final_lin = results['linear']['after_tier6']['accuracy']
    final_ker = results['kernel']['after_tier6']['accuracy']
    diff = final_ker - final_lin

    print("\n[Interpretation]")
    # Lower residual = better (more signal removed)
    # Linear achieves lower residual = linear works BETTER
    if final_lin < final_ker:
        print(f"  Linear residualization ({final_lin:.2%}) outperforms kernel ({final_ker:.2%}).")
        print("  This confirms embeddings encode features LINEARLY.")
        print("  Kernel likely overfits, making poor test predictions.")
    elif abs(diff) < 0.01:
        print(f"  Minimal difference ({diff:+.2%}): Both methods equivalent.")
        print("  Linear sufficient for interpretability.")
    else:
        print(f"  Kernel residualization ({final_ker:.2%}) outperforms linear ({final_lin:.2%}).")
        print("  Features may encode embeddings nonlinearly.")

    # Save results
    # Lower residual = better. Linear < kernel means linear encoding.
    if final_lin < final_ker:
        interp = 'linear_encoding_confirmed'
    elif abs(diff) < 0.01:
        interp = 'equivalent'
    else:
        interp = 'nonlinear_encoding_possible'

    output = {
        'timestamp': datetime.now().isoformat(),
        'n_folds': 3,
        'results': results,
        'linear_final': final_lin,
        'kernel_final': final_ker,
        'interpretation': interp
    }

    output_file = RESULTS_DIR / "nonlinear_tier_test.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
