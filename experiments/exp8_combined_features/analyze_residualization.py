#!/usr/bin/env python3
"""
EXP8: Bidirectional Residualization Analysis (Fixed CV)

Key question: What unique information does each method capture?

Method (PROPER CV):
1. For each fold:
   a. Fit Ridge regression on TRAIN data only
   b. Predict on both train and test
   c. Compute residuals
   d. Classify using residuals
2. Report cross-validated accuracy

This avoids data leakage from fitting residualizer on all data.
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# Constants
DATA_DIR = Path(__file__).parent.parent.parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def load_russian_data(n_per_author=200, seed=42, embedding_model='openai'):
    """Load Russian clean dataset."""
    from clean_dataset import load_clean_dataset
    data = load_clean_dataset(
        n_per_author=n_per_author,
        seed=seed,
        include_poems=True,
        embedding_model=embedding_model
    )
    texts = [p['text'] for p in data['poems']]
    return texts, data['embeddings'], data['labels'], data['author_list']


def load_italian_data(n_per_author=200, seed=42, embedding_model='gemini'):
    """Load the published Italian dataset.

    Data comes from data/italian/ through the shared loader; see
    data/README.md. The n_per_author and seed arguments are kept for
    call-site compatibility -- the published dataset is fixed at 200
    poems per author, seed 42.
    """
    from data_loader import load_dataset

    # The published layout names this model 'qwen8b'; accept the older
    # 'qwen-8b' spelling used by earlier result files.
    model = {'qwen-8b': 'qwen8b'}.get(embedding_model, embedding_model)

    data = load_dataset('italian', embedding_model=model, include_poems=True)
    texts = [p['text'] for p in data['poems']]

    return texts, data['embeddings'], data['labels'], data['author_list']


def run_residualization_cv(X_source, X_target, y, direction_name, n_folds=5, seed=42, alpha=1.0):
    """
    Run residualization with proper CV (fit residualizer on train only).

    Args:
        X_source: Features to predict FROM (e.g., stylometry)
        X_target: Features to predict TO and residualize (e.g., embeddings)
        y: Labels
        direction_name: For logging (e.g., "stylometry → embeddings")

    Returns:
        dict with accuracy, r2, and fold details
    """
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    fold_scores_target = []  # Original target accuracy
    fold_scores_predicted = []  # Predicted target accuracy
    fold_scores_residual = []  # Residual target accuracy
    fold_r2 = []

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X_source, y)):
        # Split data
        X_src_train, X_src_test = X_source[train_idx], X_source[test_idx]
        X_tgt_train, X_tgt_test = X_target[train_idx], X_target[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # 1. Baseline: Original target accuracy
        scaler = StandardScaler()
        X_tgt_train_scaled = scaler.fit_transform(X_tgt_train)
        X_tgt_test_scaled = scaler.transform(X_tgt_test)

        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(X_tgt_train_scaled, y_train)
        fold_scores_target.append(clf.score(X_tgt_test_scaled, y_test))

        # 2. Fit residualizer on TRAIN only
        src_scaler = StandardScaler()
        X_src_train_scaled = src_scaler.fit_transform(X_src_train)
        X_src_test_scaled = src_scaler.transform(X_src_test)

        ridge = Ridge(alpha=alpha)
        ridge.fit(X_src_train_scaled, X_tgt_train)

        # Predict target from source
        tgt_pred_train = ridge.predict(X_src_train_scaled)
        tgt_pred_test = ridge.predict(X_src_test_scaled)

        # Compute R² on train
        r2 = r2_score(X_tgt_train, tgt_pred_train, multioutput='variance_weighted')
        fold_r2.append(r2)

        # 3. Predicted target accuracy
        pred_scaler = StandardScaler()
        tgt_pred_train_scaled = pred_scaler.fit_transform(tgt_pred_train)
        tgt_pred_test_scaled = pred_scaler.transform(tgt_pred_test)

        clf_pred = LogisticRegression(max_iter=2000, C=1.0)
        clf_pred.fit(tgt_pred_train_scaled, y_train)
        fold_scores_predicted.append(clf_pred.score(tgt_pred_test_scaled, y_test))

        # 4. Residual target accuracy
        resid_train = X_tgt_train - tgt_pred_train
        resid_test = X_tgt_test - tgt_pred_test

        resid_scaler = StandardScaler()
        resid_train_scaled = resid_scaler.fit_transform(resid_train)
        resid_test_scaled = resid_scaler.transform(resid_test)

        clf_resid = LogisticRegression(max_iter=2000, C=1.0)
        clf_resid.fit(resid_train_scaled, y_train)
        fold_scores_residual.append(clf_resid.score(resid_test_scaled, y_test))

    return {
        'direction': direction_name,
        'target_accuracy': float(np.mean(fold_scores_target)),
        'target_std': float(np.std(fold_scores_target)),
        'predicted_accuracy': float(np.mean(fold_scores_predicted)),
        'predicted_std': float(np.std(fold_scores_predicted)),
        'residual_accuracy': float(np.mean(fold_scores_residual)),
        'residual_std': float(np.std(fold_scores_residual)),
        'r2': float(np.mean(fold_r2)),
        'r2_std': float(np.std(fold_r2)),
        'fold_scores': {
            'target': fold_scores_target,
            'predicted': fold_scores_predicted,
            'residual': fold_scores_residual,
            'r2': fold_r2
        }
    }


def main():
    parser = argparse.ArgumentParser(description="EXP8: Bidirectional Residualization Analysis")
    parser.add_argument('--language', type=str, default='russian',
                        choices=['russian', 'italian'],
                        help='Language to analyze')
    parser.add_argument('--embedding-model', type=str, default='gemini',
                        choices=['openai', 'gemini', 'qwen8b', 'qwen-8b'],
                        help='Embedding model')
    parser.add_argument('--n-per-author', type=int, default=200,
                        help='Poems per author')
    parser.add_argument('--cv-folds', type=int, default=5,
                        help='CV folds')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Ridge alpha')
    args = parser.parse_args()

    print("=" * 70)
    print("EXP8: Bidirectional Residualization Analysis (Fixed CV)")
    print("=" * 70)
    print("\nNote: Residualizer fitted on TRAIN data only per fold (no leakage)")

    # Load data
    print(f"\n[Loading {args.language} data...]")
    if args.language == 'russian':
        texts, embeddings, labels, author_list = load_russian_data(
            n_per_author=args.n_per_author,
            seed=args.seed,
            embedding_model=args.embedding_model
        )
        embedding_model = args.embedding_model
    else:
        texts, embeddings, labels, author_list = load_italian_data(
            n_per_author=args.n_per_author,
            seed=args.seed,
            embedding_model=args.embedding_model
        )
        embedding_model = args.embedding_model

    n_samples = len(labels)
    n_authors = len(author_list)
    chance = 1 / n_authors

    print(f"  Language: {args.language}")
    print(f"  Samples: {n_samples}")
    print(f"  Authors: {n_authors}")
    print(f"  Embedding model: {embedding_model}")
    print(f"  Chance level: {chance:.1%}")

    # Extract char 3-grams
    print("\n[Extracting char 3-grams...]")
    char_vec = TfidfVectorizer(analyzer='char', ngram_range=(3, 3), max_features=2000)
    X_char = char_vec.fit_transform(texts).toarray()
    print(f"  Char 3-gram features: {X_char.shape[1]}")
    print(f"  Embedding features: {embeddings.shape[1]}")

    # Results storage
    results = {
        'timestamp': datetime.now().isoformat(),
        'language': args.language,
        'embedding_model': embedding_model,
        'settings': {
            'n_authors': n_authors,
            'n_samples': n_samples,
            'n_per_author': args.n_per_author,
            'cv_folds': args.cv_folds,
            'seed': args.seed,
            'alpha': args.alpha,
            'chance_level': chance
        },
        'directions': {}
    }

    # ==================== BASELINES ====================
    print("\n" + "=" * 70)
    print("BASELINE CLASSIFICATION")
    print("=" * 70)

    # Embeddings baseline
    print("\n[Embeddings only...]")
    kfold = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
    emb_scores = []
    for train_idx, test_idx in kfold.split(embeddings, labels):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(embeddings[train_idx])
        X_test = scaler.transform(embeddings[test_idx])
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(X_train, labels[train_idx])
        emb_scores.append(clf.score(X_test, labels[test_idx]))
    acc_emb = np.mean(emb_scores)
    std_emb = np.std(emb_scores)
    print(f"  Accuracy: {acc_emb:.1%} ± {std_emb:.1%}")
    results['baselines'] = {'embeddings': {'accuracy': acc_emb, 'std': std_emb}}

    # Stylometry baseline
    print("\n[Char 3-grams only...]")
    char_scores = []
    for train_idx, test_idx in kfold.split(X_char, labels):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_char[train_idx])
        X_test = scaler.transform(X_char[test_idx])
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(X_train, labels[train_idx])
        char_scores.append(clf.score(X_test, labels[test_idx]))
    acc_char = np.mean(char_scores)
    std_char = np.std(char_scores)
    print(f"  Accuracy: {acc_char:.1%} ± {std_char:.1%}")
    results['baselines']['stylometry'] = {'accuracy': acc_char, 'std': std_char}

    # ==================== DIRECTION 1: Stylometry → Embeddings ====================
    print("\n" + "=" * 70)
    print("DIRECTION 1: What do embeddings capture beyond stylometry?")
    print("=" * 70)

    dir1 = run_residualization_cv(
        X_source=X_char,
        X_target=embeddings,
        y=labels,
        direction_name="stylometry → embeddings",
        n_folds=args.cv_folds,
        seed=args.seed,
        alpha=args.alpha
    )
    results['directions']['stylometry_to_embeddings'] = dir1

    print(f"\n  Variance explained (R²): {dir1['r2']:.1%} ± {dir1['r2_std']:.1%}")
    print(f"  Predicted embeddings accuracy: {dir1['predicted_accuracy']:.1%} ± {dir1['predicted_std']:.1%}")
    print(f"  Residual embeddings accuracy: {dir1['residual_accuracy']:.1%} ± {dir1['residual_std']:.1%}")
    print(f"  → Unique to embeddings: {dir1['residual_accuracy']:.1%} ({dir1['residual_accuracy']/chance:.1f}× chance)")

    # ==================== DIRECTION 2: Embeddings → Stylometry ====================
    print("\n" + "=" * 70)
    print("DIRECTION 2: What does stylometry capture beyond embeddings?")
    print("=" * 70)

    dir2 = run_residualization_cv(
        X_source=embeddings,
        X_target=X_char,
        y=labels,
        direction_name="embeddings → stylometry",
        n_folds=args.cv_folds,
        seed=args.seed,
        alpha=args.alpha
    )
    results['directions']['embeddings_to_stylometry'] = dir2

    print(f"\n  Variance explained (R²): {dir2['r2']:.1%} ± {dir2['r2_std']:.1%}")
    print(f"  Predicted stylometry accuracy: {dir2['predicted_accuracy']:.1%} ± {dir2['predicted_std']:.1%}")
    print(f"  Residual stylometry accuracy: {dir2['residual_accuracy']:.1%} ± {dir2['residual_std']:.1%}")
    print(f"  → Unique to stylometry: {dir2['residual_accuracy']:.1%} ({dir2['residual_accuracy']/chance:.1f}× chance)")

    # ==================== SUMMARY ====================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n{'Method':<40} {'Accuracy':>12} {'Lift':>8}")
    print("-" * 70)
    print(f"{'Embeddings only':<40} {acc_emb:>11.1%} {acc_emb/chance:>7.1f}×")
    print(f"{'Char 3-grams only':<40} {acc_char:>11.1%} {acc_char/chance:>7.1f}×")
    print("-" * 70)
    print(f"{'Predicted emb (from char)':<40} {dir1['predicted_accuracy']:>11.1%} {dir1['predicted_accuracy']/chance:>7.1f}×")
    print(f"{'Residual emb (unique to embeddings)':<40} {dir1['residual_accuracy']:>11.1%} {dir1['residual_accuracy']/chance:>7.1f}×")
    print("-" * 70)
    print(f"{'Predicted char (from emb)':<40} {dir2['predicted_accuracy']:>11.1%} {dir2['predicted_accuracy']/chance:>7.1f}×")
    print(f"{'Residual char (unique to stylometry)':<40} {dir2['residual_accuracy']:>11.1%} {dir2['residual_accuracy']/chance:>7.1f}×")
    print("-" * 70)
    print(f"{'Chance':<40} {chance:>11.1%} {1.0:>7.1f}×")

    print(f"\n[Key Findings]")
    print(f"\n  Direction 1: Stylometry → Embeddings")
    print(f"    Stylometry explains {dir1['r2']:.1%} of embedding variance")
    print(f"    Residual embeddings: {dir1['residual_accuracy']:.1%} accuracy")

    print(f"\n  Direction 2: Embeddings → Stylometry")
    print(f"    Embeddings explain {dir2['r2']:.1%} of stylometry variance")
    print(f"    Residual stylometry: {dir2['residual_accuracy']:.1%} accuracy")

    print(f"\n[Asymmetry Analysis]")
    if dir1['r2'] > dir2['r2']:
        print(f"  Stylometry → Embeddings explains MORE variance ({dir1['r2']:.1%} vs {dir2['r2']:.1%})")
        print(f"  → Embeddings largely 'contain' stylometry information")
    else:
        print(f"  Embeddings → Stylometry explains MORE variance ({dir2['r2']:.1%} vs {dir1['r2']:.1%})")
        print(f"  → Stylometry largely 'contains' embedding information")

    if dir1['residual_accuracy'] > dir2['residual_accuracy']:
        diff = dir1['residual_accuracy'] - dir2['residual_accuracy']
        print(f"\n  Embeddings have MORE unique signal (+{diff:.1%})")
    else:
        diff = dir2['residual_accuracy'] - dir1['residual_accuracy']
        print(f"\n  Stylometry has MORE unique signal (+{diff:.1%})")
        print(f"  Character patterns are harder to capture in embeddings")

    # Save results
    model_suffix = f"_{embedding_model}" if args.language == 'russian' else ""
    output_file = RESULTS_DIR / f"residualization_{args.language}{model_suffix}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
