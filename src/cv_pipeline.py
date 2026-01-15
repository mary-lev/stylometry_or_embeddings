"""
Cross-validation pipeline with proper train-only fitting.

CRITICAL: Residualizer and Tier 2 extractor must be fitted on train only
to avoid data leakage.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.base import clone

from .features import Tier2Extractor, extract_all_features
from .residualization import Residualizer
from .classifiers import CLASSIFIERS


def residualized_cv(
    poems: List[Dict],
    embeddings: np.ndarray,
    labels: np.ndarray,
    tier2_extractor_class=Tier2Extractor,
    residualizer_method: str = 'linear',
    classifier=None,
    n_splits: int = 5,
    random_state: int = 42
) -> Dict:
    """
    Proper CV pipeline with train-only fitting for all components.

    This is the CORRECT way to run residualization experiments:
    1. Split data into train/test
    2. Fit Tier 2 extractor on TRAIN only
    3. Extract features for both train and test
    4. Fit residualizer on TRAIN only
    5. Transform BOTH train and test to residual space
    6. Fit classifier on TRAIN, evaluate on TEST

    Args:
        poems: List of poem dicts
        embeddings: Embedding matrix
        labels: Author labels
        tier2_extractor_class: Class for Tier 2 extraction
        residualizer_method: 'linear', 'kernel', or 'nystroem'
        classifier: Classifier to use (default: LogisticRegression)
        n_splits: CV folds
        random_state: Random seed

    Returns:
        Dict with accuracy stats and per-fold results
    """
    if classifier is None:
        classifier = LogisticRegression(max_iter=1000)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    results = {
        'raw_embeddings': [],
        'residualized': [],
    }

    for fold, (train_idx, test_idx) in enumerate(skf.split(embeddings, labels)):
        # Split data
        poems_train = [poems[i] for i in train_idx]
        poems_test = [poems[i] for i in test_idx]
        emb_train = embeddings[train_idx]
        emb_test = embeddings[test_idx]
        y_train = labels[train_idx]
        y_test = labels[test_idx]

        # Step 1: Fit Tier 2 extractor on TRAIN only
        tier2_extractor = tier2_extractor_class(random_state=random_state)
        tier2_extractor.fit(poems_train)

        # Step 2: Extract features for BOTH train and test
        features_train = extract_all_features(poems_train, tier2_extractor)['all']
        features_test = extract_all_features(poems_test, tier2_extractor)['all']

        # Step 3: Baseline - classify raw embeddings
        clf_raw = clone(classifier)
        clf_raw.fit(emb_train, y_train)
        acc_raw = accuracy_score(y_test, clf_raw.predict(emb_test))
        results['raw_embeddings'].append(acc_raw)

        # Step 4: Fit residualizer on TRAIN only
        residualizer = Residualizer(method=residualizer_method)
        residualizer.fit(features_train, emb_train)

        # Step 5: Transform BOTH to residual space
        resid_train = residualizer.transform(features_train, emb_train)
        resid_test = residualizer.transform(features_test, emb_test)

        # Step 6: Classify residualized embeddings
        clf_resid = clone(classifier)
        clf_resid.fit(resid_train, y_train)
        acc_resid = accuracy_score(y_test, clf_resid.predict(resid_test))
        results['residualized'].append(acc_resid)

    # Compute statistics
    return {
        'raw_embeddings': {
            'mean': np.mean(results['raw_embeddings']),
            'std': np.std(results['raw_embeddings']),
            'scores': results['raw_embeddings']
        },
        'residualized': {
            'mean': np.mean(results['residualized']),
            'std': np.std(results['residualized']),
            'scores': results['residualized']
        },
        'accuracy_drop': np.mean(results['raw_embeddings']) - np.mean(results['residualized'])
    }


def waterfall_cv(
    poems: List[Dict],
    embeddings: np.ndarray,
    labels: np.ndarray,
    classifier=None,
    n_splits: int = 5,
    random_state: int = 42
) -> Dict:
    """
    Five-tier waterfall with proper CV.

    Progressively residualizes by tier:
    1. Surface (Tier 1)
    2. + Content (Tier 1+2)
    3. + Grammar (Tier 1+2+3)
    4. + Prosody (Tier 1+2+3+4)
    5. + Non-linear (Kernel)

    Returns accuracy at each stage.
    """
    if classifier is None:
        classifier = LogisticRegression(max_iter=1000)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    stages = ['baseline', 'tier1', 'tier1+2', 'tier1+2+3', 'tier1+2+3+4', 'tier5_kernel']
    results = {stage: [] for stage in stages}

    for fold, (train_idx, test_idx) in enumerate(skf.split(embeddings, labels)):
        # Split
        poems_train = [poems[i] for i in train_idx]
        poems_test = [poems[i] for i in test_idx]
        emb_train = embeddings[train_idx]
        emb_test = embeddings[test_idx]
        y_train = labels[train_idx]
        y_test = labels[test_idx]

        # Baseline
        clf = clone(classifier)
        clf.fit(emb_train, y_train)
        results['baseline'].append(accuracy_score(y_test, clf.predict(emb_test)))

        # Fit Tier 2 extractor on train
        tier2_ext = Tier2Extractor(random_state=random_state)
        tier2_ext.fit(poems_train)

        # Extract all features
        feat_train = extract_all_features(poems_train, tier2_ext)
        feat_test = extract_all_features(poems_test, tier2_ext)

        # Progressive residualization
        cumulative_train = None
        cumulative_test = None

        for i, tier_key in enumerate(['tier1', 'tier2', 'tier3', 'tier4']):
            # Accumulate features
            if cumulative_train is None:
                cumulative_train = feat_train[tier_key]
                cumulative_test = feat_test[tier_key]
            else:
                cumulative_train = np.hstack([cumulative_train, feat_train[tier_key]])
                cumulative_test = np.hstack([cumulative_test, feat_test[tier_key]])

            # Residualize (linear)
            residualizer = Residualizer(method='linear')
            residualizer.fit(cumulative_train, emb_train)
            resid_train = residualizer.transform(cumulative_train, emb_train)
            resid_test = residualizer.transform(cumulative_test, emb_test)

            # Classify
            clf = clone(classifier)
            clf.fit(resid_train, y_train)
            acc = accuracy_score(y_test, clf.predict(resid_test))

            stage_name = f"tier1+{i+1}" if i > 0 else "tier1"
            if stage_name in results:
                results[stage_name].append(acc)

        # Tier 5: Non-linear (kernel) on all features
        all_train = feat_train['all']
        all_test = feat_test['all']

        residualizer_kernel = Residualizer(method='nystroem', kernel='rbf')
        residualizer_kernel.fit(all_train, emb_train)
        resid_train = residualizer_kernel.transform(all_train, emb_train)
        resid_test = residualizer_kernel.transform(all_test, emb_test)

        clf = clone(classifier)
        clf.fit(resid_train, y_train)
        results['tier5_kernel'].append(accuracy_score(y_test, clf.predict(resid_test)))

    # Compute statistics
    summary = {}
    for stage, scores in results.items():
        summary[stage] = {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'scores': scores
        }

    return summary


def full_experiment_cv(
    poems: List[Dict],
    embeddings: np.ndarray,
    labels: np.ndarray,
    classifiers: Dict = None,
    n_splits: int = 5,
    random_state: int = 42
) -> Dict:
    """
    Full experiment with multiple classifiers and waterfall.

    Combines:
    - Waterfall residualization
    - Multiple classifier comparison
    - Ceiling gap analysis
    """
    if classifiers is None:
        classifiers = CLASSIFIERS

    # Run waterfall with primary classifier
    waterfall_results = waterfall_cv(
        poems, embeddings, labels,
        classifier=classifiers.get('LogReg', LogisticRegression(max_iter=1000)),
        n_splits=n_splits,
        random_state=random_state
    )

    # Run classifier comparison at baseline and final residual
    from .classifiers import compare_classifiers, compute_ceiling_gap

    # Baseline comparison
    baseline_comparison = compare_classifiers(
        embeddings, labels,
        classifiers=classifiers,
        n_splits=n_splits,
        random_state=random_state
    )

    return {
        'waterfall': waterfall_results,
        'classifier_comparison': baseline_comparison,
        'ceiling_gaps': compute_ceiling_gap(baseline_comparison)
    }
