"""
Classifier configurations with conservative hyperparameters.

Primary: LogisticRegression (interpretable)
Secondary: SVM-linear (robustness check)
Sensitivity: RandomForest, MLP (ceiling estimates)
"""

import numpy as np
from typing import Dict, List, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score


# Conservative hyperparameters to prevent overfitting
CLASSIFIERS = {
    'LogReg': LogisticRegression(
        max_iter=1000,
        solver='lbfgs',
        random_state=42
    ),

    'SVM_linear': SVC(
        kernel='linear',
        C=1.0,
        random_state=42
    ),

    'RandomForest': RandomForestClassifier(
        n_estimators=100,
        max_depth=10,           # Limit depth to prevent overfitting
        max_features='sqrt',    # sqrt(n_features) per split
        min_samples_leaf=5,     # Require 5 samples per leaf
        random_state=42
    ),

    'MLP': MLPClassifier(
        hidden_layer_sizes=(128,),  # Single layer, moderate width
        alpha=0.01,                  # Strong L2 regularization (10× default)
        early_stopping=True,         # Stop when validation loss plateaus
        validation_fraction=0.1,
        max_iter=500,
        random_state=42
    ),
}

# Classifier roles
CLASSIFIER_ROLES = {
    'LogReg': 'Primary',
    'SVM_linear': 'Secondary',
    'RandomForest': 'Sensitivity',
    'MLP': 'Sensitivity',
}


def compare_classifiers(
    X: np.ndarray,
    y: np.ndarray,
    classifiers: Dict = None,
    n_splits: int = 5,
    random_state: int = 42
) -> Dict[str, Dict]:
    """
    Compare classifiers with stratified CV.

    Returns accuracy mean/std and per-fold scores for paired tests.

    Args:
        X: Feature/embedding matrix
        y: Labels
        classifiers: Dict of classifiers (default: CLASSIFIERS)
        n_splits: CV folds
        random_state: Random seed

    Returns:
        Dict with results per classifier
    """
    if classifiers is None:
        classifiers = CLASSIFIERS

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Collect fold results
    fold_results = {name: [] for name in classifiers}

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        for name, clf in classifiers.items():
            clf_clone = clone(clf)
            clf_clone.fit(X_train, y_train)
            acc = accuracy_score(y_test, clf_clone.predict(X_test))
            fold_results[name].append(acc)

    # Compute statistics
    results = {}
    for name, scores in fold_results.items():
        results[name] = {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'scores': scores,  # Keep for paired tests
            'role': CLASSIFIER_ROLES.get(name, 'Unknown')
        }

    return results


def compute_ceiling_gap(
    results: Dict[str, Dict],
    baseline: str = 'LogReg',
    n_bootstrap: int = 1000,
    ci: float = 0.95
) -> Dict[str, Tuple[float, float, float]]:
    """
    Compute ceiling gap between non-linear and linear classifiers.

    Uses bootstrap CI for proper uncertainty quantification.

    Args:
        results: Output from compare_classifiers()
        baseline: Baseline classifier name
        n_bootstrap: Bootstrap iterations
        ci: Confidence level

    Returns:
        Dict with (mean_gap, ci_low, ci_high) per non-linear classifier
    """
    baseline_scores = np.array(results[baseline]['scores'])

    gaps = {}
    for name, data in results.items():
        if name == baseline:
            continue

        scores = np.array(data['scores'])
        diffs = scores - baseline_scores

        # Bootstrap CI
        boot_means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(diffs, len(diffs), replace=True)
            boot_means.append(np.mean(sample))

        lower = np.percentile(boot_means, (1 - ci) / 2 * 100)
        upper = np.percentile(boot_means, (1 + ci) / 2 * 100)
        mean_gap = np.mean(diffs)

        gaps[name] = (mean_gap, lower, upper)

    return gaps


def interpret_ceiling_gap(gap: Tuple[float, float, float]) -> str:
    """
    Interpret ceiling gap result.

    Args:
        gap: (mean, ci_low, ci_high)

    Returns:
        Interpretation string
    """
    mean, low, high = gap

    # Check if CI includes zero
    includes_zero = low <= 0 <= high

    if includes_zero:
        return "No significant gap (CI includes 0)"

    abs_mean = abs(mean)
    if abs_mean < 0.03:
        return f"Small gap ({mean:.1%}), statistically significant"
    elif abs_mean < 0.05:
        return f"Moderate gap ({mean:.1%}), investigate further"
    elif abs_mean < 0.10:
        return f"Large gap ({mean:.1%}), check for overfitting"
    else:
        return f"Very large gap ({mean:.1%}), linear results may underestimate ceiling"
