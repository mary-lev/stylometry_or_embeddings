"""
Metrics and statistical tests for experiments.

Includes:
- Permutation tests for accuracy vs chance
- Bootstrap confidence intervals
- McNemar's test for classifier comparison
- FDR correction for multiple comparisons
"""

import numpy as np
from typing import Tuple, List, Optional
from scipy import stats
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone


def permutation_test(
    X: np.ndarray,
    y: np.ndarray,
    classifier=None,
    n_permutations: int = 1000,
    n_splits: int = 5,
    random_state: int = 42
) -> Tuple[float, float, float]:
    """
    Permutation test for classification accuracy.

    Tests H0: classifier accuracy = chance (label-permuted accuracy).

    Args:
        X: Features/embeddings
        y: Labels
        classifier: Classifier (default: LogisticRegression)
        n_permutations: Number of permutations
        n_splits: CV folds
        random_state: Random seed

    Returns:
        (observed_accuracy, p_value, chance_mean)
    """
    if classifier is None:
        classifier = LogisticRegression(max_iter=1000)

    np.random.seed(random_state)

    def cv_accuracy(X, y):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        scores = []
        for train_idx, test_idx in skf.split(X, y):
            clf = clone(classifier)
            clf.fit(X[train_idx], y[train_idx])
            scores.append(accuracy_score(y[test_idx], clf.predict(X[test_idx])))
        return np.mean(scores)

    # Observed accuracy
    observed = cv_accuracy(X, y)

    # Permuted accuracies
    permuted_scores = []
    for _ in range(n_permutations):
        y_perm = np.random.permutation(y)
        permuted_scores.append(cv_accuracy(X, y_perm))

    # P-value: proportion of permutations >= observed
    p_value = np.mean([s >= observed for s in permuted_scores])

    return observed, p_value, np.mean(permuted_scores)


def bootstrap_ci(
    scores: List[float],
    ci: float = 0.95,
    n_bootstrap: int = 10000,
    random_state: int = 42
) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval for mean.

    Args:
        scores: List of scores (e.g., per-fold accuracies)
        ci: Confidence level
        n_bootstrap: Bootstrap iterations
        random_state: Random seed

    Returns:
        (mean, ci_low, ci_high)
    """
    np.random.seed(random_state)
    scores = np.array(scores)

    boot_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(scores, len(scores), replace=True)
        boot_means.append(np.mean(sample))

    lower = np.percentile(boot_means, (1 - ci) / 2 * 100)
    upper = np.percentile(boot_means, (1 + ci) / 2 * 100)

    return np.mean(scores), lower, upper


def accuracy_with_ci(
    X: np.ndarray,
    y: np.ndarray,
    classifier=None,
    n_splits: int = 5,
    ci: float = 0.95,
    random_state: int = 42
) -> Tuple[float, float, float]:
    """
    Compute accuracy with bootstrap CI.

    Args:
        X: Features/embeddings
        y: Labels
        classifier: Classifier
        n_splits: CV folds
        ci: Confidence level
        random_state: Random seed

    Returns:
        (mean_accuracy, ci_low, ci_high)
    """
    if classifier is None:
        classifier = LogisticRegression(max_iter=1000)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = []

    for train_idx, test_idx in skf.split(X, y):
        clf = clone(classifier)
        clf.fit(X[train_idx], y[train_idx])
        scores.append(accuracy_score(y[test_idx], clf.predict(X[test_idx])))

    return bootstrap_ci(scores, ci=ci)


def mcnemar_test(
    y_true: np.ndarray,
    y_pred1: np.ndarray,
    y_pred2: np.ndarray
) -> Tuple[float, float]:
    """
    McNemar's test for comparing two classifiers.

    Tests H0: both classifiers have same error rate.

    Args:
        y_true: True labels
        y_pred1: Predictions from classifier 1
        y_pred2: Predictions from classifier 2

    Returns:
        (test_statistic, p_value)
    """
    # Contingency table
    # b: clf1 correct, clf2 wrong
    # c: clf1 wrong, clf2 correct
    correct1 = y_pred1 == y_true
    correct2 = y_pred2 == y_true

    b = np.sum(correct1 & ~correct2)  # clf1 right, clf2 wrong
    c = np.sum(~correct1 & correct2)  # clf1 wrong, clf2 right

    # McNemar's test (with continuity correction)
    if b + c == 0:
        return 0.0, 1.0

    statistic = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = 1 - stats.chi2.cdf(statistic, df=1)

    return statistic, p_value


def fdr_correction(
    p_values: List[float],
    alpha: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Benjamini-Hochberg FDR correction for multiple comparisons.

    Args:
        p_values: List of p-values
        alpha: Significance level

    Returns:
        (rejected, adjusted_p_values)
    """
    p_values = np.array(p_values)
    n = len(p_values)

    # Sort p-values
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]

    # BH procedure
    thresholds = np.arange(1, n + 1) / n * alpha

    # Find largest k where p[k] <= threshold[k]
    rejected_sorted = sorted_p <= thresholds

    # Adjusted p-values
    adjusted = np.minimum.accumulate((sorted_p * n / np.arange(1, n + 1))[::-1])[::-1]
    adjusted = np.minimum(adjusted, 1.0)

    # Reorder to original order
    rejected = np.zeros(n, dtype=bool)
    adjusted_original = np.zeros(n)

    rejected[sorted_idx] = rejected_sorted
    adjusted_original[sorted_idx] = adjusted

    return rejected, adjusted_original


def effect_size_ratio(
    accuracy: float,
    chance: float
) -> float:
    """
    Compute effect size as accuracy / chance ratio.

    Args:
        accuracy: Observed accuracy
        chance: Chance level (1/K)

    Returns:
        Ratio (e.g., 2.7× for 34% vs 12.5%)
    """
    if chance == 0:
        return float('inf')
    return accuracy / chance


def accuracy_to_distance(accuracy: float) -> float:
    """
    Convert binary classification accuracy to stylistic distance.

    Maps [0.5, 1.0] → [0, 1] (for K=2 binary).

    Args:
        accuracy: Classification accuracy (0.5 to 1.0)

    Returns:
        Distance (0 = indistinguishable, 1 = maximally different)
    """
    return 2 * (accuracy - 0.5)


def chance_level(n_classes: int) -> float:
    """
    Compute chance level for K-class classification.

    Args:
        n_classes: Number of classes

    Returns:
        Chance accuracy (1/K)
    """
    return 1.0 / n_classes
