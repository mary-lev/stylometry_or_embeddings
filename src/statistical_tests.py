"""
Statistical tests for embedding methodology study.

Provides:
1. Permutation test for accuracy > chance
2. Bootstrap confidence intervals
3. McNemar's test for comparing classifiers
4. Effect size measures (Cohen's d, lift over chance)
"""

import numpy as np
from typing import Tuple, List, Optional, Dict
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


def permutation_test_accuracy(
    X: np.ndarray,
    y: np.ndarray,
    observed_accuracy: float,
    n_permutations: int = 1000,
    n_folds: int = 5,
    random_state: int = 42
) -> Tuple[float, np.ndarray]:
    """
    Permutation test: Is observed accuracy significantly above chance?

    Null hypothesis: Labels are independent of features (accuracy = chance).

    Args:
        X: Feature matrix (n_samples, n_features)
        y: Labels (n_samples,)
        observed_accuracy: The accuracy to test
        n_permutations: Number of permutations
        n_folds: CV folds for each permutation
        random_state: Random seed

    Returns:
        p_value: Proportion of permuted accuracies >= observed
        null_distribution: Array of permuted accuracies
    """
    rng = np.random.RandomState(random_state)
    null_distribution = []

    for i in range(n_permutations):
        # Permute labels
        y_perm = rng.permutation(y)

        # Run CV classification
        kfold = StratifiedKFold(n_splits=n_folds, shuffle=True,
                                 random_state=random_state + i)
        scores = []

        for train_idx, test_idx in kfold.split(X, y_perm):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx])
            X_test = scaler.transform(X[test_idx])

            clf = LogisticRegression(max_iter=1000, solver='lbfgs')
            clf.fit(X_train, y_perm[train_idx])
            scores.append(clf.score(X_test, y_perm[test_idx]))

        null_distribution.append(np.mean(scores))

    null_distribution = np.array(null_distribution)

    # One-tailed p-value: proportion of null >= observed
    p_value = np.mean(null_distribution >= observed_accuracy)

    return p_value, null_distribution


def bootstrap_ci(
    scores: List[float],
    confidence: float = 0.95,
    n_bootstrap: int = 10000,
    random_state: int = 42
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for mean accuracy.

    Args:
        scores: List of accuracy scores (e.g., from CV folds)
        confidence: Confidence level (default 0.95 for 95% CI)
        n_bootstrap: Number of bootstrap samples
        random_state: Random seed

    Returns:
        mean: Point estimate
        ci_low: Lower bound of CI
        ci_high: Upper bound of CI
    """
    rng = np.random.RandomState(random_state)
    scores = np.array(scores)
    n = len(scores)

    # Bootstrap resampling
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(scores, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))

    bootstrap_means = np.array(bootstrap_means)

    # Percentile method
    alpha = 1 - confidence
    ci_low = np.percentile(bootstrap_means, 100 * alpha / 2)
    ci_high = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))

    return float(np.mean(scores)), float(ci_low), float(ci_high)


def bootstrap_ci_difference(
    scores1: List[float],
    scores2: List[float],
    confidence: float = 0.95,
    n_bootstrap: int = 10000,
    random_state: int = 42
) -> Tuple[float, float, float, float]:
    """
    Bootstrap CI for the difference between two accuracy distributions.

    Useful for comparing baseline vs residualized accuracy.

    Args:
        scores1: First set of scores (e.g., baseline)
        scores2: Second set of scores (e.g., after residualization)
        confidence: Confidence level
        n_bootstrap: Number of bootstrap samples
        random_state: Random seed

    Returns:
        mean_diff: Mean difference (scores1 - scores2)
        ci_low: Lower bound of CI
        ci_high: Upper bound of CI
        p_value: Two-tailed p-value for difference != 0
    """
    rng = np.random.RandomState(random_state)
    scores1 = np.array(scores1)
    scores2 = np.array(scores2)

    # Paired bootstrap (same indices for both)
    n = min(len(scores1), len(scores2))
    bootstrap_diffs = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        diff = np.mean(scores1[:n][idx]) - np.mean(scores2[:n][idx])
        bootstrap_diffs.append(diff)

    bootstrap_diffs = np.array(bootstrap_diffs)

    alpha = 1 - confidence
    ci_low = np.percentile(bootstrap_diffs, 100 * alpha / 2)
    ci_high = np.percentile(bootstrap_diffs, 100 * (1 - alpha / 2))

    # Two-tailed p-value: proportion of bootstrap diffs on wrong side of 0
    observed_diff = np.mean(scores1) - np.mean(scores2)
    if observed_diff > 0:
        p_value = 2 * np.mean(bootstrap_diffs <= 0)
    else:
        p_value = 2 * np.mean(bootstrap_diffs >= 0)

    return float(observed_diff), float(ci_low), float(ci_high), float(min(p_value, 1.0))


def mcnemar_test(
    y_true: np.ndarray,
    pred1: np.ndarray,
    pred2: np.ndarray
) -> Tuple[float, float]:
    """
    McNemar's test for comparing two classifiers on the same data.

    Tests whether the classifiers have the same error rate.

    Args:
        y_true: True labels
        pred1: Predictions from classifier 1
        pred2: Predictions from classifier 2

    Returns:
        statistic: McNemar's chi-squared statistic
        p_value: Two-tailed p-value
    """
    # Build contingency table
    correct1 = (pred1 == y_true)
    correct2 = (pred2 == y_true)

    # b: clf1 correct, clf2 wrong
    b = np.sum(correct1 & ~correct2)
    # c: clf1 wrong, clf2 correct
    c = np.sum(~correct1 & correct2)

    # McNemar's test with continuity correction
    if b + c == 0:
        return 0.0, 1.0

    statistic = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = 1 - stats.chi2.cdf(statistic, df=1)

    return float(statistic), float(p_value)


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Compute Cohen's d effect size between two groups.

    Args:
        group1: First group of values
        group2: Second group of values

    Returns:
        d: Cohen's d (positive if group1 > group2)
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    d = (np.mean(group1) - np.mean(group2)) / pooled_std
    return float(d)


def lift_over_chance(accuracy: float, n_classes: int) -> float:
    """
    Compute lift over chance level.

    Args:
        accuracy: Observed accuracy
        n_classes: Number of classes

    Returns:
        lift: accuracy / chance_level
    """
    chance = 1 / n_classes
    return accuracy / chance


def run_full_statistical_analysis(
    X: np.ndarray,
    y: np.ndarray,
    stage_name: str,
    n_classes: int,
    fold_scores: List[float],
    n_permutations: int = 1000,
    confidence: float = 0.95,
    random_state: int = 42
) -> Dict:
    """
    Run comprehensive statistical analysis for a waterfall stage.

    Args:
        X: Feature/embedding matrix
        y: Labels
        stage_name: Name of the stage (for reporting)
        n_classes: Number of classes (for chance level)
        fold_scores: CV fold scores
        n_permutations: Number of permutations for test
        confidence: CI confidence level
        random_state: Random seed

    Returns:
        Dictionary with all statistical results
    """
    chance_level = 1 / n_classes
    observed_acc = np.mean(fold_scores)

    # Bootstrap CI
    mean_acc, ci_low, ci_high = bootstrap_ci(
        fold_scores, confidence=confidence, random_state=random_state)

    # Permutation test (only if accuracy > chance)
    if observed_acc > chance_level:
        p_value, null_dist = permutation_test_accuracy(
            X, y, observed_acc, n_permutations=n_permutations,
            random_state=random_state)
    else:
        p_value = 1.0
        null_dist = np.array([chance_level])

    # Effect size
    lift = lift_over_chance(observed_acc, n_classes)

    return {
        'stage': stage_name,
        'accuracy': observed_acc,
        'accuracy_std': float(np.std(fold_scores)),
        'ci_low': ci_low,
        'ci_high': ci_high,
        'confidence': confidence,
        'chance_level': chance_level,
        'lift_over_chance': lift,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'null_mean': float(np.mean(null_dist)),
        'null_std': float(np.std(null_dist))
    }


def format_results_table(results: List[Dict]) -> str:
    """
    Format statistical results as a markdown table.

    Args:
        results: List of result dictionaries from run_full_statistical_analysis

    Returns:
        Markdown table string
    """
    lines = [
        "| Stage | Accuracy | 95% CI | vs Chance | Lift | p-value | Sig? |",
        "|-------|----------|--------|-----------|------|---------|------|"
    ]

    for r in results:
        sig = "***" if r['p_value'] < 0.001 else "**" if r['p_value'] < 0.01 else "*" if r['p_value'] < 0.05 else ""
        lines.append(
            f"| {r['stage']} | {r['accuracy']:.1%} | [{r['ci_low']:.1%}, {r['ci_high']:.1%}] | "
            f"{r['chance_level']:.1%} | {r['lift_over_chance']:.1f}x | {r['p_value']:.4f} | {sig} |"
        )

    return "\n".join(lines)
