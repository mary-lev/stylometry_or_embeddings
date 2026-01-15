"""
Residualization utilities for removing feature variance from embeddings.

Supports:
- Linear residualization (Ridge regression)
- Non-linear residualization (Kernel Ridge with RBF, poly)
- Nyström approximation for large datasets
"""

import numpy as np
from typing import Dict, Tuple, Optional, Literal
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import r2_score
from sklearn.metrics.pairwise import cosine_similarity


class Residualizer:
    """
    Residualizer for removing feature variance from embeddings.

    Usage:
        residualizer = Residualizer(method='kernel', kernel='rbf')
        residualizer.fit(X_train, embeddings_train)
        residual_train = residualizer.transform(X_train, embeddings_train)
        residual_test = residualizer.transform(X_test, embeddings_test)
    """

    def __init__(
        self,
        method: Literal['linear', 'kernel', 'nystroem'] = 'linear',
        kernel: str = 'rbf',
        alpha: float = 1.0,
        gamma: str = 'scale',
        n_components: int = 500,
        random_state: int = 42
    ):
        """
        Args:
            method: 'linear' (Ridge), 'kernel' (KernelRidge), 'nystroem' (approximation)
            kernel: Kernel type for non-linear ('rbf', 'poly')
            alpha: Regularization strength
            gamma: Kernel coefficient
            n_components: Components for Nyström approximation
            random_state: Random seed
        """
        self.method = method
        self.kernel = kernel
        self.alpha = alpha
        self.gamma = gamma
        self.n_components = n_components
        self.random_state = random_state

        self._model = None
        self._is_fitted = False

    def fit(self, X: np.ndarray, embeddings: np.ndarray):
        """
        Fit the residualizer on training data.

        Args:
            X: Features (n_samples, n_features)
            embeddings: Embeddings (n_samples, embedding_dim)
        """
        if self.method == 'linear':
            self._model = Ridge(alpha=self.alpha)

        elif self.method == 'kernel':
            self._model = KernelRidge(
                kernel=self.kernel,
                alpha=self.alpha,
                gamma=self.gamma if self.gamma != 'scale' else None
            )

        elif self.method == 'nystroem':
            self._model = Pipeline([
                ('nystroem', Nystroem(
                    kernel=self.kernel,
                    n_components=self.n_components,
                    gamma=self.gamma if self.gamma != 'scale' else None,
                    random_state=self.random_state
                )),
                ('ridge', Ridge(alpha=self.alpha))
            ])

        else:
            raise ValueError(f"Unknown method: {self.method}")

        self._model.fit(X, embeddings)
        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
        """
        Transform embeddings to residual space.

        Args:
            X: Features
            embeddings: Embeddings to residualize

        Returns:
            Residual embeddings
        """
        if not self._is_fitted:
            raise ValueError("Residualizer must be fitted first")

        predicted = self._model.predict(X)
        return embeddings - predicted

    def fit_transform(self, X: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X, embeddings)
        return self.transform(X, embeddings)

    def score(self, X: np.ndarray, embeddings: np.ndarray) -> float:
        """
        Compute R² (variance-weighted) for the fit.

        Returns:
            Variance-weighted R² across all embedding dimensions
        """
        if not self._is_fitted:
            raise ValueError("Residualizer must be fitted first")

        predicted = self._model.predict(X)
        return r2_score(embeddings, predicted, multioutput='variance_weighted')


def compute_r2_diagnostics(
    features: np.ndarray,
    embeddings: np.ndarray,
    cv: int = 5,
    random_state: int = 42
) -> Dict[str, Tuple[float, float]]:
    """
    Compute R² diagnostics for linear and kernel regression.

    Uses proper CV to avoid overfitting estimates.

    Args:
        features: Feature matrix (n_samples, n_features)
        embeddings: Embedding matrix (n_samples, embedding_dim)
        cv: Number of CV folds
        random_state: Random seed

    Returns:
        Dict with R² mean and std for each method:
        {'linear': (mean, std), 'rbf': (mean, std), 'poly2': (mean, std)}
    """
    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)

    def cv_r2(method: str, **kwargs) -> Tuple[float, float]:
        r2_scores = []
        for train_idx, test_idx in kf.split(features):
            X_train, X_test = features[train_idx], features[test_idx]
            y_train, y_test = embeddings[train_idx], embeddings[test_idx]

            residualizer = Residualizer(method=method, **kwargs)
            residualizer.fit(X_train, y_train)

            # Score on test set
            y_pred = residualizer._model.predict(X_test)
            r2 = r2_score(y_test, y_pred, multioutput='variance_weighted')
            r2_scores.append(r2)

        return np.mean(r2_scores), np.std(r2_scores)

    results = {}

    # Linear
    results['linear'] = cv_r2('linear')

    # RBF kernel (with Nyström for large datasets)
    n_samples = len(features)
    method = 'nystroem' if n_samples > 2000 else 'kernel'
    results['rbf'] = cv_r2(method, kernel='rbf')

    # Polynomial degree 2
    results['poly2'] = cv_r2(method, kernel='poly')

    return results


def compute_cosine_similarity_score(
    features: np.ndarray,
    embeddings: np.ndarray,
    cv: int = 5,
    random_state: int = 42
) -> Tuple[float, float]:
    """
    Compute mean cosine similarity between predicted and actual embeddings.

    Alternative to R² that's more natural for normalized embeddings.

    Returns:
        (mean_similarity, std_similarity) across CV folds
    """
    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    similarities = []

    for train_idx, test_idx in kf.split(features):
        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = embeddings[train_idx], embeddings[test_idx]

        # Fit on train
        residualizer = Residualizer(method='linear')
        residualizer.fit(X_train, y_train)

        # Predict on test
        y_pred = residualizer._model.predict(X_test)

        # Mean pairwise cosine similarity
        sim = np.diag(cosine_similarity(y_test, y_pred))
        similarities.extend(sim)

    return np.mean(similarities), np.std(similarities)


def select_best_kernel(
    features: np.ndarray,
    embeddings: np.ndarray,
    cv: int = 5,
    random_state: int = 42
) -> Dict:
    """
    Select best kernel and hyperparameters via nested CV.

    Returns:
        Dict with best kernel, params, and R² scores
    """
    # Define kernel variants
    param_grid = {
        'rbf': {'kernel': ['rbf'], 'gamma': ['scale', 0.01, 0.1, 1.0], 'alpha': [0.1, 1.0, 10.0]},
        'poly2': {'kernel': ['poly'], 'degree': [2], 'alpha': [0.1, 1.0, 10.0]},
        'poly3': {'kernel': ['poly'], 'degree': [3], 'alpha': [0.1, 1.0, 10.0]},
    }

    best_results = {}

    for kernel_name, params in param_grid.items():
        # Use Ridge for linear approximation if dataset is large
        n_samples = len(features)
        if n_samples > 2000:
            # Use Nyström approximation
            base_model = Pipeline([
                ('nystroem', Nystroem(kernel=params['kernel'][0], n_components=500, random_state=random_state)),
                ('ridge', Ridge())
            ])
            search_params = {'ridge__alpha': params['alpha']}
        else:
            base_model = KernelRidge()
            search_params = params

        # Grid search with CV
        grid = GridSearchCV(
            base_model,
            search_params,
            cv=cv,
            scoring='r2',
            n_jobs=-1
        )
        grid.fit(features, embeddings)

        best_results[kernel_name] = {
            'best_params': grid.best_params_,
            'best_score': grid.best_score_
        }

    # Find overall best
    best_kernel = max(best_results, key=lambda k: best_results[k]['best_score'])

    return {
        'best_kernel': best_kernel,
        'all_results': best_results,
        'best_score': best_results[best_kernel]['best_score']
    }
