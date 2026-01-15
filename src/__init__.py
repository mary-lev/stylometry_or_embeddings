"""
Embedding Methodology Study - Shared Code

Core modules for the experiment pipeline:
- data_loader: Load and balance datasets
- features: Feature extraction (139 features across 4 tiers)
- residualization: Ridge, KernelRidge, Nyström approximation
- classifiers: Classifier configurations with conservative hyperparameters
- cv_pipeline: Proper CV with train-only fitting
- metrics: R², cosine similarity, accuracy with CIs
"""

from .data_loader import load_corpus, balance_by_author
from .features import extract_all_features, TIER_DEFINITIONS
from .residualization import Residualizer, compute_r2_diagnostics
from .classifiers import CLASSIFIERS, compare_classifiers
from .cv_pipeline import residualized_cv, full_experiment_cv
from .metrics import permutation_test, bootstrap_ci, accuracy_with_ci
