#!/usr/bin/env python3
"""
EXPERIMENT 0: Classical Stylometry Baseline

Purpose: Establish baseline using traditional stylometric methods before comparing with embeddings.

Methods implemented:
1. Character n-grams (2-4 grams) - State-of-the-art for many languages
2. Word n-grams (1-2 grams)
3. Most Frequent Words + Logistic Regression
4. Burrows' Delta (proper implementation with z-scores + Manhattan distance)
5. Cosine Delta variant
6. Function words (Russian)
7. Combined features

References:
- Burrows, J. (2002). 'Delta': A Measure of Stylistic Difference
- Stamatatos, E. (2013). On the robustness of authorship attribution based on character n-gram features
- Eder, M. (2017). Short samples in authorship attribution
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter
from itertools import combinations
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score, StratifiedKFold, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Constants
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# The published dataset is fixed at 200 poems per author (see data/README.md);
# every result in the paper is computed at this size.
N_PER_AUTHOR = 200


# Russian function words (common grammatical words) - deduplicated
RUSSIAN_FUNCTION_WORDS = list(set([
    # Pronouns
    'я', 'ты', 'он', 'она', 'оно', 'мы', 'вы', 'они',
    'мне', 'тебе', 'ему', 'ей', 'нам', 'вам', 'им',
    'меня', 'тебя', 'его', 'её', 'нас', 'вас', 'их',
    'мой', 'твой', 'наш', 'ваш',
    'этот', 'тот', 'такой', 'какой', 'который', 'чей',
    'кто', 'что', 'весь', 'всё', 'сам', 'самый',
    'себя', 'себе', 'собой',
    # Prepositions
    'в', 'на', 'с', 'к', 'у', 'о', 'об', 'по', 'из', 'за',
    'от', 'до', 'для', 'при', 'над', 'под', 'перед', 'между',
    'через', 'без', 'про', 'ради', 'вместо', 'кроме',
    # Conjunctions
    'и', 'а', 'но', 'да', 'или', 'либо', 'то', 'ни',
    'чтобы', 'как', 'когда', 'если', 'хотя', 'пока',
    'потому', 'поэтому', 'так', 'будто', 'словно', 'точно',
    # Particles
    'не', 'бы', 'же', 'ли', 'ведь', 'вот', 'вон',
    'даже', 'лишь', 'только', 'уже', 'ещё', 'еще', 'именно',
    # Auxiliary verbs / common verbs
    'быть', 'есть', 'был', 'была', 'было', 'были', 'будет',
    'стать', 'стал', 'стала', 'мочь', 'может', 'могу',
    # Adverbs
    'где', 'куда', 'откуда', 'там', 'тут', 'здесь',
    'очень', 'тоже', 'также',
    'всегда', 'никогда', 'иногда', 'теперь', 'потом', 'опять',
]))


class BurrowsDeltaClassifier(BaseEstimator, ClassifierMixin):
    """
    Burrows' Delta classifier - proper implementation.

    Algorithm:
    1. Normalize word frequencies by document length
    2. Z-score normalize each feature across the corpus
    3. Compute author centroids (mean z-score vector per author)
    4. Classify by minimum Manhattan distance to author centroid
    """

    def __init__(self, distance='manhattan'):
        self.distance = distance  # 'manhattan' (classic Delta) or 'cosine'

    def fit(self, X, y):
        self.classes_ = np.unique(y)

        # Z-score normalize features
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)

        # Compute author centroids
        self.centroids_ = {}
        for cls in self.classes_:
            mask = y == cls
            self.centroids_[cls] = X_scaled[mask].mean(axis=0)

        return self

    def predict(self, X):
        X_scaled = self.scaler_.transform(X)
        predictions = []

        for x in X_scaled:
            if self.distance == 'manhattan':
                # Classic Burrows' Delta: mean absolute difference
                distances = {
                    cls: np.mean(np.abs(x - centroid))
                    for cls, centroid in self.centroids_.items()
                }
            elif self.distance == 'cosine':
                # Cosine Delta: 1 - cosine similarity
                distances = {
                    cls: 1 - cosine_similarity(x.reshape(1, -1), centroid.reshape(1, -1))[0, 0]
                    for cls, centroid in self.centroids_.items()
                }
            else:
                raise ValueError(f"Unknown distance: {self.distance}")

            predictions.append(min(distances, key=distances.get))

        return np.array(predictions)

    def score(self, X, y):
        return (self.predict(X) == y).mean()


def extract_char_ngrams(texts, n_range=(2, 4), max_features=1000):
    """Extract character n-gram features."""
    vectorizer = TfidfVectorizer(
        analyzer='char',
        ngram_range=n_range,
        max_features=max_features,
        lowercase=True
    )
    return vectorizer.fit_transform(texts), vectorizer


def extract_word_ngrams(texts, n_range=(1, 2), max_features=1000):
    """Extract word n-gram features."""
    vectorizer = TfidfVectorizer(
        analyzer='word',
        ngram_range=n_range,
        max_features=max_features,
        lowercase=True
    )
    return vectorizer.fit_transform(texts), vectorizer


def extract_mfw(texts, n_words=100):
    """Extract Most Frequent Words features with document-length normalization."""
    vectorizer = CountVectorizer(
        analyzer='word',
        max_features=n_words,
        lowercase=True
    )
    counts = vectorizer.fit_transform(texts)
    # Normalize by document length (relative frequencies)
    row_sums = counts.sum(axis=1).A1
    row_sums[row_sums == 0] = 1
    normalized = counts.multiply(1 / row_sums[:, np.newaxis])
    return normalized.toarray(), vectorizer


def extract_function_words(texts, func_words=RUSSIAN_FUNCTION_WORDS):
    """Extract function word frequencies."""
    vectorizer = CountVectorizer(
        analyzer='word',
        vocabulary=func_words,
        lowercase=True
    )
    counts = vectorizer.fit_transform(texts)
    # Normalize by document length
    row_sums = counts.sum(axis=1).A1
    row_sums[row_sums == 0] = 1
    normalized = counts.multiply(1 / row_sums[:, np.newaxis])
    return normalized, vectorizer


def run_classification(X, y, cv_folds=5, method='logreg'):
    """Run cross-validated classification."""
    if method == 'logreg':
        clf = Pipeline([
            ('scaler', StandardScaler(with_mean=False)),
            ('clf', LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1))
        ])
    elif method == 'svm':
        clf = Pipeline([
            ('scaler', StandardScaler(with_mean=False)),
            ('clf', LinearSVC(max_iter=2000, C=1.0))
        ])
    elif method == 'delta':
        clf = BurrowsDeltaClassifier(distance='manhattan')
    elif method == 'cosine_delta':
        clf = BurrowsDeltaClassifier(distance='cosine')
    else:
        raise ValueError(f"Unknown method: {method}")

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
    return float(np.mean(scores)), float(np.std(scores))


def run_pairwise_classification(X, y, authors, cv_folds=5):
    """Run pairwise binary classification like EXP2."""
    author_to_idx = {a: i for i, a in enumerate(authors)}
    results = []

    for a1, a2 in combinations(authors, 2):
        idx1 = author_to_idx[a1]
        idx2 = author_to_idx[a2]

        mask = (y == idx1) | (y == idx2)
        X_pair = X[mask]
        y_pair = (y[mask] == idx2).astype(int)

        acc, std = run_classification(X_pair, y_pair, cv_folds=cv_folds)
        results.append({
            'author1': a1,
            'author2': a2,
            'accuracy': acc,
            'accuracy_std': std
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="EXP0: Classical Stylometry Baseline")
    parser.add_argument('--cv-folds', type=int, default=5,
                        help='CV folds')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--pairwise', action='store_true',
                        help='Also run pairwise classification (slow)')
    args = parser.parse_args()

    np.random.seed(args.seed)

    print("=" * 70)
    print("EXPERIMENT 0: Classical Stylometry Baseline (Russian)")
    print("=" * 70)

    # Load clean dataset
    print("\n[Loading clean dataset...]")
    from clean_dataset import load_clean_dataset

    data = load_clean_dataset(n_per_author=N_PER_AUTHOR, seed=args.seed, include_poems=True)

    texts = [p['text'] for p in data['poems']]
    labels = data['labels']
    authors = data['author_list']

    n_authors = len(authors)
    n_samples = len(texts)
    chance = 1 / n_authors

    print(f"  Authors: {n_authors}")
    print(f"  Samples: {n_samples}")
    print(f"  Poems per author: {N_PER_AUTHOR}")
    print(f"  Chance level: {chance:.1%}")

    # Results storage
    results = {
        'timestamp': datetime.now().isoformat(),
        'language': 'russian',
        'settings': {
            'n_authors': n_authors,
            'n_samples': n_samples,
            'n_per_author': N_PER_AUTHOR,
            'cv_folds': args.cv_folds,
            'seed': args.seed,
            'chance_level': chance
        },
        'methods': {}
    }

    # Method 1: Character n-grams (2-4)
    print("\n[Method 1: Character n-grams (2-4)]")
    X_char, _ = extract_char_ngrams(texts, n_range=(2, 4), max_features=2000)
    acc, std = run_classification(X_char, labels, cv_folds=args.cv_folds)
    print(f"  Accuracy: {acc:.1%} ± {std:.1%}")
    results['methods']['char_ngrams_2_4'] = {
        'accuracy': acc, 'std': std, 'features': X_char.shape[1]
    }

    # Method 2: Character n-grams (3-3 only)
    print("\n[Method 2: Character 3-grams only]")
    X_char3, _ = extract_char_ngrams(texts, n_range=(3, 3), max_features=2000)
    acc, std = run_classification(X_char3, labels, cv_folds=args.cv_folds)
    print(f"  Accuracy: {acc:.1%} ± {std:.1%}")
    results['methods']['char_ngrams_3'] = {
        'accuracy': acc, 'std': std, 'features': X_char3.shape[1]
    }

    # Method 3: MFW + Logistic Regression
    print("\n[Method 3: MFW (100) + Logistic Regression]")
    X_mfw, _ = extract_mfw(texts, n_words=100)
    acc, std = run_classification(X_mfw, labels, cv_folds=args.cv_folds)
    print(f"  Accuracy: {acc:.1%} ± {std:.1%}")
    results['methods']['mfw_100_logreg'] = {
        'accuracy': acc, 'std': std, 'features': 100
    }

    # Method 4: Burrows' Delta (proper implementation)
    print("\n[Method 4: Burrows' Delta (MFW 100, z-scores, Manhattan)]")
    acc, std = run_classification(X_mfw, labels, cv_folds=args.cv_folds, method='delta')
    print(f"  Accuracy: {acc:.1%} ± {std:.1%}")
    results['methods']['burrows_delta_100'] = {
        'accuracy': acc, 'std': std, 'features': 100
    }

    # Method 5: Burrows' Delta with 500 MFW
    print("\n[Method 5: Burrows' Delta (MFW 500)]")
    X_mfw500, _ = extract_mfw(texts, n_words=500)
    acc, std = run_classification(X_mfw500, labels, cv_folds=args.cv_folds, method='delta')
    print(f"  Accuracy: {acc:.1%} ± {std:.1%}")
    results['methods']['burrows_delta_500'] = {
        'accuracy': acc, 'std': std, 'features': 500
    }

    # Method 6: Cosine Delta
    print("\n[Method 6: Cosine Delta (MFW 100)]")
    acc, std = run_classification(X_mfw, labels, cv_folds=args.cv_folds, method='cosine_delta')
    print(f"  Accuracy: {acc:.1%} ± {std:.1%}")
    results['methods']['cosine_delta_100'] = {
        'accuracy': acc, 'std': std, 'features': 100
    }

    # Method 7: Function words
    print("\n[Method 7: Russian Function Words]")
    X_func, func_vec = extract_function_words(texts)
    actual_func = X_func.shape[1]
    acc, std = run_classification(X_func, labels, cv_folds=args.cv_folds)
    print(f"  Accuracy: {acc:.1%} ± {std:.1%} ({actual_func} features)")
    results['methods']['function_words'] = {
        'accuracy': acc, 'std': std, 'features': actual_func
    }

    # Method 8: Word n-grams (1-2)
    print("\n[Method 8: Word n-grams (1-2)]")
    X_word, _ = extract_word_ngrams(texts, n_range=(1, 2), max_features=2000)
    acc, std = run_classification(X_word, labels, cv_folds=args.cv_folds)
    print(f"  Accuracy: {acc:.1%} ± {std:.1%}")
    results['methods']['word_ngrams_1_2'] = {
        'accuracy': acc, 'std': std, 'features': X_word.shape[1]
    }

    # Method 9: Combined (char 3-grams + function words)
    print("\n[Method 9: Combined (char 3-grams + function words)]")
    from scipy.sparse import hstack
    X_combined = hstack([X_char3, X_func])
    acc, std = run_classification(X_combined, labels, cv_folds=args.cv_folds)
    print(f"  Accuracy: {acc:.1%} ± {std:.1%}")
    results['methods']['combined_char3_func'] = {
        'accuracy': acc, 'std': std, 'features': X_combined.shape[1]
    }

    # Method 10: Full combined
    print("\n[Method 10: Full Combined (char + word + func)]")
    X_full = hstack([X_char, X_word, X_func])
    acc, std = run_classification(X_full, labels, cv_folds=args.cv_folds)
    print(f"  Accuracy: {acc:.1%} ± {std:.1%}")
    results['methods']['full_combined'] = {
        'accuracy': acc, 'std': std, 'features': X_full.shape[1]
    }

    # Summary
    print("\n" + "=" * 70)
    print(f"SUMMARY: Multiclass Classification ({n_authors} Russian authors)")
    print("=" * 70)
    print(f"{'Method':<35} {'Accuracy':>10} {'Features':>10}")
    print("-" * 60)
    for method, mdata in sorted(results['methods'].items(), key=lambda x: -x[1]['accuracy']):
        print(f"{method:<35} {mdata['accuracy']:>9.1%} {mdata['features']:>10}")
    print("-" * 60)
    print(f"{'Chance level':<35} {chance:>9.1%}")

    # Find best method
    best_method = max(results['methods'].items(), key=lambda x: x[1]['accuracy'])
    results['best_method'] = best_method[0]
    results['best_accuracy'] = best_method[1]['accuracy']

    # Pairwise classification (optional, slow)
    if args.pairwise:
        print("\n" + "=" * 70)
        print("PAIRWISE BINARY CLASSIFICATION")
        print("=" * 70)

        pairwise_methods = {
            'char_3grams': X_char3,
            'char_2_4grams': X_char,
            'word_ngrams': X_word,
            'full_combined': X_full
        }

        results['pairwise'] = {}

        for method_name, X_method in pairwise_methods.items():
            print(f"\n[Pairwise: {method_name}]")
            pairwise_results = run_pairwise_classification(
                X_method.toarray() if hasattr(X_method, 'toarray') else X_method,
                labels, authors, cv_folds=args.cv_folds
            )

            sorted_pairs = sorted(pairwise_results, key=lambda x: x['accuracy'])
            mean_acc = np.mean([r['accuracy'] for r in pairwise_results])
            min_acc = min(r['accuracy'] for r in pairwise_results)
            max_acc = max(r['accuracy'] for r in pairwise_results)

            print(f"  Mean accuracy: {mean_acc:.1%}")
            print(f"  Min: {min_acc:.1%}, Max: {max_acc:.1%}")
            print(f"  Top-3 Most Similar:")
            for r in sorted_pairs[:3]:
                print(f"    {r['accuracy']:.1%}: {r['author1'].split()[-1]} vs {r['author2'].split()[-1]}")

            results['pairwise'][method_name] = {
                'results': pairwise_results,
                'mean_accuracy': mean_acc,
                'std_accuracy': np.std([r['accuracy'] for r in pairwise_results]),
                'min_accuracy': min_acc,
                'max_accuracy': max_acc
            }

            # Save individual pairwise results for comparison with exp2
            pairwise_output = {
                'timestamp': datetime.now().isoformat(),
                'settings': {
                    'mode': 'stylometry',
                    'feature_type': method_name,
                    'n_per_author': N_PER_AUTHOR,
                    'cv_folds': args.cv_folds,
                    'seed': args.seed,
                    'n_authors': n_authors,
                    'n_pairs_processed': len(pairwise_results)
                },
                'author_list': authors,
                'pairwise_results': pairwise_results,
                'statistics': {
                    'mean_accuracy': mean_acc,
                    'std_accuracy': np.std([r['accuracy'] for r in pairwise_results]),
                    'min_accuracy': min_acc,
                    'max_accuracy': max_acc
                },
                'top_similar': sorted_pairs[:10],
                'top_different': sorted_pairs[-10:][::-1]
            }
            pairwise_file = RESULTS_DIR / f"pairwise_stylometry_{method_name}_n{N_PER_AUTHOR}.json"
            with open(pairwise_file, 'w', encoding='utf-8') as f:
                json.dump(pairwise_output, f, indent=2, ensure_ascii=False)
            print(f"  Saved to: {pairwise_file.name}")

        # Summary comparison
        print("\n" + "-" * 60)
        print("PAIRWISE CLASSIFICATION SUMMARY")
        print("-" * 60)
        print(f"{'Method':<20} {'Mean Acc':>10} {'Min':>8} {'Max':>8}")
        print("-" * 60)
        for method_name, mdata in sorted(results['pairwise'].items(), key=lambda x: -x[1]['mean_accuracy']):
            print(f"{method_name:<20} {mdata['mean_accuracy']:>9.1%} {mdata['min_accuracy']:>7.1%} {mdata['max_accuracy']:>7.1%}")

    # Save results
    output_file = RESULTS_DIR / f"stylometry_results_n{N_PER_AUTHOR}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
