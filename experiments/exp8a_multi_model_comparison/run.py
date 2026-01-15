#!/usr/bin/env python3
"""
EXPERIMENT 8a: Multi-Model Comparison with Stylometry

Compare all 7 embedding models with stylometry:
- Embeddings only
- Stylometry only (FULL: char n-grams + word n-grams + function words)
- Combined (embeddings + stylometry)
- Residualization analysis (unique signal in each)

NOTE (Jan 2026): Updated to use full_combined stylometry (61.7%) instead of
char 3-grams only (58.4%). This is the proper baseline for fair comparison.

Models:
1. Gemini gemini-embedding-001 (API, 3072d)
2. OpenAI text-embedding-3-large (API, 3072d)
3. Voyage voyage-3-large (API, 1024d)
4. Qwen3-Embedding-8B (Local, 4096d)
5. multilingual-e5-large (Local, 1024d)
6. BAAI/bge-m3 (Local, 1024d)
7. Qwen3-Embedding-0.6B (Local, 1024d)
"""

import sys
import json
import numpy as np
from pathlib import Path
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Russian function words (for full stylometry baseline)
RUSSIAN_FUNCTION_WORDS = list(set([
    'я', 'ты', 'он', 'она', 'оно', 'мы', 'вы', 'они',
    'мне', 'тебе', 'ему', 'ей', 'нам', 'вам', 'им',
    'меня', 'тебя', 'его', 'её', 'нас', 'вас', 'их',
    'мой', 'твой', 'наш', 'ваш',
    'этот', 'тот', 'такой', 'какой', 'который', 'чей',
    'кто', 'что', 'весь', 'всё', 'сам', 'самый',
    'себя', 'себе', 'собой',
    'в', 'на', 'с', 'к', 'у', 'о', 'об', 'по', 'из', 'за',
    'от', 'до', 'для', 'при', 'над', 'под', 'перед', 'между',
    'через', 'без', 'про', 'ради', 'вместо', 'кроме',
    'и', 'а', 'но', 'да', 'или', 'либо', 'то', 'ни',
    'чтобы', 'как', 'когда', 'если', 'хотя', 'пока',
    'потому', 'поэтому', 'так', 'будто', 'словно', 'точно',
    'не', 'бы', 'же', 'ли', 'ведь', 'вот', 'вон',
    'даже', 'лишь', 'только', 'уже', 'ещё', 'еще', 'именно',
    'быть', 'есть', 'был', 'была', 'было', 'были', 'будет',
    'стать', 'стал', 'стала', 'мочь', 'может', 'могу',
    'где', 'куда', 'откуда', 'там', 'тут', 'здесь',
    'очень', 'тоже', 'также',
    'всегда', 'никогда', 'иногда', 'теперь', 'потом', 'опять',
]))

DATA_DIR = Path(__file__).parent.parent.parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Model configurations (updated for anonymized repo)
MODELS = {
    'Gemini': {
        'file': 'russian/embeddings_gemini.npy',
        'type': 'API',
        'dims': 3072
    },
    'OpenAI': {
        'file': 'russian/embeddings_openai.npy',
        'type': 'API',
        'dims': 3072
    },
    'Voyage': {
        'file': 'russian/embeddings_voyage.npy',
        'type': 'API',
        'dims': 1024
    },
    'Qwen-8B': {
        'file': 'russian/embeddings_qwen8b.npy',
        'type': 'Local',
        'dims': 4096
    },
    'E5-large': {
        'file': 'russian/embeddings_e5-large.npy',
        'type': 'Local',
        'dims': 1024
    },
    'BGE-M3': {
        'file': 'russian/embeddings_bge-m3.npy',
        'type': 'Local',
        'dims': 1024
    }
}


def run_cv(X, y, cv_folds=5, seed=42):
    """Run cross-validated classification."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring='accuracy')
    return float(np.mean(scores)), float(np.std(scores))


def analyze_model(model_name, embeddings, X_char, labels, chance):
    """Run full analysis for one embedding model."""
    results = {'name': model_name}

    # 1. Embeddings only
    acc_emb, std_emb = run_cv(embeddings, labels)
    results['emb_only'] = {'acc': acc_emb, 'std': std_emb}

    # 2. Combined (char + emb)
    X_combined = np.hstack([X_char, embeddings])
    acc_comb, std_comb = run_cv(X_combined, labels)
    results['combined'] = {'acc': acc_comb, 'std': std_comb}

    # 3. Residualization: char → embeddings
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_char, embeddings)
    emb_predicted = ridge.predict(X_char)
    emb_residual = embeddings - emb_predicted

    # Variance explained
    total_var = np.var(embeddings)
    residual_var = np.var(emb_residual)
    var_explained = 1 - (residual_var / total_var)
    results['var_explained_by_char'] = var_explained

    # Residual accuracy
    acc_resid, std_resid = run_cv(emb_residual, labels)
    results['residual_emb'] = {'acc': acc_resid, 'std': std_resid}

    # 4. Reverse: embeddings → char
    ridge_rev = Ridge(alpha=1.0)
    ridge_rev.fit(embeddings, X_char)
    char_predicted = ridge_rev.predict(embeddings)
    char_residual = X_char - char_predicted

    # Variance explained (reverse)
    total_var_char = np.var(X_char)
    residual_var_char = np.var(char_residual)
    var_explained_rev = 1 - (residual_var_char / total_var_char)
    results['var_explained_by_emb'] = var_explained_rev

    # Residual char accuracy
    acc_char_resid, std_char_resid = run_cv(char_residual, labels)
    results['residual_char'] = {'acc': acc_char_resid, 'std': std_char_resid}

    return results


def main():
    print("=" * 80)
    print("EXPERIMENT 8a: Multi-Model Comparison with Stylometry")
    print("=" * 80)

    # Load data
    print("\n[Loading data...]")
    from data_loader import load_dataset
    data = load_dataset('russian', embedding_model='gemini', include_poems=True)

    texts = [p['text'] for p in data['poems']]
    labels = data['labels']

    n_samples = len(labels)
    n_authors = len(data['author_list'])
    chance = 1 / n_authors

    print(f"  Samples: {n_samples}, Authors: {n_authors}")

    # Extract FULL stylometry features (char n-grams + word n-grams + function words)
    print("\n[Extracting full stylometry features...]")

    # Char n-grams (2-4)
    char_vec = TfidfVectorizer(analyzer='char', ngram_range=(2, 4), max_features=2000)
    X_char_ngrams = char_vec.fit_transform(texts)
    print(f"  Char n-grams (2-4): {X_char_ngrams.shape[1]} features")

    # Word n-grams (1-2)
    word_vec = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), max_features=2000)
    X_word_ngrams = word_vec.fit_transform(texts)
    print(f"  Word n-grams (1-2): {X_word_ngrams.shape[1]} features")

    # Function words
    func_vec = CountVectorizer(analyzer='word', vocabulary=RUSSIAN_FUNCTION_WORDS, lowercase=True)
    X_func_counts = func_vec.fit_transform(texts)
    row_sums = X_func_counts.sum(axis=1).A1
    row_sums[row_sums == 0] = 1
    X_func = X_func_counts.multiply(1 / row_sums[:, np.newaxis])
    print(f"  Function words: {X_func.shape[1]} features")

    # Combine all stylometry features
    X_stylometry_sparse = hstack([X_char_ngrams, X_word_ngrams, X_func])
    X_stylometry = X_stylometry_sparse.toarray()
    print(f"  Total stylometry features: {X_stylometry.shape[1]}")

    # Also keep char-only for backward compatibility
    X_char = X_char_ngrams.toarray()

    # Baseline: FULL stylometry
    print("\n[Stylometry baseline (full combined)...]")
    acc_styl, std_styl = run_cv(X_stylometry, labels)
    print(f"  Full stylometry accuracy: {acc_styl:.1%} ± {std_styl:.1%}")

    # Also report char-only for comparison
    acc_char_only, std_char_only = run_cv(X_char, labels)
    print(f"  Char n-grams only: {acc_char_only:.1%} ± {std_char_only:.1%}")

    # Analyze each model
    all_results = {
        'stylometry': {
            'full': {'acc': acc_styl, 'std': std_styl, 'features': X_stylometry.shape[1]},
            'char_only': {'acc': acc_char_only, 'std': std_char_only, 'features': X_char.shape[1]}
        },
        'chance': chance,
        'n_samples': n_samples,
        'n_authors': n_authors,
        'models': {}
    }

    # Use full stylometry for comparisons
    acc_baseline = acc_styl

    print("\n" + "=" * 80)
    print("ANALYZING ALL EMBEDDING MODELS")
    print("=" * 80)

    for model_name, config in MODELS.items():
        print(f"\n[{model_name}] ({config['type']}, {config['dims']}d)")

        # Load embeddings
        emb_file = DATA_DIR / config['file']
        if not emb_file.exists():
            print(f"  WARNING: File not found: {emb_file}")
            continue

        embeddings = np.load(emb_file)
        print(f"  Loaded: {embeddings.shape}")

        # Run analysis (use full stylometry)
        results = analyze_model(model_name, embeddings, X_stylometry, labels, chance)
        results['type'] = config['type']
        results['dims'] = config['dims']

        all_results['models'][model_name] = results

        # Print summary
        print(f"  Emb only: {results['emb_only']['acc']:.1%}")
        print(f"  Combined: {results['combined']['acc']:.1%} (+{results['combined']['acc'] - acc_baseline:.1%} vs stylometry)")
        print(f"  Var explained by stylometry: {results['var_explained_by_char']:.1%}")
        print(f"  Residual emb: {results['residual_emb']['acc']:.1%} ({results['residual_emb']['acc']/chance:.1f}× chance)")

    # Save results
    results_file = RESULTS_DIR / "multi_model_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)

    print(f"\n{'Model':<12} {'Type':<6} {'Emb Only':>10} {'Styl Only':>10} {'Combined':>10} {'Gain':>8} {'Residual':>8}")
    print("-" * 80)
    print(f"{'Stylometry':<12} {'-':<6} {'-':>10} {acc_baseline:>9.1%} {'-':>10} {'-':>8} {'-':>8}")
    print("-" * 80)

    # Sort by embedding-only accuracy (to see which model is best standalone)
    sorted_models = sorted(
        all_results['models'].items(),
        key=lambda x: x[1]['emb_only']['acc'],
        reverse=True
    )

    for model_name, results in sorted_models:
        gain = results['combined']['acc'] - acc_baseline
        emb_vs_styl = results['emb_only']['acc'] - acc_baseline
        marker = "*" if results['emb_only']['acc'] < acc_baseline else ""
        print(f"{model_name:<12} {results['type']:<6} "
              f"{results['emb_only']['acc']:>9.1%}{marker} "
              f"{acc_baseline:>9.1%} "
              f"{results['combined']['acc']:>9.1%} "
              f"{gain:>+7.1%} "
              f"{results['residual_emb']['acc']:>7.1%}")

    print("-" * 80)
    print("* = embedding alone performs WORSE than stylometry")

    # Key finding
    print("\n" + "=" * 80)
    print("KEY FINDING: STYLOMETRY vs EMBEDDINGS")
    print("=" * 80)

    # Count how many models beat stylometry
    models_beat_styl = sum(1 for _, r in all_results['models'].items()
                          if r['emb_only']['acc'] > acc_baseline)
    total_models = len(all_results['models'])

    print(f"\n  Full Stylometry baseline: {acc_baseline:.1%}")
    print(f"  Models that beat stylometry: {models_beat_styl}/{total_models}")

    # Best embedding model
    best_emb = sorted_models[0]
    diff = best_emb[1]['emb_only']['acc'] - acc_baseline
    winner = "EMBEDDING" if diff > 0 else "STYLOMETRY" if diff < 0 else "TIE"
    print(f"\n  Best embedding ({best_emb[0]}): {best_emb[1]['emb_only']['acc']:.1%}")
    print(f"  Difference from stylometry: {diff:+.1%}")
    print(f"  WINNER: {winner}")

    # Best model analysis
    print("\n" + "=" * 80)
    print("BEST MODEL ANALYSIS")
    print("=" * 80)

    best_combined = max(all_results['models'].items(), key=lambda x: x[1]['combined']['acc'])
    print(f"\nBest combined performance: {best_combined[0]}")
    print(f"  Combined accuracy: {best_combined[1]['combined']['acc']:.1%}")
    print(f"  Gain over stylometry: +{best_combined[1]['combined']['acc'] - acc_baseline:.1%}")
    print(f"  Stylometry explains {best_combined[1]['var_explained_by_char']:.1%} of embedding variance")
    print(f"  Unique to embeddings: {best_combined[1]['residual_emb']['acc']:.1%}")
    print(f"  Unique to stylometry: {best_combined[1]['residual_char']['acc']:.1%}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
