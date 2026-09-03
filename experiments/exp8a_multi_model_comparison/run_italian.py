#!/usr/bin/env python3
"""
EXPERIMENT 8a: Multi-Model Comparison with Stylometry (ITALIAN)

Compare available Italian embedding models with stylometry:
- Embeddings only
- Stylometry only (FULL: char n-grams + word n-grams + function words)
- Combined (embeddings + stylometry)
- Residualization analysis (unique signal in each)

Available Italian Models:
1. Gemini gemini-embedding-001 (API, 3072d)
2. OpenAI text-embedding-3-large (API, 3072d)
3. Voyage voyage-3-large (API, 1024d)
4. multilingual-e5-large (Local, 1024d)
"""

import sys
import json
import argparse
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

# Italian function words
ITALIAN_FUNCTION_WORDS = list(set([
    # Personal pronouns
    'io', 'tu', 'lui', 'lei', 'noi', 'voi', 'loro', 'esso', 'essa',
    'mi', 'ti', 'ci', 'vi', 'si', 'lo', 'la', 'li', 'le', 'ne',
    'me', 'te', 'sé',
    # Possessive adjectives/pronouns
    'mio', 'mia', 'miei', 'mie', 'tuo', 'tua', 'tuoi', 'tue',
    'suo', 'sua', 'suoi', 'sue', 'nostro', 'nostra', 'nostri', 'nostre',
    'vostro', 'vostra', 'vostri', 'vostre', 'loro',
    # Demonstratives
    'questo', 'questa', 'questi', 'queste', 'quello', 'quella', 'quelli', 'quelle',
    'ciò', 'stesso', 'stessa', 'stessi', 'stesse',
    # Relative/interrogative pronouns
    'che', 'chi', 'cui', 'quale', 'quali', 'quanto', 'quanta', 'quanti', 'quante',
    # Indefinites
    'tutto', 'tutta', 'tutti', 'tutte', 'ogni', 'qualche', 'alcuno', 'alcuna',
    'nessuno', 'nessuna', 'altro', 'altra', 'altri', 'altre', 'molto', 'molta',
    'molti', 'molte', 'poco', 'poca', 'pochi', 'poche', 'tanto', 'tanta',
    # Prepositions
    'di', 'a', 'da', 'in', 'con', 'su', 'per', 'tra', 'fra',
    'del', 'della', 'dei', 'delle', 'dello', 'degli',
    'al', 'alla', 'ai', 'alle', 'allo', 'agli',
    'dal', 'dalla', 'dai', 'dalle', 'dallo', 'dagli',
    'nel', 'nella', 'nei', 'nelle', 'nello', 'negli',
    'sul', 'sulla', 'sui', 'sulle', 'sullo', 'sugli',
    # Articles
    'il', 'lo', 'la', 'i', 'gli', 'le', 'un', 'uno', 'una',
    # Conjunctions
    'e', 'ed', 'o', 'ma', 'però', 'quindi', 'dunque', 'perché', 'poiché',
    'quando', 'mentre', 'se', 'come', 'dove', 'finché', 'benché', 'sebbene',
    'affinché', 'purché', 'né', 'sia', 'oppure', 'ovvero', 'cioè',
    # Adverbs
    'non', 'più', 'mai', 'sempre', 'anche', 'ancora', 'già', 'ora', 'poi',
    'dove', 'qui', 'qua', 'là', 'lì', 'così', 'come', 'molto', 'poco',
    'bene', 'male', 'solo', 'soltanto', 'forse', 'quasi', 'troppo',
    # Auxiliary verbs
    'essere', 'è', 'era', 'sono', 'sei', 'siamo', 'siete', 'erano', 'sarà',
    'avere', 'ha', 'ho', 'hai', 'abbiamo', 'avete', 'hanno', 'aveva', 'avrà',
    'fare', 'fa', 'fai', 'fanno', 'faceva',
    'potere', 'può', 'posso', 'puoi', 'possiamo', 'possono', 'poteva',
    'dovere', 'deve', 'devo', 'devi', 'dobbiamo', 'devono', 'doveva',
    'volere', 'vuole', 'voglio', 'vuoi', 'vogliamo', 'vogliono', 'voleva',
]))

DATA_DIR = Path(__file__).parent.parent.parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Italian model configurations (updated for anonymized repo)
MODELS = {
    'Gemini': {
        'file': 'italian/embeddings_gemini.npy',
        'type': 'API',
        'dims': 3072
    },
    'OpenAI': {
        'file': 'italian/embeddings_openai.npy',
        'type': 'API',
        'dims': 3072
    },
    'Voyage': {
        'file': 'italian/embeddings_voyage.npy',
        'type': 'API',
        'dims': 1024
    },
    'Qwen-8B': {
        'file': 'italian/embeddings_qwen8b.npy',
        'type': 'Local',
        'dims': 4096
    },
    'E5-large': {
        'file': 'italian/embeddings_e5-large.npy',
        'type': 'Local',
        'dims': 1024
    },
}


def run_cv(X, y, cv_folds=5, seed=42):
    """Run cross-validated classification."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring='accuracy')
    return float(np.mean(scores)), float(np.std(scores))


def analyze_model(model_name, embeddings, X_stylometry, labels, chance):
    """Run full analysis for one embedding model."""
    results = {'name': model_name}

    # 1. Embeddings only
    acc_emb, std_emb = run_cv(embeddings, labels)
    results['emb_only'] = {'acc': acc_emb, 'std': std_emb}

    # 2. Combined (stylometry + embeddings)
    X_combined = np.hstack([X_stylometry, embeddings])
    acc_comb, std_comb = run_cv(X_combined, labels)
    results['combined'] = {'acc': acc_comb, 'std': std_comb}

    # 3. Residualization: stylometry → embeddings
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_stylometry, embeddings)
    emb_predicted = ridge.predict(X_stylometry)
    emb_residual = embeddings - emb_predicted

    # Variance explained
    total_var = np.var(embeddings)
    residual_var = np.var(emb_residual)
    var_explained = 1 - (residual_var / total_var)
    results['var_explained_by_styl'] = var_explained

    # Residual accuracy
    acc_resid, std_resid = run_cv(emb_residual, labels)
    results['residual_emb'] = {'acc': acc_resid, 'std': std_resid}

    # 4. Reverse: embeddings → stylometry
    ridge_rev = Ridge(alpha=1.0)
    ridge_rev.fit(embeddings, X_stylometry)
    styl_predicted = ridge_rev.predict(embeddings)
    styl_residual = X_stylometry - styl_predicted

    # Variance explained (reverse)
    total_var_styl = np.var(X_stylometry)
    residual_var_styl = np.var(styl_residual)
    var_explained_rev = 1 - (residual_var_styl / total_var_styl)
    results['var_explained_by_emb'] = var_explained_rev

    # Residual stylometry accuracy
    acc_styl_resid, std_styl_resid = run_cv(styl_residual, labels)
    results['residual_styl'] = {'acc': acc_styl_resid, 'std': std_styl_resid}

    return results


def main():
    parser = argparse.ArgumentParser(
        description="EXP8a (Italian): Multi-Model Comparison with Stylometry")
    parser.add_argument('--stylometry', dest='stylometry_set', default='char3_func',
                        choices=['char3_func', 'full'],
                        help="Stylometry feature set. 'char3_func' (default) = char "
                             "3-grams + function words, matching the paper's tables "
                             "and published figures. 'full' = char 2-4 grams + word "
                             "1-2 grams + function words.")
    stylometry_set = parser.parse_args().stylometry_set
    print(f"Stylometry feature set: {stylometry_set}")
    print("=" * 80)
    print("EXPERIMENT 8a: Multi-Model Comparison with Stylometry (ITALIAN)")
    print("=" * 80)

    # Load Italian texts and labels
    print("\n[Loading Italian data...]")
    from data_loader import load_dataset

    data = load_dataset('italian', embedding_model='gemini', include_poems=True)
    texts = [p['text'] for p in data['poems']]
    labels = data['labels']

    n_samples = len(labels)
    n_authors = len(np.unique(labels))
    chance = 1 / n_authors

    print(f"  Samples: {n_samples}, Authors: {n_authors}")
    print(f"  Chance level: {chance:.2%}")

    # Stylometry features -- see --stylometry.
    print(f"\n[Extracting stylometry features: {stylometry_set}...]")

    # Char n-grams (2-4) also serve as the "char only" reference below
    char_vec = TfidfVectorizer(analyzer='char', ngram_range=(2, 4), max_features=2000)
    X_char_ngrams = char_vec.fit_transform(texts)
    print(f"  Char n-grams (2-4): {X_char_ngrams.shape[1]} features")

    if stylometry_set == 'full':
        word_vec = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), max_features=2000)
        X_word_ngrams = word_vec.fit_transform(texts)
        print(f"  Word n-grams (1-2): {X_word_ngrams.shape[1]} features")
    else:
        char3_vec = TfidfVectorizer(analyzer='char', ngram_range=(3, 3), max_features=2000)
        X_char3 = char3_vec.fit_transform(texts)
        print(f"  Char 3-grams: {X_char3.shape[1]} features")

    # Function words
    func_vec = CountVectorizer(analyzer='word', vocabulary=ITALIAN_FUNCTION_WORDS, lowercase=True)
    X_func_counts = func_vec.fit_transform(texts)
    row_sums = X_func_counts.sum(axis=1).A1
    row_sums[row_sums == 0] = 1
    X_func = X_func_counts.multiply(1 / row_sums[:, np.newaxis])
    print(f"  Function words: {X_func.shape[1]} features")

    # Combine stylometry features
    if stylometry_set == 'full':
        X_stylometry_sparse = hstack([X_char_ngrams, X_word_ngrams, X_func])
    else:
        X_stylometry_sparse = hstack([X_char3, X_func])
    X_stylometry = X_stylometry_sparse.toarray()
    print(f"  Total stylometry features: {X_stylometry.shape[1]}")

    # Baseline: FULL stylometry
    print("\n[Stylometry baseline (full combined)...]")
    acc_styl, std_styl = run_cv(X_stylometry, labels)
    print(f"  Full stylometry accuracy: {acc_styl:.1%} ± {std_styl:.1%}")

    # Also report char-only for comparison
    X_char = X_char_ngrams.toarray()
    acc_char_only, std_char_only = run_cv(X_char, labels)
    print(f"  Char n-grams only: {acc_char_only:.1%} ± {std_char_only:.1%}")

    # Analyze each model
    all_results = {
        'corpus': 'italian',
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

    skipped = []
    for model_name, config in MODELS.items():
        print(f"\n[{model_name}] ({config['type']}, {config['dims']}d)")

        # Load embeddings
        emb_file = DATA_DIR / config['file']
        if not emb_file.exists():
            print(f"  SKIPPED: {emb_file} not downloaded")
            skipped.append(model_name)
            continue

        embeddings = np.load(emb_file)
        print(f"  Loaded: {embeddings.shape}")

        # Run analysis
        results = analyze_model(model_name, embeddings, X_stylometry, labels, chance)
        results['type'] = config['type']
        results['dims'] = config['dims']

        all_results['models'][model_name] = results

        # Print summary
        print(f"  Emb only: {results['emb_only']['acc']:.1%}")
        print(f"  Combined: {results['combined']['acc']:.1%} (+{results['combined']['acc'] - acc_baseline:.1%} vs stylometry)")
        print(f"  Var explained by stylometry: {results['var_explained_by_styl']:.1%}")
        print(f"  Residual emb: {results['residual_emb']['acc']:.1%} ({results['residual_emb']['acc']/chance:.1f}× chance)")

    if skipped:
        print("\n" + "!" * 78)
        print(f"WARNING: {len(skipped)} of {len(MODELS)} models were skipped "
              f"(embeddings not downloaded):")
        print("  " + ", ".join(skipped))
        print("Results and figures below cover only the models present. For the")
        print("full comparison run:  python download_data.py --all")
        print("!" * 78)

    # Save results
    results_file = RESULTS_DIR / "multi_model_results_italian.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE (ITALIAN)")
    print("=" * 80)

    print(f"\n{'Model':<12} {'Type':<6} {'Emb Only':>10} {'Styl Only':>10} {'Combined':>10} {'Gain':>8} {'Residual':>8}")
    print("-" * 80)
    print(f"{'Stylometry':<12} {'-':<6} {'-':>10} {acc_baseline:>9.1%} {'-':>10} {'-':>8} {'-':>8}")
    print("-" * 80)

    # Sort by embedding-only accuracy
    sorted_models = sorted(
        all_results['models'].items(),
        key=lambda x: x[1]['emb_only']['acc'],
        reverse=True
    )

    for model_name, results in sorted_models:
        gain = results['combined']['acc'] - acc_baseline
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
    print("KEY FINDING: STYLOMETRY vs EMBEDDINGS (ITALIAN)")
    print("=" * 80)

    # Count how many models beat stylometry
    models_beat_styl = sum(1 for _, r in all_results['models'].items()
                          if r['emb_only']['acc'] > acc_baseline)
    total_models = len(all_results['models'])

    print(f"\n  Full Stylometry baseline: {acc_baseline:.1%}")
    print(f"  Models that beat stylometry: {models_beat_styl}/{total_models}")

    # Best embedding model
    if sorted_models:
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
        print(f"  Stylometry explains {best_combined[1]['var_explained_by_styl']:.1%} of embedding variance")
        print(f"  Unique to embeddings: {best_combined[1]['residual_emb']['acc']:.1%}")
        print(f"  Unique to stylometry: {best_combined[1]['residual_styl']['acc']:.1%}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
