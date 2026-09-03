#!/usr/bin/env python3
"""
EXP4 Diagnostic: Compare error patterns between Embeddings and Stylometry methods.

Hypothesis: Different classification methods may be affected differently by surface
text features (length, vocabulary richness, etc.).

Tests:
1. Does text length affect embeddings and stylometry differently?
2. Is TTR (vocabulary richness) more problematic for one method?
3. Do the methods make errors on the SAME texts or different ones?
"""

import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Constants
DATA_DIR = Path(__file__).parent.parent.parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# The published dataset is fixed at 200 poems per author (see data/README.md);
# every result in the paper is computed at this size.
N_PER_AUTHOR = 200


# Russian function words
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

# Italian function words
ITALIAN_FUNCTION_WORDS = list(set([
    'il', 'lo', 'la', 'i', 'gli', 'le', 'un', 'uno', 'una',
    'di', 'a', 'da', 'in', 'con', 'su', 'per', 'tra', 'fra',
    'del', 'dello', 'della', 'dei', 'degli', 'delle',
    'al', 'allo', 'alla', 'ai', 'agli', 'alle',
    'dal', 'dallo', 'dalla', 'dai', 'dagli', 'dalle',
    'nel', 'nello', 'nella', 'nei', 'negli', 'nelle',
    'sul', 'sullo', 'sulla', 'sui', 'sugli', 'sulle',
    'io', 'tu', 'lui', 'lei', 'noi', 'voi', 'loro', 'egli', 'ella',
    'mi', 'ti', 'ci', 'vi', 'si', 'lo', 'la', 'li', 'le', 'ne',
    'me', 'te', 'sé', 'ce', 've',
    'mio', 'tuo', 'suo', 'nostro', 'vostro', 'loro',
    'questo', 'questa', 'questi', 'queste', 'quello', 'quella',
    'chi', 'che', 'cui', 'quale', 'quali', 'quanto',
    'e', 'ed', 'o', 'od', 'ma', 'però', 'anzi', 'né',
    'se', 'quando', 'mentre', 'perché', 'poiché',
    'come', 'siccome', 'benché', 'sebbene',
    'dunque', 'quindi', 'pertanto', 'perciò', 'onde', 'ché',
    'non', 'più', 'mai', 'sempre', 'già', 'ancora', 'ora',
    'poi', 'dopo', 'prima', 'dove', 'qui', 'qua', 'là', 'lì',
    'molto', 'poco', 'troppo', 'tanto', 'così', 'pure', 'anche',
    'essere', 'è', 'era', 'sono', 'sei', 'siamo', 'erano', 'fu',
    'avere', 'ha', 'ho', 'hai', 'hanno', 'abbiamo', 'aveva', 'ebbe',
]))


def load_dataset(language='russian', embedding_model='gemini', n_per_author=200, seed=42):
    """Load the published dataset with texts and embeddings (see data/README.md)."""
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
    from data_loader import load_dataset as _load

    d = _load(language, embedding_model=embedding_model, include_poems=True)
    embeddings = d['embeddings']
    labels = d['labels']
    author_list = d['author_list']
    texts = [p['text'] for p in d['poems']]
    func_words = RUSSIAN_FUNCTION_WORDS if language == 'russian' else ITALIAN_FUNCTION_WORDS

    return {
        'embeddings': embeddings,
        'labels': labels,
        'author_list': author_list,
        'texts': texts,
        'func_words': func_words
    }


def extract_surface_features(texts):
    """Extract surface features from texts."""
    features = []
    for text in texts:
        char_count = len(text)
        words = text.split()
        word_count = len(words)
        lines = text.strip().split('\n')
        line_count = len(lines)

        avg_word_len = np.mean([len(w) for w in words]) if words else 0
        unique_words = set(w.lower() for w in words)
        ttr = len(unique_words) / word_count if word_count > 0 else 0

        punct_chars = sum(1 for c in text if c in '.,;:!?—–-«»""\'()[]')
        punct_density = punct_chars / char_count if char_count > 0 else 0

        features.append({
            'char_count': char_count,
            'word_count': word_count,
            'line_count': line_count,
            'avg_word_len': avg_word_len,
            'ttr': ttr,
            'punct_density': punct_density
        })
    return features


def extract_stylometry_features(texts, func_words):
    """Extract stylometry features (char 3-grams + function words)."""
    # Character 3-grams
    char_vec = TfidfVectorizer(analyzer='char', ngram_range=(3, 3), max_features=2000)
    X_char = char_vec.fit_transform(texts)

    # Function words
    func_vec = CountVectorizer(vocabulary=func_words, lowercase=True)
    X_func = func_vec.fit_transform(texts)
    row_sums = X_func.sum(axis=1).A1
    row_sums[row_sums == 0] = 1
    X_func_norm = X_func.multiply(1 / row_sums[:, np.newaxis])

    # Combine
    X_combined = hstack([X_char, X_func_norm])
    return X_combined, char_vec, func_vec


def run_classification_with_predictions(X, y, cv_folds=5, seed=42, sparse=False):
    """Run classification and return per-sample predictions."""
    if sparse:
        clf = Pipeline([
            ('scaler', StandardScaler(with_mean=False)),
            ('clf', LogisticRegression(max_iter=2000, solver='lbfgs'))
        ])
    else:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        clf = LogisticRegression(max_iter=2000, solver='lbfgs')

    y_pred = cross_val_predict(clf, X, y, cv=cv_folds)
    y_proba = cross_val_predict(clf, X, y, cv=cv_folds, method='predict_proba')

    confidence = np.array([y_proba[i, y_pred[i]] for i in range(len(y_pred))])
    true_class_prob = np.array([y_proba[i, y[i]] for i in range(len(y))])

    return {
        'y_pred': y_pred,
        'y_proba': y_proba,
        'confidence': confidence,
        'true_class_prob': true_class_prob,
        'correct': y_pred == y,
        'accuracy': np.mean(y_pred == y)
    }


def analyze_feature_correlation(surface_features, predictions, feature_name):
    """Analyze correlation between a surface feature and misclassification."""
    feature_vals = np.array([f[feature_name] for f in surface_features])
    correct = predictions['correct']

    corr, p_val = stats.pointbiserialr(correct, feature_vals)

    mean_correct = np.mean(feature_vals[correct])
    mean_incorrect = np.mean(feature_vals[~correct])

    return {
        'feature': feature_name,
        'correlation': corr,
        'p_value': p_val,
        'mean_correct': mean_correct,
        'mean_incorrect': mean_incorrect,
        'significant': p_val < 0.05
    }


def compute_error_overlap(pred1, pred2):
    """Compute overlap between errors made by two methods."""
    errors1 = ~pred1['correct']
    errors2 = ~pred2['correct']

    both_correct = (~errors1) & (~errors2)
    both_wrong = errors1 & errors2
    only_emb_wrong = errors1 & (~errors2)
    only_styl_wrong = (~errors1) & errors2

    n_total = len(errors1)

    # Jaccard similarity of errors
    intersection = np.sum(both_wrong)
    union = np.sum(errors1 | errors2)
    jaccard = intersection / union if union > 0 else 0

    # Cohen's Kappa for error agreement
    p_o = np.mean(errors1 == errors2)  # Observed agreement
    p_e = (np.mean(errors1) * np.mean(errors2) +
           np.mean(~errors1) * np.mean(~errors2))  # Expected agreement
    kappa = (p_o - p_e) / (1 - p_e) if p_e < 1 else 0

    return {
        'both_correct': int(np.sum(both_correct)),
        'both_wrong': int(np.sum(both_wrong)),
        'only_emb_wrong': int(np.sum(only_emb_wrong)),
        'only_styl_wrong': int(np.sum(only_styl_wrong)),
        'jaccard_similarity': jaccard,
        'kappa': kappa,
        'pct_both_correct': np.sum(both_correct) / n_total,
        'pct_both_wrong': np.sum(both_wrong) / n_total,
        'pct_only_emb_wrong': np.sum(only_emb_wrong) / n_total,
        'pct_only_styl_wrong': np.sum(only_styl_wrong) / n_total
    }


def create_comparison_plot(surface_features, emb_pred, styl_pred, feature_name, language, output_dir):
    """Create comparison plot of error rates for embeddings vs stylometry."""
    feature_vals = np.array([f[feature_name] for f in surface_features])
    emb_correct = emb_pred['correct']
    styl_correct = styl_pred['correct']

    # Create bins
    n_bins = 10
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(feature_vals, percentiles)
    bin_indices = np.digitize(feature_vals, bin_edges[1:-1])

    emb_error_rates = []
    styl_error_rates = []
    bin_centers = []

    for i in range(n_bins):
        mask = bin_indices == i
        if np.sum(mask) > 0:
            emb_error_rates.append(1 - np.mean(emb_correct[mask]))
            styl_error_rates.append(1 - np.mean(styl_correct[mask]))
            bin_centers.append(np.mean(feature_vals[mask]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Bar comparison
    x = np.arange(len(emb_error_rates))
    width = 0.35

    bars1 = ax1.bar(x - width/2, emb_error_rates, width, label='Embeddings', color='steelblue', alpha=0.7)
    bars2 = ax1.bar(x + width/2, styl_error_rates, width, label='Stylometry', color='coral', alpha=0.7)

    ax1.set_xlabel(f'{feature_name} (decile)')
    ax1.set_ylabel('Error Rate')
    ax1.set_title(f'Error Rate by {feature_name}: Embeddings vs Stylometry ({language.title()})')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{i+1}' for i in range(len(emb_error_rates))])
    ax1.legend()

    # Difference plot
    diff = np.array(emb_error_rates) - np.array(styl_error_rates)
    colors = ['green' if d < 0 else 'red' for d in diff]
    ax2.bar(x, diff, color=colors, alpha=0.7)
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.set_xlabel(f'{feature_name} (decile)')
    ax2.set_ylabel('Error Difference (Emb - Styl)')
    ax2.set_title('Difference in Error Rate\n(Green = Embeddings better, Red = Stylometry better)')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{i+1}' for i in range(len(diff))])

    plt.tight_layout()
    plt.savefig(output_dir / f'comparison_{feature_name}_{language}.png', dpi=150)
    plt.close()

    return emb_error_rates, styl_error_rates


def create_overlap_plot(emb_pred, styl_pred, language, output_dir):
    """Create Venn-style visualization of error overlap."""
    overlap = compute_error_overlap(emb_pred, styl_pred)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Stacked bar
    categories = ['Errors']
    both = overlap['pct_both_wrong'] * 100
    only_emb = overlap['pct_only_emb_wrong'] * 100
    only_styl = overlap['pct_only_styl_wrong'] * 100

    ax1.bar(categories, [both], label=f'Both wrong ({overlap["both_wrong"]})', color='purple', alpha=0.7)
    ax1.bar(categories, [only_emb], bottom=[both], label=f'Only Emb wrong ({overlap["only_emb_wrong"]})', color='steelblue', alpha=0.7)
    ax1.bar(categories, [only_styl], bottom=[both + only_emb], label=f'Only Styl wrong ({overlap["only_styl_wrong"]})', color='coral', alpha=0.7)

    ax1.set_ylabel('Percentage of Total Samples')
    ax1.set_title(f'Error Decomposition ({language.title()})\nJaccard={overlap["jaccard_similarity"]:.2f}, κ={overlap["kappa"]:.2f}')
    ax1.legend(loc='upper right')
    ax1.set_ylim(0, 50)

    # Confusion matrix of correctness
    conf = np.array([
        [overlap['both_correct'], overlap['only_styl_wrong']],
        [overlap['only_emb_wrong'], overlap['both_wrong']]
    ])

    im = ax2.imshow(conf, cmap='Blues')
    ax2.set_xticks([0, 1])
    ax2.set_yticks([0, 1])
    ax2.set_xticklabels(['Styl Correct', 'Styl Wrong'])
    ax2.set_yticklabels(['Emb Correct', 'Emb Wrong'])
    ax2.set_xlabel('Stylometry')
    ax2.set_ylabel('Embeddings')
    ax2.set_title('Agreement Matrix')

    for i in range(2):
        for j in range(2):
            ax2.text(j, i, str(conf[i, j]), ha='center', va='center', fontsize=14,
                     color='white' if conf[i, j] > conf.max()/2 else 'black')

    plt.colorbar(im, ax=ax2)
    plt.tight_layout()
    plt.savefig(output_dir / f'error_overlap_{language}.png', dpi=150)
    plt.close()

    return overlap


def create_summary_comparison(correlations_emb, correlations_styl, language, output_dir):
    """Create summary comparison of correlations."""
    features = [c['feature'] for c in correlations_emb]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(features))
    width = 0.35

    emb_corrs = [c['correlation'] for c in correlations_emb]
    styl_corrs = [c['correlation'] for c in correlations_styl]

    bars1 = ax.barh(x - width/2, emb_corrs, width, label='Embeddings', color='steelblue', alpha=0.7)
    bars2 = ax.barh(x + width/2, styl_corrs, width, label='Stylometry', color='coral', alpha=0.7)

    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_yticks(x)
    ax.set_yticklabels(features)
    ax.set_xlabel('Correlation with Correctness')
    ax.set_title(f'Feature Correlations: Embeddings vs Stylometry ({language.title()})')
    ax.legend()

    # Add significance markers
    for i, (ce, cs) in enumerate(zip(correlations_emb, correlations_styl)):
        if ce['significant']:
            ax.text(emb_corrs[i], i - width/2, '*', ha='left' if emb_corrs[i] >= 0 else 'right', va='center')
        if cs['significant']:
            ax.text(styl_corrs[i], i + width/2, '*', ha='left' if styl_corrs[i] >= 0 else 'right', va='center')

    plt.tight_layout()
    plt.savefig(output_dir / f'correlation_comparison_{language}.png', dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Compare error patterns: Embeddings vs Stylometry")
    parser.add_argument('--language', type=str, default='russian', choices=['russian', 'italian'])
    parser.add_argument('--embedding-model', type=str, default='gemini', choices=['openai', 'gemini'])
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    print("=" * 70)
    print(f"ERROR PATTERN COMPARISON: Embeddings vs Stylometry ({args.language.upper()})")
    print("=" * 70)

    # Load data
    print("\n[Loading dataset...]")
    data = load_dataset(
        language=args.language,
        embedding_model=args.embedding_model,
        n_per_author=N_PER_AUTHOR,
        seed=args.seed
    )

    embeddings = data['embeddings']
    labels = data['labels']
    texts = data['texts']
    func_words = data['func_words']

    print(f"  Samples: {len(texts)}")
    print(f"  Authors: {len(data['author_list'])}")

    # Extract features
    print("\n[Extracting surface features...]")
    surface_features = extract_surface_features(texts)

    print("\n[Extracting stylometry features...]")
    X_styl, _, _ = extract_stylometry_features(texts, func_words)
    print(f"  Stylometry features: {X_styl.shape[1]}")

    # Run classifications
    print("\n[Running embedding classification...]")
    emb_pred = run_classification_with_predictions(embeddings, labels, sparse=False)
    print(f"  Embedding accuracy: {emb_pred['accuracy']:.1%}")

    print("\n[Running stylometry classification...]")
    styl_pred = run_classification_with_predictions(X_styl, labels, sparse=True)
    print(f"  Stylometry accuracy: {styl_pred['accuracy']:.1%}")

    # Analyze correlations for both methods
    print("\n[Analyzing feature correlations...]")
    feature_names = ['char_count', 'word_count', 'line_count', 'ttr', 'punct_density']

    correlations_emb = []
    correlations_styl = []

    print(f"\n  {'Feature':<15} {'Emb r':>10} {'Styl r':>10} {'Diff':>10}")
    print("  " + "-" * 50)

    for fname in feature_names:
        corr_emb = analyze_feature_correlation(surface_features, emb_pred, fname)
        corr_styl = analyze_feature_correlation(surface_features, styl_pred, fname)
        correlations_emb.append(corr_emb)
        correlations_styl.append(corr_styl)

        diff = corr_emb['correlation'] - corr_styl['correlation']
        sig_emb = "*" if corr_emb['significant'] else ""
        sig_styl = "*" if corr_styl['significant'] else ""

        print(f"  {fname:<15} {corr_emb['correlation']:>+9.3f}{sig_emb} {corr_styl['correlation']:>+9.3f}{sig_styl} {diff:>+10.3f}")

    # Error overlap analysis
    print("\n[Analyzing error overlap...]")
    overlap = compute_error_overlap(emb_pred, styl_pred)

    print(f"\n  Error Agreement Analysis:")
    print(f"    Both correct:     {overlap['both_correct']:>5} ({overlap['pct_both_correct']:.1%})")
    print(f"    Both wrong:       {overlap['both_wrong']:>5} ({overlap['pct_both_wrong']:.1%})")
    print(f"    Only Emb wrong:   {overlap['only_emb_wrong']:>5} ({overlap['pct_only_emb_wrong']:.1%})")
    print(f"    Only Styl wrong:  {overlap['only_styl_wrong']:>5} ({overlap['pct_only_styl_wrong']:.1%})")
    print(f"\n    Jaccard similarity of errors: {overlap['jaccard_similarity']:.3f}")
    print(f"    Cohen's Kappa: {overlap['kappa']:.3f}")

    # Analyze "method-specific" errors
    print("\n[Analyzing method-specific errors...]")

    emb_errors = ~emb_pred['correct']
    styl_errors = ~styl_pred['correct']

    only_emb_mask = emb_errors & (~styl_errors)
    only_styl_mask = (~emb_errors) & styl_errors

    if np.sum(only_emb_mask) > 10:
        emb_specific = [f for f, m in zip(surface_features, only_emb_mask) if m]
        print(f"\n  Texts where ONLY Embeddings fail (n={len(emb_specific)}):")
        print(f"    Avg word count: {np.mean([f['word_count'] for f in emb_specific]):.1f}")
        print(f"    Avg TTR: {np.mean([f['ttr'] for f in emb_specific]):.3f}")

    if np.sum(only_styl_mask) > 10:
        styl_specific = [f for f, m in zip(surface_features, only_styl_mask) if m]
        print(f"\n  Texts where ONLY Stylometry fails (n={len(styl_specific)}):")
        print(f"    Avg word count: {np.mean([f['word_count'] for f in styl_specific]):.1f}")
        print(f"    Avg TTR: {np.mean([f['ttr'] for f in styl_specific]):.3f}")

    # Statistical comparison of method-specific error features
    if np.sum(only_emb_mask) > 10 and np.sum(only_styl_mask) > 10:
        print("\n  Feature comparison (Emb-only vs Styl-only errors):")
        for fname in ['word_count', 'ttr']:
            emb_vals = np.array([f[fname] for f, m in zip(surface_features, only_emb_mask) if m])
            styl_vals = np.array([f[fname] for f, m in zip(surface_features, only_styl_mask) if m])
            t_stat, t_pval = stats.ttest_ind(emb_vals, styl_vals)
            sig = "*" if t_pval < 0.05 else ""
            print(f"    {fname}: Emb-only={np.mean(emb_vals):.2f}, Styl-only={np.mean(styl_vals):.2f}, t={t_stat:.2f}, p={t_pval:.3f}{sig}")

    # Create visualizations
    print("\n[Creating visualizations...]")
    for fname in ['word_count', 'ttr', 'char_count']:
        create_comparison_plot(surface_features, emb_pred, styl_pred, fname, args.language, RESULTS_DIR)

    create_overlap_plot(emb_pred, styl_pred, args.language, RESULTS_DIR)
    create_summary_comparison(correlations_emb, correlations_styl, args.language, RESULTS_DIR)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: METHOD COMPARISON")
    print("=" * 70)

    print(f"\n  Accuracy: Embeddings={emb_pred['accuracy']:.1%}, Stylometry={styl_pred['accuracy']:.1%}")
    print(f"  Error overlap (Jaccard): {overlap['jaccard_similarity']:.3f}")

    # Key findings
    print("\n  Key findings:")

    # Length sensitivity
    len_corr_emb = next(c for c in correlations_emb if c['feature'] == 'word_count')
    len_corr_styl = next(c for c in correlations_styl if c['feature'] == 'word_count')
    if abs(len_corr_emb['correlation'] - len_corr_styl['correlation']) > 0.02:
        more_sensitive = "Embeddings" if abs(len_corr_emb['correlation']) > abs(len_corr_styl['correlation']) else "Stylometry"
        print(f"    - {more_sensitive} is more sensitive to text length")
    else:
        print(f"    - Both methods equally sensitive to text length")

    # TTR sensitivity
    ttr_corr_emb = next(c for c in correlations_emb if c['feature'] == 'ttr')
    ttr_corr_styl = next(c for c in correlations_styl if c['feature'] == 'ttr')
    if abs(ttr_corr_emb['correlation'] - ttr_corr_styl['correlation']) > 0.02:
        more_sensitive = "Embeddings" if abs(ttr_corr_emb['correlation']) > abs(ttr_corr_styl['correlation']) else "Stylometry"
        print(f"    - {more_sensitive} is more sensitive to vocabulary richness (TTR)")
    else:
        print(f"    - Both methods equally sensitive to vocabulary richness")

    # Error complementarity
    complementary_errors = overlap['only_emb_wrong'] + overlap['only_styl_wrong']
    shared_errors = overlap['both_wrong']
    if complementary_errors > shared_errors:
        print(f"    - Methods are COMPLEMENTARY: more unique errors ({complementary_errors}) than shared ({shared_errors})")
    else:
        print(f"    - Methods are REDUNDANT: more shared errors ({shared_errors}) than unique ({complementary_errors})")

    # Save results
    results = {
        'language': args.language,
        'embedding_model': args.embedding_model,
        'accuracies': {
            'embeddings': float(emb_pred['accuracy']),
            'stylometry': float(styl_pred['accuracy'])
        },
        'feature_correlations': {
            'embeddings': correlations_emb,
            'stylometry': correlations_styl
        },
        'error_overlap': overlap
    }

    results_file = RESULTS_DIR / f"error_comparison_{args.language}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_file}")
    print(f"Visualizations saved to: {RESULTS_DIR}/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
