#!/usr/bin/env python3
"""
EXPERIMENT 7 EXTENDED (Italian): Five-Tier Residualization Waterfall with Stylometry

Purpose: Address the critical gap where char n-grams explain most of embedding
variance but were missing from the original Italian waterfall.

Tiers for Italian (no prosody available):
- Tier 1: Surface (20 features)
- Tier 2: Content/TF-IDF+LDA (40 features)
- Tier 3: Grammar (100 features) - Italian-specific morphology
- Tier 5: Character n-grams (2-4) - captures orthographic patterns [NEW]
- Tier 6: Word bigrams - captures collocational patterns [NEW]

Expected outcome: Residual should drop from ~23× chance to near chance level,
revealing that embeddings primarily encode character-level patterns.
"""

import sys
import json
import argparse
import warnings
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import r2_score

# Suppress warnings
warnings.filterwarnings('ignore')

# Constants
DATA_DIR = Path(__file__).parent.parent.parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Italian POS tags (Universal Dependencies)
POS_TAGS = ['NOUN', 'VERB', 'ADJ', 'ADV', 'PRON', 'DET', 'ADP', 'CCONJ',
            'SCONJ', 'PART', 'INTJ', 'NUM', 'PUNCT', 'SYM', 'X', 'PROPN', 'AUX']

# Italian morphological features
ITALIAN_MORPH_FEATURES = [
    'Definite=Def', 'Definite=Ind',
    'Gender=Masc', 'Gender=Fem', 'Gender=Neut',
    'Number=Sing', 'Number=Plur',
    'Case=Nom', 'Case=Gen', 'Case=Dat', 'Case=Acc', 'Case=Loc',
    'Degree=Pos', 'Degree=Cmp', 'Degree=Sup',
    'Mood=Ind', 'Mood=Sub', 'Mood=Imp', 'Mood=Cnd',
    'Person=1', 'Person=2', 'Person=3',
    'Tense=Past', 'Tense=Pres', 'Tense=Fut', 'Tense=Imp',
    'VerbForm=Fin', 'VerbForm=Inf', 'VerbForm=Part', 'VerbForm=Ger',
    'Voice=Act', 'Voice=Pass',
    'Polarity=Neg',
    'PronType=Prs', 'PronType=Dem', 'PronType=Rel', 'PronType=Int', 'PronType=Neg', 'PronType=Ind',
    'Poss=Yes',
    'Reflex=Yes',
    'NumType=Card', 'NumType=Ord',
    'Clitic=Yes',
]

# Italian dependency relations
ITALIAN_DEP_RELATIONS = [
    'nsubj', 'nsubj:pass', 'obj', 'iobj', 'obl', 'vocative', 'expl', 'dislocated',
    'csubj', 'ccomp', 'xcomp', 'advcl', 'advmod', 'discourse',
    'aux', 'cop', 'mark', 'nmod', 'appos', 'nummod', 'amod', 'det',
    'acl', 'acl:relcl', 'case', 'conj', 'cc', 'fixed', 'flat', 'parataxis',
    'orphan', 'root', 'punct', 'flat:name', 'compound'
]


def load_italian_dataset(n_per_author=200, seed=42, embedding_model='gemini'):
    """Load the published Italian dataset with the requested embeddings.

    Texts, linguistic features, labels and the author list all come from
    data/italian/ through the shared loader; see data/README.md. The
    n_per_author and seed arguments are kept for call-site compatibility --
    the published dataset is fixed at 200 poems per author, seed 42.
    """
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
    from data_loader import load_dataset

    # The published layout names this model 'qwen8b'; accept the older
    # 'qwen-8b' spelling used by earlier result files.
    model = {'qwen-8b': 'qwen8b'}.get(embedding_model, embedding_model)

    data = load_dataset('italian', embedding_model=model, include_poems=True)

    return {
        'embeddings': data['embeddings'],
        'labels': data['labels'],
        'author_list': data['author_list'],
        'poems': data['poems'],
        'n_per_author': data['metadata']['n_per_author'],
    }


def extract_tier1_surface(poem: dict) -> np.ndarray:
    """Extract Tier 1: Surface features (20 features)."""
    text = poem.get('text', '')
    lines = [l for l in text.strip().split('\n') if l.strip()]
    words = text.split()

    char_count = len(text)
    word_count = len(words)
    line_count = len(lines)

    word_lengths = [len(w) for w in words] if words else [0]
    avg_word_length = np.mean(word_lengths)
    std_word_length = np.std(word_lengths)

    line_lengths_chars = [len(l) for l in lines] if lines else [0]
    avg_line_length_chars = np.mean(line_lengths_chars)
    std_line_length_chars = np.std(line_lengths_chars)

    line_lengths_words = [len(l.split()) for l in lines] if lines else [0]
    avg_line_length_words = np.mean(line_lengths_words)

    n = max(word_count, 1)
    punct_count = sum(1 for c in text if c in '.,;:!?—…-–')
    punct_rate = punct_count / max(char_count, 1)

    exclaim_rate = text.count('!') / n
    question_rate = text.count('?') / n
    dash_rate = (text.count('—') + text.count('–')) / n
    ellipsis_rate = (text.count('…') + text.count('...')) / n
    comma_rate = text.count(',') / n
    semicolon_rate = text.count(';') / n
    colon_rate = text.count(':') / n

    unique_words = set(w.lower() for w in words)
    type_token_ratio = len(unique_words) / n if n > 0 else 0

    sentence_count = max(text.count('.') + text.count('!') + text.count('?'), 1)
    words_per_sentence = word_count / sentence_count

    newline_rate = line_count / n if n > 0 else 0

    return np.array([
        char_count, word_count, line_count,
        avg_word_length, std_word_length,
        avg_line_length_chars, std_line_length_chars, avg_line_length_words,
        punct_rate, exclaim_rate, question_rate, dash_rate, ellipsis_rate,
        comma_rate, semicolon_rate, colon_rate,
        type_token_ratio, sentence_count, words_per_sentence, newline_rate,
    ])


def extract_tier3_italian(poem: dict) -> np.ndarray:
    """Extract Tier 3: Italian Grammar features (100 features)."""
    lf = poem.get('linguistic_features', {})
    word_count = lf.get('word_count', 1) or 1

    pos_counts = lf.get('pos_counts', {})
    total_pos = sum(pos_counts.values()) or 1
    pos_features = [pos_counts.get(tag, 0) / total_pos for tag in POS_TAGS]

    feats = lf.get('feats', {})
    morph_features = []
    for morph_feat in ITALIAN_MORPH_FEATURES:
        if '=' in morph_feat:
            category, value = morph_feat.split('=', 1)
            if category in feats and value in feats[category]:
                morph_features.append(feats[category][value] / word_count)
            else:
                morph_features.append(0.0)
        else:
            morph_features.append(0.0)

    while len(morph_features) < 48:
        morph_features.append(0.0)

    deprels = lf.get('deprels', {})
    total_deps = sum(deprels.values()) or 1
    dep_features = [deprels.get(rel, 0) / total_deps for rel in ITALIAN_DEP_RELATIONS]

    while len(dep_features) < 35:
        dep_features.append(0.0)

    return np.array(pos_features + morph_features[:48] + dep_features[:35])


class Tier2Extractor:
    """Tier 2: Content/Topic feature extractor (40 features)."""

    def __init__(self, n_tfidf: int = 20, n_topics: int = 20, random_state: int = 42):
        self.n_tfidf = n_tfidf
        self.n_topics = n_topics
        self.random_state = random_state

        self.tfidf = TfidfVectorizer(
            max_features=1000,
            min_df=5,
            max_df=0.8,
            ngram_range=(1, 1)
        )
        self.lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=random_state,
            max_iter=20
        )
        self.is_fitted = False
        self.top_tfidf_idx = None

    def fit(self, poems):
        texts = [p.get('text', '') for p in poems]
        tfidf_matrix = self.tfidf.fit_transform(texts)
        tfidf_var = np.var(tfidf_matrix.toarray(), axis=0)
        self.top_tfidf_idx = np.argsort(tfidf_var)[-self.n_tfidf:]
        self.lda.fit(tfidf_matrix)
        self.is_fitted = True
        return self

    def transform(self, poems) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Extractor must be fitted first")
        texts = [p.get('text', '') for p in poems]
        tfidf_matrix = self.tfidf.transform(texts).toarray()
        tfidf_features = tfidf_matrix[:, self.top_tfidf_idx]
        lda_features = self.lda.transform(self.tfidf.transform(texts))
        return np.hstack([tfidf_features, lda_features])


class CharNgramExtractor:
    """Character n-gram feature extractor for Tier 5."""

    def __init__(self, n_range=(2, 4), max_features=2000):
        self.n_range = n_range
        self.max_features = max_features
        self.vectorizer = None

    def fit(self, poems):
        texts = [p['text'] for p in poems]
        self.vectorizer = TfidfVectorizer(
            analyzer='char',
            ngram_range=self.n_range,
            max_features=self.max_features,
            lowercase=True
        )
        self.vectorizer.fit(texts)
        return self

    def transform(self, poems):
        texts = [p['text'] for p in poems]
        return self.vectorizer.transform(texts).toarray()


class WordBigramExtractor:
    """Word bigram feature extractor for Tier 6."""

    def __init__(self, max_features=2000, include_unigrams=False):
        self.max_features = max_features
        self.include_unigrams = include_unigrams
        self.vectorizer = None

    def fit(self, poems):
        texts = [p['text'] for p in poems]
        n_range = (1, 2) if self.include_unigrams else (2, 2)
        self.vectorizer = TfidfVectorizer(
            analyzer='word',
            ngram_range=n_range,
            max_features=self.max_features,
            lowercase=True
        )
        self.vectorizer.fit(texts)
        return self

    def transform(self, poems):
        texts = [p['text'] for p in poems]
        return self.vectorizer.transform(texts).toarray()


def run_waterfall_cv(poems, embeddings, y, tier_order, n_authors, n_folds=5, seed=42,
                     char_ngram_features=2000, word_bigram_features=2000):
    """
    Run the extended waterfall residualization experiment with PROPER CV.
    """
    chance_level = 1 / n_authors

    # Tiers that need fitting within CV
    fitted_tiers = {'tier2', 'tier5', 'tier6'}

    # Pre-extract deterministic features
    det_tiers = [t for t in tier_order if t not in fitted_tiers]
    all_det_features = {}
    for tier in det_tiers:
        if tier == 'tier1':
            all_det_features[tier] = np.array([extract_tier1_surface(p) for p in poems])
        elif tier == 'tier3':
            all_det_features[tier] = np.array([extract_tier3_italian(p) for p in poems])

    # Initialize results storage
    stages = ['baseline'] + [f'after_{t}' for t in tier_order] + ['after_kernel']
    fold_scores = {stage: [] for stage in stages}
    fold_r2 = {stage: [] for stage in stages}  # per-tier R²
    fold_cumul_r2 = {stage: [] for stage in stages}  # cumulative R²
    tier_feature_counts = {t: [] for t in tier_order}

    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(embeddings, y)):
        print(f"    Fold {fold_idx + 1}/{n_folds}...")

        emb_train, emb_test = embeddings[train_idx], embeddings[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        train_poems = [poems[i] for i in train_idx]
        test_poems = [poems[i] for i in test_idx]

        # === BASELINE ===
        scaler = StandardScaler()
        emb_train_scaled = scaler.fit_transform(emb_train)
        emb_test_scaled = scaler.transform(emb_test)

        clf = LogisticRegression(max_iter=2000, solver='lbfgs')
        clf.fit(emb_train_scaled, y_train)
        fold_scores['baseline'].append(clf.score(emb_test_scaled, y_test))
        fold_r2['baseline'].append(0.0)
        fold_cumul_r2['baseline'].append(0.0)

        # Store original total variance for cumulative R² calculation
        # Sum of per-dimension variances (proper multivariate variance)
        original_total_var = np.sum(np.var(emb_train, axis=0))

        # === PROGRESSIVE RESIDUALIZATION ===
        current_emb_train = emb_train.copy()
        current_emb_test = emb_test.copy()
        cumul_feat_train = None
        cumul_feat_test = None

        # Fit extractors on TRAIN ONLY
        extractors = {}

        if 'tier2' in tier_order:
            extractors['tier2'] = Tier2Extractor(n_tfidf=20, n_topics=20)
            extractors['tier2'].fit(train_poems)

        if 'tier5' in tier_order:
            extractors['tier5'] = CharNgramExtractor(
                n_range=(2, 4), max_features=char_ngram_features)
            extractors['tier5'].fit(train_poems)

        if 'tier6' in tier_order:
            extractors['tier6'] = WordBigramExtractor(
                max_features=word_bigram_features, include_unigrams=False)
            extractors['tier6'].fit(train_poems)

        for tier in tier_order:
            if tier in fitted_tiers:
                if tier not in extractors:
                    continue
                tier_feat_train = extractors[tier].transform(train_poems)
                tier_feat_test = extractors[tier].transform(test_poems)
            else:
                tier_feat_train = all_det_features[tier][train_idx]
                tier_feat_test = all_det_features[tier][test_idx]

            tier_feature_counts[tier].append(tier_feat_train.shape[1])

            if cumul_feat_train is None:
                cumul_feat_train = tier_feat_train
                cumul_feat_test = tier_feat_test
            else:
                cumul_feat_train = np.hstack([cumul_feat_train, tier_feat_train])
                cumul_feat_test = np.hstack([cumul_feat_test, tier_feat_test])

            feat_scaler = StandardScaler()
            feat_train_scaled = feat_scaler.fit_transform(
                np.nan_to_num(cumul_feat_train, nan=0.0))
            feat_test_scaled = feat_scaler.transform(
                np.nan_to_num(cumul_feat_test, nan=0.0))

            ridge = Ridge(alpha=1.0)
            ridge.fit(feat_train_scaled, current_emb_train)

            pred_train = ridge.predict(feat_train_scaled)
            pred_test = ridge.predict(feat_test_scaled)

            resid_train = current_emb_train - pred_train
            resid_test = current_emb_test - pred_test

            r2 = r2_score(current_emb_train, pred_train, multioutput='variance_weighted')

            # Compute cumulative R² as fraction of original variance explained
            # Sum of per-dimension variances of residuals vs original
            resid_total_var = np.sum(np.var(resid_train, axis=0))
            cumul_r2 = 1.0 - (resid_total_var / original_total_var)

            resid_scaler = StandardScaler()
            resid_train_scaled = resid_scaler.fit_transform(resid_train)
            resid_test_scaled = resid_scaler.transform(resid_test)

            clf = LogisticRegression(max_iter=2000, solver='lbfgs')
            clf.fit(resid_train_scaled, y_train)
            score = clf.score(resid_test_scaled, y_test)

            fold_scores[f'after_{tier}'].append(score)
            fold_r2[f'after_{tier}'].append(r2)
            fold_cumul_r2[f'after_{tier}'].append(cumul_r2)

            current_emb_train = resid_train
            current_emb_test = resid_test

        # === KERNEL RESIDUALIZATION ===
        feat_scaler = StandardScaler()
        feat_train_scaled = feat_scaler.fit_transform(
            np.nan_to_num(cumul_feat_train, nan=0.0))
        feat_test_scaled = feat_scaler.transform(
            np.nan_to_num(cumul_feat_test, nan=0.0))

        n_components = min(500, feat_train_scaled.shape[0] // 2)
        kernel_pipe = Pipeline([
            ('nystroem', Nystroem(kernel='rbf', gamma=0.1,
                                  n_components=n_components, random_state=seed)),
            ('ridge', Ridge(alpha=1.0))
        ])

        kernel_pipe.fit(feat_train_scaled, current_emb_train)

        pred_train = kernel_pipe.predict(feat_train_scaled)
        pred_test = kernel_pipe.predict(feat_test_scaled)

        resid_train = current_emb_train - pred_train
        resid_test = current_emb_test - pred_test

        r2_kernel = r2_score(current_emb_train, pred_train, multioutput='variance_weighted')

        # Compute cumulative R² for kernel
        resid_total_var = np.sum(np.var(resid_train, axis=0))
        cumul_r2_kernel = 1.0 - (resid_total_var / original_total_var)

        resid_scaler = StandardScaler()
        resid_train_scaled = resid_scaler.fit_transform(resid_train)
        resid_test_scaled = resid_scaler.transform(resid_test)

        clf = LogisticRegression(max_iter=2000, solver='lbfgs')
        clf.fit(resid_train_scaled, y_train)
        score = clf.score(resid_test_scaled, y_test)

        fold_scores['after_kernel'].append(score)
        fold_r2['after_kernel'].append(r2_kernel)
        fold_cumul_r2['after_kernel'].append(cumul_r2_kernel)

    # Aggregate results
    results = {
        'n_authors': n_authors,
        'chance_level': chance_level,
        'tier_order': tier_order,
        'n_folds': n_folds,
        'stages': {},
        'fold_scores': {},
        'tier_feature_counts': {t: int(np.mean(counts)) for t, counts in tier_feature_counts.items() if counts}
    }

    for stage in stages:
        if fold_scores[stage]:
            scores = fold_scores[stage]
            mean_acc = float(np.mean(scores))
            std_acc = float(np.std(scores))
            lift = mean_acc / chance_level

            results['stages'][stage] = {
                'accuracy': mean_acc,
                'accuracy_std': std_acc,
                'lift_over_chance': lift,
                'r2': float(np.mean(fold_r2[stage])),
                'r2_std': float(np.std(fold_r2[stage])),
                'cumul_r2': float(np.mean(fold_cumul_r2[stage])),
                'cumul_r2_std': float(np.std(fold_cumul_r2[stage]))
            }
            results['fold_scores'][stage] = scores

    return results


def main():
    parser = argparse.ArgumentParser(description="EXP7 EXTENDED (Italian): Waterfall with Stylometry Tiers")
    parser.add_argument('--n-per-author', type=int, default=200)
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--embedding-model', type=str, default='gemini',
                        choices=['gemini', 'qwen8b', 'qwen-8b'],
                        help='Embedding model to use (default: gemini)')
    parser.add_argument('--char-ngram-features', type=int, default=2000)
    parser.add_argument('--word-bigram-features', type=int, default=2000)
    args = parser.parse_args()

    print("=" * 70)
    print("EXPERIMENT 7 EXTENDED (Italian): Five-Tier Waterfall with Stylometry")
    print("=" * 70)
    print("\nPurpose: Add char n-grams and word bigrams to measure true residual")
    print("Note: Italian has no prosody tier (Tier 4)")

    # Load Italian dataset
    print(f"\n[Loading Italian dataset with {args.embedding_model} embeddings...]")
    data = load_italian_dataset(n_per_author=args.n_per_author, seed=args.seed,
                                 embedding_model=args.embedding_model)

    poems = data['poems']
    embeddings = data['embeddings']
    labels = data['labels']
    unique_authors = data['author_list']

    n_samples = len(poems)
    n_authors = len(unique_authors)

    print(f"\n[Settings]")
    print(f"  Language: Italian")
    print(f"  Authors: {n_authors}")
    print(f"  Poems per author: {args.n_per_author}")
    print(f"  Total samples: {n_samples}")
    print(f"  CV folds: {args.n_folds}")
    print(f"  Embedding model: {args.embedding_model}")
    print(f"  Embedding dims: {embeddings.shape[1]}")
    print(f"  Chance level: {1/n_authors:.1%}")
    print(f"  Char n-gram features: {args.char_ngram_features}")
    print(f"  Word bigram features: {args.word_bigram_features}")

    results = {
        'timestamp': datetime.now().isoformat(),
        'version': 'extended_stylometry',
        'language': 'italian',
        'settings': {
            'n_authors': n_authors,
            'n_per_author': args.n_per_author,
            'total_samples': n_samples,
            'n_folds': args.n_folds,
            'seed': args.seed,
            'embedding_model': args.embedding_model,
            'embedding_dim': embeddings.shape[1],
            'char_ngram_features': args.char_ngram_features,
            'word_bigram_features': args.word_bigram_features
        },
        'experiments': {}
    }

    # ========================================================================
    # EXPERIMENT 1: Original 3-tier waterfall (for comparison)
    # ========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Original 3-Tier Waterfall (baseline comparison)")
    print("=" * 70)

    original_order = ['tier1', 'tier2', 'tier3']
    print(f"\n[Order]: {' -> '.join(original_order)}")

    results['experiments']['original_3tier'] = run_waterfall_cv(
        poems, embeddings, labels, original_order, n_authors,
        n_folds=args.n_folds, seed=args.seed)

    print("\n  Results:")
    for stage, data_stage in results['experiments']['original_3tier']['stages'].items():
        print(f"    {stage}: {data_stage['accuracy']:.1%} ± {data_stage['accuracy_std']:.1%} "
              f"(lift={data_stage['lift_over_chance']:.1f}×, R²={data_stage['r2']:.3f})")

    # ========================================================================
    # EXPERIMENT 2: Extended 5-tier waterfall (with stylometry)
    # ========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Extended 5-Tier Waterfall (with stylometry)")
    print("=" * 70)

    extended_order = ['tier1', 'tier2', 'tier3', 'tier5', 'tier6']
    print(f"\n[Order]: {' -> '.join(extended_order)}")
    print("  Tier 5 = Char n-grams (2-4)")
    print("  Tier 6 = Word bigrams")

    results['experiments']['extended_5tier'] = run_waterfall_cv(
        poems, embeddings, labels, extended_order, n_authors,
        n_folds=args.n_folds, seed=args.seed,
        char_ngram_features=args.char_ngram_features,
        word_bigram_features=args.word_bigram_features)

    print("\n  Results:")
    for stage, data_stage in results['experiments']['extended_5tier']['stages'].items():
        print(f"    {stage}: {data_stage['accuracy']:.1%} ± {data_stage['accuracy_std']:.1%} "
              f"(lift={data_stage['lift_over_chance']:.1f}×, R²={data_stage['r2']:.3f})")

    print(f"\n  Feature counts: {results['experiments']['extended_5tier']['tier_feature_counts']}")

    # ========================================================================
    # EXPERIMENT 3: Only stylometry tiers (char n-grams + word bigrams)
    # ========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Stylometry-Only Waterfall (char n-grams + word bigrams)")
    print("=" * 70)

    stylometry_order = ['tier5', 'tier6']
    print(f"\n[Order]: {' -> '.join(stylometry_order)}")

    results['experiments']['stylometry_only'] = run_waterfall_cv(
        poems, embeddings, labels, stylometry_order, n_authors,
        n_folds=args.n_folds, seed=args.seed,
        char_ngram_features=args.char_ngram_features,
        word_bigram_features=args.word_bigram_features)

    print("\n  Results:")
    for stage, data_stage in results['experiments']['stylometry_only']['stages'].items():
        print(f"    {stage}: {data_stage['accuracy']:.1%} ± {data_stage['accuracy_std']:.1%} "
              f"(lift={data_stage['lift_over_chance']:.1f}×, R²={data_stage['r2']:.3f})")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY: Comparison of Waterfall Experiments (Italian)")
    print("=" * 70)

    chance = 1 / n_authors
    print(f"\n  Chance level: {chance:.1%}")

    final_original = results['experiments']['original_3tier']['stages']['after_kernel']['accuracy']
    final_extended = results['experiments']['extended_5tier']['stages']['after_kernel']['accuracy']
    final_stylometry = results['experiments']['stylometry_only']['stages']['after_kernel']['accuracy']

    r2_original = results['experiments']['original_3tier']['stages']['after_kernel']['r2']
    r2_extended = results['experiments']['extended_5tier']['stages']['after_kernel']['r2']
    r2_stylometry = results['experiments']['stylometry_only']['stages']['after_kernel']['r2']

    print(f"\n  {'Experiment':<35} {'Final Acc':>10} {'Lift':>8} {'R²':>8}")
    print(f"  {'-' * 65}")
    print(f"  {'Original (3 tiers: S+C+G)':<35} {final_original:>9.1%} {final_original/chance:>7.1f}× {r2_original:>7.3f}")
    print(f"  {'Extended (5 tiers: +Char+Word)':<35} {final_extended:>9.1%} {final_extended/chance:>7.1f}× {r2_extended:>7.3f}")
    print(f"  {'Stylometry only (Char+Word)':<35} {final_stylometry:>9.1%} {final_stylometry/chance:>7.1f}× {r2_stylometry:>7.3f}")
    print(f"  {'-' * 65}")
    print(f"  {'Chance':<35} {chance:>9.1%}")

    # Key insight
    print(f"\n[Key Finding]")
    print(f"  Original residual:  {final_original:.1%} ({final_original/chance:.1f}× chance)")
    print(f"  Extended residual:  {final_extended:.1%} ({final_extended/chance:.1f}× chance)")
    print(f"  Reduction:          {final_original - final_extended:.1%} ({(final_original - final_extended)/final_original*100:.0f}% drop)")

    # Cumulative R² breakdown
    after_tier3_cumul = results['experiments']['extended_5tier']['stages']['after_tier3']['cumul_r2']
    after_tier5_cumul = results['experiments']['extended_5tier']['stages']['after_tier5']['cumul_r2']
    after_tier6_cumul = results['experiments']['extended_5tier']['stages']['after_tier6']['cumul_r2']
    after_kernel_cumul = results['experiments']['extended_5tier']['stages']['after_kernel']['cumul_r2']

    print(f"\n[Cumulative Variance Explained]")
    print(f"  After Tier 3 (Grammar):      R² = {after_tier3_cumul:.1%}")
    print(f"  After Tier 5 (Char n-grams): R² = {after_tier5_cumul:.1%} (+{after_tier5_cumul - after_tier3_cumul:.1%})")
    print(f"  After Tier 6 (Word bigrams): R² = {after_tier6_cumul:.1%} (+{after_tier6_cumul - after_tier5_cumul:.1%})")
    print(f"  After Kernel:                R² = {after_kernel_cumul:.1%} (+{after_kernel_cumul - after_tier6_cumul:.1%})")

    # Store comparison
    results['comparison'] = {
        'original_3tier': {
            'final_accuracy': final_original,
            'lift_over_chance': final_original / chance,
            'r2': r2_original
        },
        'extended_5tier': {
            'final_accuracy': final_extended,
            'lift_over_chance': final_extended / chance,
            'r2': r2_extended
        },
        'stylometry_only': {
            'final_accuracy': final_stylometry,
            'lift_over_chance': final_stylometry / chance,
            'r2': r2_stylometry
        },
        'reduction': {
            'absolute': final_original - final_extended,
            'relative': (final_original - final_extended) / final_original if final_original > 0 else 0
        }
    }

    # Save results
    results_file = RESULTS_DIR / f"waterfall_extended_italian_{args.embedding_model}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults saved to: {results_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
