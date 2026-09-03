#!/usr/bin/env python3
"""
Cross-Topic Validation: Do Embeddings Encode Topic or Style?

Purpose: Test whether embeddings generalize across topics (genuine stylistic signal)
or exploit topical cues (topic confound).

Method:
1. Assign each poem a dominant LDA topic
2. Split poems by topic: train on Topic A, test on Topic B
3. If accuracy drops sharply → embeddings encode topic, not style
4. If accuracy holds → genuine authorial signal

Reference: Manolache et al. found BERT exploits topical rather than stylistic cues.
"""

import sys
import json
import argparse
import warnings
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from clean_dataset import load_clean_dataset
from statistical_tests import bootstrap_ci

# Suppress warnings
warnings.filterwarnings('ignore')

# Constants
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# The published dataset is fixed at 200 poems per author (see data/README.md);
# every result in the paper is computed at this size.
N_PER_AUTHOR = 200



def fit_lda_topics(poems, n_topics=20, random_state=42, linguistic_features=None):
    """Fit LDA and return topic distributions for each poem.

    Uses lemmas from linguistic annotations with stopwords filtered.
    """
    # Russian stopwords (common function words) - as lemmas
    russian_stopwords = {
        # Pronouns (lemma forms)
        'я', 'ты', 'он', 'она', 'оно', 'мы', 'вы', 'они',
        'мой', 'твой', 'его', 'её', 'ее', 'наш', 'ваш', 'их',
        'этот', 'тот', 'такой', 'какой', 'который', 'чей',
        'кто', 'что', 'весь', 'всё', 'все', 'сам', 'самый',
        'себя', 'свой',
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
        # Common verbs (auxiliary/modal) - lemma forms
        'быть', 'мочь', 'хотеть', 'знать', 'стать', 'идти',
        # Adverbs
        'где', 'куда', 'откуда', 'там', 'тут', 'здесь',
        'очень', 'тоже', 'также', 'ещё', 'еще',
        'всегда', 'никогда', 'иногда', 'теперь', 'потом', 'опять',
        # Punctuation that might appear as tokens
        '.', ',', '!', '?', ':', ';', '-', '—', '«', '»', '"', "'",
    }

    # Extract lemmas from linguistic features
    lemma_texts = []
    for i, p in enumerate(poems):
        # Get linguistic features - from parameter if provided, else from poem
        if linguistic_features is not None and i < len(linguistic_features):
            ling = linguistic_features[i]
        else:
            ling = p.get('linguistic_features', {})
        lemmas = ling.get('lemmas', []) if ling else []
        # Filter stopwords and short tokens
        filtered = [l.lower() for l in lemmas
                   if l.lower() not in russian_stopwords
                   and len(l) >= 3
                   and l.isalpha()]
        lemma_texts.append(' '.join(filtered))

    # TF-IDF on lemmatized texts
    tfidf = TfidfVectorizer(
        max_features=1000,
        min_df=5,
        max_df=0.8,
        lowercase=True
    )
    tfidf_matrix = tfidf.fit_transform(lemma_texts)

    # LDA
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=random_state,
        max_iter=20
    )
    topic_distributions = lda.fit_transform(tfidf_matrix)

    # Get top words for each topic
    feature_names = tfidf.get_feature_names_out()
    topic_words = []
    for topic_idx, topic in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-6:-1]]
        topic_words.append(top_words)

    return topic_distributions, topic_words, lda, tfidf


def assign_dominant_topics(topic_distributions, threshold=0.3):
    """Assign dominant topic to each poem. Returns -1 if no clear dominant."""
    dominant_topics = []
    for dist in topic_distributions:
        max_prob = np.max(dist)
        if max_prob >= threshold:
            dominant_topics.append(np.argmax(dist))
        else:
            dominant_topics.append(-1)  # No clear dominant topic
    return np.array(dominant_topics)


def run_baseline_cv(embeddings, labels, n_folds=5, seed=42):
    """Run standard CV (random split) as baseline."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(embeddings)

    clf = LogisticRegression(max_iter=2000, solver='lbfgs')
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scores = cross_val_score(clf, X_scaled, labels, cv=cv, scoring='accuracy')

    return float(np.mean(scores)), float(np.std(scores)), list(scores)


def run_cross_topic_validation(embeddings, labels, dominant_topics, author_list,
                                min_samples_per_topic=10, seed=42):
    """
    Run cross-topic validation:
    - For each topic pair (A, B), train on A, test on B
    - Only include authors present in both topics
    """
    unique_topics = np.unique(dominant_topics)
    unique_topics = unique_topics[unique_topics >= 0]  # Exclude -1 (no dominant)

    results = []

    for train_topic in unique_topics:
        for test_topic in unique_topics:
            if train_topic == test_topic:
                continue

            # Get indices for train and test topics
            train_mask = dominant_topics == train_topic
            test_mask = dominant_topics == test_topic

            train_labels = labels[train_mask]
            test_labels = labels[test_mask]

            # Find authors present in both
            train_authors = set(train_labels)
            test_authors = set(test_labels)
            common_authors = train_authors & test_authors

            if len(common_authors) < 5:  # Need at least 5 authors
                continue

            # Filter to common authors
            train_keep = np.isin(train_labels, list(common_authors))
            test_keep = np.isin(test_labels, list(common_authors))

            X_train = embeddings[train_mask][train_keep]
            y_train = train_labels[train_keep]
            X_test = embeddings[test_mask][test_keep]
            y_test = test_labels[test_keep]

            # Check minimum samples
            if len(X_train) < min_samples_per_topic or len(X_test) < min_samples_per_topic:
                continue

            # Map labels to 0..n_authors-1
            author_to_idx = {a: i for i, a in enumerate(sorted(common_authors))}
            y_train_mapped = np.array([author_to_idx[a] for a in y_train])
            y_test_mapped = np.array([author_to_idx[a] for a in y_test])

            # Train and evaluate
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            clf = LogisticRegression(max_iter=2000, solver='lbfgs')
            clf.fit(X_train_scaled, y_train_mapped)
            accuracy = clf.score(X_test_scaled, y_test_mapped)

            chance = 1 / len(common_authors)

            results.append({
                'train_topic': int(train_topic),
                'test_topic': int(test_topic),
                'n_common_authors': len(common_authors),
                'n_train': len(X_train),
                'n_test': len(X_test),
                'accuracy': float(accuracy),
                'chance': float(chance),
                'lift': float(accuracy / chance)
            })

    return results


def run_leave_one_topic_out(embeddings, labels, dominant_topics, seed=42):
    """
    Leave-one-topic-out validation:
    - For each topic, train on ALL other topics, test on this topic
    - This tests generalization to unseen topic distributions
    """
    unique_topics = np.unique(dominant_topics)
    unique_topics = unique_topics[unique_topics >= 0]

    results = []

    for held_out_topic in unique_topics:
        # Split by topic
        test_mask = dominant_topics == held_out_topic
        train_mask = (dominant_topics >= 0) & (dominant_topics != held_out_topic)

        train_labels = labels[train_mask]
        test_labels = labels[test_mask]

        # Find common authors
        train_authors = set(train_labels)
        test_authors = set(test_labels)
        common_authors = train_authors & test_authors

        if len(common_authors) < 5:
            continue

        # Filter to common authors
        train_keep = np.isin(train_labels, list(common_authors))
        test_keep = np.isin(test_labels, list(common_authors))

        X_train = embeddings[train_mask][train_keep]
        y_train = train_labels[train_keep]
        X_test = embeddings[test_mask][test_keep]
        y_test = test_labels[test_keep]

        if len(X_test) < 20:  # Need enough test samples
            continue

        # Map labels
        author_to_idx = {a: i for i, a in enumerate(sorted(common_authors))}
        y_train_mapped = np.array([author_to_idx[a] for a in y_train])
        y_test_mapped = np.array([author_to_idx[a] for a in y_test])

        # Train and evaluate
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        clf = LogisticRegression(max_iter=2000, solver='lbfgs')
        clf.fit(X_train_scaled, y_train_mapped)
        accuracy = clf.score(X_test_scaled, y_test_mapped)

        chance = 1 / len(common_authors)

        results.append({
            'held_out_topic': int(held_out_topic),
            'n_authors': len(common_authors),
            'n_train': len(X_train),
            'n_test': len(X_test),
            'accuracy': float(accuracy),
            'chance': float(chance),
            'lift': float(accuracy / chance)
        })

    return results


def run_within_topic_cv(embeddings, labels, dominant_topics, n_folds=5, seed=42):
    """
    Within-topic CV: For each major topic, run CV only on poems from that topic.
    This tests: can we distinguish authors WITHIN a topic?
    """
    unique_topics = np.unique(dominant_topics)
    unique_topics = unique_topics[unique_topics >= 0]

    results = []

    for topic in unique_topics:
        topic_mask = dominant_topics == topic
        topic_labels = labels[topic_mask]
        topic_embeddings = embeddings[topic_mask]

        # Need enough samples and authors
        unique_authors = np.unique(topic_labels)
        if len(unique_authors) < 5 or len(topic_embeddings) < 50:
            continue

        # Check that each author has enough samples
        author_counts = defaultdict(int)
        for l in topic_labels:
            author_counts[l] += 1

        # Filter to authors with at least 5 samples in this topic
        valid_authors = [a for a, c in author_counts.items() if c >= 5]
        if len(valid_authors) < 5:
            continue

        valid_mask = np.isin(topic_labels, valid_authors)
        X = topic_embeddings[valid_mask]
        y = topic_labels[valid_mask]

        # Map labels
        author_to_idx = {a: i for i, a in enumerate(sorted(set(y)))}
        y_mapped = np.array([author_to_idx[a] for a in y])

        # Run CV
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        clf = LogisticRegression(max_iter=2000, solver='lbfgs')
        cv = StratifiedKFold(n_splits=min(n_folds, min(np.bincount(y_mapped))),
                            shuffle=True, random_state=seed)

        try:
            scores = cross_val_score(clf, X_scaled, y_mapped, cv=cv, scoring='accuracy')

            chance = 1 / len(set(y))

            results.append({
                'topic': int(topic),
                'n_authors': len(set(y)),
                'n_samples': len(X),
                'accuracy': float(np.mean(scores)),
                'accuracy_std': float(np.std(scores)),
                'chance': float(chance),
                'lift': float(np.mean(scores) / chance)
            })
        except Exception as e:
            print(f"  Skipping topic {topic}: {e}")
            continue

    return results


def main():
    parser = argparse.ArgumentParser(description="Cross-Topic Validation")
    parser.add_argument('--n-topics', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--embedding-model', type=str, default='gemini',
                        choices=['openai', 'gemini'])
    parser.add_argument('--topic-threshold', type=float, default=0.25,
                        help='Min probability to assign dominant topic')
    args = parser.parse_args()

    print("=" * 70)
    print("CROSS-TOPIC VALIDATION: Do Embeddings Encode Topic or Style?")
    print("=" * 70)
    print("\nIf embeddings exploit topic cues (Manolache et al. concern):")
    print("  → Cross-topic accuracy should drop sharply")
    print("If embeddings capture genuine authorial style:")
    print("  → Cross-topic accuracy should remain high")

    # Load data
    print(f"\n[Loading dataset with {args.embedding_model} embeddings...]")
    data = load_clean_dataset(
        n_per_author=N_PER_AUTHOR,
        seed=args.seed,
        include_poems=True,
        embedding_model=args.embedding_model
    )

    poems = data['poems']
    embeddings = data['embeddings']
    labels = data['labels']
    author_list = data['author_list']
    linguistic_features = data.get('linguistic_features', None)

    n_samples = len(poems)
    n_authors = len(author_list)
    chance = 1 / n_authors

    print(f"  Authors: {n_authors}")
    print(f"  Samples: {n_samples}")
    print(f"  Chance level: {chance:.1%}")

    # Fit LDA
    print(f"\n[Fitting LDA with {args.n_topics} topics...]")
    topic_dists, topic_words, lda, tfidf = fit_lda_topics(
        poems, n_topics=args.n_topics, random_state=args.seed,
        linguistic_features=linguistic_features)

    # Show topic words
    print("\n  Top topics:")
    for i, words in enumerate(topic_words[:5]):
        print(f"    Topic {i}: {', '.join(words)}")

    # Assign dominant topics
    dominant_topics = assign_dominant_topics(topic_dists, threshold=args.topic_threshold)
    n_assigned = np.sum(dominant_topics >= 0)
    print(f"\n  Poems with clear dominant topic: {n_assigned}/{n_samples} ({n_assigned/n_samples:.1%})")

    # Topic distribution
    topic_counts = defaultdict(int)
    for t in dominant_topics:
        if t >= 0:
            topic_counts[t] += 1

    print("\n  Top 5 topics by poem count:")
    for topic, count in sorted(topic_counts.items(), key=lambda x: -x[1])[:5]:
        print(f"    Topic {topic}: {count} poems ({', '.join(topic_words[topic][:3])})")

    results = {
        'timestamp': datetime.now().isoformat(),
        'settings': {
            'n_authors': n_authors,
            'n_samples': n_samples,
            'n_topics': args.n_topics,
            'embedding_model': args.embedding_model,
            'topic_threshold': args.topic_threshold,
            'n_poems_with_topic': int(n_assigned)
        },
        'topic_words': {i: words for i, words in enumerate(topic_words)},
        'experiments': {}
    }

    # ========================================================================
    # EXPERIMENT 1: Baseline (random CV)
    # ========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Baseline (Standard 5-fold CV)")
    print("=" * 70)

    baseline_acc, baseline_std, baseline_scores = run_baseline_cv(
        embeddings, labels, n_folds=5, seed=args.seed)

    print(f"\n  Accuracy: {baseline_acc:.1%} ± {baseline_std:.1%}")
    print(f"  Lift over chance: {baseline_acc/chance:.1f}×")

    results['experiments']['baseline'] = {
        'accuracy': baseline_acc,
        'accuracy_std': baseline_std,
        'fold_scores': baseline_scores,
        'chance': chance,
        'lift': baseline_acc / chance
    }

    # ========================================================================
    # EXPERIMENT 2: Leave-One-Topic-Out
    # ========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Leave-One-Topic-Out Validation")
    print("=" * 70)
    print("\nTrain on all topics except one, test on held-out topic")

    loto_results = run_leave_one_topic_out(
        embeddings, labels, dominant_topics, seed=args.seed)

    if loto_results:
        loto_accs = [r['accuracy'] for r in loto_results]
        loto_mean = np.mean(loto_accs)
        loto_std = np.std(loto_accs)

        print(f"\n  Mean accuracy: {loto_mean:.1%} ± {loto_std:.1%}")
        print(f"  Baseline: {baseline_acc:.1%}")
        print(f"  Drop: {baseline_acc - loto_mean:.1%}")

        print("\n  Per-topic results:")
        for r in sorted(loto_results, key=lambda x: -x['accuracy'])[:5]:
            topic_id = r['held_out_topic']
            topic_name = ', '.join(topic_words[topic_id][:3])
            print(f"    Topic {topic_id} ({topic_name}): {r['accuracy']:.1%} "
                  f"({r['n_test']} test, {r['n_authors']} authors)")

        results['experiments']['leave_one_topic_out'] = {
            'results': loto_results,
            'mean_accuracy': loto_mean,
            'std_accuracy': loto_std,
            'drop_from_baseline': baseline_acc - loto_mean
        }
    else:
        print("\n  Not enough data for leave-one-topic-out validation")

    # ========================================================================
    # EXPERIMENT 3: Within-Topic CV
    # ========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Within-Topic Classification")
    print("=" * 70)
    print("\nCan we distinguish authors WITHIN a single topic?")
    print("(Topic is held constant, so only style differs)")

    within_results = run_within_topic_cv(
        embeddings, labels, dominant_topics, n_folds=5, seed=args.seed)

    if within_results:
        within_accs = [r['accuracy'] for r in within_results]
        within_mean = np.mean(within_accs)
        within_std = np.std(within_accs)

        # Weight by number of samples
        weighted_acc = sum(r['accuracy'] * r['n_samples'] for r in within_results)
        weighted_acc /= sum(r['n_samples'] for r in within_results)

        print(f"\n  Mean accuracy (across topics): {within_mean:.1%} ± {within_std:.1%}")
        print(f"  Weighted mean accuracy: {weighted_acc:.1%}")
        print(f"  Baseline (all topics): {baseline_acc:.1%}")

        print("\n  Per-topic results:")
        for r in sorted(within_results, key=lambda x: -x['n_samples'])[:5]:
            topic_id = r['topic']
            topic_name = ', '.join(topic_words[topic_id][:3])
            print(f"    Topic {topic_id} ({topic_name}): {r['accuracy']:.1%} "
                  f"({r['n_samples']} samples, {r['n_authors']} authors, {r['lift']:.1f}× chance)")

        results['experiments']['within_topic'] = {
            'results': within_results,
            'mean_accuracy': within_mean,
            'std_accuracy': within_std,
            'weighted_accuracy': weighted_acc
        }
    else:
        print("\n  Not enough data for within-topic validation")

    # ========================================================================
    # EXPERIMENT 4: Cross-Topic Pairs
    # ========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Cross-Topic Pairs (Train A → Test B)")
    print("=" * 70)

    cross_results = run_cross_topic_validation(
        embeddings, labels, dominant_topics, author_list, seed=args.seed)

    if cross_results:
        cross_accs = [r['accuracy'] for r in cross_results]
        cross_mean = np.mean(cross_accs)
        cross_std = np.std(cross_accs)

        print(f"\n  Mean accuracy: {cross_mean:.1%} ± {cross_std:.1%}")
        print(f"  Baseline: {baseline_acc:.1%}")
        print(f"  Drop: {baseline_acc - cross_mean:.1%}")

        # Best and worst pairs
        sorted_pairs = sorted(cross_results, key=lambda x: x['accuracy'])

        print("\n  Best topic pairs (highest transfer):")
        for r in sorted_pairs[-3:]:
            train_name = ', '.join(topic_words[r['train_topic']][:2])
            test_name = ', '.join(topic_words[r['test_topic']][:2])
            print(f"    {r['train_topic']}→{r['test_topic']}: {r['accuracy']:.1%} "
                  f"({train_name} → {test_name})")

        print("\n  Worst topic pairs (lowest transfer):")
        for r in sorted_pairs[:3]:
            train_name = ', '.join(topic_words[r['train_topic']][:2])
            test_name = ', '.join(topic_words[r['test_topic']][:2])
            print(f"    {r['train_topic']}→{r['test_topic']}: {r['accuracy']:.1%} "
                  f"({train_name} → {test_name})")

        results['experiments']['cross_topic_pairs'] = {
            'results': cross_results,
            'mean_accuracy': cross_mean,
            'std_accuracy': cross_std,
            'drop_from_baseline': baseline_acc - cross_mean
        }
    else:
        print("\n  Not enough data for cross-topic validation")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY: Topic vs Style")
    print("=" * 70)

    print(f"\n  {'Experiment':<35} {'Accuracy':>10} {'vs Baseline':>12}")
    print(f"  {'-' * 60}")
    print(f"  {'Baseline (random CV)':<35} {baseline_acc:>9.1%} {'—':>12}")

    if 'leave_one_topic_out' in results['experiments']:
        loto = results['experiments']['leave_one_topic_out']
        drop = loto['drop_from_baseline']
        print(f"  {'Leave-one-topic-out':<35} {loto['mean_accuracy']:>9.1%} {-drop:>+11.1%}")

    if 'within_topic' in results['experiments']:
        within = results['experiments']['within_topic']
        diff = within['weighted_accuracy'] - baseline_acc
        print(f"  {'Within-topic (weighted)':<35} {within['weighted_accuracy']:>9.1%} {diff:>+11.1%}")

    if 'cross_topic_pairs' in results['experiments']:
        cross = results['experiments']['cross_topic_pairs']
        drop = cross['drop_from_baseline']
        print(f"  {'Cross-topic pairs (mean)':<35} {cross['mean_accuracy']:>9.1%} {-drop:>+11.1%}")

    # Interpretation
    print("\n[Interpretation]")

    if 'cross_topic_pairs' in results['experiments']:
        drop = results['experiments']['cross_topic_pairs']['drop_from_baseline']
        cross_acc = results['experiments']['cross_topic_pairs']['mean_accuracy']

        if drop < 0.05:
            print(f"  ✓ Cross-topic drop is small ({drop:.1%})")
            print(f"  ✓ Embeddings generalize across topics")
            print(f"  → GENUINE STYLISTIC SIGNAL (not topic confound)")
        elif drop < 0.15:
            print(f"  ~ Moderate cross-topic drop ({drop:.1%})")
            print(f"  ~ Embeddings partially exploit topic cues")
            print(f"  → MIXED: Some style, some topic")
        else:
            print(f"  ✗ Large cross-topic drop ({drop:.1%})")
            print(f"  ✗ Embeddings heavily exploit topic cues")
            print(f"  → TOPIC CONFOUND (Manolache et al. concern applies)")

        # Check if still above chance
        if cross_acc > chance * 2:
            print(f"\n  Even cross-topic accuracy ({cross_acc:.1%}) is {cross_acc/chance:.1f}× chance")
            print(f"  → Some genuine authorial signal remains")

    # Save results
    output_file = RESULTS_DIR / f"cross_topic_validation_{args.embedding_model}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults saved to: {output_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
