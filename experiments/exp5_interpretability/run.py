#!/usr/bin/env python3
"""
EXPERIMENT 5: Probing Tasks for Embedding Interpretability

Purpose: What linguistic properties do embeddings encode linearly?

Probing Tasks:
1. Meter: Iamb (Я) vs Trochee (Х) - binary syllabic meters
2. Meter: Binary (Я/Х) vs Ternary (Ан/Аф/Д) - meter family classification
3. Meter: All 5 meters (Я, Х, Д, Ан, Аф) - multiclass
4. Clausula: Regular vs Free (вольная)
5. Feet: 4-foot vs 5-foot
6. Rhyme: Cross (ABAB) vs Paired (AABB)
7. Topic: Dominant LDA topic (Top 2 topics)

Note: Residual analysis is covered in EXP7 Waterfall.
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Constants
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# The published dataset is fixed at 200 poems per author (see data/README.md);
# every result in the paper is computed at this size.
N_PER_AUTHOR = 200



def load_data(n_per_author=200, seed=42, embedding_model='gemini'):
    """Load clean dataset with specified embeddings."""
    from clean_dataset import load_clean_dataset

    print(f"[Loading clean dataset with {embedding_model} embeddings...]")
    data = load_clean_dataset(
        n_per_author=n_per_author,
        seed=seed,
        include_poems=True,
        embedding_model=embedding_model
    )

    print(f"  Authors: {len(data['author_list'])}")
    print(f"  Samples: {len(data['labels'])}")
    print(f"  Embedding dim: {data['embeddings'].shape[1]}")

    return data


def run_probe(embeddings, labels, name, seed=42):
    """Run a single probing task with balanced classes."""
    # Get class indices
    classes = np.unique(labels)
    if len(classes) != 2:
        return None

    idx_0 = np.where(labels == classes[0])[0]
    idx_1 = np.where(labels == classes[1])[0]

    n_0, n_1 = len(idx_0), len(idx_1)

    # Balance classes
    np.random.seed(seed)
    n_per = min(n_0, n_1)

    if n_per < 100:
        print(f"  {name}: Skipped (insufficient samples: {n_per})")
        return None

    idx_0_balanced = np.random.choice(idx_0, n_per, replace=False)
    idx_1_balanced = np.random.choice(idx_1, n_per, replace=False)

    idx_balanced = np.concatenate([idx_0_balanced, idx_1_balanced])
    X = embeddings[idx_balanced]
    y = np.array([0] * n_per + [1] * n_per)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Run cross-validation
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring='accuracy')

    acc = float(np.mean(scores))
    std = float(np.std(scores))

    print(f"  {name}: {acc:.1%} ± {std:.1%} (n={2*n_per}, balanced)")

    return {
        'accuracy': acc,
        'accuracy_std': std,
        'n_samples': 2 * n_per,
        'n_class_0': n_0,
        'n_class_1': n_1,
        'class_labels': [str(classes[0]), str(classes[1])]
    }


def run_multiclass_probe(embeddings, labels, name, class_names, seed=42, min_per_class=50):
    """Run a multiclass probing task with balanced classes."""
    from collections import Counter

    classes = np.unique(labels)
    n_classes = len(classes)

    if n_classes < 2:
        print(f"  {name}: Skipped (only {n_classes} class)")
        return None

    # Count samples per class
    class_counts = Counter(labels)
    min_count = min(class_counts.values())

    if min_count < min_per_class:
        print(f"  {name}: Skipped (min class has only {min_count} samples)")
        return None

    # Balance by sampling min_count from each class
    np.random.seed(seed)
    balanced_indices = []

    for cls in classes:
        cls_indices = np.where(labels == cls)[0]
        sampled = np.random.choice(cls_indices, min_count, replace=False)
        balanced_indices.extend(sampled)

    balanced_indices = np.array(balanced_indices)
    X = embeddings[balanced_indices]
    y_raw = labels[balanced_indices]

    # Remap to 0, 1, 2, ...
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    y = np.array([class_to_idx[cls] for cls in y_raw])

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Run cross-validation with multiclass
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed, multi_class='multinomial')
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring='accuracy')

    acc = float(np.mean(scores))
    std = float(np.std(scores))
    chance = 1.0 / n_classes

    n_total = len(balanced_indices)
    print(f"  {name}: {acc:.1%} ± {std:.1%} (n={n_total}, {n_classes} classes, chance={chance:.1%})")

    return {
        'accuracy': acc,
        'accuracy_std': std,
        'n_samples': n_total,
        'n_classes': n_classes,
        'n_per_class': min_count,
        'chance_level': chance,
        'class_counts': {str(k): int(v) for k, v in class_counts.items()},
        'class_names': class_names
    }


def probe_meter(poems, embeddings, seed=42):
    """Probe: Iamb (Я) vs Trochee (Х)"""
    labels = []
    mask = []

    for poem in poems:
        prosody = poem.get('prosody', {})
        meter = prosody.get('meter', '') if prosody else ''

        if meter == 'Я':
            labels.append(0)
            mask.append(True)
        elif meter == 'Х':
            labels.append(1)
            mask.append(True)
        else:
            mask.append(False)

    mask = np.array(mask)
    if mask.sum() < 200:
        return None

    return run_probe(
        embeddings[mask],
        np.array(labels),
        "Meter (Iamb vs Trochee)",
        seed
    )


def probe_meter_binary_ternary(poems, embeddings, seed=42):
    """
    Probe: Binary-syllabic (Я/Х) vs Ternary-syllabic (Ан/Аф/Д) meters.

    Binary meters have 2-syllable feet: Iamb (uS), Trochee (Su)
    Ternary meters have 3-syllable feet: Anapest (uuS), Amphibrach (uSu), Dactyl (Suu)
    """
    binary_meters = {'Я', 'Х'}
    ternary_meters = {'Ан', 'Аф', 'Д'}

    labels = []
    mask = []

    for poem in poems:
        prosody = poem.get('prosody', {})
        meter = prosody.get('meter', '') if prosody else ''

        # Handle complex meters - take first component
        if '#' in meter:
            meter = meter.split('#')[0].strip()
        if ',' in meter:
            meter = meter.split(',')[0].strip()

        if meter in binary_meters:
            labels.append(0)  # Binary
            mask.append(True)
        elif meter in ternary_meters:
            labels.append(1)  # Ternary
            mask.append(True)
        else:
            mask.append(False)

    mask = np.array(mask)
    if mask.sum() < 200:
        return None

    result = run_probe(
        embeddings[mask],
        np.array(labels),
        "Meter (Binary vs Ternary)",
        seed
    )

    if result:
        result['class_labels'] = ['Binary (Я/Х)', 'Ternary (Ан/Аф/Д)']

    return result


def probe_meter_multiclass(poems, embeddings, seed=42):
    """
    Probe: All 5 main meter types (multiclass classification).

    Classes: Я (Iamb), Х (Trochee), Д (Dactyl), Ан (Anapest), Аф (Amphibrach)
    """
    main_meters = ['Я', 'Х', 'Д', 'Ан', 'Аф']
    meter_names = {
        'Я': 'Iamb',
        'Х': 'Trochee',
        'Д': 'Dactyl',
        'Ан': 'Anapest',
        'Аф': 'Amphibrach'
    }

    labels = []
    valid_indices = []

    for i, poem in enumerate(poems):
        prosody = poem.get('prosody', {})
        meter = prosody.get('meter', '') if prosody else ''

        # Handle complex meters - take first component
        if '#' in meter:
            meter = meter.split('#')[0].strip()
        if ',' in meter:
            meter = meter.split(',')[0].strip()

        if meter in main_meters:
            labels.append(meter)
            valid_indices.append(i)

    if len(valid_indices) < 500:
        print(f"  Meter (5-class): Skipped (insufficient samples: {len(valid_indices)})")
        return None

    valid_indices = np.array(valid_indices)
    labels = np.array(labels)

    return run_multiclass_probe(
        embeddings[valid_indices],
        labels,
        "Meter (5-class: Я/Х/Д/Ан/Аф)",
        class_names=meter_names,
        seed=seed,
        min_per_class=50
    )


def probe_clausula(poems, embeddings, seed=42):
    """Probe: Regular vs Free clausula"""
    labels = []
    mask = []

    for poem in poems:
        prosody = poem.get('prosody', {})
        clausula = prosody.get('clausula', '') if prosody else ''

        # Extract main clausula type (before colon)
        clausula_type = clausula.split(':')[0].strip().lower() if clausula else ''

        if 'регулярная' in clausula_type:
            labels.append(0)
            mask.append(True)
        elif 'вольная' in clausula_type:
            labels.append(1)
            mask.append(True)
        else:
            mask.append(False)

    mask = np.array(mask)
    if mask.sum() < 200:
        return None

    return run_probe(
        embeddings[mask],
        np.array(labels),
        "Clausula (Regular vs Free)",
        seed
    )


def probe_feet(poems, embeddings, seed=42):
    """Probe: 4-foot vs 5-foot meter"""
    labels = []
    mask = []

    for poem in poems:
        prosody = poem.get('prosody', {})
        feet = prosody.get('feet', '') if prosody else ''

        if feet == '4':
            labels.append(0)
            mask.append(True)
        elif feet == '5':
            labels.append(1)
            mask.append(True)
        else:
            mask.append(False)

    mask = np.array(mask)
    if mask.sum() < 200:
        return None

    return run_probe(
        embeddings[mask],
        np.array(labels),
        "Feet (4-foot vs 5-foot)",
        seed
    )


def probe_rhyme(poems, embeddings, seed=42):
    """Probe: Cross (ABAB) vs Paired (AABB) rhyme"""
    labels = []
    mask = []

    for poem in poems:
        prosody = poem.get('prosody', {})
        rhyme = prosody.get('rhyme', '') if prosody else ''

        # Extract main rhyme type (before colon)
        rhyme_type = rhyme.split(':')[0].strip().lower() if rhyme else ''

        if 'перекрестная' in rhyme_type:  # Cross rhyme ABAB
            labels.append(0)
            mask.append(True)
        elif 'парная' in rhyme_type:  # Paired rhyme AABB
            labels.append(1)
            mask.append(True)
        else:
            mask.append(False)

    mask = np.array(mask)
    if mask.sum() < 200:
        return None

    return run_probe(
        embeddings[mask],
        np.array(labels),
        "Rhyme (Cross vs Paired)",
        seed
    )


def probe_topic(poems, embeddings, seed=42, n_topics=20, linguistic_features=None):
    """
    Probe: Dominant LDA topic (Top 2 most frequent topics).

    Method:
    1. Use lemmas from linguistic annotations (Stanza)
    2. Fit LDA on TF-IDF of lemmatized poems (with stopwords removed)
    3. Assign each poem its dominant topic (argmax)
    4. Find the two most frequent topics
    5. Probe: Topic A vs Topic B
    """
    print("  [Fitting LDA for topic probing (using lemmas)...]")

    # Russian stopwords (common function words) - now as lemmas
    russian_stopwords = {
        # Pronouns (lemma forms)
        'я', 'ты', 'он', 'она', 'оно', 'мы', 'вы', 'они',
        'себя', 'свой', 'мой', 'твой', 'наш', 'ваш', 'его', 'её', 'их',
        'этот', 'тот', 'кто', 'что', 'какой', 'который', 'чей',
        'сам', 'весь', 'всякий', 'каждый', 'иной', 'другой',
        # Prepositions
        'в', 'во', 'на', 'за', 'из', 'к', 'ко', 'о', 'об', 'обо',
        'от', 'по', 'под', 'при', 'про', 'с', 'со', 'у', 'до', 'для',
        'без', 'между', 'через', 'над', 'перед', 'около', 'после',
        # Conjunctions
        'и', 'а', 'но', 'да', 'или', 'либо', 'ни', 'то', 'как',
        'что', 'чтобы', 'если', 'хотя', 'когда', 'пока', 'чем',
        'потому', 'поэтому', 'однако', 'зато', 'же', 'ли',
        # Particles
        'не', 'ни', 'бы', 'б', 'же', 'ж', 'ведь', 'уж',
        'вот', 'вон', 'лишь', 'только', 'даже', 'именно',
        'ещё', 'еще', 'уже', 'разве', 'неужели',
        # Common verbs (auxiliary/modal) - lemma forms
        'быть', 'стать', 'мочь', 'хотеть', 'знать', 'видеть',
        # Common adverbs
        'так', 'там', 'тут', 'здесь', 'где', 'куда', 'откуда',
        'когда', 'тогда', 'теперь', 'сейчас', 'потом', 'опять',
        'очень', 'совсем', 'почти', 'более', 'менее', 'столь',
        'как', 'почему', 'зачем', 'сколько',
        # Other common words
        'один', 'два', 'три', 'нет', 'ничто', 'никто',
        # Punctuation that might appear as lemmas
        ',', '.', '!', '?', ':', ';', '-', '—', '–', '...', '«', '»',
    }

    # Extract lemmas from linguistic features (join into text for TF-IDF)
    # linguistic_features can be passed as parameter or extracted from poems
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
                   and len(l) > 2
                   and l.isalpha()]
        lemma_texts.append(' '.join(filtered))

    # Fit TF-IDF on lemmatized texts
    tfidf = TfidfVectorizer(
        max_features=1000,
        min_df=5,
        max_df=0.8,
        ngram_range=(1, 1)
    )
    tfidf_matrix = tfidf.fit_transform(lemma_texts)

    # Fit LDA
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=seed,
        max_iter=20
    )
    topic_dist = lda.fit_transform(tfidf_matrix)

    # Get dominant topic per poem
    dominant_topics = np.argmax(topic_dist, axis=1)

    # Find two most frequent topics
    topic_counts = np.bincount(dominant_topics, minlength=n_topics)
    top_2_topics = np.argsort(topic_counts)[-2:][::-1]  # Descending order

    topic_a, topic_b = top_2_topics[0], top_2_topics[1]
    count_a, count_b = topic_counts[topic_a], topic_counts[topic_b]

    print(f"  Top topics: #{topic_a} (n={count_a}), #{topic_b} (n={count_b})")

    # Get top words for each topic (for interpretation)
    feature_names = tfidf.get_feature_names_out()
    top_words_a = [feature_names[i] for i in lda.components_[topic_a].argsort()[-5:][::-1]]
    top_words_b = [feature_names[i] for i in lda.components_[topic_b].argsort()[-5:][::-1]]

    print(f"  Topic {topic_a} words: {', '.join(top_words_a)}")
    print(f"  Topic {topic_b} words: {', '.join(top_words_b)}")

    # Create binary labels for these two topics
    mask = (dominant_topics == topic_a) | (dominant_topics == topic_b)
    labels = np.where(dominant_topics[mask] == topic_a, 0, 1)

    if mask.sum() < 200:
        print(f"  Topic probe: Skipped (insufficient samples: {mask.sum()})")
        return None

    result = run_probe(
        embeddings[mask],
        labels,
        f"Topic (#{topic_a} vs #{topic_b})",
        seed
    )

    if result:
        # Add topic metadata
        result['topic_a'] = int(topic_a)
        result['topic_b'] = int(topic_b)
        result['topic_a_words'] = top_words_a
        result['topic_b_words'] = top_words_b
        result['n_lda_topics'] = n_topics

    return result


def main():
    parser = argparse.ArgumentParser(description="EXP5: Probing Tasks")
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--embedding-model', type=str, default='gemini',
                        choices=['openai', 'gemini'],
                        help='Embedding model (default: gemini)')
    args = parser.parse_args()

    np.random.seed(args.seed)

    print("=" * 60)
    print("EXPERIMENT 5: Probing Tasks for Embedding Interpretability")
    print("=" * 60)

    # Load data
    data = load_data(n_per_author=N_PER_AUTHOR, seed=args.seed,
                     embedding_model=args.embedding_model)
    poems = data['poems']
    embeddings = data['embeddings']

    # Results storage
    results = {
        'timestamp': datetime.now().isoformat(),
        'settings': {
            'n_authors': len(data['author_list']),
            'n_per_author': N_PER_AUTHOR,
            'total_samples': len(poems),
            'embedding_model': args.embedding_model,
            'embedding_dim': embeddings.shape[1],
            'seed': args.seed
        },
        'probes': {}
    }

    # Run probing tasks
    print("\n[Running probing tasks...]")
    print("-" * 40)

    # 1a. Meter probe: Iamb vs Trochee (original)
    result = probe_meter(poems, embeddings, args.seed)
    if result:
        results['probes']['meter_iamb_vs_trochee'] = result

    # 1b. Meter probe: Binary vs Ternary (NEW)
    result = probe_meter_binary_ternary(poems, embeddings, args.seed)
    if result:
        results['probes']['meter_binary_vs_ternary'] = result

    # 1c. Meter probe: 5-class multiclass (NEW)
    result = probe_meter_multiclass(poems, embeddings, args.seed)
    if result:
        results['probes']['meter_5class'] = result

    # 2. Clausula probe
    result = probe_clausula(poems, embeddings, args.seed)
    if result:
        results['probes']['clausula_regular_vs_free'] = result

    # 3. Feet probe
    result = probe_feet(poems, embeddings, args.seed)
    if result:
        results['probes']['feet_4_vs_5'] = result

    # 4. Rhyme probe
    result = probe_rhyme(poems, embeddings, args.seed)
    if result:
        results['probes']['rhyme_cross_vs_paired'] = result

    # 5. Topic probe (semantic)
    linguistic_features = data.get('linguistic_features', None)
    result = probe_topic(poems, embeddings, args.seed, linguistic_features=linguistic_features)
    if result:
        results['probes']['topic_lda'] = result

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\n{'Probe':<35} {'Accuracy':<12} {'Chance':<10} {'Samples':<10}")
    print("-" * 67)

    for name, probe in results['probes'].items():
        acc = probe['accuracy']
        n = probe['n_samples']
        # Get chance level (50% for binary, 1/n for multiclass)
        chance = probe.get('chance_level', 0.5)
        print(f"{name:<35} {acc:>6.1%}       {chance:>6.1%}     {n:<10}")

    # Interpretation
    print("\n[Interpretation]")
    for name, probe in results['probes'].items():
        acc = probe['accuracy']
        chance = probe.get('chance_level', 0.5)
        lift = acc / chance if chance > 0 else 0

        # Effect size based on lift over chance
        if lift >= 1.5:
            level = "STRONG"
        elif lift >= 1.3:
            level = "MODERATE"
        elif lift >= 1.1:
            level = "WEAK"
        else:
            level = "NOT ENCODED"
        print(f"  {name}: {level} ({acc:.1%}, {lift:.1f}× chance)")

    # Save results
    output_file = RESULTS_DIR / "probing_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_file}")

    return results


if __name__ == "__main__":
    main()
