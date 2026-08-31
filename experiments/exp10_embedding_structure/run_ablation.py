#!/usr/bin/env python3
"""
Text Ablation Study

Purpose: Determine which text aspects embeddings are most sensitive to.
Modify poems systematically, re-embed, and measure distance from original.

Ablation types:
1. remove_punctuation: Strip all punctuation
2. shuffle_words: Random word order within each line
3. shuffle_lines: Random line order
4. lowercase: Convert to lowercase
5. mask_content: Replace nouns/verbs/adjectives with [MASK]

Method:
1. Sample 290 poems (10 per author × 29 authors)
2. Apply each ablation type
3. Re-embed modified texts via Gemini API
4. Compute cosine distance from original embedding
5. Report which ablation causes largest shift

Usage:
    python run_ablation.py               # default: reproduce offline from the
                                         # published per-poem distances in
                                         # results/ablation_results.json.
                                         # No API key, no cost.
    python run_ablation.py --regenerate  # redo steps 3-4 through the Gemini API.
                                         # Needs GEMINI_API_KEY and COSTS MONEY.
    python run_ablation.py --dry-run     # preview perturbations and call count.

The perturbed texts are generated here and so are absent from the Zenodo release,
which means their embeddings cannot be distributed. The resulting per-poem cosine
distances are distributed instead, and every published number derives from those.
"""

import os
import sys
import json
import time
import re
import argparse
import warnings
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from dotenv import load_dotenv

from sklearn.metrics.pairwise import cosine_similarity

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from clean_dataset import load_clean_dataset

# Suppress warnings
warnings.filterwarnings('ignore')

# Load environment variables
ENV_FILE = Path(__file__).parent.parent.parent / ".env"
load_dotenv(ENV_FILE)

# Constants
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = "gemini-embedding-001"
EMBEDDING_DIM = 3072
BATCH_SIZE = 50
RATE_LIMIT_DELAY = 1.0  # seconds between batches

# Russian function words (stopwords) - preserved during content masking
RUSSIAN_STOPWORDS = {
    'я', 'ты', 'он', 'она', 'оно', 'мы', 'вы', 'они',
    'мой', 'твой', 'его', 'её', 'наш', 'ваш', 'их',
    'этот', 'тот', 'такой', 'какой', 'который',
    'весь', 'всё', 'все', 'сам', 'самый',
    'и', 'а', 'но', 'да', 'или', 'что', 'как', 'где', 'когда',
    'если', 'чтобы', 'потому', 'хотя', 'ведь', 'же', 'ли', 'бы',
    'в', 'на', 'с', 'к', 'о', 'у', 'за', 'из', 'по', 'от', 'до',
    'для', 'при', 'без', 'над', 'под', 'про', 'через', 'между',
    'не', 'ни', 'уже', 'ещё', 'только', 'даже', 'вот', 'вон',
    'очень', 'так', 'там', 'тут', 'здесь', 'сюда', 'туда',
    'быть', 'есть', 'был', 'была', 'было', 'были', 'будет',
    'можно', 'нужно', 'надо', 'нет', 'него', 'неё', 'ним', 'ней'
}


def get_gemini_embeddings(texts, task_type="CLASSIFICATION"):
    """Get embeddings for a list of texts using Gemini API."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=GEMINI_API_KEY)

    result = client.models.embed_content(
        model=MODEL,
        contents=texts,
        config=types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=EMBEDDING_DIM
        )
    )

    return [emb.values for emb in result.embeddings]


def get_embeddings_batched(texts, batch_size=BATCH_SIZE):
    """Get embeddings in batches with rate limiting."""
    all_embeddings = []
    n_batches = (len(texts) + batch_size - 1) // batch_size

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_num = i // batch_size + 1

        print(f"    Batch {batch_num}/{n_batches}...", end=" ", flush=True)
        embeddings = get_gemini_embeddings(batch)
        all_embeddings.extend(embeddings)
        print("done")

        if i + batch_size < len(texts):
            time.sleep(RATE_LIMIT_DELAY)

    return np.array(all_embeddings)


# Ablation functions

def ablate_remove_punctuation(text):
    """Remove all punctuation from text."""
    # Keep only letters, numbers, and whitespace
    cleaned = re.sub(r'[^\w\s]', '', text)
    # Normalize whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def ablate_shuffle_words(text, seed=None):
    """Shuffle word order within each line."""
    if seed is not None:
        np.random.seed(seed)

    lines = text.split('\n')
    shuffled_lines = []

    for line in lines:
        words = line.split()
        if len(words) > 1:
            np.random.shuffle(words)
        shuffled_lines.append(' '.join(words))

    return '\n'.join(shuffled_lines)


def ablate_shuffle_lines(text, seed=None):
    """Shuffle line order."""
    if seed is not None:
        np.random.seed(seed)

    lines = text.split('\n')
    lines = [l for l in lines if l.strip()]  # Remove empty lines
    np.random.shuffle(lines)
    return '\n'.join(lines)


def ablate_lowercase(text):
    """Convert to lowercase."""
    return text.lower()


def ablate_mask_content(poem, mask_token="[MASK]"):
    """Replace content words with mask token.

    Uses lemmas from linguistic_features to identify content words.
    Function words (from RUSSIAN_STOPWORDS) are preserved.
    """
    text = poem['text']
    ling = poem.get('linguistic_features', {})

    if not ling:
        # Fallback: mask all long words (likely content words)
        words = text.split()
        masked = [mask_token if len(w.strip('.,!?;:-—""«»()')) > 4 else w for w in words]
        return ' '.join(masked)

    lemmas = ling.get('lemmas', [])
    if not lemmas:
        return text

    # Get content lemmas (not in stopwords, long enough)
    content_lemmas = set()
    for lemma in lemmas:
        lemma_lower = lemma.lower()
        if (lemma_lower not in RUSSIAN_STOPWORDS and
            len(lemma) >= 3 and
            lemma.isalpha()):
            content_lemmas.add(lemma_lower)

    # Mask words whose lowercase form starts with any content lemma
    result_words = []
    for word in text.split():
        # Clean word for matching
        clean = word.lower().strip('.,!?;:-—""«»()')

        # Check if word matches any content lemma
        is_content = any(clean.startswith(lem[:4]) for lem in content_lemmas if len(lem) >= 4)

        if is_content and len(clean) >= 4:
            # Preserve punctuation around the word
            prefix = ''
            suffix = ''
            for c in word:
                if c.isalpha():
                    break
                prefix += c
            for c in reversed(word):
                if c.isalpha():
                    break
                suffix = c + suffix
            result_words.append(prefix + mask_token + suffix)
        else:
            result_words.append(word)

    return ' '.join(result_words)


def ablate_function_words_only(poem):
    """Keep only function words, remove content words."""
    ling = poem.get('linguistic_features', {})
    if not ling:
        return poem['text']

    pos_tags = ling.get('pos_tags', [])
    tokens = ling.get('tokens', [])

    if not pos_tags or not tokens:
        return poem['text']

    # Function word POS tags
    function_pos = {'ADP', 'AUX', 'CCONJ', 'DET', 'PART', 'PRON', 'SCONJ', 'PUNCT'}

    # Keep only function words
    function_tokens = [t for t, p in zip(tokens, pos_tags) if p in function_pos]

    return ' '.join(function_tokens)


def sample_poems_stratified(poems, embeddings, labels, n_per_author=10, seed=42):
    """Sample n poems per author for ablation study."""
    np.random.seed(seed)

    sampled_poems = []
    sampled_embeddings = []
    sampled_labels = []
    sampled_indices = []

    # Group by author
    author_poems = defaultdict(list)
    for i, (poem, emb, label) in enumerate(zip(poems, embeddings, labels)):
        author_poems[label].append((i, poem, emb))

    # Sample from each author
    for label in sorted(author_poems.keys()):
        items = author_poems[label]
        if len(items) >= n_per_author:
            selected = np.random.choice(len(items), n_per_author, replace=False)
        else:
            selected = range(len(items))

        for idx in selected:
            i, poem, emb = items[idx]
            sampled_indices.append(i)
            sampled_poems.append(poem)
            sampled_embeddings.append(emb)
            sampled_labels.append(label)

    return (sampled_poems, np.array(sampled_embeddings),
            np.array(sampled_labels), sampled_indices)


def compute_cosine_distances(original_embeddings, modified_embeddings):
    """Compute cosine distance between original and modified embeddings."""
    # Cosine similarity
    sims = np.array([
        cosine_similarity([orig], [mod])[0, 0]
        for orig, mod in zip(original_embeddings, modified_embeddings)
    ])
    # Convert to distance
    distances = 1 - sims
    return distances


def report_ablation_results(results, settings, save=False):
    """Print the summary, redraw the figure, and optionally re-save results.

    Takes the per-ablation distance statistics, so it works identically whether
    they were just computed via the API or loaded from the published
    results/ablation_results.json.
    """
    print("\n" + "=" * 70)
    print("SUMMARY: Embedding Sensitivity to Text Modifications")
    print("=" * 70)

    sorted_results = sorted(results.items(),
                            key=lambda x: x[1]['mean_distance'], reverse=True)

    print(f"\nRanked by embedding shift (cosine distance):")
    for i, (name, res) in enumerate(sorted_results, 1):
        print(f"  {i}. {name}: {res['mean_distance']:.4f} ± {res['std_distance']:.4f}")

    most_sensitive = sorted_results[0][0]
    least_sensitive = sorted_results[-1][0]

    print(f"\n  Most sensitive to: {most_sensitive}")
    print(f"  Least sensitive to: {least_sensitive}")

    print("\n[Creating visualization...]")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    names = [name for name, _ in sorted_results]
    means = [res['mean_distance'] for _, res in sorted_results]
    stds = [res['std_distance'] for _, res in sorted_results]

    ax.barh(names, means, xerr=stds, color='#4C72B0', capsize=5)
    ax.set_xlabel('Cosine Distance from Original')
    ax.set_title('Embedding Sensitivity to Text Modifications')
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'ablation_sensitivity.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {RESULTS_DIR}/ablation_sensitivity.png")

    if save:
        output_results = {
            'timestamp': datetime.now().isoformat(),
            'settings': settings,
            'results': results,
            'ranking': [name for name, _ in sorted_results],
            'interpretation': {
                'most_sensitive': most_sensitive,
                'least_sensitive': least_sensitive
            }
        }
        output_file = RESULTS_DIR / 'ablation_results.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {output_file}")

    print("\n" + "=" * 70)
    print("ABLATION STUDY COMPLETE")
    print("=" * 70)

    return 0


def main():
    parser = argparse.ArgumentParser(description="Text Ablation Study")
    parser.add_argument('--n-per-author', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--embedding-model', type=str, default='gemini',
                        choices=['gemini'])  # Only Gemini supported for re-embedding
    parser.add_argument('--sample-per-author', type=int, default=10,
                        help='Number of poems to sample per author for ablation')
    parser.add_argument('--regenerate', action='store_true',
                        help='Re-embed the perturbed texts through the Gemini API '
                             'instead of using the published distances. Requires '
                             'GEMINI_API_KEY and COSTS MONEY (290 x 5 = 1450 calls). '
                             'Not needed to reproduce the published result.')
    parser.add_argument('--dry-run', action='store_true',
                        help='Preview the perturbations and API call count for '
                             '--regenerate, without calling the API')
    parser.add_argument('--from-results', action='store_true',
                        help='Reproduce from the published per-poem distances. This '
                             'is the default; the flag is accepted for explicitness.')
    args = parser.parse_args()

    print("=" * 70)
    print("TEXT ABLATION STUDY")
    print("=" * 70)

    if not args.regenerate and not args.dry_run:
        # Default: offline reproduction. The perturbed texts do not exist in the
        # Zenodo release (they are generated here), so their embeddings cannot
        # be shipped -- but the resulting per-poem cosine distances are, which
        # is what every number in the paper is computed from.
        results_file = RESULTS_DIR / 'ablation_results.json'
        if not results_file.exists():
            print(f"\nERROR: {results_file} not found.")
            print("Re-run with --regenerate to recompute it via the Gemini API "
                  "(requires an API key and incurs cost).")
            return 1
        with open(results_file, encoding='utf-8') as f:
            published = json.load(f)
        results = published['results']
        print(f"\n[Reproducing from published distances: {results_file.name}]")
        print(f"  Sampled poems: {published['settings']['n_sampled']}")
        print(f"  Ablations: {len(results)}")
        print("  No API calls, no cost. Use --regenerate to recompute from scratch.")
        return report_ablation_results(results, published['settings'])

    if not GEMINI_API_KEY and not args.dry_run:
        print("\nERROR: --regenerate needs GEMINI_API_KEY in the environment.")
        print("Run without --regenerate to reproduce the published analysis offline,")
        print("or add --dry-run to preview the call count without calling the API.")
        return 1

    print("\n[--regenerate: perturbed texts will be re-embedded via the Gemini API]")
    print("  This calls a paid API. Run without --regenerate to reproduce offline.")

    # Load data
    print(f"\n[Loading dataset with {args.embedding_model} embeddings...]")
    data = load_clean_dataset(
        n_per_author=args.n_per_author,
        seed=args.seed,
        include_poems=True,
        embedding_model=args.embedding_model
    )

    poems = data['poems']
    embeddings = data['embeddings']
    labels = data['labels']
    author_list = data['author_list']

    n_samples = len(poems)
    n_authors = len(author_list)

    print(f"  Total samples: {n_samples}")
    print(f"  Authors: {n_authors}")

    # Sample poems for ablation
    print(f"\n[Sampling {args.sample_per_author} poems per author...]")
    (sampled_poems, sampled_embeddings, sampled_labels,
     sampled_indices) = sample_poems_stratified(
        poems, embeddings, labels,
        n_per_author=args.sample_per_author,
        seed=args.seed
    )

    n_sampled = len(sampled_poems)
    print(f"  Sampled: {n_sampled} poems")

    # Define ablations
    ablations = {
        'remove_punctuation': lambda p: ablate_remove_punctuation(p['text']),
        'shuffle_words': lambda p: ablate_shuffle_words(p['text'], seed=args.seed),
        'shuffle_lines': lambda p: ablate_shuffle_lines(p['text'], seed=args.seed),
        'lowercase': lambda p: ablate_lowercase(p['text']),
        'mask_content': lambda p: ablate_mask_content(p),
    }

    n_ablations = len(ablations)
    total_api_calls = n_sampled * n_ablations

    print(f"\n[Ablation types: {n_ablations}]")
    for name in ablations:
        print(f"  - {name}")
    print(f"\n  Total API calls needed: {total_api_calls}")

    if args.dry_run:
        print("\n[DRY RUN - not calling API]")
        print("  Run with --regenerate (without --dry-run) to actually re-embed;")
        print("  that calls the paid API. Run with no flags to reproduce the")
        print("  published analysis offline at no cost.")
        return 0

    # Apply ablations and get embeddings
    results = {}

    for ablation_name, ablation_fn in ablations.items():
        print(f"\n[Ablation: {ablation_name}]")

        # Apply ablation
        print("  Applying ablation...")
        modified_texts = [ablation_fn(p) for p in sampled_poems]

        # Show example
        print(f"  Example original: {sampled_poems[0]['text'][:80]}...")
        print(f"  Example modified: {modified_texts[0][:80]}...")

        # Get embeddings
        print("  Getting embeddings...")
        modified_embeddings = get_embeddings_batched(modified_texts)

        # Compute distances
        print("  Computing distances...")
        distances = compute_cosine_distances(sampled_embeddings, modified_embeddings)

        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        median_dist = np.median(distances)

        print(f"  Distance: {mean_dist:.4f} ± {std_dist:.4f} (median: {median_dist:.4f})")

        results[ablation_name] = {
            'mean_distance': float(mean_dist),
            'std_distance': float(std_dist),
            'median_distance': float(median_dist),
            'min_distance': float(np.min(distances)),
            'max_distance': float(np.max(distances)),
            'distances': distances.tolist()
        }

    settings = {
        'embedding_model': args.embedding_model,
        'n_sampled': n_sampled,
        'sample_per_author': args.sample_per_author,
        'n_authors': n_authors,
        'n_ablations': n_ablations,
        'seed': args.seed
    }
    return report_ablation_results(results, settings, save=True)


if __name__ == "__main__":
    sys.exit(main())
