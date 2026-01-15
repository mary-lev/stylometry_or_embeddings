#!/usr/bin/env python3
"""
Data loader for the anonymized repository.

Provides a consistent interface to load the pre-computed clean datasets
for Russian and Italian poetry corpora.

Usage:
    from data_loader import load_dataset

    # Load Russian corpus with Gemini embeddings
    data = load_dataset('russian', embedding_model='gemini')

    # Load Italian corpus
    data = load_dataset('italian', embedding_model='gemini')
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Data directory (relative to this file)
DATA_DIR = Path(__file__).parent.parent / "data"


def load_dataset(
    language: str = 'russian',
    embedding_model: str = 'gemini',
    include_poems: bool = False
) -> Dict:
    """
    Load the clean, balanced dataset.

    Args:
        language: 'russian' or 'italian'
        embedding_model: 'gemini' or 'qwen8b'
        include_poems: If True, load full poem texts (Italian only)

    Returns:
        dict with:
            - embeddings: np.array (n_samples, embedding_dim)
            - labels: np.array (n_samples,) - author indices
            - author_list: list of author names
            - author_to_idx: dict mapping author name to index
            - metadata: dict with dataset parameters
            - poems: list of poem dicts (only if include_poems=True and Italian)
            - prosody: dict of prosodic features (Russian only)
    """
    lang_dir = DATA_DIR / language

    if not lang_dir.exists():
        raise FileNotFoundError(
            f"Data directory not found: {lang_dir}\n"
            f"Available languages: russian, italian"
        )

    # Load dataset metadata
    metadata_file = lang_dir / "dataset_metadata.json"
    with open(metadata_file, encoding='utf-8') as f:
        metadata = json.load(f)

    # Load embeddings
    emb_file = lang_dir / f"embeddings_{embedding_model}.npy"
    if not emb_file.exists():
        raise FileNotFoundError(
            f"Embeddings not found: {emb_file}\n"
            f"Available models: gemini, openai, voyage, qwen8b, e5-large, bge-m3"
        )

    embeddings = np.load(emb_file)
    labels = np.load(lang_dir / "labels.npy")

    author_list = metadata['author_list']

    result = {
        'embeddings': embeddings,
        'labels': labels,
        'author_list': author_list,
        'author_to_idx': {a: i for i, a in enumerate(author_list)},
        'idx_to_author': {i: a for i, a in enumerate(author_list)},
        'metadata': {
            'language': language,
            'n_per_author': metadata['parameters']['n_per_author'],
            'n_authors': metadata['statistics']['n_authors'],
            'n_poems': metadata['statistics']['n_poems_total'],
            'seed': metadata['parameters']['seed'],
            'embedding_model': embedding_model,
            'embedding_dim': embeddings.shape[1]
        }
    }

    # Load prosody for Russian
    if language == 'russian':
        prosody_file = lang_dir / "prosody.json"
        if prosody_file.exists():
            with open(prosody_file, encoding='utf-8') as f:
                result['prosody'] = json.load(f)

    # Load linguistic features (available for both Russian and Italian)
    ling_file = lang_dir / "linguistic_features.json"
    if ling_file.exists():
        with open(ling_file, encoding='utf-8') as f:
            result['linguistic_features'] = json.load(f)

    # Load poems if requested
    if include_poems:
        poems_file = lang_dir / "poems.json"
        if poems_file.exists():
            with open(poems_file, encoding='utf-8') as f:
                poems = json.load(f)

            # Merge prosody into poems for Russian
            if language == 'russian' and 'prosody' in result:
                prosody_data = result['prosody']
                for poem in poems:
                    idx = str(poem.get('idx', ''))
                    if idx in prosody_data:
                        poem['prosody'] = prosody_data[idx]

            result['poems'] = poems

    return result


def load_clean_dataset(
    n_per_author: int = 200,
    seed: int = 42,
    include_poems: bool = False,
    embedding_model: str = 'gemini',
    language: str = 'russian'
) -> Dict:
    """
    Compatibility wrapper for existing experiment scripts.

    Maps to load_dataset() with appropriate parameters.
    """
    return load_dataset(
        language=language,
        embedding_model=embedding_model,
        include_poems=include_poems
    )


def get_author_embeddings(data: Dict, author: str) -> np.ndarray:
    """Get embeddings for a specific author."""
    author_idx = data['author_to_idx'][author]
    mask = data['labels'] == author_idx
    return data['embeddings'][mask]


def get_pair_data(
    data: Dict,
    author1: str,
    author2: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Get balanced data for a pair of authors."""
    idx1 = data['author_to_idx'][author1]
    idx2 = data['author_to_idx'][author2]

    mask1 = data['labels'] == idx1
    mask2 = data['labels'] == idx2

    emb1 = data['embeddings'][mask1]
    emb2 = data['embeddings'][mask2]

    embeddings = np.vstack([emb1, emb2])
    labels = np.array([0] * len(emb1) + [1] * len(emb2))

    return embeddings, labels


def get_all_author_pairs(data: Dict) -> List[Tuple[str, str]]:
    """Get all valid author pairs for pairwise classification."""
    authors = data['author_list']
    pairs = []
    for i, a1 in enumerate(authors):
        for a2 in authors[i+1:]:
            pairs.append((a1, a2))
    return pairs


if __name__ == "__main__":
    # Test loading
    print("Testing data loader...")

    for lang in ['russian', 'italian']:
        print(f"\n{lang.upper()}:")
        try:
            data = load_dataset(lang, embedding_model='gemini')
            print(f"  Embeddings shape: {data['embeddings'].shape}")
            print(f"  Labels shape: {data['labels'].shape}")
            print(f"  Authors: {data['metadata']['n_authors']}")
            print(f"  Poems per author: {data['metadata']['n_per_author']}")
        except FileNotFoundError as e:
            print(f"  Not found: {e}")
