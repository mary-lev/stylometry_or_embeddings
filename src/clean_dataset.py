#!/usr/bin/env python3
"""
Compatibility wrapper for clean dataset loading.

This module provides backward compatibility with experiment scripts
that import from clean_dataset instead of data_loader.
"""

try:
    from .data_loader import (
        load_dataset,
        load_clean_dataset,
        get_author_embeddings,
        get_pair_data,
        get_all_author_pairs
    )
except ImportError:
    from data_loader import (
        load_dataset,
        load_clean_dataset,
        get_author_embeddings,
        get_pair_data,
        get_all_author_pairs
    )

__all__ = [
    'load_dataset',
    'load_clean_dataset',
    'get_author_embeddings',
    'get_pair_data',
    'get_all_author_pairs'
]
