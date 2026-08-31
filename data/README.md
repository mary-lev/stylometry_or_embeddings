# Data Description

This directory contains pre-computed embeddings and metadata for the poetry corpora used in this study.

**Note:** Large files (embeddings, poems.json) are hosted on Zenodo due to GitHub size limitations.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18260458.svg)](https://doi.org/10.5281/zenodo.18260458)

Run `python download_data.py` from the repository root to download them automatically.

## Directory Structure

```
data/
├── russian/
│   ├── dataset_metadata.json    # Author list, dataset parameters
│   ├── embeddings_gemini.npy    # Gemini text-embedding-001 (5800 x 3072)
│   ├── embeddings_openai.npy    # OpenAI text-embedding-3-large (5800 x 3072)
│   ├── embeddings_voyage.npy    # Voyage-3-large (5800 x 1024)
│   ├── embeddings_qwen8b.npy    # Qwen3-Embedding-8B (5800 x 4096)
│   ├── embeddings_e5-large.npy  # E5-large multilingual (5800 x 1024)
│   ├── embeddings_bge-m3.npy    # BGE-M3 (5800 x 1024)
│   ├── labels.npy               # Author indices (5800,)
│   ├── linguistic_features.json # POS, morphology, dependencies
│   ├── poems.json               # Poem texts (prosody included inline)
│   └── prosody.json             # Prosodic annotations from RNC
│
└── italian/
    ├── dataset_metadata.json    # Author list, dataset parameters
    ├── embeddings_gemini.npy    # Gemini text-embedding-001 (10400 x 3072)
    ├── embeddings_openai.npy    # OpenAI text-embedding-3-large (10400 x 3072)
    ├── embeddings_voyage.npy    # Voyage-3-large (10400 x 1024)
    ├── embeddings_qwen8b.npy    # Qwen3-Embedding-8B (10400 x 4096)
    ├── embeddings_e5-large.npy  # E5-large multilingual (10400 x 1024)
    ├── labels.npy               # Author indices (10400,)
    ├── linguistic_features.json # POS, morphology, dependencies
    └── poems.json               # Poem texts
```

## Russian Corpus

- **Source**: PoeTree project (poems from Stihi.ru matched with Russian National Corpus)
- **Authors**: 29 poets (1850-1930)
- **Poems per author**: 200 (balanced)
- **Total samples**: 5,800
- **Prosodic annotations**: Meter, feet, clausula, rhyme scheme (from RNC)

### Author List

1. Aleksandr Aleksandrovich Blok
2. Aleksandr Ivanovich Tinyakov
3. Aleksey Nikolaevich Apukhtin
4. Andrey Bely
5. Anna Andreevna Akhmatova
6. Apollon Nikolaevich Maykov
7. Afanasiy Afanasievich Fet
8. Boris Leonidovich Pasternak
9. Valeriy Yakovlevich Bryusov
10. Vladimir Ivanovich Narbut
11. Vyacheslav Ivanovich Ivanov
12. Georgiy Vladimirovich Ivanov
13. Dmitriy Sergeevich Merezhkovskiy
14. Zinaida Nikolaevna Gippius
15. Ivan Alekseevich Bunin
16. Innokentiy Fedorovich Annenskiy
17. Konstantin Dmitrievich Balmont
18. Konstantin Konstantinovich Sluchevskiy
19. Lev Lvovich Kobylinskiy
20. Maksimilian Aleksandrovich Voloshin
21. Marina Ivanovna Tsvetaeva
22. Mirra Aleksandrovna Lokhvitskaya
23. Mikhail Alekseevich Kuzmin
24. Mikhail Arkadievich Svetlov
25. Nikolay Stepanovich Gumilev
26. Osip Emilyevich Mandelshtam
27. Semyon Yakovlevich Nadson
28. Sergey Mitrofanovich Gorodetskiy
29. Yurgis Kazimirovich Baltrushaytis

## Italian Corpus

- **Source**: PoeTree project (Biblioteca Italiana via LirIta corpus)
- **Authors**: 52 poets (1200-1900)
- **Poems per author**: 200 (balanced)
- **Total samples**: 10,400
- **Linguistic annotations**: POS tags, morphological features, dependency relations

## File Formats

### Embeddings (*.npy)
NumPy arrays of shape `(n_samples, embedding_dim)`:
- Gemini: 3,072 dimensions
- Qwen3-8B: 4,096 dimensions

### Labels (labels.npy)
NumPy array of shape `(n_samples,)` containing integer author indices.
Use `dataset_metadata.json['author_list']` to map indices to author names.

### Prosody (prosody.json)
Dictionary mapping poem indices to prosodic features:
```json
{
  "0": {
    "meter": "Я",           // Iamb
    "feet": "4",            // 4-foot
    "clausula": "жм",       // Alternating feminine-masculine
    "rhyme": "перекрестная" // Cross rhyme (ABAB)
  },
  ...
}
```

### Linguistic Features (linguistic_features.json)

The two corpora ship this file in different shapes; `load_dataset()` handles
both and attaches the features to each poem as `poem['linguistic_features']`.

**Russian** — a list aligned with `poems.json` row order:
```json
[
  {
    "pos_counts": {"NOUN": 27, "VERB": 15, ...},
    "feats": {"Case": {"Nom": 18, "Gen": 12, ...}, ...},
    "deprels": {"nsubj": 5, "obj": 3, ...},
    "word_count": 95
  },
  ...
]
```

**Italian** — keyed by the poem `id` field from `poems.json`:
```json
{
  "metadata": {...},
  "poems": {
    "bibit000068-206": {
      "pos_counts": {...}, "feats": {...},
      "deprels": {...}, "word_count": 95
    },
    ...
  }
}
```

## Loading Data

```python
from src.data_loader import load_dataset

# Load Russian corpus with Gemini embeddings
data = load_dataset('russian', embedding_model='gemini')

print(data['embeddings'].shape)  # (5800, 3072)
print(data['labels'].shape)      # (5800,)
print(len(data['author_list']))  # 29

# Load Italian corpus
data_it = load_dataset('italian', embedding_model='gemini')
```

## Citation

If you use this data, please cite:

[Citation information to be added after publication]

## License

The poem texts are from publicly available corpora (PoeTree, Russian National Corpus).
Embeddings are provided for research purposes only.
