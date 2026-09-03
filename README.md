# Stylometry or Embeddings? Authorship Attribution for Russian and Italian Poetry

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21736209.svg)](https://doi.org/10.5281/zenodo.21736209)

This repository contains code and data for reproducing the experiments in the paper:

> **Stylometry or Embeddings? Authorship Attribution for Russian and Italian Poetry**
>
> *Maria Levchenko (University of Bologna)*
>
> Journal of Computational Literary Studies


## Overview

We investigate what LLM embeddings encode about authorial style through a residualization analysis of two poetry corpora:
- **Russian**: 5,800 poems by 29 poets (1850-1930)
- **Italian**: 10,400 poems by 52 poets (1200-1900)

### Key Findings

1. **For Russian poetry, residual signal collapses to near chance (1.1×)** after accounting for character n-grams and word bigrams
2. **Character n-grams explain 63% of embedding variance** — far more than all interpretable linguistic features combined (~11%)
3. **Embeddings and stylometry fail on different texts** (39% error overlap, Jaccard), indicating complementarity


## Quick Start

```bash
# Clone repository
git clone https://github.com/mary-lev/stylometry_or_embeddings.git
cd stylometry_or_embeddings

# Install dependencies
pip install -r requirements.txt

# Get the data from Zenodo (~251 MB) and verify it
python download_data.py

# Reproduce two published results in a couple of seconds
python experiments/exp7_waterfall/run_permutation_test.py
python experiments/exp10_embedding_structure/run_ablation.py
```

`download_data.py` downloads whatever is missing and then checksums every
file, so it is also the way to confirm an existing download. It is safe to
re-run: files already on disk are checked, not fetched again.

If those three commands succeed, the installation is correct. The section
below runs the actual experiments, shortest first.

---

## Reproduction Sequence

Commands are grouped by cost so you can stop at any stage. Stages 1–3 run on
the default download from Quick Start; only stage 4 needs more data.

**Nearly every command runs as written, with no arguments.** Defaults are the
published settings: Russian, Gemini embeddings, 200 poems per author, seed 42,
5-fold CV. Stages 2 and 3 are argument-free entirely; stage 1 has one argument
and stage 4 has three, each selecting something genuinely different —
`--language italian` for the Italian permutation test, `--all` for the extra
embeddings, and `--embedding-model qwen8b` for the two Qwen3-8B columns.

The comment after each command is the headline number it should print, so you
can check as you go. Full expected output for each is in *Reproducing Results*
below. Timings are from one 20-core machine and are indicative.

### Stage 1 — verify (about 3 seconds in total)

These re-derive published numbers from the results files already in the
repository. Nothing is recomputed, so this is the fastest way to confirm the
installation is sound. (The data itself was already checksummed by
`download_data.py` in Quick Start.)

```bash
python experiments/exp7_waterfall/run_permutation_test.py     # RU: not above chance
python experiments/exp7_waterfall/run_permutation_test.py --language italian
                                                              # IT: above chance, p<0.001
python experiments/exp10_embedding_structure/run_ablation.py  # mask 0.134 / shuffle 0.027
python experiments/exp7_waterfall/make_waterfall_figures.py   # writes 2 PNGs
```

### Stage 2 — fast experiments (a few minutes each)

```bash
python experiments/exp5_interpretability/run.py               # topic probe 81.3%
python experiments/exp5_interpretability/run_cross_topic.py   # cross-topic 35.7%
python experiments/exp8_combined_features/run.py              # 61.3 / 59.2 / 70.5
python experiments/exp8_combined_features/analyze_residualization.py
                                                              # residual embeddings ~12.5%
python experiments/exp4_multiclass/run.py                     # multiclass 61.3%
python experiments/exp2_binary_pairs/run_italian.py           # Italian pairwise
```

### Stage 3 — main results (10–40 minutes each)

```bash
# Table 3, the main result. Italian is the slowest here, about 40 minutes.
python experiments/exp7_waterfall/run_extended.py             # 61.3 -> 3.8% (1.1x chance)
python experiments/exp7_waterfall/run_italian_extended.py     # 67.0 -> 8.9% (4.6x chance)

# Tables 2 (Italian), 4 and 6, and the frequency bands
python experiments/exp8_combined_features/run_italian.py      # 67.1 / 79.7 / 81.8
python experiments/exp4_multiclass/run_length_sensitivity.py  # 61.3/57.8 -> 99.7/100.0
python experiments/exp7_waterfall/probe_discriminative_features.py
                                                              # vocabulary diversity R2 0.77
python experiments/exp7_waterfall/analyze_frequency_bands.py  # 500-2000 band highest, 77.6%

# Pairwise attribution and error analysis
python experiments/exp2_binary_pairs/run.py                   # mean over 406 pairs 89.6%
python experiments/exp2_binary_pairs/analyze_method_differences.py
                                                              # per-poem 93.8% / 94.5%
python experiments/exp4_multiclass/analyze_errors_comparison.py
                                                              # error overlap 39%, TTR -0.16
python experiments/exp7_waterfall/test_nonlinear_tiers.py     # kernel 47.8% vs linear 4.3%
```

### Stage 4 — needs the full download or hours of compute

```bash
python download_data.py --all        # all embedding models, ~896 MB

# Figure 1 compares seven models
python experiments/exp8a_multi_model_comparison/run.py           # Gemini highest, 61.3%
python experiments/exp8a_multi_model_comparison/run_italian.py   # Gemini highest, 67.1%

# The Qwen3-8B columns of Table 3 -- the only place a model must be named
python experiments/exp7_waterfall/run_extended.py --embedding-model qwen8b
                                                                 # 50.7 -> 3.8%
python experiments/exp7_waterfall/run_italian_extended.py --embedding-model qwen8b
                                                                 # 57.6 -> 9.4%

# Italian pairwise: 1,326 author pairs, allow 2-3 hours
python experiments/exp2_binary_pairs/analyze_method_differences_italian.py
                                                                 # per-poem 96.1% / 98.5%
python experiments/exp2_binary_pairs/plot_binary_error_by_length.py
```

Two further options exist but are **not needed** to reproduce anything: 
`run_permutation_test.py --recompute` (~3.5 h Russian, ~7 h Italian) and
`run_ablation.py --regenerate` (calls a paid API). Stage 1 reproduces both
results from the published artifacts.

---

## Data

**Note:** Large files (embeddings, poems) are hosted on Zenodo due to GitHub size limitations.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18260458.svg)](https://doi.org/10.5281/zenodo.18260458)

`download_data.py` fetches what the experiments need and checksums every file
afterwards. Re-running it is how you confirm an existing download; nothing is
fetched twice.

```bash
python download_data.py              # what the experiments need (~251 MB)
python download_data.py --all        # also the other embedding models (~896 MB)
python download_data.py --verify     # check only, download nothing
python download_data.py --force      # re-download even if files are present
```

See `data/README.md` for detailed documentation.

**Corpora:**
- **Russian**: 29 poets, 200 poems each, with prosodic annotations from Russian National Corpus
- **Italian**: 52 poets, 200 poems each, spanning seven centuries (1200-1900)

**Embeddings (6 models):**
- Gemini text-embedding-001 (3,072 dimensions) — API
- OpenAI text-embedding-3-large (3,072 dimensions) — API
- Voyage-3-large (1,024 dimensions) — API
- Qwen3-Embedding-8B (4,096 dimensions) — Local
- E5-large multilingual (1,024 dimensions) — Local
- BGE-M3 (1,024 dimensions) — Local, Russian only

---

## Requirements

- Python 3.8+
- NumPy >= 1.21.0
- scikit-learn >= 1.0.0
- SciPy >= 1.7.0
- pandas >= 1.3.0
- matplotlib >= 3.4.0

See `requirements.txt` for full dependencies.

---


## Reproducing Results

### Figure 1: Embedding Model Comparison

```bash
python download_data.py --all      # this comparison needs every model
python experiments/exp8a_multi_model_comparison/run.py
python experiments/exp8a_multi_model_comparison/run_italian.py
```

**Note**: this is the one experiment that needs more than the default
download. With
only the Gemini embeddings downloaded it still runs, but compares a single
model and says so loudly; the figure then shows one bar instead of the table
below.

Pre-computed results: `experiments/exp8a_multi_model_comparison/results/`

**Expected output:**

![Model Comparison](figures/embedding_model_comparison.png)

| Model | Russian Accuracy | Italian Accuracy |
|-------|-----------------|------------------|
| Gemini text-embedding-001 | **61.3%** | **67.1%** |
| OpenAI text-embedding-3-large | 56.1% | 65.4% |
| Voyage-3-large | 51.6% | 59.4% |
| Qwen3-Embedding-8B | 50.8% | 57.6% |
| E5-large | 47.2% | 61.3% |
| BGE-M3 | 41.9% | — (Russian only) |

Chance is 3.4% (Russian) and 1.9% (Italian). Gemini ranks highest and Qwen3-8B is
the strongest open-source model in both corpora, which is why those two are carried
through the main analysis.

---

### Table 3: Waterfall Residualization (Main Result)

```bash
# Russian corpus, Gemini embeddings (the default)
python experiments/exp7_waterfall/run_extended.py

# Italian corpus, Gemini embeddings (the default)
python experiments/exp7_waterfall/run_italian_extended.py

# The same two with Qwen3-8B instead (needs `download_data.py --all`)
python experiments/exp7_waterfall/run_extended.py --embedding-model qwen8b
python experiments/exp7_waterfall/run_italian_extended.py --embedding-model qwen8b
```

The Qwen3-8B pair is optional: the Gemini columns reproduce the main result on
their own, and they run on the default download.

Pre-computed results: `experiments/exp7_waterfall/results/`

**Expected output:**

![Waterfall Residualization](figures/waterfall_figure.png)

| Stage | RU Gemini | RU Qwen3-8B | IT Gemini | IT Qwen3-8B |
|-------|-----------|-------------|-----------|-------------|
| **Baseline embeddings** | 61.3% | 50.7% | 67.0% | 57.6% |
| After interpretable (T1-T4) | 40.0% | 32.2% | 46.0% | 41.1% |
| After char n-grams (T5) | 10.2% | 8.3% | 16.1% | 16.8% |
| After word bigrams (T6) | 3.8% | 4.0% | 8.7% | 9.5% |
| **Final residual (kernel control)** | **3.8%** | **3.8%** | **8.9%** | **9.4%** |
| | | | | |
| **Final lift vs. chance** | **1.1×** | **1.1×** | **4.6×** | **4.9×** |
| Chance level | 3.4% | 3.4% | 1.9% | 1.9% |

**Key insight**: For Russian, the residual drops to chance after removing character-level features.

---

### Table 2: Stylometry Baseline & Combined Results

```bash
python experiments/exp0_stylometry_baseline/run.py
python experiments/exp8_combined_features/run.py
python experiments/exp8_combined_features/run_italian.py
```

Pre-computed results: `experiments/exp8_combined_features/results/`

**Expected output:**

![Russian Comparison](figures/combined_comparison.png)

![Italian Comparison](figures/combined_comparison_italian.png)

| Method | Russian (29 authors) | Italian (52 authors) |
|--------|---------------------|----------------------|
| Embeddings (Gemini) | 61.3% ± 1.3% | 67.1% ± 0.5% |
| Stylometry | 59.2% ± 1.1% | 79.7% ± 1.3% |
| **Combined** | **70.5% ± 0.9%** | **81.8% ± 1.1%** |
| Chance | 3.4% | 1.9% |

---

### Table 4: Effect of Text Length

```bash
python experiments/exp4_multiclass/run_length_sensitivity.py
```

Pre-computed results: `experiments/exp4_multiclass/results/`

**Expected output:**

![Error by Length](figures/binary_fig2_error_by_length.png)

| Concatenation | Words | Gemini Embeddings | Stylometry |
|---------------|-------|-------------------|------------|
| 1 poem | 100 | 61.3% | 57.8% |
| 5 poems | 502 | 95.3% | 96.2% |
| 10 poems | 1,004 | 99.1% | 99.5% |
| 20 poems | 2,008 | 99.7% | 100.0% |

**Finding**: Text length is the primary limiting factor for single-poem attribution.

---

### Tables 6-7: Probing Analysis

```bash
# Classification probes (semantic and prosodic properties)
python experiments/exp5_interpretability/run.py

# Continuous probes (lexical and punctuation properties, Ridge regression)
python experiments/exp7_waterfall/probe_discriminative_features.py
```

Pre-computed results: `experiments/exp5_interpretability/results/probing_results.json`
and `experiments/exp7_waterfall/results/probe_discriminative_gemini.json`

**Expected output:**

#### Continuous Properties (Ridge Regression)

| Property | R² | Encoding |
|----------|-----|----------|
| Vocabulary diversity (1-100 MFW) | 0.77 | Strong |
| Vocabulary diversity (100-500 MFW) | 0.68 | Strong |
| Vocabulary diversity (500-2000 MFW) | 0.63 | Strong |
| Exclamation rate | 0.53 | Moderate |
| Ellipsis rate | 0.44 | Moderate |
| Em-dash rate | 0.30 | Moderate |
| Hyphen rate | -0.37 | Not encoded |
| Colon rate | < 0 | Not encoded |

#### Classification Properties (Logistic Regression)

| Property | Type | Accuracy | Chance | Lift |
|----------|------|----------|--------|------|
| Topic (Inner vs. Nature) | Semantic | 81.3% | 50% | 1.6× |
| Meter (Iamb vs. Trochee) | Prosodic | 65.1% | 50% | 1.3× |
| Feet (4 vs. 5) | Prosodic | 63.3% | 50% | 1.3× |
| Rhyme (ABAB vs. AABB) | Prosodic | 61.0% | 50% | 1.2× |
| Clausula (regular vs. free) | Prosodic | 56.2% | 50% | 1.1× |
| Meter (5-class) | Prosodic | 32.8% | 20% | 1.6× |

**Finding**: Embeddings encode semantic content (topic: 81%) more strongly than prosodic structure (56-65%).

---

### Table 8: Cross-Topic Validation

```bash
python experiments/exp5_interpretability/run_cross_topic.py
```

Pre-computed results: `experiments/exp5_interpretability/results/cross_topic_validation_gemini.json`

**Expected output:**

| Condition | Accuracy | Δ from baseline |
|-----------|----------|-----------------|
| Baseline (random CV) | 61.3% | — |
| Leave-one-topic-out | 55.8% | -5.5 pp |
| Within-topic | 55.5% | -5.8 pp |
| **Cross-topic** | **35.7%** | **-25.6 pp** |

**Finding**: ~40% of signal is topic-dependent, ~60% reflects genuine stylistic patterns.

---

### Permutation Test (Residual Significance)

Tests whether the final residual accuracy is significantly above chance.

```bash
python experiments/exp7_waterfall/run_permutation_test.py
python experiments/exp7_waterfall/run_permutation_test.py --language italian
```

Runs in seconds by default — the published results files store all 1,000 null
accuracies, so the test is re-derived from them rather than recomputed.

**Expected output:**

| | Russian | Italian |
|---|---------|---------|
| Null mean | 3.4% | 1.9% |
| Null 95% CI | [3.0%, 3.9%] | [1.7%, 2.2%] |
| Observed residual | 3.9% | 8.9% |
| p-value | ~0.06 | < 0.001 |
| Within null 95% CI | yes | no |
| Conclusion | **not** above chance | above chance |

For Russian the null distribution is dense right around the residual, so the
exact p-value is sensitive to the residual's second decimal — a residual of
3.78%, 3.80% or 3.86% gives p = 0.089, 0.080 or 0.058 respectively. All are
above 0.05 and inside the null 95% CI, so the conclusion is unaffected, but
do not expect the p-value to reproduce to two decimals. Italian is
unambiguous (8.9% sits far outside the null).

Pre-computed results: `experiments/exp7_waterfall/results/permutation_test_*.json`

<details>
<summary>Regenerating the null distribution (optional, hours)</summary>

`--recompute` reshuffles labels 1,000 times and re-classifies each time:
roughly **3.5 h for Russian and 7 h for Italian** (53 s setup + ~13 s per
permutation). This is not needed to verify the published result.

```bash
python experiments/exp7_waterfall/run_permutation_test.py --recompute
```

Note that this script and `run_extended.py` residualize differently: the
waterfall fits per fold on training data only (residual 3.8% / 8.9%, the value
the paper reports), while the permutation test residualizes once over all data
to hold the residuals fixed while labels are shuffled (giving a lower
`observed_accuracy` of 1.2% / 2.5% in the results file). The null distribution
is label-independent and applies to either; the verification above compares it
against the waterfall residual.
</details>

---

### Vocabulary Frequency Bands

```bash
python experiments/exp7_waterfall/analyze_frequency_bands.py
```

Pre-computed results: `experiments/exp7_waterfall/results/frequency_band_analysis.json`

**Expected output:**

| Frequency band | Word type | Accuracy |
|----------------|-----------|----------|
| 1-100 | Function words | 76.4% |
| 100-500 | Style markers | 76.6% |
| 500-2000 | Topic vocabulary | **77.6%** |
| 2000-5000 | Rare vocabulary | 74.0% |

**Finding**: The prose pattern inverts — topic vocabulary is the most
author-discriminative band for poetry, not function words.

---

### Binary Attribution Error by Text Length

```bash
# Per-poem method comparison (produces method_differences_*.json)
python experiments/exp2_binary_pairs/analyze_method_differences.py

# Figure, plotted from the analysis above
python experiments/exp2_binary_pairs/plot_binary_error_by_length.py
```

Pre-computed results: `experiments/exp2_binary_pairs/results/`

---

### Embedding Ablation (Perturbation Analysis)

Measures how far embeddings move when content words are masked versus when word
order is shuffled — the basis for the claim that semantic content dominates
embedding geometry (cosine distance 0.134 vs. 0.027).

```bash
python experiments/exp10_embedding_structure/run_ablation.py
```

Runs offline by default — no API key, no cost.

**Expected output:**

| Modification | Cosine distance from original |
|--------------|-------------------------------|
| Mask content words | **0.1337** ± 0.0243 |
| Shuffle word order | 0.0266 ± 0.0081 |
| Shuffle line order | 0.0201 ± 0.0092 |
| Remove punctuation | 0.0191 ± 0.0055 |
| Lowercase | 0.0042 ± 0.0030 |

**Finding**: Masking content words moves embeddings ~5× further than shuffling
word order — semantic content dominates embedding geometry.

Pre-computed results: `experiments/exp10_embedding_structure/results/ablation_results.json`,
which stores the per-poem cosine distance for all 290 sampled poems across all five
ablations. Every number above is recomputed from that file.

<details>
<summary>Regenerating the distances from scratch (optional, requires an API key)</summary>

Unlike every other experiment, this one perturbs the poem text and must re-embed
the *modified* text, so the Zenodo embeddings — which cover the original poems
only — cannot supply it. Regenerating therefore calls the Gemini API once per
poem per ablation (290 × 5 = 1,450 calls) and costs money. This is **not**
required to reproduce the published result — the default command above already
does that.

```bash
export GEMINI_API_KEY=YOUR_API_KEY_HERE
python experiments/exp10_embedding_structure/run_ablation.py --regenerate

# Preview the perturbations and call count without calling the API
python experiments/exp10_embedding_structure/run_ablation.py --dry-run
```
</details>

---

### Regenerating the Figures

Every figure in `figures/` is generated from a results file, not hand-edited.
Each plotting script reads the results of the experiment above it in the
Reproduction Sequence, and names the missing command if you run it too early.

| Figure | Plotting script | Needs |
|--------|-----------------|-------|
| `waterfall_figure.png`, `waterfall_figure_qwen.png` | `exp7_waterfall/make_waterfall_figures.py` | ships with results — runs immediately |
| `embedding_model_comparison.png` | `exp8a_multi_model_comparison/visualize_model_comparison.py` | stage 4 `exp8a/run.py` |
| `combined_comparison.png`, `combined_comparison_italian.png` | `exp8a_multi_model_comparison/visualize{,_italian}.py` | stage 4 `exp8a/run{,_italian}.py` |
| `binary_fig2_error_by_length.png` | `exp2_binary_pairs/plot_binary_error_by_length.py` | stage 3 + 4 `analyze_method_differences{,_italian}.py` |

**Note on Qwen-0.6B**: Figure 1 includes a seventh model, Qwen3-Embedding-0.6B,
whose embeddings are **not** part of the Zenodo release. `exp8a/run.py` skips it
automatically when the file is absent and reports which models were skipped,
producing the six-model figure instead.

---

## Repository Structure

```
.
├── data/                 # Downloaded from Zenodo (see download_data.py)
│   ├── russian/          # Russian poetry corpus (5,800 poems)
│   │   ├── embeddings_gemini.npy     # 68 MB
│   │   ├── embeddings_openai.npy     # 136 MB
│   │   ├── embeddings_voyage.npy     # 23 MB
│   │   ├── embeddings_qwen8b.npy     # 91 MB
│   │   ├── embeddings_e5-large.npy   # 23 MB
│   │   ├── embeddings_bge-m3.npy     # 23 MB
│   │   ├── poems.json
│   │   └── linguistic_features.json
│   └── italian/          # Italian poetry corpus (10,400 poems)
│       ├── embeddings_gemini.npy     # 122 MB
│       ├── embeddings_openai.npy     # 122 MB
│       ├── embeddings_voyage.npy     # 41 MB
│       ├── embeddings_qwen8b.npy     # 163 MB
│       ├── embeddings_e5-large.npy   # 41 MB
│       ├── poems.json
│       └── linguistic_features.json
│
├── src/                  # Core library
│   ├── data_loader.py    # Data loading utilities
│   ├── features.py       # Feature extraction (175 features)
│   ├── statistical_tests.py
│   └── ...
│
├── experiments/          # Experiment scripts + pre-computed results
│   ├── exp0_stylometry_baseline/   # Stylometry baselines
│   ├── exp2_binary_pairs/          # Pairwise attribution, error by length
│   ├── exp4_multiclass/            # Multiclass attribution, text length
│   ├── exp5_interpretability/      # Classification probes, cross-topic
│   ├── exp7_waterfall/             # Residualization waterfall, continuous
│   │                               #   probes, frequency bands
│   ├── exp8_combined_features/     # Combined features, bidirectional
│   ├── exp8a_multi_model_comparison/  # Embedding model comparison
│   └── exp10_embedding_structure/  # Ablation / perturbation analysis
│
├── figures/              # Publication figures
│
├── download_data.py      # Download data from Zenodo
├── requirements.txt      # Python dependencies
└── LICENSE               # MIT License
```

---

## Feature Tiers

Our residualization waterfall uses 6 feature tiers:

| Tier | Features | Description |
|------|----------|-------------|
| 1 | 20 | **Surface**: length, punctuation, TTR |
| 2 | 40 | **Content**: TF-IDF + LDA topics |
| 3 | 100 | **Grammar**: POS, morphology, dependencies |
| 4 | 15 | **Prosody**: meter, rhyme, clausula (Russian only) |
| 5 | 2,000 | **Character n-grams** (2-4) |
| 6 | 2,000 | **Word bigrams** |

---


## Citation

If you use this code or data, please cite:

```bibtex
@article{levchenko2026stylometry,
  title={Stylometry or Embeddings? Authorship Attribution for Russian and Italian Poetry},
  author={Levchenko, Maria},
  journal={Journal of Computational Literary Studies},
  year={2026}
}
```

The code itself can be cited via its Zenodo archive: [10.5281/zenodo.21736209](https://doi.org/10.5281/zenodo.21736209).

---

## License

This code is released under the MIT License. See `LICENSE` for details.

The poem texts are sourced from publicly available corpora (PoeTree project).
Embeddings are provided for research purposes only.
