"""
Feature extraction for stylometric analysis.

Tier 1: Surface & Tokenization (20 features)
Tier 2: Content/Topic (40 features) - TF-IDF + LDA
Tier 3: Grammar (100 features) - POS + Morphology + Dependencies
Tier 4: Prosody (15 features) - Meter, rhyme, clausula, strophics
"""

import numpy as np
from collections import Counter
from typing import Dict, List, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Tier definitions with feature counts
TIER_DEFINITIONS = {
    'tier1_surface': 20,
    'tier2_content': 40,
    'tier3_grammar': 100,
    'tier4_prosody': 15,
    'total': 175
}

# =============================================================================
# TIER 1: Surface Features (20)
# =============================================================================

def extract_tier1_surface(poem: Dict) -> np.ndarray:
    """
    Extract Tier 1: Surface & Tokenization features (20 features).

    Features:
    1. char_count - total character count
    2. word_count - total word count
    3. line_count - number of lines
    4. avg_word_length - mean word length
    5. std_word_length - std dev of word length
    6. avg_line_length_chars - mean characters per line
    7. std_line_length_chars - std dev of characters per line
    8. avg_line_length_words - mean words per line
    9. punct_rate - punctuation per character
    10. exclaim_rate - ! per word
    11. question_rate - ? per word
    12. dash_rate - —/– per word
    13. ellipsis_rate - …/... per word
    14. comma_rate - , per word
    15. semicolon_rate - ; per word
    16. colon_rate - : per word
    17. type_token_ratio - unique words / total words
    18. sentence_count - count of terminal punctuation (.!?)
    19. words_per_sentence - avg words per sentence
    20. newline_rate - line breaks per word
    """
    text = poem.get('text', '')
    lines = [l for l in text.strip().split('\n') if l.strip()]
    words = text.split()

    # Basic counts
    char_count = len(text)
    word_count = len(words)
    line_count = len(lines)

    # Word length stats
    word_lengths = [len(w) for w in words] if words else [0]
    avg_word_length = np.mean(word_lengths)
    std_word_length = np.std(word_lengths)

    # Line length stats (in characters)
    line_lengths_chars = [len(l) for l in lines] if lines else [0]
    avg_line_length_chars = np.mean(line_lengths_chars)
    std_line_length_chars = np.std(line_lengths_chars)

    # Line length in words
    line_lengths_words = [len(l.split()) for l in lines] if lines else [0]
    avg_line_length_words = np.mean(line_lengths_words)

    # Punctuation rates (per word)
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

    # Type-token ratio (lexical diversity)
    unique_words = set(w.lower() for w in words)
    type_token_ratio = len(unique_words) / n if n > 0 else 0

    # Sentence-level features
    sentence_count = text.count('.') + text.count('!') + text.count('?')
    sentence_count = max(sentence_count, 1)
    words_per_sentence = word_count / sentence_count

    # Newline rate
    newline_rate = line_count / n if n > 0 else 0

    return np.array([
        char_count,
        word_count,
        line_count,
        avg_word_length,
        std_word_length,
        avg_line_length_chars,
        std_line_length_chars,
        avg_line_length_words,
        punct_rate,
        exclaim_rate,
        question_rate,
        dash_rate,
        ellipsis_rate,
        comma_rate,
        semicolon_rate,
        colon_rate,
        type_token_ratio,
        sentence_count,
        words_per_sentence,
        newline_rate,
    ])

TIER1_FEATURE_NAMES = [
    'char_count', 'word_count', 'line_count',
    'avg_word_length', 'std_word_length',
    'avg_line_length_chars', 'std_line_length_chars', 'avg_line_length_words',
    'punct_rate', 'exclaim_rate', 'question_rate', 'dash_rate', 'ellipsis_rate',
    'comma_rate', 'semicolon_rate', 'colon_rate',
    'type_token_ratio', 'sentence_count', 'words_per_sentence', 'newline_rate'
]


# =============================================================================
# MODEL-SPECIFIC TOKENIZATION FEATURES
# =============================================================================
# These features capture how each embedding model tokenizes the text.
# A Russian word like "перекрёстный" might be 1 token in one model but 3 in another.

# Tokenizer cache to avoid repeated loading
_TOKENIZER_CACHE = {}


def get_tokenizer(model_name: str):
    """
    Get tokenizer for a specific model. Cached for efficiency.

    Supported models:
    - 'openai': tiktoken cl100k_base
    - 'gemini': tiktoken cl100k_base (approximation)
    - 'voyage': tiktoken cl100k_base (approximation)
    - 'qwen8b', 'qwen06b': Qwen/Qwen2.5-7B-Instruct tokenizer
    - 'e5-large': intfloat/multilingual-e5-large
    - 'bge-m3': BAAI/bge-m3
    """
    if model_name in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[model_name]

    tokenizer = None

    if model_name in ['openai', 'gemini', 'voyage']:
        # Use tiktoken for OpenAI-style tokenizers
        try:
            import tiktoken
            tokenizer = ('tiktoken', tiktoken.get_encoding("cl100k_base"))
        except ImportError:
            pass

    elif model_name in ['qwen8b', 'qwen06b']:
        try:
            from transformers import AutoTokenizer
            tokenizer = ('hf', AutoTokenizer.from_pretrained(
                "Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True))
        except Exception:
            pass

    elif model_name == 'e5-large':
        try:
            from transformers import AutoTokenizer
            tokenizer = ('hf', AutoTokenizer.from_pretrained(
                "intfloat/multilingual-e5-large"))
        except Exception:
            pass

    elif model_name == 'bge-m3':
        try:
            from transformers import AutoTokenizer
            tokenizer = ('hf', AutoTokenizer.from_pretrained("BAAI/bge-m3"))
        except Exception:
            pass

    _TOKENIZER_CACHE[model_name] = tokenizer
    return tokenizer


def compute_tokens_per_word(text: str, model_name: str = 'openai') -> float:
    """
    Compute the tokens-per-word ratio for a given text and model.

    This measures how efficiently a model tokenizes the text.
    Higher values indicate more subword fragmentation.

    Args:
        text: The poem text
        model_name: Name of the embedding model

    Returns:
        tokens_per_word ratio (float)
    """
    words = text.split()
    word_count = len(words)

    if word_count == 0:
        return 0.0

    tokenizer = get_tokenizer(model_name)

    if tokenizer is None:
        # Fallback: estimate based on average Russian word length
        # Russian uses ~1.5 tokens per word on average in BPE tokenizers
        return 1.5

    tokenizer_type, tok = tokenizer

    try:
        if tokenizer_type == 'tiktoken':
            token_count = len(tok.encode(text))
        else:  # HuggingFace
            token_count = len(tok.encode(text, add_special_tokens=False))
    except Exception:
        return 1.5  # Fallback

    return token_count / word_count


def extract_tokenization_features(poems: list, model_name: str = 'openai') -> np.ndarray:
    """
    Extract model-specific tokenization features for a list of poems.

    Returns array of shape (n_poems, 1) with tokens_per_word for each poem.
    """
    features = []
    for poem in poems:
        text = poem.get('text', '')
        tpw = compute_tokens_per_word(text, model_name)
        features.append([tpw])

    return np.array(features)

# =============================================================================
# TIER 3: Grammar Features (100)
# =============================================================================

# POS tags (Universal Dependencies for Russian)
POS_TAGS = ['NOUN', 'VERB', 'ADJ', 'ADV', 'PRON', 'DET', 'ADP', 'CCONJ',
            'SCONJ', 'PART', 'INTJ', 'NUM', 'PUNCT', 'SYM', 'X', 'PROPN', 'AUX']

# Morphological features with all values present in data
MORPH_FEATURES = [
    # Animacy (2)
    'Animacy=Anim', 'Animacy=Inan',
    # Aspect (2)
    'Aspect=Imp', 'Aspect=Perf',
    # Case (7)
    'Case=Nom', 'Case=Gen', 'Case=Dat', 'Case=Acc', 'Case=Ins', 'Case=Loc', 'Case=Voc',
    # Degree (3)
    'Degree=Pos', 'Degree=Cmp', 'Degree=Sup',
    # Gender (3)
    'Gender=Masc', 'Gender=Fem', 'Gender=Neut',
    # Mood (3)
    'Mood=Ind', 'Mood=Imp', 'Mood=Cnd',
    # Number (2)
    'Number=Sing', 'Number=Plur',
    # Person (3)
    'Person=1', 'Person=2', 'Person=3',
    # Tense (3)
    'Tense=Past', 'Tense=Pres', 'Tense=Fut',
    # VerbForm (5)
    'VerbForm=Fin', 'VerbForm=Inf', 'VerbForm=Part', 'VerbForm=Conv', 'VerbForm=Ger',
    # Voice (3)
    'Voice=Act', 'Voice=Pass', 'Voice=Mid',
    # Polarity (1)
    'Polarity=Neg',
    # Variant (1)
    'Variant=Short',
    # PronType (6)
    'PronType=Prs', 'PronType=Dem', 'PronType=Rel', 'PronType=Int', 'PronType=Neg', 'PronType=Ind',
    # Poss (1)
    'Poss=Yes',
    # Reflex (1)
    'Reflex=Yes',
    # NumType (2)
    'NumType=Card', 'NumType=Ord',
    # Foreign (1)
    'Foreign=Yes',
]

# Dependency relations (comprehensive list)
DEP_RELATIONS = [
    'nsubj', 'nsubj:pass', 'obj', 'iobj', 'obl', 'vocative', 'expl', 'dislocated',
    'csubj', 'ccomp', 'xcomp', 'advcl', 'advmod', 'discourse',
    'aux', 'cop', 'mark', 'nmod', 'appos', 'nummod', 'nummod:gov', 'amod', 'det',
    'acl', 'acl:relcl', 'case', 'conj', 'cc', 'fixed', 'flat', 'parataxis',
    'orphan', 'root', 'punct'
]


def extract_tier3_grammar(poem: Dict) -> np.ndarray:
    """
    Extract Tier 3: Grammar features (100 features).

    17 POS + 48 morph + 35 deprel = 100 features

    Requires poem to have 'linguistic_features' field with:
    - pos_counts: dict like {'ADJ': 9, 'NOUN': 27, ...}
    - feats: dict of dicts like {'Case': {'Nom': 18, ...}, ...}
    - deprels: dict like {'amod': 9, 'nsubj': 5, ...}
    - word_count: int
    """
    lf = poem.get('linguistic_features', {})
    word_count = lf.get('word_count', 1) or 1

    # POS tag distribution (17 features) - normalized
    pos_counts = lf.get('pos_counts', {})
    total_pos = sum(pos_counts.values()) or 1
    pos_features = [pos_counts.get(tag, 0) / total_pos for tag in POS_TAGS]

    # Morphological features (48 features) - normalized counts
    feats = lf.get('feats', {})
    morph_features = []
    for morph_feat in MORPH_FEATURES:
        if '=' in morph_feat:
            category, value = morph_feat.split('=', 1)
            if category in feats and value in feats[category]:
                morph_features.append(feats[category][value] / word_count)
            else:
                morph_features.append(0.0)
        else:
            morph_features.append(0.0)

    # Dependency relations (35 features) - normalized
    deprels = lf.get('deprels', {})
    total_deps = sum(deprels.values()) or 1
    dep_features = [deprels.get(rel, 0) / total_deps for rel in DEP_RELATIONS]

    return np.array(pos_features + morph_features + dep_features)

TIER3_FEATURE_NAMES = (
    [f'pos_{tag}' for tag in POS_TAGS] +
    [f'morph_{f.replace("=", "_")}' for f in MORPH_FEATURES] +
    [f'dep_{rel}' for rel in DEP_RELATIONS]
)

# =============================================================================
# TIER 4: Prosody Features (15)
# =============================================================================

# Known meter types in Russian poetry
METER_TYPES = ['Я', 'Х', 'Д', 'Ан', 'Аф']  # Iamb, Trochee, Dactyl, Anapest, Amphibrach

def extract_tier4_prosody(poem: Dict) -> np.ndarray:
    """
    Extract Tier 4: Prosody features (15 features).

    Requires poem to have 'prosody' field with:
    - meter: meter type (Я, Х, Д, Ан, Аф, etc.)
    - feet: number of feet (2-6 typical)
    - clausula: ending pattern (жм, ж, м, etc.)
    - rhyme: rhyme scheme (перекрестная, парная, опоясывающая, etc.)
    - strophics: stanza size (4, 6, 8, etc.)
    - line_count: total lines in poem

    Features:
    1-5: meter one-hot (Я, Х, Д, Ан, Аф)
    6: feet count (normalized 2-6)
    7: is_alternating (жм clausula)
    8: is_feminine (ж clausula)
    9: is_masculine (м clausula)
    10: is_crossed_rhyme (абаб)
    11: is_paired_rhyme (аабб)
    12: is_enclosing_rhyme (абба)
    13: stanza_size (normalized)
    14: is_sonnet (14 lines)
    15: line_count (normalized by log)
    """
    prosody = poem.get('prosody', {})

    # Meter one-hot (5 features)
    meter = prosody.get('meter', '')
    # Handle complex meters (take first)
    if '#' in meter:
        meter = meter.split('#')[0].strip()
    meter_features = [1.0 if meter == m else 0.0 for m in METER_TYPES]

    # Feet count (1 feature)
    feet_str = str(prosody.get('feet', '0'))
    if '#' in feet_str:
        feet_str = feet_str.split('#')[0].strip()
    # Extract numeric part
    feet_num = ''
    for c in feet_str:
        if c.isdigit():
            feet_num += c
            break
    feet = int(feet_num) if feet_num else 0
    feet_normalized = feet / 6.0  # Normalize to 0-1 range (typical 2-6)

    # Clausula (3 features)
    clausula = prosody.get('clausula', '')
    is_alternating = 1.0 if 'жм' in clausula or 'мж' in clausula else 0.0
    is_feminine = 1.0 if 'ж' in clausula and 'м' not in clausula else 0.0
    is_masculine = 1.0 if 'м' in clausula and 'ж' not in clausula else 0.0

    # Rhyme scheme (3 features)
    rhyme = prosody.get('rhyme', '')
    is_crossed = 1.0 if 'перекрестн' in rhyme or 'абаб' in rhyme else 0.0
    is_paired = 1.0 if 'парн' in rhyme or 'аабб' in rhyme else 0.0
    is_enclosing = 1.0 if 'опоясыва' in rhyme or 'абба' in rhyme else 0.0

    # Stanza size (1 feature)
    strophics_str = str(prosody.get('strophics', '0'))
    if '#' in strophics_str:
        strophics_str = strophics_str.split('#')[0].strip()
    strophics = int(strophics_str) if strophics_str.isdigit() else 0
    stanza_normalized = strophics / 14.0  # Normalize (typical 4-14)

    # Is sonnet (14 lines, 1 feature)
    line_count_str = str(prosody.get('line_count', '0'))
    line_count = int(line_count_str) if line_count_str.isdigit() else 0
    is_sonnet = 1.0 if line_count == 14 else 0.0

    # Line count normalized (1 feature)
    line_count_normalized = np.log1p(line_count) / 5.0  # log scale

    return np.array([
        *meter_features,  # 5 features
        feet_normalized,  # 1 feature
        is_alternating, is_feminine, is_masculine,  # 3 features
        is_crossed, is_paired, is_enclosing,  # 3 features
        stanza_normalized,  # 1 feature
        is_sonnet,  # 1 feature
        line_count_normalized,  # 1 feature
    ])

TIER4_FEATURE_NAMES = [
    'meter_Я', 'meter_Х', 'meter_Д', 'meter_Ан', 'meter_Аф',
    'feet_normalized',
    'clausula_alternating', 'clausula_feminine', 'clausula_masculine',
    'rhyme_crossed', 'rhyme_paired', 'rhyme_enclosing',
    'stanza_normalized', 'is_sonnet', 'line_count_normalized'
]

# =============================================================================
# TIER 2: Content Features (40)
# =============================================================================

class Tier2Extractor:
    """
    Tier 2: Content/Topic feature extractor (40 features).

    Uses TF-IDF (20 features) + LDA topics (20 features).
    Must be fitted on training data only to avoid data leakage.

    Method:
    1. Fit TF-IDF on training texts (max 1000 features, min_df=5, max_df=0.8)
    2. Fit LDA on TF-IDF matrix (20 topics)
    3. For each poem: extract top-20 highest-variance TF-IDF features + 20 topic proportions
    """

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

    def fit(self, poems: List[Dict]):
        """Fit on training poems."""
        texts = [p.get('text', '') for p in poems]

        # Fit TF-IDF
        tfidf_matrix = self.tfidf.fit_transform(texts)

        # Find top-variance TF-IDF indices
        tfidf_var = np.var(tfidf_matrix.toarray(), axis=0)
        self.top_tfidf_idx = np.argsort(tfidf_var)[-self.n_tfidf:]

        # Fit LDA on TF-IDF
        self.lda.fit(tfidf_matrix)

        self.is_fitted = True
        return self

    def transform(self, poems: List[Dict]) -> np.ndarray:
        """Transform poems to Tier 2 features."""
        if not self.is_fitted:
            raise ValueError("Extractor must be fitted first")

        texts = [p.get('text', '') for p in poems]

        # TF-IDF features (top n_tfidf by variance)
        tfidf_matrix = self.tfidf.transform(texts).toarray()
        tfidf_features = tfidf_matrix[:, self.top_tfidf_idx]

        # LDA topic distributions
        lda_features = self.lda.transform(self.tfidf.transform(texts))

        return np.hstack([tfidf_features, lda_features])

    def fit_transform(self, poems: List[Dict]) -> np.ndarray:
        """Fit and transform in one step (for training data)."""
        self.fit(poems)
        return self.transform(poems)


# =============================================================================
# Combined Feature Extraction
# =============================================================================

def extract_all_features(
    poems: List[Dict],
    tier2_extractor: Optional[Tier2Extractor] = None
) -> Dict[str, np.ndarray]:
    """
    Extract all features for given poems.

    Args:
        poems: List of poem dicts
        tier2_extractor: Fitted Tier2Extractor (or None to skip Tier 2)

    Returns:
        Dict with keys: tier1, tier2, tier3, tier4, all
    """
    n = len(poems)

    # Tier 1: Surface
    tier1 = np.array([extract_tier1_surface(p) for p in poems])

    # Tier 2: Content (requires fitted extractor)
    if tier2_extractor is not None:
        tier2 = tier2_extractor.transform(poems)
    else:
        tier2 = np.zeros((n, TIER_DEFINITIONS['tier2_content']))

    # Tier 3: Grammar
    tier3 = np.array([extract_tier3_grammar(p) for p in poems])

    # Tier 4: Prosody
    tier4 = np.array([extract_tier4_prosody(p) for p in poems])

    # Combined
    all_features = np.hstack([tier1, tier2, tier3, tier4])

    return {
        'tier1': tier1,
        'tier2': tier2,
        'tier3': tier3,
        'tier4': tier4,
        'all': all_features
    }


def get_feature_names() -> Dict[str, List[str]]:
    """Get feature names for each tier."""
    return {
        'tier1': TIER1_FEATURE_NAMES,
        'tier3': TIER3_FEATURE_NAMES,
        'tier4': TIER4_FEATURE_NAMES,
    }
