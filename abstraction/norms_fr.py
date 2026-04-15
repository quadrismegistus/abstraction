"""
French word norms: Bonin (2018) concreteness + Desrochers & Thompson (2009) imageability.

Mirrors the schema of `norms.py` so downstream scoring/plotting code works unchanged.
Source labels: BON-Conc (concreteness), DES-Imag (imageability).
Auxiliary ratings (Bonin context/valence/arousal, Desrochers frequency) loaded but
not part of the abstract/concrete contrast by default.
"""

import os

import numpy as np
import pandas as pd

from .config import (
    PATH_BONIN, PATH_DESROCHERS, PATH_NORMS_FR, ZCUT,
)
from .norms import _add_series_to_norms, get_contrasts, classify_word
from .utils import zfy


_NLTK_STOPWORDS_FR = None


def get_nltk_stopwords_fr():
    """Return NLTK French stopwords as a frozenset. Cached after first call."""
    global _NLTK_STOPWORDS_FR
    if _NLTK_STOPWORDS_FR is None:
        from nltk.corpus import stopwords as _sw
        _NLTK_STOPWORDS_FR = frozenset(_sw.words("french"))
    return _NLTK_STOPWORDS_FR


# ---------------------------------------------------------------------------
# Loaders for individual sources
# ---------------------------------------------------------------------------

def load_bonin():
    """Bonin et al. (2018) — concreteness/context/valence/arousal for 1,659 French words."""
    df = pd.read_excel(PATH_BONIN, sheet_name="Norms")
    df["word"] = df["items"].astype(str).str.strip().str.lower()
    df = df[df["word"].str.len() > 0].drop_duplicates("word").set_index("word")
    return df


def load_desrochers():
    """Desrochers & Thompson (2009) — imageability and subjective frequency for 3,600 French nouns."""
    df = pd.read_excel(PATH_DESROCHERS, sheet_name="Normative Data")
    df["word"] = df["NOUN"].astype(str).str.strip().str.lower()
    df = df[df["word"].str.len() > 0].drop_duplicates("word").set_index("word")
    return df


# ---------------------------------------------------------------------------
# Norm generation
# ---------------------------------------------------------------------------

def gen_norms_bonin(norms):
    df = load_bonin()
    _add_series_to_norms(df["conc_mean"], "Abs-Conc.BON-Conc", norms)


def gen_norms_desrochers(norms):
    df = load_desrochers()
    _add_series_to_norms(df["IMAGE_Mean"], "Abs-Conc.DES-Imag", norms)


def gen_orignorms_fr():
    """Generate and save French original (empirical) word norms from all sources."""
    norms = []
    gen_norms_bonin(norms)
    gen_norms_desrochers(norms)
    df = pd.DataFrame(norms).drop_duplicates(["word", "source"], keep="first")
    df = df.pivot(index="word", columns="source", values="z")
    os.makedirs(os.path.dirname(PATH_NORMS_FR), exist_ok=True)
    df.to_csv(PATH_NORMS_FR)
    return df


def get_orignorms_fr(remove_stopwords=True, force=False):
    if force or not os.path.exists(PATH_NORMS_FR):
        gen_orignorms_fr()
    df = pd.read_csv(PATH_NORMS_FR).set_index("word")
    if remove_stopwords:
        df = df[~df.index.isin(get_nltk_stopwords_fr())]
    df["Abs-Conc.Median"] = df.median(axis=1)
    return df


# ---------------------------------------------------------------------------
# Contrasts and fields
# ---------------------------------------------------------------------------

def get_origcontrasts_fr(remove_stopwords=True):
    """Build French abstract/concrete contrast word sets from Bonin + Desrochers."""
    df = get_orignorms_fr(remove_stopwords=False)
    if remove_stopwords:
        df = df[~df.index.isin(get_nltk_stopwords_fr())]
    return get_contrasts(df)


def classify_word_fr(z, zcut=ZCUT):
    return classify_word(z, zcut=zcut)


# ---------------------------------------------------------------------------
# Allnorms-shaped output (for score_corpus_freqs)
# ---------------------------------------------------------------------------

def get_allnorms_fr(remove_stopwords=True):
    """Return French norms in the same schema as `get_allnorms()`.

    Columns have a '.orig' suffix so scoring's _pct_*_orig logic engages.
    No vector-norm periods (C16–C21) — add once French Word2Vec models exist.
    """
    df = get_orignorms_fr(remove_stopwords=remove_stopwords)
    df.columns = [c + ".orig" for c in df.columns]
    return df
