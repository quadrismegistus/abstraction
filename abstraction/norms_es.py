"""
Spanish word norms: Guasch et al. (2015) + optional EsPal (Duchon et al. 2013).

  - Guasch et al. (2015): 1,400 words, concreteness + imageability, 1-9 scale.
    Data: BRM supplementary, https://doi.org/10.3758/s13428-015-0684-y
    File: data/fields/sources/es_norms/Guasch2015.xlsx
    Columns: Word, CON_M (concreteness), IMA_M (imageability)

  - EsPal (Duchon et al. 2013): 6,326 words, concreteness + imageability, 1-7 scale.
    Only loaded if data/fields/sources/es_norms/EsPal.xlsx exists.
    Columns: Word, Concreteness_M, Imageability_M  (update _ESPAL_*_COL if different)

Source labels: GUA-Conc, GUA-Imag, ESP-Conc, ESP-Imag
Mirrors English/French/German schema exactly.
"""

import os
from collections import defaultdict

import pandas as pd

from .config import (
    PATH_ESPAL, PATH_GUASCH, PATH_NORMS_ES,
    PATH_VECNORMS_ES, PATH_ALLNORMS_ES, ZCUT,
)
from .norms import _add_series_to_norms, get_contrasts, classify_word
from .utils import read_df, save_df


_NLTK_STOPWORDS_ES = None


# NOTE on `remove_stopwords` semantics across languages: English (`norms.py`)
# filters against a curated ~180K-entry stopwords+names list (function words,
# honorifics, proper names), lowercasing both the list and the norms index
# before comparing. There is no equivalent list for Spanish -- this module only
# filters the ~200-word NLTK Spanish function-word list, so far fewer
# non-content words are excluded here than in English. Building an 180K-scale
# list for Spanish is out of scope; treat `remove_stopwords=True` results as
# NOT directly comparable in coverage across languages.
#
# Words in the orig-norms sources are already lowercased at load time, but
# vector-norm vocabularies come straight from corpus text and can include
# capitalized/sentence-initial forms (e.g. "El", "Y"); NLTK's Spanish stopword
# list is all-lowercase. We therefore lowercase the index only for the
# membership test below, so capitalized function words aren't missed.
def get_nltk_stopwords_es():
    global _NLTK_STOPWORDS_ES
    if _NLTK_STOPWORDS_ES is None:
        from nltk.corpus import stopwords as _sw
        _NLTK_STOPWORDS_ES = frozenset(_sw.words("spanish"))
    return _NLTK_STOPWORDS_ES


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

# EsPal column names as published in BRM supplementary material.
# If your download uses different names, update these constants.
_ESPAL_WORD_COL = "Word"
_ESPAL_CONC_COL = "Concreteness_M"
_ESPAL_IMAG_COL = "Imageability_M"


def load_espal():
    """EsPal (Duchon et al. 2013) — concreteness + imageability for ~6,326 Spanish words.

    Scale: 1-7 (1=abstract/unimaginable, 7=concrete/imaginable).
    """
    df = pd.read_excel(PATH_ESPAL)
    df["word"] = df[_ESPAL_WORD_COL].astype(str).str.strip().str.lower()
    df = df[df["word"].str.len() > 0].drop_duplicates("word").set_index("word")
    return df


_GUASCH_WORD_COL = "Word"
_GUASCH_CONC_COL = "CON_M"
_GUASCH_IMAG_COL = "IMA_M"


def load_guasch():
    """Guasch et al. (2015) — concreteness + imageability for 1,400 Spanish words (1-9 scale)."""
    df = pd.read_excel(PATH_GUASCH)
    df["word"] = df[_GUASCH_WORD_COL].astype(str).str.strip().str.lower()
    df = df[df["word"].str.len() > 0].drop_duplicates("word").set_index("word")
    return df


# ---------------------------------------------------------------------------
# Norm generation
# ---------------------------------------------------------------------------

def gen_orignorms_es():
    """Generate and save Spanish original (empirical) word norms from Guasch (+ EsPal if present)."""
    norms = []

    guasch = load_guasch()
    _add_series_to_norms(guasch[_GUASCH_CONC_COL].dropna(), "Abs-Conc.GUA-Conc", norms)
    _add_series_to_norms(guasch[_GUASCH_IMAG_COL].dropna(), "Abs-Conc.GUA-Imag", norms)

    if os.path.exists(PATH_ESPAL):
        espal = load_espal()
        _add_series_to_norms(espal[_ESPAL_CONC_COL].dropna(), "Abs-Conc.ESP-Conc", norms)
        if _ESPAL_IMAG_COL in espal.columns:
            _add_series_to_norms(espal[_ESPAL_IMAG_COL].dropna(), "Abs-Conc.ESP-Imag", norms)

    df = pd.DataFrame(norms).drop_duplicates(["word", "source"], keep="first")
    df = df.pivot(index="word", columns="source", values="z")
    os.makedirs(os.path.dirname(PATH_NORMS_ES), exist_ok=True)
    df.to_csv(PATH_NORMS_ES)
    return df


def get_orignorms_es(remove_stopwords=True, force=False):
    if force or not os.path.exists(PATH_NORMS_ES):
        gen_orignorms_es()
    df = pd.read_csv(PATH_NORMS_ES).set_index("word")
    if remove_stopwords:
        df = df[~df.index.str.lower().isin(get_nltk_stopwords_es())]
    df["Abs-Conc.Median"] = df.median(axis=1)
    return df


# ---------------------------------------------------------------------------
# Contrasts and fields
# ---------------------------------------------------------------------------

def get_origcontrasts_es(remove_stopwords=True):
    """Build Spanish abstract/concrete contrast word sets from EsPal + Guasch."""
    df = get_orignorms_es(remove_stopwords=False)
    if remove_stopwords:
        df = df[~df.index.str.lower().isin(get_nltk_stopwords_es())]
    return get_contrasts(df)


def classify_word_es(z, zcut=ZCUT):
    return classify_word(z, zcut=zcut)


# ---------------------------------------------------------------------------
# Allnorms (orig + optional vecnorms)
# ---------------------------------------------------------------------------

def get_vecnorms_es(remove_stopwords=True):
    df = pd.read_pickle(PATH_VECNORMS_ES)
    if remove_stopwords:
        df = df[~df.index.str.lower().isin(get_nltk_stopwords_es())]
    colgroups = defaultdict(set)
    for col in df.columns:
        if col.count(".") != 2:
            continue
        contrast, source, _period = col.split(".")
        colgroups[f"{contrast}.{source}"] |= {col}
    for group, cols in colgroups.items():
        df[f"{group}.median"] = df[list(cols)].median(axis=1)
    return df


def get_allnorms_es(remove_stopwords=True, force=False):
    if not force and os.path.exists(PATH_ALLNORMS_ES):
        df = read_df(PATH_ALLNORMS_ES)
        if remove_stopwords:
            df = df[~df.index.str.lower().isin(get_nltk_stopwords_es())]
        return df

    orig = get_orignorms_es(remove_stopwords=False)
    orig.columns = [c + ".orig" for c in orig.columns]

    if os.path.exists(PATH_VECNORMS_ES):
        vec = get_vecnorms_es(remove_stopwords=False)
        combined = vec.join(orig, how="outer")
    else:
        combined = orig

    save_df(combined, PATH_ALLNORMS_ES)
    if remove_stopwords:
        combined = combined[~combined.index.str.lower().isin(get_nltk_stopwords_es())]
    return combined


def gen_vecnorms_es(model_dir=None, bin_year_by=100, num_proc=1):
    from .config import PATH_MODELS_ES
    from .models import gen_vecnorms
    gen_vecnorms(
        bin_year_by=bin_year_by,
        num_proc=num_proc,
        model_dir=model_dir or PATH_MODELS_ES,
        contrasts=get_origcontrasts_es(),
        output_path=PATH_VECNORMS_ES,
        regenerate_allnorms=False,
    )
    get_allnorms_es(force=True)
