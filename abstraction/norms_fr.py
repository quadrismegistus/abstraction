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

from collections import defaultdict

from .config import (
    PATH_BONIN, PATH_DESROCHERS, PATH_NORMS_FR,
    PATH_VECNORMS_FR, PATH_ALLNORMS_FR, ZCUT,
)
from .utils import read_df, save_df
from .norms import _add_series_to_norms, get_contrasts, classify_word
from .utils import zfy


_NLTK_STOPWORDS_FR = None


# NOTE on `remove_stopwords` semantics across languages: English (`norms.py`)
# filters against a curated ~180K-entry stopwords+names list (function words,
# honorifics, proper names), lowercasing both the list and the norms index
# before comparing. There is no equivalent list for French -- this module only
# filters the ~200-word NLTK French function-word list, so far fewer
# non-content words are excluded here than in English. Building an 180K-scale
# list for French is out of scope; treat `remove_stopwords=True` results as NOT
# directly comparable in coverage across languages.
#
# Words in the orig-norms sources are already lowercased at load time, but
# vector-norm vocabularies come straight from corpus text and can include
# capitalized/sentence-initial forms (e.g. "Le", "Et"); NLTK's French stopword
# list is all-lowercase. We therefore lowercase the index only for the
# membership test below, so capitalized function words aren't missed.
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
        df = df[~df.index.str.lower().isin(get_nltk_stopwords_fr())]
    df["Abs-Conc.Median"] = df.median(axis=1)
    return df


# ---------------------------------------------------------------------------
# Contrasts and fields
# ---------------------------------------------------------------------------

def get_origcontrasts_fr(remove_stopwords=True):
    """Build French abstract/concrete contrast word sets from Bonin + Desrochers."""
    df = get_orignorms_fr(remove_stopwords=False)
    if remove_stopwords:
        df = df[~df.index.str.lower().isin(get_nltk_stopwords_fr())]
    return get_contrasts(df)


def classify_word_fr(z, zcut=ZCUT):
    return classify_word(z, zcut=zcut)


# ---------------------------------------------------------------------------
# Allnorms-shaped output (for score_corpus_freqs)
# ---------------------------------------------------------------------------

def get_vecnorms_fr(remove_stopwords=True):
    """Return French vector-based norms, with period-median columns added."""
    import pandas as pd
    df = pd.read_pickle(PATH_VECNORMS_FR)
    if remove_stopwords:
        df = df[~df.index.str.lower().isin(get_nltk_stopwords_fr())]
    colgroups = defaultdict(set)
    for col in df.columns:
        if col.count(".") != 2:
            continue
        contrast, source, _period = col.split(".")
        colgroups[f"{contrast}.{source}"] |= {col}
    for group, cols in colgroups.items():
        df[f"{group}.median"] = df[list(cols)].median(axis=1)
    return df


def get_allnorms_fr(remove_stopwords=True, force=False):
    """Return French norms in the same schema as `get_allnorms()`.

    Columns have a '.orig' suffix for orig norms and '.{period}' for vec norms.
    If no vec norms exist on disk, returns orig-only.
    """
    import os
    import pandas as pd

    if not force and os.path.exists(PATH_ALLNORMS_FR):
        df = read_df(PATH_ALLNORMS_FR)
        if remove_stopwords:
            df = df[~df.index.str.lower().isin(get_nltk_stopwords_fr())]
        return df

    orig = get_orignorms_fr(remove_stopwords=False)
    orig.columns = [c + ".orig" for c in orig.columns]

    if os.path.exists(PATH_VECNORMS_FR):
        vec = get_vecnorms_fr(remove_stopwords=False)
        combined = vec.join(orig, how="outer")
    else:
        combined = orig

    save_df(combined, PATH_ALLNORMS_FR)
    if remove_stopwords:
        combined = combined[~combined.index.str.lower().isin(get_nltk_stopwords_fr())]
    return combined


def gen_vecnorms_fr(model_dir=None, bin_year_by=100, num_proc=1):
    """Generate French vecnorms from trained French Word2Vec models.

    Uses `get_origcontrasts_fr()` for contrast word sets. Writes to
    PATH_VECNORMS_FR. Does NOT touch English allnorms.
    """
    from .config import PATH_MODELS_FR
    from .models import gen_vecnorms
    gen_vecnorms(
        bin_year_by=bin_year_by,
        num_proc=num_proc,
        model_dir=model_dir or PATH_MODELS_FR,
        contrasts=get_origcontrasts_fr(),
        output_path=PATH_VECNORMS_FR,
        regenerate_allnorms=False,
    )
    # Regenerate French allnorms
    get_allnorms_fr(force=True)
