"""
German word norms: human-rated concreteness and imageability from multiple sources.

Concreteness sources (from Conde et al. 2026 spreadsheet):
  - Charbonnier & Wartena (2020): 4,008 words, 7-point (composite of Baschek+Wippich+Lahl+Kanske)
  - Lahl et al. (2009): 2,365 nouns, 11-point
  - Kanske & Kotz (2010): 971 nouns, 9-point

Imageability sources:
  - Võ et al. (2006): 2,826 words, 7-point
  - Schmidtke et al. (2014): 1,034 words, 9-point (from separate file + Conde)
  - Schröter & Schroeder (2017): 1,152 words, 7-point
  - Grandy et al. (2020): 2,559 nouns, 0-100 scale (average of young+old raters)

All columns independently z-scored, median taken across sources.
"""

import os
from collections import defaultdict

import pandas as pd

from .config import (
    PATH_CONDE, PATH_SCHMIDTKE, PATH_NORMS_DE,
    PATH_VECNORMS_DE, PATH_ALLNORMS_DE, ZCUT,
)
from .norms import _add_series_to_norms, get_contrasts, classify_word
from .utils import read_df, save_df


_NLTK_STOPWORDS_DE = None


# NOTE on `remove_stopwords` semantics across languages: English (`norms.py`)
# filters against a curated ~180K-entry stopwords+names list (function words,
# honorifics, proper names), lowercasing both the list and the norms index
# before comparing. There is no equivalent list for German -- this module only
# filters the ~200-word NLTK German function-word list, so far fewer non-content
# words are excluded here than in English. Building an 180K-scale list for
# German is out of scope; treat `remove_stopwords=True` results as NOT directly
# comparable in coverage across languages.
#
# Case handling is also language-specific: NLTK's German stopword list is
# all-lowercase function words, but German nouns are legitimately capitalized
# (and vector-norm vocabularies come straight from corpus text, so they retain
# real capitalization). We therefore lowercase the index only for the
# membership test below -- never the stored index/word column -- so a
# sentence-initial "Und" is correctly recognized as the stopword "und" while a
# capitalized noun like "Stein" is never mistaken for one.
def get_nltk_stopwords_de():
    global _NLTK_STOPWORDS_DE
    if _NLTK_STOPWORDS_DE is None:
        from nltk.corpus import stopwords as _sw
        _NLTK_STOPWORDS_DE = frozenset(_sw.words("german"))
    return _NLTK_STOPWORDS_DE


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_conde():
    """Load Conde et al. (2026) spreadsheet, return DataFrame indexed by lowercase word."""
    df = pd.read_excel(PATH_CONDE, sheet_name="Sheet1")
    df["word"] = df["word"].astype(str).str.strip().str.lower()
    df = df[df["word"].str.len() > 0].drop_duplicates("word").set_index("word")
    return df


def load_schmidtke():
    """Load Schmidtke et al. (2014) imageability ratings."""
    df = pd.read_excel(PATH_SCHMIDTKE, sheet_name="anew_ger")
    df["word"] = df["word"].astype(str).str.strip().str.lower()
    df = df[df["word"].str.len() > 0].drop_duplicates("word").set_index("word")
    return df


# ---------------------------------------------------------------------------
# Norm generation
# ---------------------------------------------------------------------------

_HUMAN_CONCRETENESS = {
    "Abs-Conc.CHB-Conc": "Charb_concr_human",
    "Abs-Conc.LAH-Conc": "Lahl_concr_human",
    "Abs-Conc.KAN-Conc": "Kanske_concr_human",
}

_HUMAN_IMAGEABILITY = {
    "Abs-Conc.VO-Imag": "Vo_ima",
    "Abs-Conc.SCM-Imag": "Schmi_ima_human",
    "Abs-Conc.SCR-Imag": "Schrö_ima_human",
    "Abs-Conc.GRD-Imag": None,  # computed: average of Grandy young+old
}


def gen_orignorms_de():
    """Generate and save German original (empirical) word norms from all human sources."""
    conde = load_conde()
    schmidtke = load_schmidtke()
    norms = []

    for source, col in _HUMAN_CONCRETENESS.items():
        series = conde[col].dropna()
        if col == "Kanske_concr_human":
            series = -series
        _add_series_to_norms(series, source, norms)

    for source, col in _HUMAN_IMAGEABILITY.items():
        if source == "Abs-Conc.SCM-Imag":
            # Schmidtke et al. (2014) imageability. Conde's `Schmi_ima_human` column
            # is a verbatim subset of the standalone Schmidtke2014.xlsx file (raw
            # values identical on the overlap -- verified against the source data),
            # but the standalone file also has a handful of extra words absent from
            # Conde. Build ONE raw series over the union of both, then z-score once,
            # so every word in this column shares a single z-scale. (Previously the
            # Conde subset and the non-overlapping extras were z-scored separately
            # -- over different, differently-sized populations -- yet stamped with
            # the same "Abs-Conc.SCM-Imag" column label, silently mixing two
            # incompatible z-scales under one name.)
            base = conde[col].dropna()
            extra = schmidtke[~schmidtke.index.isin(base.index)]["IMA_MEAN"].dropna()
            series = pd.concat([base, extra])
            series = series[~series.index.duplicated(keep="first")]
        elif col is not None:
            series = conde[col].dropna()
        else:
            grandy = conde[["Grandy_ima_young", "Grandy_ima_old"]].mean(axis=1).dropna()
            series = grandy
        _add_series_to_norms(series, source, norms)

    df = pd.DataFrame(norms).drop_duplicates(["word", "source"], keep="first")
    df = df.pivot(index="word", columns="source", values="z")
    os.makedirs(os.path.dirname(PATH_NORMS_DE), exist_ok=True)
    df.to_csv(PATH_NORMS_DE)
    return df


def get_orignorms_de(remove_stopwords=True, force=False):
    if force or not os.path.exists(PATH_NORMS_DE):
        gen_orignorms_de()
    df = pd.read_csv(PATH_NORMS_DE).set_index("word")
    if remove_stopwords:
        df = df[~df.index.str.lower().isin(get_nltk_stopwords_de())]
    df["Abs-Conc.Median"] = df.median(axis=1)
    return df


# ---------------------------------------------------------------------------
# Contrasts and fields
# ---------------------------------------------------------------------------

def get_origcontrasts_de(remove_stopwords=True):
    df = get_orignorms_de(remove_stopwords=False)
    if remove_stopwords:
        df = df[~df.index.str.lower().isin(get_nltk_stopwords_de())]
    return get_contrasts(df)


def classify_word_de(z, zcut=ZCUT):
    return classify_word(z, zcut=zcut)


# ---------------------------------------------------------------------------
# Allnorms (orig + optional vecnorms)
# ---------------------------------------------------------------------------

def get_vecnorms_de(remove_stopwords=True):
    df = pd.read_pickle(PATH_VECNORMS_DE)
    if remove_stopwords:
        df = df[~df.index.str.lower().isin(get_nltk_stopwords_de())]
    colgroups = defaultdict(set)
    for col in df.columns:
        if col.count(".") != 2:
            continue
        contrast, source, _period = col.split(".")
        colgroups[f"{contrast}.{source}"] |= {col}
    for group, cols in colgroups.items():
        df[f"{group}.median"] = df[list(cols)].median(axis=1)
    return df


def get_allnorms_de(remove_stopwords=True, force=False):
    if not force and os.path.exists(PATH_ALLNORMS_DE):
        df = read_df(PATH_ALLNORMS_DE)
        if remove_stopwords:
            df = df[~df.index.str.lower().isin(get_nltk_stopwords_de())]
        return df

    orig = get_orignorms_de(remove_stopwords=False)
    orig.columns = [c + ".orig" for c in orig.columns]

    if os.path.exists(PATH_VECNORMS_DE):
        vec = get_vecnorms_de(remove_stopwords=False)
        combined = vec.join(orig, how="outer")
    else:
        combined = orig

    save_df(combined, PATH_ALLNORMS_DE)
    if remove_stopwords:
        combined = combined[~combined.index.str.lower().isin(get_nltk_stopwords_de())]
    return combined


def gen_vecnorms_de(model_dir=None, bin_year_by=100, num_proc=1):
    from .config import PATH_MODELS_DE
    from .models import gen_vecnorms
    gen_vecnorms(
        bin_year_by=bin_year_by,
        num_proc=num_proc,
        model_dir=model_dir or PATH_MODELS_DE,
        contrasts=get_origcontrasts_de(),
        output_path=PATH_VECNORMS_DE,
        regenerate_allnorms=False,
    )
    get_allnorms_de(force=True)
