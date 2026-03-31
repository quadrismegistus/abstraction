"""
Word norms: loading psycholinguistic concreteness/imageability ratings,
computing vector-based norms, and classifying words into semantic fields.
"""

import os
import shutil
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import (
    FIELD_DIR, SOURCE_DIR, ZCUT, PATH_NORMS, PATH_ALLNORMS, PATH_VECNORMS,
    REMOVE_STOPWORDS, BAD_SOURCES,
)
from .tokenize import get_stopwords, get_stopwords_and_names
from .utils import zfy, download_tqdm, read_df, save_df


# ---------------------------------------------------------------------------
# Adding norm series
# ---------------------------------------------------------------------------

def _add_series_to_norms(series, source, norms):
    series_z = zfy(series)
    for word, z in zip(series_z.index, series_z):
        if not isinstance(word, str) or not word or not word[0].isalpha():
            continue
        norms.append({"word": word, "score": series[word], "z": z, "source": source})


# ---------------------------------------------------------------------------
# Generating norms from published sources
# ---------------------------------------------------------------------------

def gen_norms_paivio(norms):
    path_pdf = os.path.join(SOURCE_DIR, "Paivio1968.pdf")
    path_csv = os.path.join(SOURCE_DIR, "Paivio1968.csv")
    if not os.path.exists(path_csv):
        from tabula import read_pdf
        dfs = read_pdf(path_pdf, pages="10-25")
        df = pd.concat(dfs)
        header = ["Noun", "IMAG_M", "x", "IMAG_SD", "CONC_M", "y", "CONC_SD",
                   "MEANP_M", "z", "MEANP_SD", "F"]
        df.columns = header + list(df.columns[len(header):])
        df["word"] = df.Noun.str.lower()
        df[["word", "IMAG_M", "IMAG_SD", "CONC_M", "CONC_SD",
            "MEANP_M", "MEANP_SD", "F"]].iloc[2:].to_csv(path_csv, index=False)
    df = pd.read_csv(path_csv).set_index("word")
    _add_series_to_norms(df.CONC_M, "Abs-Conc.PAV-Conc", norms)
    _add_series_to_norms(df.IMAG_M, "Abs-Conc.PAV-Imag", norms)


def gen_norms_mrc(norms):
    url = "https://ota.bodleian.ox.ac.uk/repository/xmlui/bitstream/handle/20.500.12024/1054/1054.zip?sequence=3&isAllowed=y"
    path_zip = os.path.join(SOURCE_DIR, "mrc.zip")
    path_dic = os.path.join(SOURCE_DIR, "mrc2.dct")
    if not os.path.exists(path_dic):
        download_tqdm(url, path_zip)
        from zipfile import ZipFile
        with ZipFile(path_zip) as zf:
            for member in zf.namelist():
                fn = os.path.basename(member)
                if fn == "mrc2.dct":
                    with zf.open(member) as src, open(path_dic, "wb") as dst:
                        shutil.copyfileobj(src, dst)
        os.remove(path_zip)

    parser = {"CONC": (29, 31), "IMAG": (32, 34), "AOA": (41, 43),
              "BROWN_FREQ": (22, 25), "FAM": (26, 28), "MEANC": (35, 37), "MEANP": (38, 40)}
    rows = []
    with open(path_dic) as f:
        for ln in f:
            pos = ln[44] if len(ln) > 44 else ""
            if pos not in {"N", "J", "V", "A"}:
                continue
            word = ln.strip().split()[-1].split("|")[0].strip().lower()
            irreg = ln[50] if len(ln) > 50 else ""
            if irreg == "N":
                word = word[1:]
            dx = {"word": word}
            for field, (start, stop) in parser.items():
                val = int(ln[start - 1 : stop])
                dx[field] = val if 100 <= val <= 700 else np.nan
            rows.append(dx)
    df = pd.DataFrame(rows).groupby("word").median().reset_index().set_index("word")
    _add_series_to_norms(df["CONC"], "Abs-Conc.MRC-Conc", norms)
    _add_series_to_norms(df["IMAG"], "Abs-Conc.MRC-Imag", norms)


def gen_norms_brysbaert(norms):
    url = "http://crr.ugent.be/papers/Concreteness_ratings_Brysbaert_et_al_BRM.txt"
    path = os.path.join(SOURCE_DIR, "Concreteness_ratings_Brysbaert_et_al_BRM.txt")
    if not os.path.exists(path):
        download_tqdm(url, path)
    df = pd.read_csv(path, sep="\t").set_index("Word")
    df["word"] = df.index
    df = df[df.word.apply(lambda x: isinstance(x, str) and x and " " not in x)]
    _add_series_to_norms(df["Conc.M"], "Abs-Conc.MT-Conc", norms)


def gen_norms_lsn(norms):
    url = "https://osf.io/48wsc/download"
    path = os.path.join(SOURCE_DIR, "Lancaster_sensorimotor_norms_for_39707_words.csv")
    if not os.path.exists(path):
        download_tqdm(url, path)
    df = pd.read_csv(path)
    df["Word"] = df.Word.str.lower()
    df = df[~df.Word.str.contains(" ")].set_index("Word")
    _add_series_to_norms(df["Visual.mean"], "Abs-Conc.LSN-Imag", norms)
    _add_series_to_norms(df["Haptic.mean"], "Abs-Conc.LSN-Hapt", norms)


def gen_orignorms():
    """Generate and save original (empirical) word norms from all sources."""
    os.makedirs(SOURCE_DIR, exist_ok=True)
    norms = []
    for func in tqdm([gen_norms_paivio, gen_norms_mrc, gen_norms_brysbaert, gen_norms_lsn],
                     desc="Building norms from sources"):
        func(norms)
    df = pd.DataFrame(norms).drop_duplicates(["word", "source"], keep="first")
    df = df.pivot(index="word", columns="source", values="z")
    os.makedirs(os.path.dirname(PATH_NORMS), exist_ok=True)
    df.to_csv(PATH_NORMS)


def get_orignorms(remove_stopwords=REMOVE_STOPWORDS):
    df = pd.read_csv(PATH_NORMS).set_index("word")
    if remove_stopwords:
        df = df.loc[[w for w in df.index if w not in get_stopwords_and_names()]]
    df["Abs-Conc.Median"] = df.median(axis=1)
    return df


# ---------------------------------------------------------------------------
# Fields: classify words as Abstract / Concrete / Neither
# ---------------------------------------------------------------------------

def get_contrasts(dfnorms, zcut=ZCUT):
    contrasts = []
    for col in dfnorms.columns:
        parts = col.split(".")
        contrast, source = parts[0], parts[1]
        period = parts[2] if len(parts) > 2 else "orig"
        neg, pos = contrast.split("-")
        series = dfnorms[col]
        pos_words = set(series[series >= zcut].index)
        neg_words = set(series[series <= -zcut].index)
        neither_words = set(series.dropna().index) - pos_words - neg_words
        contrasts.append({
            "contrast": contrast, "source": source, "period": period,
            "neg": neg_words, "pos": pos_words, "neither": neither_words,
        })
    return contrasts


def get_fields_from_norms(dfnorms, zcut=ZCUT, remove_stopwords=True):
    if remove_stopwords:
        dfnorms = dfnorms.loc[list(set(dfnorms.index) - get_stopwords_and_names())]
    fields = {}
    for cdx in get_contrasts(dfnorms, zcut=zcut):
        neg, pos = cdx["contrast"].split("-")
        period = "." + cdx["period"] if cdx["period"] != "orig" else ""
        prefix = f"{cdx['contrast']}.{cdx['source']}"
        fields[f"{prefix}.{neg}{period}"] = cdx["neg"]
        fields[f"{prefix}.{pos}{period}"] = cdx["pos"]
        fields[f"{prefix}.Neither{period}"] = cdx["neither"]
    return fields


def get_origfields():
    return get_fields_from_norms(get_orignorms())


def get_origcontrasts(remove_stopwords=REMOVE_STOPWORDS):
    df = get_orignorms(remove_stopwords=False)
    if remove_stopwords:
        exclude = get_stopwords_and_names()
        df = df.loc[[w for w in df.index if w not in exclude]]
    return get_contrasts(df)


# ---------------------------------------------------------------------------
# Classify abstract/concrete for a single z-score
# ---------------------------------------------------------------------------

def classify_word(z, zcut=ZCUT):
    if z >= zcut:
        return "Concrete"
    if z <= -zcut:
        return "Abstract"
    return "Neither"


# ---------------------------------------------------------------------------
# Norms as long-form DataFrame (for plotting)
# ---------------------------------------------------------------------------

NORM_SOURCE_ORDER = [
    "PAV-Conc", "MRC-Conc", "MT-Conc", "PAV-Imag", "MRC-Imag",
    "LSN-Imag", "LSN-Hapt", "LSN-Aud", "LSN-Perc", "LSN-Sens", "Median",
    "orig", "C20", "C19", "C18", "C17", "C16", "median",
]


def format_norms_as_long(dfnorms, zcut=ZCUT):
    rows = []
    for col in dfnorms.columns:
        parts = col.split(".")
        contrast, source = parts[0], parts[1]
        period = parts[2] if len(parts) > 2 else ""
        source_label = f"{source}.{period}" if period else source
        source_type = "Conc" if source.split("-")[-1] == "Conc" else "Imag"
        series = dfnorms[col].dropna()
        for word, z in zip(series.index, series):
            rows.append({
                "word": word, "z": z, "source": source_label,
                "source_type": source_type, "decision": classify_word(z, zcut),
                "order": NORM_SOURCE_ORDER.index(source_label) if source_label in NORM_SOURCE_ORDER else 0,
            })
    return pd.DataFrame(rows).sort_values("z")


# ---------------------------------------------------------------------------
# Vector-based norms (from historical Word2Vec models)
# ---------------------------------------------------------------------------

def get_vecnorms(remove_stopwords=REMOVE_STOPWORDS):
    df = pd.read_csv(PATH_VECNORMS).set_index("word")
    if remove_stopwords:
        df = df.loc[[w for w in df.index if w not in get_stopwords_and_names()]]
    # add median across periods for each contrast.source group
    colgroups = defaultdict(set)
    for col in df.columns:
        if col.count(".") != 2:
            continue
        contrast, source, _period = col.split(".")
        colgroups[f"{contrast}.{source}"] |= {col}
    for group, cols in colgroups.items():
        df[f"{group}.median"] = df[list(cols)].median(axis=1)
    return df


def get_allnorms(remove_stopwords=REMOVE_STOPWORDS, force=False):
    if not force and os.path.exists(PATH_ALLNORMS):
        return read_df(PATH_ALLNORMS)
    orig = get_orignorms(remove_stopwords=remove_stopwords)
    orig.columns = [c + ".orig" for c in orig.columns]
    vec = get_vecnorms(remove_stopwords=remove_stopwords)
    combined = vec.join(orig, how="outer")
    save_df(combined, PATH_ALLNORMS)
    return combined


def get_allcontrasts(remove_stopwords=REMOVE_STOPWORDS):
    df = get_allnorms(remove_stopwords=False)
    if remove_stopwords:
        exclude = get_stopwords_and_names()
        df = df.loc[[w for w in df.index if w not in exclude]]
    return get_contrasts(df)


def get_allfields():
    return get_fields_from_norms(get_allnorms())


# ---------------------------------------------------------------------------
# Correlation stats
# ---------------------------------------------------------------------------

def corr_norms(dfnorms):
    from scipy.stats import pearsonr
    corr_r = dfnorms.corr()
    corr_p = dfnorms.corr(method=lambda x, y: pearsonr(x, y)[1])
    r_melt = corr_r.reset_index().melt("index").set_index(["index", "variable"])
    p_melt = corr_p.reset_index().melt("index").set_index(["index", "variable"])
    cordf = r_melt.join(p_melt, rsuffix="_p").reset_index()
    cordf = cordf[cordf["index"] < cordf["variable"]].sort_values("value")
    return cordf
