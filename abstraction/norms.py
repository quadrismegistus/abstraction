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
    PATH_ICNORMS, REMOVE_STOPWORDS, BAD_SOURCES,
)
from .tokenize import get_stopwords, get_stopwords_and_names
from .utils import zfy, download_tqdm, read_df, save_df


_NLTK_STOPWORDS = None

# ---------------------------------------------------------------------------
# Allnorms artifact fingerprint (cache invalidation; audit §4.1-4.4)
# ---------------------------------------------------------------------------

# Fingerprints cached by (path, mtime_ns, size) so a stat() is the only cost
# per call — the 425MB pickle is never loaded just to fingerprint.
_NORMS_VERSION_CACHE = {}


def _allnorms_path(lang="en"):
    """Path to the allnorms artifact for a language code ('en'/'fr'/'de'/'es')."""
    if lang == "en":
        return PATH_ALLNORMS
    from .config import PATH_ALLNORMS_FR, PATH_ALLNORMS_DE, PATH_ALLNORMS_ES
    paths = {"fr": PATH_ALLNORMS_FR, "de": PATH_ALLNORMS_DE, "es": PATH_ALLNORMS_ES}
    if lang not in paths:
        raise ValueError(f"Unknown allnorms language {lang!r} (expected en/fr/de/es)")
    return paths[lang]


def _cheap_allnorms_columns(path):
    """Column names of the artifact, but only when readable without a full load.

    Feather/parquet expose their schema cheaply via pyarrow; pickle (the
    current .pkl.gz format) does not, so we return None rather than load it.
    """
    try:
        if path.endswith(".parquet"):
            import pyarrow.parquet as pq
            return list(pq.read_schema(path).names)
        if path.endswith(".feather"):
            import pyarrow as pa
            with pa.memory_map(path) as source:
                return list(pa.ipc.open_file(source).schema.names)
    except Exception:
        return None
    return None


def norms_version(lang="en"):
    """Short stable fingerprint (12 hex chars) of the current allnorms artifact.

    Built from the allnorms file's (path, mtime, size) — plus its column
    names when they are cheaply readable (feather/parquet schema; never a
    full pickle load). Regenerating the allnorms file changes its mtime/size
    and therefore the fingerprint, letting downstream caches (freqs_cache.db,
    trajectory cache, etc.) self-invalidate.

    A missing file yields a deterministic fingerprint of the path alone, so
    callers on data-less machines (e.g. CI) never crash here.
    """
    import hashlib
    path = _allnorms_path(lang)
    try:
        st = os.stat(path)
        stat_key = (path, st.st_mtime_ns, st.st_size)
    except OSError:
        stat_key = (path, None, None)
    cached = _NORMS_VERSION_CACHE.get(stat_key)
    if cached is not None:
        return cached
    payload = f"{stat_key[0]}|{stat_key[1]}|{stat_key[2]}"
    cols = _cheap_allnorms_columns(path) if stat_key[1] is not None else None
    if cols is not None:
        payload += "|" + ",".join(map(str, cols))
    version = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    _NORMS_VERSION_CACHE[stat_key] = version
    return version


def get_nltk_stopwords():
    """Return NLTK English stopwords as a frozenset. Cached after first call."""
    global _NLTK_STOPWORDS
    if _NLTK_STOPWORDS is None:
        from nltk.corpus import stopwords as _sw
        _NLTK_STOPWORDS = frozenset(_sw.words("english"))
    return _NLTK_STOPWORDS


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
        exclude = get_stopwords_and_names()
        df = df[~df.index.str.lower().isin(exclude)]
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
        if "-" not in contrast:
            # not a contrast column (e.g. IC.* information-content norms)
            continue
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
        dfnorms = dfnorms[~dfnorms.index.str.lower().isin(get_stopwords_and_names())]
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
    """Build contrast word sets (abstract/concrete/neither) from empirical norms.

    When remove_stopwords=True, filters only NLTK function words — not names
    or the broader stopwords.txt. Content words like 'servant', 'body' etc.
    participate in defining the abstract/concrete axis.
    """
    df = get_orignorms(remove_stopwords=False)
    if remove_stopwords:
        df = df[~df.index.isin(get_nltk_stopwords())]
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
    df = pd.read_pickle(PATH_VECNORMS)
    if remove_stopwords:
        exclude = get_stopwords_and_names()
        df = df[~df.index.str.lower().isin(exclude)]
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


def gen_ic_norms(model_dir=None):
    """Generate IC (information content) norms from word2vec vocab.txt files.

    For each corpus-period vocab, computes IC = log2(total / count) in bits.
    Aggregates: median across corpora within century, then cross-century median.
    Saves to PATH_ICNORMS.
    """
    from .config import PATH_MODELS
    if model_dir is None:
        model_dir = PATH_MODELS

    PERIOD_MAP = {
        "1500-1600": "C16", "1600-1700": "C17", "1700-1800": "C18",
        "1800-1900": "C19", "1900-2000": "C20", "2000-2100": "C21",
    }

    all_series = {}
    for corpus_dir in sorted(os.listdir(model_dir)):
        cp = os.path.join(model_dir, corpus_dir)
        if not os.path.isdir(cp):
            continue
        for period_dir in sorted(os.listdir(cp)):
            plabel = PERIOD_MAP.get(period_dir)
            if not plabel:
                continue
            vpath = os.path.join(cp, period_dir, "run_01", "vocab.txt")
            if not os.path.isfile(vpath):
                continue
            vocab, total = {}, 0
            with open(vpath) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        vocab[parts[0]] = int(parts[1])
                        total += int(parts[1])
            col = f"IC.{corpus_dir}.{plabel}"
            ic = {w: np.log2(total / c) for w, c in vocab.items()}
            all_series[col] = pd.Series(ic, dtype="float32")

    ic_df = pd.DataFrame(all_series)
    for plabel in sorted(set(PERIOD_MAP.values())):
        cols = [c for c in ic_df.columns if c.endswith(f".{plabel}")]
        if cols:
            ic_df[f"IC.Median.{plabel}"] = ic_df[cols].median(axis=1)
    med_cols = sorted(c for c in ic_df.columns if c.startswith("IC.Median.C"))
    ic_df["IC.Median.median"] = ic_df[med_cols].median(axis=1)
    save_df(ic_df, PATH_ICNORMS)
    return ic_df


def get_ic_norms(remove_stopwords=REMOVE_STOPWORDS, force=False):
    if force or not os.path.exists(PATH_ICNORMS):
        df = gen_ic_norms()
    else:
        df = read_df(PATH_ICNORMS)
    if remove_stopwords:
        exclude = get_stopwords_and_names()
        df = df[~df.index.str.lower().isin(exclude)]
    return df


def get_allnorms(remove_stopwords=REMOVE_STOPWORDS, force=False):
    if not force and os.path.exists(PATH_ALLNORMS):
        df = read_df(PATH_ALLNORMS)
        if remove_stopwords:
            exclude = get_stopwords_and_names()
            df = df[~df.index.str.lower().isin(exclude)]
        return df
    # Always save the FULL unfiltered table — filtering happens on read
    orig = get_orignorms(remove_stopwords=False)
    orig.columns = [c + ".orig" for c in orig.columns]
    vec = get_vecnorms(remove_stopwords=False)
    ic = get_ic_norms(remove_stopwords=False)
    combined = vec.join(orig, how="outer").join(ic, how="outer")
    save_df(combined, PATH_ALLNORMS)
    if remove_stopwords:
        exclude = get_stopwords_and_names()
        combined = combined[~combined.index.str.lower().isin(exclude)]
    return combined


def get_allcontrasts(remove_stopwords=REMOVE_STOPWORDS):
    """Build contrast word sets from all norms (empirical + vector).

    Filters only NLTK function words for contrast construction.
    """
    df = get_allnorms(remove_stopwords=False)
    if remove_stopwords:
        df = df[~df.index.isin(get_nltk_stopwords())]
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
