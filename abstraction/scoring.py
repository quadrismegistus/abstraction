"""
Text scoring and passage analysis utilities.
"""

import json
import os

import numpy as np
import pandas as pd
from scipy.stats import zscore
from tqdm import tqdm

from .config import COUNT_DIR, PSGS_DIR, SCORES_DIR, PATH_CORPORA
from .corpus import load_corpus
from .norms import get_allnorms
from .tokenize import tokenize_agnostic
from .counting import count_absconc, count_absconc_path
from .utils import read_df, save_df


# ---------------------------------------------------------------------------
# Simple text scoring
# ---------------------------------------------------------------------------

_NORM_DICTS = {}


def get_norm_dict(col="Abs-Conc.Median.median"):
    if col not in _NORM_DICTS:
        _NORM_DICTS[col] = get_allnorms()[col].dropna().to_dict()
    return _NORM_DICTS[col]


def score_psg(txt, col="Abs-Conc.Median.median"):
    """Score a passage's mean concreteness (negative = abstract, positive = concrete)."""
    scores = get_norm_dict(col)
    toks = [x for x in tokenize_agnostic(txt.lower()) if x in scores]
    return sum(scores[x] for x in toks) / len(toks) if toks else np.nan


# ---------------------------------------------------------------------------
# Frequency-based scoring (for pre-computed word frequency files)
# ---------------------------------------------------------------------------

def score_freqs(freqs, col="Abs-Conc.Median.median"):
    """Score from a word frequency dict {word: count, ...}.

    Returns the count-weighted mean concreteness score. Useful for corpora
    that store pre-tokenized frequency files (JSON) rather than raw text.
    """
    scores = get_norm_dict(col)
    total_score = 0.0
    total_count = 0
    for word, count in freqs.items():
        word = word.lower()
        if word in scores:
            total_score += scores[word] * count
            total_count += count
    return total_score / total_count if total_count else np.nan


def score_freqs_file(path, col="Abs-Conc.Median.median"):
    """Score a JSON word-frequency file."""
    with open(path) as f:
        freqs = json.load(f)
    return score_freqs(freqs, col=col)


# ---------------------------------------------------------------------------
# Word-level scoring (per-token scores for visualization)
# ---------------------------------------------------------------------------

def score_words(txt, col="Abs-Conc.Median.median"):
    """Tokenize a text and return a DataFrame with per-word concreteness scores.

    Each row is a token with its position, the raw token, and its z-score
    (NaN if the word is not in the norm vocabulary). Useful for density plots
    and color-coded passage rendering.
    """
    scores = get_norm_dict(col)
    tokens = tokenize_agnostic(txt.lower())
    rows = []
    for i, tok in enumerate(tokens):
        if tok and tok[0].isalpha():
            rows.append({
                "position": i,
                "word": tok,
                "score": scores.get(tok, np.nan),
            })
    df = pd.DataFrame(rows)
    if len(df):
        df["is_abstract"] = df["score"] <= -1.0
        df["is_concrete"] = df["score"] >= 1.0
    return df


# ---------------------------------------------------------------------------
# Passage-level analysis
# ---------------------------------------------------------------------------

def _binz(z, zcut=1):
    if z >= zcut:
        return "Abs"
    if z <= -zcut:
        return "Conc"
    return "Neither"


def _biny(y, by=100, miny=1500, offy=1400):
    return y // by * by if y >= miny else offy


def get_all_passages(corpus_name="CanonFiction"):
    """Load passage-level counts with z-scores and year bins."""
    df = pd.read_csv(os.path.join(COUNT_DIR, f"data.absconc.{corpus_name}.psgs.v5.csv.gz"))
    df["abs-conc"] = df["num_abs"] - df["num_conc"]
    for k in ["abs-conc", "num_abs", "num_conc", "num_neither"]:
        df[f"{k}_z"] = zscore(df[k])
    df["zbin"] = df["abs-conc_z"].apply(_binz)
    meta = load_corpus(corpus_name).metadata
    id2year = dict(zip(meta["id"], meta["year"]))
    df["year"] = df["id"].map(id2year)
    df["ybin"] = df["year"].apply(_biny)
    return df


def sample_passages(df, sample_by=("ybin", "zbin"), n=100):
    return (
        df[df["zbin"].notna()]
        .groupby(list(sample_by))
        .sample(n=n, replace=True)
        .drop_duplicates()
        .sample(frac=1)
    )


# ---------------------------------------------------------------------------
# Book-level passage generation
# ---------------------------------------------------------------------------

def gen_bookpassages(corpus_name, text_id, sources=None, periods=None, save=False):
    """Generate passage-level counts for a single text."""
    if sources is None:
        sources = {"Median"}
    if periods is None:
        periods = {"median"}
    corpus = load_corpus(corpus_name)
    path = corpus.text_path(text_id)
    ld = count_absconc_path(
        path, sources=sources, periods=periods,
        incl_psg=True, modernize=True,
    )
    df = pd.DataFrame(ld)
    if len(df):
        df["abs-conc"] = df["num_abs"] - df["num_conc"]
        df["id"] = text_id
        df = df.merge(corpus.metadata, on="id", how="left")
    if save:
        save_bookpassages(df, text_id)
    return df


def save_bookpassages(df, fname, stacklen=100):
    """Save passages as markdown files for reading/annotation."""
    odir = os.path.join(PSGS_DIR, f"psgs_{fname}")
    os.makedirs(odir, exist_ok=True)
    units = []
    done = 0
    for i, row in df.iterrows():
        psg = row["passage"].replace("\\\\", "\n").strip()
        while "\n\n" in psg:
            psg = psg.replace("\n\n", "\n")
        psg = psg.replace("\n", "\n>\t")
        abs_conc = row["num_abs"] - row["num_conc"]
        units.append(f"\n\n\n> {psg}\n")
        if len(units) >= stacklen or i == len(df) - 1:
            done += 1
            ofn = os.path.join(odir, f"psgs_{fname}_{str(done).zfill(4)}.md")
            with open(ofn, "w") as f:
                f.write("\n\n".join(units))
            units = []


# ---------------------------------------------------------------------------
# Corpus-level frequency scoring (score all freqs JSONs across corpora)
# ---------------------------------------------------------------------------

def _walk_freqs(freqs_dir):
    """Yield (text_id, path) for every .json file under freqs_dir, recursively."""
    for root, _dirs, files in os.walk(freqs_dir):
        for fn in files:
            if fn.endswith(".json"):
                path = os.path.join(root, fn)
                text_id = os.path.relpath(path, freqs_dir).removesuffix(".json")
                yield text_id, path


def _score_freqs_allnorms(path, allnorms):
    """Score a single freqs JSON file against all norm columns at once.

    Returns a dict of {norm_col: weighted_mean_score} or empty dict on error.
    """
    try:
        with open(path) as f:
            freqs = json.load(f)
    except Exception:
        return {}
    if not freqs:
        return {}
    words = list(freqs.keys())
    counts = np.array([freqs[w] for w in words])
    words_lower = [w.lower() for w in words]
    matched = allnorms.reindex(words_lower)
    notna = matched.notna()
    weighted = matched.mul(counts, axis=0)
    col_counts = notna.mul(counts, axis=0).sum()
    col_sums = weighted.sum()
    scores = col_sums / col_counts
    return scores[col_counts > 0].to_dict()


def score_corpus_freqs(corpus_dir, allnorms=None):
    """Score all freqs/*.json files in a corpus directory against all norms.

    Parameters
    ----------
    corpus_dir : str
        Path to a corpus directory containing a freqs/ subdirectory.
    allnorms : DataFrame, optional
        Pre-loaded allnorms DataFrame. Loaded automatically if not provided.

    Returns
    -------
    DataFrame with 'id' column plus one column per norm, or empty DataFrame
    if no freqs/ directory exists.
    """
    freqs_dir = os.path.join(corpus_dir, "freqs")
    if not os.path.isdir(freqs_dir):
        return pd.DataFrame()
    if allnorms is None:
        allnorms = get_allnorms()
    allnorms = allnorms[allnorms.index.notna() & ~allnorms.index.duplicated()]
    rows = []
    for text_id, path in _walk_freqs(freqs_dir):
        scores = _score_freqs_allnorms(path, allnorms)
        if scores:
            scores["id"] = text_id
            rows.append(scores)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    cols = ["id"] + [c for c in df.columns if c != "id"]
    return df[cols]


def score_all_corpora(
    corpora_dir=PATH_CORPORA,
    output_dir=SCORES_DIR,
    force=False,
):
    """Score all corpora that have freqs/ folders.

    Saves one parquet per corpus to output_dir. Deduplicates corpora
    whose freqs/ directories resolve to the same real path (e.g. hathi
    subcorpora sharing a symlinked freqs/ tree).

    Parameters
    ----------
    corpora_dir : str
        Parent directory containing corpus subdirectories.
    output_dir : str
        Where to save per-corpus score files.
    force : bool
        If False, skip corpora that already have a saved score file.

    Returns
    -------
    dict of {corpus_name: DataFrame}
    """
    os.makedirs(output_dir, exist_ok=True)
    allnorms = get_allnorms()
    allnorms = allnorms[allnorms.index.notna() & ~allnorms.index.duplicated()]

    # discover corpora with freqs/ dirs, dedup by realpath
    seen_realpaths = {}
    corpora = []
    for name in sorted(os.listdir(corpora_dir)):
        corpus_dir = os.path.join(corpora_dir, name)
        freqs_dir = os.path.join(corpus_dir, "freqs")
        if not os.path.isdir(freqs_dir):
            continue
        real = os.path.realpath(freqs_dir)
        if real in seen_realpaths:
            print(f"  Skipping {name}/freqs/ (same as {seen_realpaths[real]})")
            continue
        seen_realpaths[real] = name
        corpora.append((name, corpus_dir))

    # collect all (corpus_name, text_id, path) triples for a single progress bar
    all_files = []
    skip_corpora = set()
    for name, corpus_dir in corpora:
        out_path = os.path.join(output_dir, f"{name}.pkl")
        if not force and os.path.exists(out_path):
            skip_corpora.add(name)
            continue
        freqs_dir = os.path.join(corpus_dir, "freqs")
        for text_id, path in _walk_freqs(freqs_dir):
            all_files.append((name, text_id, path))

    results = {}
    for name in skip_corpora:
        out_path = os.path.join(output_dir, f"{name}.pkl")
        print(f"  {name}: already scored, loading")
        results[name] = read_df(out_path)

    # score all files with a single progress bar
    corpus_rows = {}
    pbar = tqdm(all_files, desc="Scoring", unit="file")
    for name, text_id, path in pbar:
        pbar.set_postfix_str(name, refresh=False)
        scores = _score_freqs_allnorms(path, allnorms)
        if scores:
            scores["id"] = text_id
            corpus_rows.setdefault(name, []).append(scores)

    # assemble and save per-corpus DataFrames
    for name, corpus_dir in corpora:
        if name in skip_corpora:
            continue
        rows = corpus_rows.get(name, [])
        if rows:
            df = pd.DataFrame(rows)
            cols = ["id"] + [c for c in df.columns if c != "id"]
            df = df[cols]
            out_path = os.path.join(output_dir, f"{name}.pkl")
            save_df(df, out_path)
            print(f"  {name}: scored {len(df)} texts")
        else:
            df = pd.DataFrame()
            print(f"  {name}: no valid freqs files")
        results[name] = df
    return results


def printpsg(row):
    """Format a passage row as a markdown string."""
    psg = row["passage"].replace("\\\\", "\n").strip()
    while "\n\n" in psg:
        psg = psg.replace("\n\n", "\n")
    psg = psg.replace("\n", "\n>\t").replace("''", '"')
    return f"""
> ... {psg} ...
>
> -- {row.get("author", "Unknown")}, _{row.get("title", "Unknown")}_ ({row.get("year", "?")})
>    - Abstract words ({row['num_abs']}), Concrete words ({row['num_conc']})
>    - Abs - Conc = {row.get('abs-conc', '?')}
"""
