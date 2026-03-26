"""
Text scoring and passage analysis utilities.
"""

import json
import os

import numpy as np
import pandas as pd
from scipy.stats import zscore
from tqdm import tqdm

from .config import COUNT_DIR, DIST_DIR, PSGS_DIR, SCORES_DIR, PATH_CORPORA
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


def _get_csv_columns(allnorms):
    """Return the canonical column order for score CSVs."""
    return ["id"] + sorted(allnorms.columns.tolist())


def _load_done_ids(csv_path):
    """Read the 'id' column from an existing score CSV, or return empty set."""
    if not os.path.exists(csv_path):
        return set()
    try:
        return set(pd.read_csv(csv_path, usecols=["id"])["id"])
    except Exception:
        return set()


def score_corpus_freqs(corpus_dir, allnorms=None, output_path=None):
    """Score all freqs/*.json files in a corpus directory against all norms.

    If output_path is provided, appends rows to a CSV incrementally and
    skips IDs that are already present. This makes the function resumable.

    Parameters
    ----------
    corpus_dir : str
        Path to a corpus directory containing a freqs/ subdirectory.
    allnorms : DataFrame, optional
        Pre-loaded allnorms DataFrame. Loaded automatically if not provided.
    output_path : str, optional
        Path to a CSV file for incremental output. If None, returns results
        in memory only.

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
    columns = _get_csv_columns(allnorms)

    # check what's already done
    done_ids = _load_done_ids(output_path) if output_path else set()

    # open CSV for appending
    csv_file = None
    writer = None
    if output_path:
        file_exists = os.path.exists(output_path) and os.path.getsize(output_path) > 0
        csv_file = open(output_path, "a", newline="")
        import csv
        writer = csv.DictWriter(csv_file, fieldnames=columns, extrasaction="ignore", restval="")
        if not file_exists:
            writer.writeheader()

    rows = []
    try:
        for text_id, path in _walk_freqs(freqs_dir):
            if text_id in done_ids:
                continue
            scores = _score_freqs_allnorms(path, allnorms)
            if scores:
                scores["id"] = text_id
                rows.append(scores)
                if writer:
                    writer.writerow(scores)
    finally:
        if csv_file:
            csv_file.close()

    if not rows and not done_ids:
        return pd.DataFrame()

    # return full DataFrame (existing + new)
    if output_path and done_ids:
        return pd.read_csv(output_path)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df[columns]


def score_all_corpora(
    corpora_dir=PATH_CORPORA,
    output_dir=SCORES_DIR,
    force=False,
):
    """Score all corpora that have freqs/ folders.

    Saves one CSV per corpus to output_dir/v7/, appending incrementally.
    Resumable: skips already-scored text IDs within each corpus.
    Deduplicates corpora whose freqs/ directories resolve to the same
    real path (e.g. hathi subcorpora sharing a symlinked freqs/ tree).

    Parameters
    ----------
    corpora_dir : str
        Parent directory containing corpus subdirectories.
    output_dir : str
        Where to save per-corpus score files.
    force : bool
        If True, delete existing CSVs and re-score from scratch.

    Returns
    -------
    dict of {corpus_name: DataFrame}
    """
    output_dir = os.path.join(output_dir, "v7")
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

    # collect all (corpus_name, text_id, path) triples, skipping done IDs
    all_files = []
    done_counts = {}
    for name, corpus_dir in corpora:
        out_path = os.path.join(output_dir, f"{name}.csv")
        if force and os.path.exists(out_path):
            os.remove(out_path)
        done_ids = _load_done_ids(out_path)
        done_counts[name] = len(done_ids)
        freqs_dir = os.path.join(corpus_dir, "freqs")
        for text_id, path in _walk_freqs(freqs_dir):
            if text_id not in done_ids:
                all_files.append((name, text_id, path))
        if done_ids:
            print(f"  {name}: {len(done_ids)} already done, {sum(1 for n, _, _ in all_files if n == name)} remaining")

    # open one CSV writer per corpus
    import csv
    columns = _get_csv_columns(allnorms)
    writers = {}
    file_handles = {}

    def get_writer(name):
        if name not in writers:
            out_path = os.path.join(output_dir, f"{name}.csv")
            file_exists = os.path.exists(out_path) and os.path.getsize(out_path) > 0
            fh = open(out_path, "a", newline="")
            w = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
            if not file_exists:
                w.writeheader()
            writers[name] = w
            file_handles[name] = fh
        return writers[name]

    # score with a single progress bar
    pbar = tqdm(all_files, desc="Scoring", unit="file")
    try:
        for name, text_id, path in pbar:
            pbar.set_postfix_str(name, refresh=False)
            scores = _score_freqs_allnorms(path, allnorms)
            if scores:
                scores["id"] = text_id
                get_writer(name).writerow(scores)
    finally:
        for fh in file_handles.values():
            fh.close()

    # load and return results
    results = {}
    for name, _ in corpora:
        out_path = os.path.join(output_dir, f"{name}.csv")
        if os.path.exists(out_path):
            try:
                results[name] = pd.read_csv(out_path)
            except Exception as e:
                print(f"  Warning: could not read {name}.csv: {e}")
                results[name] = pd.DataFrame()
        else:
            results[name] = pd.DataFrame()
    return results


# ---------------------------------------------------------------------------
# Z-score distribution counting (CDF bins per text per norm)
# ---------------------------------------------------------------------------

DEFAULT_BIN_EDGES = np.round(np.arange(-3.0, 3.05, 0.1), 1)


def _count_freqs_allnorms(path, allnorms, bin_edges):
    """Compute cumulative z-score distributions for a single freqs JSON.

    For each norm column, bins words by z-score weighted by frequency,
    then returns cumulative proportions at each bin edge.

    Returns a dict like {"{norm}_cdf_{edge}": proportion, ...}.
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

    result = {}
    for col in allnorms.columns:
        vals = matched[col].values
        mask = np.isfinite(vals)
        if mask.sum() == 0:
            continue
        w = counts[mask].astype(float)
        v = vals[mask]
        # Histogram with bin_edges; values outside range go into first/last bins
        hist, _ = np.histogram(v, bins=bin_edges, weights=w)
        total = w.sum()
        cdf = np.cumsum(hist) / total
        for edge, prop in zip(bin_edges[1:], cdf):
            result[f"{col}_cdf_{edge:.1f}"] = round(float(prop), 6)
    return result


def _get_cdf_columns(allnorms, bin_edges):
    """Return the canonical column order for CDF CSVs."""
    cols = ["id"]
    for norm_col in sorted(allnorms.columns):
        for edge in bin_edges[1:]:
            cols.append(f"{norm_col}_cdf_{edge:.1f}")
    return cols


def count_corpus_freqs(corpus_dir, allnorms=None, output_path=None,
                       bin_edges=None, norm_filter=None):
    """Count z-score distributions for all freqs/*.json in a corpus.

    Mirrors score_corpus_freqs but outputs cumulative proportions in
    z-score bins instead of weighted-mean scores.

    Parameters
    ----------
    corpus_dir : str
        Corpus directory with a freqs/ subdirectory.
    allnorms : DataFrame, optional
        Pre-loaded allnorms. Loaded automatically if None.
    output_path : str, optional
        CSV path for incremental output.
    bin_edges : array-like, optional
        Z-score bin edges. Default: -3.0 to 3.0 in 0.1 steps.
    norm_filter : list of str, optional
        If given, only count these norm columns (e.g. ["Abs-Conc.Median.median"]).

    Returns
    -------
    DataFrame with 'id' column plus CDF columns, or empty DataFrame.
    """
    freqs_dir = os.path.join(corpus_dir, "freqs")
    if not os.path.isdir(freqs_dir):
        return pd.DataFrame()
    if allnorms is None:
        allnorms = get_allnorms()
    allnorms = allnorms[allnorms.index.notna() & ~allnorms.index.duplicated()]
    if norm_filter:
        allnorms = allnorms[[c for c in allnorms.columns if c in norm_filter]]
    if bin_edges is None:
        bin_edges = DEFAULT_BIN_EDGES
    bin_edges = np.asarray(bin_edges)
    columns = _get_cdf_columns(allnorms, bin_edges)

    done_ids = _load_done_ids(output_path) if output_path else set()

    csv_file = None
    writer = None
    if output_path:
        import csv as csv_mod
        file_exists = os.path.exists(output_path) and os.path.getsize(output_path) > 0
        csv_file = open(output_path, "a", newline="")
        writer = csv_mod.DictWriter(csv_file, fieldnames=columns,
                                    extrasaction="ignore", restval="")
        if not file_exists:
            writer.writeheader()

    rows = []
    try:
        for text_id, path in _walk_freqs(freqs_dir):
            if text_id in done_ids:
                continue
            cdfs = _count_freqs_allnorms(path, allnorms, bin_edges)
            if cdfs:
                cdfs["id"] = text_id
                rows.append(cdfs)
                if writer:
                    writer.writerow(cdfs)
    finally:
        if csv_file:
            csv_file.close()

    if not rows and not done_ids:
        return pd.DataFrame()
    if output_path and done_ids:
        return pd.read_csv(output_path)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)[columns]


def count_all_corpora(
    corpora_dir=PATH_CORPORA,
    output_dir=None,
    force=False,
    norm_filter=None,
    bin_edges=None,
):
    """Count z-score distributions for all corpora with freqs/ folders.

    Mirrors score_all_corpora. Saves one CSV per corpus to output_dir/v1/.

    Parameters
    ----------
    corpora_dir : str
        Parent directory containing corpus subdirectories.
    output_dir : str, optional
        Where to save per-corpus count files. Default: DIST_DIR.
    force : bool
        If True, delete existing CSVs and re-count from scratch.
    norm_filter : list of str, optional
        Only count these norm columns.
    bin_edges : array-like, optional
        Z-score bin edges. Default: -3.0 to 3.0 in 0.1 steps.
    """
    if output_dir is None:
        output_dir = COUNT_DIR
    output_dir = os.path.join(output_dir, "v1")
    os.makedirs(output_dir, exist_ok=True)

    allnorms = get_allnorms()
    allnorms = allnorms[allnorms.index.notna() & ~allnorms.index.duplicated()]
    if norm_filter:
        allnorms = allnorms[[c for c in allnorms.columns if c in norm_filter]]
    if bin_edges is None:
        bin_edges = DEFAULT_BIN_EDGES
    bin_edges = np.asarray(bin_edges)
    columns = _get_cdf_columns(allnorms, bin_edges)

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

    # collect all (corpus_name, text_id, path), skipping done IDs
    all_files = []
    for name, corpus_dir in corpora:
        out_path = os.path.join(output_dir, f"{name}.csv")
        if force and os.path.exists(out_path):
            os.remove(out_path)
        done_ids = _load_done_ids(out_path)
        freqs_dir = os.path.join(corpus_dir, "freqs")
        for text_id, path in _walk_freqs(freqs_dir):
            if text_id not in done_ids:
                all_files.append((name, text_id, path))
        if done_ids:
            print(f"  {name}: {len(done_ids)} already done, "
                  f"{sum(1 for n, _, _ in all_files if n == name)} remaining")

    # open one CSV writer per corpus
    import csv as csv_mod
    writers = {}
    file_handles = {}

    def get_writer(name):
        if name not in writers:
            out_path = os.path.join(output_dir, f"{name}.csv")
            file_exists = os.path.exists(out_path) and os.path.getsize(out_path) > 0
            fh = open(out_path, "a", newline="")
            w = csv_mod.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
            if not file_exists:
                w.writeheader()
            writers[name] = w
            file_handles[name] = fh
        return writers[name]

    pbar = tqdm(all_files, desc="Counting", unit="file")
    try:
        for name, text_id, path in pbar:
            pbar.set_postfix_str(name, refresh=False)
            cdfs = _count_freqs_allnorms(path, allnorms, bin_edges)
            if cdfs:
                cdfs["id"] = text_id
                get_writer(name).writerow(cdfs)
    finally:
        for fh in file_handles.values():
            fh.close()

    results = {}
    for name, _ in corpora:
        out_path = os.path.join(output_dir, f"{name}.csv")
        if os.path.exists(out_path):
            try:
                results[name] = pd.read_csv(out_path)
            except Exception as e:
                print(f"  Warning: could not read {name}.csv: {e}")
                results[name] = pd.DataFrame()
        else:
            results[name] = pd.DataFrame()
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
