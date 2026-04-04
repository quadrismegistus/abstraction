"""
Text scoring and passage analysis utilities.
"""

import json
import os

import numpy as np
import pandas as pd
from scipy.stats import zscore
from tqdm import tqdm

import sqlite3

from .config import COUNT_DIR, DIST_DIR, PSGS_DIR, SCORES_DIR, PATH_CORPORA
from .corpus import load_corpus
from .norms import get_allnorms
from .tokenize import tokenize_agnostic, get_spelling_modernizer
from .counting import count_absconc, count_absconc_path
from .utils import read_df, save_df


# ---------------------------------------------------------------------------
# Freqs score cache (sqlite-backed, keyed by relative path + modernize flag)
# ---------------------------------------------------------------------------

FREQS_CACHE_PATH = os.path.join(SCORES_DIR, "freqs_cache.db")
_CORPORA_PREFIX = os.path.expanduser("~/lltk_data/corpora/")


def _freqs_relpath(abspath):
    """Convert absolute freqs path to relative key under ~/lltk_data/corpora/."""
    if abspath.startswith(_CORPORA_PREFIX):
        return abspath[len(_CORPORA_PREFIX):]
    return abspath


def _load_freqs_cache(modernize=False):
    """Load all cached scores into a dict: relpath -> scores_dict."""
    if not os.path.exists(FREQS_CACHE_PATH):
        return {}
    mod_int = 1 if modernize else 0
    conn = sqlite3.connect(FREQS_CACHE_PATH)
    try:
        rows = conn.execute(
            "SELECT freqs_key, scores_json FROM freqs_scores WHERE modernized = ?",
            (mod_int,),
        ).fetchall()
    except sqlite3.OperationalError:
        return {}
    finally:
        conn.close()
    cache = {}
    for key, sj in rows:
        cache[key] = json.loads(sj)
    return cache


def _save_freqs_cache(new_entries, modernize=False):
    """Write new cache entries to sqlite. new_entries: list of (relpath, scores_dict)."""
    if not new_entries:
        return
    os.makedirs(os.path.dirname(FREQS_CACHE_PATH), exist_ok=True)
    mod_int = 1 if modernize else 0
    conn = sqlite3.connect(FREQS_CACHE_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS freqs_scores (
            freqs_key TEXT NOT NULL,
            modernized INTEGER NOT NULL,
            scores_json TEXT NOT NULL,
            PRIMARY KEY (freqs_key, modernized)
        )
    """)
    conn.executemany(
        "INSERT OR REPLACE INTO freqs_scores (freqs_key, modernized, scores_json) VALUES (?, ?, ?)",
        [(k, mod_int, json.dumps(v)) for k, v in new_entries],
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Simple text scoring
# ---------------------------------------------------------------------------

_NORM_DICTS = {}


def get_norm_dict(col="Abs-Conc.Median.median"):
    if col not in _NORM_DICTS:
        _NORM_DICTS[col] = get_allnorms()[col].dropna().to_dict()
    return _NORM_DICTS[col]


def _modernize_score(word, norm_dict, spelling_d):
    """Look up word in norms, preferring modernized spelling when available.

    If the word has a modern form in the spelling dict AND that form is in
    the norms, uses the modern form's score. This ensures historical variants
    like "vertue" get "virtue"'s score rather than a truncated vecnorm median.
    Falls back to the raw word's score if no modernization applies.

    Returns (score, matched_word) or (None, None) if no match.
    """
    modern = spelling_d.get(word)
    if modern is not None and modern in norm_dict:
        return norm_dict[modern], modern
    if word in norm_dict:
        return norm_dict[word], word
    return None, None


def score_psg(txt, col="Abs-Conc.Median.median"):
    """Score a passage's mean concreteness (negative = abstract, positive = concrete)."""
    scores = get_norm_dict(col)
    spelling_d = get_spelling_modernizer()
    total, n = 0.0, 0
    for tok in tokenize_agnostic(txt.lower()):
        s, _ = _modernize_score(tok, scores, spelling_d)
        if s is not None:
            total += s
            n += 1
    return total / n if n else np.nan


# ---------------------------------------------------------------------------
# Frequency-based scoring (for pre-computed word frequency files)
# ---------------------------------------------------------------------------

def score_freqs(freqs, col="Abs-Conc.Median.median"):
    """Score from a word frequency dict {word: count, ...}.

    Returns the count-weighted mean concreteness score. Useful for corpora
    that store pre-tokenized frequency files (JSON) rather than raw text.
    """
    scores = get_norm_dict(col)
    spelling_d = get_spelling_modernizer()
    total_score = 0.0
    total_count = 0
    for word, count in freqs.items():
        word = word.lower()
        s, _ = _modernize_score(word, scores, spelling_d)
        if s is not None:
            total_score += s * count
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
    (NaN if the word is not in the norm vocabulary). Falls back to modernized
    spelling for unmatched words. Useful for density plots and color-coded
    passage rendering.
    """
    scores = get_norm_dict(col)
    spelling_d = get_spelling_modernizer()
    tokens = tokenize_agnostic(txt.lower())
    rows = []
    for i, tok in enumerate(tokens):
        if tok and tok[0].isalpha():
            s, _ = _modernize_score(tok, scores, spelling_d)
            rows.append({
                "position": i,
                "word": tok,
                "score": s if s is not None else np.nan,
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
        incl_psg=True, modernize=False,
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


def _modernize_word_list(words_lower, norm_index, spelling_d):
    """Modernize words, preferring modern form when available in norms."""
    result = []
    for w in words_lower:
        modern = spelling_d.get(w)
        if modern is not None and modern in norm_index:
            result.append(modern)
        else:
            result.append(w)
    return result


_NORMS_ARRAYS_CACHE = None


def _get_norms_arrays(allnorms):
    """Precompute numpy arrays + word→index dict for fast scoring.

    Returns (word2idx, values, columns) where:
    - word2idx: dict mapping word → row index
    - values: numpy float64 array (n_words × n_cols)
    - columns: list of column names
    """
    global _NORMS_ARRAYS_CACHE
    if _NORMS_ARRAYS_CACHE is not None:
        return _NORMS_ARRAYS_CACHE
    word2idx = {w: i for i, w in enumerate(allnorms.index)}
    values = allnorms.values.astype(np.float64)
    columns = allnorms.columns.tolist()
    _NORMS_ARRAYS_CACHE = (word2idx, values, columns)
    return _NORMS_ARRAYS_CACHE


def _score_freqs_dict_allnorms(freqs, allnorms, spelling_d=None):
    """Score a word-frequency dict against all norm columns at once.

    Returns a dict of {norm_col: weighted_mean_score} or empty dict.
    Uses precomputed numpy arrays with O(1) word→index lookup.
    """
    if not freqs:
        return {}
    word2idx, values, columns = _get_norms_arrays(allnorms)

    # Build matched indices and counts
    indices = []
    counts = []
    for word, count in freqs.items():
        w = word.lower()
        if spelling_d:
            mod = spelling_d.get(w)
            if mod and mod in word2idx:
                w = mod
        idx = word2idx.get(w)
        if idx is not None:
            indices.append(idx)
            counts.append(count)

    if not indices:
        return {}

    # Vectorized scoring: extract matched rows, multiply by counts
    matched = values[indices]  # (n_matched × n_cols)
    counts_arr = np.array(counts, dtype=np.float64)[:, np.newaxis]  # (n_matched × 1)
    notna = ~np.isnan(matched)
    weighted = np.where(notna, matched * counts_arr, 0.0)
    col_counts = np.where(notna, counts_arr, 0.0).sum(axis=0)
    col_sums = weighted.sum(axis=0)

    result = {}
    for i, col in enumerate(columns):
        if col_counts[i] > 0:
            result[col] = col_sums[i] / col_counts[i]
    return result


def _score_freqs_allnorms(path, allnorms, spelling_d=None):
    """Score a single freqs JSON file against all norm columns at once."""
    try:
        with open(path) as f:
            freqs = json.load(f)
    except Exception:
        return {}
    return _score_freqs_dict_allnorms(freqs, allnorms, spelling_d)


def _get_csv_columns(allnorms):
    """Return the canonical column order for score CSVs."""
    return ["id"] + sorted(allnorms.columns.tolist())


def _load_done_ids(csv_path):
    """Read the 'id' column from an existing score CSV, or return empty set."""
    if not os.path.exists(csv_path):
        return set()
    try:
        return set(pd.read_csv(csv_path, usecols=["id"], dtype={"id": str})["id"])
    except Exception:
        return set()


def score_corpus_freqs(corpus_dir, allnorms=None, output_path=None,
                       modernize=False):
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
    modernize : bool
        If True (default), prefer modernized spelling for norm lookups.

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
    spelling_d = get_spelling_modernizer() if modernize else None

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
            scores = _score_freqs_allnorms(path, allnorms, spelling_d)
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


def _version_dir(base, version, modernize):
    """Append version subdirectory, with '-raw' suffix when unmodernized."""
    suffix = "" if modernize else "-raw"
    return os.path.join(base, f"{version}{suffix}")


# ---------------------------------------------------------------------------
# Multiprocessing worker for scoring
# ---------------------------------------------------------------------------

# Module-level references inherited by forked workers
_worker_allnorms = None
_worker_spelling_d = None
_worker_min_words = 100


def _init_worker(allnorms, spelling_d, min_words, freqs_cache=None):
    """Initialize worker process with shared data."""
    global _worker_allnorms, _worker_spelling_d, _worker_min_words, _worker_freqs_cache
    _worker_allnorms = allnorms
    _worker_spelling_d = spelling_d
    _worker_min_words = min_words
    _worker_freqs_cache = freqs_cache or {}
    # Pre-build the norms arrays cache in this worker
    _get_norms_arrays(allnorms)


def _score_one_text(args):
    """Score a single text from its freqs path. Runs in worker process."""
    text_id, freqs_path = args
    try:
        with open(freqs_path) as f:
            freqs = json.load(f)
    except Exception:
        return None
    if sum(freqs.values()) < _worker_min_words:
        return None
    scores = _score_freqs_dict_allnorms(freqs, _worker_allnorms, _worker_spelling_d)
    if scores:
        scores["id"] = text_id
        return scores
    return None


def score_all_corpora(
    output_dir=SCORES_DIR,
    force=False,
    modernize=False,
    only=None,
    min_words=100,
    num_proc=1,
):
    """Score corpora using LLTK text objects for freqs access.

    Iterates LLTK corpora and their texts, scoring each text's freqs()
    against all norms. Saves one CSV per corpus, resumable.

    Parameters
    ----------
    output_dir : str
        Where to save per-corpus score files.
    force : bool
        If True, delete existing CSVs and re-score from scratch.
    modernize : bool
        If True, prefer modernized spelling for norm lookups.
        Outputs go to v8/ (modernized) or v8-raw/ (unmodernized).
    only : list of str, optional
        If provided, score only these corpora. If None, defaults to
        ARC_CORPORA. Pass only="all" to score every LLTK corpus with freqs.
    num_proc : int
        Number of parallel worker processes. Default 1 (no parallelism).
        Workers inherit allnorms via fork for zero-copy sharing.

    Returns
    -------
    dict of {corpus_name: DataFrame}
    """
    import csv
    import sys
    sys.path.insert(0, os.path.expanduser("~/github/lltk"))
    import lltk
    from .analysis import ARC_CORPORA, EXCLUDE_CORPORA

    output_dir = _version_dir(output_dir, "v8", modernize)
    os.makedirs(output_dir, exist_ok=True)
    allnorms = get_allnorms()
    allnorms = allnorms[allnorms.index.notna() & ~allnorms.index.duplicated()]
    columns = _get_csv_columns(allnorms)
    spelling_d = get_spelling_modernizer() if modernize else None

    # Pre-build norms arrays in main process (inherited by workers via fork)
    _get_norms_arrays(allnorms)

    # Resolve which corpora to include
    if only == "all":
        include = None
    elif only is not None:
        include = set(only)
    else:
        include = set(ARC_CORPORA)

    # Discover LLTK corpora
    skip = {"estc", "hathi", "test_fixture", "tmp", "BigHist"}
    # Only apply EXCLUDE_CORPORA when not explicitly naming corpora
    explicit = isinstance(only, list)
    corpus_list = []
    for corpus_name, corpus in lltk.corpora():
        cid = corpus.id
        if cid in skip:
            continue
        if not explicit and cid in EXCLUDE_CORPORA:
            continue
        if include is not None and cid not in include:
            continue
        corpus_list.append((cid, corpus))

    print(f"Scoring {len(corpus_list)} corpora (num_proc={num_proc})")

    # Set up multiprocessing pool if requested
    pool = None
    if num_proc > 1:
        import multiprocessing as mp
        # Use fork to inherit allnorms arrays (copy-on-write)
        ctx = mp.get_context("fork")
        pool = ctx.Pool(
            num_proc,
            initializer=_init_worker,
            initargs=(allnorms, spelling_d, min_words),
        )

    results = {}
    for cid, corpus in corpus_list:
        out_path = os.path.join(output_dir, f"{cid}.csv")
        if force and os.path.exists(out_path):
            os.remove(out_path)
        done_ids = _load_done_ids(out_path)

        # Collect (text_id, freqs_path) tuples from LLTK
        work_items = []
        for t in corpus.texts():
            if t.id in done_ids:
                continue
            pf = t.path_freqs
            if pf and os.path.exists(pf):
                work_items.append((t.id, pf))

        if not work_items and done_ids:
            print(f"  {cid}: {len(done_ids)} already done, 0 remaining")
            continue
        if not work_items:
            continue

        print(f"  {cid}: {len(done_ids)} done, {len(work_items)} to score")

        file_exists = os.path.exists(out_path) and os.path.getsize(out_path) > 0
        fh = open(out_path, "a", newline="")
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()

        try:
            if pool is not None:
                # Parallel scoring
                iterator = pool.imap_unordered(_score_one_text, work_items, chunksize=100)
            else:
                # Sequential scoring (same worker function for consistency)
                _init_worker(allnorms, spelling_d, min_words)
                iterator = map(_score_one_text, work_items)

            for result in tqdm(iterator, total=len(work_items), desc=f"  {cid}", unit="text"):
                if result is not None:
                    writer.writerow(result)
        finally:
            fh.close()

        try:
            results[cid] = pd.read_csv(out_path)
        except Exception:
            results[cid] = pd.DataFrame()

    if pool is not None:
        pool.close()
        pool.join()

    return results


# ---------------------------------------------------------------------------
# Scoring synthetic (arc) corpora
# ---------------------------------------------------------------------------

ARC_CORPUS_IDS = ["arc_fiction", "arc_poetry", "arc_periodical", "arc_essays"]


def _score_one_text_with_id(args):
    """Score a single text, returning (_id, source_corpus, scores, cache_new).

    Returns a dict with scores plus _id/source_corpus.  Also includes a
    '_cache_new' key: list of (relpath, scores_dict) for paths that were
    cache misses and need to be persisted.

    If multiple freqs_paths are given (from match group members), each is
    scored independently and the results are averaged per norm column.
    """
    _id, source_corpus, freqs_paths = args
    if isinstance(freqs_paths, str):
        freqs_paths = [freqs_paths]

    all_scores = []
    cache_new = []
    for fp in freqs_paths:
        relpath = _freqs_relpath(fp)
        cached = _worker_freqs_cache.get(relpath)
        if cached is not None:
            all_scores.append(cached)
            continue
        try:
            with open(fp) as f:
                freqs = json.load(f)
        except Exception:
            continue
        if sum(freqs.values()) < _worker_min_words:
            continue
        scores = _score_freqs_dict_allnorms(freqs, _worker_allnorms, _worker_spelling_d)
        if scores:
            all_scores.append(scores)
            cache_new.append((relpath, scores))

    if not all_scores:
        return None

    # Average across match group versions
    if len(all_scores) == 1:
        result = all_scores[0].copy()
    else:
        result = {}
        all_keys = set()
        for s in all_scores:
            all_keys.update(s.keys())
        for k in all_keys:
            vals = [s[k] for s in all_scores if k in s]
            result[k] = sum(vals) / len(vals)

    result["_id"] = _id
    result["source_corpus"] = source_corpus
    result["_cache_new"] = cache_new
    return result


def score_arc_corpora(
    output_dir=SCORES_DIR,
    force=False,
    modernize=False,
    only=None,
    min_words=100,
    num_proc=1,
):
    """Score synthetic arc corpora (arc_fiction, arc_poetry, etc.).

    Each synthetic corpus produces one CSV with _id and source_corpus columns.
    Texts are deduplicated by the SyntheticCorpus definition.

    Parameters
    ----------
    only : list of str, optional
        Which arc corpora to score. Default: all in ARC_CORPUS_IDS.
    """
    import csv
    import sys
    sys.path.insert(0, os.path.expanduser("~/github/lltk"))
    import lltk

    output_dir = _version_dir(output_dir, "v8", modernize)
    os.makedirs(output_dir, exist_ok=True)
    allnorms = get_allnorms()
    allnorms = allnorms[allnorms.index.notna() & ~allnorms.index.duplicated()]
    columns = ["_id", "source_corpus"] + sorted(allnorms.columns.tolist())
    spelling_d = get_spelling_modernizer() if modernize else None

    _get_norms_arrays(allnorms)

    # Load freqs score cache
    print("  Loading freqs score cache...", flush=True)
    freqs_cache = _load_freqs_cache(modernize=modernize)
    print(f"  Cache: {len(freqs_cache)} entries loaded", flush=True)

    include = set(only) if only else set(ARC_CORPUS_IDS)

    # Set up multiprocessing
    pool = None
    if num_proc > 1:
        import multiprocessing as mp
        ctx = mp.get_context("fork")
        pool = ctx.Pool(
            num_proc,
            initializer=_init_worker,
            initargs=(allnorms, spelling_d, min_words, freqs_cache),
        )
    else:
        _init_worker(allnorms, spelling_d, min_words, freqs_cache)

    results = {}
    for arc_id in sorted(include):
        print(f"  {arc_id}: loading corpus...", flush=True)
        corpus = lltk.load(arc_id)
        if corpus is None:
            print(f"  {arc_id}: not found in LLTK")
            continue

        out_path = os.path.join(output_dir, f"{arc_id}.csv")
        if force and os.path.exists(out_path):
            os.remove(out_path)
        done_ids = set()
        if os.path.exists(out_path):
            try:
                done_ids = set(pd.read_csv(out_path, usecols=["_id"], dtype={"_id": str})["_id"])
            except Exception:
                pass

        # Collect work items: (_id, source_corpus, freqs_paths)
        # Use match_group_texts to score all versions of a text and average.
        # Deduplicate by freqs set: if multiple texts from corpus.texts()
        # resolve to the same match group freqs, keep only the first (which
        # is the dedup winner by rank).
        print(f"  {arc_id}: collecting texts and match group freqs...", flush=True)
        work_items = []
        seen_freqs_sets = set()  # frozenset of freqs paths already queued
        n_skipped = 0
        n_no_freqs = 0
        n_match_group_extras = 0
        n_dedup_collapsed = 0
        for t in corpus.texts(progress=True):
            _id = t._id if hasattr(t, '_id') else f"_{t.corpus.id}/{t.id}"
            if _id in done_ids:
                n_skipped += 1
                continue
            # Gather freqs from all match group members
            freqs_paths = []
            try:
                for m in t.match_group_texts:
                    pf = getattr(m, 'path_freqs', None)
                    if pf and os.path.exists(pf):
                        freqs_paths.append(pf)
            except Exception:
                # Fallback to just this text
                pf = t.path_freqs
                if pf and os.path.exists(pf):
                    freqs_paths = [pf]
            if not freqs_paths:
                n_no_freqs += 1
                continue
            # Deduplicate: skip if we already have a work item with the same freqs
            freqs_key = frozenset(freqs_paths)
            if freqs_key in seen_freqs_sets:
                n_dedup_collapsed += 1
                continue
            seen_freqs_sets.add(freqs_key)
            if len(freqs_paths) > 1:
                n_match_group_extras += len(freqs_paths) - 1
            source = t.corpus.id if hasattr(t, 'corpus') and t.corpus else arc_id
            work_items.append((_id, source, freqs_paths))

        print(f"  {arc_id}: {len(work_items)} to score, "
              f"{n_skipped} already done, {n_no_freqs} without freqs, "
              f"{n_match_group_extras} extra match group versions, "
              f"{n_dedup_collapsed} match group duplicates collapsed", flush=True)

        if not work_items:
            continue

        file_exists = os.path.exists(out_path) and os.path.getsize(out_path) > 0
        fh = open(out_path, "a", newline="")
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()

        all_cache_new = []
        try:
            if pool is not None:
                iterator = pool.imap_unordered(_score_one_text_with_id, work_items, chunksize=100)
            else:
                iterator = map(_score_one_text_with_id, work_items)

            for result in tqdm(iterator, total=len(work_items), desc=f"  {arc_id}", unit="text"):
                if result is not None:
                    cache_new = result.pop("_cache_new", [])
                    all_cache_new.extend(cache_new)
                    writer.writerow(result)
        finally:
            fh.close()

        # Persist new cache entries
        if all_cache_new:
            print(f"  {arc_id}: caching {len(all_cache_new)} new freqs scores", flush=True)
            _save_freqs_cache(all_cache_new, modernize=modernize)

        try:
            results[arc_id] = pd.read_csv(out_path)
        except Exception:
            results[arc_id] = pd.DataFrame()

    if pool is not None:
        pool.close()
        pool.join()

    return results


# ---------------------------------------------------------------------------
# Z-score distribution counting (CDF bins per text per norm)
# ---------------------------------------------------------------------------

DEFAULT_BIN_EDGES = np.round(np.arange(-3.0, 3.05, 0.1), 1)


def _count_freqs_allnorms(path, allnorms, bin_edges, spelling_d=None):
    """Compute frequency-weighted z-score histograms for a single freqs JSON.

    For each norm column, bins words by z-score weighted by word frequency.
    Only non-zero bins are stored (sparse).

    Returns a dict like {norm_col: {"-1.0": 22, "-0.9": 33, ...}, ...}
    where values are raw frequency-weighted counts (integers).
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
    if spelling_d:
        words_lower = _modernize_word_list(words_lower, set(allnorms.index), spelling_d)
    matched = allnorms.reindex(words_lower)

    result = {}
    for col in allnorms.columns:
        vals = matched[col].values
        mask = np.isfinite(vals)
        if mask.sum() == 0:
            continue
        w = counts[mask].astype(float)
        v = vals[mask]
        hist, _ = np.histogram(v, bins=bin_edges, weights=w)
        # Store sparse: only bins with nonzero counts
        norm_hist = {}
        for edge, count in zip(bin_edges[1:], hist):
            c = int(round(count))
            if c > 0:
                norm_hist[f"{edge:.1f}"] = c
        if norm_hist:
            result[col] = norm_hist
    return result


def _load_done_jsonl(jsonl_path):
    """Load existing JSONL, return {id: {norm_set}} and list of all records."""
    records = {}
    if not os.path.exists(jsonl_path):
        return records, []
    lines = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            tid = rec.get("id")
            if tid is not None:
                records[tid] = rec
                lines.append(rec)
    return records, lines


def count_corpus_freqs(corpus_dir, allnorms=None, output_path=None,
                       bin_edges=None, norm_filter=None, modernize=False):
    """Count z-score distributions for all freqs/*.json in a corpus.

    Outputs one JSONL line per text:
        {"id": "...", "Abs-Conc.Median.median": {"-1.0": 0.19, ...}, ...}

    Resumable: skips texts already present with all requested norms.
    If re-run with additional norms, merges new norms into existing records.

    Parameters
    ----------
    corpus_dir : str
        Corpus directory with a freqs/ subdirectory.
    allnorms : DataFrame, optional
        Pre-loaded allnorms. Loaded automatically if None.
    output_path : str, optional
        JSONL path for incremental output.
    bin_edges : array-like, optional
        Z-score bin edges. Default: -3.0 to 3.0 in 0.1 steps.
    norm_filter : list of str, optional
        If given, only count these norm columns.
    modernize : bool
        If True (default), prefer modernized spelling for norm lookups.

    Returns
    -------
    List of dicts (one per text).
    """
    freqs_dir = os.path.join(corpus_dir, "freqs")
    if not os.path.isdir(freqs_dir):
        return []
    if allnorms is None:
        allnorms = get_allnorms()
    allnorms = allnorms[allnorms.index.notna() & ~allnorms.index.duplicated()]
    if norm_filter:
        allnorms = allnorms[[c for c in allnorms.columns if c in norm_filter]]
    if bin_edges is None:
        bin_edges = DEFAULT_BIN_EDGES
    bin_edges = np.asarray(bin_edges)

    requested_norms = set(allnorms.columns)
    spelling_d = get_spelling_modernizer() if modernize else None

    # Load existing records and check which texts need (re-)processing
    existing, all_records = _load_done_jsonl(output_path) if output_path else ({}, [])
    done_ids = set()
    needs_update = {}  # id -> existing record that needs new norms merged in
    for tid, rec in existing.items():
        rec_norms = set(k for k in rec if k != "id")
        if requested_norms.issubset(rec_norms):
            done_ids.add(tid)
        else:
            needs_update[tid] = rec

    new_records = []
    fh = None
    try:
        if output_path:
            # Append mode: new texts get appended; updated texts rewritten at end
            fh = open(output_path, "a")

        for text_id, path in _walk_freqs(freqs_dir):
            if text_id in done_ids:
                continue
            cdfs = _count_freqs_allnorms(path, allnorms, bin_edges, spelling_d)
            if not cdfs:
                continue

            if text_id in needs_update:
                # Merge new norms into existing record
                merged = needs_update.pop(text_id)
                merged.update(cdfs)
                new_records.append(merged)
            else:
                rec = {"id": text_id}
                rec.update(cdfs)
                new_records.append(rec)
                if fh:
                    fh.write(json.dumps(rec) + "\n")
    finally:
        if fh:
            fh.close()

    # If we had to update existing records, rewrite the whole file
    if needs_update is not None and any(
        tid not in done_ids and tid not in {r["id"] for r in new_records}
        for tid in existing
    ) or any(r["id"] in existing for r in new_records):
        if output_path and new_records:
            # Rebuild: existing (unchanged) + updated/new
            final = {}
            for rec in all_records:
                final[rec["id"]] = rec
            for rec in new_records:
                final[rec["id"]] = rec
            with open(output_path, "w") as f:
                for rec in final.values():
                    f.write(json.dumps(rec) + "\n")

    return all_records + [r for r in new_records if r["id"] not in existing]


def count_all_corpora(
    corpora_dir=PATH_CORPORA,
    output_dir=None,
    force=False,
    norm_filter=None,
    bin_edges=None,
    modernize=False,
):
    """Count z-score distributions for all corpora with freqs/ folders.

    Saves one JSONL file per corpus to output_dir/v2/.
    Resumable: skips already-counted texts. Re-running with additional
    norms merges new norms into existing records.

    Parameters
    ----------
    corpora_dir : str
        Parent directory containing corpus subdirectories.
    output_dir : str, optional
        Where to save per-corpus count files. Default: COUNT_DIR.
    force : bool
        If True, delete existing files and re-count from scratch.
    norm_filter : list of str, optional
        Only count these norm columns.
    bin_edges : array-like, optional
        Z-score bin edges. Default: -3.0 to 3.0 in 0.1 steps.
    modernize : bool
        If True (default), prefer modernized spelling for norm lookups.
        Outputs go to v2/ (modernized) or v2-raw/ (unmodernized).
    """
    if output_dir is None:
        output_dir = COUNT_DIR
    output_dir = _version_dir(output_dir, "v2", modernize)
    os.makedirs(output_dir, exist_ok=True)

    allnorms = get_allnorms()
    allnorms = allnorms[allnorms.index.notna() & ~allnorms.index.duplicated()]
    if norm_filter:
        allnorms = allnorms[[c for c in allnorms.columns if c in norm_filter]]
    if bin_edges is None:
        bin_edges = DEFAULT_BIN_EDGES
    bin_edges = np.asarray(bin_edges)

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

    requested_norms = set(allnorms.columns)
    spelling_d = get_spelling_modernizer() if modernize else None

    # collect files to process, skipping done IDs
    all_files = []
    for name, corpus_dir in corpora:
        out_path = os.path.join(output_dir, f"{name}.jsonl")
        if force and os.path.exists(out_path):
            os.remove(out_path)
        existing, _ = _load_done_jsonl(out_path)
        n_done = 0
        for tid, rec in existing.items():
            rec_norms = set(k for k in rec if k != "id")
            if requested_norms.issubset(rec_norms):
                n_done += 1
        freqs_dir = os.path.join(corpus_dir, "freqs")
        n_new = 0
        for text_id, path in _walk_freqs(freqs_dir):
            if text_id not in existing or not requested_norms.issubset(
                set(k for k in existing[text_id] if k != "id")
            ):
                all_files.append((name, text_id, path))
                n_new += 1
        if n_done:
            print(f"  {name}: {n_done} already done, {n_new} remaining")

    # open one JSONL file handle per corpus; track records for merging
    file_handles = {}
    corpus_records = {}  # name -> {id: record}

    def get_handle(name):
        if name not in file_handles:
            out_path = os.path.join(output_dir, f"{name}.jsonl")
            existing, _ = _load_done_jsonl(out_path)
            corpus_records[name] = existing
            file_handles[name] = open(out_path, "a")
        return file_handles[name]

    pbar = tqdm(all_files, desc="Counting", unit="file")
    needs_rewrite = set()
    try:
        for name, text_id, path in pbar:
            pbar.set_postfix_str(name, refresh=False)
            cdfs = _count_freqs_allnorms(path, allnorms, bin_edges, spelling_d)
            if not cdfs:
                continue

            fh = get_handle(name)
            if text_id in corpus_records[name]:
                # Merge new norms into existing record — needs rewrite
                corpus_records[name][text_id].update(cdfs)
                needs_rewrite.add(name)
            else:
                rec = {"id": text_id}
                rec.update(cdfs)
                corpus_records[name][text_id] = rec
                fh.write(json.dumps(rec) + "\n")
    finally:
        for fh in file_handles.values():
            fh.close()

    # Rewrite files that had merged records
    for name in needs_rewrite:
        out_path = os.path.join(output_dir, f"{name}.jsonl")
        with open(out_path, "w") as f:
            for rec in corpus_records[name].values():
                f.write(json.dumps(rec) + "\n")

    return {name: list(corpus_records.get(name, {}).values()) for name, _ in corpora}


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
