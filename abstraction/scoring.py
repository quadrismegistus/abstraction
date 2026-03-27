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
from .tokenize import tokenize_agnostic, get_spelling_modernizer
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


def _modernize_score(word, norm_dict, spelling_d):
    """Look up word in norms; if not found, try modernized spelling.

    Returns (score, matched_word) or (None, None) if no match.
    """
    if word in norm_dict:
        return norm_dict[word], word
    modern = spelling_d.get(word)
    if modern is not None and modern in norm_dict:
        return norm_dict[modern], modern
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


def _modernize_word_list(words_lower, norm_index, spelling_d):
    """For each word not in norm_index, try modernized spelling."""
    result = []
    for w in words_lower:
        if w in norm_index:
            result.append(w)
        else:
            modern = spelling_d.get(w)
            result.append(modern if modern is not None and modern in norm_index else w)
    return result


def _score_freqs_allnorms(path, allnorms, spelling_d=None):
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
    if spelling_d:
        words_lower = _modernize_word_list(words_lower, set(allnorms.index), spelling_d)
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
    spelling_d = get_spelling_modernizer()

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
    spelling_d = get_spelling_modernizer()
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
            scores = _score_freqs_allnorms(path, allnorms, spelling_d)
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
                       bin_edges=None, norm_filter=None):
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
    spelling_d = get_spelling_modernizer()

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
):
    """Count z-score distributions for all corpora with freqs/ folders.

    Saves one JSONL file per corpus to output_dir/v1/.
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
    spelling_d = get_spelling_modernizer()

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
