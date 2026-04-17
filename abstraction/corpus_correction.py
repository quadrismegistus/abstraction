"""
Corpus bias correction via within-match-group paired comparisons.

Different corpora have systematic score biases from transcription quality
(OCR vs hand-transcribed), edition selection, text completeness, etc.
This module estimates those biases from texts that appear in multiple
corpora via LLTK's match groups — the same work, different transcriptions.

Method: within-group demeaning + OLS on corpus dummies.
"""

import json
import os
import sqlite3

import numpy as np
import pandas as pd

from .config import SCORES_DIR


DEFAULT_SCORE_COL = "Abs-Conc.Median.median"
REFERENCE_CORPUS = "ecco_tcp"
BIAS_PATH = os.path.join(SCORES_DIR, "corpus_bias_coefficients.json")
FREQS_CACHE_PATH = os.path.join(SCORES_DIR, "freqs_cache.db")


def load_match_group_scores(score_col=DEFAULT_SCORE_COL, modernize=False):
    """Load per-freqs-file scores joined with match group + corpus info.

    Returns DataFrame with columns: group_id, corpus, freqs_key, score
    filtered to groups with 2+ distinct corpora.
    """
    import sys
    sys.path.insert(0, os.path.expanduser("~/github/lltk"))
    import lltk

    # Load cached scores
    mod_int = 1 if modernize else 0
    conn = sqlite3.connect(FREQS_CACHE_PATH)
    cache_rows = conn.execute(
        "SELECT freqs_key, scores_json FROM freqs_scores WHERE modernized = ?",
        (mod_int,),
    ).fetchall()
    conn.close()

    scores = {}
    for key, sj in cache_rows:
        try:
            d = json.loads(sj)
            v = d.get(score_col)
            if v is not None:
                scores[key] = v
        except (json.JSONDecodeError, TypeError):
            pass
    print(f"  Cache: {len(scores)} entries with {score_col}")

    # Load match groups + freqs paths from LLTK DB
    df = lltk.db.conn.execute("""
        SELECT t._id, t.corpus, t.path_freqs,
               mg.group_id AS group_id
        FROM texts t
        JOIN match_db.match_groups mg ON t._id = mg._id
        WHERE t.path_freqs IS NOT NULL
    """).fetchdf()
    print(f"  DB: {len(df)} texts with freqs + match groups")

    # Join with cached scores
    df["score"] = df["path_freqs"].map(scores)
    df = df.dropna(subset=["score"])
    print(f"  Joined: {len(df)} scored texts in match groups")

    # Keep only groups with 2+ distinct corpora
    multi = df.groupby("group_id").filter(lambda g: g["corpus"].nunique() > 1)
    print(f"  Multi-corpus groups: {multi['group_id'].nunique()} "
          f"({len(multi)} texts)")

    return multi[["group_id", "corpus", "path_freqs", "score"]].reset_index(drop=True)


def estimate_corpus_bias(
    df=None,
    score_col=DEFAULT_SCORE_COL,
    reference_corpus=REFERENCE_CORPUS,
    min_group_overlap=10,
    modernize=False,
):
    """Estimate corpus bias coefficients from within-match-group variation.

    Parameters
    ----------
    df : DataFrame, optional
        Output of load_match_group_scores(). Loaded if not provided.
    reference_corpus : str
        Corpus to use as the zero baseline.
    min_group_overlap : int
        Minimum match groups a corpus must appear in to get a coefficient.

    Returns
    -------
    dict with keys:
        coefficients: {corpus: bias_value}  (reference corpus = 0)
        se: {corpus: standard_error}
        n_groups: {corpus: n_match_groups_used}
        reference: str
        score_col: str
        connected_components: list of lists
        uncalibrated: list of corpus names not connected to reference
    """
    if df is None:
        df = load_match_group_scores(score_col=score_col, modernize=modernize)

    # Filter corpora with too few groups
    corpus_group_counts = df.groupby("corpus")["group_id"].nunique()
    keep_corpora = set(corpus_group_counts[corpus_group_counts >= min_group_overlap].index)
    df = df[df["corpus"].isin(keep_corpora)].copy()

    # Re-filter to multi-corpus groups after removing rare corpora
    df = df.groupby("group_id").filter(lambda g: g["corpus"].nunique() > 1)

    if len(df) == 0:
        print("  No multi-corpus groups after filtering")
        return None

    # Check connectivity
    components = _find_connected_components(df)
    ref_component = None
    for comp in components:
        if reference_corpus in comp:
            ref_component = comp
            break

    if ref_component is None:
        print(f"  Warning: reference corpus '{reference_corpus}' not found in "
              f"any component. Using largest component's first corpus.")
        ref_component = max(components, key=len)
        reference_corpus = sorted(ref_component)[0]

    uncalibrated = []
    for comp in components:
        if comp is not ref_component:
            uncalibrated.extend(sorted(comp))

    # Mean score per (group, corpus) — average if multiple freqs per corpus per group
    gm = df.groupby(["group_id", "corpus"])["score"].mean().reset_index()

    # Within-group demeaning
    group_means = gm.groupby("group_id")["score"].transform("mean")
    gm["demeaned"] = gm["score"] - group_means

    # Build corpus dummies (drop reference corpus)
    corpora = sorted(set(gm["corpus"]) - {reference_corpus})
    corpus_to_idx = {c: i for i, c in enumerate(corpora)}

    X = np.zeros((len(gm), len(corpora)))
    for i, c in enumerate(gm["corpus"]):
        if c in corpus_to_idx:
            X[i, corpus_to_idx[c]] = 1.0

    y = gm["demeaned"].values

    # OLS: y = X @ beta + eps
    beta, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)

    # Standard errors
    n, p = X.shape
    resid = y - X @ beta
    mse = (resid ** 2).sum() / max(n - p, 1)
    try:
        XtX_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(XtX_inv) * mse)

    # Build result
    coefficients = {reference_corpus: 0.0}
    se_dict = {reference_corpus: 0.0}
    n_groups = {reference_corpus: int(corpus_group_counts.get(reference_corpus, 0))}

    for c, b, s in zip(corpora, beta, se):
        coefficients[c] = float(b)
        se_dict[c] = float(s)
        n_groups[c] = int(corpus_group_counts.get(c, 0))

    result = {
        "coefficients": coefficients,
        "se": se_dict,
        "n_groups": n_groups,
        "reference": reference_corpus,
        "score_col": score_col,
        "connected_components": [sorted(c) for c in components],
        "uncalibrated": uncalibrated,
    }

    # Print summary
    print(f"\n  Corpus bias coefficients (reference: {reference_corpus}):")
    print(f"  {'Corpus':>25s}  {'Bias':>8s}  {'SE':>8s}  {'Groups':>6s}")
    print(f"  {'-'*55}")
    for c in sorted(coefficients, key=lambda x: coefficients[x]):
        print(f"  {c:>25s}  {coefficients[c]:+8.4f}  {se_dict[c]:8.4f}  {n_groups[c]:6d}")

    if uncalibrated:
        print(f"\n  Uncalibrated (not connected to {reference_corpus}): "
              f"{', '.join(uncalibrated)}")

    return result


def _find_connected_components(df):
    """Find connected components in the corpus overlap graph."""
    # Build adjacency from shared match groups
    edges = set()
    for _, g in df.groupby("group_id"):
        corpora = sorted(g["corpus"].unique())
        for i in range(len(corpora)):
            for j in range(i + 1, len(corpora)):
                edges.add((corpora[i], corpora[j]))

    # Union-find
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[a] = b

    all_corpora = set(df["corpus"].unique())
    for c in all_corpora:
        find(c)
    for a, b in edges:
        union(a, b)

    components = {}
    for c in all_corpora:
        root = find(c)
        components.setdefault(root, set()).add(c)

    return list(components.values())


def save_corpus_bias(bias_dict, path=None):
    """Save corpus bias coefficients to JSON."""
    path = path or BIAS_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(bias_dict, f, indent=2)
    print(f"  Saved to {path}")


def load_corpus_bias(path=None):
    """Load corpus bias coefficients from JSON."""
    path = path or BIAS_PATH
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# Language-specific coefficient files. Each is calibrated against its own
# reference corpus (EN → ecco_tcp, FR → gallica_literary_fictions), so
# corrections are per-language. Since corpus names don't overlap across
# languages, a merged dict is safe for heterogeneous queries.
_LANG_BIAS_FILES = {
    "en": "corpus_bias_coefficients.json",
    "fr": "corpus_bias_coefficients_fr.json",
    "de": "corpus_bias_coefficients_de.json",
}


def load_all_corpus_bias():
    """Load and merge per-language corpus bias coefficients.

    Returns a dict shaped like the single-language version but containing
    coefficients from all available languages. Used by the web app to
    apply per-text correction regardless of which language arc a text
    belongs to (arc_fiction reps correct against English coeffs, arc_fiction_fr
    against French, etc. — one unified lookup since corpus names don't collide).
    """
    merged = {"coefficients": {}, "se": {}, "_sources": {}}
    for lang, fn in _LANG_BIAS_FILES.items():
        p = os.path.join(SCORES_DIR, fn)
        if not os.path.exists(p):
            continue
        with open(p) as f:
            b = json.load(f)
        coeffs = b.get("coefficients", {})
        merged["coefficients"].update(coeffs)
        merged["se"].update(b.get("se", {}))
        for corpus_name in coeffs:
            merged["_sources"][corpus_name] = lang
    if not merged["coefficients"]:
        return None
    return merged


def correct_scores_df(df, score_col=DEFAULT_SCORE_COL, corpus_col="corpus_name",
                      bias=None):
    """Subtract corpus bias from scores in a DataFrame.

    Parameters
    ----------
    df : DataFrame
        Must have score_col and corpus_col columns.
    bias : dict, optional
        Output of estimate_corpus_bias() or load_corpus_bias().
        Loaded from disk if not provided.

    Returns
    -------
    DataFrame with score_col values corrected in place.
    """
    if bias is None:
        bias = load_corpus_bias()
    if bias is None:
        return df

    coefficients = bias["coefficients"]
    df = df.copy()
    corrections = df[corpus_col].map(coefficients).fillna(0.0)
    df[score_col] = df[score_col] - corrections
    return df


# CLI entry point
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Estimate corpus bias coefficients")
    parser.add_argument("--score-col", default=DEFAULT_SCORE_COL)
    parser.add_argument("--reference", default=REFERENCE_CORPUS)
    parser.add_argument("--min-overlap", type=int, default=10)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    result = estimate_corpus_bias(
        score_col=args.score_col,
        reference_corpus=args.reference,
        min_group_overlap=args.min_overlap,
    )
    if result:
        save_corpus_bias(result, args.output)
