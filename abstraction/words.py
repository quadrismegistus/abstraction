"""
Word-trend analysis: which words drive the abstraction arc?

Two complementary analyses:
1. Correlation: which words' frequency trajectories track the concreteness trend?
2. Contribution: which words' frequency × z-score most explain the shift?
"""

import json
import os

import numpy as np
import pandas as pd
from scipy.stats import zscore
from tqdm import tqdm

from .config import PATH_CORPORA, SCORES_DIR
from .corpus import load_corpus, _camel_to_snake
from .norms import get_allnorms
from .scoring import _walk_freqs


# ---------------------------------------------------------------------------
# Aggregate word frequencies by decade
# ---------------------------------------------------------------------------

def aggregate_freqs_by_decade(corpus_name, min_year=1500, max_year=2020,
                              decade_len=10, top_n=10000,
                              corpora_dir=None):
    """Load all freq files for a corpus, aggregate word frequencies by decade.

    Parameters
    ----------
    corpus_name : str
        Corpus directory name (snake_case).
    min_year, max_year : int
        Year range to include.
    decade_len : int
        Bin width in years. Default 10.
    top_n : int
        Keep only the top N most frequent words (across all decades).
    corpora_dir : str, optional
        Path to corpora directory. Default PATH_CORPORA.

    Returns
    -------
    DataFrame
        Rows = decades, columns = words, values = total word count per decade.
        Index is the decade start year.
    """
    if corpora_dir is None:
        corpora_dir = PATH_CORPORA
    corpus_dir = os.path.join(corpora_dir, corpus_name)
    freqs_dir = os.path.join(corpus_dir, "freqs")

    # Load metadata for years
    corpus = load_corpus(corpus_name)
    meta = corpus.metadata
    if "year" not in meta.columns:
        raise ValueError(f"No 'year' column in {corpus_name} metadata")
    meta["year"] = pd.to_numeric(meta["year"], errors="coerce")
    id_to_year = meta.set_index("id")["year"].dropna().to_dict()

    # Walk freq files, accumulate per-decade word counts
    decade_freqs = {}  # {decade: Counter-like dict}
    n_matched = 0

    for text_id, path in tqdm(list(_walk_freqs(freqs_dir)),
                              desc=f"Loading {corpus_name} freqs"):
        # Try to match text_id to metadata
        year = id_to_year.get(text_id)
        if year is None:
            # Try common ID normalizations
            for alt in [text_id.replace("/", "."),
                        text_id.replace("/", ".").split(".")[-1],
                        ".".join(text_id.split("/")[:2])]:
                year = id_to_year.get(alt)
                if year is not None:
                    break
        if year is None or not (min_year <= year < max_year):
            continue

        decade = int(year // decade_len * decade_len)

        try:
            with open(path) as f:
                freqs = json.load(f)
        except Exception:
            continue

        if decade not in decade_freqs:
            decade_freqs[decade] = {}
        bucket = decade_freqs[decade]
        for word, count in freqs.items():
            # Only alphabetic words
            if word and word[0].isalpha():
                w = word.lower()
                bucket[w] = bucket.get(w, 0) + count
        n_matched += 1

    if not decade_freqs:
        raise ValueError(f"No texts matched metadata for {corpus_name}")

    print(f"  {n_matched} texts matched across {len(decade_freqs)} decades")

    # Build DataFrame
    df = pd.DataFrame(decade_freqs).T.fillna(0).sort_index()
    df.index.name = "decade"

    # Keep only top_n most frequent words overall
    if top_n and len(df.columns) > top_n:
        totals = df.sum(axis=0)
        keep = totals.nlargest(top_n).index
        df = df[keep]

    return df


def load_aggregate_freqs(corpus_names, cache_dir=None, force=False, **kwargs):
    """Aggregate freqs across multiple corpora, with optional caching.

    Parameters
    ----------
    corpus_names : list of str
        Corpus directory names.
    cache_dir : str, optional
        Directory to cache per-corpus aggregates. If None, no caching.
    force : bool
        Re-aggregate even if cache exists.
    **kwargs
        Passed to aggregate_freqs_by_decade.

    Returns
    -------
    DataFrame
        Combined decade × word frequency matrix (summed across corpora).
    """
    combined = None
    for name in corpus_names:
        cache_path = None
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            cache_path = os.path.join(cache_dir, f"{name}_decade_freqs.pkl")

        if cache_path and os.path.exists(cache_path) and not force:
            print(f"Loading cached {name}")
            df = pd.read_pickle(cache_path)
        else:
            df = aggregate_freqs_by_decade(name, **kwargs)
            if cache_path:
                df.to_pickle(cache_path)

        if combined is None:
            combined = df
        else:
            combined = combined.add(df, fill_value=0)

    return combined.fillna(0)


# ---------------------------------------------------------------------------
# Word-trend correlation
# ---------------------------------------------------------------------------

def correlate_words_with_trend(decade_freqs, score_col="Abs-Conc.Median.median",
                               min_total_freq=100, method="cosine"):
    """Correlate each word's frequency trajectory with the aggregate concreteness trend.

    Parameters
    ----------
    decade_freqs : DataFrame
        Rows = decades, columns = words, values = frequency counts.
        From aggregate_freqs_by_decade().
    score_col : str
        Norm column for computing the aggregate trend and word scores.
    min_total_freq : int
        Minimum total frequency across all decades to include a word.
    method : str
        'cosine' for cosine similarity, 'pearson' for Pearson correlation.

    Returns
    -------
    DataFrame
        Columns: word, freq, z_score, correlation, correlation_z,
                 category (Abstract/Concrete/Neither).
        Sorted by correlation (ascending = most anti-correlated with trend).
    """
    allnorms = get_allnorms()
    if score_col not in allnorms.columns:
        raise ValueError(f"Unknown norm column: {score_col}")

    norm_scores = allnorms[score_col].dropna()

    # Filter to words that appear in norms
    shared_words = sorted(set(decade_freqs.columns) & set(norm_scores.index))
    if not shared_words:
        raise ValueError("No overlap between freq words and norm vocabulary")

    freq_df = decade_freqs[shared_words]

    # Filter by minimum frequency
    totals = freq_df.sum(axis=0)
    keep = totals[totals >= min_total_freq].index
    freq_df = freq_df[keep]

    # Normalize to proportions per decade (each decade sums to 1)
    row_sums = freq_df.sum(axis=1)
    prop_df = freq_df.div(row_sums, axis=0)

    # Compute aggregate concreteness trend per decade
    # weighted mean concreteness = sum(proportion * z_score) for each decade
    word_scores = norm_scores[prop_df.columns]
    trend = prop_df.values @ word_scores.values  # decades × 1

    # Correlate each word's proportion trajectory with the trend
    if method == "cosine":
        from numpy.linalg import norm as np_norm
        trend_normed = trend / (np_norm(trend) + 1e-12)
        word_trajs = prop_df.values.T  # words × decades
        word_norms = np.sqrt((word_trajs ** 2).sum(axis=1)) + 1e-12
        correlations = (word_trajs @ trend_normed) / word_norms
    elif method == "pearson":
        correlations = np.array([
            np.corrcoef(prop_df[w].values, trend)[0, 1]
            for w in prop_df.columns
        ])
    else:
        raise ValueError(f"Unknown method: {method}")

    # Build results DataFrame
    results = pd.DataFrame({
        "word": prop_df.columns,
        "freq": totals[prop_df.columns].values,
        "z_score": word_scores.values,
        "correlation": correlations,
    })

    # Classify
    results["category"] = "Neither"
    results.loc[results["z_score"] <= -1.0, "category"] = "Abstract"
    results.loc[results["z_score"] >= 1.0, "category"] = "Concrete"

    # Z-score the correlations for easier interpretation
    results["correlation_z"] = zscore(results["correlation"])
    results["freq_log"] = np.log10(results["freq"].clip(lower=1))

    results = results.sort_values("correlation", ascending=True).reset_index(drop=True)
    return results


# ---------------------------------------------------------------------------
# Contribution decomposition
# ---------------------------------------------------------------------------

def word_contributions(decade_freqs, score_col="Abs-Conc.Median.median",
                       period_early=(1700, 1780), period_late=(1850, 1950),
                       min_total_freq=100):
    """Compute each word's contribution to the concreteness shift between two periods.

    contribution = (frequency_change) × (concreteness_z_score)

    A word with negative z-score (abstract) that decreased in frequency
    contributes positively to concretization (less abstract → more concrete overall).

    Parameters
    ----------
    decade_freqs : DataFrame
        Rows = decades, columns = words.
    score_col : str
        Norm column.
    period_early : tuple (start, end)
        The "abstract peak" period (inclusive start, exclusive end).
    period_late : tuple (start, end)
        The "concrete" period.
    min_total_freq : int
        Minimum total frequency.

    Returns
    -------
    DataFrame
        Columns: word, z_score, freq_early, freq_late, freq_change,
                 contribution, category.
        Sorted by contribution (largest positive = most concretizing).
    """
    allnorms = get_allnorms()
    norm_scores = allnorms[score_col].dropna()

    shared = sorted(set(decade_freqs.columns) & set(norm_scores.index))
    freq_df = decade_freqs[shared]

    # Filter by frequency
    totals = freq_df.sum(axis=0)
    keep = totals[totals >= min_total_freq].index
    freq_df = freq_df[keep]

    # Normalize to proportions
    row_sums = freq_df.sum(axis=1)
    prop_df = freq_df.div(row_sums, axis=0)

    # Compute mean proportions in each period
    early_mask = (prop_df.index >= period_early[0]) & (prop_df.index < period_early[1])
    late_mask = (prop_df.index >= period_late[0]) & (prop_df.index < period_late[1])

    if early_mask.sum() == 0:
        raise ValueError(f"No decades in early period {period_early}")
    if late_mask.sum() == 0:
        raise ValueError(f"No decades in late period {period_late}")

    freq_early = prop_df.loc[early_mask].mean(axis=0)
    freq_late = prop_df.loc[late_mask].mean(axis=0)
    freq_change = freq_late - freq_early

    word_scores = norm_scores[freq_df.columns]

    # Contribution: freq_change × z_score
    # If a concrete word (positive z) increased (positive change) → positive contribution (concretizing)
    # If an abstract word (negative z) decreased (negative change) → positive contribution (concretizing)
    contribution = freq_change * word_scores

    results = pd.DataFrame({
        "word": freq_df.columns,
        "z_score": word_scores.values,
        "freq_early": freq_early.values,
        "freq_late": freq_late.values,
        "freq_change": freq_change.values,
        "contribution": contribution.values,
        "freq_total": totals[freq_df.columns].values,
    })

    results["category"] = "Neither"
    results.loc[results["z_score"] <= -1.0, "category"] = "Abstract"
    results.loc[results["z_score"] >= 1.0, "category"] = "Concrete"

    results["freq_change_pct"] = (results["freq_change"] / results["freq_early"].clip(lower=1e-10)) * 100

    results = results.sort_values("contribution", ascending=False).reset_index(drop=True)
    return results


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def summarize_contributions(contrib_df, n=20):
    """Print a readable summary of top contributing words.

    Parameters
    ----------
    contrib_df : DataFrame
        From word_contributions().
    n : int
        Number of top words to show per direction.
    """
    print("=" * 72)
    print(f"TOP {n} CONCRETIZING WORDS (drove language toward concreteness)")
    print("=" * 72)
    top_conc = contrib_df.head(n)
    for _, row in top_conc.iterrows():
        direction = "rose" if row["freq_change"] > 0 else "fell"
        print(f"  {row['word']:20s}  z={row['z_score']:+.2f} ({row['category']:>8s})  "
              f"freq {direction} {abs(row['freq_change_pct']):>6.1f}%  "
              f"contrib={row['contribution']:+.2e}")

    print()
    print("=" * 72)
    print(f"TOP {n} ABSTRACTING WORDS (drove language toward abstraction)")
    print("=" * 72)
    bottom = contrib_df.tail(n).iloc[::-1]
    for _, row in bottom.iterrows():
        direction = "rose" if row["freq_change"] > 0 else "fell"
        print(f"  {row['word']:20s}  z={row['z_score']:+.2f} ({row['category']:>8s})  "
              f"freq {direction} {abs(row['freq_change_pct']):>6.1f}%  "
              f"contrib={row['contribution']:+.2e}")


def summarize_correlations(corr_df, n=20):
    """Print a readable summary of most/least correlated words.

    Parameters
    ----------
    corr_df : DataFrame
        From correlate_words_with_trend().
    n : int
        Number of words per direction.
    """
    print("=" * 72)
    print(f"TOP {n} WORDS TRACKING THE CONCRETIZING TREND")
    print("  (frequency rises as language gets more concrete)")
    print("=" * 72)
    top = corr_df.tail(n).iloc[::-1]
    for _, row in top.iterrows():
        print(f"  {row['word']:20s}  z={row['z_score']:+.2f} ({row['category']:>8s})  "
              f"r={row['correlation']:+.3f}  freq={row['freq']:,.0f}")

    print()
    print("=" * 72)
    print(f"TOP {n} WORDS TRACKING THE ABSTRACTING TREND")
    print("  (frequency rises as language gets more abstract)")
    print("=" * 72)
    top_abs = corr_df.head(n)
    for _, row in top_abs.iterrows():
        print(f"  {row['word']:20s}  z={row['z_score']:+.2f} ({row['category']:>8s})  "
              f"r={row['correlation']:+.3f}  freq={row['freq']:,.0f}")
