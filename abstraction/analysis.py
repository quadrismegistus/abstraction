"""
Arc analysis: detecting and quantifying the rise-and-fall pattern
of abstract language across literary history.
"""

import os

import numpy as np
import pandas as pd
from scipy import stats

from .config import SCORES_DIR, PATH_CORPORA
from .corpus import load_corpus, _camel_to_snake
from .tokenize import tokenize_agnostic
from .utils import read_df


# ---------------------------------------------------------------------------
# Genre harmonization
# ---------------------------------------------------------------------------

# Corpus-level genre: if the corpus name implies a single genre, use it
CORPUS_GENRE = {
    "chadwyck": "Fiction",
    "chadwyck_drama": "Drama",
    "chadwyck_poetry": "Poetry",
    "chicago": "Fiction",
    "gale_amfic": "Fiction",
    "gildedage": "Fiction",
    "internet_archive": "Fiction",
    "markmark": "Fiction",
    "fanfic": "Fiction",
    "canon_fiction": "Fiction",
    "litlab": "Fiction",
    "hathi_novels": "Fiction",
    "hathi_stories": "Fiction",
    "hathi_tales": "Fiction",
    "hathi_romances": "Fiction",
    "hathi_essays": "Essay/Treatise",
    "hathi_treatises": "Essay/Treatise",
    "hathi_sermons": "Sermon",
    "hathi_letters": "Letters",
    "hathi_bio": "Biography",
    "hathi_proclamations": "Proclamation",
    "hathi_almanacs": "Almanac",
    "old_bailey": "Legal",
    "oldbailey": "Legal",
    "sotu": "Political",
    "spectator": "Periodical",
    "new_yorker": "Periodical",
    "pmla": "Criticism",
    "bpo": "Periodical",
}

# Map raw genre values to harmonized categories
_GENRE_MAP = {
    # Poetry
    "Verse": "Poetry", "Poetry": "Poetry", "Poem": "Poetry",
    "Sonnet": "Poetry", "Lyric": "Poetry", "Heroic couplets": "Poetry",
    "Ballad": "Poetry", "Ode": "Poetry", "Hymn": "Poetry",
    "Epigram": "Poetry", "Metrical Psalm": "Poetry", "Elegy": "Poetry",
    "Epitaph": "Poetry", "Prologue": "Poetry", "Epilogue": "Poetry",
    # Fiction
    "Fiction": "Fiction", "Novel": "Fiction", "Tale": "Fiction",
    "Story": "Fiction", "FIC": "Fiction", "FanFiction": "Fiction",
    "Romance": "Fiction", "Gothic": "Fiction", "Historical": "Fiction",
    "Silver Fork": "Fiction", "New Woman": "Fiction", "Epistolary": "Fiction",
    "Picaresque": "Fiction", "Oriental": "Fiction", "Anti-Jacobin": "Fiction",
    "Jacobin": "Fiction", "National tale": "Fiction", "Evangelical": "Fiction",
    "DET": "Fiction", "ROM": "Fiction", "FANT": "Fiction",
    "SCI": "Fiction", "SOC": "Fiction", "HIST": "Fiction",
    "ADV": "Fiction", "WEST": "Fiction",
    # Drama
    "Drama": "Drama",
    # Essay / Treatise
    "Treatise": "Essay/Treatise", "Essay": "Essay/Treatise",
    "Discourse": "Essay/Treatise",
    # Letters
    "Letter": "Letters", "Letters": "Letters",
    # Biography
    "Biography": "Biography",
    # Sermon
    "Sermon": "Sermon",
    # Nonfiction catchall
    "Non-Fiction": "Nonfiction", "NF": "Nonfiction",
    # Periodical
    "Magazine": "Periodical", "MAG": "Periodical",
    "News": "Periodical", "NEWS": "Periodical",
    "Periodical": "Periodical",
    # Other media
    "SPOK": "Spoken", "ACAD": "Academic", "Film": "Film",
}

# Genre columns to try, in priority order
_GENRE_COL_PRIORITY = [
    "major_genre", "genre", "attgenre", "medium",
    "genre_label", "documentType", "ObjectType",
]

# Vague genre values that should fall through to title-based detection
_VAGUE_GENRES = {"Prose", "Print", "BOOK", ""}

# Keywords to detect genre from title
_TITLE_KEYWORDS = {
    "novel": "Fiction", "tale": "Fiction", "romance": "Fiction",
    "story": "Fiction", "stories": "Fiction",
    "sermon": "Sermon", "sermons": "Sermon",
    "essay": "Essay/Treatise", "essays": "Essay/Treatise",
    "treatise": "Essay/Treatise", "discourse": "Essay/Treatise",
    "letter": "Letters", "letters": "Letters",
    "poem": "Poetry", "poems": "Poetry", "ode": "Poetry",
    "hymn": "Poetry", "hymns": "Poetry", "ballad": "Poetry",
}


def _genre_from_title(title):
    """Infer genre from title keywords as a fallback."""
    if not isinstance(title, str):
        return ""
    words = [w.lower() for w in tokenize_agnostic(title)]
    for kw, genre in _TITLE_KEYWORDS.items():
        if any(w.startswith(kw) for w in words):
            return genre
    return ""


def _detect_genre_col(df):
    """Find the best genre column in a DataFrame."""
    for col in _GENRE_COL_PRIORITY:
        if col in df.columns:
            return col
    return None


def harmonize_genre(df, corpus_name=None):
    """Add a 'genre_harmonized' column to a scored+metadata DataFrame.

    Resolution order:
    1. Corpus-level genre (if corpus implies a single genre)
    2. Row-level genre from metadata column, mapped through _GENRE_MAP
    3. Title-based fallback
    4. Empty string if unresolvable
    """
    df = df.copy()
    snake = _camel_to_snake(corpus_name) if corpus_name else None

    # 1. Corpus-level override
    corpus_genre = CORPUS_GENRE.get(snake, "") if snake else ""

    # 2. Find the best genre column
    genre_col = _detect_genre_col(df)

    def _resolve(row):
        # corpus-level genre takes precedence for single-genre corpora
        if corpus_genre:
            return corpus_genre

        # row-level genre
        if genre_col:
            raw = str(row.get(genre_col, "")).strip()
            if raw and raw not in _VAGUE_GENRES:
                mapped = _GENRE_MAP.get(raw, "")
                if mapped:
                    return mapped
                # try as-is if it looks like a real genre
                return raw

        # title fallback
        title = row.get("title", "")
        g = _genre_from_title(title)
        if g:
            return g

        return ""

    df["genre_harmonized"] = df.apply(_resolve, axis=1)
    return df


# ---------------------------------------------------------------------------
# Loading and merging scores with metadata
# ---------------------------------------------------------------------------

def _find_id_col(meta):
    """Find the ID column in metadata, preferring 'id' then 'htid'."""
    for col in ["id", "htid"]:
        if col in meta.columns:
            return col
    return meta.columns[0]


def load_scores(corpus_name, scores_dir=None, version="v7", harmonize=True):
    """Load scored texts for a corpus and merge with metadata.

    Returns a DataFrame with score columns plus metadata (year, genre, etc.).
    If harmonize=True, adds a 'genre_harmonized' column.

    Handles ID format mismatches (e.g. hathi subcorpora where freqs paths
    use slashes but metadata uses dots in IDs).
    """
    if scores_dir is None:
        scores_dir = os.path.join(SCORES_DIR, version)
    snake = _camel_to_snake(corpus_name) if corpus_name[0].isupper() else corpus_name
    path = os.path.join(scores_dir, f"{snake}.csv")

    # If no corpus-specific scores, try falling back to parent hathi scores
    if not os.path.exists(path) and snake.startswith("hathi_"):
        parent_path = os.path.join(scores_dir, "hathi.csv")
        if os.path.exists(parent_path):
            path = parent_path

    if not os.path.exists(path):
        raise FileNotFoundError(f"No scores file: {path}")

    scores = pd.read_csv(path)
    corpus = load_corpus(corpus_name)
    meta = corpus.metadata

    # Find the right ID column in metadata
    id_col = _find_id_col(meta)
    if id_col != "id":
        meta = meta.rename(columns={id_col: "id"})

    # Try direct merge first
    merged = scores.merge(meta, on="id", how="inner")

    # If merge is poor, try slash↔dot normalization (hathi ID format mismatch)
    if len(merged) < len(scores) * 0.1 and len(scores) > 0:
        scores_norm = scores.copy()
        scores_norm["id"] = scores_norm["id"].str.replace("/", ".", n=1)
        merged_norm = scores_norm.merge(meta, on="id", how="inner")
        if len(merged_norm) > len(merged):
            merged = merged_norm

    if harmonize:
        merged = harmonize_genre(merged, corpus_name=corpus_name)
    return merged


def get_score_columns(df):
    """Return the norm score column names from a merged scores DataFrame."""
    return [c for c in df.columns if c.startswith("Abs-Conc.")]


# ---------------------------------------------------------------------------
# Quadratic fit: score ~ β₀ + β₁·year + β₂·year²
# ---------------------------------------------------------------------------

def fit_quadratic(years, scores):
    """Fit a quadratic model and return summary statistics.

    Parameters
    ----------
    years : array-like
        Year values.
    scores : array-like
        Score values (e.g. abstractness).

    Returns
    -------
    dict with keys: beta0, beta1, beta2, beta2_p, peak_year, r2, n
    """
    mask = np.isfinite(years) & np.isfinite(scores)
    y = np.asarray(years)[mask].astype(float)
    s = np.asarray(scores)[mask].astype(float)
    if len(y) < 10:
        return _empty_quad()

    # center years for numerical stability
    y_center = y.mean()
    yc = y - y_center

    X = np.column_stack([np.ones(len(yc)), yc, yc ** 2])
    try:
        beta, residuals, rank, sv = np.linalg.lstsq(X, s, rcond=None)
    except np.linalg.LinAlgError:
        return _empty_quad()

    b0, b1, b2 = beta
    s_pred = X @ beta
    ss_res = np.sum((s - s_pred) ** 2)
    ss_tot = np.sum((s - s.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # p-value for β₂
    n = len(y)
    p = X.shape[1]
    if n > p:
        mse = ss_res / (n - p)
        try:
            cov = mse * np.linalg.inv(X.T @ X)
            se_b2 = np.sqrt(cov[2, 2])
            t_stat = b2 / se_b2
            beta2_p = 2 * stats.t.sf(abs(t_stat), n - p)
        except np.linalg.LinAlgError:
            beta2_p = np.nan
    else:
        beta2_p = np.nan

    # peak year (vertex of parabola)
    peak_year = -b1 / (2 * b2) + y_center if b2 != 0 else np.nan

    return {
        "quad_beta0": b0,
        "quad_beta1": b1,
        "quad_beta2": b2,
        "quad_beta2_p": beta2_p,
        "quad_peak_year": peak_year,
        "quad_r2": r2,
        "quad_n": n,
    }


def _empty_quad():
    return {k: np.nan for k in [
        "quad_beta0", "quad_beta1", "quad_beta2", "quad_beta2_p",
        "quad_peak_year", "quad_r2", "quad_n",
    ]}


# ---------------------------------------------------------------------------
# Piecewise linear fit: two slopes joined at a breakpoint
# ---------------------------------------------------------------------------

def fit_piecewise(years, scores, break_year=None, search_range=(1650, 1850), search_step=10):
    """Fit a piecewise linear model with one breakpoint.

    Parameters
    ----------
    years : array-like
        Year values.
    scores : array-like
        Score values.
    break_year : int, optional
        Fixed breakpoint. If None, searches for the best fit.
    search_range : tuple
        (min_year, max_year) to search for breakpoint.
    search_step : int
        Step size for breakpoint search.

    Returns
    -------
    dict with keys: break_year, slope_before, slope_after, slope_before_p,
    slope_after_p, r2, n, n_before, n_after
    """
    mask = np.isfinite(years) & np.isfinite(scores)
    y = np.asarray(years)[mask].astype(float)
    s = np.asarray(scores)[mask].astype(float)
    if len(y) < 20:
        return _empty_piecewise()

    if break_year is not None:
        return _fit_piecewise_at(y, s, break_year)

    # grid search for best breakpoint
    best = None
    best_r2 = -np.inf
    lo, hi = search_range
    for by in range(lo, hi + 1, search_step):
        n_before = np.sum(y <= by)
        n_after = np.sum(y > by)
        if n_before < 10 or n_after < 10:
            continue
        result = _fit_piecewise_at(y, s, by)
        if result["pw_r2"] > best_r2:
            best_r2 = result["pw_r2"]
            best = result
    return best if best is not None else _empty_piecewise()


def _fit_piecewise_at(y, s, break_year):
    """Fit piecewise linear at a specific breakpoint."""
    before = y <= break_year
    after = y > break_year
    n_before = before.sum()
    n_after = after.sum()

    if n_before < 5 or n_after < 5:
        return _empty_piecewise()

    # fit each segment
    slope_b, intercept_b, r_b, p_b, se_b = stats.linregress(y[before], s[before])
    slope_a, intercept_a, r_a, p_a, se_a = stats.linregress(y[after], s[after])

    # overall R² of the piecewise model
    s_pred = np.empty_like(s)
    s_pred[before] = intercept_b + slope_b * y[before]
    s_pred[after] = intercept_a + slope_a * y[after]
    ss_res = np.sum((s - s_pred) ** 2)
    ss_tot = np.sum((s - s.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        "pw_break_year": break_year,
        "pw_slope_before": slope_b,
        "pw_slope_after": slope_a,
        "pw_slope_before_p": p_b,
        "pw_slope_after_p": p_a,
        "pw_r2": r2,
        "pw_n": len(y),
        "pw_n_before": int(n_before),
        "pw_n_after": int(n_after),
    }


def _empty_piecewise():
    return {k: np.nan for k in [
        "pw_break_year", "pw_slope_before", "pw_slope_after",
        "pw_slope_before_p", "pw_slope_after_p", "pw_r2",
        "pw_n", "pw_n_before", "pw_n_after",
    ]}


# ---------------------------------------------------------------------------
# Combined arc analysis
# ---------------------------------------------------------------------------

DEFAULT_MIN_YEAR = 1600
DEFAULT_MAX_YEAR = 2000
DEFAULT_AGG_BIN = 10  # aggregate by decade


def fit_arc(df, score_col="Abs-Conc.Median.median", year_col="year",
            min_year=DEFAULT_MIN_YEAR, max_year=DEFAULT_MAX_YEAR,
            agg_bin=DEFAULT_AGG_BIN, min_texts_per_bin=3, **kw):
    """Run both quadratic and piecewise fits on a scored DataFrame.

    By default, aggregates scores by decade before fitting, giving each
    time period equal weight regardless of how many texts it contains.

    Parameters
    ----------
    df : DataFrame
        Must have year_col and score_col columns.
    score_col : str
        Column name for the abstractness score.
    year_col : str
        Column name for the year.
    min_year, max_year : int, optional
        Restrict analysis to a year range.
    agg_bin : int or None
        Bin size in years for aggregation (default 10 = decades).
        Set to None to fit on individual texts.
    min_texts_per_bin : int
        Drop bins with fewer texts than this.

    Returns
    -------
    dict combining quadratic and piecewise results, plus metadata.
    """
    sub = df[[year_col, score_col]].copy()
    sub[year_col] = pd.to_numeric(sub[year_col], errors="coerce")
    sub = sub.dropna()
    if min_year is not None:
        sub = sub[sub[year_col] >= min_year]
    if max_year is not None:
        sub = sub[sub[year_col] <= max_year]

    n_texts = len(sub)

    if agg_bin is not None and len(sub) > 0:
        sub = sub.copy()
        sub["_bin"] = (sub[year_col] // agg_bin) * agg_bin
        agg = sub.groupby("_bin").agg(
            score=(score_col, "mean"),
            n_texts=(score_col, "count"),
        ).reset_index()
        agg = agg[agg.n_texts >= min_texts_per_bin]
        years = agg["_bin"].values.astype(float)
        scores = agg["score"].values
        n_bins = len(agg)
    else:
        years = sub[year_col].values
        scores = sub[score_col].values
        n_bins = len(sub)

    result = {"score_col": score_col, "n_texts": n_texts, "n_bins": n_bins}
    if len(years) > 0:
        result["year_min"] = int(years.min())
        result["year_max"] = int(years.max())
    else:
        result["year_min"] = np.nan
        result["year_max"] = np.nan

    result.update(fit_quadratic(years, scores))
    result.update(fit_piecewise(years, scores, **kw))
    return result


def fit_arc_corpus(corpus_name, score_col="Abs-Conc.Median.median", **kw):
    """Load scores for a corpus and fit the arc."""
    df = load_scores(corpus_name)
    result = fit_arc(df, score_col=score_col, **kw)
    result["corpus"] = corpus_name
    return result


def fit_arc_all_corpora(score_col="Abs-Conc.Median.median",
                        scores_dir=None, version="v7", **kw):
    """Fit arc for all scored corpora. Returns a DataFrame of results."""
    if scores_dir is None:
        scores_dir = os.path.join(SCORES_DIR, version)
    if not os.path.isdir(scores_dir):
        raise FileNotFoundError(f"No scores directory: {scores_dir}")

    results = []
    for fn in sorted(os.listdir(scores_dir)):
        if not fn.endswith(".csv"):
            continue
        corpus_name = fn.removesuffix(".csv")
        try:
            df = load_scores(corpus_name, scores_dir=scores_dir, version=version)
        except (FileNotFoundError, Exception) as e:
            print(f"  Skipping {corpus_name}: {e}")
            continue
        result = fit_arc(df, score_col=score_col, **kw)
        result["corpus"] = corpus_name
        results.append(result)

    return pd.DataFrame(results)


def fit_arc_by_genre(df, score_col="Abs-Conc.Median.median",
                     genre_col="genre_harmonized", min_texts=30, **kw):
    """Fit arc separately for each genre in a DataFrame.

    Parameters
    ----------
    df : DataFrame
        Scored + metadata DataFrame (must have genre_col and year columns).
    score_col : str
        Column to fit.
    genre_col : str
        Column with genre labels.
    min_texts : int
        Skip genres with fewer texts than this.

    Returns
    -------
    DataFrame of arc results, one row per genre.
    """
    results = []
    for genre, gdf in df.groupby(genre_col):
        if not genre or len(gdf) < min_texts:
            continue
        result = fit_arc(gdf, score_col=score_col, **kw)
        result["genre"] = genre
        result["corpus"] = genre  # for summarize_arc display
        results.append(result)
    return pd.DataFrame(results)


def fit_arc_all_by_genre(score_col="Abs-Conc.Median.median",
                         scores_dir=None, version="v7", min_texts=30, **kw):
    """Load all scored corpora, harmonize genres, and fit arc per genre.

    Pools texts across corpora by harmonized genre, then fits one arc
    per genre. Returns a DataFrame of results.
    """
    if scores_dir is None:
        scores_dir = os.path.join(SCORES_DIR, version)
    if not os.path.isdir(scores_dir):
        raise FileNotFoundError(f"No scores directory: {scores_dir}")

    all_dfs = []
    for fn in sorted(os.listdir(scores_dir)):
        if not fn.endswith(".csv"):
            continue
        corpus_name = fn.removesuffix(".csv")
        try:
            df = load_scores(corpus_name, scores_dir=scores_dir, version=version)
            df["corpus_name"] = corpus_name
            all_dfs.append(df)
        except Exception as e:
            print(f"  Skipping {corpus_name}: {e}")
            continue

    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    return fit_arc_by_genre(combined, score_col=score_col,
                            min_texts=min_texts, **kw)


def summarize_arc(result):
    """Format an arc result dict as a human-readable string."""
    lines = []
    corpus = result.get("corpus", "?")
    n_texts = result.get('n_texts', result.get('n', '?'))
    n_bins = result.get('n_bins', '')
    bin_str = f", {n_bins} bins" if n_bins and n_bins != n_texts else ""
    lines.append(f"=== {corpus} ({n_texts} texts{bin_str}, {result.get('year_min', '?')}-{result.get('year_max', '?')}) ===")

    # quadratic
    b2 = result.get("quad_beta2", np.nan)
    p2 = result.get("quad_beta2_p", np.nan)
    peak = result.get("quad_peak_year", np.nan)
    r2q = result.get("quad_r2", np.nan)
    # scores are concreteness (negative=abstract, positive=concrete)
    # β₂ > 0 means U in concreteness = inverted-U in abstractness = the expected arc
    if b2 > 0:
        arc_dir = "abstractness rises then falls (expected arc)"
    elif b2 < 0:
        arc_dir = "abstractness falls then rises (inverse of expected)"
    else:
        arc_dir = "flat"
    sig = "***" if p2 < 0.001 else "**" if p2 < 0.01 else "*" if p2 < 0.05 else "n.s."
    lines.append(f"  Quadratic: {arc_dir} {sig}, peak abstractness ~{peak:.0f}, R²={r2q:.3f}" if np.isfinite(peak) else "  Quadratic: insufficient data")

    # piecewise
    by = result.get("pw_break_year", np.nan)
    sb = result.get("pw_slope_before", np.nan)
    sa = result.get("pw_slope_after", np.nan)
    pb = result.get("pw_slope_before_p", np.nan)
    pa = result.get("pw_slope_after_p", np.nan)
    r2p = result.get("pw_r2", np.nan)
    if np.isfinite(by):
        sb_sig = "***" if pb < 0.001 else "**" if pb < 0.01 else "*" if pb < 0.05 else "n.s."
        sa_sig = "***" if pa < 0.001 else "**" if pa < 0.01 else "*" if pa < 0.05 else "n.s."
        # slopes per century (negative slope = growing abstractness, positive = growing concreteness)
        sb_dir = "abstracting" if sb < 0 else "concretizing"
        sa_dir = "abstracting" if sa < 0 else "concretizing"
        lines.append(f"  Piecewise (break {by:.0f}): before {sb_dir} {sb*100:+.4f}/century {sb_sig}, after {sa_dir} {sa*100:+.4f}/century {sa_sig}, R²={r2p:.3f}")
    else:
        lines.append("  Piecewise: insufficient data")

    return "\n".join(lines)
