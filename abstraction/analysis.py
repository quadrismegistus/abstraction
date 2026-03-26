"""
Arc analysis: detecting and quantifying the rise-and-fall pattern
of abstract language across literary history.
"""

import os
from tqdm import tqdm
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

    # Ensure both ID columns are strings
    scores["id"] = scores["id"].astype(str)
    meta["id"] = meta["id"].astype(str)

    # Try direct merge first
    merged = scores.merge(meta, on="id", how="inner")

    # Try progressively more aggressive normalizations if merge is poor
    def _try_merge(score_ids, meta_ids_col=None):
        nonlocal merged
        s = scores.copy()
        m = meta
        if score_ids is not None:
            s["id"] = score_ids
        if meta_ids_col is not None:
            m = meta.copy()
            m["id"] = meta_ids_col
        candidate = s.merge(m, on="id", how="inner")
        if len(candidate) > len(merged):
            merged = candidate

    if len(merged) < len(scores) * 0.5 and len(scores) > 0:
        score_ids = scores["id"]

        # slash→dot (hathi)
        _try_merge(score_ids.str.replace("/", ".", n=1))

        # zero-padded numeric (chicago): meta "1" -> scores "00000001"
        sample_sid = score_ids.iloc[0]
        if sample_sid.isdigit() and len(sample_sid) > 4:
            pad_len = len(sample_sid)
            _try_merge(None, meta["id"].apply(
                lambda x: str(x).split(".")[0].zfill(pad_len) if str(x).replace(".", "").isdigit() else x
            ))

        # underscore↔space (gildedage)
        _try_merge(None, meta["id"].str.replace("_", " "))

        # htid→path: "nyp.334330..." -> "nyp/334/330..."
        def _htid_to_path(htid):
            if "." in htid:
                prefix, rest = htid.split(".", 1)
                return f"{prefix}/{rest[:3]}/{rest[3:]}"
            return htid
        _try_merge(None, meta["id"].apply(_htid_to_path))

        # 3-segment→2-segment: "chi/086/546157" -> "chi/086546157"
        # (freqs walk produces 3-level paths but metadata uses 2-level)
        def _collapse_3seg(sid):
            parts = sid.split("/")
            if len(parts) == 3:
                return f"{parts[0]}/{parts[1]}{parts[2]}"
            return sid
        _try_merge(score_ids.apply(_collapse_3seg))

    if harmonize:
        merged = harmonize_genre(merged, corpus_name=corpus_name)
    if "year" in merged.columns:
        merged["year"] = pd.to_numeric(merged["year"], errors="coerce")
        merged = _apply_year_range(merged, snake)
    return merged


def get_score_columns(df):
    """Return the norm score column names from a merged scores DataFrame."""
    return [c for c in df.columns if c.startswith("Abs-Conc.")]


# ---------------------------------------------------------------------------
# Corpus dummy matrix
# ---------------------------------------------------------------------------

def _make_dummies(groups):
    """Build a dummy matrix from group labels (dropping the first level)."""
    groups = np.asarray(groups)
    levels = sorted(set(groups))
    if len(levels) <= 1:
        return np.zeros((len(groups), 0))
    # drop first level (reference category)
    dummies = np.column_stack([
        (groups == lvl).astype(float) for lvl in levels[1:]
    ])
    return dummies


def _ols_with_pvalue(X, s, coef_idx):
    """Fit OLS, return (beta, p-value, R²) for coefficient at coef_idx."""
    try:
        beta, residuals, rank, sv = np.linalg.lstsq(X, s, rcond=None)
    except np.linalg.LinAlgError:
        return None, np.nan, np.nan

    s_pred = X @ beta
    ss_res = np.sum((s - s_pred) ** 2)
    ss_tot = np.sum((s - s.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    n, p = X.shape
    if n > p:
        mse = ss_res / (n - p)
        try:
            cov = mse * np.linalg.inv(X.T @ X)
            se = np.sqrt(cov[coef_idx, coef_idx])
            t_stat = beta[coef_idx] / se
            pval = 2 * stats.t.sf(abs(t_stat), n - p)
        except np.linalg.LinAlgError:
            pval = np.nan
    else:
        pval = np.nan

    return beta, pval, r2


# ---------------------------------------------------------------------------
# Quadratic fit: score ~ β₀ + β₁·year + β₂·year² [+ corpus dummies]
# ---------------------------------------------------------------------------

def fit_quadratic(years, scores, groups=None):
    """Fit a quadratic model and return summary statistics.

    Parameters
    ----------
    years : array-like
        Year values.
    scores : array-like
        Score values (e.g. abstractness).
    groups : array-like, optional
        Corpus/group labels for fixed-effect dummies.

    Returns
    -------
    dict with keys: beta0, beta1, beta2, beta2_p, peak_year, r2, n
    """
    mask = np.isfinite(years) & np.isfinite(scores)
    y = np.asarray(years)[mask].astype(float)
    s = np.asarray(scores)[mask].astype(float)
    g = np.asarray(groups)[mask] if groups is not None else None
    if len(y) < 10:
        return _empty_quad()

    # center years for numerical stability
    y_center = y.mean()
    yc = y - y_center

    # design matrix: [intercept, year, year², corpus_dummies...]
    X = np.column_stack([np.ones(len(yc)), yc, yc ** 2])
    if g is not None:
        dummies = _make_dummies(g)
        if dummies.shape[1] > 0:
            X = np.column_stack([X, dummies])

    beta, beta2_p, r2 = _ols_with_pvalue(X, s, coef_idx=2)
    if beta is None:
        return _empty_quad()

    b0, b1, b2 = beta[0], beta[1], beta[2]

    # peak year (vertex of parabola)
    peak_year = -b1 / (2 * b2) + y_center if b2 != 0 else np.nan

    return {
        "quad_beta0": b0,
        "quad_beta1": b1,
        "quad_beta2": b2,
        "quad_beta2_p": beta2_p,
        "quad_peak_year": peak_year,
        "quad_r2": r2,
        "quad_n": len(y),
    }


def _empty_quad():
    return {k: np.nan for k in [
        "quad_beta0", "quad_beta1", "quad_beta2", "quad_beta2_p",
        "quad_peak_year", "quad_r2", "quad_n",
    ]}


# ---------------------------------------------------------------------------
# Piecewise linear fit: two slopes joined at a breakpoint
# ---------------------------------------------------------------------------

def fit_piecewise(years, scores, groups=None, break_year=None,
                  search_range=(1650, 1850), search_step=10):
    """Fit a piecewise linear model with one breakpoint.

    Parameters
    ----------
    years : array-like
        Year values.
    scores : array-like
        Score values.
    groups : array-like, optional
        Corpus/group labels for fixed-effect dummies.
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
    g = np.asarray(groups)[mask] if groups is not None else None
    if len(y) < 20:
        return _empty_piecewise()

    if break_year is not None:
        return _fit_piecewise_at(y, s, break_year, g)

    # grid search for best breakpoint
    best = None
    best_r2 = -np.inf
    lo, hi = search_range
    for by in range(lo, hi + 1, search_step):
        n_before = np.sum(y <= by)
        n_after = np.sum(y > by)
        if n_before < 10 or n_after < 10:
            continue
        result = _fit_piecewise_at(y, s, by, g)
        if result["pw_r2"] > best_r2:
            best_r2 = result["pw_r2"]
            best = result
    return best if best is not None else _empty_piecewise()


def _fit_piecewise_at(y, s, break_year, groups=None):
    """Fit piecewise linear at a specific breakpoint, with optional corpus dummies.

    Model: score ~ slope_before * year_before + slope_after * year_after + corpus_dummies
    where year_before = year if year <= break, else 0 (and vice versa).
    """
    before = y <= break_year
    after = y > break_year
    n_before = int(before.sum())
    n_after = int(after.sum())

    if n_before < 5 or n_after < 5:
        return _empty_piecewise()

    # Design matrix: [intercept, year_before, year_after, dummies...]
    # year_before = year - break for year <= break, 0 otherwise
    # year_after = year - break for year > break, 0 otherwise
    yb = np.where(before, y - break_year, 0.0)
    ya = np.where(after, y - break_year, 0.0)
    X = np.column_stack([np.ones(len(y)), yb, ya])
    if groups is not None:
        dummies = _make_dummies(groups)
        if dummies.shape[1] > 0:
            X = np.column_stack([X, dummies])

    try:
        beta, residuals, rank, sv = np.linalg.lstsq(X, s, rcond=None)
    except np.linalg.LinAlgError:
        return _empty_piecewise()

    slope_b, slope_a = beta[1], beta[2]
    s_pred = X @ beta
    ss_res = np.sum((s - s_pred) ** 2)
    ss_tot = np.sum((s - s.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # p-values for the two slope coefficients
    n, p = X.shape
    p_b, p_a = np.nan, np.nan
    if n > p:
        mse = ss_res / (n - p)
        try:
            cov = mse * np.linalg.inv(X.T @ X)
            se_b = np.sqrt(cov[1, 1])
            se_a = np.sqrt(cov[2, 2])
            p_b = 2 * stats.t.sf(abs(slope_b / se_b), n - p)
            p_a = 2 * stats.t.sf(abs(slope_a / se_a), n - p)
        except np.linalg.LinAlgError:
            pass

    return {
        "pw_break_year": break_year,
        "pw_slope_before": slope_b,
        "pw_slope_after": slope_a,
        "pw_slope_before_p": p_b,
        "pw_slope_after_p": p_a,
        "pw_r2": r2,
        "pw_n": len(y),
        "pw_n_before": n_before,
        "pw_n_after": n_after,
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

# Corpora to exclude from cross-corpus analyses
EXCLUDE_CORPORA = {
    "artfl",        # French
    "dta",          # German
    "evans_tcp0",   # duplicate of evans_tcp
    "oldbailey0",   # duplicate of oldbailey
    "txtlab",
    "fanfic"
}

# Per-corpus year bounds to filter outlier texts.
# Keys are snake_case corpus names; values are (min_year, max_year).
# Use None for an open bound, e.g. ("chicago", (None, 1930)).
CORPUS_YEAR_RANGE = {
    "chadwyck": (1500, 1900),
    "chadwyck_poetry": (1500, 1999),
}


def _apply_year_range(df, corpus_name, year_col="year"):
    """Filter rows outside the corpus's configured year range, if any."""
    bounds = CORPUS_YEAR_RANGE.get(corpus_name)
    if bounds is None:
        return df
    lo, hi = bounds
    if lo is not None:
        df = df[df[year_col] >= lo]
    if hi is not None:
        df = df[df[year_col] <= hi]
    return df


def fit_arc(df, score_col="Abs-Conc.Median.median", year_col="year",
            min_year=DEFAULT_MIN_YEAR, max_year=DEFAULT_MAX_YEAR,
            agg_bin=DEFAULT_AGG_BIN, min_texts_per_bin=3,
            corpus_col=None, **kw):
    """Run both quadratic and piecewise fits on a scored DataFrame.

    By default, aggregates scores by decade before fitting, giving each
    time period equal weight regardless of how many texts it contains.

    When corpus_col is provided, includes corpus fixed effects (dummy
    variables) to absorb baseline differences between corpora. In this
    mode, aggregation is by (decade, corpus) to preserve corpus identity,
    and the regressions estimate the shared time trend after controlling
    for corpus-level intercepts.

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
    corpus_col : str, optional
        Column with corpus labels. If provided, includes corpus fixed
        effects in the regression to control for baseline differences.

    Returns
    -------
    dict combining quadratic and piecewise results, plus metadata.
    """
    keep_cols = [year_col, score_col]
    if corpus_col and corpus_col in df.columns:
        keep_cols.append(corpus_col)
    sub = df[keep_cols].copy()
    sub[year_col] = pd.to_numeric(sub[year_col], errors="coerce")
    sub = sub.dropna(subset=[year_col, score_col])
    if min_year is not None:
        sub = sub[sub[year_col] >= min_year]
    if max_year is not None:
        sub = sub[sub[year_col] <= max_year]

    n_texts = len(sub)
    groups = None

    if agg_bin is not None and len(sub) > 0:
        sub["_bin"] = (sub[year_col] // agg_bin) * agg_bin
        if corpus_col and corpus_col in sub.columns:
            # aggregate by (decade, corpus) to preserve corpus identity
            agg = sub.groupby(["_bin", corpus_col]).agg(
                score=(score_col, "mean"),
                n_texts=(score_col, "count"),
            ).reset_index()
            agg = agg[agg.n_texts >= min_texts_per_bin]
            years = agg["_bin"].values.astype(float)
            scores = agg["score"].values
            groups = agg[corpus_col].values
        else:
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
        if corpus_col and corpus_col in sub.columns:
            groups = sub[corpus_col].values
        n_bins = len(sub)

    result = {"score_col": score_col, "n_texts": n_texts, "n_bins": n_bins}
    if len(years) > 0:
        result["year_min"] = int(years.min())
        result["year_max"] = int(years.max())
    else:
        result["year_min"] = np.nan
        result["year_max"] = np.nan

    if groups is not None:
        result["n_corpora"] = len(set(groups))

    result.update(fit_quadratic(years, scores, groups=groups))
    result.update(fit_piecewise(years, scores, groups=groups, **kw))
    return result


def fit_arc_corpus(corpus_name, score_col="Abs-Conc.Median.median", **kw):
    """Load scores for a corpus and fit the arc."""
    df = load_scores(corpus_name)
    result = fit_arc(df, score_col=score_col, **kw)
    result["corpus"] = corpus_name
    return result


def fit_arc_all_corpora(score_col="Abs-Conc.Median.median",
                        combined_df=None,
                        scores_dir=None, version="v7",
                        exclude=EXCLUDE_CORPORA, **kw):
    """Fit arc for all scored corpora. Returns a DataFrame of results.

    Parameters
    ----------
    combined_df : DataFrame, optional
        Pre-loaded combined DataFrame (e.g. from load_all_scored()).
        If provided, skips loading from disk.
    """
    if combined_df is not None:
        results = []
        for corpus_name, cdf in combined_df.groupby("corpus_name"):
            result = fit_arc(cdf, score_col=score_col, **kw)
            result["corpus"] = corpus_name
            results.append(result)
        return pd.DataFrame(results)

    if scores_dir is None:
        scores_dir = os.path.join(SCORES_DIR, version)
    if not os.path.isdir(scores_dir):
        raise FileNotFoundError(f"No scores directory: {scores_dir}")

    results = []
    for fn in tqdm(sorted(os.listdir(scores_dir))):
        if not fn.endswith(".csv"):
            continue
        corpus_name = fn.removesuffix(".csv")
        if corpus_name in exclude:
            continue
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
                         combined_df=None,
                         scores_dir=None, version="v7", min_texts=30,
                         corpus_fixed_effects=True,
                         exclude=EXCLUDE_CORPORA, **kw):
    """Load all scored corpora, harmonize genres, and fit arc per genre.

    Pools texts across corpora by harmonized genre, then fits one arc
    per genre. When corpus_fixed_effects=True (default), includes corpus
    dummy variables to absorb baseline differences between corpora.
    Returns a DataFrame of results.

    Parameters
    ----------
    combined_df : DataFrame, optional
        Pre-loaded combined DataFrame (e.g. from load_all_scored()).
        If provided, skips loading from disk.
    """
    if combined_df is not None:
        combined = combined_df
    else:
        combined = load_all_scored(scores_dir=scores_dir, version=version,
                                   exclude=exclude)
    if len(combined) == 0:
        return pd.DataFrame()

    if corpus_fixed_effects:
        kw["corpus_col"] = "corpus_name"
    return fit_arc_by_genre(combined, score_col=score_col,
                            min_texts=min_texts, **kw)


def load_all_scored(scores_dir=None, version="v7", exclude=EXCLUDE_CORPORA):
    """Load all scored corpora, harmonize genres, return a combined DataFrame.

    This is the data-loading step extracted from fit_arc_all_by_genre,
    useful when you need the underlying data (not just fit summaries).
    """
    if scores_dir is None:
        scores_dir = os.path.join(SCORES_DIR, version)
    if not os.path.isdir(scores_dir):
        raise FileNotFoundError(f"No scores directory: {scores_dir}")

    all_dfs = []
    iterr = tqdm(sorted(os.listdir(scores_dir)))
    for fn in iterr:
        if not fn.endswith(".csv"):
            continue
        corpus_name = fn.removesuffix(".csv")
        iterr.set_description(f"Loading {corpus_name}")
        if corpus_name in exclude:
            continue
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
    if "year" in combined.columns:
        combined["year"] = pd.to_numeric(combined["year"], errors="coerce")
    return combined


def adjust_scores(df, score_col="Abs-Conc.Median.median", year_col="year",
                  corpus_col="corpus_name", min_year=DEFAULT_MIN_YEAR,
                  max_year=DEFAULT_MAX_YEAR, agg_bin=DEFAULT_AGG_BIN,
                  min_texts_per_bin=3, model="quadratic"):
    """Fit a regression with corpus fixed effects and return adjusted scores.

    Returns a DataFrame with columns:
        year, score (raw), adjusted (corpus-corrected), fitted (trend line),
        corpus, n_texts
    The 'adjusted' column removes corpus-specific intercepts while preserving
    the shared time trend: adjusted = score - corpus_effect.
    The 'fitted' column is the predicted trend (intercept + time terms only).

    Parameters
    ----------
    model : str
        "quadratic" or "piecewise". Determines the trend shape.
    """
    keep_cols = [year_col, score_col]
    if corpus_col and corpus_col in df.columns:
        keep_cols.append(corpus_col)
    sub = df[keep_cols].copy()
    sub[year_col] = pd.to_numeric(sub[year_col], errors="coerce")
    sub = sub.dropna(subset=[year_col, score_col])
    if min_year is not None:
        sub = sub[sub[year_col] >= min_year]
    if max_year is not None:
        sub = sub[sub[year_col] <= max_year]

    if len(sub) == 0:
        return pd.DataFrame()

    # Aggregate by (decade, corpus)
    sub["_bin"] = (sub[year_col] // agg_bin) * agg_bin
    if corpus_col and corpus_col in sub.columns:
        agg = sub.groupby(["_bin", corpus_col]).agg(
            score=(score_col, "mean"),
            n_texts=(score_col, "count"),
        ).reset_index()
        agg = agg[agg.n_texts >= min_texts_per_bin]
        groups = agg[corpus_col].values
    else:
        agg = sub.groupby("_bin").agg(
            score=(score_col, "mean"),
            n_texts=(score_col, "count"),
        ).reset_index()
        agg = agg[agg.n_texts >= min_texts_per_bin]
        groups = None

    years = agg["_bin"].values.astype(float)
    scores = agg["score"].values

    mask = np.isfinite(years) & np.isfinite(scores)
    y = years[mask]
    s = scores[mask]
    g = groups[mask] if groups is not None else None
    agg_masked = agg[mask].copy()

    if len(y) < 10:
        return pd.DataFrame()

    def _fit_and_adjust(X_trend, X, s, n_trend_cols):
        """Fit OLS, return (fitted, adjusted, fitted_se)."""
        try:
            beta, _, _, _ = np.linalg.lstsq(X, s, rcond=None)
        except np.linalg.LinAlgError:
            return None
        fitted = X_trend @ beta[:n_trend_cols]
        if X.shape[1] > n_trend_cols:
            corpus_effect = X[:, n_trend_cols:] @ beta[n_trend_cols:]
        else:
            corpus_effect = np.zeros(len(s))
        adjusted = s - corpus_effect

        # Standard error of the fitted trend
        resid = s - X @ beta
        n, p = X.shape
        mse = (resid ** 2).sum() / max(n - p, 1)
        try:
            XtX_inv = np.linalg.inv(X.T @ X)
        except np.linalg.LinAlgError:
            XtX_inv = np.linalg.pinv(X.T @ X)
        # Variance of fitted = X_trend @ Cov(beta_trend) @ X_trend'
        # but beta_trend covariance is the top-left block of mse * (X'X)^-1
        cov_trend = mse * XtX_inv[:n_trend_cols, :n_trend_cols]
        # Per-point SE: sqrt(x_i @ cov_trend @ x_i')
        fitted_se = np.sqrt(np.sum((X_trend @ cov_trend) * X_trend, axis=1))
        return fitted, adjusted, fitted_se

    _POLY_MODELS = {"quadratic": 2, "cubic": 3, "quartic": 4}

    if model in _POLY_MODELS:
        degree = _POLY_MODELS[model]
        y_center = y.mean()
        yc = y - y_center

        # Design matrix: [intercept, year, year², ..., year^d, corpus_dummies...]
        X_trend = np.column_stack([yc ** i for i in range(degree + 1)])
        if g is not None:
            dummies = _make_dummies(g)
            X = np.column_stack([X_trend, dummies]) if dummies.shape[1] > 0 else X_trend
        else:
            X = X_trend

        fit_result = _fit_and_adjust(X_trend, X, s, degree + 1)
        if fit_result is None:
            return pd.DataFrame()
        fitted, adjusted, fitted_se = fit_result

    elif model == "piecewise":
        # First find the best breakpoint
        pw_result = fit_piecewise(y, s, groups=g)
        break_year = pw_result.get("pw_break_year", np.nan)
        if not np.isfinite(break_year):
            return pd.DataFrame()

        before = y <= break_year
        after = y > break_year
        yb = np.where(before, y - break_year, 0.0)
        ya = np.where(after, y - break_year, 0.0)

        X_trend = np.column_stack([np.ones(len(y)), yb, ya])
        if g is not None:
            dummies = _make_dummies(g)
            X = np.column_stack([X_trend, dummies]) if dummies.shape[1] > 0 else X_trend
        else:
            X = X_trend

        fit_result = _fit_and_adjust(X_trend, X, s, 3)
        if fit_result is None:
            return pd.DataFrame()
        fitted, adjusted, fitted_se = fit_result
    else:
        raise ValueError(f"Unknown model: {model!r} (use 'quadratic', 'cubic', 'quartic', or 'piecewise')")

    # Build result DataFrame
    result = pd.DataFrame({
        "year": y,
        "score": s,
        "adjusted": adjusted,
        "fitted": fitted,
        "fitted_se": fitted_se,
        "n_texts": agg_masked["n_texts"].values,
    })
    if g is not None:
        result["corpus"] = g

    return result


def summarize_arc(result):
    """Format an arc result dict as a human-readable string."""
    lines = []
    corpus = result.get("corpus", "?")
    n_texts = result.get('n_texts', result.get('n', '?'))
    n_bins = result.get('n_bins', '')
    n_corpora = result.get('n_corpora', '')
    bin_str = f", {n_bins} bins" if n_bins and n_bins != n_texts else ""
    corp_str = f", {n_corpora} corpora" if n_corpora else ""
    lines.append(f"=== {corpus} ({n_texts} texts{bin_str}{corp_str}, {result.get('year_min', '?')}-{result.get('year_max', '?')}) ===")

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
