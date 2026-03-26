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
from .utils import read_df


# ---------------------------------------------------------------------------
# Loading and merging scores with metadata
# ---------------------------------------------------------------------------

def load_scores(corpus_name, scores_dir=None, version="v7"):
    """Load scored texts for a corpus and merge with metadata.

    Returns a DataFrame with score columns plus metadata (year, genre, etc.).
    """
    if scores_dir is None:
        scores_dir = os.path.join(SCORES_DIR, version)
    snake = _camel_to_snake(corpus_name) if corpus_name[0].isupper() else corpus_name
    path = os.path.join(scores_dir, f"{snake}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No scores file: {path}")
    scores = pd.read_csv(path)
    corpus = load_corpus(corpus_name)
    merged = scores.merge(corpus.metadata, on="id", how="inner")
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

def fit_arc(df, score_col="Abs-Conc.Median.median", year_col="year",
            min_year=None, max_year=None, **kw):
    """Run both quadratic and piecewise fits on a scored DataFrame.

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

    Returns
    -------
    dict combining quadratic and piecewise results, plus metadata.
    """
    sub = df[[year_col, score_col]].dropna()
    if min_year is not None:
        sub = sub[sub[year_col] >= min_year]
    if max_year is not None:
        sub = sub[sub[year_col] <= max_year]

    years = sub[year_col].values
    scores = sub[score_col].values

    result = {"score_col": score_col, "n": len(sub)}
    if len(sub) > 0:
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


def summarize_arc(result):
    """Format an arc result dict as a human-readable string."""
    lines = []
    corpus = result.get("corpus", "?")
    lines.append(f"=== {corpus} ({result.get('n', '?')} texts, {result.get('year_min', '?')}-{result.get('year_max', '?')}) ===")

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
