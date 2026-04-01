"""
Arc analysis: detecting and quantifying the rise-and-fall pattern
of abstract language across literary history.
"""

import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy import stats

import json
from .config import COUNT_DIR, SCORES_DIR, PATH_CORPORA
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


def _merge_with_metadata(df, corpus_name, harmonize=True):
    """Merge a DataFrame (with 'id' column) against corpus metadata.

    Handles ID format mismatches (slash→dot, zero-padding, htid→path, etc.)
    and applies genre harmonization, year fixes, and year-range filtering.
    """
    snake = _camel_to_snake(corpus_name) if corpus_name[0].isupper() else corpus_name
    corpus = load_corpus(corpus_name)
    meta = corpus.metadata

    id_col = _find_id_col(meta)
    if id_col != "id":
        meta = meta.rename(columns={id_col: "id"})

    df["id"] = df["id"].astype(str)
    meta["id"] = meta["id"].astype(str)

    merged = df.merge(meta, on="id", how="inner")

    def _try_merge(score_ids, meta_ids_col=None):
        nonlocal merged
        s = df.copy()
        m = meta
        if score_ids is not None:
            s["id"] = score_ids
        if meta_ids_col is not None:
            m = meta.copy()
            m["id"] = meta_ids_col
        candidate = s.merge(m, on="id", how="inner")
        if len(candidate) > len(merged):
            merged = candidate

    if len(merged) < len(df) * 0.5 and len(df) > 0:
        score_ids = df["id"]

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
        def _collapse_3seg(sid):
            parts = sid.split("/")
            if len(parts) == 3:
                return f"{parts[0]}/{parts[1]}{parts[2]}"
            return sid
        _try_merge(score_ids.apply(_collapse_3seg))

    if harmonize:
        merged = harmonize_genre(merged, corpus_name=corpus_name)
    if snake == "chadwyck_poetry":
        merged = _fix_chadwyck_poetry_year(merged)
    if "year" in merged.columns:
        merged["year"] = pd.to_numeric(merged["year"], errors="coerce")
        merged = _apply_year_range(merged, snake)
    return merged


def load_scores(corpus_name, scores_dir=None, version="v8-raw", harmonize=True):
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
    return _merge_with_metadata(scores, corpus_name, harmonize=harmonize)


def _fix_chadwyck_poetry_year(df):
    """Re-estimate year for chadwyck_poetry using publication date.

    The raw metadata sets year = author_dob + 30 for every row.  This
    replaces it with attpubn1 (actual publication year) when that date
    falls within the author's lifetime (DOB–DOD), and drops rows where
    it doesn't or where DOD is unknown for authors born before 1950.
    """
    needed = {"author_dob", "author_dod", "attpubn1"}
    if not needed.issubset(df.columns):
        return df

    dob = pd.to_numeric(df["author_dob"], errors="coerce")
    dod = pd.to_numeric(df["author_dod"], errors="coerce")
    pub = pd.to_numeric(df["attpubn1"], errors="coerce")

    has_dod = dod.notna() & (dod > 0)
    has_pub = pub.notna() & (pub > 0)

    # Drop: no DOD and born before 1950 (can't verify pub date)
    drop_no_dod = ~has_dod & (dob < 1950)

    # Drop: has DOD + pub year but pub year outside lifetime
    outside_lifetime = has_dod & has_pub & ((pub < dob) | (pub > dod))

    keep = ~drop_no_dod & ~outside_lifetime & has_pub
    out = df[keep].copy()
    out["year"] = pub[keep]
    return out


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


def report_piecewise(combined_df, genres=None,
                     score_col="Abs-Conc.Median.median",
                     corpus_col="corpus_name",
                     min_year=1600, max_year=2020,
                     agg_bin=10, min_texts_per_bin=3,
                     search_range=(1650, 1850), search_step=10,
                     invert=True):
    """Report piecewise regression statistics per genre.

    Returns a DataFrame with one row per genre: breakpoint, slopes,
    slope SEs, p-values, R², and sample sizes.
    """
    if genres is None:
        gcounts = combined_df["genre_harmonized"].value_counts()
        genres = gcounts[gcounts >= 30].index.tolist()

    rows = []
    for genre in genres:
        gdf = combined_df[combined_df["genre_harmonized"] == genre]
        # Aggregate by (decade, corpus)
        sub = gdf[[score_col, "year", corpus_col]].copy()
        sub["year"] = pd.to_numeric(sub["year"], errors="coerce")
        sub = sub.dropna(subset=["year", score_col])
        sub = sub[(sub["year"] >= min_year) & (sub["year"] <= max_year)]
        if len(sub) < 30:
            continue
        sub["_bin"] = (sub["year"] // agg_bin) * agg_bin
        agg = sub.groupby(["_bin", corpus_col]).agg(
            score=(score_col, "mean"),
            n_texts=(score_col, "count"),
        ).reset_index()
        agg = agg[agg.n_texts >= min_texts_per_bin]
        if len(agg) < 20:
            continue

        y = agg["_bin"].values.astype(float)
        s = agg["score"].values
        if invert:
            s = -s
        g = agg[corpus_col].values

        pw = fit_piecewise(y, s, groups=g,
                           search_range=search_range,
                           search_step=search_step)

        # Also compute slope SEs from the full fit for reporting
        break_year = pw.get("pw_break_year", np.nan)
        se_before = np.nan
        se_after = np.nan
        if np.isfinite(break_year):
            before = y <= break_year
            after = y > break_year
            yb = np.where(before, y - break_year, 0.0)
            ya = np.where(after, y - break_year, 0.0)
            X = np.column_stack([np.ones(len(y)), yb, ya])
            dummies = _make_dummies(g)
            if dummies.shape[1] > 0:
                X = np.column_stack([X, dummies])
            try:
                beta, _, _, _ = np.linalg.lstsq(X, s, rcond=None)
                resid = s - X @ beta
                n, p = X.shape
                mse = (resid ** 2).sum() / max(n - p, 1)
                cov = mse * np.linalg.inv(X.T @ X)
                se_before = np.sqrt(cov[1, 1])
                se_after = np.sqrt(cov[2, 2])
            except np.linalg.LinAlgError:
                pass

        rows.append({
            "genre": genre,
            "breakpoint": pw["pw_break_year"],
            "slope_before": pw["pw_slope_before"],
            "slope_before_se": se_before,
            "slope_before_p": pw["pw_slope_before_p"],
            "slope_after": pw["pw_slope_after"],
            "slope_after_se": se_after,
            "slope_after_p": pw["pw_slope_after_p"],
            "r2": pw["pw_r2"],
            "n_total": pw["pw_n"],
            "n_before": pw["pw_n_before"],
            "n_after": pw["pw_n_after"],
            "n_texts_total": int(sub["n_texts"].sum()) if "n_texts" in sub.columns else len(sub),
        })

    return pd.DataFrame(rows)


def report_arc(combined_df=None, genres=None,
               score_col="Abs-Conc.Median.median",
               corpus_col="corpus_name",
               min_year=1600, max_year=2020,
               agg_bin=10, min_texts_per_bin=3,
               search_range=(1650, 1850), search_step=10,
               print_result=True):
    """Report piecewise arc statistics with ratios for each genre.

    Computes piecewise regression, raw decade means at key decades
    (start trough, peak, end trough), and abstractness ratios using
    raw scores. Ratios are only reported when both values have the
    same sign (i.e. both abstract or both concrete); NaN otherwise.

    Parameters
    ----------
    combined_df : DataFrame, optional
        Pre-loaded combined scored DataFrame. If None, calls load_all_scored().
    genres : list, optional
        Genres to report. Default: Fiction, Poetry, Periodical.
    print_result : bool
        If True, print a prose summary alongside the DataFrame.

    Returns
    -------
    DataFrame with one row per genre.
    """
    if combined_df is None:
        combined_df = load_all_scored()

    if genres is None:
        genres = ["Fiction", "Poetry", "Periodical"]

    # --- Collect raw decade means per genre (inverted: abstractness up) ---
    genre_dec = {}
    for genre in genres:
        gdf = combined_df[combined_df["genre_harmonized"] == genre].copy()
        gdf["year"] = pd.to_numeric(gdf["year"], errors="coerce")
        gdf = gdf.dropna(subset=["year", score_col])
        gdf = gdf[(gdf["year"] >= min_year) & (gdf["year"] <= max_year)]
        gdf["decade"] = (gdf["year"] // agg_bin) * agg_bin
        dec = -gdf.groupby("decade")[score_col].mean()
        dec_n = gdf.groupby("decade")[score_col].count()
        genre_dec[genre] = (dec, dec_n, gdf)

    rows = []
    prose_lines = []
    for genre in genres:
        dec_raw, dec_n, gdf = genre_dec[genre]
        text_sd = gdf[score_col].std()

        # Piecewise fit (on aggregated, corpus-adjusted data) to find key decades
        adj = adjust_scores(gdf, score_col=score_col, corpus_col=corpus_col,
                            min_year=min_year, max_year=max_year,
                            agg_bin=agg_bin, min_texts_per_bin=min_texts_per_bin,
                            model="piecewise")
        if adj.empty:
            continue
        adj_dec = -adj.groupby("year")["adjusted"].mean()

        peak_yr = int(adj_dec.idxmax())
        before = adj_dec[adj_dec.index <= peak_yr]
        after = adj_dec[adj_dec.index >= peak_yr]
        start_yr = int(before.idxmin())
        end_yr = int(after.idxmin())

        # Piecewise regression stats
        sub = gdf[["year", score_col, corpus_col]].copy()
        sub["_bin"] = (sub["year"] // agg_bin) * agg_bin
        agg = sub.groupby(["_bin", corpus_col]).agg(
            score=(score_col, "mean"),
            n_texts=(score_col, "count"),
        ).reset_index()
        agg = agg[agg.n_texts >= min_texts_per_bin]
        y = agg["_bin"].values.astype(float)
        s = -agg["score"].values  # invert
        g = agg[corpus_col].values
        pw = fit_piecewise(y, s, groups=g,
                           search_range=search_range,
                           search_step=search_step)

        # Slope SEs
        break_year = pw.get("pw_break_year", np.nan)
        se_before = se_after = np.nan
        if np.isfinite(break_year):
            bm = y <= break_year
            am = y > break_year
            yb = np.where(bm, y - break_year, 0.0)
            ya = np.where(am, y - break_year, 0.0)
            X = np.column_stack([np.ones(len(y)), yb, ya])
            dummies = _make_dummies(g)
            if dummies.shape[1] > 0:
                X = np.column_stack([X, dummies])
            try:
                beta, _, _, _ = np.linalg.lstsq(X, s, rcond=None)
                resid = s - X @ beta
                n, p = X.shape
                mse = (resid ** 2).sum() / max(n - p, 1)
                cov = mse * np.linalg.inv(X.T @ X)
                se_before = np.sqrt(cov[1, 1])
                se_after = np.sqrt(cov[2, 2])
            except np.linalg.LinAlgError:
                pass

        # Raw values at key decades
        raw_start = float(dec_raw[start_yr])
        raw_peak = float(dec_raw[peak_yr])
        raw_end = float(dec_raw[end_yr])

        n_start = int(dec_n[start_yr])
        n_peak = int(dec_n[peak_yr])
        n_end = int(dec_n[end_yr])

        rise_sd = (raw_peak - raw_start) / text_sd
        fall_sd = (raw_peak - raw_end) / text_sd

        def _safe_ratio(a, b):
            """Return a/b only when both have the same sign; NaN otherwise."""
            if a * b > 0:
                return a / b
            return np.nan

        rows.append({
            "genre": genre,
            "breakpoint": pw["pw_break_year"],
            "start_decade": start_yr,
            "peak_decade": peak_yr,
            "end_decade": end_yr,
            "raw_start": raw_start,
            "raw_peak": raw_peak,
            "raw_end": raw_end,
            "rise_sd": rise_sd,
            "fall_sd": fall_sd,
            "peak_vs_start": _safe_ratio(raw_peak, raw_start),
            "peak_vs_end": _safe_ratio(raw_peak, raw_end),
            "start_vs_end": _safe_ratio(raw_start, raw_end),
            "slope_before": pw["pw_slope_before"],
            "slope_before_se": se_before,
            "slope_before_p": pw["pw_slope_before_p"],
            "slope_after": pw["pw_slope_after"],
            "slope_after_se": se_after,
            "slope_after_p": pw["pw_slope_after_p"],
            "r2": pw["pw_r2"],
            "n_bins": pw["pw_n"],
            "n_texts_start": n_start,
            "n_texts_peak": n_peak,
            "n_texts_end": n_end,
            "n_texts_total": len(gdf),
        })

        if print_result:
            ratio_peak_start = _safe_ratio(raw_peak, raw_start)
            ratio_peak_end = _safe_ratio(raw_peak, raw_end)
            ratio_str_rise = f"{ratio_peak_start:.2f}x" if np.isfinite(ratio_peak_start) else "N/A (sign change)"
            ratio_str_fall = f"{ratio_peak_end:.2f}x" if np.isfinite(ratio_peak_end) else "N/A (sign change)"
            prose_lines.append(
                f"{genre}: abstractness rises from {raw_start:.4f} in the "
                f"{start_yr}s (n={n_start:,}) to {raw_peak:.4f} in the "
                f"{peak_yr}s (n={n_peak:,}): {ratio_str_rise}, "
                f"+{rise_sd:.2f} SD. Then falls to {raw_end:.4f} in the "
                f"{end_yr}s (n={n_end:,}): peak vs end {ratio_str_fall}. "
                f"Piecewise breakpoint at {int(pw['pw_break_year'])}; "
                f"rise slope = {pw['pw_slope_before']:+.4f}/decade "
                f"(p = {pw['pw_slope_before_p']:.1e}), "
                f"fall slope = {pw['pw_slope_after']:+.4f}/decade "
                f"(p = {pw['pw_slope_after_p']:.1e}); "
                f"R² = {pw['pw_r2']:.3f}; "
                f"n = {len(gdf):,} texts."
            )

    df = pd.DataFrame(rows)

    if print_result and prose_lines:
        print()
        for line in prose_lines:
            print(line)
            print()
        print(f"(Ratios use raw abstractness scores; reported only when "
              f"both values have the same sign.)")

    return df


# ---------------------------------------------------------------------------
# Combined arc analysis
# ---------------------------------------------------------------------------

DEFAULT_MIN_YEAR = 1600
DEFAULT_MAX_YEAR = 2000
DEFAULT_AGG_BIN = 10  # aggregate by decade

# Corpora that contribute to the three main genres (Fiction, Poetry, Periodical).
# Used as the default for score-corpora and arc analysis.
ARC_CORPORA = [
    # Fiction
    "canon_fiction", "chadwyck", "chicago", "gale_amfic", "gildedage",
    "hathi_englit", "internet_archive", "litlab", "long_arc_prestige", "markmark",
    # Poetry
    "chadwyck_poetry", "eebo_tcp",
    # Periodical
    "bpo", "coha", "new_yorker", "spectator",
    # Other (contribute texts to multiple genres via metadata)
    "clmet", "sellers", "tedjdh",
]

# Corpora to exclude from cross-corpus analyses
EXCLUDE_CORPORA = {
    "artfl",        # French
    "dta",          # German
    "evans_tcp0",   # duplicate of evans_tcp
    "oldbailey0",   # duplicate of oldbailey
    "txtlab",
    "fanfic",
    "coca"
    # "chadwyck_poetry",
}

# Per-corpus year bounds to filter outlier texts.
# Keys are snake_case corpus names; values are (min_year, max_year).
# Use None for an open bound, e.g. ("chicago", (None, 1930)).
CORPUS_YEAR_RANGE = {
    "chadwyck": (1500, 1900),
    "chadwyck_poetry": (1500, 2020),
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
                        scores_dir=None, version="v8-raw",
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
                         scores_dir=None, version="v8-raw", min_texts=30,
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


def load_all_scored(scores_dir=None, version="v8-raw", exclude=EXCLUDE_CORPORA):
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


# ---------------------------------------------------------------------------
# Count-based analysis (proportions from z-score bin counts)
# ---------------------------------------------------------------------------

def _load_counts_jsonl(path):
    """Load a JSONL counts file into a list of dicts."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def pct_in_range(rec, norm="Abs-Conc.Median.median", lo=None, hi=None):
    """Compute the proportion of words in a z-score range for one text.

    Parameters
    ----------
    rec : dict
        A single JSONL record with norm -> {bin_edge: count} structure.
    norm : str
        Norm column name.
    lo, hi : float, optional
        Z-score bounds (inclusive). If lo is None, no lower bound.
        If hi is None, no upper bound.

    Returns
    -------
    float or NaN — proportion of words in [lo, hi].
    """
    bins = rec.get(norm)
    if not bins:
        return np.nan
    total = 0
    in_range = 0
    for edge_str, count in bins.items():
        edge = float(edge_str)
        total += count
        if (lo is None or edge <= lo) if hi is None else \
           (hi is None or edge > hi) if lo is None else \
           False:
            pass
        # Simpler logic:
        include = True
        if hi is not None and edge > hi:
            include = False
        if lo is not None and edge <= lo:
            include = False
        if include:
            in_range += count
    if total == 0:
        return np.nan
    return in_range / total


def pct_abstract(rec, norm="Abs-Conc.Median.median", cutoff=-1.0):
    """Proportion of words with z-score ≤ cutoff (abstract words)."""
    bins = rec.get(norm)
    if not bins:
        return np.nan
    total = sum(bins.values())
    if total == 0:
        return np.nan
    abstract = sum(c for e, c in bins.items() if float(e) <= cutoff)
    return abstract / total


def pct_concrete(rec, norm="Abs-Conc.Median.median", cutoff=1.0):
    """Proportion of words with z-score > cutoff (concrete words)."""
    bins = rec.get(norm)
    if not bins:
        return np.nan
    total = sum(bins.values())
    if total == 0:
        return np.nan
    concrete = sum(c for e, c in bins.items() if float(e) > cutoff)
    return concrete / total


def load_counts(corpus_name, counts_dir=None, version="v2-raw",
                norm="Abs-Conc.Median.median",
                abs_cutoff=-1.0, conc_cutoff=1.0,
                harmonize=True):
    """Load counts for a corpus, compute pct_abstract and pct_concrete.

    Parameters
    ----------
    abs_cutoff : float
        Z-score cutoff for abstract words (z ≤ abs_cutoff). Default -1.0.
    conc_cutoff : float
        Z-score cutoff for concrete words (z > conc_cutoff). Default 1.0.

    Returns a DataFrame with columns: id, pct_abstract, pct_concrete,
    n_words, year, genre_harmonized, corpus_name, etc.
    """
    if counts_dir is None:
        counts_dir = os.path.join(COUNT_DIR, version)
    snake = _camel_to_snake(corpus_name) if corpus_name[0].isupper() else corpus_name
    path = os.path.join(counts_dir, f"{snake}.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No counts file: {path}")

    records = _load_counts_jsonl(path)
    rows = []
    for rec in records:
        pa = pct_abstract(rec, norm=norm, cutoff=abs_cutoff)
        pc = pct_concrete(rec, norm=norm, cutoff=conc_cutoff)
        total = sum(rec.get(norm, {}).values())
        rows.append({
            "id": rec["id"],
            "pct_abstract": pa,
            "pct_concrete": pc,
            "n_words": total,
        })
    df = pd.DataFrame(rows)
    return _merge_with_metadata(df, corpus_name, harmonize=harmonize)


def load_all_counts(counts_dir=None, version="v2-raw",
                    norm="Abs-Conc.Median.median",
                    abs_cutoff=-1.0, conc_cutoff=1.0,
                    exclude=EXCLUDE_CORPORA):
    """Load counts for all corpora, return a combined DataFrame.

    Each text gets pct_abstract and pct_concrete at the given cutoffs.
    """
    if counts_dir is None:
        counts_dir = os.path.join(COUNT_DIR, version)
    if not os.path.isdir(counts_dir):
        raise FileNotFoundError(f"No counts directory: {counts_dir}")

    all_dfs = []
    iterr = tqdm(sorted(os.listdir(counts_dir)))
    for fn in iterr:
        if not fn.endswith(".jsonl"):
            continue
        corpus_name = fn.removesuffix(".jsonl")
        iterr.set_description(f"Loading {corpus_name}")
        if corpus_name in exclude:
            continue
        try:
            df = load_counts(corpus_name, counts_dir=counts_dir,
                             version=version, norm=norm,
                             abs_cutoff=abs_cutoff, conc_cutoff=conc_cutoff)
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


def _report_one_measure(genre, gdf, score_col, corpus_col, label,
                        min_year, max_year, agg_bin, min_texts_per_bin,
                        search_range, search_step):
    """Run piecewise fit for one genre and one measure (abstract or concrete).

    Returns (row_dict, prose_string) or (None, None) if insufficient data.
    """
    sub_gdf = gdf.dropna(subset=[score_col])
    if len(sub_gdf) < 30:
        return None, None

    sub_gdf = sub_gdf.copy()
    sub_gdf["decade"] = (sub_gdf["year"] // agg_bin) * agg_bin
    dec_mean = sub_gdf.groupby("decade")[score_col].mean()
    dec_n = sub_gdf.groupby("decade")[score_col].count()

    adj = adjust_scores(sub_gdf, score_col=score_col, corpus_col=corpus_col,
                        min_year=min_year, max_year=max_year,
                        agg_bin=agg_bin, min_texts_per_bin=min_texts_per_bin,
                        model="piecewise")
    if adj.empty:
        return None, None

    adj_dec = adj.groupby("year")["adjusted"].mean()
    peak_yr = int(adj_dec.idxmax())
    trough_before = adj_dec[adj_dec.index <= peak_yr]
    trough_after = adj_dec[adj_dec.index >= peak_yr]
    start_yr = int(trough_before.idxmin())
    end_yr = int(trough_after.idxmin())

    # Piecewise stats
    sub = sub_gdf[["year", score_col, corpus_col]].copy()
    sub["_bin"] = (sub["year"] // agg_bin) * agg_bin
    agg = sub.groupby(["_bin", corpus_col]).agg(
        score=(score_col, "mean"),
        n_texts=(score_col, "count"),
    ).reset_index()
    agg = agg[agg.n_texts >= min_texts_per_bin]
    y = agg["_bin"].values.astype(float)
    s = agg["score"].values
    g = agg[corpus_col].values
    pw = fit_piecewise(y, s, groups=g,
                       search_range=search_range,
                       search_step=search_step)

    pct_start = dec_mean.get(start_yr, np.nan) * 100
    pct_peak = dec_mean.get(peak_yr, np.nan) * 100
    pct_end = dec_mean.get(end_yr, np.nan) * 100

    def _ratio(a, b):
        return a / b if b > 0 else np.nan

    peak_vs_start = _ratio(pct_peak, pct_start)
    peak_vs_end = _ratio(pct_peak, pct_end)
    start_vs_end = _ratio(pct_start, pct_end)

    row = {
        f"{label}_breakpoint": pw["pw_break_year"],
        f"{label}_start_decade": start_yr,
        f"{label}_peak_decade": peak_yr,
        f"{label}_end_decade": end_yr,
        f"{label}_pct_start": pct_start,
        f"{label}_pct_peak": pct_peak,
        f"{label}_pct_end": pct_end,
        f"{label}_peak_vs_start": peak_vs_start,
        f"{label}_peak_vs_end": peak_vs_end,
        f"{label}_start_vs_end": start_vs_end,
        f"{label}_slope_before": pw["pw_slope_before"],
        f"{label}_slope_before_p": pw["pw_slope_before_p"],
        f"{label}_slope_after": pw["pw_slope_after"],
        f"{label}_slope_after_p": pw["pw_slope_after_p"],
        f"{label}_r2": pw["pw_r2"],
        f"{label}_n_texts_start": int(dec_n.get(start_yr, 0)),
        f"{label}_n_texts_peak": int(dec_n.get(peak_yr, 0)),
        f"{label}_n_texts_end": int(dec_n.get(end_yr, 0)),
    }

    prose = (
        f"  {label.capitalize()}:\n"
        f"    {start_yr}s: {pct_start:.1f}%  →  {peak_yr}s: {pct_peak:.1f}% (peak)  →  {end_yr}s: {pct_end:.1f}%\n"
        f"    Rise:  {pct_start:.1f}% → {pct_peak:.1f}% = {peak_vs_start:.1f}x ({start_yr}s→{peak_yr}s)\n"
        f"    Fall:  {pct_peak:.1f}% → {pct_end:.1f}% = {peak_vs_end:.1f}x ({peak_yr}s→{end_yr}s)\n"
        f"    Net:   {pct_start:.1f}% → {pct_end:.1f}% = {start_vs_end:.1f}x ({start_yr}s→{end_yr}s)\n"
        f"    Breakpoint {int(pw['pw_break_year'])}; R² = {pw['pw_r2']:.3f}"
    )
    return row, prose


def report_arc_counts(combined_df=None, genres=None,
                      norm="Abs-Conc.Median.median",
                      abs_cutoff=-1.0, conc_cutoff=1.0,
                      corpus_col="corpus_name",
                      min_year=DEFAULT_MIN_YEAR, max_year=2020,
                      agg_bin=10, min_texts_per_bin=3,
                      search_range=(1650, 1850), search_step=10,
                      print_result=True):
    """Report piecewise arc statistics for both abstract and concrete proportions.

    Uses pct_abstract (z ≤ abs_cutoff) and pct_concrete (z > conc_cutoff),
    giving ratio-scale values where ratios are meaningful.

    Parameters
    ----------
    combined_df : DataFrame, optional
        Pre-loaded from load_all_counts(). If None, loads automatically.
    genres : list, optional
        Genres to report. Default: Fiction, Poetry, Periodical.
    abs_cutoff : float
        Z-score cutoff for abstract words (default -1.0).
    conc_cutoff : float
        Z-score cutoff for concrete words (default 1.0).
    print_result : bool
        If True, print prose summary.

    Returns
    -------
    DataFrame with one row per genre, columns for both abstract and concrete.
    """
    if combined_df is None:
        combined_df = load_all_counts(norm=norm, abs_cutoff=abs_cutoff,
                                      conc_cutoff=conc_cutoff)

    if genres is None:
        genres = ["Fiction", "Poetry", "Periodical"]

    fit_kw = dict(min_year=min_year, max_year=max_year,
                  agg_bin=agg_bin, min_texts_per_bin=min_texts_per_bin,
                  search_range=search_range, search_step=search_step)

    rows = []
    prose_lines = []

    for genre in genres:
        gdf = combined_df[combined_df["genre_harmonized"] == genre].copy()
        gdf["year"] = pd.to_numeric(gdf["year"], errors="coerce")
        gdf = gdf[(gdf["year"] >= min_year) & (gdf["year"] <= max_year)]
        if len(gdf) < 30:
            continue

        row = {"genre": genre, "n_texts": len(gdf)}

        # Run piecewise on abstract to find the key decades
        abs_row, abs_prose = _report_one_measure(
            genre, gdf, "pct_abstract", corpus_col, "abstract", **fit_kw)
        conc_row, conc_prose = _report_one_measure(
            genre, gdf, "pct_concrete", corpus_col, "concrete", **fit_kw)

        if abs_row:
            row.update(abs_row)
        if conc_row:
            row.update(conc_row)

        # Cross-measure: concrete values at the abstract key decades
        if abs_row:
            gdf_c = gdf.copy()
            gdf_c["decade"] = (gdf_c["year"] // agg_bin) * agg_bin
            abs_start = int(abs_row["abstract_start_decade"])
            abs_peak = int(abs_row["abstract_peak_decade"])
            abs_end = int(abs_row["abstract_end_decade"])
            conc_dec = gdf_c.groupby("decade")["pct_concrete"].mean() * 100

            conc_at_abs_start = conc_dec.get(abs_start, np.nan)
            conc_at_abs_peak = conc_dec.get(abs_peak, np.nan)
            conc_at_abs_end = conc_dec.get(abs_end, np.nan)

            row["conc_at_abs_start"] = conc_at_abs_start
            row["conc_at_abs_peak"] = conc_at_abs_peak
            row["conc_at_abs_end"] = conc_at_abs_end

            # Abstract-to-concrete ratio at key decades
            def _ratio(a, c):
                return a / c if c > 0 else np.nan
            row["abs_conc_ratio_start"] = _ratio(abs_row["abstract_pct_start"], conc_at_abs_start)
            row["abs_conc_ratio_peak"] = _ratio(abs_row["abstract_pct_peak"], conc_at_abs_peak)
            row["abs_conc_ratio_end"] = _ratio(abs_row["abstract_pct_end"], conc_at_abs_end)

        if print_result and abs_row:
            a_s = abs_row["abstract_pct_start"]
            a_p = abs_row["abstract_pct_peak"]
            a_e = abs_row["abstract_pct_end"]
            c_s = conc_at_abs_start
            c_p = conc_at_abs_peak
            c_e = conc_at_abs_end
            r_s = row["abs_conc_ratio_start"]
            r_p = row["abs_conc_ratio_peak"]
            r_e = row["abs_conc_ratio_end"]

            def _r(a, b):
                return a / b if b > 0 else np.nan

            def _fmt_ratio(r):
                """Format abs/conc ratio, showing inverse when < 1."""
                if r < 1:
                    return f"{r:.1f}:1 (1:{1/r:.1f} conc/abs)"
                return f"{r:.1f}:1"

            genre_prose = [
                f"{genre} (n = {len(gdf):,}):",
                f"  Rise ({abs_start}s → {abs_peak}s):",
                f"    Abstract: {a_s:.1f}% → {a_p:.1f}% ({_r(a_p, a_s):.1f}x)",
                f"    Concrete: {c_s:.1f}% → {c_p:.1f}% ({_r(c_s, c_p):.1f}x decline)" if c_p < c_s else
                f"    Concrete: {c_s:.1f}% → {c_p:.1f}% ({_r(c_p, c_s):.1f}x increase)",
                f"    Abs/Conc ratio: {_fmt_ratio(r_s)} → {_fmt_ratio(r_p)} ({_r(r_p, r_s):.1f}x)",
                f"  Fall ({abs_peak}s → {abs_end}s):",
                f"    Abstract: {a_p:.1f}% → {a_e:.1f}% ({_r(a_p, a_e):.1f}x decline)",
                f"    Concrete: {c_p:.1f}% → {c_e:.1f}% ({_r(c_e, c_p):.1f}x increase)" if c_e > c_p else
                f"    Concrete: {c_p:.1f}% → {c_e:.1f}% ({_r(c_p, c_e):.1f}x decline)",
                f"    Abs/Conc ratio: {_fmt_ratio(r_p)} → {_fmt_ratio(r_e)} ({_r(r_p, r_e):.1f}x decline)",
                f"  Net ({abs_start}s → {abs_end}s):",
                f"    Abstract: {a_s:.1f}% → {a_e:.1f}% ({_r(a_s, a_e):.1f}x decline)" if a_e < a_s else
                f"    Abstract: {a_s:.1f}% → {a_e:.1f}% ({_r(a_e, a_s):.1f}x increase)",
                f"    Concrete: {c_s:.1f}% → {c_e:.1f}% ({_r(c_e, c_s):.1f}x increase)" if c_e > c_s else
                f"    Concrete: {c_s:.1f}% → {c_e:.1f}% ({_r(c_s, c_e):.1f}x decline)",
                f"    Abs/Conc ratio: {_fmt_ratio(r_s)} → {_fmt_ratio(r_e)} ({_r(r_s, r_e):.1f}x decline)" if r_e < r_s else
                f"    Abs/Conc ratio: {_fmt_ratio(r_s)} → {_fmt_ratio(r_e)} ({_r(r_e, r_s):.1f}x increase)",
                f"  Breakpoint {int(abs_row['abstract_breakpoint'])}; "
                f"R² abstract = {abs_row['abstract_r2']:.3f}"
                + (f", R² concrete = {conc_row['concrete_r2']:.3f}" if conc_row else ""),
            ]
            prose_lines.append("\n".join(genre_prose))

        rows.append(row)

    df = pd.DataFrame(rows)

    if print_result and prose_lines:
        print()
        for block in prose_lines:
            print(block)
            print()
        print(f"(Abstract: z ≤ {abs_cutoff}; Concrete: z > {conc_cutoff}; "
              f"proportions are frequency-weighted.)")

    return df


# ---------------------------------------------------------------------------
# Combined report: scores + counts in one view
# ---------------------------------------------------------------------------

def report_full(scores_df=None, counts_df=None, genres=None,
                score_col="Abs-Conc.Median.median",
                norm="Abs-Conc.Median.median",
                abs_cutoff=-1.0, conc_cutoff=1.0,
                corpus_col="corpus_name",
                min_year=1600, max_year=2020,
                agg_bin=10, min_texts_per_bin=3,
                search_range=(1650, 1850), search_step=10,
                scores_version="v8-raw", counts_version="v2-raw"):
    """Generate a combined report from score-based and count-based analyses.

    Returns (markdown_string, summary_df) where markdown_string contains
    both a table and prose suitable for inclusion in a README.

    Parameters
    ----------
    scores_df : DataFrame, optional
        Pre-loaded from load_all_scored(). Loaded automatically if None.
    counts_df : DataFrame, optional
        Pre-loaded from load_all_counts(). Loaded automatically if None.
    genres : list, optional
        Genres to report. Default: Fiction, Poetry, Periodical.
    scores_version : str
        Version directory for scores (default "v8-raw").
    counts_version : str
        Version directory for counts (default "v2-raw").
    """
    if genres is None:
        genres = ["Fiction", "Poetry", "Periodical"]

    # Load data if needed
    if scores_df is None:
        scores_df = load_all_scored(version=scores_version)
    if counts_df is None:
        counts_df = load_all_counts(version=counts_version, norm=norm,
                                    abs_cutoff=abs_cutoff,
                                    conc_cutoff=conc_cutoff)

    fit_kw = dict(
        corpus_col=corpus_col, min_year=min_year, max_year=max_year,
        agg_bin=agg_bin, min_texts_per_bin=min_texts_per_bin,
        search_range=search_range, search_step=search_step,
    )

    # Run both reports silently
    score_result = report_arc(
        combined_df=scores_df, genres=genres, score_col=score_col,
        print_result=False, **fit_kw,
    )
    count_result = report_arc_counts(
        combined_df=counts_df, genres=genres, norm=norm,
        abs_cutoff=abs_cutoff, conc_cutoff=conc_cutoff,
        print_result=False, **fit_kw,
    )

    def _p_stars(p):
        if not np.isfinite(p):
            return ""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return "n.s."

    def _fmt_ratio(r):
        if r < 1:
            return f"{r:.1f}:1 (1:{1/r:.1f} conc/abs)"
        return f"{r:.1f}:1"

    def _safe_ratio(a, b):
        return a / b if b > 0 else np.nan

    lines = []

    # --- Summary table ---
    lines.append("### Summary (piecewise regression with corpus fixed effects)")
    lines.append("")
    lines.append("| Genre | Texts | Breakpoint | Rise slope | Fall slope | R² (scores) | R² (abstract %) | R² (concrete %) |")
    lines.append("|---|---:|---:|---|---|---:|---:|---:|")

    for _, sr in score_result.iterrows():
        genre = sr["genre"]
        cr = count_result[count_result["genre"] == genre]
        cr = cr.iloc[0] if len(cr) else None

        n = f"{int(sr['n_texts_total']):,}"
        bp = int(sr["breakpoint"])
        rise_p = _p_stars(sr["slope_before_p"])
        fall_p = _p_stars(sr["slope_after_p"])
        rise_slope = f"{sr['slope_before']:+.4f}/dec {rise_p}"
        fall_slope = f"{sr['slope_after']:+.4f}/dec {fall_p}"
        r2_scores = f"{sr['r2']:.3f}"
        r2_abs = f"{cr['abstract_r2']:.3f}" if cr is not None else "—"
        r2_conc = f"{cr['concrete_r2']:.3f}" if cr is not None else "—"

        lines.append(f"| {genre} | {n} | {bp} | {rise_slope} | {fall_slope} | {r2_scores} | {r2_abs} | {r2_conc} |")

    lines.append("")

    # --- Per-genre detail ---
    for _, sr in score_result.iterrows():
        genre = sr["genre"]
        cr = count_result[count_result["genre"] == genre]
        cr = cr.iloc[0] if len(cr) else None
        if cr is None:
            continue

        start_yr = int(sr["start_decade"])
        peak_yr = int(sr["peak_decade"])
        end_yr = int(sr["end_decade"])

        # Count-based values at score-based key decades
        a_s = cr["abstract_pct_start"]
        a_p = cr["abstract_pct_peak"]
        a_e = cr["abstract_pct_end"]
        c_s = cr["conc_at_abs_start"]
        c_p = cr["conc_at_abs_peak"]
        c_e = cr["conc_at_abs_end"]
        r_s = cr["abs_conc_ratio_start"]
        r_p = cr["abs_conc_ratio_peak"]
        r_e = cr["abs_conc_ratio_end"]

        lines.append(f"#### {genre} (n = {int(sr['n_texts_total']):,})")
        lines.append("")

        # Scores summary
        lines.append("**Scores** (continuous weighted-mean concreteness, inverted so abstractness is up):")
        lines.append(f"- {start_yr}s: {sr['raw_start']:.4f} → {peak_yr}s: {sr['raw_peak']:.4f} → {end_yr}s: {sr['raw_end']:.4f}")
        lines.append(f"- Rise: +{sr['rise_sd']:.2f} SD | Fall: +{sr['fall_sd']:.2f} SD")
        lines.append(f"- Breakpoint: {int(sr['breakpoint'])} | R² = {sr['r2']:.3f}")
        rise_p_str = _p_stars(sr["slope_before_p"])
        fall_p_str = _p_stars(sr["slope_after_p"])
        lines.append(f"- Rise slope: {sr['slope_before']:+.4f}/decade (p = {sr['slope_before_p']:.1e}) {rise_p_str}")
        lines.append(f"- Fall slope: {sr['slope_after']:+.4f}/decade (p = {sr['slope_after_p']:.1e}) {fall_p_str}")
        lines.append("")

        # Counts table
        lines.append(f"**Word proportions** (abstract: z ≤ {abs_cutoff}, concrete: z > {conc_cutoff}):")
        lines.append("")
        lines.append("| Phase | Abstract | Concrete | Abs/Conc ratio |")
        lines.append("|---|---|---|---|")

        lines.append(
            f"| {start_yr}s (start) | {a_s:.1f}% | {c_s:.1f}% | {_fmt_ratio(r_s)} |"
        )
        lines.append(
            f"| {peak_yr}s (peak) | {a_p:.1f}% | {c_p:.1f}% | {_fmt_ratio(r_p)} |"
        )
        lines.append(
            f"| {end_yr}s (end) | {a_e:.1f}% | {c_e:.1f}% | {_fmt_ratio(r_e)} |"
        )
        lines.append(
            f"| **Rise** ({start_yr}s→{peak_yr}s) "
            f"| {_safe_ratio(a_p, a_s):.1f}x "
            f"| {_safe_ratio(c_s, c_p):.1f}x decline "
            f"| {_safe_ratio(r_p, r_s):.1f}x |"
        )
        lines.append(
            f"| **Fall** ({peak_yr}s→{end_yr}s) "
            f"| {_safe_ratio(a_p, a_e):.1f}x decline "
            f"| {_safe_ratio(c_e, c_p):.1f}x increase "
            f"| {_safe_ratio(r_p, r_e):.1f}x decline |"
        )
        lines.append(
            f"| **Net** ({start_yr}s→{end_yr}s) "
            f"| {_safe_ratio(max(a_s, a_e), min(a_s, a_e)):.1f}x {'decline' if a_e < a_s else 'increase'} "
            f"| {_safe_ratio(max(c_s, c_e), min(c_s, c_e)):.1f}x {'increase' if c_e > c_s else 'decline'} "
            f"| {_safe_ratio(max(r_s, r_e), min(r_s, r_e)):.1f}x {'decline' if r_e < r_s else 'increase'} |"
        )

        lines.append("")
        lines.append(
            f"R² abstract = {cr['abstract_r2']:.3f}, "
            f"R² concrete = {cr['concrete_r2']:.3f}"
        )
        lines.append("")

    # --- Prose ---
    lines.append("### Prose summary")
    lines.append("")

    for _, sr in score_result.iterrows():
        genre = sr["genre"]
        cr = count_result[count_result["genre"] == genre]
        cr = cr.iloc[0] if len(cr) else None
        if cr is None:
            continue

        start_yr = int(sr["start_decade"])
        peak_yr = int(sr["peak_decade"])
        end_yr = int(sr["end_decade"])
        a_s = cr["abstract_pct_start"]
        a_p = cr["abstract_pct_peak"]
        a_e = cr["abstract_pct_end"]
        c_s = cr["conc_at_abs_start"]
        c_p = cr["conc_at_abs_peak"]
        c_e = cr["conc_at_abs_end"]
        r_s = cr["abs_conc_ratio_start"]
        r_p = cr["abs_conc_ratio_peak"]
        r_e = cr["abs_conc_ratio_end"]

        ratio_rise = _safe_ratio(r_p, r_s)
        ratio_fall = _safe_ratio(r_p, r_e)

        prose = (
            f"**{genre}** (n = {int(sr['n_texts_total']):,}): "
            f"Abstractness rises from the {start_yr}s to a peak in the {peak_yr}s "
            f"(+{sr['rise_sd']:.2f} SD), "
            f"then falls through the {end_yr}s (+{sr['fall_sd']:.2f} SD). "
            f"At peak, {genre.lower()} has {_fmt_ratio(r_p)} abstract-to-concrete words, "
        )
        if r_s >= 1:
            prose += f"up from {_fmt_ratio(r_s)} in the {start_yr}s ({ratio_rise:.1f}x). "
        else:
            prose += f"up from {_fmt_ratio(r_s)} in the {start_yr}s. "
        if r_e < 1:
            prose += (
                f"By the {end_yr}s, the ratio inverts to {_fmt_ratio(r_e)} — "
                f"a {ratio_fall:.1f}x decline from peak. "
            )
        else:
            prose += f"By the {end_yr}s it falls to {_fmt_ratio(r_e)}. "
        prose += (
            f"Piecewise breakpoint at {int(sr['breakpoint'])}; "
            f"rise slope = {sr['slope_before']:+.4f}/decade "
            f"(p = {sr['slope_before_p']:.1e}), "
            f"fall slope = {sr['slope_after']:+.4f}/decade "
            f"(p = {sr['slope_after_p']:.1e}); "
            f"R² = {sr['r2']:.3f}."
        )
        lines.append(prose)
        lines.append("")

    lines.append(
        f"*(Scores: continuous weighted-mean concreteness, inverted. "
        f"Proportions: frequency-weighted, abstract z ≤ {abs_cutoff}, "
        f"concrete z > {conc_cutoff}. "
        f"All regressions include corpus fixed effects.)*"
    )

    md = "\n".join(lines)

    # Merge DataFrames
    merged = score_result.merge(count_result, on="genre", how="outer",
                                suffixes=("_score", "_count"))

    return md, merged


def report_compare(genres=None,
                   score_col="Abs-Conc.Median.median",
                   norm="Abs-Conc.Median.median",
                   abs_cutoff=-1.0, conc_cutoff=1.0,
                   corpus_col="corpus_name",
                   min_year=1600, max_year=2020,
                   agg_bin=10, min_texts_per_bin=3,
                   search_range=(1650, 1850), search_step=10):
    """Compare raw vs modernized results across scores and counts.

    Loads all four datasets (v8-raw, v8, v2-raw, v2) and produces a
    side-by-side comparison table plus per-genre detail.

    Returns (markdown_string, dict_of_dataframes).
    """
    if genres is None:
        genres = ["Fiction", "Poetry", "Periodical"]

    fit_kw = dict(
        corpus_col=corpus_col, min_year=min_year, max_year=max_year,
        agg_bin=agg_bin, min_texts_per_bin=min_texts_per_bin,
        search_range=search_range, search_step=search_step,
    )

    # Load all four datasets
    variants = {
        "scores_raw": ("v8-raw", "scores"),
        "scores_mod": ("v8", "scores"),
        "counts_raw": ("v2-raw", "counts"),
        "counts_mod": ("v2", "counts"),
    }
    data = {}
    for key, (version, kind) in variants.items():
        try:
            if kind == "scores":
                data[key] = load_all_scored(version=version)
            else:
                data[key] = load_all_counts(version=version, norm=norm,
                                            abs_cutoff=abs_cutoff,
                                            conc_cutoff=conc_cutoff)
        except FileNotFoundError:
            print(f"  Warning: {version}/ not found, skipping {key}")
            data[key] = None

    # Run reports for each available variant
    results = {}
    if data.get("scores_raw") is not None:
        results["scores_raw"] = report_arc(
            combined_df=data["scores_raw"], genres=genres,
            score_col=score_col, print_result=False, **fit_kw)
    if data.get("scores_mod") is not None:
        results["scores_mod"] = report_arc(
            combined_df=data["scores_mod"], genres=genres,
            score_col=score_col, print_result=False, **fit_kw)
    if data.get("counts_raw") is not None:
        results["counts_raw"] = report_arc_counts(
            combined_df=data["counts_raw"], genres=genres, norm=norm,
            abs_cutoff=abs_cutoff, conc_cutoff=conc_cutoff,
            print_result=False, **fit_kw)
    if data.get("counts_mod") is not None:
        results["counts_mod"] = report_arc_counts(
            combined_df=data["counts_mod"], genres=genres, norm=norm,
            abs_cutoff=abs_cutoff, conc_cutoff=conc_cutoff,
            print_result=False, **fit_kw)

    def _p_stars(p):
        if not np.isfinite(p):
            return ""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return "n.s."

    def _fmt_ratio(r):
        if not np.isfinite(r):
            return "—"
        if r < 1:
            return f"{r:.1f}:1 (1:{1/r:.1f} conc/abs)"
        return f"{r:.1f}:1"

    def _safe_ratio(a, b):
        return a / b if b > 0 else np.nan

    def _get_row(df, genre):
        if df is None:
            return None
        rows = df[df["genre"] == genre]
        return rows.iloc[0] if len(rows) else None

    lines = []

    lines.append("## Raw vs modernized spelling: comparison")
    lines.append("")

    # --- Per-genre comparison ---
    for genre in genres:
        sr_raw = _get_row(results.get("scores_raw"), genre)
        sr_mod = _get_row(results.get("scores_mod"), genre)
        cr_raw = _get_row(results.get("counts_raw"), genre)
        cr_mod = _get_row(results.get("counts_mod"), genre)

        n_texts = int(sr_raw["n_texts_total"]) if sr_raw is not None else (
            int(sr_mod["n_texts_total"]) if sr_mod is not None else "?")

        lines.append(f"### {genre} (n = {n_texts:,})")
        lines.append("")

        # Scores comparison table
        lines.append("#### Scores (continuous weighted-mean)")
        lines.append("")
        lines.append("| | Raw | Modernized |")
        lines.append("|---|---|---|")

        def _score_row(label, key, sr):
            if sr is None:
                return "—"
            return f"{sr[key]}"

        if sr_raw is not None or sr_mod is not None:
            for label, key, fmt in [
                ("Breakpoint", "breakpoint", lambda v: f"{int(v)}"),
                ("Start decade", "start_decade", lambda v: f"{int(v)}s"),
                ("Peak decade", "peak_decade", lambda v: f"{int(v)}s"),
                ("End decade", "end_decade", lambda v: f"{int(v)}s"),
                ("Raw start", "raw_start", lambda v: f"{v:.4f}"),
                ("Raw peak", "raw_peak", lambda v: f"{v:.4f}"),
                ("Raw end", "raw_end", lambda v: f"{v:.4f}"),
                ("Rise (SD)", "rise_sd", lambda v: f"+{v:.2f}"),
                ("Fall (SD)", "fall_sd", lambda v: f"+{v:.2f}"),
                ("Rise slope", "slope_before", lambda v: f"{v:+.4f}/dec"),
                ("Rise p", "slope_before_p", lambda v: f"{v:.1e} {_p_stars(v)}"),
                ("Fall slope", "slope_after", lambda v: f"{v:+.4f}/dec"),
                ("Fall p", "slope_after_p", lambda v: f"{v:.1e} {_p_stars(v)}"),
                ("R²", "r2", lambda v: f"{v:.3f}"),
            ]:
                raw_val = fmt(sr_raw[key]) if sr_raw is not None else "—"
                mod_val = fmt(sr_mod[key]) if sr_mod is not None else "—"
                lines.append(f"| {label} | {raw_val} | {mod_val} |")

        lines.append("")

        # Counts comparison table
        lines.append(f"#### Word proportions (abstract: z ≤ {abs_cutoff}, concrete: z > {conc_cutoff})")
        lines.append("")
        lines.append("| | Raw | Modernized |")
        lines.append("|---|---|---|")

        if cr_raw is not None or cr_mod is not None:
            for label, key, fmt in [
                ("Breakpoint (abstract)", "abstract_breakpoint", lambda v: f"{int(v)}"),
                ("Abstract start", "abstract_pct_start", lambda v: f"{v:.1f}%"),
                ("Abstract peak", "abstract_pct_peak", lambda v: f"{v:.1f}%"),
                ("Abstract end", "abstract_pct_end", lambda v: f"{v:.1f}%"),
                ("Concrete at start", "conc_at_abs_start", lambda v: f"{v:.1f}%"),
                ("Concrete at peak", "conc_at_abs_peak", lambda v: f"{v:.1f}%"),
                ("Concrete at end", "conc_at_abs_end", lambda v: f"{v:.1f}%"),
                ("Abs/Conc ratio start", "abs_conc_ratio_start", lambda v: _fmt_ratio(v)),
                ("Abs/Conc ratio peak", "abs_conc_ratio_peak", lambda v: _fmt_ratio(v)),
                ("Abs/Conc ratio end", "abs_conc_ratio_end", lambda v: _fmt_ratio(v)),
                ("R² (abstract)", "abstract_r2", lambda v: f"{v:.3f}"),
                ("R² (concrete)", "concrete_r2", lambda v: f"{v:.3f}"),
            ]:
                raw_val = fmt(cr_raw[key]) if cr_raw is not None else "—"
                mod_val = fmt(cr_mod[key]) if cr_mod is not None else "—"
                lines.append(f"| {label} | {raw_val} | {mod_val} |")

            # Add ratio changes
            for cr_label, cr in [("Raw", cr_raw), ("Modernized", cr_mod)]:
                if cr is None:
                    continue
                r_s = cr["abs_conc_ratio_start"]
                r_p = cr["abs_conc_ratio_peak"]
                r_e = cr["abs_conc_ratio_end"]
                # We'll add these as summary rows below

        lines.append("")

        # Ratio change summary
        lines.append("**Abs/Conc ratio changes:**")
        lines.append("")
        lines.append("| Phase | Raw | Modernized |")
        lines.append("|---|---|---|")
        for phase, get_a, get_b in [
            ("Rise", "abs_conc_ratio_start", "abs_conc_ratio_peak"),
            ("Fall", "abs_conc_ratio_peak", "abs_conc_ratio_end"),
            ("Net", "abs_conc_ratio_start", "abs_conc_ratio_end"),
        ]:
            cells = []
            for cr in [cr_raw, cr_mod]:
                if cr is None:
                    cells.append("—")
                    continue
                a, b = cr[get_a], cr[get_b]
                if phase == "Rise":
                    cells.append(f"{_fmt_ratio(a)} → {_fmt_ratio(b)} ({_safe_ratio(b, a):.1f}x)")
                elif phase == "Fall":
                    cells.append(f"{_fmt_ratio(a)} → {_fmt_ratio(b)} ({_safe_ratio(a, b):.1f}x decline)")
                else:
                    change = _safe_ratio(max(a, b), min(a, b))
                    direction = "decline" if b < a else "increase"
                    cells.append(f"{_fmt_ratio(a)} → {_fmt_ratio(b)} ({change:.1f}x {direction})")
            lines.append(f"| {phase} | {cells[0]} | {cells[1]} |")

        lines.append("")

    lines.append(
        f"*(Scores: continuous weighted-mean concreteness, inverted. "
        f"Proportions: frequency-weighted, abstract z ≤ {abs_cutoff}, "
        f"concrete z > {conc_cutoff}. "
        f"All regressions include corpus fixed effects.)*"
    )

    md = "\n".join(lines)
    return md, results
