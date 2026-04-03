"""Arc endpoints: aggregated decade bins and paginated raw texts."""

from fastapi import APIRouter, Query

from ..db import get_connection
from ..models import (
    ArcAggregated, ArcBin, ArcText, ArcTexts,
    CorpusArc, CorpusArcBin,
    GenreArc, AdjustedPoint, LoessPoint, ArcStats,
)

router = APIRouter()

DEFAULT_COL = "Abs-Conc.Median.median"


def _build_where(genre: list[str], corpus: list[str],
                 year_min: float | None, year_max: float | None,
                 col: str):
    """Build WHERE clause and params for filtering (DuckDB positional params)."""
    clauses = [f'"{col}" IS NOT NULL', "year IS NOT NULL"]
    params: list = []

    if genre:
        # Support both arc_corpus IDs and regular genre names
        arc_genres = [g for g in genre if g.startswith("arc_")]
        reg_genres = [g for g in genre if not g.startswith("arc_")]
        genre_clauses = []
        if arc_genres:
            placeholders = ",".join("?" for _ in arc_genres)
            genre_clauses.append(f"arc_corpus IN ({placeholders})")
            params.extend(arc_genres)
        if reg_genres:
            placeholders = ",".join("?" for _ in reg_genres)
            genre_clauses.append(f"genre IN ({placeholders})")
            params.extend(reg_genres)
        if genre_clauses:
            clauses.append(f"({' OR '.join(genre_clauses)})")
    if corpus:
        placeholders = ",".join("?" for _ in corpus)
        clauses.append(f"corpus_name IN ({placeholders})")
        params.extend(corpus)
    if year_min is not None:
        clauses.append(f"year >= {year_min}")
    if year_max is not None:
        clauses.append(f"year <= {year_max}")

    return " AND ".join(clauses), params


@router.get("/aggregated", response_model=ArcAggregated)
def arc_aggregated(
    col: str = DEFAULT_COL,
    genre: list[str] = Query(default=[]),
    corpus: list[str] = Query(default=[]),
    year_min: float | None = None,
    year_max: float | None = None,
    bin_size: int = 10,
):
    """Return decade-binned summary statistics for the arc plot."""
    where, params = _build_where(genre, corpus, year_min, year_max, col)
    conn = get_connection()

    rows = conn.execute(f"""
        SELECT CAST(year / {bin_size} AS INT) * {bin_size} AS decade,
               "{col}", year
        FROM texts
        WHERE {where}
        ORDER BY decade
    """, params).fetchall()

    from collections import defaultdict
    buckets: dict[int, list[float]] = defaultdict(list)
    for decade, score, _year in rows:
        if score is not None and decade is not None:
            buckets[int(decade)].append(score)

    import numpy as np
    bins = []
    for decade in sorted(buckets):
        vals = buckets[decade]
        arr = np.array(vals)
        bins.append(ArcBin(
            decade=decade,
            mean=float(np.mean(arr)),
            median=float(np.median(arr)),
            q25=float(np.percentile(arr, 25)),
            q75=float(np.percentile(arr, 75)),
            n=len(vals),
        ))

    total = sum(b.n for b in bins)
    return ArcAggregated(bins=bins, total=total)


@router.get("/by-corpus", response_model=list[CorpusArc])
def arc_by_corpus(
    col: str = DEFAULT_COL,
    genre: list[str] = Query(default=[]),
    corpus: list[str] = Query(default=[]),
    year_min: float | None = None,
    year_max: float | None = None,
    bin_size: int = 10,
):
    """Return per-corpus decade-binned means for the macro arc plot."""
    where, params = _build_where(genre, corpus, year_min, year_max, col)
    conn = get_connection()

    rows = conn.execute(f"""
        SELECT corpus_name, genre,
               CAST(year / {bin_size} AS INT) * {bin_size} AS decade,
               "{col}"
        FROM texts
        WHERE {where}
        ORDER BY corpus_name, decade
    """, params).fetchall()

    from collections import defaultdict
    import numpy as np

    corpus_data: dict[str, dict] = {}
    for corpus_name, genre_val, decade, score in rows:
        if score is None or decade is None:
            continue
        if corpus_name not in corpus_data:
            corpus_data[corpus_name] = {
                "genre_counts": defaultdict(int),
                "decades": defaultdict(list),
            }
        corpus_data[corpus_name]["genre_counts"][genre_val or ""] += 1
        corpus_data[corpus_name]["decades"][int(decade)].append(score)

    results = []
    for cname in sorted(corpus_data):
        cd = corpus_data[cname]
        decades = cd["decades"]
        bins = []
        total = 0
        for decade in sorted(decades):
            vals = decades[decade]
            bins.append(CorpusArcBin(
                decade=decade,
                mean=float(np.mean(vals)),
                n=len(vals),
            ))
            total += len(vals)
        top_genre = max(cd["genre_counts"], key=cd["genre_counts"].get) if cd["genre_counts"] else None
        results.append(CorpusArc(
            corpus=cname,
            genre=top_genre or None,
            n_texts=total,
            bins=bins,
        ))

    return results


@router.get("/texts", response_model=ArcTexts)
def arc_texts(
    col: str = DEFAULT_COL,
    genre: list[str] = Query(default=[]),
    corpus: list[str] = Query(default=[]),
    year_min: float | None = None,
    year_max: float | None = None,
    page: int = 0,
    page_size: int = 5000,
    period_matched: bool = False,
):
    """Return paginated scored texts for scatter plot overlay."""
    import pandas as pd
    from ...analysis import assign_period_score, CENTURY_BINS

    col_parts = col.split(".")
    source = col_parts[1] if len(col_parts) >= 2 else "Median"

    conn = get_connection()

    if period_matched:
        # Load all per-century columns for this source
        score_cols = [r[0] for r in conn.execute("DESCRIBE scores").fetchall()
                      if r[0].startswith(f"Abs-Conc.{source}.")]
        col_sql = ", ".join(f'"{c}"' for c in score_cols)

        where, params = _build_where(genre, corpus, year_min, year_max, score_cols[0] if score_cols else col)
        # Override the NOT NULL check to require any period column
        where = where.replace(f'"{score_cols[0]}" IS NOT NULL', "1=1") if score_cols else where

        total = conn.execute(
            f"SELECT COUNT(*) FROM texts WHERE {where}", params
        ).fetchone()[0]

        df = conn.execute(f"""
            SELECT _id, corpus_name, year, author, title, genre, {col_sql}
            FROM texts
            WHERE {where}
            ORDER BY year
            LIMIT {page_size} OFFSET {page * page_size}
        """, params).fetchdf()

        if len(df) > 0:
            df = assign_period_score(df, source=source)
            texts = [
                ArcText(
                    id=row["_id"], corpus=row["corpus_name"], year=row.get("year"),
                    author=row.get("author"), title=row.get("title"), genre=row.get("genre"),
                    score=row.get("period_score"),
                )
                for _, row in df.iterrows()
            ]
        else:
            texts = []
    else:
        where, params = _build_where(genre, corpus, year_min, year_max, col)

        total = conn.execute(
            f"SELECT COUNT(*) FROM texts WHERE {where}", params
        ).fetchone()[0]

        rows = conn.execute(f"""
            SELECT _id, corpus_name, year, author, title, genre, "{col}"
            FROM texts
            WHERE {where}
            ORDER BY year
            LIMIT {page_size} OFFSET {page * page_size}
        """, params).fetchall()

        texts = [
            ArcText(
                id=r[0], corpus=r[1], year=r[2],
                author=r[3], title=r[4], genre=r[5], score=r[6],
            )
            for r in rows
        ]

    return ArcTexts(texts=texts, total=total, page=page, page_size=page_size)


@router.get("/by-genre", response_model=list[GenreArc])
def arc_by_genre(
    col: str = DEFAULT_COL,
    genre: list[str] = Query(default=["arc_fiction"]),
    corpus: list[str] = Query(default=[]),
    year_min: float = 1580,
    year_max: float = 2020,
    loess_span: float = 0.3,
    invert: bool = True,
    period_matched: bool = False,
    corpus_adjusted: bool = False,
    model: str = "quadratic",
    bin_size: int = 10,
):
    """Return corpus-adjusted decade bins + LOESS per arc corpus.

    The 'genre' parameter accepts arc corpus IDs (arc_fiction, arc_poetry, etc.)
    or regular genre names for backward compatibility.
    """
    import pandas as pd
    from ...analysis import adjust_scores, assign_period_score

    col_parts = col.split(".")
    source = col_parts[1] if len(col_parts) >= 2 else "Median"

    conn = get_connection()

    # Build source corpus filter
    corpus_sql = ""
    if corpus:
        corpus_list = ", ".join(f"'{c}'" for c in corpus)
        corpus_sql = f" AND corpus_name IN ({corpus_list})"

    # Load data for all requested genres/arc_corpora
    all_dfs = []
    for g in genre:
        # Check if g is an arc_corpus name or a genre name
        is_arc = g.startswith("arc_")

        if period_matched:
            score_cols = [r[0] for r in conn.execute("DESCRIBE scores").fetchall()
                          if r[0].startswith(f"Abs-Conc.{source}.")]
            col_sql = ", ".join(f'"{c}"' for c in score_cols)
            if is_arc:
                gdf = conn.execute(f"""
                    SELECT _id, corpus_name, year, arc_corpus, {col_sql}
                    FROM texts
                    WHERE arc_corpus = '{g}'
                      AND year IS NOT NULL
                      AND year >= {year_min} AND year <= {year_max}{corpus_sql}
                """).fetchdf()
            else:
                gdf = conn.execute(f"""
                    SELECT _id, corpus_name, year, genre, {col_sql}
                    FROM texts
                    WHERE genre = '{g}'
                      AND year IS NOT NULL
                      AND year >= {year_min} AND year <= {year_max}{corpus_sql}
                """).fetchdf()
        else:
            if is_arc:
                gdf = conn.execute(f"""
                    SELECT _id, corpus_name, year, arc_corpus, "{col}"
                    FROM texts
                    WHERE arc_corpus = '{g}'
                      AND year IS NOT NULL AND "{col}" IS NOT NULL
                      AND year >= {year_min} AND year <= {year_max}{corpus_sql}
                """).fetchdf()
            else:
                gdf = conn.execute(f"""
                    SELECT _id, corpus_name, year, genre, "{col}"
                    FROM texts
                    WHERE genre = '{g}'
                      AND year IS NOT NULL AND "{col}" IS NOT NULL
                      AND year >= {year_min} AND year <= {year_max}{corpus_sql}
                """).fetchdf()

        if len(gdf) > 0:
            gdf["_genre_label"] = g
            all_dfs.append(gdf)

    if not all_dfs:
        return []

    df = pd.concat(all_dfs, ignore_index=True)

    if period_matched:
        df = assign_period_score(df, source=source)
        score_col = "period_score"
    else:
        score_col = col

    fe = []
    if period_matched:
        fe.append("norm_period")
    if corpus_adjusted:
        fe.append("corpus_name")

    results = []
    for g in genre:
        gdf = df[df["_genre_label"] == g]
        if period_matched:
            gdf = gdf.dropna(subset=[score_col])
        if len(gdf) < 30:
            continue

        adj = adjust_scores(
            gdf, score_col=score_col, min_year=year_min, max_year=year_max,
            corpus_col="corpus_name",
            fixed_effects=fe if fe else None,
            model=model, agg_bin=bin_size,
        )
        if adj.empty:
            continue

        sign = -1.0 if invert else 1.0

        # Choose which values to use for the main LOESS
        main_col = "adjusted" if corpus_adjusted else "score"

        points = []
        for _, row in adj.iterrows():
            points.append(AdjustedPoint(
                year=float(row["year"]),
                score=float(row["score"]) * sign,
                adjusted=float(row["adjusted"]) * sign,
                n_texts=int(row["n_texts"]),
                corpus=row.get("corpus"),
            ))

        # Main LOESS (raw by default, adjusted if toggled)
        main_vals = adj[main_col].values * sign
        loess_points = _compute_loess(
            adj["year"].values, main_vals, span=loess_span,
        )

        # Secondary LOESS (the other one, for comparison)
        other_col = "score" if corpus_adjusted else "adjusted"
        other_vals = adj[other_col].values * sign
        loess_raw_points = _compute_loess(
            adj["year"].values, other_vals, span=loess_span,
        )

        stats = _compute_arc_stats(adj, sign, loess_points)

        # Aggregate LOESS: weighted mean per year bin across all corpora
        import numpy as np
        agg_years = adj["year"].values
        agg_scores = (adj[main_col].values * sign * adj["n_texts"].values)
        agg_weights = adj["n_texts"].values
        # Group by year
        unique_years = np.unique(agg_years)
        agg_means = np.array([
            agg_scores[agg_years == y].sum() / agg_weights[agg_years == y].sum()
            for y in unique_years
        ])
        loess_agg_points = _compute_loess(unique_years, agg_means, span=loess_span)

        n_corpora = adj["corpus"].nunique() if "corpus" in adj.columns else 1
        results.append(GenreArc(
            genre=g,
            points=points,
            loess=loess_points,
            loess_raw=loess_raw_points,
            loess_aggregate=loess_agg_points,
            stats=stats,
            n_texts_total=int(adj["n_texts"].sum()),
            n_corpora=n_corpora,
        ))

    return results


def _compute_arc_stats(adj, sign, loess_points):
    """Compute piecewise regression stats + peak/start/end from LOESS."""
    import numpy as np
    from ...analysis import fit_piecewise

    years = adj["year"].values
    scores = adj["adjusted"].values * sign
    groups = adj["corpus"].values if "corpus" in adj.columns else None
    n_texts = int(adj["n_texts"].sum())
    n_corpora = int(adj["corpus"].nunique()) if "corpus" in adj.columns else 1

    pw = fit_piecewise(years, scores, groups=groups)

    breakpoint = pw.get("pw_break_year")
    rise_slope = pw.get("pw_slope_before")
    fall_slope = pw.get("pw_slope_after")
    if rise_slope is not None and np.isfinite(rise_slope):
        rise_slope *= 10
    if fall_slope is not None and np.isfinite(fall_slope):
        fall_slope *= 10

    peak_year = peak_score = start_score = end_score = None
    if loess_points:
        peak_pt = max(loess_points, key=lambda p: p.fitted)
        peak_year = int(round(peak_pt.year))
        peak_score = peak_pt.fitted
        start_score = loess_points[0].fitted
        end_score = loess_points[-1].fitted

    change_sd = None
    if peak_score is not None and end_score is not None:
        sd = np.std(scores)
        if sd > 0:
            change_sd = round((peak_score - end_score) / sd, 2)

    def _clean(v):
        if v is None:
            return None
        try:
            if np.isnan(v):
                return None
        except (TypeError, ValueError):
            pass
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return round(float(v), 6)
        return v

    return ArcStats(
        n_texts=n_texts,
        n_corpora=n_corpora,
        breakpoint=_clean(breakpoint),
        rise_slope=_clean(rise_slope),
        fall_slope=_clean(fall_slope),
        rise_slope_p=_clean(pw.get("pw_slope_before_p")),
        fall_slope_p=_clean(pw.get("pw_slope_after_p")),
        r2=_clean(pw.get("pw_r2")),
        peak_year=_clean(peak_year),
        peak_score=_clean(peak_score),
        start_score=_clean(start_score),
        end_score=_clean(end_score),
        change_sd=_clean(change_sd),
    )


def _compute_loess(years, values, span=0.3, n_points=200):
    """Compute LOESS smooth with SE band."""
    from statsmodels.nonparametric.smoothers_lowess import lowess
    import numpy as np

    order = np.argsort(years)
    x = years[order].astype(float)
    y = values[order].astype(float)

    result = lowess(y, x, frac=span, return_sorted=True)
    lx, ly = result[:, 0], result[:, 1]

    residuals = y - np.interp(x, lx, ly)
    window = max(3, int(len(x) * span))
    se_vals = np.full_like(ly, np.std(residuals))
    for i in range(len(ly)):
        lo = max(0, i - window // 2)
        hi = min(len(residuals), i + window // 2 + 1)
        if hi - lo >= 3:
            se_vals[i] = np.std(residuals[lo:hi])

    points = []
    for i in range(len(lx)):
        points.append(LoessPoint(
            year=float(lx[i]),
            fitted=float(ly[i]),
            se_lo=float(ly[i] - se_vals[i]),
            se_hi=float(ly[i] + se_vals[i]),
        ))
    return points
