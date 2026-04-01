"""Arc endpoints: aggregated decade bins and paginated raw texts."""

from fastapi import APIRouter, Query

from ..db import get_connection
from ..models import (
    ArcAggregated, ArcBin, ArcText, ArcTexts,
    CorpusArc, CorpusArcBin,
    GenreArc, AdjustedPoint, LoessPoint,
)

router = APIRouter()

DEFAULT_COL = "Abs-Conc.Median.median"


def _build_where(genre: list[str], corpus: list[str],
                 year_min: float | None, year_max: float | None,
                 col: str):
    """Build WHERE clause and params for filtering."""
    clauses = [f'"{col}" IS NOT NULL', "year IS NOT NULL"]
    params: list = []

    if genre:
        placeholders = ",".join("?" for _ in genre)
        clauses.append(f"genre_harmonized IN ({placeholders})")
        params.extend(genre)
    if corpus:
        placeholders = ",".join("?" for _ in corpus)
        clauses.append(f"corpus_name IN ({placeholders})")
        params.extend(corpus)
    if year_min is not None:
        clauses.append("year >= ?")
        params.append(year_min)
    if year_max is not None:
        clauses.append("year <= ?")
        params.append(year_max)

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

    # SQLite doesn't have percentile functions, so we fetch raw values
    # and compute in Python. For ~2M rows this is still fast (<1s).
    rows = conn.execute(f"""
        SELECT CAST(year / ? AS INT) * ? AS decade,
               "{col}", year
        FROM texts
        WHERE {where}
        ORDER BY decade
    """, [bin_size, bin_size] + params).fetchall()
    conn.close()

    # Group by decade and compute stats
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
        SELECT corpus_name, genre_harmonized,
               CAST(year / ? AS INT) * ? AS decade,
               "{col}"
        FROM texts
        WHERE {where}
        ORDER BY corpus_name, decade
    """, [bin_size, bin_size] + params).fetchall()
    conn.close()

    from collections import defaultdict
    import numpy as np

    # Group by corpus → decade → scores, track genre counts
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
        # Use the most common genre for this corpus
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
):
    """Return paginated scored texts for scatter plot overlay."""
    where, params = _build_where(genre, corpus, year_min, year_max, col)
    conn = get_connection()

    # Total count
    total = conn.execute(
        f"SELECT COUNT(*) FROM texts WHERE {where}", params
    ).fetchone()[0]

    # Paginated rows
    rows = conn.execute(f"""
        SELECT id, corpus_name, year, author, title, genre_harmonized, "{col}"
        FROM texts
        WHERE {where}
        ORDER BY year
        LIMIT ? OFFSET ?
    """, params + [page_size, page * page_size]).fetchall()
    conn.close()

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
    genre: list[str] = Query(default=["Fiction", "Poetry", "Periodical"]),
    year_min: float = 1600,
    year_max: float = 2020,
    loess_span: float = 0.3,
    invert: bool = True,
    period_matched: bool = False,
):
    """Return corpus-adjusted decade bins + LOESS per genre.

    Calls analysis.adjust_scores() per genre to remove corpus fixed effects,
    then fits LOESS on the adjusted points. Matches the book figure approach.

    When period_matched=True, each text is scored with its century-matched
    vecnorms (e.g. C17 texts use Abs-Conc.Median.C17) and norm_period is
    added as a second fixed effect to absorb hinge points at century boundaries.
    """
    import pandas as pd
    from ...analysis import adjust_scores, assign_period_score

    # Parse source from col (e.g. "Abs-Conc.Median.median" → "Median")
    col_parts = col.split(".")
    source = col_parts[1] if len(col_parts) >= 2 else "Median"

    conn = get_connection()

    if period_matched:
        # Load all per-century columns for this source + the median fallback
        all_cols = [r[1] for r in conn.execute("PRAGMA table_info(texts)").fetchall()]
        source_cols = [c for c in all_cols if c.startswith(f"Abs-Conc.{source}.")]
        col_sql = ", ".join(f'"{c}"' for c in source_cols)
        rows = conn.execute(f"""
            SELECT id, corpus_name, year, genre_harmonized, {col_sql}
            FROM texts
            WHERE year IS NOT NULL AND year >= ? AND year <= ?
        """, [year_min, year_max]).fetchall()
        columns = ["id", "corpus_name", "year", "genre_harmonized"] + source_cols
    else:
        rows = conn.execute(f"""
            SELECT id, corpus_name, year, genre_harmonized, "{col}"
            FROM texts
            WHERE year IS NOT NULL AND "{col}" IS NOT NULL
                  AND year >= ? AND year <= ?
        """, [year_min, year_max]).fetchall()
        columns = ["id", "corpus_name", "year", "genre_harmonized", col]

    conn.close()

    df = pd.DataFrame(rows, columns=columns)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")

    if period_matched:
        df = assign_period_score(df, source=source)
        score_col = "period_score"
        fe = ["norm_period"]
    else:
        score_col = col
        fe = None

    results = []
    for g in genre:
        gdf = df[df["genre_harmonized"] == g]
        if period_matched:
            gdf = gdf.dropna(subset=[score_col])
        if len(gdf) < 30:
            continue

        adj = adjust_scores(
            gdf, score_col=score_col, min_year=year_min, max_year=year_max,
            fixed_effects=fe,
        )
        if adj.empty:
            continue

        sign = -1.0 if invert else 1.0

        # Build points
        points = []
        for _, row in adj.iterrows():
            points.append(AdjustedPoint(
                year=float(row["year"]),
                score=float(row["score"]) * sign,
                adjusted=float(row["adjusted"]) * sign,
                n_texts=int(row["n_texts"]),
                corpus=row.get("corpus"),
            ))

        # LOESS on adjusted scores
        loess_points = _compute_loess(
            adj["year"].values,
            adj["adjusted"].values * sign,
            span=loess_span,
        )

        results.append(GenreArc(
            genre=g,
            points=points,
            loess=loess_points,
            n_texts_total=int(adj["n_texts"].sum()),
            n_corpora=adj["corpus"].nunique() if "corpus" in adj.columns else 1,
        ))

    return results


def _compute_loess(years, values, span=0.3, n_points=200):
    """Compute LOESS smooth with SE band."""
    from statsmodels.nonparametric.smoothers_lowess import lowess
    import numpy as np

    # Sort
    order = np.argsort(years)
    x = years[order].astype(float)
    y = values[order].astype(float)

    # Fit LOESS
    result = lowess(y, x, frac=span, return_sorted=True)
    lx, ly = result[:, 0], result[:, 1]

    # Estimate SE via residuals (simple approach: local residual SD)
    residuals = y - np.interp(x, lx, ly)
    # Rolling window SE approximation
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
