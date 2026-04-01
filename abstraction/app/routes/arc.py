"""Arc endpoints: aggregated decade bins and paginated raw texts."""

from fastapi import APIRouter, Query

from ..db import get_connection
from ..models import ArcAggregated, ArcBin, ArcText, ArcTexts, CorpusArc, CorpusArcBin

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
