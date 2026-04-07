"""Arc endpoints: aggregated decade bins and paginated raw texts."""

from fastapi import APIRouter, Query

from ..db import get_connection
from ..models import (
    ArcAggregated, ArcBin, ArcText, ArcTexts,
    CorpusArc, CorpusArcBin,
    GenreArc, AdjustedPoint, LoessPoint, ArcStats,
    AggGenreArc, AggBinPoint,
)

router = APIRouter()

DEFAULT_COL = "Abs-Conc.Median.median"


def _build_where(genre: list[str], corpus: list[str],
                 year_min: float | None, year_max: float | None,
                 col: str, genre_raw: str | None = None):
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
    if genre_raw:
        clauses.append("genre_raw LIKE ?")
        params.append(f"%{genre_raw}%")
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
    genre_raw: str | None = None,
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

        where, params = _build_where(genre, corpus, year_min, year_max, score_cols[0] if score_cols else col, genre_raw=genre_raw)
        # Override the NOT NULL check to require any period column
        where = where.replace(f'"{score_cols[0]}" IS NOT NULL', "1=1") if score_cols else where

        total = conn.execute(
            f"SELECT COUNT(*) FROM texts WHERE {where}", params
        ).fetchone()[0]

        # Check if version columns exist in scores table
        score_table_cols = {r[0] for r in conn.execute("DESCRIBE scores").fetchall()}
        has_versions = "_n_versions" in score_table_cols
        version_sql = ', "_n_versions", "_score_sd"' if has_versions else ""

        df = conn.execute(f"""
            SELECT _id, corpus_name, year, author, title, genre, genre_raw, genre_enriched_source, is_translated{version_sql}, {col_sql}
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
                    genre_raw=row.get("genre_raw"),
                    genre_enriched_source=row.get("genre_enriched_source"),
                    is_translated=bool(row["is_translated"]) if pd.notna(row.get("is_translated")) else None,
                    score=row.get("period_score"),
                    n_versions=int(row["_n_versions"]) if has_versions and pd.notna(row.get("_n_versions")) else None,
                    score_sd=float(row["_score_sd"]) if has_versions and pd.notna(row.get("_score_sd")) else None,
                )
                for _, row in df.iterrows()
            ]
        else:
            texts = []
    else:
        where, params = _build_where(genre, corpus, year_min, year_max, col, genre_raw=genre_raw)

        score_table_cols = {r[0] for r in conn.execute("DESCRIBE scores").fetchall()}
        has_versions = "_n_versions" in score_table_cols
        version_sql = ', "_n_versions", "_score_sd"' if has_versions else ""

        total = conn.execute(
            f"SELECT COUNT(*) FROM texts WHERE {where}", params
        ).fetchone()[0]

        rows = conn.execute(f"""
            SELECT _id, corpus_name, year, author, title, genre, genre_raw, genre_enriched_source, is_translated, "{col}"{version_sql}
            FROM texts
            WHERE {where}
            ORDER BY year
            LIMIT {page_size} OFFSET {page * page_size}
        """, params).fetchall()

        texts = [
            ArcText(
                id=r[0], corpus=r[1], year=r[2],
                author=r[3], title=r[4], genre=r[5], genre_raw=r[6], genre_enriched_source=r[7],
                is_translated=r[8], score=r[9],
                n_versions=r[10] if has_versions and len(r) > 10 else None,
                score_sd=r[11] if has_versions and len(r) > 11 else None,
            )
            for r in rows
        ]

    return ArcTexts(texts=texts, total=total, page=page, page_size=page_size)


@router.get("/aggregate", response_model=list[AggGenreArc])
def arc_aggregate(
    col: str = DEFAULT_COL,
    genre: list[str] = Query(default=["arc_fiction"]),
    corpus: list[str] = Query(default=[]),
    year_min: float = 1565,
    year_max: float = 2020,
    loess_span: float = 0.2,
    invert: bool = True,
    period_matched: bool = False,
    bin_size: int = 5,
    split_by: str | None = None,
    is_translated: str | None = None,
    min_texts: int = 1,
    corpus_corrected: bool = False,
):
    """Aggregate arc: simple mean per year bin across ALL texts (no corpus weighting).

    Each text contributes equally regardless of which corpus it comes from.
    No corpus fixed effects. Just bin by year, mean, LOESS.
    """
    import pandas as pd
    import numpy as np
    from ...analysis import assign_period_score, CENTURY_BINS
    from ..models import AggGenreArc, AggBinPoint

    col_parts = col.split(".")
    source = col_parts[1] if len(col_parts) >= 2 else "Median"

    # Load corpus bias coefficients if requested
    corpus_bias = None
    if corpus_corrected:
        from ...corpus_correction import load_corpus_bias
        corpus_bias = load_corpus_bias()

    conn = get_connection()

    corpus_filter = ""
    if corpus and len(corpus) > 0:
        cl = ", ".join(f"'{c}'" for c in corpus)
        corpus_filter = f" AND corpus_name IN ({cl})"

    # is_translated filter (top-level column in LLTK DuckDB)
    translated_filter = ""
    if is_translated == "true":
        translated_filter = " AND is_translated = true"
    elif is_translated == "false":
        translated_filter = " AND (is_translated IS NULL OR is_translated = false)"

    # Extra columns to select for split_by and corpus correction
    extra_cols = ""
    valid_splits = {"genre_raw", "corpus_name", "is_translated"}
    extra_col_set = set()
    if split_by and split_by in valid_splits:
        extra_col_set.add(split_by)
    if corpus_bias:
        extra_col_set.add("corpus_name")
    if extra_col_set:
        extra_cols = ", " + ", ".join(extra_col_set)

    sign = -1.0 if invert else 1.0
    results = []

    for g in genre:
        is_arc = g.startswith("arc_")
        filter_col = "arc_corpus" if is_arc else "genre"

        if period_matched:
            score_cols = [r[0] for r in conn.execute("DESCRIBE scores").fetchall()
                          if r[0].startswith(f"Abs-Conc.{source}.")]
            col_sql = ", ".join(f'"{c}"' for c in score_cols)
            df = conn.execute(f"""
                SELECT year{extra_cols}, {col_sql}
                FROM texts
                WHERE {filter_col} = '{g}' AND year IS NOT NULL
                  AND year >= {year_min} AND year <= {year_max}{corpus_filter}{translated_filter}
            """).fetchdf()

            if len(df) < 30:
                continue

            df = assign_period_score(df, source=source)
            df["_score"] = df["period_score"]
            df = df.dropna(subset=["_score"])
        else:
            df = conn.execute(f"""
                SELECT year{extra_cols}, "{col}" as _score
                FROM texts
                WHERE {filter_col} = '{g}' AND year IS NOT NULL AND "{col}" IS NOT NULL
                  AND year >= {year_min} AND year <= {year_max}{corpus_filter}{translated_filter}
            """).fetchdf()

            if len(df) < 30:
                continue

        # Apply corpus bias correction before sign flip
        if corpus_bias and "corpus_name" in df.columns:
            coefficients = corpus_bias.get("coefficients", {})
            df["_score"] = df["_score"] - df["corpus_name"].map(coefficients).fillna(0.0)

        df["_score"] = df["_score"] * sign

        # Determine split column
        split_col = None
        if split_by and split_by in df.columns:
            split_col = split_by
        elif split_by == "is_translated" and "is_translated" in df.columns:
            split_col = "is_translated"

        if split_col:
            _add_agg_arc_with_subgroups(results, df, g, split_col, bin_size, loess_span, min_texts=min_texts)
        else:
            _add_agg_arc(results, df, g, bin_size, loess_span, min_texts=min_texts)

    return results


def _add_agg_arc(results, df, label, bin_size, loess_span, min_texts=1):
    """Helper: bin a DataFrame and add an AggGenreArc to results."""
    df = df.copy()
    df["_bin"] = (df["year"] // bin_size).astype(int) * bin_size
    agg = df.groupby("_bin").agg(
        mean=("_score", "mean"),
        n_texts=("_score", "count"),
    ).reset_index()

    # Filter by min_texts
    agg = agg[agg["n_texts"] >= min_texts]

    points = [
        AggBinPoint(year=float(row["_bin"]), mean=float(row["mean"]), n_texts=int(row["n_texts"]))
        for _, row in agg.iterrows()
    ]

    loess_pts = _compute_loess(
        agg["_bin"].values.astype(float),
        agg["mean"].values.astype(float),
        span=loess_span,
    ) if len(agg) >= 5 else []

    results.append(AggGenreArc(
        genre=label,
        points=points,
        loess=loess_pts,
        n_texts_total=int(agg["n_texts"].sum()),
    ))


def _add_agg_arc_with_subgroups(results, df, label, split_col, bin_size, loess_span, top_n=10, min_texts=1):
    """Bin with subgroup-tagged points: top N values + Other + Aggregate.

    The LOESS line is computed on the overall aggregate.  Points are broken
    out by subgroup so the frontend can color them independently.
    """
    import re
    import numpy as np

    df = df.copy()
    df["_bin"] = (df["year"] // bin_size).astype(int) * bin_size

    # Parse genre_raw into a cleaner label
    if split_col == "genre_raw":
        def _parse(val):
            if not val or str(val) == "" or str(val) == "nan":
                return "(unknown)"
            parts = [p.strip() for p in re.split(r"[|,]", str(val))]
            for p in parts:
                if p and p != "Fiction" and p != "":
                    return p
            return parts[0] if parts else "(unknown)"
        df["_subgroup"] = df[split_col].apply(_parse)
    else:
        df["_subgroup"] = df[split_col].fillna("(unknown)").astype(str)

    # Identify top N subgroups by text count
    counts = df["_subgroup"].value_counts()
    top_labels = list(counts.head(top_n).index)
    df["_subgroup_display"] = df["_subgroup"].where(
        df["_subgroup"].isin(top_labels), "Other"
    )

    # Aggregate LOESS (overall, ignoring subgroups) — filtered by min_texts
    agg_all = df.groupby("_bin").agg(
        mean=("_score", "mean"),
        n_texts=("_score", "count"),
    ).reset_index()
    agg_filtered = agg_all[agg_all["n_texts"] >= min_texts]
    loess_pts = _compute_loess(
        agg_filtered["_bin"].values.astype(float),
        agg_filtered["mean"].values.astype(float),
        span=loess_span,
    ) if len(agg_filtered) >= 5 else []

    # Build points: per subgroup × bin + aggregate — filtered by min_texts
    points = []
    for (sg, b), sdf in df.groupby(["_subgroup_display", "_bin"]):
        vals = sdf["_score"].dropna()
        if len(vals) < min_texts:
            continue
        points.append(AggBinPoint(
            year=float(b), mean=float(np.mean(vals)),
            n_texts=len(vals), subgroup=str(sg),
        ))

    # Aggregate points — also filtered by min_texts
    for _, row in agg_filtered.iterrows():
        points.append(AggBinPoint(
            year=float(row["_bin"]), mean=float(row["mean"]),
            n_texts=int(row["n_texts"]), subgroup="Aggregate",
        ))

    results.append(AggGenreArc(
        genre=label,
        points=points,
        loess=loess_pts,
        n_texts_total=int(agg_all["n_texts"].sum()),
    ))


@router.get("/by-genre", response_model=list[GenreArc])
def arc_by_genre(
    col: str = DEFAULT_COL,
    genre: list[str] = Query(default=["arc_fiction"]),
    corpus: list[str] = Query(default=[]),
    year_min: float = 1565,
    year_max: float = 2020,
    loess_span: float = 0.3,
    invert: bool = True,
    period_matched: bool = False,
    corpus_adjusted: bool = False,
    model: str = "quadratic",
    bin_size: int = 10,
    is_translated: str | None = None,
    corpus_corrected: bool = False,
    min_texts: int = 1,
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
    if corpus and len(corpus) > 0:
        corpus_list = ", ".join(f"'{c}'" for c in corpus)
        corpus_sql = f" AND corpus_name IN ({corpus_list})"

    # Build translation filter
    translated_sql = ""
    if is_translated == "true":
        translated_sql = " AND is_translated = true"
    elif is_translated == "false":
        translated_sql = " AND (is_translated IS NULL OR is_translated = false)"

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
                      AND year >= {year_min} AND year <= {year_max}{corpus_sql}{translated_sql}
                """).fetchdf()
            else:
                gdf = conn.execute(f"""
                    SELECT _id, corpus_name, year, genre, {col_sql}
                    FROM texts
                    WHERE genre = '{g}'
                      AND year IS NOT NULL
                      AND year >= {year_min} AND year <= {year_max}{corpus_sql}{translated_sql}
                """).fetchdf()
        else:
            if is_arc:
                gdf = conn.execute(f"""
                    SELECT _id, corpus_name, year, arc_corpus, "{col}"
                    FROM texts
                    WHERE arc_corpus = '{g}'
                      AND year IS NOT NULL AND "{col}" IS NOT NULL
                      AND year >= {year_min} AND year <= {year_max}{corpus_sql}{translated_sql}
                """).fetchdf()
            else:
                gdf = conn.execute(f"""
                    SELECT _id, corpus_name, year, genre, "{col}"
                    FROM texts
                    WHERE genre = '{g}'
                      AND year IS NOT NULL AND "{col}" IS NOT NULL
                      AND year >= {year_min} AND year <= {year_max}{corpus_sql}{translated_sql}
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

        # Load corpus bias for match-group-based correction
        cb = None
        if corpus_corrected:
            from ...corpus_correction import load_corpus_bias
            cb = load_corpus_bias()

        adj = adjust_scores(
            gdf, score_col=score_col, min_year=year_min, max_year=year_max,
            corpus_col="corpus_name",
            fixed_effects=fe if fe else None,
            corpus_bias=cb,
            model=model, agg_bin=bin_size,
            min_texts_per_bin=min_texts,
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


@router.post("/print")
def arc_print(arcs: list[GenreArc]):
    """Render a print-quality PNG from the same data the frontend displays."""
    import os
    import matplotlib
    matplotlib.use("Agg")
    import plotnine as p9
    import pandas as pd
    import numpy as np
    from fastapi.responses import FileResponse
    from ...config import PATH_DATA

    genre_labels = {
        "arc_fiction": "Fiction", 
        "arc_poetry": "Poetry",
        "arc_periodical": "Periodical", 
        "arc_essays": "Essays",
        "arc_biography": "Biography",
        "arc_sermons": "Sermons",
    }

    all_points = []
    all_loess = []

    for arc in arcs:
        label = genre_labels.get(arc.genre, arc.genre)

        # Aggregate points by year (weighted by n_texts)
        bins: dict[float, dict] = {}
        for p in arc.points:
            yr = p.year
            if yr not in bins:
                bins[yr] = {"sum_score": 0, "sum_adj": 0, "n": 0}
            bins[yr]["sum_score"] += p.score * p.n_texts
            bins[yr]["sum_adj"] += p.adjusted * p.n_texts
            bins[yr]["n"] += p.n_texts

        for yr, b in sorted(bins.items()):
            all_points.append({
                "year": yr,
                "score": b["sum_adj"] / b["n"],  # use adjusted (matches aggregate view)
                "n_texts": b["n"],
                "genre": label,
            })

        for lp in arc.loess_aggregate:
            all_loess.append({
                "year": lp.year, "fitted": lp.fitted,
                "se_lo": lp.se_lo, "se_hi": lp.se_hi,
                "genre": label,
            })

    if not all_points:
        from fastapi import HTTPException
        raise HTTPException(404, "No data")

    agg_df = pd.DataFrame(all_points)
    ldf = pd.DataFrame(all_loess)

    # Define specific colors for genres: 
    # - Fiction (arc_fiction): black
    # - Poetry (arc_poetry): mid-gray
    # - All others: very light gray

    genre_colors = {
        "Fiction": "#000000",         # black
        "Poetry": "#888888",          # mid-gray
        "Periodical": "#bdbdbd",      # slightly less light gray
        "Essays": "#bdbdbd",          # slightly less light gray
        "Biography": "#bdbdbd",       # slightly less light gray
        "Sermons": "#bdbdbd",         # slightly less light gray
    }

    # Add or update "color" column for agg_df/ldf if needed (optional, but not required for scale)
    color_scale = p9.scale_color_manual(
        values=genre_colors,
        name="Genre",
    )
    fill_scale = p9.scale_fill_manual(
        values=genre_colors,
        name="Genre",
    )

    agg_df["genre"] = pd.Categorical(agg_df["genre"], categories=[g for g in genre_colors.keys() if g in agg_df["genre"].unique()])
    ldf["genre"] = pd.Categorical(ldf["genre"], categories=[g for g in genre_colors.keys() if g in ldf["genre"].unique()])
    
    fig = (
        p9.ggplot(agg_df, p9.aes(x="year", y="score"))
        + p9.geom_hline(yintercept=0, color="black", size=0.5, alpha=0.5, linetype="dashed")
        + p9.geom_ribbon(
            p9.aes(x="year", ymin="se_lo", ymax="se_hi", fill="genre"),
            data=ldf, alpha=0.1, inherit_aes=False,
        )
        + p9.geom_point(
            p9.aes(size="n_texts", shape="genre", color="genre"),
            alpha=0.4,
        )
        + p9.geom_line(
            p9.aes(x="year", y="fitted", linetype="genre", color="genre"),
            data=ldf, size=0.8, inherit_aes=False,
        )
        + p9.scale_size_continuous(range=(1, 5), name="Texts")
        # + p9.scale_fill_grey(start=0.5, end=0.8)
        # + p9.scale_color_grey(start=0.0, end=0.4)
        + p9.labs(
            x="Year",
            y="<< More concrete | More abstract >>                             ",
            shape="Genre", linetype="Genre", color="Genre", fill="Genre", size="# Texts"
        )
        + p9.theme_minimal()
        + p9.theme(legend_position="right", figure_size=(10, 7))
        + color_scale
        + fill_scale
    )

    fig_dir = os.path.join(PATH_DATA, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    out_path = os.path.join(fig_dir, "arc_print.png")
    fig.save(out_path, dpi=300)

    return FileResponse(out_path, media_type="image/png")


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
    global_se = np.std(residuals)
    window = max(5, int(len(x) * span))
    se_vals = np.full_like(ly, global_se)
    for i in range(len(ly)):
        lo = max(0, i - window // 2)
        hi = min(len(residuals), i + window // 2 + 1)
        if hi - lo >= 5:
            local_se = np.std(residuals[lo:hi])
            se_vals[i] = min(local_se, global_se)  # clamp to global

    points = []
    for i in range(len(lx)):
        points.append(LoessPoint(
            year=float(lx[i]),
            fitted=float(ly[i]),
            se_lo=float(ly[i] - se_vals[i]),
            se_hi=float(ly[i] + se_vals[i]),
        ))
    return points
