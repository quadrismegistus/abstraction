"""Shift-share decomposition of abstractness changes between periods."""

from fastapi import APIRouter, Query
from pydantic import BaseModel

from ..db import get_connection

router = APIRouter()

DEFAULT_COL = "Abs-Conc.Median.median"


class DecompRow(BaseModel):
    genre: str
    share_early: float
    share_late: float
    mean_early: float
    mean_late: float
    n_early: int
    n_late: int
    composition_effect: float
    within_effect: float
    interaction: float
    total_effect: float


class DecompResult(BaseModel):
    period_early: str
    period_late: str
    overall_mean_early: float
    overall_mean_late: float
    overall_change: float
    total_composition: float
    total_within: float
    total_interaction: float
    rows: list[DecompRow]


def _parse_genre_raw(genre_raw: str | None) -> str:
    """Extract the most specific non-'Fiction' genre from 'X | Y | Z' format."""
    if not genre_raw or genre_raw == "":
        return "(unknown)"
    parts = [p.strip() for p in str(genre_raw).split("|")]
    for p in parts:
        if p and p != "Fiction" and p != "":
            return p
    return parts[0] if parts else "(unknown)"


@router.get("/shift-share", response_model=DecompResult)
def shift_share(
    col: str = DEFAULT_COL,
    genre: str = "arc_fiction",
    year_early_min: int = 1700,
    year_early_max: int = 1780,
    year_late_min: int = 1850,
    year_late_max: int = 1950,
    corpus: list[str] = Query(default=[]),
    invert: bool = True,
    period_matched: bool = True,
    min_texts: int = 10,
    is_translated: str | None = None,
):
    """Shift-share decomposition of abstractness change between two periods.

    Decomposes the change in mean score into:
    - Composition effect: genre mix changed (more novels, fewer romances)
    - Within-genre effect: scores within each genre changed
    - Interaction: both changed simultaneously
    """
    import pandas as pd
    import numpy as np
    from ...analysis import assign_period_score

    conn = get_connection()

    is_arc = genre.startswith("arc_")
    filter_col = "arc_corpus" if is_arc else "genre"

    corpus_filter = ""
    if corpus and len(corpus) > 0:
        cl = ", ".join(f"'{c}'" for c in corpus)
        corpus_filter = f" AND corpus_name IN ({cl})"

    translated_filter = ""
    if is_translated == "true":
        translated_filter = " AND is_translated = true"
    elif is_translated == "false":
        translated_filter = " AND (is_translated IS NULL OR is_translated = false)"

    # Load both periods
    if period_matched:
        col_parts = col.split(".")
        source = col_parts[1] if len(col_parts) >= 2 else "Median"
        score_cols = [r[0] for r in conn.execute("DESCRIBE scores").fetchall()
                      if r[0].startswith(f"Abs-Conc.{source}.")]
        col_sql = ", ".join(f'"{c}"' for c in score_cols)

        df = conn.execute(f"""
            SELECT year, genre_raw, {col_sql}
            FROM texts
            WHERE {filter_col} = '{genre}' AND year IS NOT NULL
              AND ((year >= {year_early_min} AND year <= {year_early_max})
                OR (year >= {year_late_min} AND year <= {year_late_max}))
              {corpus_filter}{translated_filter}
        """).fetchdf()

        if len(df) == 0:
            from fastapi import HTTPException
            raise HTTPException(404, "No data")

        df = assign_period_score(df, source=source)
        df["_score"] = df["period_score"]
        df = df.dropna(subset=["_score"])
    else:
        df = conn.execute(f"""
            SELECT year, genre_raw, "{col}" as _score
            FROM texts
            WHERE {filter_col} = '{genre}' AND year IS NOT NULL AND "{col}" IS NOT NULL
              AND ((year >= {year_early_min} AND year <= {year_early_max})
                OR (year >= {year_late_min} AND year <= {year_late_max}))
              {corpus_filter}{translated_filter}
        """).fetchdf()

    if len(df) == 0:
        from fastapi import HTTPException
        raise HTTPException(404, "No data")

    sign = -1.0 if invert else 1.0
    df["_score"] = df["_score"] * sign

    # Parse genre_raw
    df["_genre"] = df["genre_raw"].apply(_parse_genre_raw)

    # Split into early/late
    early = df[(df["year"] >= year_early_min) & (df["year"] <= year_early_max)]
    late = df[(df["year"] >= year_late_min) & (df["year"] <= year_late_max)]

    if len(early) == 0 or len(late) == 0:
        from fastapi import HTTPException
        raise HTTPException(404, "No data in one of the periods")

    # Compute per-genre stats
    def genre_stats(sub):
        total = len(sub)
        g = sub.groupby("_genre").agg(
            mean=("_score", "mean"),
            n=("_score", "count"),
        ).reset_index()
        g["share"] = g["n"] / total
        return g

    ge = genre_stats(early).rename(columns={"mean": "mean_early", "n": "n_early", "share": "share_early"})
    gl = genre_stats(late).rename(columns={"mean": "mean_late", "n": "n_late", "share": "share_late"})

    # Merge (outer join so we see genres that exist in only one period)
    merged = ge.merge(gl, on="_genre", how="outer").fillna(0)

    # Filter to genres with enough texts in at least one period
    merged = merged[(merged["n_early"] >= min_texts) | (merged["n_late"] >= min_texts)]

    # Compute decomposition
    overall_early = early["_score"].mean()
    overall_late = late["_score"].mean()
    overall_change = overall_late - overall_early

    rows = []
    for _, r in merged.iterrows():
        d_share = r["share_late"] - r["share_early"]
        d_mean = r["mean_late"] - r["mean_early"]

        comp = d_share * r["mean_early"]        # composition effect
        within = r["share_early"] * d_mean       # within-genre effect
        interaction = d_share * d_mean           # interaction

        rows.append(DecompRow(
            genre=r["_genre"],
            share_early=round(r["share_early"], 4),
            share_late=round(r["share_late"], 4),
            mean_early=round(r["mean_early"], 4),
            mean_late=round(r["mean_late"], 4),
            n_early=int(r["n_early"]),
            n_late=int(r["n_late"]),
            composition_effect=round(comp, 6),
            within_effect=round(within, 6),
            interaction=round(interaction, 6),
            total_effect=round(comp + within + interaction, 6),
        ))

    # Sort by absolute total effect
    rows.sort(key=lambda r: abs(r.total_effect), reverse=True)

    return DecompResult(
        period_early=f"{year_early_min}-{year_early_max}",
        period_late=f"{year_late_min}-{year_late_max}",
        overall_mean_early=round(overall_early, 4),
        overall_mean_late=round(overall_late, 4),
        overall_change=round(overall_change, 4),
        total_composition=round(sum(r.composition_effect for r in rows), 4),
        total_within=round(sum(r.within_effect for r in rows), 4),
        total_interaction=round(sum(r.interaction for r in rows), 4),
        rows=rows,
    )
