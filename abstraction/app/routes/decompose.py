"""Shift-share decomposition of abstractness changes between periods."""

from fastapi import APIRouter, Query
from pydantic import BaseModel

from ..db import get_connection

router = APIRouter()

DEFAULT_COL = "Abs-Conc.Median.median"


class DecompRow(BaseModel):
    category: str
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
    decompose_by: str
    period_early: str
    period_late: str
    overall_mean_early: float
    overall_mean_late: float
    overall_change: float
    total_composition: float
    total_within: float
    total_interaction: float
    rows: list[DecompRow]


class MultiDecompResult(BaseModel):
    results: list[DecompResult]


def _parse_genre_raw(genre_raw: str | None) -> str:
    """Extract the most specific non-'Fiction' genre from 'X | Y | Z' format."""
    if not genre_raw or genre_raw == "":
        return "(unknown)"
    parts = [p.strip() for p in str(genre_raw).split("|")]
    for p in parts:
        if p and p != "Fiction" and p != "":
            return p
    return parts[0] if parts else "(unknown)"


def _decompose(df_early, df_late, cat_col, min_texts=5):
    """Run shift-share decomposition on two DataFrames with _score and cat_col."""
    import numpy as np

    if len(df_early) == 0 or len(df_late) == 0:
        return None

    overall_early = df_early["_score"].mean()
    overall_late = df_late["_score"].mean()
    overall_change = overall_late - overall_early

    n_early_total = len(df_early)
    n_late_total = len(df_late)

    def genre_stats(sub, total):
        g = sub.groupby(cat_col).agg(
            mean=("_score", "mean"),
            n=("_score", "count"),
        ).reset_index()
        g["share"] = g["n"] / total
        return g

    ge = genre_stats(df_early, n_early_total).rename(
        columns={"mean": "mean_early", "n": "n_early", "share": "share_early"})
    gl = genre_stats(df_late, n_late_total).rename(
        columns={"mean": "mean_late", "n": "n_late", "share": "share_late"})

    merged = ge.merge(gl, on=cat_col, how="outer").fillna(0)
    merged = merged[(merged["n_early"] >= min_texts) | (merged["n_late"] >= min_texts)]

    rows = []
    for _, r in merged.iterrows():
        d_share = r["share_late"] - r["share_early"]
        d_mean = r["mean_late"] - r["mean_early"]

        comp = d_share * r["mean_early"]
        within = r["share_early"] * d_mean
        interaction = d_share * d_mean

        rows.append(DecompRow(
            category=str(r[cat_col]),
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

    rows.sort(key=lambda r: abs(r.total_effect), reverse=True)

    return DecompResult(
        decompose_by=cat_col,
        period_early="",  # filled by caller
        period_late="",
        overall_mean_early=round(overall_early, 4),
        overall_mean_late=round(overall_late, 4),
        overall_change=round(overall_change, 4),
        total_composition=round(sum(r.composition_effect for r in rows), 4),
        total_within=round(sum(r.within_effect for r in rows), 4),
        total_interaction=round(sum(r.interaction for r in rows), 4),
        rows=rows,
    )


@router.get("/shift-share", response_model=MultiDecompResult)
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
    min_texts: int = 5,
    is_translated: str | None = None,
):
    """Shift-share decomposition by genre_raw, corpus, and is_translated.

    Returns three decompositions simultaneously:
    - By genre_raw: which specific genres drove the change?
    - By corpus: which corpora drove the change?
    - By is_translated: how much is translation vs original?
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

    if period_matched:
        col_parts = col.split(".")
        source = col_parts[1] if len(col_parts) >= 2 else "Median"
        score_cols = [r[0] for r in conn.execute("DESCRIBE scores").fetchall()
                      if r[0].startswith(f"Abs-Conc.{source}.")]
        col_sql = ", ".join(f'"{c}"' for c in score_cols)

        df = conn.execute(f"""
            SELECT year, genre_raw, corpus_name, is_translated, {col_sql}
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
            SELECT year, genre_raw, corpus_name, is_translated, "{col}" as _score
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
    df["_genre_raw"] = df["genre_raw"].apply(_parse_genre_raw)

    # Parse is_translated to readable labels
    df["_translated"] = df["is_translated"].apply(
        lambda x: "Translated" if x is True or x == "True" else
                  "Original" if x is False or x == "False" else "(unknown)")

    # Split into early/late
    early = df[(df["year"] >= year_early_min) & (df["year"] <= year_early_max)]
    late = df[(df["year"] >= year_late_min) & (df["year"] <= year_late_max)]

    if len(early) == 0 or len(late) == 0:
        from fastapi import HTTPException
        raise HTTPException(404, "No data in one of the periods")

    period_early = f"{year_early_min}-{year_early_max}"
    period_late = f"{year_late_min}-{year_late_max}"

    results = []

    # Decompose by genre_raw
    r = _decompose(early, late, "_genre_raw", min_texts)
    if r:
        r.decompose_by = "genre_raw"
        r.period_early = period_early
        r.period_late = period_late
        results.append(r)

    # Decompose by corpus
    r = _decompose(early, late, "corpus_name", min_texts)
    if r:
        r.decompose_by = "corpus"
        r.period_early = period_early
        r.period_late = period_late
        results.append(r)

    # Decompose by is_translated
    r = _decompose(early, late, "_translated", min_texts)
    if r:
        r.decompose_by = "is_translated"
        r.period_early = period_early
        r.period_late = period_late
        results.append(r)

    return MultiDecompResult(results=results)
