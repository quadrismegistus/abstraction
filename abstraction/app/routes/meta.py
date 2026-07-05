"""Metadata endpoints: corpora, norms, genres."""

import re

from fastapi import APIRouter

from ..db import get_connection, RAW_CORPORA
from ..models import CorpusInfo, NormInfo

router = APIRouter()


# Human-readable labels for norm sources
SOURCE_LABELS = {
    "PAV-Conc": "Paivio Concreteness",
    "PAV-Imag": "Paivio Imageability",
    "MRC-Conc": "MRC Concreteness",
    "MRC-Imag": "MRC Imageability",
    "MT-Conc": "Brysbaert Concreteness",
    "LSN-Imag": "Lancaster Imagery",
    "LSN-Hapt": "Lancaster Haptic",
    "Median": "Median (all sources)",
}

PERIOD_LABELS = {
    "C16": "16th century",
    "C17": "17th century",
    "C18": "18th century",
    "C19": "19th century",
    "C20": "20th century",
    "C21": "21st century",
    "median": "Median (all centuries)",
    "orig": "Original (empirical)",
}


@router.get("/corpora", response_model=list[CorpusInfo])
def list_corpora():
    conn = get_connection()
    # Single pass: year stats over dated texts + distinct genres over all texts
    # (genres deliberately include undated rows, matching the old per-corpus query).
    df = conn.execute("""
        SELECT corpus_name,
               countIf(year IS NOT NULL) AS n_texts,
               minIf(year, year IS NOT NULL) AS year_min,
               maxIf(year, year IS NOT NULL) AS year_max,
               groupUniqArrayIf(genre, genre IS NOT NULL AND genre != '') AS genres
        FROM texts
        GROUP BY corpus_name
        HAVING n_texts > 0
        ORDER BY corpus_name
    """).fetchdf()

    return [
        CorpusInfo(
            name=row["corpus_name"], n_texts=int(row["n_texts"]),
            year_min=row["year_min"], year_max=row["year_max"],
            genres=sorted(row["genres"]),
        )
        for _, row in df.iterrows()
    ]


@router.get("/norms", response_model=list[NormInfo])
def list_norms():
    conn = get_connection()
    # DuckDB: get column names from the scores table
    columns = [row[0] for row in conn.execute("DESCRIBE scores").fetchall()]

    norms = []
    for col in columns:
        m = re.match(r"Abs-Conc\.(.+)\.(.+)", col)
        if m:
            source, period = m.groups()
            source_label = SOURCE_LABELS.get(source, source)
            period_label = PERIOD_LABELS.get(period, period)
            norms.append(NormInfo(
                col=col,
                source=source,
                period=period,
                label=f"{source_label} ({period_label})",
            ))

    return norms


@router.get("/genres", response_model=list[str])
def list_genres():
    conn = get_connection()
    # Arc corpora first, then regular genres
    arcs = conn.execute(
        "SELECT DISTINCT arc_corpus FROM scores WHERE arc_corpus IS NOT NULL ORDER BY arc_corpus"
    ).fetchall()
    genres = conn.execute(
        "SELECT DISTINCT genre FROM texts WHERE genre IS NOT NULL AND genre != '' ORDER BY genre"
    ).fetchall()
    arc_list = [r[0] for r in arcs]
    genre_list = [r[0] for r in genres]
    return arc_list + [g for g in genre_list if g not in arc_list]


@router.get("/raw-corpora")
def list_raw_corpora():
    """Return available raw (unfiltered) corpora with labels, langs, and text counts."""
    conn = get_connection()
    results = []
    for corpus_id, cfg in RAW_CORPORA.items():
        try:
            row = conn.execute(
                "SELECT COUNT(*) FROM texts WHERE arc_corpus = ?", [corpus_id]
            ).fetchone()
            n = row[0] if row else 0
        except Exception:
            n = 0
        results.append({
            "id": corpus_id,
            "label": cfg["label"],
            "lang": cfg["lang"],
            "n_texts": n,
        })
    return results
