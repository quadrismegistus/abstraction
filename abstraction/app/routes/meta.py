"""Metadata endpoints: corpora, norms, genres."""

import re

from fastapi import APIRouter

from ..db import get_connection
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
    "median": "Median (all centuries)",
    "orig": "Original (empirical)",
}


@router.get("/corpora", response_model=list[CorpusInfo])
def list_corpora():
    conn = get_connection()
    rows = conn.execute("""
        SELECT corpus_name,
               COUNT(*) as n_texts,
               MIN(year) as year_min,
               MAX(year) as year_max
        FROM texts
        WHERE year IS NOT NULL
        GROUP BY corpus_name
        ORDER BY corpus_name
    """).fetchall()

    results = []
    for name, n, ymin, ymax in rows:
        genres = [r[0] for r in conn.execute(
            "SELECT DISTINCT genre_harmonized FROM texts WHERE corpus_name = ? AND genre_harmonized IS NOT NULL",
            (name,)
        ).fetchall()]
        results.append(CorpusInfo(
            name=name, n_texts=n,
            year_min=ymin, year_max=ymax,
            genres=sorted(genres),
        ))

    conn.close()
    return results


@router.get("/norms", response_model=list[NormInfo])
def list_norms():
    conn = get_connection()
    # Get column names from the table
    cursor = conn.execute("PRAGMA table_info(texts)")
    columns = [row[1] for row in cursor.fetchall()]
    conn.close()

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
    rows = conn.execute(
        "SELECT DISTINCT genre_harmonized FROM texts WHERE genre_harmonized IS NOT NULL ORDER BY genre_harmonized"
    ).fetchall()
    conn.close()
    return [r[0] for r in rows]
