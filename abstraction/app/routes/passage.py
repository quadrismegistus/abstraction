"""Passage endpoint: word-level scoring and colored HTML rendering."""

import numpy as np
from fastapi import APIRouter, HTTPException, Query

from ...scoring import score_words
from ...passages import render_passage_html
from ..models import PassageResponse, ScoredWord, ScoreRequest

router = APIRouter()

DEFAULT_COL = "Abs-Conc.Median.median"


def _score_text(text: str, col: str) -> PassageResponse:
    """Score a text passage and return word-level data + HTML."""
    df = score_words(text, col=col)
    words = []
    for _, row in df.iterrows():
        words.append(ScoredWord(
            position=int(row["position"]),
            word=row["word"],
            score=float(row["score"]) if not np.isnan(row["score"]) else None,
            is_abstract=bool(row.get("is_abstract", False)),
            is_concrete=bool(row.get("is_concrete", False)),
        ))

    html = render_passage_html(text, col=col)

    return PassageResponse(text=text, words=words, html=html)


@router.get("/{corpus}/{text_id:path}/{chunk_index}", response_model=PassageResponse)
def get_passage(
    corpus: str,
    text_id: str,
    chunk_index: int,
    col: str = DEFAULT_COL,
    chunk_size: int = Query(default=500, ge=50, le=5000),
):
    """Get a specific passage chunk with word-level scoring."""
    # Load the text and extract the requested chunk
    from .trajectory import _get_text_and_metadata, _chunk_text_simple

    txt, _meta, lltk_text = _get_text_and_metadata(corpus, text_id)
    if txt is None:
        raise HTTPException(status_code=404, detail=f"Text not found: {corpus}/{text_id}")

    # Try LLTK chunking
    chunk_text = None
    if lltk_text is not None:
        try:
            passages = lltk_text.passages(n=chunk_size)
            for i, psg in enumerate(passages.texts()):
                if i == chunk_index:
                    chunk_text = psg.txt
                    break
        except Exception:
            pass

    # Fallback: simple word-split
    if chunk_text is None:
        chunks = _chunk_text_simple(txt, chunk_size)
        if chunk_index < 0 or chunk_index >= len(chunks):
            raise HTTPException(status_code=404, detail=f"Chunk index {chunk_index} out of range (0-{len(chunks)-1})")
        chunk_text = chunks[chunk_index]["text"]

    return _score_text(chunk_text, col)


@router.post("/score", response_model=PassageResponse)
def score_arbitrary_text(req: ScoreRequest):
    """Score arbitrary user-provided text."""
    return _score_text(req.text, req.col)
