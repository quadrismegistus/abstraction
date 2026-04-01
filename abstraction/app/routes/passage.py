"""Passage endpoint: word-level scoring and colored HTML rendering."""

import numpy as np
from fastapi import APIRouter, HTTPException, Query

from ...scoring import get_norm_dict, _modernize_score
from ...tokenize import tokenize_agnostic, get_spelling_modernizer
from ...passages import render_passage_html
from ..models import PassageResponse, PassageToken, ScoreRequest

router = APIRouter()

DEFAULT_COL = "Abs-Conc.Median.median"


def _score_text(text: str, col: str) -> PassageResponse:
    """Score a text passage, returning all tokens (words + punctuation) + HTML.

    Mirrors the logic in passages.py:_render_paragraph — iterates over all
    tokens from tokenize_agnostic, scores alphabetic ones, passes through
    punctuation as-is.
    """
    scores = get_norm_dict(col)
    spelling_d = get_spelling_modernizer()
    # Tokenize original text (preserve case), lowercase only for lookup
    tokens_raw = tokenize_agnostic(text)

    tokens = []
    n_abs = 0
    n_conc = 0
    n_neutral = 0

    for tok in tokens_raw:
        if not tok:
            continue

        if not tok[0].isalpha():
            # Skip whitespace-only tokens (spaces, newlines) —
            # the frontend handles spacing between tokens.
            if tok.strip() == '':
                continue
            # Meaningful punctuation (commas, periods, dashes, etc.)
            tokens.append(PassageToken(text=tok, is_punct=True))
            continue

        # Alphabetic word — score it (lowercase for lookup, keep original case)
        s, _ = _modernize_score(tok.lower(), scores, spelling_d)
        z = float(s) if s is not None else None
        is_abs = z is not None and z <= -1.0
        is_conc = z is not None and z >= 1.0

        if is_abs:
            n_abs += 1
        elif is_conc:
            n_conc += 1
        else:
            n_neutral += 1

        tokens.append(PassageToken(
            text=tok,
            is_punct=False,
            score=z,
            is_abstract=is_abs,
            is_concrete=is_conc,
        ))

    html = render_passage_html(text, col=col)

    return PassageResponse(
        text=text,
        tokens=tokens,
        html=html,
        n_abstract=n_abs,
        n_concrete=n_conc,
        n_neutral=n_neutral,
    )


@router.get("/{corpus}/{text_id:path}/{chunk_index}", response_model=PassageResponse)
def get_passage(
    corpus: str,
    text_id: str,
    chunk_index: int,
    col: str = DEFAULT_COL,
    chunk_size: int = Query(default=500, ge=50, le=5000),
):
    """Get a specific passage chunk with word-level scoring."""
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
