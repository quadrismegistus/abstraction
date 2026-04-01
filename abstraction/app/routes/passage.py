"""Passage endpoint: server-rendered HTML with data-z attributes."""

from fastapi import APIRouter, HTTPException, Query

from ...passages import render_passage_body, render_passage_html
from ...scoring import get_norm_dict, _modernize_score
from ...tokenize import tokenize_agnostic, get_spelling_modernizer
from ..models import PassageResponse, ScoreRequest

router = APIRouter()

DEFAULT_COL = "Abs-Conc.Median.median"


def _count_words(txt: str, col: str):
    """Count abstract/concrete/neutral words in text."""
    scores = get_norm_dict(col)
    spelling_d = get_spelling_modernizer()
    tokens = tokenize_agnostic(txt)
    n_abs = n_conc = n_neutral = 0
    for tok in tokens:
        if not tok or not tok[0].isalpha():
            continue
        s, _ = _modernize_score(tok.lower(), scores, spelling_d)
        if s is None:
            n_neutral += 1
        elif s <= -1.0:
            n_abs += 1
        elif s >= 1.0:
            n_conc += 1
        else:
            n_neutral += 1
    return n_abs, n_conc, n_neutral


def _score_text(text: str, col: str) -> PassageResponse:
    """Score a text passage using centralized rendering from passages.py."""
    body_html = render_passage_body(text, col=col, mode="color")
    print_body_html = render_passage_body(text, col=col, mode="print")
    print_html = render_passage_html(text, col=col, show_legend=True)
    n_abs, n_conc, n_neutral = _count_words(text, col)

    return PassageResponse(
        text=text,
        body_html=body_html,
        print_body_html=print_body_html,
        print_html=print_html,
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
