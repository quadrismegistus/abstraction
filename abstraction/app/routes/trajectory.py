"""Trajectory endpoint: intra-text abstractness scored over passage chunks."""

import json
import os

import numpy as np
from fastapi import APIRouter, HTTPException, Query

from ...config import PATH_DATA
from ...scoring import score_freqs, score_psg
from ..models import TrajectoryChunk, TrajectoryResponse

router = APIRouter()

DEFAULT_COL = "Abs-Conc.Median.median"
CACHE_DIR = os.path.join(PATH_DATA, "stash", "trajectories")


def _cache_path(corpus: str, text_id: str, chunk_size: int, col: str):
    safe_col = col.replace(".", "_")
    safe_id = text_id.replace("/", "__")
    d = os.path.join(CACHE_DIR, corpus)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{safe_id}_n{chunk_size}_{safe_col}.json")


def _load_cached(path):
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError):
            os.remove(path)
    return None


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        import numpy as _np
        if isinstance(obj, (_np.integer,)):
            return int(obj)
        if isinstance(obj, (_np.floating,)):
            return float(obj)
        if isinstance(obj, _np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _save_cache(path, data):
    with open(path, "w") as f:
        json.dump(data, f, cls=_NumpyEncoder)


def _chunk_text_simple(text: str, chunk_size: int):
    """Split text into chunks of approximately chunk_size words."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk_words = words[i : i + chunk_size]
        chunks.append({
            "text": " ".join(chunk_words),
            "n_words": len(chunk_words),
            "start_word": i,
        })
    return chunks


def _snake_to_camel(name: str) -> str:
    """Convert snake_case corpus name to CamelCase."""
    return "".join(w.capitalize() for w in name.split("_"))


def _get_text_and_metadata(corpus_name: str, text_id: str):
    """Try LLTK first, fall back to abstraction Corpus."""
    camel = _snake_to_camel(corpus_name)

    # Try LLTK
    try:
        import lltk
        c = lltk.load(camel)
        if c is not None:
            t = c[text_id]
            if t is not None:
                txt = t.txt
                if txt:
                    meta = {}
                    for k in ("title", "author", "year"):
                        v = t.get(k, None)
                        if v is not None:
                            meta[k] = v
                    return txt, meta, t
    except Exception:
        pass

    # Fallback: abstraction Corpus
    from ...corpus import load_corpus
    import glob
    try:
        corpus = load_corpus(camel)

        # Try direct read first
        txt = None
        try:
            txt = corpus.read_text(text_id)
        except FileNotFoundError:
            pass

        # If direct read fails, search for matching text file
        if not txt:
            txt_dir = os.path.join(corpus.path, "txt")
            # Try glob match: *{text_id}*
            pattern = os.path.join(txt_dir, f"*{text_id}*")
            matches = glob.glob(pattern)
            if matches:
                with open(matches[0], encoding="utf-8", errors="ignore") as f:
                    txt = f.read()

        # Also try matching text_id against metadata IDs
        meta = {}
        row = corpus.metadata[corpus.metadata["id"].str.contains(text_id, regex=False)]
        if len(row) == 0:
            row = corpus.metadata[corpus.metadata["id"] == text_id]
        if len(row):
            for k in ("title", "author", "year"):
                if k in row.columns:
                    v = row.iloc[0][k]
                    if v is not None and str(v) != "nan":
                        meta[k] = v

            # If we still don't have text, try reading by the matched metadata ID
            if not txt:
                full_id = row.iloc[0]["id"]
                try:
                    txt = corpus.read_text(full_id)
                except FileNotFoundError:
                    pass

        if txt:
            return txt, meta, None
    except Exception:
        pass

    return None, None, None


def _score_via_lltk(lltk_text, chunk_size: int, col: str):
    """Use LLTK passages API for sentence-boundary-respecting chunks."""
    chunks = []
    try:
        passages = lltk_text.passages(n=chunk_size)
        for i, psg in enumerate(passages.texts()):
            freqs = psg.freqs()
            score = score_freqs(dict(freqs), col=col) if freqs else None
            n_words = psg.get("num_words", sum(freqs.values()) if freqs else 0)
            start = psg.get("word_start", i * chunk_size)
            chunks.append(TrajectoryChunk(
                index=i,
                score=float(score) if score is not None and not np.isnan(score) else None,
                n_words=int(n_words),
                start_word=int(start),
            ))
    except Exception:
        return None
    return chunks if chunks else None


@router.get("/{corpus}/{text_id:path}", response_model=TrajectoryResponse)
def get_trajectory(
    corpus: str,
    text_id: str,
    col: str = DEFAULT_COL,
    chunk_size: int = Query(default=500, ge=50, le=5000),
):
    """Compute abstractness trajectory for a single text."""
    # Check cache
    cp = _cache_path(corpus, text_id, chunk_size, col)
    cached = _load_cached(cp)
    if cached is not None:
        return TrajectoryResponse(**cached)

    # Load text
    txt, meta, lltk_text = _get_text_and_metadata(corpus, text_id)
    if txt is None:
        raise HTTPException(status_code=404, detail=f"Text not found: {corpus}/{text_id}")

    meta = meta or {}
    # Sanitize metadata for JSON serialization (numpy types → Python builtins)
    meta = {k: int(v) if hasattr(v, 'item') and isinstance(v.item(), int) else
               float(v) if hasattr(v, 'item') else
               str(v) if not isinstance(v, (str, int, float, type(None))) else v
            for k, v in meta.items()}

    # Try LLTK sentence-boundary chunking first
    chunks = None
    if lltk_text is not None:
        chunks = _score_via_lltk(lltk_text, chunk_size, col)

    # Fallback: simple word-split chunking
    if chunks is None:
        raw_chunks = _chunk_text_simple(txt, chunk_size)
        chunks = []
        for i, rc in enumerate(raw_chunks):
            score = score_psg(rc["text"], col=col)
            chunks.append(TrajectoryChunk(
                index=i,
                score=float(score) if score is not None and not np.isnan(score) else None,
                n_words=rc["n_words"],
                start_word=rc["start_word"],
            ))

    # Overall score
    scores = [c.score for c in chunks if c.score is not None]
    overall = float(np.mean(scores)) if scores else None

    result = TrajectoryResponse(
        metadata=meta,
        chunks=chunks,
        overall_score=overall,
    )

    # Cache
    _save_cache(cp, result.model_dump())

    return result
