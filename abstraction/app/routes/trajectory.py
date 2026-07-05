"""Trajectory endpoint: intra-text abstractness scored over passage chunks."""

import json
import os

import numpy as np
from fastapi import APIRouter, HTTPException, Query

from ...config import PATH_DATA
from ...norms import norms_version
from ...scoring import score_freqs, score_psg
from ..models import TrajectoryChunk, TrajectoryResponse
from ..validation import validate_col, validate_corpus_name, validate_text_id

router = APIRouter()

DEFAULT_COL = "Abs-Conc.Median.median"
CACHE_DIR = os.path.join(PATH_DATA, "stash", "trajectories")


def _cache_path(corpus: str, text_id: str, chunk_size: int, col: str,
                norms_ver: str = ""):
    """On-disk cache path for a trajectory.

    The allnorms fingerprint is baked into the filename (audit §4.1) so that
    regenerating the norms self-invalidates the cache: files written under
    old norms simply stop matching and become orphans.
    """
    safe_col = col.replace(".", "_")
    safe_id = text_id.replace("/", "__")
    d = os.path.join(CACHE_DIR, corpus)
    os.makedirs(d, exist_ok=True)
    suffix = f"_{norms_ver}" if norms_ver else ""
    return os.path.join(d, f"{safe_id}_n{chunk_size}_{safe_col}{suffix}.json")


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


# Corpora that are authoritatively non-English. Used when the per-text `lang`
# metadata is missing (most corpora don't populate it — LLTK's canonical source
# is `lltk.texts.lang`, populated by `lltk db-detect-langs`, but the abstraction
# web app doesn't always have it on hand).
_CORPUS_LANG_DEFAULTS = {
    "german_fiction": "de",
    "dta": "de",
    "german_pd": "de",
    "arc_fiction_de": "de",
    "artfl": "fr",
    "gallica_literary_fictions": "fr",
    "french_pd_books": "fr",
    "arc_fiction_fr": "fr",
    "arc_fiction_es": "es",
}

_SUPPORTED_LANGS = {"en", "fr", "de", "es"}


def resolve_lang(corpus: str, meta: dict | None = None, explicit: str | None = None) -> str:
    """Pick the norm language for a text.

    Priority: explicit query param > per-text `lang` metadata > corpus default > 'en'.
    """
    for candidate in (explicit, (meta or {}).get("lang")):
        if candidate in _SUPPORTED_LANGS:
            return candidate
    return _CORPUS_LANG_DEFAULTS.get(corpus.lower(), "en")


def _get_text_and_metadata(corpus_name: str, text_id: str):
    """Try LLTK first, fall back to abstraction Corpus."""
    camel = _snake_to_camel(corpus_name)

    # First, resolve the full metadata ID (score IDs are often simplified)
    from ...corpus import load_corpus
    import glob

    full_id = text_id
    meta = {}
    try:
        corpus = load_corpus(camel)
        # Try partial match against metadata
        row = corpus.metadata[corpus.metadata["id"].str.contains(text_id, regex=False)]
        if len(row) == 0:
            row = corpus.metadata[corpus.metadata["id"] == text_id]
        if len(row):
            full_id = str(row.iloc[0]["id"])
            for k in ("title", "author", "year", "lang"):
                if k in row.columns:
                    v = row.iloc[0][k]
                    if v is not None and str(v) != "nan":
                        meta[k] = v
    except Exception:
        pass

    # Try LLTK with the full ID (gives sentence-boundary chunks)
    try:
        import lltk
        c = lltk.load(camel)
        if c is not None:
            # Try full_id first, then original text_id
            t = None
            for tid in [full_id, text_id]:
                try:
                    t = c[tid]
                    if t is not None and t.txt:
                        break
                except Exception:
                    continue
            if t is not None:
                txt = t.txt
                if txt:
                    # Fill in metadata from LLTK if we didn't get it from corpus
                    for k in ("title", "author", "year", "lang"):
                        if k not in meta:
                            v = t.get(k, None)
                            if v is not None:
                                meta[k] = v
                    return txt, meta, t
    except Exception:
        pass

    # Fallback: read text file directly
    try:
        corpus = load_corpus(camel)
        txt = None

        # Try reading by full_id, then text_id
        for tid in [full_id, text_id]:
            try:
                txt = corpus.read_text(tid)
                if txt:
                    break
            except FileNotFoundError:
                continue

        # Last resort: glob search (containment-checked: every candidate
        # must resolve inside the corpus txt/ directory)
        if not txt:
            txt_dir = os.path.join(corpus.path, "txt")
            base = os.path.realpath(txt_dir)
            pattern = os.path.join(txt_dir, f"*{text_id}*")
            matches = [
                m for m in glob.glob(pattern)
                if os.path.realpath(m).startswith(base + os.sep)
            ]
            if matches:
                with open(matches[0], encoding="utf-8", errors="ignore") as f:
                    txt = f.read()

        if txt:
            return txt, meta, None
    except Exception:
        pass

    return None, None, None


def _score_via_lltk(lltk_text, chunk_size: int, col: str, lang: str = "en"):
    """Use LLTK passages API for sentence-boundary-respecting chunks."""
    chunks = []
    try:
        passages = lltk_text.passages(n=chunk_size)
        for i, psg in enumerate(passages.texts()):
            freqs = psg.freqs()
            score = score_freqs(dict(freqs), col=col, lang=lang) if freqs else None
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
    period_matched: bool = False,
    lang: str | None = None,
):
    """Compute abstractness trajectory for a single text."""
    corpus = validate_corpus_name(corpus)
    text_id = validate_text_id(text_id)
    col = validate_col(col)

    def _cp_for(resolved: str) -> str:
        key = col if not period_matched else col + "_pm"
        if resolved != "en":
            key = f"{key}_{resolved}"
        # Non-English scoring can fall back to English norms for columns
        # missing from the language's allnorms (scoring.get_norm_dict), so
        # stamp non-English caches with BOTH fingerprints.
        ver = norms_version(resolved)
        if resolved != "en":
            ver = f"{ver}-{norms_version('en')}"
        return _cache_path(corpus, text_id, chunk_size, key, norms_ver=ver)

    # Early lang guess from corpus default + explicit query param, used for the
    # first cache lookup. Per-text `meta['lang']` may override later.
    early_lang = resolve_lang(corpus, None, lang)
    cp = _cp_for(early_lang)
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

    resolved_lang = resolve_lang(corpus, meta, lang)
    meta["lang"] = resolved_lang

    # If meta yielded a different lang than our early guess, re-check cache.
    if resolved_lang != early_lang:
        cp = _cp_for(resolved_lang)
        cached = _load_cached(cp)
        if cached is not None:
            return TrajectoryResponse(**cached)

    # Resolve period-matched column
    if period_matched:
        year = meta.get("year")
        if year is not None:
            from ...analysis import CENTURY_BINS
            col_parts = col.split(".")
            source = col_parts[1] if len(col_parts) >= 2 else "Median"
            try:
                yr = float(year)
                for lo, hi, label in CENTURY_BINS:
                    if lo <= yr < hi:
                        col = f"Abs-Conc.{source}.{label}"
                        break
            except (TypeError, ValueError):
                pass

    # Try LLTK sentence-boundary chunking first
    chunks = None
    if lltk_text is not None:
        chunks = _score_via_lltk(lltk_text, chunk_size, col, lang=resolved_lang)

    # Fallback: simple word-split chunking
    if chunks is None:
        raw_chunks = _chunk_text_simple(txt, chunk_size)
        chunks = []
        for i, rc in enumerate(raw_chunks):
            score = score_psg(rc["text"], col=col, lang=resolved_lang)
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
