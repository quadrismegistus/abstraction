"""Shared request-parameter validation for the web app routes.

Defends two layers:

- the SQL layer: identifier-like params (``col``) that routes interpolate
  into ClickHouse SQL as quoted identifiers, validated against the actual
  column set of the ``abstraction.scores`` / ``scores_rep`` tables;
- the filesystem layer: ``corpus`` / ``text_id`` path params that reach
  ``Corpus.read_text()`` / glob lookups, checked for traversal.

String *values* (genre names, corpus filters, ...) are NOT handled here —
routes must pass those through ``?`` placeholders (see ``db._translate_qmarks``).
"""

import re
import threading

from fastapi import HTTPException

# Norm-column shape: letters, digits, underscore, dot, hyphen.
# (Real columns look like "Abs-Conc.Median.median", "IC.blbooks.C17",
# "_n_versions".)  This alone makes f-string interpolation inside a
# double-quoted ClickHouse identifier safe; the membership check below
# additionally guarantees the column actually exists.
_COL_RE = re.compile(r"^[A-Za-z0-9_.\-]+$")
_MAX_COL_LEN = 200

# Columns of the scores tables that are identifiers, not score columns.
_NON_SCORE_COLS = {"_id", "arc_corpus", "source_corpus"}

_cols_lock = threading.Lock()
_known_cols: set[str] | None = None


def _fetch_score_cols() -> "set[str] | None":
    """DESCRIBE the CH scores tables; return their column set, or None if CH
    is unreachable / the tables don't exist yet."""
    from .db import get_connection

    cols: set[str] = set()
    try:
        conn = get_connection()
    except Exception:
        return None
    for table in ("scores", "scores_rep"):
        try:
            cols.update(r[0] for r in conn.execute(f"DESCRIBE {table}").fetchall())
        except Exception:
            continue
    cols -= _NON_SCORE_COLS
    return cols or None


def get_known_cols(refresh: bool = False) -> "set[str] | None":
    """Cached column set of the scores tables. ``refresh=True`` re-fetches
    (used on a cache miss so an ``--refresh`` rebuild that adds columns
    doesn't require an app restart)."""
    global _known_cols
    with _cols_lock:
        if _known_cols is None or refresh:
            fetched = _fetch_score_cols()
            if fetched is not None:
                _known_cols = fetched
        return _known_cols


def validate_col(col: str) -> str:
    """Validate a user-supplied norm-column name; raise HTTPException(400)
    on anything that isn't a known score column.

    1. Shape check (identifier-safe characters only) — prevents SQL
       injection outright.
    2. Membership check against the live CH column set (cached; one
       refresh-on-miss). Skipped if ClickHouse is unreachable, in which
       case the shape check still holds (needed so the trajectory/passage
       endpoints, which don't require CH, keep working offline).
    """
    if not col or len(col) > _MAX_COL_LEN or not _COL_RE.match(col):
        raise HTTPException(status_code=400, detail="Invalid column name")
    known = get_known_cols()
    if known is None:
        return col
    if col not in known:
        known = get_known_cols(refresh=True)
        if known is not None and col not in known:
            raise HTTPException(status_code=400, detail=f"Unknown column: {col}")
    return col


def validate_corpus_name(corpus: str) -> str:
    """Corpus names are single path segments (snake_case dirs under
    PATH_CORPORA). Reject anything that could change directories."""
    if (
        not corpus
        or corpus in (".", "..")
        or "/" in corpus
        or "\\" in corpus
        or "\x00" in corpus
    ):
        raise HTTPException(status_code=400, detail="Invalid corpus name")
    return corpus


def validate_text_id(text_id: str) -> str:
    """Text ids legitimately contain '/' (e.g.
    'Eighteenth-Century_Fiction/richards.01'), so separators are allowed;
    reject absolute paths, '..' segments, and control characters.
    Final containment is additionally enforced in ``Corpus.text_path``."""
    if not text_id or "\x00" in text_id:
        raise HTTPException(status_code=400, detail="Invalid text id")
    if text_id.startswith("/") or text_id.startswith("\\"):
        raise HTTPException(status_code=400, detail="Invalid text id")
    for parts in (text_id.split("/"), text_id.split("\\")):
        if ".." in parts:
            raise HTTPException(status_code=400, detail="Invalid text id")
    return text_id
