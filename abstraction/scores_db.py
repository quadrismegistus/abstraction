"""
Unified per-text scores database.

Stores raw 1:1 scores (text_id → norm columns) per language. Aggregation
(match-group averaging, dedup, etc.) is deferred to query time so different
consumers can pick different policies — see `aggregate.py` for the readers.

Tables:
  scores_en  — English per-text scores. Columns: _id PK + all English norm cols.
  scores_fr  — French per-text scores. Columns: _id PK + all French norm cols.
  scores_de  — German per-text scores. Same pattern.
  scoring_meta — bookkeeping: (table_name, n_rows, last_updated, notes).

Schema is built dynamically from the columns of whatever DataFrame is being
written, so adding a new norm column doesn't require a migration — just
re-run scoring with the updated allnorms and the table grows columns
automatically (new rows get the new column; old rows are NULL there).
"""

import os
from datetime import datetime, timezone
from typing import Iterable

import duckdb
import pandas as pd

from .config import PATH_SCORES_DB

LANG_TABLES = {"en": "scores_en", "fr": "scores_fr", "de": "scores_de"}


def _connect(db_path=None, read_only=False):
    db_path = db_path or PATH_SCORES_DB
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    return duckdb.connect(db_path, read_only=read_only)


def init_db(db_path=None):
    """Create empty scores DB with bookkeeping table. Per-language tables
    are created lazily on first write so the schema can mirror allnorms."""
    con = _connect(db_path)
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS scoring_meta (
            table_name VARCHAR PRIMARY KEY,
            n_rows BIGINT,
            last_updated TIMESTAMP,
            notes VARCHAR
        )
        """
    )
    con.close()


def _quote_col(c: str) -> str:
    return '"' + c.replace('"', '""') + '"'


def _ensure_table(con, table: str, score_cols: Iterable[str]):
    """Create the per-language scores table if missing, with all needed cols.
    If already exists, ALTER TABLE to add any new score columns."""
    exists = con.execute(
        "SELECT 1 FROM information_schema.tables WHERE table_name = ?",
        [table],
    ).fetchone()
    cols_sql = ",\n            ".join(f"{_quote_col(c)} DOUBLE" for c in score_cols)
    if not exists:
        con.execute(
            f"""
            CREATE TABLE {table} (
                _id VARCHAR PRIMARY KEY,
                {cols_sql}
            )
            """
        )
        return

    existing = {
        row[0]
        for row in con.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = ?",
            [table],
        ).fetchall()
    }
    for c in score_cols:
        if c not in existing:
            con.execute(f"ALTER TABLE {table} ADD COLUMN {_quote_col(c)} DOUBLE")


def write_scores(
    df: pd.DataFrame,
    lang: str,
    db_path: str = None,
    upsert: bool = True,
):
    """Insert per-text scores into scores_<lang>.

    Parameters
    ----------
    df : DataFrame
        Must have an `_id` column. Other columns are treated as score cols.
    lang : str
        Language key — one of 'en', 'fr', 'de'.
    upsert : bool
        If True (default), replace existing rows for the same _id. If False,
        skip ids already present.
    """
    if lang not in LANG_TABLES:
        raise ValueError(f"lang must be one of {list(LANG_TABLES)}, got {lang!r}")
    if "_id" not in df.columns:
        raise ValueError("df must have an `_id` column")
    table = LANG_TABLES[lang]
    score_cols = [c for c in df.columns if c != "_id"]
    if not score_cols:
        raise ValueError("df has no score columns besides _id")

    con = _connect(db_path)
    init_db(db_path)
    _ensure_table(con, table, score_cols)

    # Use a temp view so we can do INSERT ... SELECT
    con.register("incoming_df", df)

    if upsert:
        # Delete then insert (DuckDB has UPSERT but it's painful with dynamic cols)
        ids = df["_id"].tolist()
        if ids:
            # Drop in batches to avoid huge IN clause
            for i in range(0, len(ids), 10000):
                batch = ids[i : i + 10000]
                ph = ",".join(["?"] * len(batch))
                con.execute(f"DELETE FROM {table} WHERE _id IN ({ph})", batch)

    cols_sql = ", ".join(_quote_col(c) for c in (["_id"] + score_cols))
    con.execute(
        f"INSERT INTO {table} ({cols_sql}) SELECT {cols_sql} FROM incoming_df"
    )
    n = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    con.execute(
        """
        INSERT OR REPLACE INTO scoring_meta (table_name, n_rows, last_updated, notes)
        VALUES (?, ?, ?, ?)
        """,
        [table, n, datetime.now(timezone.utc), None],
    )
    con.unregister("incoming_df")
    con.close()
    return n


def read_scores(lang: str, ids=None, db_path: str = None) -> pd.DataFrame:
    """Read per-text scores. Optionally filter to a specific id set."""
    if lang not in LANG_TABLES:
        raise ValueError(f"lang must be one of {list(LANG_TABLES)}, got {lang!r}")
    table = LANG_TABLES[lang]
    con = _connect(db_path, read_only=True)
    if ids is None:
        df = con.execute(f"SELECT * FROM {table}").fetchdf()
    else:
        ids = list(ids)
        # Use temp table for big id sets
        con.execute("CREATE TEMP TABLE _read_ids (_id VARCHAR PRIMARY KEY)")
        for i in range(0, len(ids), 10000):
            batch = ids[i : i + 10000]
            ph = ",".join(["(?)"] * len(batch))
            con.execute(f"INSERT INTO _read_ids VALUES {ph}", batch)
        df = con.execute(
            f"SELECT s.* FROM {table} s JOIN _read_ids r ON s._id = r._id"
        ).fetchdf()
    con.close()
    return df


def db_stats(db_path: str = None) -> pd.DataFrame:
    """Return current row counts per table."""
    con = _connect(db_path, read_only=True)
    try:
        df = con.execute(
            "SELECT * FROM scoring_meta ORDER BY table_name"
        ).fetchdf()
    finally:
        con.close()
    return df
