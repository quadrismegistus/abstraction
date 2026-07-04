"""
ClickHouse-backed database layer for the web app.

Replaces the previous DuckDB/scores.duckdb + LLTK metadb ATTACH plumbing.
All data lives on the local CH server:
  - abstraction.scores_{en,fr,de}    raw per-text scores (1:1 with lltk.texts)
  - abstraction.scores                wide union: per-arc, within_lang_group dedup
  - abstraction.scores_rep            wide union: per-arc, rep_only (no averaging)
  - abstraction.texts      VIEW       scores JOIN lltk.texts
  - abstraction.texts_rep  VIEW       scores_rep JOIN lltk.texts

Routes consume `abstraction.texts` / `abstraction.texts_rep` via a thin
DuckDB-compatibility shim (`CHConn`) so the existing `conn.execute(sql, params)
.fetchall() / .fetchdf()` idiom keeps working.
"""

import threading
from typing import Any, Optional, Sequence

import clickhouse_connect
import pandas as pd

from ..aggregate import get_arc_scores, get_corpus_scores


CH_HOST = "localhost"
CH_PORT = 8123
CH_USER = "lltk"
CH_PASSWORD = "lltk"
CH_DB = "abstraction"

_local = threading.local()


# Arcs whose scores come from the new CH scores_{lang} tables.
NEW_PIPELINE_ARCS = {
    "arc_fiction":    {"lang": "en"},
    "arc_fiction_fr": {"lang": "fr"},
    "arc_fiction_de": {"lang": "de"},
    "arc_poetry":     {"lang": "en"},
    "arc_biography":  {"lang": "en"},
    "arc_essays":     {"lang": "en"},
    "arc_periodical": {"lang": "en"},
    "arc_sermons":    {"lang": "en"},
}

# General-purpose historical corpora (no genre filter, no match-group dedup).
# Displayed as background reference lines in the web app.
RAW_CORPORA = {
    "spanish_pd_books": {"lang": "es", "label": "Spanish PD Books"},
    "french_pd_books":  {"lang": "fr", "label": "French PD Books"},
    "german_pd":        {"lang": "de", "label": "German PD"},
    "blbooks":          {"lang": "en", "label": "BL Books"},
}


# ──────────────────────────────────────────────────────────────────────
# DuckDB-compat shim: conn.execute(sql, params).fetchall() / .fetchdf()
# ──────────────────────────────────────────────────────────────────────

class _Cursor:
    """Mimics a DuckDB cursor so routes don't need to change shape."""

    def __init__(self, client, sql: str, params: Optional[Sequence] = None):
        self._client = client
        self._sql, self._pdict = _translate_qmarks(sql, params)
        self._df: Optional[pd.DataFrame] = None

    def _run(self) -> pd.DataFrame:
        if self._df is None:
            self._df = self._client.query_df(self._sql, parameters=self._pdict)
        return self._df

    @staticmethod
    def _clean_row(r):
        """Replace pandas NA with None so pydantic/json can serialize."""
        return tuple(None if pd.isna(v) else v for v in r)

    def fetchall(self) -> list[tuple]:
        df = self._run()
        return [self._clean_row(r) for r in df.itertuples(index=False, name=None)]

    def fetchone(self):
        df = self._run()
        if len(df) == 0:
            return None
        return self._clean_row(df.iloc[0])

    def fetchdf(self) -> pd.DataFrame:
        return self._run()


def _translate_qmarks(sql: str, params: Optional[Sequence]) -> tuple[str, dict]:
    """Convert DuckDB-style `?` positional params to CH `%(pN)s` named params.

    Naive but sufficient for the routes here — they never embed `?` inside
    string literals.
    """
    if not params:
        return sql, {}
    out_parts = []
    idx = 0
    for ch in sql:
        if ch == "?":
            out_parts.append(f"%(p{idx})s")
            idx += 1
        else:
            out_parts.append(ch)
    new_sql = "".join(out_parts)
    pdict = {f"p{i}": v for i, v in enumerate(params)}
    if idx != len(params):
        raise ValueError(
            f"parameter count mismatch: sql has {idx} placeholders, "
            f"{len(params)} values provided"
        )
    return new_sql, pdict


class CHConn:
    """Wraps a clickhouse_connect client with a duckdb-style execute() API."""

    def __init__(self, client):
        self._client = client

    def execute(self, sql: str, params: Optional[Sequence] = None) -> _Cursor:
        return _Cursor(self._client, sql, params)

    def close(self):
        self._client.close()


# ──────────────────────────────────────────────────────────────────────
# Connection factory
# ──────────────────────────────────────────────────────────────────────

def _raw_client(database: str = CH_DB):
    return clickhouse_connect.get_client(
        host=CH_HOST, port=CH_PORT,
        username=CH_USER, password=CH_PASSWORD,
        database=database,
        # Bound how long a slow/heavy query can hold a FastAPI worker thread:
        # client-side connect + send/receive timeouts, plus a server-side
        # execution cap slightly below the receive timeout.
        connect_timeout=10,
        send_receive_timeout=120,
        settings={"max_execution_time": 110},
    )


def get_connection() -> CHConn:
    """Per-thread CHConn wrapper. Routes call conn.execute(sql, params)."""
    conn = getattr(_local, "conn", None)
    if conn is None:
        conn = CHConn(_raw_client())
        _local.conn = conn
    return conn


# ──────────────────────────────────────────────────────────────────────
# Init: build abstraction.scores + abstraction.scores_rep + VIEWs
# ──────────────────────────────────────────────────────────────────────

def _backtick(c: str) -> str:
    return "`" + c.replace("`", "``") + "`"


def _build_arc_scores_df(dedup_mode: str) -> tuple[pd.DataFrame, list[str]]:
    """Call get_arc_scores for curated arcs and get_corpus_scores for raw corpora;
    concat into one wide DataFrame with (_id, arc_corpus, source_corpus, <score cols>).

    Returns (df, score_cols) where score_cols is the union of score cols across
    all arcs. Missing cols per-arc become NaN via pd.concat.
    """
    dfs = []
    for arc_name, cfg in NEW_PIPELINE_ARCS.items():
        try:
            df = get_arc_scores(arc_name, lang=cfg["lang"], dedup=dedup_mode)
        except Exception as e:
            print(f"    {arc_name}: FAILED ({e})")
            continue
        df["arc_corpus"] = arc_name
        df["source_corpus"] = arc_name
        dfs.append(df)
        print(f"    {arc_name}: {len(df):,} reps")

    for corpus_name, cfg in RAW_CORPORA.items():
        try:
            df = get_corpus_scores(corpus_name, lang=cfg["lang"])
        except Exception as e:
            print(f"    {corpus_name} (raw): FAILED ({e})")
            continue
        if df.empty:
            print(f"    {corpus_name} (raw): 0 texts — skipping")
            continue
        df["arc_corpus"] = corpus_name
        df["source_corpus"] = corpus_name
        dfs.append(df)
        print(f"    {corpus_name} (raw): {len(df):,} texts")

    if not dfs:
        return pd.DataFrame(), []
    combined = pd.concat(dfs, ignore_index=True)
    non_score_cols = {"_id", "arc_corpus", "source_corpus", "_n_versions"}
    score_cols = [c for c in combined.columns if c not in non_score_cols]
    return combined, score_cols


def _ddl_scores_table(table: str, score_cols: list[str], has_versions: bool) -> str:
    defs = [
        "`_id` String",
        "`arc_corpus` LowCardinality(String)",
        "`source_corpus` LowCardinality(String)",
    ]
    if has_versions:
        defs.append("`_n_versions` Nullable(UInt32)")
    for c in score_cols:
        defs.append(f"{_backtick(c)} Nullable(Float32) CODEC(ZSTD(3))")
    cols_sql = ",\n    ".join(defs)
    return f"""
CREATE TABLE {CH_DB}.{table} (
    {cols_sql}
) ENGINE = MergeTree() ORDER BY (arc_corpus, `_id`)
"""


def _create_texts_view(admin_client, view_name: str, scores_table: str):
    """Create abstraction.{view_name} as scores_table JOIN lltk.texts."""
    admin_client.command(f"DROP VIEW IF EXISTS {CH_DB}.{view_name}")
    admin_client.command(f"""
        CREATE VIEW {CH_DB}.{view_name} AS
        SELECT
            s.`_id` AS `_id`,
            -- Fall back to source_corpus label if LLTK doesn't have a match
            if(m.corpus = '', s.source_corpus, m.corpus) AS corpus_name,
            s.arc_corpus AS arc_corpus,
            m.title AS title,
            m.author AS author,
            m.year AS year,
            m.genre AS genre,
            m.genre_raw AS genre_raw,
            m.genre_enriched_source AS genre_enriched_source,
            m.is_translated AS is_translated,
            m.n_words AS n_words,
            m.author_norm AS author_norm,
            s.* EXCEPT (`_id`, arc_corpus, source_corpus)
        FROM {CH_DB}.{scores_table} s
        LEFT JOIN (SELECT * FROM lltk.texts FINAL) m ON s.`_id` = m.`_id`
    """)


def _tables_healthy(admin) -> bool:
    """Check whether scores + scores_rep + both VIEWs exist and have rows."""
    try:
        existing = {r[0] for r in admin.query(f"SHOW TABLES FROM {CH_DB}").result_rows}
        for name in ("scores", "scores_rep", "texts", "texts_rep"):
            if name not in existing:
                return False
        for table in ("scores", "scores_rep"):
            n = admin.query(f"SELECT count() FROM {CH_DB}.{table}").result_rows[0][0]
            if n == 0:
                return False
        return True
    except Exception:
        return False


def init_db(force: bool = False):
    """Build abstraction.scores, abstraction.scores_rep, and matching VIEWs
    on ClickHouse.

    Idempotent. By default skips rebuild if tables exist and have rows; pass
    `force=True` (or set `ABSTRACTION_REFRESH=1`) to drop + rebuild from scratch.
    """
    import os as _os
    if not force and _os.environ.get("ABSTRACTION_REFRESH") == "1":
        force = True

    admin = _raw_client()

    if not force and _tables_healthy(admin):
        counts = {
            t: admin.query(f"SELECT count() FROM {CH_DB}.{t}").result_rows[0][0]
            for t in ("scores", "scores_rep")
        }
        print(f"[app] Scores tables fresh (scores={counts['scores']:,}, scores_rep={counts['scores_rep']:,}); skipping rebuild. Pass --refresh to force.")
        admin.close()
        return

    print("[app] Building CH-side scores tables and VIEWs...")

    for table in ("scores", "scores_rep"):
        admin.command(f"DROP VIEW IF EXISTS {CH_DB}.{'texts' if table == 'scores' else 'texts_rep'}")
        admin.command(f"DROP TABLE IF EXISTS {CH_DB}.{table}")

    for table, mode in (("scores", "within_lang_group"), ("scores_rep", "rep_only")):
        print(f"\n  Building {table} (dedup={mode})...")
        df, score_cols = _build_arc_scores_df(mode)
        if df.empty:
            print(f"    no data for {table}; skipping")
            continue
        has_versions = "_n_versions" in df.columns
        admin.command(_ddl_scores_table(table, score_cols, has_versions))

        insert_cols = ["_id", "arc_corpus", "source_corpus"]
        if has_versions:
            insert_cols.append("_n_versions")
        insert_cols.extend(score_cols)
        for c in insert_cols:
            if c not in df.columns:
                df[c] = None
        df = df[insert_cols]

        CHUNK = 100_000
        for i in range(0, len(df), CHUNK):
            admin.insert_df(table, df.iloc[i:i + CHUNK])
        n = admin.query(f"SELECT count() FROM {CH_DB}.{table}").result_rows[0][0]
        print(f"    {table}: {n:,} rows, {len(score_cols)} score cols, has_versions={has_versions}")

    # Strip cross-lang leakage: per-arc, delete reps whose texts.lang doesn't
    # match the arc's declared language. This catches texts that CuratedCorpus
    # picked before lang detection flipped without clobbering legitimately
    # non-English arcs (arc_fiction_fr, arc_fiction_de).
    for table in ("scores", "scores_rep"):
        existing = admin.query(f"SHOW TABLES FROM {CH_DB}").result_rows
        if table not in {r[0] for r in existing}:
            continue
        total_pruned = 0
        for arc_name, cfg in NEW_PIPELINE_ARCS.items():  # raw corpora skip pruning — already lang-filtered
            expected_lang = cfg["lang"]
            before = admin.query(
                f"SELECT count() FROM {CH_DB}.{table} WHERE arc_corpus = %(a)s",
                parameters={"a": arc_name},
            ).result_rows[0][0]
            admin.command(f"""
                DELETE FROM {CH_DB}.{table}
                WHERE arc_corpus = %(arc)s
                  AND _id IN (
                    SELECT s._id FROM {CH_DB}.{table} s
                    INNER JOIN (SELECT _id, lang FROM lltk.texts FINAL) t ON s._id = t._id
                    WHERE s.arc_corpus = %(arc)s
                      AND t.lang != %(lang)s
                      AND t.lang IS NOT NULL
                )
            """, parameters={"arc": arc_name, "lang": expected_lang})
            after = admin.query(
                f"SELECT count() FROM {CH_DB}.{table} WHERE arc_corpus = %(a)s",
                parameters={"a": arc_name},
            ).result_rows[0][0]
            if before != after:
                total_pruned += (before - after)
                print(f"    {table}/{arc_name}: pruned {before - after} lang-mismatch rows")
        if total_pruned:
            print(f"    {table}: total pruned {total_pruned}")

    for scores_table, view_name in (("scores", "texts"), ("scores_rep", "texts_rep")):
        existing = admin.query(f"SHOW TABLES FROM {CH_DB}").result_rows
        if scores_table not in {r[0] for r in existing}:
            continue
        _create_texts_view(admin, view_name, scores_table)
        sample = admin.query(f"SELECT count() FROM {CH_DB}.{view_name}").result_rows[0][0]
        print(f"  VIEW {CH_DB}.{view_name}: {sample:,} rows")

    # Reset any per-thread connections so the next get_connection() sees new views.
    _local.conn = None
    print("\n[app] CH scores tables + VIEWs ready.")
    admin.close()
