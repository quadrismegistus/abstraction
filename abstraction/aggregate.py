"""
Downstream aggregation of per-text scores from `scores.duckdb`.

Per-text scores are stored 1:1 in `scores_<lang>` tables. This module
provides aggregation policies for arc analysis: representatives only,
within-language match-group averaging, etc.

By keeping aggregation here (not in the scorer), different consumers
(notebook, web app, paper figures) can pick different policies without
re-scoring, and cross-language match groups can be excluded from averaging
without contaminating the scores DB.
"""

import os
import sys
from typing import Iterable, Literal, Optional

import duckdb
import pandas as pd

from .config import PATH_SCORES_DB
from .scores_db import LANG_TABLES, _quote_col


PATH_LLTK_METADB = os.path.expanduser("~/lltk_data/data/metadb.duckdb")
PATH_LLTK_MATCHES = os.path.expanduser("~/lltk_data/data/metadb_matches.duckdb")


DedupMode = Literal["rep_only", "within_lang_group"]


def get_arc_scores(
    arc: str,
    lang: str = "en",
    score_cols: Optional[Iterable[str]] = None,
    dedup: DedupMode = "within_lang_group",
    scores_db_path: str = None,
    matches_db_path: str = None,
    metadb_path: str = None,
    cross_lang_arc: Optional[str] = None,
    con: duckdb.DuckDBPyConnection = None,
) -> pd.DataFrame:
    """Aggregate per-text scores into per-rep scores for an arc corpus.

    Parameters
    ----------
    arc : str
        LLTK arc corpus name, e.g. 'arc_fiction'. The reps are loaded from
        LLTK metadata via the metadb. If you want to bypass LLTK, pass `con`
        with `arc_reps` already attached/created.
    lang : str
        Language of the scores table to query. 'en'|'fr'|'de'.
    score_cols : iterable of str, optional
        Score columns to aggregate. Default: all columns of scores_<lang>.
    dedup : 'rep_only' | 'within_lang_group'
        - rep_only: return each rep's own per-text score (no aggregation).
        - within_lang_group: average all match-group members' scores per rep.
          Cross-language groups are excluded if `cross_lang_arc` is provided.
    cross_lang_arc : str, optional
        Companion arc in another language whose match-group overlap with `arc`
        should disqualify a group from averaging. E.g. cross_lang_arc='arc_fiction_fr'
        when arc='arc_fiction' excludes the 97 known translation-pair groups.
    con : DuckDBPyConnection, optional
        Pre-existing connection. Useful if LLTK has DBs locked.

    Returns
    -------
    DataFrame keyed by rep _id with one column per requested score.
    """
    if lang not in LANG_TABLES:
        raise ValueError(f"lang must be one of {list(LANG_TABLES)}, got {lang!r}")
    table = LANG_TABLES[lang]

    # Load LLTK metadata FIRST — this opens metadb and matches via LLTK's own
    # conn. Doing this before our duckdb.connect avoids file-handle conflicts.
    sys.path.insert(0, os.path.expanduser("~/github/lltk"))
    import lltk

    arc_corpus = lltk.load(arc)
    if arc_corpus is None:
        raise ValueError(f"LLTK corpus {arc!r} not found")
    rep_ids = sorted(set(arc_corpus.load_metadata()["_id"]))

    cl_ids = None
    if cross_lang_arc:
        cl_corpus = lltk.load(cross_lang_arc)
        if cl_corpus is None:
            raise ValueError(f"LLTK corpus {cross_lang_arc!r} not found")
        cl_ids = sorted(set(cl_corpus.load_metadata()["_id"]))

    # For match groups, query LLTK's conn directly (already open).
    lltk_conn = lltk.db.conn
    try:
        lltk_conn.execute("SELECT 1 FROM match_db.match_groups LIMIT 1")
    except Exception:
        lltk_conn.execute(f"ATTACH '{matches_db_path or PATH_LLTK_MATCHES}' AS match_db (READ_ONLY)")

    owns_con = con is None
    if owns_con:
        # New conn that attaches scores DB only — no LLTK DBs to avoid lock conflicts.
        con = duckdb.connect(":memory:")
        con.execute("PRAGMA threads=8")
        con.execute(f"ATTACH '{scores_db_path or PATH_SCORES_DB}' AS sdb (READ_ONLY)")
    scores_ref = f"sdb.{table}"

    # Resolve score_cols
    if score_cols is None:
        cols = con.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = 'sdb' AND table_name = ? "
            "AND column_name != '_id' ORDER BY ordinal_position",
            [table],
        ).fetchall()
        score_cols = [c[0] for c in cols]
    score_cols = list(score_cols)
    if not score_cols:
        raise ValueError(f"no score columns found in sdb.{table}")
    if dedup == "rep_only":
        # Load rep ids into our conn and JOIN with scores
        con.execute("CREATE TEMP TABLE _arc_reps (_id VARCHAR PRIMARY KEY)")
        for i in range(0, len(rep_ids), 10000):
            batch = rep_ids[i : i + 10000]
            ph = ",".join(["(?)"] * len(batch))
            con.execute(f"INSERT INTO _arc_reps VALUES {ph}", batch)
        cols_sql = ", ".join(f"s.{_quote_col(c)}" for c in score_cols)
        df = con.execute(
            f"""
            SELECT s._id AS _id, {cols_sql}
            FROM {scores_ref} s
            JOIN _arc_reps r ON s._id = r._id
            """
        ).fetchdf()
        con.execute("DROP TABLE _arc_reps")
        if owns_con:
            con.close()
        return df

    if dedup != "within_lang_group":
        raise ValueError(f"unknown dedup mode: {dedup!r}")

    # within_lang_group: query LLTK's conn for the rep→member mapping (small),
    # then load it into our conn for the JOIN against scores.
    rep_ids_param = rep_ids
    cl_clause = ""
    cl_param = []
    if cl_ids is not None:
        cl_clause = """
            AND mg1.group_id NOT IN (
                SELECT mg_x.group_id
                FROM match_db.match_groups mg_x
                WHERE mg_x._id IN (SELECT UNNEST(?::VARCHAR[]))
            )
        """
        cl_param = [cl_ids]

    rep_to_member = lltk_conn.execute(
        f"""
        WITH arc_reps AS (
            SELECT UNNEST(?::VARCHAR[]) AS _id
        ),
        arc_with_groups AS (
            SELECT r._id AS rep_id, mg1.group_id
            FROM arc_reps r
            JOIN match_db.match_groups mg1 ON r._id = mg1._id
            WHERE 1=1
            {cl_clause}
        ),
        with_members AS (
            SELECT awg.rep_id, mg2._id AS member_id
            FROM arc_with_groups awg
            JOIN match_db.match_groups mg2 ON awg.group_id = mg2.group_id
        ),
        -- Reps with no usable groups (no match groups at all, or only
        -- cross-lang groups) fall back to their own _id as the single member.
        singletons AS (
            SELECT r._id AS rep_id, r._id AS member_id
            FROM arc_reps r
            WHERE r._id NOT IN (SELECT awg.rep_id FROM arc_with_groups awg)
        )
        SELECT rep_id, member_id FROM with_members
        UNION ALL
        SELECT rep_id, member_id FROM singletons
        """,
        [rep_ids_param] + cl_param,
    ).fetchdf()

    con.register("_rtm_df", rep_to_member)
    con.execute("CREATE TEMP TABLE _rep_to_member AS SELECT * FROM _rtm_df")
    con.unregister("_rtm_df")

    avg_exprs = ", ".join(f"AVG(s.{_quote_col(c)}) AS {_quote_col(c)}" for c in score_cols)
    df = con.execute(
        f"""
        SELECT m.rep_id AS _id,
               COUNT(DISTINCT s._id) AS _n_versions,
               {avg_exprs}
        FROM _rep_to_member m
        JOIN {scores_ref} s ON m.member_id = s._id
        GROUP BY m.rep_id
        """
    ).fetchdf()

    con.execute("DROP TABLE IF EXISTS _rep_to_member")
    if owns_con:
        con.close()

    return df
