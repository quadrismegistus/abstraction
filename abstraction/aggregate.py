"""
Downstream aggregation of per-text scores from ClickHouse `abstraction.scores_*`.

Per-text scores are stored 1:1 in `abstraction.scores_{en,fr,de}` on the same CH
server as `lltk.match_groups` / `lltk.texts`. This module provides aggregation
policies for arc analysis: representatives only, within-language match-group
averaging.

By keeping aggregation here (not in the scorer), different consumers (notebook,
web app, paper figures) can pick different policies without re-scoring. The
per-language scores table also prevents cross-language contamination at JOIN
time.
"""

import os
import sys
from typing import Iterable, Literal, Optional

import pandas as pd


LANG_TABLES = {"en": "scores_en", "fr": "scores_fr", "de": "scores_de", "es": "scores_es"}

CH_HOST = "localhost"
CH_PORT = 8123
CH_USER = "lltk"
CH_PASSWORD = "lltk"


DedupMode = Literal["rep_only", "within_lang_group"]


def _ch_client():
    import clickhouse_connect
    return clickhouse_connect.get_client(
        host=CH_HOST, port=CH_PORT,
        username=CH_USER, password=CH_PASSWORD,
    )


def _backtick(c: str) -> str:
    return "`" + c.replace("`", "``") + "`"


def _load_arc_reps(arc: str) -> list[str]:
    """Load representative _ids for an LLTK arc corpus.

    Uses LLTK's CuratedCorpus metadata which already applies dedup (by='oldest'
    for ArcFiction) and annotation-based exclusions/additions.
    """
    sys.path.insert(0, os.path.expanduser("~/github/lltk"))
    import lltk

    arc_corpus = lltk.load(arc)
    if arc_corpus is None:
        raise ValueError(f"LLTK corpus {arc!r} not found")
    return sorted(set(arc_corpus.metadata()["_id"]))


def _resolve_score_cols(ch, table: str, score_cols) -> list[str]:
    if score_cols is None:
        cols = ch.query(f"SELECT name FROM system.columns WHERE database='abstraction' AND table=%(t)s ORDER BY position",
                        parameters={"t": table}).result_rows
        score_cols = [r[0] for r in cols if r[0] != "_id"]
    score_cols = list(score_cols)
    if not score_cols:
        raise ValueError(f"no score columns found in abstraction.{table}")
    return score_cols


def _insert_temp_reps(ch, rep_ids: list[str]) -> str:
    """Create a short-lived Memory table of rep _ids and return its fully-qualified name.

    Memory tables are session-local to the CH server (not thread-local on client) so
    we include a unique suffix per call to avoid collision across concurrent workers.
    """
    import uuid
    name = f"abstraction._arc_reps_{uuid.uuid4().hex[:12]}"
    ch.command(f"CREATE TABLE {name} (`_id` String) ENGINE = Memory")
    ch.insert(name.split(".", 1)[1], [[rid] for rid in rep_ids],
              column_names=["_id"], database="abstraction")
    return name


def _drop_table(ch, fq_name: str):
    try:
        ch.command(f"DROP TABLE IF EXISTS {fq_name}")
    except Exception:
        pass


def get_arc_scores(
    arc: str,
    lang: str = "en",
    score_cols: Optional[Iterable[str]] = None,
    dedup: DedupMode = "within_lang_group",
    ch_client=None,
) -> pd.DataFrame:
    """Aggregate per-text scores into per-rep scores for an arc corpus.

    Parameters
    ----------
    arc : str
        LLTK arc corpus name, e.g. 'arc_fiction'. Reps loaded from LLTK metadata.
    lang : str
        Language of the scores table to query. 'en'|'fr'|'de'.
    score_cols : iterable of str, optional
        Score columns to aggregate. Default: all columns of scores_<lang>.
    dedup : 'rep_only' | 'within_lang_group'
        - rep_only: return each rep's own per-text score (no aggregation).
        - within_lang_group: average all match-group members' scores per rep.
          Cross-language members are naturally excluded because they don't
          live in the same-language scores table. Singletons (reps with no
          match group) pass through as their own score.
    ch_client : clickhouse_connect client, optional
        Pre-existing client. A fresh one is created if None.

    Returns
    -------
    DataFrame keyed by rep _id with one column per requested score.
    For within_lang_group mode, also includes `_n_versions` (group-member count).
    """
    if lang not in LANG_TABLES:
        raise ValueError(f"lang must be one of {list(LANG_TABLES)}, got {lang!r}")
    table = LANG_TABLES[lang]
    fq_table = f"abstraction.{table}"

    rep_ids = _load_arc_reps(arc)
    if not rep_ids:
        raise ValueError(f"arc {arc!r} returned no rep _ids")

    owns_client = ch_client is None
    ch = ch_client if ch_client is not None else _ch_client()

    score_cols = _resolve_score_cols(ch, table, score_cols)

    reps_table = _insert_temp_reps(ch, rep_ids)
    try:
        if dedup == "rep_only":
            cols_sql = ", ".join(f"s.{_backtick(c)} AS {_backtick(c)}" for c in score_cols)
            df = ch.query_df(f"""
                SELECT s._id AS _id, {cols_sql}
                FROM {fq_table} s
                INNER JOIN {reps_table} r ON s._id = r._id
            """)
            return df

        if dedup != "within_lang_group":
            raise ValueError(f"unknown dedup mode: {dedup!r}")

        # within_lang_group: for each rep, find match group members (or
        # fall back to the rep itself for singletons), AVG scores over them.
        # Use FINAL on match_groups since it's ReplacingMergeTree.
        # group_id=0 is a real value in match_groups, so detect singletons via
        # anti-join on _id rather than a sentinel check on group_id.
        avg_exprs = ", ".join(f"avg(s.{_backtick(c)}) AS {_backtick(c)}" for c in score_cols)

        df = ch.query_df(f"""
            WITH
              mg AS (SELECT _id, group_id FROM lltk.match_groups FINAL),
              matched AS (
                SELECT r._id AS rep_id, mg2._id AS member_id
                FROM {reps_table} r
                INNER JOIN mg mg1 ON r._id = mg1._id
                INNER JOIN mg mg2 ON mg1.group_id = mg2.group_id
              ),
              singletons AS (
                SELECT r._id AS rep_id, r._id AS member_id
                FROM {reps_table} r
                WHERE r._id NOT IN (SELECT _id FROM mg)
              ),
              members AS (
                SELECT rep_id, member_id FROM matched
                UNION ALL
                SELECT rep_id, member_id FROM singletons
              )
            SELECT m.rep_id AS _id,
                   uniqExact(s._id) AS _n_versions,
                   {avg_exprs}
            FROM members m
            INNER JOIN {fq_table} s ON m.member_id = s._id
            GROUP BY m.rep_id
        """)
        return df
    finally:
        _drop_table(ch, reps_table)
        if owns_client:
            ch.close()


def get_corpus_scores(
    corpus: str,
    lang: str = "en",
    score_cols: Optional[Iterable[str]] = None,
    ch_client=None,
) -> pd.DataFrame:
    """Return per-text scores for all confirmed-lang texts in a raw corpus.

    No match-group dedup. Filters by lltk.texts.lang to exclude texts whose
    language detection contradicts the declared lang (unreliable corpora like
    internet_archive can include wrong-lang texts).
    """
    if lang not in LANG_TABLES:
        raise ValueError(f"lang must be one of {list(LANG_TABLES)}, got {lang!r}")
    table = LANG_TABLES[lang]
    fq_table = f"abstraction.{table}"

    owns_client = ch_client is None
    ch = ch_client if ch_client is not None else _ch_client()
    try:
        score_cols = _resolve_score_cols(ch, table, score_cols)
        cols_sql = ", ".join(f"s.{_backtick(c)}" for c in score_cols)
        df = ch.query_df(f"""
            SELECT s._id AS _id, {cols_sql}
            FROM {fq_table} s
            INNER JOIN (SELECT _id, lang FROM lltk.texts FINAL) t ON s._id = t._id
            WHERE s._id LIKE %(prefix)s
              AND (t.lang IS NULL OR t.lang = %(lang)s)
        """, parameters={"prefix": f"_{corpus}/%", "lang": lang})
        return df
    finally:
        if owns_client:
            ch.close()


def passage_abstractness_by_lang(
    langs: Iterable[str] = ("en", "fr"),
    period_bins: Iterable[int] = (1600, 1650, 1700, 1750, 1800),
    score_col: str = "Abs-Conc.Median.median",
    scheme: str = "p500",
    invert: bool = True,
    ch_client=None,
) -> pd.DataFrame:
    """Mean passage-level abstractness per (lang, period) bin.

    Queries abstraction.passage_scores joined to lltk.texts. Returns long-form
    with columns: lang, period, n, mean_abs, se_abs. Period labels match the
    'START-END' format used by largeliterarymodels.analysis.compare_cross_language
    so the two dataframes concatenate cleanly.

    Args:
        langs: languages to include (each must exist in lltk.texts.lang).
        period_bins: bin edges; N bins produce N-1 periods.
        score_col: which key from the scores Map to pull.
        scheme: passage scheme filter.
        invert: if True (default), negate the raw score so + = more abstract.
    """
    bins = list(period_bins)
    if len(bins) < 2:
        raise ValueError("period_bins needs >= 2 edges")
    langs = list(langs)

    owns_client = ch_client is None
    ch = ch_client if ch_client is not None else _ch_client()
    sign = "-" if invert else ""
    # Build CASE WHEN for period labelling
    case_parts = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        case_parts.append(f"WHEN t.year >= {lo} AND t.year < {hi} THEN '{lo}-{hi-1}'")
    case_sql = "CASE " + " ".join(case_parts) + " END"
    try:
        df = ch.query_df(f"""
            SELECT {case_sql} AS period,
                   t.lang AS lang,
                   count() AS n,
                   {sign}avg(ps.scores[%(col)s]) AS mean_abs,
                   stddevSamp(ps.scores[%(col)s]) / sqrt(count()) AS se_abs
            FROM abstraction.passage_scores ps
            INNER JOIN (SELECT _id, lang, year FROM lltk.texts FINAL) t
              ON ps._id = t._id
            WHERE ps.scheme = %(scheme)s
              AND t.lang IN %(langs)s
              AND t.year >= {bins[0]} AND t.year < {bins[-1]}
            GROUP BY period, lang
            ORDER BY period, lang
        """, parameters={"col": score_col, "scheme": scheme, "langs": langs})
        return df.dropna(subset=["period"]).reset_index(drop=True)
    finally:
        if owns_client:
            ch.close()
