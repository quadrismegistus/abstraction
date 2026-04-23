"""
Server-side ClickHouse scoring for a language's corpora.

Steps:
  1. Upload abstraction.allnorms_{lang} (word + score cols) to CH
  2. CREATE abstraction.scores_{lang} (DDL mirrors scores_de/fr/en)
  3. INSERT...SELECT with ARRAY JOIN + LEFT JOIN against allnorms — runs
     entirely in CH, ~minutes for 100K+ texts vs hours client-side.

Usage:
    python scripts/score_serverside_ch.py --lang es --corpora spanish_pd_books impact_es
    python scripts/score_serverside_ch.py --lang es --corpora spanish_pd_books --drop
    python scripts/score_serverside_ch.py --lang es --skip-allnorms   # allnorms already on CH
    python scripts/score_serverside_ch.py --lang es --dry-run         # print SQL, don't INSERT
"""

import argparse
import sys
import os
import time

import clickhouse_connect
import pandas as pd

sys.path.insert(0, os.path.expanduser("~/github/lltk"))

CH_HOST = "localhost"
CH_PORT = 8123
CH_USER = "lltk"
CH_PASSWORD = "lltk"
CH_DB = "abstraction"
ALLNORMS_CHUNK = 200_000


def get_ch():
    return clickhouse_connect.get_client(
        host=CH_HOST, port=CH_PORT,
        username=CH_USER, password=CH_PASSWORD,
        database=CH_DB,
    )


def load_allnorms(lang: str):
    if lang == "es":
        from abstraction.norms_es import get_allnorms_es
        return get_allnorms_es(remove_stopwords=True)
    if lang == "fr":
        from abstraction.norms_fr import get_allnorms_fr
        return get_allnorms_fr(remove_stopwords=True)
    if lang == "de":
        from abstraction.norms_de import get_allnorms_de
        return get_allnorms_de(remove_stopwords=True)
    from abstraction.norms import get_allnorms
    return get_allnorms(remove_stopwords=True)


# ── allnorms upload ────────────────────────────────────────────────────────────

def upload_allnorms(lang: str, drop: bool = False):
    ch = get_ch()
    table = f"allnorms_{lang}"
    allnorms = load_allnorms(lang)
    allnorms = allnorms[allnorms.index.notna() & ~allnorms.index.duplicated()].copy()
    allnorms.index.name = "word"
    score_cols = list(allnorms.columns)
    n = len(allnorms)
    print(f"[allnorms_{lang}] {n:,} words × {len(score_cols)} cols")

    if drop:
        ch.command(f"DROP TABLE IF EXISTS {CH_DB}.{table}")

    existing = {r[0] for r in ch.query(f"SHOW TABLES FROM {CH_DB}").result_rows}
    if table in existing and not drop:
        ch_n = ch.query(f"SELECT count() FROM {CH_DB}.{table}").result_rows[0][0]
        if ch_n == n:
            print(f"[allnorms_{lang}] already on CH with {ch_n:,} rows — skipping upload")
            return score_cols
        print(f"[allnorms_{lang}] CH has {ch_n:,} rows != {n:,} — rebuilding")
        ch.command(f"DROP TABLE IF EXISTS {CH_DB}.{table}")

    col_defs = ["`word` String"] + [
        f"`{c}` Nullable(Float32)" for c in score_cols
    ]
    ddl = (
        f"CREATE TABLE {CH_DB}.{table} (\n    "
        + ",\n    ".join(col_defs)
        + f"\n) ENGINE = MergeTree() ORDER BY word"
    )
    ch.command(ddl)
    print(f"[allnorms_{lang}] created table")

    df = allnorms.reset_index()
    t0 = time.time()
    inserted = 0
    for i in range(0, n, ALLNORMS_CHUNK):
        chunk = df.iloc[i : i + ALLNORMS_CHUNK].copy()
        for c in score_cols:
            chunk[c] = chunk[c].astype("float32")
        ch.insert_df(table, chunk)
        inserted += len(chunk)
        print(f"  uploaded {inserted:,}/{n:,} ({100*inserted/n:.1f}%)")
    print(f"[allnorms_{lang}] upload done in {time.time()-t0:.0f}s")
    return score_cols


# ── scores table DDL ───────────────────────────────────────────────────────────

def create_scores_table(lang: str, score_cols: list, drop: bool = False):
    ch = get_ch()
    table = f"scores_{lang}"
    if drop:
        ch.command(f"DROP TABLE IF EXISTS {CH_DB}.{table}")
    existing = {r[0] for r in ch.query(f"SHOW TABLES FROM {CH_DB}").result_rows}
    if table in existing:
        n = ch.query(f"SELECT count() FROM {CH_DB}.{table}").result_rows[0][0]
        print(f"[scores_{lang}] table already exists ({n:,} rows) — use --drop to rebuild")
        return
    col_defs = ["`_id` String"] + [
        f"`{c}` Nullable(Float32) CODEC(ZSTD(3))" for c in score_cols
    ]
    ddl = (
        f"CREATE TABLE {CH_DB}.{table} (\n    "
        + ",\n    ".join(col_defs)
        + f"\n) ENGINE = MergeTree() ORDER BY `_id`"
    )
    ch.command(ddl)
    print(f"[scores_{lang}] created table ({len(score_cols)} score cols)")


# ── server-side INSERT ─────────────────────────────────────────────────────────

def build_insert_sql(lang: str, score_cols: list, corpus_filters: list) -> str:
    """Build INSERT...SELECT with ARRAY JOIN + allnorms LEFT JOIN."""
    exprs = []
    for c in score_cols:
        qc = f"`{c}`"
        exprs.append(
            f"    sumIf(toFloat32(cnt) * n.{qc}, isNotNull(n.{qc})) /\n"
            f"    nullIf(sumIf(toFloat32(cnt), isNotNull(n.{qc})), 0) AS {qc}"
        )
    exprs_sql = ",\n".join(exprs)

    where_clauses = " OR ".join(
        f"tf._id LIKE '_{corpus}/%'" for corpus in corpus_filters
    )

    return f"""INSERT INTO {CH_DB}.scores_{lang}
SELECT
    tf._id,
{exprs_sql}
FROM lltk.text_freqs tf FINAL
ARRAY JOIN mapKeys(tf.freqs) AS word, mapValues(tf.freqs) AS cnt
LEFT JOIN {CH_DB}.allnorms_{lang} n ON n.word = word
WHERE {where_clauses}
GROUP BY tf._id
SETTINGS max_memory_usage = 30000000000, max_threads = 8"""


def run_insert(lang: str, score_cols: list, corpus_filters: list, dry_run: bool = False):
    sql = build_insert_sql(lang, score_cols, corpus_filters)
    print(f"\n[scores_{lang}] INSERT SQL:\n{sql}\n")
    if dry_run:
        print("[dry-run] skipping INSERT")
        return

    ch = get_ch()
    # Count expected texts
    where = " OR ".join(f"_id LIKE '_{c}/%'" for c in corpus_filters)
    expected = ch.query(
        f"SELECT count() FROM lltk.text_freqs FINAL WHERE {where}"
    ).result_rows[0][0]
    print(f"[scores_{lang}] expected ~{expected:,} texts")

    t0 = time.time()
    ch.command(sql)
    elapsed = time.time() - t0

    n = ch.query(f"SELECT count() FROM {CH_DB}.scores_{lang}").result_rows[0][0]
    print(f"[scores_{lang}] done in {elapsed:.0f}s — {n:,} rows inserted")

    # Sanity check: median col
    med = "Abs-Conc.Median.median"
    if med in score_cols:
        stats = ch.query(
            f"SELECT avg(`{med}`), stddevSamp(`{med}`), countIf(isNotNull(`{med}`)) "
            f"FROM {CH_DB}.scores_{lang}"
        ).result_rows[0]
        print(f"[scores_{lang}] {med}: mean={stats[0]:.4f} sd={stats[1]:.4f} n={stats[2]:,}")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="es", choices=["en", "fr", "de", "es"])
    ap.add_argument("--corpora", nargs="+", default=["spanish_pd_books", "impact_es"],
                    help="LLTK corpus names to score (default: spanish_pd_books impact_es)")
    ap.add_argument("--drop", action="store_true", help="Drop and rebuild both tables")
    ap.add_argument("--skip-allnorms", action="store_true",
                    help="Skip allnorms upload (assume already on CH)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print INSERT SQL but don't execute")
    args = ap.parse_args()

    if args.skip_allnorms:
        # Read score_cols from existing CH table
        ch = get_ch()
        rows = ch.query(f"DESCRIBE TABLE {CH_DB}.allnorms_{args.lang}").result_rows
        score_cols = [r[0] for r in rows if r[0] != "word"]
        print(f"[allnorms_{args.lang}] using existing CH table ({len(score_cols)} cols)")
    else:
        score_cols = upload_allnorms(args.lang, drop=args.drop)

    create_scores_table(args.lang, score_cols, drop=args.drop)
    run_insert(args.lang, score_cols, args.corpora, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
