"""
Migrate scores.duckdb -> ClickHouse abstraction.scores_{en,fr,de}.

Agreed DDL (2026-04-19):
  MergeTree() ORDER BY _id, all score cols Nullable(Float32) CODEC(ZSTD(3)).
  _id plain String. Backticked `Abs-Conc.{source}.{period}` column names preserved.

Usage:
  python scripts/migrate_scores_to_ch.py --lang fr --dry-run   # creates + loads; safe to rerun
  python scripts/migrate_scores_to_ch.py --lang all            # en + de + fr
  python scripts/migrate_scores_to_ch.py --lang fr --drop      # drop first, then recreate

Dry-run is the same as a real run — it's all idempotent (DROP TABLE IF EXISTS + CREATE
+ INSERT). The --drop flag is explicit for when you want to force a rebuild.
"""

import argparse
import time
import duckdb
import clickhouse_connect


DUCKDB_PATH = "data/scores/scores.duckdb"
CH_HOST = "localhost"
CH_PORT = 8123
CH_USER = "lltk"
CH_PASSWORD = "lltk"
CH_DB = "abstraction"
CHUNK_SIZE = 100_000


def get_ch_client():
    return clickhouse_connect.get_client(
        host=CH_HOST,
        port=CH_PORT,
        username=CH_USER,
        password=CH_PASSWORD,
        database=CH_DB,
    )


def get_duckdb_conn():
    return duckdb.connect(DUCKDB_PATH, read_only=True)


def ddl_for_table(table_name: str, score_cols: list[str]) -> str:
    """Build CREATE TABLE DDL with backticked Nullable(Float32) score columns."""
    col_defs = ["`_id` String"]
    for c in score_cols:
        col_defs.append(f"`{c}` Nullable(Float32) CODEC(ZSTD(3))")
    cols_sql = ",\n    ".join(col_defs)
    return f"""
CREATE TABLE {CH_DB}.{table_name} (
    {cols_sql}
) ENGINE = MergeTree() ORDER BY `_id`
"""


def migrate_one(lang: str, drop: bool = False):
    duck = get_duckdb_conn()
    ch = get_ch_client()
    table = f"scores_{lang}"

    cols_df = duck.execute(f"DESCRIBE scores_{lang}").fetchdf()
    all_cols = cols_df["column_name"].tolist()
    assert all_cols[0] == "_id", f"expected _id first, got {all_cols[0]}"
    score_cols = all_cols[1:]
    n_rows = duck.execute(f"SELECT COUNT(*) FROM scores_{lang}").fetchone()[0]

    print(f"[{lang}] source: {n_rows:,} rows × {len(score_cols)} score cols")

    if drop:
        print(f"[{lang}] dropping existing {CH_DB}.{table} ...")
        ch.command(f"DROP TABLE IF EXISTS {CH_DB}.{table}")

    existing = ch.query(f"SHOW TABLES FROM {CH_DB}").result_rows
    existing_names = {r[0] for r in existing}
    if table in existing_names and not drop:
        ch_n = ch.query(f"SELECT COUNT(*) FROM {CH_DB}.{table}").result_rows[0][0]
        if ch_n == n_rows:
            print(f"[{lang}] {table} already exists with {ch_n:,} rows — skipping load (use --drop to rebuild)")
            return
        else:
            print(f"[{lang}] {table} exists but has {ch_n:,} rows != {n_rows:,} source — rebuilding")
            ch.command(f"DROP TABLE {CH_DB}.{table}")

    print(f"[{lang}] creating {CH_DB}.{table} ...")
    ch.command(ddl_for_table(table, score_cols))

    print(f"[{lang}] streaming in {CHUNK_SIZE:,}-row chunks ...")
    select_cols = ", ".join([f'"{c}"' for c in all_cols])
    cur = duck.execute(f"SELECT {select_cols} FROM scores_{lang}")

    inserted = 0
    t0 = time.time()
    while True:
        batch = cur.fetch_df_chunk(CHUNK_SIZE // 2048 + 1)
        if batch is None or len(batch) == 0:
            break
        ch.insert_df(table, batch)
        inserted += len(batch)
        elapsed = time.time() - t0
        rate = inserted / elapsed if elapsed > 0 else 0
        eta = (n_rows - inserted) / rate / 60 if rate > 0 else 0
        print(f"  [{lang}] inserted {inserted:,}/{n_rows:,} ({100*inserted/n_rows:.1f}%, rate={rate:,.0f}/s, ETA={eta:.1f}min)")

    elapsed = time.time() - t0
    print(f"[{lang}] insert done in {elapsed:.1f}s ({inserted/elapsed:,.0f} rows/s)")

    ch_n = ch.query(f"SELECT COUNT(*) FROM {CH_DB}.{table}").result_rows[0][0]
    print(f"[{lang}] CH row count: {ch_n:,} (source {n_rows:,}) — {'MATCH' if ch_n == n_rows else 'MISMATCH'}")

    median_col = "Abs-Conc.Median.median"
    if median_col in score_cols:
        src_stats = duck.execute(f'''
            SELECT AVG("{median_col}"), STDDEV("{median_col}"),
                   MIN("{median_col}"), MAX("{median_col}"),
                   COUNT("{median_col}")
            FROM scores_{lang}
        ''').fetchone()
        ch_stats = ch.query(f'''
            SELECT avg(`{median_col}`), stddevSamp(`{median_col}`),
                   min(`{median_col}`), max(`{median_col}`),
                   count(`{median_col}`)
            FROM {CH_DB}.{table}
        ''').result_rows[0]
        print(f"[{lang}] {median_col} src: mean={src_stats[0]:.4f} sd={src_stats[1]:.4f} range=[{src_stats[2]:.4f},{src_stats[3]:.4f}] n={src_stats[4]:,}")
        print(f"[{lang}] {median_col} ch:  mean={ch_stats[0]:.4f} sd={ch_stats[1]:.4f} range=[{ch_stats[2]:.4f},{ch_stats[3]:.4f}] n={ch_stats[4]:,}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["en", "de", "fr", "all"], required=True)
    ap.add_argument("--drop", action="store_true", help="drop target tables first")
    args = ap.parse_args()

    langs = ["en", "de", "fr"] if args.lang == "all" else [args.lang]
    for lang in langs:
        migrate_one(lang, drop=args.drop)
        print()


if __name__ == "__main__":
    main()
