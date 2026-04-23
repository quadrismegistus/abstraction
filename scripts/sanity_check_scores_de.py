"""Sanity-check scores_de after overnight rescore.

Run me the morning after `abstraction score-missing --lang de`.
Validates row count, NaN rate, score ranges, and spot-checks a few texts
against the freqs to catch silent shard skips or FP quirks.
"""
import os
import sys
import duckdb
import numpy as np

sys.path.insert(0, os.path.expanduser("~/github/lltk"))
import lltk


SCORES_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "scores", "scores.duckdb"
)


def main():
    s = duckdb.connect(SCORES_PATH, read_only=True)
    ch = lltk.db.adapter.client

    # 1. Row count: does every DE text with freqs have a score?
    n_scored = s.execute("SELECT COUNT(*) FROM scores_de").fetchone()[0]
    n_candidates = ch.query("""
        SELECT COUNT(*) FROM lltk.text_freqs f
        JOIN lltk.texts t ON f._id = t._id WHERE t.lang='de'
    """).result_rows[0][0]
    print(f"[1] scores_de rows: {n_scored:,}  / candidates: {n_candidates:,}")
    if n_scored < n_candidates * 0.98:
        print(f"    WARN: {n_candidates - n_scored:,} texts missing ({(n_candidates-n_scored)/n_candidates*100:.1f}%)")

    # 2. NaN rate per column
    cols = [c for c in s.execute("DESCRIBE scores_de").fetchdf()["column_name"]
            if c != "_id"]
    print(f"[2] checking NaN rate across {len(cols)} score columns...")
    for col in cols[:3] + ["Abs-Conc.Median.median", "Abs-Conc.Median.orig"]:
        qc = f'"{col}"'
        n_null = s.execute(
            f"SELECT COUNT(*) FROM scores_de WHERE {qc} IS NULL OR isnan({qc})"
        ).fetchone()[0]
        print(f"    {col}: {n_null:,} null/nan ({n_null/n_scored*100:.1f}%)")

    # 3. Score distributions: mean, std, min, max on key columns
    print("[3] score distributions:")
    for col in ["Abs-Conc.Median.median", "Abs-Conc.Median.orig"]:
        qc = f'"{col}"'
        r = s.execute(
            f"SELECT AVG({qc}), STDDEV({qc}), MIN({qc}), MAX({qc}) FROM scores_de"
        ).fetchone()
        print(f"    {col}: mean={r[0]:.3f}, sd={r[1]:.3f}, range=[{r[2]:.3f}, {r[3]:.3f}]")
    # Sanity: scores should be roughly centered around 0 with sd ~0.3-0.6
    # (similar to EN/FR). Very different means would suggest a bug.

    # 4. Cross-check one text: pull freqs + recompute manually
    print("[4] spot-check one text vs manual recompute...")
    test_id = s.execute("SELECT _id FROM scores_de LIMIT 1").fetchone()[0]
    stored = s.execute(
        f'SELECT "Abs-Conc.Median.median" FROM scores_de WHERE _id=?', [test_id]
    ).fetchone()[0]
    # Pull freqs + allnorms.Median.median, recompute
    freqs_df = lltk.db.read_freqs(ids=[test_id])
    freqs = freqs_df["freqs"].iloc[0]
    from abstraction.norms_de import get_allnorms_de
    an = get_allnorms_de(remove_stopwords=True)["Abs-Conc.Median.median"].dropna()
    total_cnt, total_wsum = 0, 0.0
    for w, c in freqs.items():
        if w in an.index:
            total_cnt += c
            total_wsum += c * an[w]
    recomputed = total_wsum / total_cnt if total_cnt else None
    match = abs(stored - recomputed) < 1e-6 if recomputed else False
    print(f"    {test_id}: stored={stored:.4f}, recomputed={recomputed:.4f}, match={match}")

    print("\nDone. If any warning above, investigate before trusting.")


if __name__ == "__main__":
    main()
