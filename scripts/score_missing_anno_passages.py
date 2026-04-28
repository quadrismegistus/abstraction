"""Score p500 passages for texts that have annotations but no passage_scores rows.

Targeted version of `abstraction score-passages` — only processes texts that
appear in lltk.passage_annotations but are missing from abstraction.passage_scores.
"""
import sys
import time

import clickhouse_connect
from tqdm import tqdm

sys.path.insert(0, "abstraction")
from abstraction.aggregate import CH_HOST, CH_PORT, CH_USER, CH_PASSWORD
from abstraction.norms import get_allnorms
from abstraction.scoring import score_text_allcols, build_allnorms_index


def main(lang: str = "en", batch_size: int = 200):
    print(f"loading allnorms ({lang})...")
    allnorms = get_allnorms(remove_stopwords=True)
    print(f"  {len(allnorms):,} words × {len(allnorms.columns)} cols")
    norm_index = build_allnorms_index(allnorms)

    client = clickhouse_connect.get_client(
        host=CH_HOST, port=CH_PORT, username=CH_USER, password=CH_PASSWORD,
    )

    target_ids_sql = """
        SELECT DISTINCT _id FROM lltk.passage_annotations
        WHERE _id NOT IN (
            SELECT DISTINCT _id FROM abstraction.passage_scores
            WHERE scheme = 'p500' AND lang = {lang:String}
        )
    """
    target_ids = [r[0] for r in client.query(target_ids_sql, parameters={"lang": lang}).result_rows]
    print(f"target texts: {len(target_ids)}")

    count_sql = """
        SELECT count() FROM lltk.passages
        WHERE scheme = 'p500' AND _id IN {ids:Array(String)}
    """
    total = client.query(count_sql, parameters={"ids": target_ids}).result_rows[0][0]
    print(f"target passages: {total:,}")

    fetch_sql = """
        SELECT p._id, p.scheme, p.seq, p.text
        FROM lltk.passages p
        WHERE p.scheme = 'p500' AND p._id IN {ids:Array(String)}
        ORDER BY p._id, p.seq
        LIMIT {limit:UInt32} OFFSET {offset:UInt32}
    """

    offset = 0
    written = 0
    t0 = time.time()
    with tqdm(total=total, unit="psg") as pbar:
        while True:
            rows = client.query(
                fetch_sql,
                parameters={"ids": target_ids, "limit": batch_size, "offset": offset},
            ).result_rows
            if not rows:
                break

            insert_rows = []
            for _id, scheme, seq, text in rows:
                scores = score_text_allcols(text, allnorms, index=norm_index)
                if scores:
                    insert_rows.append((_id, scheme, seq, lang, scores))

            if insert_rows:
                client.insert(
                    "abstraction.passage_scores",
                    insert_rows,
                    column_names=["_id", "scheme", "seq", "lang", "scores"],
                )
                written += len(insert_rows)
            pbar.update(len(rows))
            offset += batch_size

    elapsed = time.time() - t0
    print(f"\ndone: wrote {written:,} score rows in {elapsed:.0f}s ({written/elapsed:.0f} psg/s)")
    client.close()


if __name__ == "__main__":
    main()
