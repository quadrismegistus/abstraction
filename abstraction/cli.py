"""Command-line interface for the abstraction package."""

import argparse
import sys


def cmd_score_corpus(args):
    import os
    from .config import PATH_CORPORA, SCORES_DIR
    from .scoring import score_corpus_freqs, _version_dir, get_allnorms

    corpus_dir = os.path.join(PATH_CORPORA, args.corpus)
    if not os.path.isdir(corpus_dir):
        print(f"Corpus directory not found: {corpus_dir}", file=sys.stderr)
        sys.exit(1)

    allnorms = get_allnorms()
    out_dir = _version_dir(SCORES_DIR, "v8", args.modernize)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.corpus}.csv")

    if args.force and os.path.exists(out_path):
        os.remove(out_path)

    df = score_corpus_freqs(corpus_dir, allnorms=allnorms, output_path=out_path,
                            modernize=args.modernize)
    if len(df) > 0:
        print(f"Scored {len(df)} texts -> {out_path}")
    else:
        print(f"No freqs files found in {corpus_dir}/freqs/")


def cmd_score_corpora(args):
    from .scoring import score_all_corpora
    if args.all:
        only = "all"
    elif args.corpora:
        only = args.corpora
    else:
        only = None  # defaults to ARC_CORPORA
    score_all_corpora(force=args.force, modernize=args.modernize, only=only,
                      num_proc=args.workers)


def cmd_score_arcs(args):
    from .scoring import score_arc_corpora
    only = args.arcs if args.arcs else None
    score_arc_corpora(force=args.force, modernize=args.modernize, only=only,
                      num_proc=args.workers)


def cmd_score_ids(args):
    """Score an LLTK corpus's texts via DuckDB freqs DB (1:1, no aggregation)."""
    import os, sys, csv, time
    import pandas as pd
    from .config import SCORES_DIR
    from .scoring import score_ids_duckdb

    # Pick allnorms by language
    if args.lang == "fr":
        from .norms_fr import get_allnorms_fr as get_allnorms
    elif args.lang == "de":
        from .norms_de import get_allnorms_de as get_allnorms
    elif args.lang == "es":
        from .norms_es import get_allnorms_es as get_allnorms
    else:
        from .norms import get_allnorms

    # Load corpus metadata via LLTK
    try:
        sys.path.insert(0, os.path.expanduser("~/github/lltk"))
        import lltk
    except ImportError:
        print("LLTK not importable; install or set PYTHONPATH", file=sys.stderr)
        sys.exit(1)

    c = lltk.load(args.corpus)
    if c is None:
        print(f"LLTK corpus '{args.corpus}' not found", file=sys.stderr)
        sys.exit(1)
    meta = c.load_metadata()
    if "_id" not in meta.columns:
        # Derive _id from corpus name + row id (LLTK convention: _corpusname/id)
        corpus_id = c.id if hasattr(c, 'id') else args.corpus
        id_col = meta.index if meta.index.name == "id" else meta.get("id", meta.index)
        meta = meta.copy()
        meta["_id"] = [f"_{corpus_id}/{row_id}" for row_id in id_col]
    ids = meta["_id"].tolist()
    print(f"  {args.corpus}: {len(ids)} texts in metadata")

    # Output path: v8-raw/{corpus}.csv (raw = no spelling modernization)
    out_dir = os.path.join(SCORES_DIR, "v8-raw")
    os.makedirs(out_dir, exist_ok=True)
    out_path = args.output or os.path.join(out_dir, f"{args.corpus}.csv")

    # Resume: drop already-scored ids unless --force
    if args.force and os.path.exists(out_path):
        os.remove(out_path)
    done_ids = set()
    if os.path.exists(out_path):
        try:
            done_ids = set(pd.read_csv(out_path, usecols=["_id"])["_id"])
            print(f"  resume: {len(done_ids)} already scored")
        except Exception as e:
            print(f"  could not resume ({e}); starting fresh")
            os.remove(out_path)

    todo = [i for i in ids if i not in done_ids]
    if not todo:
        print(f"  nothing to do; output at {out_path}")
        return

    print(f"  loading allnorms ({args.lang})...")
    allnorms = get_allnorms(remove_stopwords=True)

    # LLTK has metadb_freqs.duckdb attached on its conn already; reuse it
    # so we don't hit the file-handle conflict.
    con = freqs_table = None
    try:
        if hasattr(lltk, "db") and hasattr(lltk.db, "conn"):
            attached = lltk.db.conn.execute(
                "SELECT database_name FROM duckdb_databases() WHERE path LIKE '%metadb_freqs%'"
            ).fetchall()
            if attached:
                con = lltk.db.conn
                freqs_table = f"{attached[0][0]}.text_freqs"
                print(f"  reusing LLTK conn (freqs attached as {attached[0][0]})")
    except Exception as e:
        print(f"  could not reuse LLTK conn ({e}); opening own")

    print(f"  scoring {len(todo)} texts via DuckDB...")
    t0 = time.time()
    df = score_ids_duckdb(
        todo, allnorms, shard_size=args.shard_size, verbose=True,
        con=con, freqs_table=freqs_table or "fdb.text_freqs",
    )
    elapsed = time.time() - t0
    print(f"  done: {len(df)} scored in {elapsed:.1f}s ({len(df)/elapsed:.0f}/s)")

    # Append (or write) using csv module to handle commas in _id
    mode = "a" if done_ids else "w"
    with open(out_path, mode, newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        if not done_ids:
            writer.writerow(df.columns.tolist())
        for row in df.itertuples(index=False, name=None):
            writer.writerow(row)
    print(f"  wrote {out_path}")


def cmd_score_missing(args):
    """Score every text-with-freqs that isn't yet in scores.duckdb, routing
    per-text by LLTK's `texts.lang`. Idempotent."""
    from .scoring import score_all_missing
    results = score_all_missing(
        lang=args.lang,
        batch_size=args.batch_size,
        limit=args.limit,
        dry_run=args.dry_run,
    )
    print("\nSummary:")
    for lg, info in results.items():
        print(
            f"  scores_{lg}: added {info['added']:,}, "
            f"total {info['total']:,} (candidates: {info['candidates']:,})"
        )


def cmd_check_freqs(args):
    from .corpus import check_freqs_coverage
    df = check_freqs_coverage(corpus_name=args.corpus)
    df = df.sort_values("pct_coverage", ascending=False)
    # format for display
    df["coverage"] = df["pct_coverage"].apply(lambda x: f"{x:.1f}%")
    print(df[["corpus", "n_metadata", "n_freqs", "n_overlap", "coverage"]].to_string(index=False))


def cmd_fix_hathi_englit(args):
    from .corpus import fix_hathi_englit
    genres = tuple(args.genres.split(","))
    fix_hathi_englit(genres=genres)


def cmd_count_corpora(args):
    from .scoring import count_all_corpora
    norm_filter = args.norms.split(",") if args.norms else None
    count_all_corpora(force=args.force, norm_filter=norm_filter,
                      modernize=args.modernize)


def cmd_count_corpus(args):
    import os
    from .config import PATH_CORPORA, COUNT_DIR
    from .scoring import count_corpus_freqs, _version_dir

    corpus_dir = os.path.join(PATH_CORPORA, args.corpus)
    if not os.path.isdir(corpus_dir):
        print(f"Corpus directory not found: {corpus_dir}", file=sys.stderr)
        sys.exit(1)
    out_dir = _version_dir(COUNT_DIR, "v2", args.modernize)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.corpus}.jsonl")
    if args.force and os.path.exists(out_path):
        os.remove(out_path)
    norm_filter = args.norms.split(",") if args.norms else None
    records = count_corpus_freqs(corpus_dir, output_path=out_path,
                                 norm_filter=norm_filter, modernize=args.modernize)
    if records:
        print(f"Counted {len(records)} texts -> {out_path}")
    else:
        print(f"No freqs files found in {corpus_dir}/freqs/")


def cmd_report_arc_counts(args):
    from .analysis import report_arc_counts, load_all_counts
    genres = args.genres.split(",") if args.genres else None
    version = "v2" if args.modernize else "v2-raw"
    combined_df = load_all_counts(version=version, norm=args.norm,
                                  abs_cutoff=args.abs_cutoff,
                                  conc_cutoff=args.conc_cutoff)
    df = report_arc_counts(
        combined_df=combined_df,
        genres=genres,
        norm=args.norm,
        abs_cutoff=args.abs_cutoff,
        conc_cutoff=args.conc_cutoff,
        min_year=args.min_year,
        max_year=args.max_year,
        print_result=True,
    )
    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"\nSaved to {args.csv}")


def cmd_report_full(args):
    if args.compare:
        from .analysis import report_compare
        genres = args.genres.split(",") if args.genres else None
        md, results = report_compare(
            genres=genres,
            abs_cutoff=args.abs_cutoff,
            conc_cutoff=args.conc_cutoff,
            min_year=args.min_year,
            max_year=args.max_year,
        )
        print(md)
        if args.output:
            with open(args.output, "w") as f:
                f.write(md + "\n")
            print(f"\nSaved markdown to {args.output}")
        return

    from .analysis import report_full
    genres = args.genres.split(",") if args.genres else None
    scores_version = "v8" if args.modernize else "v8-raw"
    counts_version = "v2" if args.modernize else "v2-raw"
    md, df = report_full(
        genres=genres,
        abs_cutoff=args.abs_cutoff,
        conc_cutoff=args.conc_cutoff,
        min_year=args.min_year,
        max_year=args.max_year,
        scores_version=scores_version,
        counts_version=counts_version,
    )
    print(md)
    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"\nSaved DataFrame to {args.csv}")
    if args.output:
        with open(args.output, "w") as f:
            f.write(md + "\n")
        print(f"\nSaved markdown to {args.output}")


def cmd_train_model(args):
    import logging
    import time
    from .models import gen_model
    if args.verbose:
        logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
        logging.getLogger('gensim').setLevel(logging.INFO)
    skipgram_path = args.skipgrams
    print(f"Training on: {skipgram_path}")
    print(f"  runs={args.runs}, workers={args.workers}, dims={args.dims}, "
          f"min_count={args.min_count}, window={args.window}")
    if args.num_skips:
        print(f"  num_skips={args.num_skips:,}")
    else:
        print(f"  num_skips=all (no cap)")
    t0 = time.time()
    gen_model(
        skipgram_path,
        num_runs=args.runs,
        num_workers=args.workers,
        min_count=args.min_count,
        num_dimensions=args.dims,
        skipgram_size=args.window,
        num_skips=args.num_skips,
    )
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.0f}s")


def cmd_train_all(args):
    import glob
    import logging
    import os
    import time
    from .models import gen_model
    if args.verbose:
        logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
        logging.getLogger('gensim').setLevel(logging.INFO)
    model_dir = args.model_dir.rstrip("/")
    skipgram_files = sorted(glob.glob(os.path.join(model_dir, "*", "*", "skipgrams.txt.gz")))
    if not skipgram_files:
        print(f"No skipgrams.txt.gz files found under {model_dir}")
        sys.exit(1)
    print(f"Found {len(skipgram_files)} skipgram files under {model_dir}")
    for i, sf in enumerate(skipgram_files, 1):
        rel = sf.replace(model_dir + "/", "").replace("/skipgrams.txt.gz", "")
        print(f"\n[{i}/{len(skipgram_files)}] {rel}")
        t0 = time.time()
        gen_model(
            sf,
            num_runs=args.runs,
            num_workers=args.workers,
            min_count=args.min_count,
            num_dimensions=args.dims,
            skipgram_size=args.window,
            num_skips=args.num_skips,
        )
        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.0f}s")
    print("\nAll done.")


def cmd_train_skipgrams(args):
    from .models import gen_skipgrams_corpus
    output_dir = args.output_dir
    print(f"Generating skipgrams for {args.corpus} (period_len={args.period_len})")
    if output_dir:
        print(f"  output_dir={output_dir}")
    if args.fast:
        print("  fast mode (no sentence tokenization)")
    gen_skipgrams_corpus(
        args.corpus,
        period_len=args.period_len,
        min_year=args.min_year,
        max_year=args.max_year,
        num_proc=args.workers,
        force=args.force,
        output_dir=output_dir,
        fast=args.fast,
        max_skipgrams=args.max_skipgrams,
    )
    print("Done")


def cmd_app(args):
    import os
    import signal
    import subprocess

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if args.refresh:
        # Tells the uvicorn child process (via lifespan -> init_db) to force
        # rebuild abstraction.scores / scores_rep on CH.
        os.environ["ABSTRACTION_REFRESH"] = "1"
        print("--refresh: will rebuild CH scores tables on startup")
    frontend_dir = os.path.join(project_root, "frontend")

    procs = []

    if not args.frontend_only:
        print(f"Starting FastAPI backend on {args.host}:{args.port}...")
        backend = subprocess.Popen(
            ["uvicorn", "abstraction.app:app", "--reload",
             "--host", args.host, "--port", str(args.port)],
            cwd=project_root,
        )
        procs.append(backend)

    if not args.backend_only:
        if not os.path.isdir(frontend_dir):
            print(f"Frontend directory not found: {frontend_dir}")
            sys.exit(1)
        print(f"Starting SvelteKit frontend on {args.host}:{args.frontend_port}...")
        frontend = subprocess.Popen(
            ["npm", "run", "dev", "--", "--host", args.host, "--port", str(args.frontend_port)],
            cwd=frontend_dir,
        )
        procs.append(frontend)

    if not procs:
        print("Nothing to start (both --backend-only and --frontend-only?)")
        sys.exit(1)

    print(f"\nBackend:  http://localhost:{args.port}/docs")
    print(f"Frontend: http://localhost:{args.frontend_port}")
    print("Press Ctrl+C to stop.\n")

    def _shutdown(sig, frame):
        for p in procs:
            p.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # Wait for any process to exit
    try:
        for p in procs:
            p.wait()
    except KeyboardInterrupt:
        _shutdown(None, None)


def cmd_genre_tag_shift(args):
    """Shift-share decomposition of abstractness change by genre tags (Oaxaca-Blinder).

    For each genre tag facet (form/mode/register), computes how much of the change
    in mean abstractness between period A and period B is due to:
      - Composition: genre mix shifting between periods
      - Within: genres themselves becoming more/less abstract
      - Interaction: correlated composition + within change
    """
    import pandas as pd
    import clickhouse_connect
    from .app.routes.decompose import _decompose

    def parse_period(s):
        try:
            start, end = s.split("-")
            return int(start), int(end)
        except Exception:
            print(f"Bad period format {s!r} — expected START-END e.g. 1600-1700", file=sys.stderr)
            sys.exit(1)

    start_a, end_a = parse_period(args.period_a)
    start_b, end_b = parse_period(args.period_b)

    arc = args.arc
    col = args.col
    if args.facet == "all":
        facets = ["form", "mode", "register", "flat"]
    elif args.facet == "flat":
        facets = ["flat"]
    else:
        facets = [args.facet]
    sign = -1.0 if args.invert else 1.0

    client = clickhouse_connect.get_client(
        host="localhost", port=8123, username="lltk", password="lltk",
        database="abstraction",
    )

    # Fetch all tags per (_id, facet) as arrays — groupArray preserves all tags
    # so multi-tagged texts (e.g. [novel, novella]) are exploded into both buckets
    # rather than collapsed to one. facet='unknown' filtered out.
    year_min = min(start_a, start_b)
    year_max = max(end_a, end_b)
    sql = f"""
    SELECT
        s._id                 AS _id,
        any(t.year)           AS year,
        any(s.`{col}`)        AS _score,
        gt.facet,
        groupArray(gt.tag)    AS tags
    FROM abstraction.scores s
    JOIN (SELECT _id, year FROM lltk.texts FINAL) t ON s._id = t._id
    LEFT JOIN lltk.text_genre_tags gt ON s._id = gt._id
    WHERE s.arc_corpus = %(arc)s
      AND s.`{col}` IS NOT NULL
      AND t.year >= %(year_min)s AND t.year < %(year_max)s
      AND (
        (t.year >= %(start_a)s AND t.year < %(end_a)s)
        OR (t.year >= %(start_b)s AND t.year < %(end_b)s)
      )
      AND (gt.facet IS NULL OR gt.facet != 'unknown')
    GROUP BY s._id, gt.facet
    """
    raw = client.query_df(sql, parameters=dict(
        arc=arc, start_a=start_a, end_a=end_a,
        start_b=start_b, end_b=end_b,
        year_min=year_min, year_max=year_max,
    ))
    client.close()

    if raw.empty:
        print(f"No scored texts found for arc={arc} in given periods.", file=sys.stderr)
        sys.exit(1)

    # Base: one row per text
    base = raw[["_id", "year", "_score"]].drop_duplicates("_id").copy()
    base["_score"] = base["_score"] * sign

    # Explode tags: one row per (_id, facet, tag). Multi-tag texts appear once
    # per tag so they contribute to every bucket they belong to.
    tagged = (
        raw[raw["facet"].notna()]
        .explode("tags")
        .rename(columns={"tags": "tag"})
        .dropna(subset=["tag"])
        .drop_duplicates(subset=["_id", "facet", "tag"])
        [["_id", "facet", "tag"]]
    )

    # Per-facet decomposition: merge base with that facet's tags, run _decompose
    early_base = base[(base["year"] >= start_a) & (base["year"] < end_a)]
    late_base  = base[(base["year"] >= start_b) & (base["year"] < end_b)]

    if early_base.empty or late_base.empty:
        print("No data in one or both periods.", file=sys.stderr)
        sys.exit(1)

    print(f"\narc={arc}  col={col}  invert={args.invert}")
    print(f"Period A: {start_a}–{end_a}  (N={len(early_base):,}  mean={early_base['_score'].mean():.4f})")
    print(f"Period B: {start_b}–{end_b}  (N={len(late_base):,}  mean={late_base['_score'].mean():.4f})")
    print(f"Overall change: {late_base['_score'].mean() - early_base['_score'].mean():+.4f}")

    all_results = []
    for facet in facets:
        # Build per-facet df: explode multi-tag texts into multiple rows.
        # "flat" pools all tags across facets (deduplicated by (_id, tag)).
        if facet == "flat":
            facet_tags = tagged[["_id", "tag"]].drop_duplicates().rename(columns={"tag": "_tag"})
        else:
            facet_tags = tagged[tagged["facet"] == facet][["_id", "tag"]].rename(columns={"tag": "_tag"})
        early = early_base.merge(facet_tags, on="_id", how="left")
        late  = late_base.merge(facet_tags, on="_id", how="left")
        early["_tag"] = early["_tag"].fillna("(untagged)")
        late["_tag"]  = late["_tag"].fillna("(untagged)")

        result = _decompose(early, late, "_tag", min_texts=args.min_count)
        if result is None:
            print(f"\n[{facet}] No result (too few texts per tag?)")
            continue

        result.decompose_by = f"genre_tag:{facet}"
        result.period_early = f"{start_a}-{end_a}"
        result.period_late = f"{start_b}-{end_b}"
        all_results.append(result)

        print(f"\n{'─'*90}")
        print(f"  Facet: {facet}   composition={result.total_composition:+.4f}  "
              f"within={result.total_within:+.4f}  interaction={result.total_interaction:+.4f}")
        print(f"{'─'*90}")
        hdr = (f"  {'tag':<28} {'n_a':>6} {'n_b':>6} {'mean_a':>8} {'mean_b':>8}"
               f" {'comp':>8} {'within':>8} {'inter':>8} {'total':>8}")
        print(hdr)
        print(f"  {'-'*86}")
        for row in result.rows:
            print(f"  {row.category:<28} {row.n_early:>6} {row.n_late:>6}"
                  f" {row.mean_early:>8.4f} {row.mean_late:>8.4f}"
                  f" {row.composition_effect:>8.4f} {row.within_effect:>8.4f}"
                  f" {row.interaction:>8.4f} {row.total_effect:>8.4f}")

    if args.csv and all_results:
        rows_data = []
        for res in all_results:
            facet_label = res.decompose_by.split(":")[1] if ":" in res.decompose_by else res.decompose_by
            for row in res.rows:
                rows_data.append({
                    "facet": facet_label,
                    "tag": row.category,
                    "n_a": row.n_early, "n_b": row.n_late,
                    "mean_a": row.mean_early, "mean_b": row.mean_late,
                    "share_a": row.share_early, "share_b": row.share_late,
                    "composition_effect": row.composition_effect,
                    "within_effect": row.within_effect,
                    "interaction": row.interaction,
                    "total_effect": row.total_effect,
                })
        pd.DataFrame(rows_data).to_csv(args.csv, index=False)
        print(f"\nSaved to {args.csv}")


def cmd_score_passages(args):
    """Score every passage in lltk.passages → abstraction.passage_scores.

    Stores all norm columns as Map(String, Float32) so downstream queries can
    extract any score: scores['Abs-Conc.Median.median']. Tokenization uses
    tokenize_agnostic (regex word-boundary splitter, not str.split).
    """
    import time
    import numpy as np
    import clickhouse_connect
    from .aggregate import CH_HOST, CH_PORT, CH_USER, CH_PASSWORD
    from .scoring import score_text_allcols, build_allnorms_index

    lang = args.lang

    print(f"  loading allnorms ({lang})...")
    if lang == "fr":
        from .norms_fr import get_allnorms_fr
        allnorms = get_allnorms_fr(remove_stopwords=True)
    elif lang == "de":
        from .norms_de import get_allnorms_de
        allnorms = get_allnorms_de(remove_stopwords=True)
    else:
        from .norms import get_allnorms
        allnorms = get_allnorms(remove_stopwords=True)
    print(f"  allnorms: {len(allnorms):,} words × {len(allnorms.columns)} columns")
    norm_index = build_allnorms_index(allnorms)

    client = clickhouse_connect.get_client(
        host=CH_HOST, port=CH_PORT, username=CH_USER, password=CH_PASSWORD,
    )

    client.command("""
        CREATE TABLE IF NOT EXISTS abstraction.passage_scores (
            _id     String,
            scheme  String,
            seq     UInt32,
            lang    String,
            scores  Map(String, Float32)
        ) ENGINE = MergeTree()
        ORDER BY (_id, scheme, seq)
    """)

    count_sql = """
        SELECT count()
        FROM lltk.passages p
        LEFT JOIN lltk.text_langs tl ON p._id = tl._id
        WHERE coalesce(tl.lang_detected, p.lang) = {lang:String}
    """
    total = client.query(count_sql, parameters={"lang": lang}).result_rows[0][0]
    print(f"  lltk.passages: {total:,} {lang} passages")

    if args.force:
        client.command(
            "ALTER TABLE abstraction.passage_scores DELETE WHERE lang = {lang:String}",
            parameters={"lang": lang},
        )
        done = set()
        print("  --force: cleared existing scores")
    else:
        done_rows = client.query(
            "SELECT _id, scheme, seq FROM abstraction.passage_scores WHERE lang = {lang:String}",
            parameters={"lang": lang},
        ).result_rows
        done = {(r[0], r[1], r[2]) for r in done_rows}
        if done:
            print(f"  resume: {len(done):,} already scored")

    from tqdm import tqdm

    batch_size = args.batch_size
    offset = 0

    fetch_sql = """
        SELECT p._id, p.scheme, p.seq, p.text
        FROM lltk.passages p
        LEFT JOIN lltk.text_langs tl ON p._id = tl._id
        WHERE coalesce(tl.lang_detected, p.lang) = {lang:String}
        ORDER BY p._id, p.scheme, p.seq
        LIMIT {limit:UInt32} OFFSET {offset:UInt32}
    """

    with tqdm(total=total - len(done), unit="psg", desc=f"score-passages ({lang})") as pbar:
        while True:
            rows = client.query(
                fetch_sql,
                parameters={"lang": lang, "limit": batch_size, "offset": offset},
            ).result_rows

            if not rows:
                break

            insert_rows = []
            for _id, scheme, seq, text in rows:
                if (_id, scheme, seq) in done:
                    continue
                scores = score_text_allcols(text, allnorms, index=norm_index)
                if scores:
                    insert_rows.append((_id, scheme, seq, lang, scores))

            if insert_rows:
                client.insert(
                    "abstraction.passage_scores",
                    insert_rows,
                    column_names=["_id", "scheme", "seq", "lang", "scores"],
                )
                pbar.update(len(insert_rows))

            offset += batch_size

    print(f"  done")
    client.close()


def cmd_estimate_corpus_bias(args):
    from .corpus_correction import estimate_corpus_bias, save_corpus_bias
    result = estimate_corpus_bias(
        score_col=args.score_col,
        reference_corpus=args.reference,
        min_group_overlap=args.min_overlap,
    )
    if result:
        save_corpus_bias(result)


def cmd_gen_vecnorms(args):
    import time
    lang = getattr(args, 'lang', 'en') or 'en'
    print(f"Generating vector norms lang={lang} (period_len={args.period_len})")
    t0 = time.time()
    if lang == 'fr':
        from .norms_fr import gen_vecnorms_fr
        gen_vecnorms_fr(model_dir=getattr(args, 'model_dir', None),
                        bin_year_by=args.period_len, num_proc=args.workers)
    elif lang == 'de':
        from .norms_de import gen_vecnorms_de
        gen_vecnorms_de(model_dir=getattr(args, 'model_dir', None),
                        bin_year_by=args.period_len, num_proc=args.workers)
    elif lang == 'es':
        from .norms_es import gen_vecnorms_es
        gen_vecnorms_es(model_dir=getattr(args, 'model_dir', None),
                        bin_year_by=args.period_len, num_proc=args.workers)
    else:
        from .models import gen_vecnorms
        gen_vecnorms(bin_year_by=args.period_len, num_proc=args.workers,
                     model_dir=getattr(args, 'model_dir', None))
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.0f}s")


def cmd_report_arc(args):
    from .analysis import report_arc, load_all_scored
    genres = args.genres.split(",") if args.genres else None
    version = "v8" if args.modernize else "v8-raw"
    combined_df = load_all_scored(version=version)
    df = report_arc(
        combined_df=combined_df,
        genres=genres,
        min_year=args.min_year,
        max_year=args.max_year,
        print_result=True,
    )
    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"\nSaved to {args.csv}")


def main():
    parser = argparse.ArgumentParser(prog="abstraction", description="Abstraction CLI")
    sub = parser.add_subparsers(dest="command")

    # score-corpora: score all corpora with freqs/ folders
    p = sub.add_parser("score-corpora", help="Score corpora (default: arc corpora only)")
    p.add_argument("corpora", nargs="*", help="Specific corpora to score (e.g. canon_fiction ecco)")
    p.add_argument("--all", action="store_true", help="Score ALL LLTK corpora with freqs (default: arc corpora only)")
    p.add_argument("--force", action="store_true", help="Re-score even if output exists")
    p.add_argument("--workers", "-j", type=int, default=1, help="Parallel worker processes (default: 1)")
    p.add_argument("--modernize", action="store_true", help="Enable spelling modernization (output to v8/ instead of v8-raw/)")

    # score-arcs: score synthetic arc corpora (arc_fiction, arc_poetry, etc.)
    p = sub.add_parser("score-arcs", help="Score synthetic arc corpora (deduplicated by genre)")
    p.add_argument("arcs", nargs="*", help="Specific arc corpora (e.g. arc_fiction arc_poetry)")
    p.add_argument("--force", action="store_true", help="Re-score even if output exists")
    p.add_argument("--workers", "-j", type=int, default=1, help="Parallel worker processes (default: 1)")
    p.add_argument("--modernize", action="store_true", help="Enable spelling modernization")

    # score-corpus: score a single corpus
    p = sub.add_parser("score-corpus", help="Score a single corpus by directory name")
    p.add_argument("corpus", help="Corpus directory name (e.g. canon_fiction)")
    p.add_argument("--force", action="store_true", help="Re-score even if output exists")
    p.add_argument("--modernize", action="store_true", help="Enable spelling modernization")

    # score-ids: 1:1 DuckDB-backed scorer for any LLTK corpus, language-aware
    p = sub.add_parser(
        "score-ids",
        help="Score LLTK corpus texts via DuckDB freqs DB (1:1, no match-group averaging)",
    )
    p.add_argument("corpus", help="LLTK corpus name (e.g. arc_fiction_fr, gallica_literary_fictions)")
    p.add_argument("--lang", choices=["en", "fr", "de", "es"], default="en",
                   help="Language: chooses get_allnorms vs get_allnorms_fr/de/es (default: en)")
    p.add_argument("--force", action="store_true", help="Re-score even if output exists")
    p.add_argument("--output", "-o", default=None, help="Output CSV path (default: data/scores/v8-raw/{corpus}.csv)")
    p.add_argument("--shard-size", type=int, default=20000, help="Texts per DuckDB query (default: 20000)")

    # score-missing: unified "score everything not yet in scores.duckdb"
    p = sub.add_parser(
        "score-missing",
        help="Score all texts with freqs that aren't in scores.duckdb yet, routed per-text by LLTK texts.lang",
    )
    p.add_argument("--lang", choices=["all", "en", "fr", "de"], default="all",
                   help="Language to score (default: all)")
    p.add_argument("--batch-size", type=int, default=10000,
                   help="Texts per scoring batch (default: 10000)")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap total texts scored per lang (for smoke tests)")
    p.add_argument("--dry-run", action="store_true",
                   help="Report what would be scored without running")

    # check-freqs: check metadata-to-freqs coverage
    p = sub.add_parser("check-freqs", help="Check freqs coverage for corpora")
    p.add_argument("corpus", nargs="?", default=None, help="Corpus name (default: all)")

    # fix-hathi-englit: unpack TSV archives into freqs JSONs
    p = sub.add_parser("fix-hathi-englit", help="Unpack hathi_englit TSV archives into freqs JSONs")
    p.add_argument("--genres", default="fiction,poetry", help="Comma-separated genres (default: fiction,poetry)")

    # count-corpora: count z-score distributions for all corpora
    p = sub.add_parser("count-corpora", help="Count z-score distributions for all corpora with freqs/")
    p.add_argument("--force", action="store_true", help="Re-count even if output exists")
    p.add_argument("--norms", default=None, help="Comma-separated norm columns to count (default: all)")
    p.add_argument("--modernize", action="store_true", help="Enable spelling modernization (output to v2/ instead of v2-raw/)")

    # count-corpus: count z-score distributions for one corpus
    p = sub.add_parser("count-corpus", help="Count z-score distributions for a single corpus")
    p.add_argument("corpus", help="Corpus directory name (e.g. canon_fiction)")
    p.add_argument("--force", action="store_true", help="Re-count even if output exists")
    p.add_argument("--norms", default=None, help="Comma-separated norm columns to count (default: all)")
    p.add_argument("--modernize", action="store_true", help="Enable spelling modernization (output to v2/ instead of v2-raw/)")

    # report-full: combined score + count report
    p = sub.add_parser("report-full", help="Combined report: scores + word proportions + prose")
    p.add_argument("--genres", default=None, help="Comma-separated genres (default: Fiction,Poetry,Periodical)")
    p.add_argument("--abs-cutoff", type=float, default=-1.0, help="Z-score cutoff for abstract words (default: -1.0)")
    p.add_argument("--conc-cutoff", type=float, default=1.0, help="Z-score cutoff for concrete words (default: 1.0)")
    p.add_argument("--min-year", type=int, default=1600)
    p.add_argument("--max-year", type=int, default=2020)
    p.add_argument("--csv", default=None, help="Save merged DataFrame to CSV")
    p.add_argument("--output", "-o", default=None, help="Save markdown to file")
    p.add_argument("--modernize", action="store_true", help="Use spelling-modernized data")
    p.add_argument("--compare", action="store_true", help="Compare raw vs modernized side by side (loads all 4 datasets)")

    # train-model: train Word2Vec from a skipgrams file
    p = sub.add_parser("train-model", help="Train Word2Vec model from a skipgrams file")
    p.add_argument("skipgrams", help="Path to skipgrams.txt.gz file")
    p.add_argument("--runs", type=int, default=1, help="Number of training runs (default: 1)")
    p.add_argument("--workers", type=int, default=8, help="Number of threads (default: 8)")
    p.add_argument("--dims", type=int, default=100, help="Embedding dimensions (default: 100)")
    p.add_argument("--min-count", type=int, default=10, help="Min word frequency (default: 10)")
    p.add_argument("--window", type=int, default=10, help="Context window size (default: 10)")
    p.add_argument("--num-skips", type=int, default=None, help="Max skipgrams to sample (default: all)")
    p.add_argument("--verbose", "-v", action="store_true", help="Show gensim training progress")

    # train-all: train models for all skipgram files under a directory
    p = sub.add_parser("train-all", help="Train Word2Vec models for all skipgram files under a directory")
    p.add_argument("model_dir", help="Root model directory (e.g. /Volumes/diderot/DH/data/models_century5)")
    p.add_argument("--runs", type=int, default=5, help="Number of training runs (default: 5)")
    p.add_argument("--workers", type=int, default=8, help="Number of threads (default: 8)")
    p.add_argument("--dims", type=int, default=100, help="Embedding dimensions (default: 100)")
    p.add_argument("--min-count", type=int, default=10, help="Min word frequency (default: 10)")
    p.add_argument("--window", type=int, default=10, help="Context window size (default: 10)")
    p.add_argument("--num-skips", type=int, default=None, help="Max skipgrams to sample (default: all)")
    p.add_argument("--verbose", "-v", action="store_true", help="Show gensim training progress")

    # train-skipgrams: generate skipgram files from a corpus
    p = sub.add_parser("train-skipgrams", help="Generate skipgram files from a corpus by period")
    p.add_argument("corpus", help="Corpus name (e.g. CanonFiction)")
    p.add_argument("--period-len", type=int, default=100, help="Period length in years (default: 100)")
    p.add_argument("--min-year", type=int, default=None)
    p.add_argument("--max-year", type=int, default=None)
    p.add_argument("--workers", type=int, default=1, help="Parallel processes (default: 1)")
    p.add_argument("--force", action="store_true", help="Regenerate even if files exist")
    p.add_argument("--output-dir", default=None, help="Output directory (default: data/models/)")
    p.add_argument("--fast", action="store_true", help="Skip sentence tokenization (fixed-size chunks, 10-50x faster)")
    p.add_argument("--max-skipgrams", type=int, default=None, metavar="N",
                   help="Cap skipgrams per period at N (texts shuffled, writing stops at cap)")

    # gen-vecnorms: generate vector-based word norms from trained models
    p = sub.add_parser("gen-vecnorms", help="Generate vector norms from trained models")
    p.add_argument("--lang", default="en", choices=["en", "fr", "de", "es"], help="Language (default: en)")
    p.add_argument("--period-len", type=int, default=100, help="Period length for binning (default: 100)")
    p.add_argument("--model-dir", default=None, help="Model directory (default: data/models/)")
    p.add_argument("--workers", type=int, default=1, help="Parallel processes (default: 1)")

    # report-arc: piecewise arc report with ratios (score-based)
    p = sub.add_parser("report-arc", help="Report piecewise arc statistics per genre")
    p.add_argument("--genres", default=None, help="Comma-separated genres (default: Fiction,Poetry,Periodical)")
    p.add_argument("--min-year", type=int, default=1600)
    p.add_argument("--max-year", type=int, default=2020)
    p.add_argument("--csv", default=None, help="Save results to CSV")
    p.add_argument("--modernize", action="store_true", help="Use spelling-modernized scores (v8 instead of v8-raw)")

    # report-arc-counts: piecewise arc report with count-based proportions
    p = sub.add_parser("report-arc-counts", help="Report arc statistics using word proportions")
    p.add_argument("--genres", default=None, help="Comma-separated genres (default: Fiction,Poetry,Periodical)")
    p.add_argument("--abs-cutoff", type=float, default=-1.0, help="Z-score cutoff for abstract words (z ≤ cutoff, default: -1.0)")
    p.add_argument("--conc-cutoff", type=float, default=1.0, help="Z-score cutoff for concrete words (z > cutoff, default: 1.0)")
    p.add_argument("--norm", default="Abs-Conc.Median.median", help="Norm column")
    p.add_argument("--min-year", type=int, default=1600)
    p.add_argument("--max-year", type=int, default=2020)
    p.add_argument("--csv", default=None, help="Save results to CSV")
    p.add_argument("--modernize", action="store_true", help="Use spelling-modernized counts (v2 instead of v2-raw)")

    # genre-tag-shift: shift-share decomposition of abstractness by genre tags
    p = sub.add_parser(
        "genre-tag-shift",
        help="Shift-share decomposition of abstractness change by genre tags (Oaxaca-Blinder)",
    )
    p.add_argument("--period-a", required=True, metavar="START-END",
                   help="Earlier period, exclusive end e.g. 1640-1680")
    p.add_argument("--period-b", required=True, metavar="START-END",
                   help="Later period, exclusive end e.g. 1740-1780")
    p.add_argument("--arc", default="arc_fiction",
                   help="Arc corpus name (default: arc_fiction)")
    p.add_argument("--col", default="Abs-Conc.Median.median",
                   help="Score column (default: Abs-Conc.Median.median)")
    p.add_argument("--facet", default="form", choices=["form", "mode", "register", "flat", "all"],
                   help="Genre tag facet (default: form); 'flat' pools all tags across facets; 'all' runs all four")
    p.add_argument("--invert", action="store_true", default=True,
                   help="Negate scores so positive = more abstract (default: True)")
    p.add_argument("--no-invert", dest="invert", action="store_false")
    p.add_argument("--min-count", type=int, default=5,
                   help="Min texts per tag to include (default: 5)")
    p.add_argument("--csv", default=None, help="Save results to CSV")

    # score-passages: score every passage in lltk.passages → abstraction.passage_scores
    p = sub.add_parser(
        "score-passages",
        help="Score all passages in lltk.passages and write to abstraction.passage_scores",
    )
    p.add_argument("--lang", choices=["en", "fr", "de"], default="en",
                   help="Language to score (default: en; uses lang_detected where available)")
    p.add_argument("--batch-size", type=int, default=5000,
                   help="Passages per CH fetch batch (default: 5000)")
    p.add_argument("--force", action="store_true",
                   help="Delete and re-score existing rows for this lang")

    # estimate-corpus-bias
    p = sub.add_parser("estimate-corpus-bias", help="Estimate corpus bias coefficients from match group comparisons")
    p.add_argument("--score-col", default="Abs-Conc.Median.median", help="Score column to use")
    p.add_argument("--reference", default="ecco_tcp", help="Reference corpus (bias=0)")
    p.add_argument("--min-overlap", type=int, default=10, help="Min match groups per corpus")

    # app: start web app servers
    p = sub.add_parser("app", help="Start the web app (FastAPI backend + SvelteKit frontend)")
    p.add_argument("--backend-only", action="store_true", help="Start only the FastAPI backend")
    p.add_argument("--frontend-only", action="store_true", help="Start only the SvelteKit frontend")
    p.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1, use 0.0.0.0 for network access)")
    p.add_argument("--port", type=int, default=1709, help="Backend port (default: 1709)")
    p.add_argument("--frontend-port", type=int, default=1784, help="Frontend port (default: 1784)")
    p.add_argument("--refresh", action="store_true", help="Force rebuild of abstraction.scores / scores_rep on CH at startup")

    args = parser.parse_args()
    if args.command == "app":
        cmd_app(args)
    elif args.command == "report-full":
        cmd_report_full(args)
    elif args.command == "score-corpus":
        cmd_score_corpus(args)
    elif args.command == "score-corpora":
        cmd_score_corpora(args)
    elif args.command == "score-arcs":
        cmd_score_arcs(args)
    elif args.command == "score-ids":
        cmd_score_ids(args)
    elif args.command == "score-missing":
        cmd_score_missing(args)
    elif args.command == "check-freqs":
        cmd_check_freqs(args)
    elif args.command == "fix-hathi-englit":
        cmd_fix_hathi_englit(args)
    elif args.command == "count-corpora":
        cmd_count_corpora(args)
    elif args.command == "count-corpus":
        cmd_count_corpus(args)
    elif args.command == "report-arc":
        cmd_report_arc(args)
    elif args.command == "report-arc-counts":
        cmd_report_arc_counts(args)
    elif args.command == "train-model":
        cmd_train_model(args)
    elif args.command == "train-all":
        cmd_train_all(args)
    elif args.command == "train-skipgrams":
        cmd_train_skipgrams(args)
    elif args.command == "gen-vecnorms":
        cmd_gen_vecnorms(args)
    elif args.command == "genre-tag-shift":
        cmd_genre_tag_shift(args)
    elif args.command == "score-passages":
        cmd_score_passages(args)
    elif args.command == "estimate-corpus-bias":
        cmd_estimate_corpus_bias(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
