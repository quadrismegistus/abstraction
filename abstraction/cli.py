"""Command-line interface for the abstraction package."""

import argparse
import sys


def cmd_score_corpora(args):
    from .scoring import score_all_corpora
    only = "all" if args.all else None
    score_all_corpora(force=args.force, modernize=args.modernize, only=only)


def cmd_score_corpus(args):
    import os
    from .config import PATH_CORPORA, SCORES_DIR
    from .scoring import score_corpus_freqs, _version_dir

    corpus_dir = os.path.join(PATH_CORPORA, args.corpus)
    if not os.path.isdir(corpus_dir):
        print(f"Corpus directory not found: {corpus_dir}", file=sys.stderr)
        sys.exit(1)
    out_dir = _version_dir(SCORES_DIR, "v8", args.modernize)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.corpus}.csv")
    if args.force and os.path.exists(out_path):
        os.remove(out_path)
    df = score_corpus_freqs(corpus_dir, output_path=out_path, modernize=args.modernize)
    if len(df):
        print(f"Scored {len(df)} texts -> {out_path}")
    else:
        print(f"No freqs files found in {corpus_dir}/freqs/")


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
    gen_skipgrams_corpus(
        args.corpus,
        period_len=args.period_len,
        min_year=args.min_year,
        max_year=args.max_year,
        num_proc=args.workers,
        force=args.force,
        output_dir=output_dir,
    )
    print("Done")


def cmd_app(args):
    import os
    import signal
    import subprocess

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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


def cmd_gen_vecnorms(args):
    import time
    from .models import gen_vecnorms
    print(f"Generating vector norms (period_len={args.period_len})")
    t0 = time.time()
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
    p.add_argument("--all", action="store_true", help="Score ALL corpora with freqs/ folders (default: arc corpora only)")
    p.add_argument("--force", action="store_true", help="Re-score even if output exists")
    p.add_argument("--modernize", action="store_true", help="Enable spelling modernization (output to v8/ instead of v8-raw/)")

    # score-corpus: score a single corpus
    p = sub.add_parser("score-corpus", help="Score a single corpus")
    p.add_argument("corpus", help="Corpus directory name (e.g. canon_fiction)")
    p.add_argument("--force", action="store_true", help="Re-score even if output exists")
    p.add_argument("--modernize", action="store_true", help="Enable spelling modernization (output to v8/ instead of v8-raw/)")

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

    # gen-vecnorms: generate vector-based word norms from trained models
    p = sub.add_parser("gen-vecnorms", help="Generate vector norms from trained models")
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

    # app: start web app servers
    p = sub.add_parser("app", help="Start the web app (FastAPI backend + SvelteKit frontend)")
    p.add_argument("--backend-only", action="store_true", help="Start only the FastAPI backend")
    p.add_argument("--frontend-only", action="store_true", help="Start only the SvelteKit frontend")
    p.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1, use 0.0.0.0 for network access)")
    p.add_argument("--port", type=int, default=1709, help="Backend port (default: 1709)")
    p.add_argument("--frontend-port", type=int, default=1784, help="Frontend port (default: 1784)")

    args = parser.parse_args()
    if args.command == "app":
        cmd_app(args)
    elif args.command == "report-full":
        cmd_report_full(args)
    elif args.command == "score-corpora":
        cmd_score_corpora(args)
    elif args.command == "score-corpus":
        cmd_score_corpus(args)
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
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
