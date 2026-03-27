"""Command-line interface for the abstraction package."""

import argparse
import sys


def cmd_score_corpora(args):
    from .scoring import score_all_corpora
    score_all_corpora(force=args.force)


def cmd_score_corpus(args):
    import os
    from .config import PATH_CORPORA, SCORES_DIR
    from .scoring import score_corpus_freqs

    corpus_dir = os.path.join(PATH_CORPORA, args.corpus)
    if not os.path.isdir(corpus_dir):
        print(f"Corpus directory not found: {corpus_dir}", file=sys.stderr)
        sys.exit(1)
    out_dir = os.path.join(SCORES_DIR, "v8")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.corpus}.csv")
    if args.force and os.path.exists(out_path):
        os.remove(out_path)
    df = score_corpus_freqs(corpus_dir, output_path=out_path)
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
    count_all_corpora(force=args.force, norm_filter=norm_filter)


def cmd_count_corpus(args):
    import os
    from .config import PATH_CORPORA, COUNT_DIR
    from .scoring import count_corpus_freqs

    corpus_dir = os.path.join(PATH_CORPORA, args.corpus)
    if not os.path.isdir(corpus_dir):
        print(f"Corpus directory not found: {corpus_dir}", file=sys.stderr)
        sys.exit(1)
    out_dir = os.path.join(COUNT_DIR, "v2")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.corpus}.jsonl")
    if args.force and os.path.exists(out_path):
        os.remove(out_path)
    norm_filter = args.norms.split(",") if args.norms else None
    records = count_corpus_freqs(corpus_dir, output_path=out_path, norm_filter=norm_filter)
    if records:
        print(f"Counted {len(records)} texts -> {out_path}")
    else:
        print(f"No freqs files found in {corpus_dir}/freqs/")


def cmd_report_arc_counts(args):
    from .analysis import report_arc_counts
    genres = args.genres.split(",") if args.genres else None
    df = report_arc_counts(
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


def cmd_report_arc(args):
    from .analysis import report_arc
    genres = args.genres.split(",") if args.genres else None
    df = report_arc(
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
    p = sub.add_parser("score-corpora", help="Score all corpora with freqs/ folders")
    p.add_argument("--force", action="store_true", help="Re-score even if output exists")

    # score-corpus: score a single corpus
    p = sub.add_parser("score-corpus", help="Score a single corpus")
    p.add_argument("corpus", help="Corpus directory name (e.g. canon_fiction)")
    p.add_argument("--force", action="store_true", help="Re-score even if output exists")

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

    # count-corpus: count z-score distributions for one corpus
    p = sub.add_parser("count-corpus", help="Count z-score distributions for a single corpus")
    p.add_argument("corpus", help="Corpus directory name (e.g. canon_fiction)")
    p.add_argument("--force", action="store_true", help="Re-count even if output exists")
    p.add_argument("--norms", default=None, help="Comma-separated norm columns to count (default: all)")

    # report-arc: piecewise arc report with ratios (score-based)
    p = sub.add_parser("report-arc", help="Report piecewise arc statistics per genre")
    p.add_argument("--genres", default=None, help="Comma-separated genres (default: Fiction,Poetry,Periodical)")
    p.add_argument("--min-year", type=int, default=1600)
    p.add_argument("--max-year", type=int, default=2020)
    p.add_argument("--csv", default=None, help="Save results to CSV")

    # report-arc-counts: piecewise arc report with count-based proportions
    p = sub.add_parser("report-arc-counts", help="Report arc statistics using word proportions")
    p.add_argument("--genres", default=None, help="Comma-separated genres (default: Fiction,Poetry,Periodical)")
    p.add_argument("--abs-cutoff", type=float, default=-1.0, help="Z-score cutoff for abstract words (z ≤ cutoff, default: -1.0)")
    p.add_argument("--conc-cutoff", type=float, default=1.0, help="Z-score cutoff for concrete words (z > cutoff, default: 1.0)")
    p.add_argument("--norm", default="Abs-Conc.Median.median", help="Norm column")
    p.add_argument("--min-year", type=int, default=1600)
    p.add_argument("--max-year", type=int, default=2020)
    p.add_argument("--csv", default=None, help="Save results to CSV")

    args = parser.parse_args()
    if args.command == "score-corpora":
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
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
