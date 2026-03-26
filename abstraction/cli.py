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
    from .utils import save_df

    corpus_dir = os.path.join(PATH_CORPORA, args.corpus)
    if not os.path.isdir(corpus_dir):
        print(f"Corpus directory not found: {corpus_dir}", file=sys.stderr)
        sys.exit(1)
    df = score_corpus_freqs(corpus_dir)
    if len(df):
        os.makedirs(SCORES_DIR, exist_ok=True)
        out_path = os.path.join(SCORES_DIR, f"{args.corpus}.pkl")
        save_df(df, out_path)
        print(f"Scored {len(df)} texts -> {out_path}")
    else:
        print(f"No freqs files found in {corpus_dir}/freqs/")


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

    args = parser.parse_args()
    if args.command == "score-corpora":
        cmd_score_corpora(args)
    elif args.command == "score-corpus":
        cmd_score_corpus(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
