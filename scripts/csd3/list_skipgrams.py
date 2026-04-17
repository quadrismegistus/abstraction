#!/usr/bin/env python3
"""Enumerate skipgrams files under a model directory that still need training runs.

Prints one path per line (stdout), ordered. The line number (1-based) is the
SLURM array task index. Use with `scripts/csd3/train_w2v.sh`.

Usage:
    python list_skipgrams.py data/models --runs 5
    python list_skipgrams.py data/models_fr --runs 5 --incomplete-only
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def count_complete_runs(skipgrams_path: Path) -> int:
    parent = skipgrams_path.parent
    n = 0
    for d in sorted(parent.glob("run_*")):
        if (d / "model.bin").exists():
            n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model_dir", help="Root model directory (contains <corpus>/<period>/skipgrams.txt.gz)")
    ap.add_argument("--runs", type=int, default=5, help="Target number of runs per skipgrams file")
    ap.add_argument("--incomplete-only", action="store_true",
                    help="Only list skipgrams files with fewer than --runs completed runs")
    ap.add_argument("--show-counts", action="store_true",
                    help="Print existing run counts alongside paths (for humans, not for sbatch --array)")
    args = ap.parse_args()

    root = Path(args.model_dir).resolve()
    if not root.is_dir():
        print(f"error: {root} is not a directory", file=sys.stderr)
        return 2

    paths = sorted(root.rglob("skipgrams.txt.gz"))
    for p in paths:
        done = count_complete_runs(p)
        if args.incomplete_only and done >= args.runs:
            continue
        if args.show_counts:
            rel = p.relative_to(root)
            print(f"{rel}\t{done}/{args.runs}")
        else:
            print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
