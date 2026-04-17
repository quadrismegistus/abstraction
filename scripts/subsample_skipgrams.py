#!/usr/bin/env python3
"""Stream a gzipped skipgrams file and write a subsample.

Two modes:
  --prob P        Keep each line with probability P. Single pass, O(1) memory.
                  Output line count is approximate (~P * input lines). PREFERRED.
  --target N      Estimates probability from file size (10 MB sample, assumes ~5x
                  gzip ratio), then stream-filters with a hard cap at N lines.
                  WARNING: the ratio guess can be off by ~2x for dense skipgram
                  files — you may end up with half the lines you wanted. If you
                  know the input's line count (or can guess from similar files),
                  compute P yourself and pass --prob instead.

Reservoir sampling for exact counts would require holding N lines in memory —
for N=150M and ~30-byte lines that's ~4.5 GB. Probabilistic is cheaper and
fine for skipgram training (gensim's W2V is insensitive to exact row count).

Note: Python's gzip module is single-threaded. For very large files (>10 GB
compressed) a shell pipeline is much faster:

    pigz -dc IN.txt.gz | gawk 'BEGIN{srand(42)} rand() < 0.1' | pigz -c > OUT.txt.gz

Usage:
    python scripts/subsample_skipgrams.py IN.txt.gz OUT.txt.gz --prob 0.1
    python scripts/subsample_skipgrams.py IN.txt.gz OUT.txt.gz --target 150000000
"""
from __future__ import annotations

import argparse
import gzip
import os
import random
import sys
import time


def _open_in(path):
    return gzip.open(path, "rb") if path.endswith(".gz") else open(path, "rb")


def _open_out(path, compresslevel=6):
    return (
        gzip.open(path, "wb", compresslevel=compresslevel)
        if path.endswith(".gz")
        else open(path, "wb")
    )


def estimate_prob_from_size(path: str, target: int, sample_bytes: int = 10_000_000) -> float:
    """Estimate what --prob to use to hit approximately `target` lines."""
    total_bytes = os.path.getsize(path)
    # Count lines in the first sample_bytes of decompressed stream
    bytes_read = 0
    lines_read = 0
    with _open_in(path) as f:
        while bytes_read < sample_bytes:
            line = f.readline()
            if not line:
                break
            bytes_read += len(line)
            lines_read += 1
    if lines_read == 0:
        raise RuntimeError(f"{path} appears empty")

    # This assumes gzip compression ratio is uniform — usually close enough
    # for skipgram files (token streams are low-entropy).
    # Bytes read are decompressed bytes; we need to extrapolate to full
    # decompressed size. A rough assumption: compression ratio is constant,
    # so decompressed_total ≈ total_bytes * (decompressed_sample / compressed_sample_read).
    # Since readline() works on decompressed bytes, we don't know the
    # compressed offset. Fallback: assume skipgrams gzip at ~5x.
    est_decompressed = total_bytes * 5.0
    est_total_lines = est_decompressed * (lines_read / bytes_read)
    prob = target / est_total_lines
    return max(min(prob, 1.0), 1e-6), est_total_lines


def subsample(in_path: str, out_path: str, prob: float, seed: int, max_lines: int | None, progress_every: int):
    rng = random.Random(seed)
    n_in = 0
    n_out = 0
    t0 = time.time()
    with _open_in(in_path) as fin, _open_out(out_path) as fout:
        for line in fin:
            n_in += 1
            if rng.random() < prob:
                fout.write(line)
                n_out += 1
                if max_lines is not None and n_out >= max_lines:
                    print(f"  hit max_lines cap at {n_out:,} kept (read {n_in:,})", file=sys.stderr, flush=True)
                    break
            if n_in % progress_every == 0:
                dt = time.time() - t0
                print(
                    f"  {n_in:,} read, {n_out:,} kept ({n_out/n_in:.2%}) — "
                    f"{n_in/dt/1e6:.1f}M lines/s",
                    file=sys.stderr,
                    flush=True,
                )
    dt = time.time() - t0
    print(
        f"done: {n_in:,} read, {n_out:,} kept ({n_out/max(n_in,1):.2%}) in {dt:.0f}s",
        file=sys.stderr,
        flush=True,
    )
    return n_in, n_out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", help="Input skipgrams file (.txt or .txt.gz)")
    ap.add_argument("output", help="Output path (.txt.gz recommended)")
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--prob", type=float, help="Keep each line with this probability (0..1)")
    grp.add_argument("--target", type=int, help="Approximate target line count")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")
    ap.add_argument("--progress-every", type=int, default=10_000_000, help="Progress line every N (default: 10M)")
    args = ap.parse_args()

    if args.prob is not None:
        prob = args.prob
        max_lines = None
    else:
        prob, est_total = estimate_prob_from_size(args.input, args.target)
        print(
            f"estimate: ~{est_total/1e9:.2f}B input lines → using prob={prob:.4f} "
            f"to target {args.target:,} (with cap)",
            file=sys.stderr,
            flush=True,
        )
        max_lines = args.target

    subsample(args.input, args.output, prob, args.seed, max_lines, args.progress_every)
    return 0


if __name__ == "__main__":
    sys.exit(main())
