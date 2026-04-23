"""Translate Brysbaert English concreteness lexicon into German + French.

Produces supplementary anchor data for the German and French vec-norm
pipelines, where native psycholinguistic norms are noun-dominated and
verb-sparse. The LLM never sees Brysbaert ratings — they're attached
downstream after translation, avoiding circularity.

Pipeline:
  Phase 1: stratified 1000-word sample (250 per POS). Inspect ambiguity
           distribution, spot-check per-POS failure modes.
  Phase 2: full 27,949-word run. HashStash caches Phase 1 so it's free.

Usage:
    python scripts/translate_brysbaert.py --sample 1000
    python scripts/translate_brysbaert.py --sample all
"""

import argparse
import os
import sys

import pandas as pd

from largeliterarymodels.tasks.translate_word import (
    TranslationTask, format_word_for_translation,
)


BRYSBAERT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "fields", "sources",
    "Concreteness_ratings_Brysbaert_et_al_BRM.txt",
)
OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "data", "fields", "sources",
)

CONTENT_POS = ("Noun", "Verb", "Adjective", "Adverb")
DEFAULT_MODEL = "claude-haiku-4-5-20251001"


def load_filtered_brysbaert():
    df = pd.read_csv(BRYSBAERT_PATH, sep="\t")
    df = df[df["Bigram"] == 0]
    df = df[df["Dom_Pos"].isin(CONTENT_POS)]
    df = df.reset_index(drop=True)
    print(f"Brysbaert rows after filter: {len(df):,}", file=sys.stderr)
    print(df["Dom_Pos"].value_counts().to_string(), file=sys.stderr)
    return df


def stratified_sample(df, n_per_pos=250, seed=0):
    parts = []
    for pos in CONTENT_POS:
        subset = df[df["Dom_Pos"] == pos]
        k = min(n_per_pos, len(subset))
        parts.append(subset.sample(n=k, random_state=seed))
    return pd.concat(parts).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="1000",
                    help="'all' for full run, or integer total (stratified 4-way).")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--output", default=None,
                    help="Output CSV path. Defaults to brysbaert_translations_{phase}.csv "
                         "in data/fields/sources/.")
    args = ap.parse_args()

    brys = load_filtered_brysbaert()

    if args.sample == "all":
        subset = brys
        phase = "full"
    else:
        n = int(args.sample)
        n_per_pos = n // len(CONTENT_POS)
        subset = stratified_sample(brys, n_per_pos=n_per_pos)
        phase = f"phase1_n{len(subset)}"

    print(f"\nRunning translation on {len(subset):,} words ({phase})",
          file=sys.stderr)
    print(f"Model: {args.model}, workers: {args.workers}", file=sys.stderr)

    prompts = [format_word_for_translation(r["Word"], r["Dom_Pos"])
               for _, r in subset.iterrows()]
    metadata_list = [{"word": r["Word"], "pos": r["Dom_Pos"],
                      "source": "brysbaert"}
                     for _, r in subset.iterrows()]

    task = TranslationTask()
    task.map(prompts, metadata_list=metadata_list,
             model=args.model, num_workers=args.workers)

    df = task.df
    df = df[df["model"] == args.model].copy()

    # Attach Brysbaert ratings downstream (LLM never saw them).
    df = df.merge(
        subset[["Word", "Conc.M", "Conc.SD", "Dom_Pos"]],
        left_on="meta_word", right_on="Word", how="left",
    ).drop(columns=["Word"])

    # Ambiguity signals.
    df["de_ambig"] = (df[["de_1", "de_2", "de_3"]].ne("")
                      .fillna(False).sum(axis=1))
    df["fr_ambig"] = (df[["fr_1", "fr_2", "fr_3"]].ne("")
                      .fillna(False).sum(axis=1))

    output_path = args.output or os.path.join(
        OUTPUT_DIR, f"brysbaert_translations_{phase}.csv")
    df.to_csv(output_path, index=False)
    print(f"\n✓ Wrote {len(df):,} rows to {output_path}", file=sys.stderr)

    # Summary.
    print("\n----- Ambiguity distribution -----")
    print(f"DE spread (# non-empty candidates):")
    print(df["de_ambig"].value_counts().sort_index().to_string())
    print(f"\nFR spread:")
    print(df["fr_ambig"].value_counts().sort_index().to_string())
    print(f"\nRows with sense_note: {(df['sense_note'] != '').sum():,} / {len(df):,}")

    print("\n----- Example high-ambiguity entries (de_ambig >= 3) -----")
    hi = df[df["de_ambig"] >= 3].head(10)
    print(hi[["meta_word", "meta_pos", "de_1", "de_2", "de_3",
              "fr_1", "sense_note"]].to_string(index=False))

    print("\n----- Sanity sample per POS -----")
    for pos in CONTENT_POS:
        sample = df[df["meta_pos"] == pos].head(3)
        if not len(sample):
            continue
        print(f"\n{pos}:")
        print(sample[["meta_word", "Conc.M", "de_1", "de_2", "fr_1", "fr_2"]]
              .to_string(index=False))


if __name__ == "__main__":
    main()
