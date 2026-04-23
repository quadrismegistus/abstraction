"""Prototype: does the Brysbaert-translated German anchor set fix the
`stehen`/`wolle`/`fliegen`-type miscoring that motivated this whole thread?

Pipeline:
  1. Load Brysbaert primary translations (27,949 rows, 'word', 'pos', 'de',
     'fr', 'concreteness'). These are human concreteness ratings from
     Brysbaert attached to LLM-translated German lemmas.
  2. Load one well-stocked German W2V model (1800-1900 default; large vocab).
  3. Build a Brysbaert-anchored contrast axis: concrete = de words where
     Brysbaert rating z-scores to >=+1; abstract = z-score <=-1. This
     gives us an axis with real verb coverage.
  4. Also build the NATIVE German anchor axis from Conde+Schmidtke orig
     norms (what we've been using) as the baseline.
  5. Project diagnostic words onto both axes, compare.
  6. Bonus: combined axis (native + brysbaert-de).

This is not a production pipeline — single model, no period matching, no
cross-run averaging. Purpose is to answer: does verb-anchor supplementation
change verb-scoring in the expected direction?
"""

import os
import sys

import numpy as np
import pandas as pd

from abstraction.config import PATH_MODELS_DE
from abstraction.models import load_model, MODEL_MIN_COUNT
from abstraction.norms_de import get_orignorms_de

BRYS_DE_CSV = os.path.join(
    os.path.dirname(__file__), "..", "data", "fields", "sources",
    "brysbaert_translations_primary.csv",
)
MODEL_PATH = os.path.join(PATH_MODELS_DE, "german_pd", "1800-1900", "run_01", "model.bin")
MODEL_VOCAB = os.path.join(PATH_MODELS_DE, "german_pd", "1800-1900", "run_01", "vocab.txt")

ZCUT = 1.0

# Words we already know are miscored or diagnostic
DIAGNOSTICS = [
    # Verbs we identified as problematic earlier
    ("stehen", "current score ~-1.26 (abstract), human says +0.65 (concrete)"),
    ("sehen", "~-1.27 (abstract), human -0.62 (mildly abstract)"),
    ("fliegen", "~+1.54 (concrete), human +1.50 (concrete) — agrees"),
    ("wolle", "~-0.48 (abstract-ish), human +0.76 (concrete) — homograph"),
    ("sagen", "~-0.79 (abstract), no human norm"),
    ("bewegen", "~-0.79 (abstract), no human norm"),
    ("tragen", "new Brysbaert example — carry/wear/bear"),
    # Concrete-verb controls
    ("laufen", "should stay concrete"),
    ("essen", "should stay concrete"),
    ("schlagen", "should stay concrete"),
    # Abstract-verb controls
    ("denken", "should stay abstract"),
    ("glauben", "should stay abstract"),
    ("wissen", "should stay abstract"),
    # Noun controls
    ("stadt", "concrete noun"),
    ("freiheit", "abstract noun"),
    ("tisch", "concrete noun"),
]


def build_axis(label, anchors_df, kv):
    """anchors_df has columns: word, rating. Rating z-scored, anchors picked at |z|>=1."""
    z = (anchors_df["rating"] - anchors_df["rating"].mean()) / anchors_df["rating"].std()
    conc = anchors_df[z >= ZCUT]["word"].tolist()
    abst = anchors_df[z <= -ZCUT]["word"].tolist()
    conc_in = [w for w in conc if w in kv]
    abst_in = [w for w in abst if w in kv]
    print(f"  {label}: concrete anchors {len(conc_in)}/{len(conc)}, abstract {len(abst_in)}/{len(abst)}",
          file=sys.stderr)
    if len(conc_in) < 15 or len(abst_in) < 15:
        return None
    cvec = np.mean([kv[w] for w in conc_in], axis=0)
    avec = np.mean([kv[w] for w in abst_in], axis=0)
    axis = cvec - avec
    return axis / np.linalg.norm(axis)


def score_vocab(axis, kv):
    """Return {word: z_score} across vocabulary."""
    vocab = list(kv.index_to_key)
    mat = np.stack([kv[w] for w in vocab])
    mat_norm = mat / np.linalg.norm(mat, axis=1, keepdims=True)
    proj = mat_norm @ axis
    z = (proj - proj.mean()) / proj.std()
    return dict(zip(vocab, z))


def main():
    print("Loading Brysbaert primaries ...", file=sys.stderr)
    brys = pd.read_csv(BRYS_DE_CSV)
    # Filter to rows with valid DE translation
    brys = brys[brys["de"].notna() & (brys["de"] != "")]
    # Lowercase DE for W2V lookup (W2V vocab is lowercase)
    brys["de_lc"] = brys["de"].str.lower()
    # For multi-word phrases, keep only first token (closest approximation)
    brys["de_lc"] = brys["de_lc"].str.split().str[0]
    # Drop duplicates — multiple English words may map to same German
    brys = brys.drop_duplicates(subset=["de_lc"])
    print(f"  {len(brys):,} unique German anchor candidates", file=sys.stderr)
    print(f"  POS breakdown: {brys['pos'].value_counts().to_dict()}", file=sys.stderr)

    # Brysbaert rating: 1-5 concreteness. Use as-is for z-scoring.
    brys_anchors = brys[["de_lc", "concreteness"]].rename(
        columns={"de_lc": "word", "concreteness": "rating"}
    )

    print("\nLoading German orig norms ...", file=sys.stderr)
    native = get_orignorms_de(remove_stopwords=False)
    # Use cross-source median as the anchor rating
    native_median = native.median(axis=1)
    native_anchors = pd.DataFrame({
        "word": native_median.index,
        "rating": native_median.values,
    })
    native_anchors = native_anchors.dropna()
    print(f"  {len(native_anchors):,} native anchor candidates", file=sys.stderr)

    print(f"\nLoading W2V model: {MODEL_PATH}", file=sys.stderr)
    model = load_model(MODEL_PATH, MODEL_VOCAB, min_count=MODEL_MIN_COUNT)
    kv = model.wv
    print(f"  vocab: {len(kv.index_to_key):,} words", file=sys.stderr)

    print("\nBuilding axes ...", file=sys.stderr)
    ax_native = build_axis("NATIVE (Conde+Schmidtke)", native_anchors, kv)
    ax_brys = build_axis("BRYSBAERT-DE (translated)", brys_anchors, kv)

    # Combined: union of both anchor tables, rating z-scored within source first
    # then combined
    def zscore(df):
        df = df.copy()
        df["rating_z"] = (df["rating"] - df["rating"].mean()) / df["rating"].std()
        return df
    native_z = zscore(native_anchors).rename(columns={"rating_z": "rating"}).drop(columns=["rating"], errors="ignore")
    native_z["rating"] = (native_anchors["rating"] - native_anchors["rating"].mean()) / native_anchors["rating"].std()
    brys_z = brys_anchors.copy()
    brys_z["rating"] = (brys_anchors["rating"] - brys_anchors["rating"].mean()) / brys_anchors["rating"].std()
    combined = pd.concat([native_z[["word", "rating"]], brys_z[["word", "rating"]]])
    combined = combined.groupby("word")["rating"].mean().reset_index()
    ax_combined = build_axis("COMBINED (native + brysbaert)", combined, kv)

    print("\nProjecting vocab onto each axis ...", file=sys.stderr)
    scores_native = score_vocab(ax_native, kv) if ax_native is not None else {}
    scores_brys = score_vocab(ax_brys, kv) if ax_brys is not None else {}
    scores_combined = score_vocab(ax_combined, kv) if ax_combined is not None else {}

    print("\n" + "=" * 90)
    print(f"{'word':<14} {'NATIVE':>8} {'BRYSBAERT':>11} {'COMBINED':>10}  note")
    print("=" * 90)
    for w, note in DIAGNOSTICS:
        def fmt(d):
            v = d.get(w)
            return "   —   " if v is None else f"{v:+7.3f}"
        in_brys = w in brys["de_lc"].values
        tag = "  [B]" if in_brys else ""
        print(f"{w:<14} {fmt(scores_native):>8} {fmt(scores_brys):>11} "
              f"{fmt(scores_combined):>10}  {note}{tag}")

    # Correlation check: how much do the three axes agree on the shared vocab?
    shared = set(scores_native) & set(scores_brys) & set(scores_combined)
    if shared:
        shared_sample = list(shared)[:30000]
        df = pd.DataFrame({
            "native": [scores_native[w] for w in shared_sample],
            "brys":   [scores_brys[w] for w in shared_sample],
            "combined": [scores_combined[w] for w in shared_sample],
        })
        print(f"\n--- Inter-axis correlation on {len(shared_sample):,} shared words ---")
        print(df.corr().round(3).to_string())

    # How many Brysbaert anchors enter through VERB that were missing in native?
    print("\n--- Coverage gain on verbs ---")
    brys_verbs = brys[brys["pos"] == "Verb"]["de_lc"].tolist()
    native_verb_coverage = sum(1 for w in brys_verbs if w in native_median.index and pd.notna(native_median.loc[w]))
    print(f"Brysbaert verbs (DE): {len(brys_verbs):,}")
    print(f"  also in native norms: {native_verb_coverage} ({100*native_verb_coverage/len(brys_verbs):.1f}%)")
    print(f"  NEW verb anchors from Brysbaert: {len(brys_verbs) - native_verb_coverage:,}")


if __name__ == "__main__":
    main()
