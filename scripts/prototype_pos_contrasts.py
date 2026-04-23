"""Prototype: per-POS abstract/concrete contrasts on English W2V vecnorms.

Hypothesis: with unbalanced POS on the two contrast poles, the vec axis leaks
word-class into the concreteness signal. Splitting anchors by POS and
projecting each word onto its own POS axis should produce more intra-POS
semantic variance and less inter-POS bias.

This is exploratory only — does not modify production norms. Loads one W2V
model, POS-tags vocab and anchors, builds per-POS axes, and prints a
comparison table for diagnostic words. If the numbers look better than the
global axis, we promote the approach into gen_vecnorms().
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import spacy
from collections import defaultdict

from abstraction.config import PATH_MODELS
from abstraction.models import load_model, MODEL_MIN_COUNT
from abstraction.norms import get_orignorms

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Use BLBooks 1800-1900 as our test model — large vocab, well-sampled period.
MODEL_PATH = os.path.join(PATH_MODELS, "blbooks", "1800-1900", "run_01", "model.bin")
MODEL_VOCAB = os.path.join(PATH_MODELS, "blbooks", "1800-1900", "run_01", "vocab.txt")

# MRC-Imag is the best-balanced EN source for per-POS anchors (has verbs).
# MT-Conc has even more data but is MegaTurk crowdsourced; let's start with
# the canonical MRC psycholinguistic database. Can swap later.
SOURCES_TO_TEST = ["MRC-Imag", "MT-Conc"]

ZCUT = 1.0  # anchor threshold

# Diagnostic words spanning POS classes + known-surprise cases.
DIAGNOSTIC_WORDS = [
    # Concrete nouns (should score concrete)
    "servant", "body", "child", "hand", "dog", "stone", "tree", "house",
    # Abstract nouns (should score abstract)
    "freedom", "liberty", "justice", "virtue", "truth", "idea", "concept",
    # Action verbs (previously flagged as ambiguous on vec axis)
    "stand", "sit", "walk", "run", "fly", "fall", "jump", "throw",
    # Perception/cognition verbs (expect abstract)
    "see", "hear", "think", "believe", "know", "understand", "decide",
    # Modal/auxiliary verbs
    "should", "must", "could", "would", "shall",
    # Speech verbs
    "say", "tell", "speak", "ask",
    # Concrete adjectives
    "red", "hot", "cold", "tall", "round",
    # Abstract adjectives
    "happy", "sad", "true", "good", "free", "just",
    # Adverbs
    "quickly", "slowly", "truly", "hardly", "really",
]


def main():
    # ---- 1. Load model ----
    print(f"Loading W2V model from {MODEL_PATH} ...", file=sys.stderr)
    model = load_model(MODEL_PATH, MODEL_VOCAB, min_count=MODEL_MIN_COUNT)
    if model is None:
        print("Failed to load model.", file=sys.stderr)
        return
    kv = model.wv if hasattr(model, "wv") else model
    print(f"  vocab: {len(kv.index_to_key):,} words", file=sys.stderr)

    # ---- 2. POS-tag anchor vocabulary ----
    print("Loading spaCy en_core_web_sm ...", file=sys.stderr)
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

    print("Loading EN orig norms and tagging ...", file=sys.stderr)
    orig = get_orignorms(remove_stopwords=False)
    anchor_words = [w for w in orig.index if isinstance(w, str) and w in kv]
    print(f"  {len(anchor_words):,} anchor words present in model vocab", file=sys.stderr)

    # Tag each anchor word as a single-token document. Single-word context is
    # weak — spaCy guesses from morphology and defaults. Good enough for a
    # prototype; production would tag in running text.
    word_pos = {}
    for w, doc in zip(anchor_words, nlp.pipe(anchor_words, batch_size=1024)):
        word_pos[w] = doc[0].pos_ if len(doc) else "X"

    # ---- 3. Build per-source, per-POS anchor sets ----
    print("\nPer-source per-POS anchor counts:", file=sys.stderr)
    axes = {}  # (source, pos) -> (concrete_centroid, abstract_centroid, axis_vec)
    for source in SOURCES_TO_TEST:
        col = f"Abs-Conc.{source}"
        if col not in orig.columns:
            print(f"  {source}: missing column — skip", file=sys.stderr)
            continue
        series = orig[col]
        for pos in ("NOUN", "VERB", "ADJ", "ADV", "ALL"):
            if pos == "ALL":
                subset = [w for w in anchor_words if pd.notna(series.get(w))]
            else:
                subset = [w for w in anchor_words
                          if word_pos.get(w) == pos and pd.notna(series.get(w))]
            conc = [w for w in subset if series[w] >= ZCUT]
            abst = [w for w in subset if series[w] <= -ZCUT]
            print(f"  {source}/{pos:<4}  conc={len(conc):>4}  abs={len(abst):>4}",
                  file=sys.stderr)
            if len(conc) < 15 or len(abst) < 15:
                continue
            cvec = np.mean([kv[w] for w in conc], axis=0)
            avec = np.mean([kv[w] for w in abst], axis=0)
            axis = cvec - avec
            axis = axis / np.linalg.norm(axis)
            axes[(source, pos)] = axis

    # ---- 4. POS-tag diagnostic words and project onto their POS axis ----
    print("\nTagging diagnostic words ...", file=sys.stderr)
    diag_pos = {}
    for w, doc in zip(DIAGNOSTIC_WORDS, nlp.pipe(DIAGNOSTIC_WORDS, batch_size=64)):
        diag_pos[w] = doc[0].pos_ if len(doc) else "X"

    # ---- 5. Project every word onto every axis; z-score per axis across vocab ----
    print("Computing projections + z-scoring ...", file=sys.stderr)
    # Build a matrix of all vocab vectors for fast projection.
    all_vecs = np.stack([kv[w] for w in kv.index_to_key])  # (V, D)
    all_vecs_norm = all_vecs / np.linalg.norm(all_vecs, axis=1, keepdims=True)
    vocab_idx = {w: i for i, w in enumerate(kv.index_to_key)}

    axis_scores = {}  # (source, pos) -> z-scored projection array (len V)
    for key, axis in axes.items():
        proj = all_vecs_norm @ axis  # cosine sim between unit axis and unit-normed vecs
        # z-score across full vocab so scales match the original vecnorm pipeline
        proj_z = (proj - proj.mean()) / proj.std()
        axis_scores[key] = proj_z

    # ---- 6. Compare on diagnostic words ----
    rows = []
    for w in DIAGNOSTIC_WORDS:
        if w not in vocab_idx:
            continue
        idx = vocab_idx[w]
        pos = diag_pos.get(w, "X")
        row = {"word": w, "POS (tagger)": pos}
        for source in SOURCES_TO_TEST:
            global_key = (source, "ALL")
            pos_key = (source, pos)
            row[f"{source}.global"] = axis_scores[global_key][idx] if global_key in axis_scores else np.nan
            row[f"{source}.bypos"]  = axis_scores[pos_key][idx] if pos_key in axis_scores else np.nan
            # human rating for reference
            col_name = f"Abs-Conc.{source}"
            if col_name in orig.columns and w in orig.index:
                row[f"{source}.orig"] = orig.at[w, col_name]
            else:
                row[f"{source}.orig"] = np.nan
        rows.append(row)

    df = pd.DataFrame(rows)
    pd.set_option("display.float_format", lambda x: f"{x:+.2f}" if pd.notna(x) else "   —")
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print("\n===== Diagnostic word comparison =====")
    print(df.to_string(index=False))

    # ---- 7. Aggregate: per-POS mean shift between global and bypos ----
    print("\n===== Mean |global − bypos| shift by POS =====")
    # Compute over the full vocab (not just diagnostic), per source.
    for source in SOURCES_TO_TEST:
        print(f"\n{source}:")
        for pos in ("NOUN", "VERB", "ADJ", "ADV"):
            if (source, pos) not in axis_scores or (source, "ALL") not in axis_scores:
                print(f"  {pos}: (missing axis)")
                continue
            # Only consider words whose auto-tag matches this POS, for fair
            # comparison. Tag a random 10K sample for speed.
            rng = np.random.default_rng(0)
            sample_idx = rng.choice(len(kv.index_to_key), size=10000, replace=False)
            sample_words = [kv.index_to_key[i] for i in sample_idx]
            sample_docs = list(nlp.pipe(sample_words, batch_size=1024))
            sample_tags = [d[0].pos_ if len(d) else "X" for d in sample_docs]
            mask = np.array([t == pos for t in sample_tags])
            if not mask.any():
                print(f"  {pos}: no words in sample")
                continue
            gsel = axis_scores[(source, "ALL")][sample_idx][mask]
            psel = axis_scores[(source, pos)][sample_idx][mask]
            print(f"  {pos}: n={mask.sum():>4}  "
                  f"global_mean={gsel.mean():+.3f}  "
                  f"bypos_mean={psel.mean():+.3f}  "
                  f"mean_diff={(psel - gsel).mean():+.3f}  "
                  f"mean_|shift|={np.abs(psel - gsel).mean():.3f}")


if __name__ == "__main__":
    main()
