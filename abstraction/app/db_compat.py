"""Backward-compatible loading of regular (non-arc) score CSVs.

Used as a fallback when arc_*.csv files don't exist yet.
"""

import os

import pandas as pd
from tqdm import tqdm


def load_regular_csvs(scores_dir, csv_files):
    """Load regular per-corpus score CSVs, adding _id and source_corpus columns."""
    try:
        import sys
        sys.path.insert(0, os.path.expanduser("~/github/lltk"))
        from lltk.corpus.hathi.hathi import hathi_id_normalize
    except ImportError:
        hathi_id_normalize = None

    from ..analysis import EXCLUDE_CORPORA

    all_dfs = []
    for fn in tqdm(csv_files, desc="  Regular corpora"):
        corpus_name = fn.removesuffix(".csv")
        if corpus_name in EXCLUDE_CORPORA:
            continue
        path = os.path.join(scores_dir, fn)
        try:
            df = pd.read_csv(path, dtype={"id": str})
        except Exception:
            continue
        df["source_corpus"] = corpus_name
        df["arc_corpus"] = corpus_name  # no arc grouping

        # Build _id from corpus + id
        if hathi_id_normalize and corpus_name.startswith("hathi"):
            df["_id"] = "_" + corpus_name + "/" + df["id"].apply(hathi_id_normalize)
        elif corpus_name == "chicago":
            df["_id"] = "_" + corpus_name + "/" + df["id"].apply(lambda x: str(x).zfill(8))
        elif corpus_name == "gildedage":
            df["_id"] = "_" + corpus_name + "/" + df["id"].str.replace(" ", "_")
        else:
            df["_id"] = "_" + corpus_name + "/" + df["id"].astype(str)

        all_dfs.append(df)

    return all_dfs
