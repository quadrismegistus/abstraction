"""
Sliding-window counting of abstract/concrete words in texts.
"""

import os
from collections import Counter

import pandas as pd
from tqdm import tqdm

from .config import (
    COUNT_DIR, COUNT_WINDOW_LEN, MODERNIZE_SPELLING,
    SOURCES_FOR_COUNTING,
)
from .corpus import load_corpus, pmap_iter
from .norms import get_allcontrasts
from .tokenize import tokenize


# ---------------------------------------------------------------------------
# Norm contrasts cache
# ---------------------------------------------------------------------------

_NORM_CONTRASTS = None


def get_norms_for_counting(sources=None, periods=None):
    global _NORM_CONTRASTS
    if _NORM_CONTRASTS is None:
        _NORM_CONTRASTS = [
            dx for dx in get_allcontrasts(remove_stopwords=True)
            if dx["source"] in SOURCES_FOR_COUNTING
        ]
    return [
        dx for dx in _NORM_CONTRASTS
        if (not sources or dx["source"] in sources)
        and (not periods or dx["period"] in periods)
    ]


# ---------------------------------------------------------------------------
# Core counting
# ---------------------------------------------------------------------------

def _count_window(dx, recog_tokens, all_tokens=None, incl_psg=False, vocab_len=5):
    """Count abstract/concrete words in a single window of recognized tokens."""
    tokenset = set(recog_tokens)
    countd = Counter(recog_tokens)
    result = {
        "num_words": len([w for w in (all_tokens or recog_tokens) if w and w[0].isalpha()]),
        "num_tokens": len(recog_tokens),
        "num_types": len(tokenset),
        "contrast": dx["contrast"],
        "source": dx["source"],
        "period": dx["period"],
    }

    total = 0
    key_map = {"neg": "abs", "pos": "conc", "neither": "neither"}
    for key, label in key_map.items():
        shared = set(dx[key]) & tokenset
        num = sum(countd[w] for w in shared)
        result[f"num_{label}"] = num
        total += num
        if not incl_psg:
            result[label] = ", ".join(sorted(shared, key=lambda w: -countd[w])[:vocab_len])
    result["num_total"] = total

    if incl_psg and all_tokens is not None:
        parens = {"(", "]"}
        psg = []
        for tok in all_tokens:
            if tok in {"n't"} or (not tok[0].isalpha() and tok[0] not in parens):
                if psg:
                    psg[-1] += tok
                    continue
            tokl = tok.lower()
            if tokl in dx["neg"]:
                tok = f"<i><b>{tok}</b></i>"
            elif tokl in dx["pos"]:
                tok = f"<i><u>{tok}</u></i>"
            elif tokl in dx["neither"]:
                tok = f"<i>{tok}</i>"
            if psg and psg[-1] in parens:
                psg[-1] += tok
            else:
                psg.append(tok)
        result["passage"] = " ".join(psg).strip().replace("\n", "\n<br>")

    return result


def count_absconc(txt, window_len=COUNT_WINDOW_LEN, keep_last=True,
                  sources=None, periods=None, incl_psg=False,
                  modernize=MODERNIZE_SPELLING):
    """Count abstract/concrete words in sliding windows over a text."""
    tokens = tokenize(txt, lower=False, modernize=modernize)
    results = []

    for dx in get_norms_for_counting(sources=sources, periods=periods):
        allwords = dx["neg"] | dx["pos"] | dx["neither"]
        all_tokens, recog_tokens = [], []

        for i, tok in enumerate(tokens):
            tokl = tok.lower()
            all_tokens.append(tok)
            if tokl in allwords:
                recog_tokens.append(tokl)

            if len(recog_tokens) >= window_len:
                cdx = _count_window(dx, recog_tokens, all_tokens, incl_psg=incl_psg)
                cdx["slice"] = len(results) + 1
                cdx["tok_i"] = i + 1
                results.append(cdx)
                all_tokens, recog_tokens = [], []

        if keep_last and recog_tokens:
            cdx = _count_window(dx, recog_tokens, all_tokens, incl_psg=incl_psg)
            cdx["slice"] = len(results) + 1
            cdx["tok_i"] = i + 2 if tokens else 0
            results.append(cdx)

    return results


def count_absconc_path(path, **kwargs):
    """Count abstract/concrete words in a text file."""
    if path.endswith(".gz") and os.path.exists(path[:-3]):
        path = path[:-3]
    with open(path, encoding="utf-8", errors="ignore") as f:
        results = count_absconc(f.read(), **kwargs)
    for dx in results:
        dx["path"] = path
    return results


def count_absconc_psg(txt, sources=None, periods=None, **kwargs):
    """Count with passage text included, return sorted DataFrame."""
    if sources is None:
        sources = {"Median"}
    if periods is None:
        periods = {"median"}
    df = pd.DataFrame(count_absconc(
        txt, incl_psg=True, sources=sources, periods=periods,
        window_len=COUNT_WINDOW_LEN, **kwargs,
    ))
    if len(df):
        df["abs-conc"] = df["num_abs"] - df["num_conc"]
        df = df.sort_values("abs-conc")
    return df


# ---------------------------------------------------------------------------
# Corpus-level counting
# ---------------------------------------------------------------------------

def count_absconc_corpus(corpus_name, num_proc=1, incl_psg=False, ofn=None):
    """Count abstract/concrete words across all texts in a corpus."""
    corpus = load_corpus(corpus_name)
    meta = corpus.metadata
    paths = [corpus.text_path(tid) for tid in meta["id"]]
    path2id = dict(zip(paths, meta["id"]))

    func = count_absconc_path
    if incl_psg:
        def func(path):
            try:
                with open(path, encoding="utf-8", errors="ignore") as f:
                    ld = count_absconc(
                        f.read(), sources={"Median"}, periods={"median"},
                        incl_psg=True,
                    )
                for dx in ld:
                    dx["path"] = path
                return ld
            except Exception as e:
                print(f"Error counting {path}: {e}")
                return []

    if not ofn:
        psg_tag = ".psgs" if incl_psg else ""
        ofn = os.path.join(COUNT_DIR, f"data.absconc.{corpus_name}{psg_tag}.csv.gz")

    os.makedirs(os.path.dirname(ofn), exist_ok=True)

    from .utils import writegen

    def _gen():
        for pathld in pmap_iter(func, paths, num_proc=num_proc,
                                desc=f"Counting abstract/concrete words in {corpus_name}"):
            for dx in pathld:
                dx["id"] = path2id.get(dx.pop("path", ""))
                yield dx

    header = ["id", "slice", "source", "period", "num_abs", "num_conc",
              "num_neither", "num_total", "num_types"]
    if incl_psg:
        header.append("passage")
    else:
        header.extend(["abs", "conc", "neither"])

    writegen(ofn, _gen, header=header)
