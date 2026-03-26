import csv
import gzip
import json
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import zscore
from tqdm import tqdm


def zfy(series):
    """Z-score a numeric series, dropping NaN."""
    series = pd.to_numeric(series, errors="coerce").dropna()
    return pd.Series(zscore(series), index=series.index)


def read_df(fn):
    if ".feather" in fn or fn.endswith(".ft"):
        return pd.read_feather(fn)
    elif ".csv" in fn:
        return pd.read_csv(fn)
    elif ".pkl" in fn:
        return pd.read_pickle(fn)
    elif ".xls" in fn:
        return pd.read_excel(fn)
    elif ".jsonl" in fn:
        return pd.read_json(fn, lines=True)
    elif ".json" in fn:
        return pd.read_json(fn)
    else:
        raise ValueError(f"Unknown file type: {fn}")


def save_df(df, fn):
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    if fn.endswith(".feather") or fn.endswith(".ft"):
        df.reset_index().to_feather(fn)
    elif fn.endswith(".pkl"):
        df.to_pickle(fn)
    elif fn.endswith(".csv.gz"):
        df.to_csv(fn, index=True)
    elif fn.endswith(".csv"):
        df.to_csv(fn, index=True)
    else:
        raise ValueError(f"Unknown file type: {fn}")


def writegen(fn, generator, header=None, args=(), kwargs=None):
    """Write an iterable of dicts to CSV (optionally gzipped)."""
    if kwargs is None:
        kwargs = {}
    iterator = generator(*args, **kwargs)
    if not header:
        first = next(iterator)
        header = sorted(first.keys())
        # re-chain the first element back
        import itertools
        iterator = itertools.chain([first], iterator)

    opener = gzip.open(fn, "wt") if fn.endswith(".gz") else open(fn, "w")
    with opener as f:
        writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        for dx in iterator:
            writer.writerow(dx)
    print(">> saved:", fn)


def writegen_jsonl(fn, generator, args=(), kwargs=None):
    if kwargs is None:
        kwargs = {}
    with open(fn, "w") as f:
        for dx in generator(*args, **kwargs):
            f.write(json.dumps(dx) + "\n")
    print(">> saved:", fn)


def download_tqdm(url, save_to):
    import requests
    r = requests.get(url, stream=True)
    total_size = int(r.headers.get("content-length", 0))
    with open(save_to, "wb") as f:
        for chunk in tqdm(r.iter_content(32 * 1024), total=total_size, unit="B", unit_scale=True):
            if chunk:
                f.write(chunk)
    return save_to


def cleanhtml(raw_html):
    return re.sub(r"<.*?>", "", raw_html)


def get_slices(lst, slice_length=None, num_slices=None, keep_runts=True):
    if not slice_length and not num_slices:
        return [lst]
    if not slice_length:
        slice_length = max(1, len(lst) // num_slices)
    chunks = [lst[i : i + slice_length] for i in range(0, len(lst), slice_length)]
    if keep_runts:
        return chunks
    return [c for c in chunks if len(c) == slice_length]


def parse_json_str(response):
    try:
        response = response.split("```json", 1)[-1]
        parts = response.split("```")
        response = parts[1] if len(parts) > 1 and parts[1].strip() else parts[0]
        return json.loads(response.strip())
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        return None


def get_avgs_df(df, gby=("genre", "corpus", "decade"), y="Abs-Conc.Median.median",
                min_texts=None):
    """Group, standardize, and aggregate scores with mean/stderr/count.

    Useful for plotting trends with error bars across genre, corpus, and decade.
    """
    df = df.copy()
    if min_texts:
        df = df.groupby(list(gby)).filter(lambda x: x["num_texts"].sum() >= min_texts)
    df[y] = (df[y] - df[y].mean()) / df[y].std()
    stats_df = (
        df.groupby(list(gby))[y]
        .agg(
            mean="mean",
            stderr=lambda x: x.std() / np.sqrt(len(x)),
            count="count",
        )
    )
    return stats_df.sort_index()


def sent_tokenize_exact(text):
    """Split text into sentence chunks preserving all characters exactly."""
    from nltk.tokenize import PunktSentenceTokenizer
    tokenizer = PunktSentenceTokenizer()
    spans = list(tokenizer.span_tokenize(text))
    chunks = []
    last = 0
    for start, end in spans:
        if start > last:
            chunks.append(text[last:start])
        chunks.append(text[start:end])
        last = end
    if last < len(text):
        chunks.append(text[last:])
    return chunks
