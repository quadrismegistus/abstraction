"""
Historical word embedding models: skipgram generation, Word2Vec training,
vector field computation, and distance calculations.
"""

import gzip
import os
import random
import time
from collections import defaultdict

import gensim
import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import (
    PATH_MODELS, MODEL_MIN_COUNT, MODEL_NUM_DIM,
    MODEL_PERIOD_LEN, PATH_VECNORMS, FIELD_DIR,
)
from .corpus import load_corpus, pmap, pmap_iter
from .norms import get_origcontrasts, _add_series_to_norms
from .tokenize import tokenize, tokenize_sentences
from .utils import zfy


# ---------------------------------------------------------------------------
# Skipgram generation
# ---------------------------------------------------------------------------

def yield_sentences_from_text(txt, min_len=10):
    """Yield word-lists from sentence-tokenized text."""
    buf = []
    for sent in tokenize_sentences(txt):
        buf.extend(w for w in tokenize(sent.lower()) if w and w[0].isalpha())
        if len(buf) >= min_len:
            yield buf
            buf = []


def save_skipgrams_from_paths(paths, ofn, min_len=10):
    os.makedirs(os.path.dirname(ofn), exist_ok=True)
    opener = gzip.open(ofn, "wt") if ofn.endswith(".gz") else open(ofn, "w")
    with opener as f:
        for path in tqdm(paths, desc="Generating skipgrams"):
            if not os.path.exists(path):
                continue
            with open(path, encoding="utf-8", errors="ignore") as tf:
                txt = tf.read()
            for sent in yield_sentences_from_text(txt, min_len=min_len):
                f.write(" ".join(sent) + "\n")


def gen_skipgrams_corpus(corpus_name, period_len=MODEL_PERIOD_LEN,
                         min_year=None, max_year=None, num_proc=1, force=False,
                         output_dir=None):
    """Generate skipgram files for each time period in a corpus."""
    corpus = load_corpus(corpus_name)
    oroot = os.path.join(output_dir or PATH_MODELS, corpus.id)
    df = corpus.metadata.copy()
    df["period"] = df["year"].apply(
        lambda y: f"{int(y) // period_len * period_len}-{int(y) // period_len * period_len + period_len}"
    )
    if min_year:
        df = df[df.year >= min_year]
    if max_year:
        df = df[df.year < max_year]

    objs = []
    for period, pdf in sorted(df.groupby("period")):
        paths = [corpus.text_path(tid) for tid in pdf["id"]]
        ofn = os.path.join(oroot, period, "skipgrams.txt.gz")
        if not force and os.path.exists(ofn):
            continue
        objs.append((paths, ofn))

    def _do(obj):
        save_skipgrams_from_paths(obj[0], obj[1])

    pmap(_do, objs, num_proc=num_proc, desc="Generating skipgrams by period")


def load_skipgrams(fn, num_skips=None, max_memory_gb=2.0):
    """Load skipgrams from a file into memory, optionally sampling.

    Returns a list of word-lists. If the file is larger than max_memory_gb
    (estimated), returns a StreamingSkipgrams iterator instead to avoid
    memory issues.
    """
    # Conservative: 2.2GB compressed BPO → 151GB in Python (70x).
    # Use compressed size directly with a low threshold.
    file_size_gb = os.path.getsize(fn) / (1024**3)

    if file_size_gb > max_memory_gb:
        print(f"  Large file ({file_size_gb:.1f}GB compressed). Streaming from disk.")
        return StreamingSkipgrams(fn, num_skips)

    opener = gzip.open(fn, "rb") if fn.endswith(".gz") else open(fn, "rb")
    sentences = []
    with opener as f:
        for line in tqdm(f, desc=f"Loading {os.path.basename(fn)}"):
            if isinstance(line, bytes):
                line = line.decode("utf-8")
            sentences.append(line.strip().split())
    if num_skips is not None and num_skips < len(sentences):
        sentences = random.sample(sentences, num_skips)
    return sentences


class StreamingSkipgrams:
    """Streams skipgrams from a file, re-reading on each epoch.

    Used for large files that don't fit in memory.
    """

    def __init__(self, fn, num_skips=None):
        self.fn = fn
        self.num_skips = num_skips
        if num_skips is not None:
            self.total_lines = self._count_lines()
            sample_size = min(num_skips, self.total_lines)
            self.sampled_lines = set(random.sample(range(self.total_lines), sample_size))
        else:
            self.sampled_lines = None

    def _count_lines(self):
        opener = gzip.open(self.fn, "rb") if self.fn.endswith(".gz") else open(self.fn)
        with opener as f:
            for i, _ in enumerate(f):
                pass
        return i + 1

    def __iter__(self):
        opener = gzip.open(self.fn, "rb") if self.fn.endswith(".gz") else open(self.fn)
        with opener as f:
            for i, line in enumerate(f):
                if self.sampled_lines is not None and i not in self.sampled_lines:
                    continue
                if isinstance(line, bytes):
                    line = line.decode("utf-8")
                yield line.strip().split()


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def _train_single_model(args):
    ifn, ofn_txt, ofn_bin, ofn_vocab, train_kwargs, num_skips = args
    sentences = load_skipgrams(ifn, num_skips)
    model = gensim.models.Word2Vec(sentences, **train_kwargs)
    model.wv.fill_norms()
    model.wv.save_word2vec_format(ofn_txt, ofn_vocab)
    model.save(ofn_bin)


def gen_model(skipgram_path, num_runs=1, num_workers=8, min_count=MODEL_MIN_COUNT,
              num_dimensions=MODEL_NUM_DIM, skipgram_size=10, num_skips=None,
              num_proc=1):
    objs = []
    for run in range(num_runs):
        odir = os.path.join(os.path.dirname(skipgram_path), f"run_{str(run + 1).zfill(2)}")
        os.makedirs(odir, exist_ok=True)
        ofn_bin = os.path.join(odir, "model.bin")
        if os.path.exists(ofn_bin):
            continue
        objs.append((
            skipgram_path,
            os.path.join(odir, "model.txt.gz"),
            ofn_bin,
            os.path.join(odir, "vocab.txt"),
            dict(workers=num_workers, sg=1, min_count=min_count,
                 vector_size=num_dimensions, window=skipgram_size),
            num_skips,
        ))
    if objs:
        pmap(_train_single_model, objs, num_proc=num_proc)


def gen_models_corpus(corpus_name, period_len=MODEL_PERIOD_LEN, **kwargs):
    skipgrams = sorted([
        d["path"] for d in get_model_paths(model_fn="skipgrams.txt.gz", period_len=period_len)
        if d["corpus"] == corpus_name
    ])
    for fn in skipgrams:
        gen_model(fn, **kwargs)


# ---------------------------------------------------------------------------
# Model loading and paths
# ---------------------------------------------------------------------------

def get_model_paths(model_dir=None, model_fn="model.bin", vocab_fn="vocab.txt", period_len=None):
    if model_dir is None:
        model_dir = PATH_MODELS
    results = []
    for root, dirs, fns in os.walk(model_dir):
        if model_fn not in fns:
            continue
        parts = root.replace(model_dir, "").strip("/").split("/")
        if len(parts) >= 3 and "run_" in parts[-1]:
            corpus, period, run = parts[-3], parts[-2], parts[-1]
        elif len(parts) >= 2:
            corpus, period = parts[-2], parts[-1]
            run = None
        else:
            continue
        dx = {
            "corpus": corpus,
            "period_start": period.split("-")[0],
            "period_end": period.split("-")[-1],
            "path": os.path.join(root, model_fn),
            "path_vocab": os.path.join(root, vocab_fn),
        }
        if run is not None:
            dx["run"] = run
        if period_len:
            try:
                if int(dx["period_end"]) - int(dx["period_start"]) != period_len:
                    continue
            except ValueError:
                continue
        results.append(dx)
    return results


def load_model(path, path_vocab=None, min_count=None):
    if path.endswith(".bin") and os.path.exists(path):
        model = gensim.models.Word2Vec.load(path, mmap="r")
    elif path.endswith(".txt.gz") and os.path.exists(path):
        if path_vocab and os.path.exists(path_vocab):
            model = gensim.models.KeyedVectors.load_word2vec_format(path, path_vocab)
        else:
            model = gensim.models.KeyedVectors.load_word2vec_format(path)
    else:
        return None
    if min_count:
        _filter_model(model, min_count)
    return model


def _filter_model(model, min_count):
    kv = model.wv if hasattr(model, "wv") else model
    words_ok = {w for w in kv.key_to_index if kv.get_vecattr(w, "count") >= min_count}
    _restrict_keyed_vectors(kv, words_ok)


def _restrict_keyed_vectors(kv, keep_words):
    """Remove words not in keep_words from a KeyedVectors instance (gensim 4.x)."""
    new_vectors = []
    new_index = []
    for word in kv.index_to_key:
        if word in keep_words:
            new_index.append(word)
            new_vectors.append(kv[word])
    kv.index_to_key = new_index
    kv.key_to_index = {w: i for i, w in enumerate(new_index)}
    kv.vectors = np.array(new_vectors)
    if hasattr(kv, "vectors_norm"):
        kv.vectors_norm = None  # will be recomputed on next access


# ---------------------------------------------------------------------------
# Vector field computation
# ---------------------------------------------------------------------------

def get_centroid(kv, words):
    vecs = [kv[w] for w in words if w in kv]
    return np.mean(vecs, axis=0) if vecs else None


def compute_contrast_vector(kv, words_pos, words_neg=None):
    centroid = get_centroid(kv, words_pos)
    if centroid is None:
        return None
    if words_neg:
        neg = get_centroid(kv, words_neg)
        if neg is not None:
            return centroid - neg
    return centroid


def get_fieldvecs_in_model(model, contrasts):
    kv = model.wv if hasattr(model, "wv") else model
    field2vec = {}
    for cdx in contrasts:
        key = f"{cdx['contrast']}.{cdx['source']}"
        vec = compute_contrast_vector(kv, cdx["pos"], cdx["neg"])
        if vec is not None:
            field2vec[key] = vec
    return field2vec


def compute_vec2vec_dists(x2vec, y2vec):
    """Compute cosine distances between two sets of vectors."""
    from scipy.spatial.distance import cosine
    rows = []
    for x, xv in x2vec.items():
        for y, yv in y2vec.items():
            try:
                rows.append({"word": x, "field": y, "dist": cosine(xv, yv)})
            except Exception:
                continue
    return pd.DataFrame(rows).pivot(index="word", columns="field", values="dist")


# ---------------------------------------------------------------------------
# Generate vector-based norms
# ---------------------------------------------------------------------------

def _gen_vecnorms_for_model(pathd):
    model = load_model(pathd["path"], pathd.get("path_vocab"), min_count=MODEL_MIN_COUNT)
    if model is None:
        return []
    kv = model.wv if hasattr(model, "wv") else model
    field2vec = get_fieldvecs_in_model(model, get_origcontrasts())
    word2vec = {w: kv[w] for w in kv.index_to_key}
    dfdist = compute_vec2vec_dists(word2vec, field2vec)
    norms = []
    for col in dfdist.columns:
        _add_series_to_norms(dfdist[col], source=col, norms=norms)
    return norms


def gen_vecnorms(bin_year_by=MODEL_PERIOD_LEN, num_proc=1):
    """Generate vector-based word norms aggregated by time period."""
    def periodize(y):
        y = int(y)
        if bin_year_by == 100:
            return f"C{(y // 100) + 1}"
        elif bin_year_by == 50:
            return f"C{(y // 100) + 1}{'e' if int(str(y)[2]) < 5 else 'l'}"
        return str(y // bin_year_by * bin_year_by)

    paths_df = pd.DataFrame(get_model_paths())
    paths_df["period"] = paths_df["period_start"].apply(periodize)

    word2field2z = defaultdict(dict)
    for period, pdf in sorted(paths_df.groupby("period")):
        vecnorm_rows = []
        for (corpus, ps, pe), cdf in sorted(pdf.groupby(["corpus", "period_start", "period_end"])):
            rows = []
            for pathd in cdf.to_dict("records"):
                rows.extend(_gen_vecnorms_for_model(pathd))
            if rows:
                cdf_norms = pd.DataFrame(rows).groupby(["word", "source"]).median().reset_index()
                cdf_norms["corpus"] = corpus
                vecnorm_rows.append(cdf_norms)
        if vecnorm_rows:
            newdf = pd.concat(vecnorm_rows)
            newdf = newdf.groupby(["word", "source"]).median().reset_index()
            for _, row in newdf.iterrows():
                word, field, z = row["word"], row["source"], row["z"]
                if not word or not word[0].isalpha():
                    continue
                word2field2z[word][f"{field}.{period}"] = z

    rows = [{"word": w, **fields} for w, fields in word2field2z.items()]
    df = pd.DataFrame(rows).set_index("word")
    os.makedirs(os.path.dirname(PATH_VECNORMS), exist_ok=True)
    df.to_csv(PATH_VECNORMS)
