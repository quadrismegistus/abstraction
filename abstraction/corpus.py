"""
Corpus access layer — replaces lltk dependency.

Expects corpora at PATH_CORPORA/[corpus_name]/ with:
  - metadata.csv (must contain an 'id' column)
  - txt/[text_id].txt
"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

import pandas as pd
from tqdm import tqdm

from .config import PATH_CORPORA


class Corpus:
    def __init__(self, name, root=PATH_CORPORA):
        self.name = name
        # lltk convention: CamelCase name -> snake_case directory
        self.id = _camel_to_snake(name)
        self.path = os.path.join(root, self.id)
        self._metadata = None

    @property
    def metadata(self):
        if self._metadata is None:
            meta_csv = os.path.join(self.path, "metadata.csv")
            meta_xls = os.path.join(self.path, "metadata.xls")
            if os.path.exists(meta_csv):
                self._metadata = pd.read_csv(meta_csv)
            elif os.path.exists(meta_xls):
                self._metadata = pd.read_excel(meta_xls)
            else:
                raise FileNotFoundError(f"No metadata found at {self.path}")
        return self._metadata

    def text_path(self, text_id):
        return os.path.join(self.path, "txt", f"{text_id}.txt")

    def text_paths(self):
        return [
            (row["id"], self.text_path(row["id"]))
            for _, row in self.metadata.iterrows()
        ]

    def read_text(self, text_id):
        path = self.text_path(text_id)
        with open(path, encoding="utf-8", errors="ignore") as f:
            return f.read()


def load_corpus(name, root=PATH_CORPORA):
    return Corpus(name, root=root)


def _camel_to_snake(name):
    import re
    s = re.sub(r"(?<=[a-z0-9])([A-Z])", r"_\1", name)
    return s.lower()


# ---------------------------------------------------------------------------
# Parallel map utilities — replaces lltk.pmap / lltk.pmap_iter
# ---------------------------------------------------------------------------

def pmap(func, items, num_proc=1, desc=None):
    """Parallel map returning a list of results."""
    if num_proc <= 1:
        return [func(item) for item in tqdm(items, desc=desc)]
    with ProcessPoolExecutor(max_workers=num_proc) as pool:
        futures = [pool.submit(func, item) for item in items]
        results = []
        for f in tqdm(as_completed(futures), total=len(futures), desc=desc):
            results.append(f.result())
        return results


# ---------------------------------------------------------------------------
# Hathi EngLit: unpack TSV word frequencies into freqs JSONs
# ---------------------------------------------------------------------------

def _htid_to_freqs_path(htid, freqs_dir):
    """Convert a Hathi htid to a freqs JSON path.

    'nyp.33433067374037' -> freqs_dir/nyp/334/33067374037.json
    'dul1.ark+=13960=t03x9108q' -> freqs_dir/dul1/ark/+=13960=t03x9108q.json
    """
    prefix, rest = htid.split(".", 1)
    subdir = rest[:3]
    fname = rest[3:] + ".json"
    return os.path.join(freqs_dir, prefix, subdir, fname)


def fix_hathi_englit(genres=("fiction", "poetry"), root=PATH_CORPORA):
    """Untar Hathi EngLit TSV archives and convert to freqs JSONs.

    Processes fiction and poetry tar.gz files from hathi_englit/raw/,
    reads each TSV (word[tab]count), and saves as JSON in hathi_englit/freqs/.
    Skips files that already exist.

    Parameters
    ----------
    genres : tuple of str
        Which genres to process (e.g. ("fiction", "poetry", "drama")).
    root : str
        Path to corpora directory.
    """
    import json
    import tarfile

    corpus_dir = os.path.join(root, "hathi_englit")
    raw_dir = os.path.join(corpus_dir, "raw")
    freqs_dir = os.path.join(corpus_dir, "freqs")

    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(f"No raw directory: {raw_dir}")

    # find all tar.gz files for requested genres
    archives = []
    for fn in sorted(os.listdir(raw_dir)):
        if not fn.endswith(".tar.gz"):
            continue
        genre = fn.split("_")[0]
        if genre in genres:
            archives.append(os.path.join(raw_dir, fn))

    print(f"Found {len(archives)} archives for genres: {genres}")

    total_new = 0
    total_skipped = 0
    for archive_path in tqdm(archives, desc="Archives"):
        try:
            with tarfile.open(archive_path, "r:gz") as tar:
                members = [m for m in tar.getmembers() if m.name.endswith(".tsv")]
                for member in tqdm(members, desc=os.path.basename(archive_path),
                                   leave=False, unit="file"):
                    # extract htid from filename: "yale.39002073112030.tsv"
                    basename = os.path.basename(member.name)
                    htid = basename.removesuffix(".tsv")
                    out_path = _htid_to_freqs_path(htid, freqs_dir)

                    if os.path.exists(out_path):
                        total_skipped += 1
                        continue

                    # read TSV: word\tcount
                    f = tar.extractfile(member)
                    if f is None:
                        continue
                    freqs = {}
                    for line in f.read().decode("utf-8", errors="ignore").splitlines():
                        parts = line.split("\t", 1)
                        if len(parts) != 2:
                            continue
                        word, count_str = parts
                        word = word.strip()
                        if not word:
                            continue
                        try:
                            count = int(count_str.strip())
                        except ValueError:
                            continue
                        if count > 0:
                            freqs[word] = freqs.get(word, 0) + count

                    if freqs:
                        os.makedirs(os.path.dirname(out_path), exist_ok=True)
                        with open(out_path, "w") as of:
                            json.dump(freqs, of)
                        total_new += 1
        except Exception as e:
            print(f"  Error processing {archive_path}: {e}")
            continue

    print(f"Done: {total_new} new, {total_skipped} already existed")


# ---------------------------------------------------------------------------
# Freqs coverage check
# ---------------------------------------------------------------------------

def check_freqs_coverage(corpus_name=None, root=PATH_CORPORA):
    """Check how many metadata IDs have corresponding freqs JSON files.

    Parameters
    ----------
    corpus_name : str, optional
        Single corpus to check. If None, checks all corpora with freqs/ dirs.
    root : str
        Path to corpora directory.

    Returns
    -------
    DataFrame with columns: corpus, n_metadata, n_freqs, n_overlap, pct_coverage
    """
    from .scoring import _walk_freqs

    if corpus_name:
        names = [_camel_to_snake(corpus_name) if corpus_name[0].isupper() else corpus_name]
    else:
        names = sorted(d for d in os.listdir(root)
                       if os.path.isdir(os.path.join(root, d, "freqs")))

    rows = []
    for name in tqdm(names, desc="Checking coverage"):
        corpus_dir = os.path.join(root, name)
        freqs_dir = os.path.join(corpus_dir, "freqs")
        if not os.path.isdir(freqs_dir):
            continue

        # load metadata IDs — try multiple columns, keep best overlap
        meta_id_sets = {}
        try:
            c = Corpus(name, root=root)
            meta = c.metadata
            for col in ["id", "htid"]:
                if col in meta.columns:
                    meta_id_sets[col] = set(meta[col].dropna().astype(str))
        except Exception:
            pass
        meta_ids = meta_id_sets.get("id", meta_id_sets.get("htid", set()))

        # collect freqs IDs
        freqs_ids = {tid for tid, _ in _walk_freqs(freqs_dir)}

        # try direct overlap
        overlap = meta_ids & freqs_ids

        # try ID normalizations across all available ID columns
        for id_set in meta_id_sets.values():
            if not id_set:
                continue
            # direct
            direct = id_set & freqs_ids
            if len(direct) > len(overlap):
                overlap = direct
                meta_ids = id_set

            # slash→dot
            freqs_ids_dot = {fid.replace("/", ".", 1) for fid in freqs_ids}
            overlap_dot = id_set & freqs_ids_dot
            if len(overlap_dot) > len(overlap):
                overlap = overlap_dot
                meta_ids = id_set

            # htid→path format: "nyp.334330..." -> "nyp/334/330..."
            as_freqs = set()
            for mid in id_set:
                if "." in mid:
                    prefix, rest = mid.split(".", 1)
                    as_freqs.add(f"{prefix}/{rest[:3]}/{rest[3:]}")
            overlap_htid = as_freqs & freqs_ids
            if len(overlap_htid) > len(overlap):
                overlap = overlap_htid
                meta_ids = id_set

        rows.append({
            "corpus": name,
            "n_metadata": len(meta_ids),
            "n_freqs": len(freqs_ids),
            "n_overlap": len(overlap),
            "pct_coverage": len(overlap) / len(meta_ids) * 100 if meta_ids else 0,
        })

    return pd.DataFrame(rows)


def pmap_iter(func, items, num_proc=1, desc=None):
    """Parallel map yielding results as they complete."""
    if num_proc <= 1:
        for item in tqdm(items, desc=desc):
            yield func(item)
        return
    with ProcessPoolExecutor(max_workers=num_proc) as pool:
        futures = [pool.submit(func, item) for item in items]
        for f in tqdm(as_completed(futures), total=len(futures), desc=desc):
            yield f.result()
