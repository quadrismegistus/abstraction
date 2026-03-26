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
