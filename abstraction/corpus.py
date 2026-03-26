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
