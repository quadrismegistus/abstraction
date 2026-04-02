"""
DuckDB database for the web app.

Reads metadata from LLTK's DuckDB (~/lltk_data/data/metadb.duckdb) and
scores from CSV files (data/scores/v8-raw/), joining them into a single
'texts' view.

On first run (or when score CSVs are newer than the scores DB), loads
score CSVs into a local DuckDB file. LLTK metadata is read directly
via ATTACH — no copying.
"""

import os

import duckdb

from ..config import PATH_DATA, SCORES_DIR


SCORES_DB_FILENAME = "scores.duckdb"
SCORES_VERSION = "v8-raw"
LLTK_DB_PATH = os.path.expanduser("~/lltk_data/data/metadb.duckdb")

import threading

_local = threading.local()


def _scores_db_path():
    return os.path.join(PATH_DATA, SCORES_DB_FILENAME)


def _scores_dir():
    return os.path.join(SCORES_DIR, SCORES_VERSION)


def get_connection():
    """Get a per-thread DuckDB connection with scores + LLTK metadata.

    Each thread gets its own connection (DuckDB connections are not thread-safe).
    The LLTK DB is attached read-only for JOIN queries.
    """
    conn = getattr(_local, 'conn', None)
    if conn is None:
        conn = _build_connection()
        _local.conn = conn
    return conn


def _build_connection():
    """Create a DuckDB connection with scores + metadata."""
    db_path = _scores_db_path()
    conn = duckdb.connect(db_path, read_only=True)

    # Attach LLTK's metadata DB read-only
    if os.path.exists(LLTK_DB_PATH):
        try:
            conn.execute(f"ATTACH '{LLTK_DB_PATH}' AS lltk (READ_ONLY)")
        except duckdb.BinderException:
            pass  # already attached

    # Create the joined view
    try:
        conn.execute("""
            CREATE OR REPLACE TEMP VIEW texts AS
            SELECT
                s.id,
                s.corpus_name,
                m.title,
                m.author,
                m.year,
                m.genre,
                m.genre_raw,
                s.* EXCLUDE (id, corpus_name)
            FROM scores s
            LEFT JOIN lltk.texts m
                ON s.corpus_name = m.corpus
                AND s.id_normalized = m.id
        """)
    except Exception as e:
        print(f"[app] Could not create texts view: {e}")

    return conn


def _is_stale(db_path):
    """Check if scores DB needs rebuilding."""
    if not os.path.exists(db_path):
        return True
    if os.path.getsize(db_path) == 0:
        return True
    try:
        conn = duckdb.connect(db_path, read_only=True)
        conn.execute("SELECT 1 FROM scores LIMIT 1")
        conn.close()
    except Exception:
        return True
    db_mtime = os.path.getmtime(db_path)
    scores_dir = _scores_dir()
    if not os.path.isdir(scores_dir):
        return False
    for fn in os.listdir(scores_dir):
        if fn.endswith(".csv"):
            if os.path.getmtime(os.path.join(scores_dir, fn)) > db_mtime:
                return True
    return False


def init_db(db_path=None):
    """Build the scores DuckDB if stale or missing."""
    if db_path is None:
        db_path = _scores_db_path()

    if not _is_stale(db_path):
        print(f"[app] Scores database is fresh: {db_path}")
        return

    print(f"[app] Building scores database from {SCORES_VERSION} CSVs...")

    import pandas as pd
    from ..analysis import EXCLUDE_CORPORA

    scores_dir = _scores_dir()
    if not os.path.isdir(scores_dir):
        print(f"[app] No scores directory: {scores_dir}")
        return

    # Normalize hathi IDs to match LLTK's canonical form
    try:
        import sys
        sys.path.insert(0, os.path.expanduser("~/github/lltk"))
        from lltk.corpus.hathi.hathi import hathi_id_normalize
    except ImportError:
        hathi_id_normalize = None

    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = duckdb.connect(db_path)

    all_dfs = []
    from tqdm import tqdm
    csv_files = sorted(fn for fn in os.listdir(scores_dir) if fn.endswith(".csv"))
    for fn in tqdm(csv_files, desc="Loading scores"):
        corpus_name = fn.removesuffix(".csv")
        if corpus_name in EXCLUDE_CORPORA:
            continue
        path = os.path.join(scores_dir, fn)
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"  Skipping {corpus_name}: {e}")
            continue
        df["corpus_name"] = corpus_name
        df["id"] = df["id"].astype(str)

        # Normalize IDs to match LLTK canonical form
        if hathi_id_normalize and corpus_name.startswith("hathi"):
            df["id_normalized"] = df["id"].apply(hathi_id_normalize)
        elif corpus_name == "chicago":
            # Chicago: freqs use bare numbers, LLTK uses zero-padded 8-digit
            df["id_normalized"] = df["id"].apply(lambda x: str(x).zfill(8))
        else:
            df["id_normalized"] = df["id"]

        all_dfs.append(df)

    if not all_dfs:
        print("[app] No score data found.")
        conn.close()
        return

    combined = pd.concat(all_dfs, ignore_index=True)
    conn.execute("CREATE TABLE scores AS SELECT * FROM combined")
    conn.execute("CREATE INDEX idx_scores_corpus ON scores (corpus_name)")
    conn.execute("CREATE INDEX idx_scores_id ON scores (id_normalized)")

    n_rows = conn.execute("SELECT COUNT(*) FROM scores").fetchone()[0]
    n_corpora = conn.execute("SELECT COUNT(DISTINCT corpus_name) FROM scores").fetchone()[0]
    conn.close()

    print(f"[app] Scores database ready: {n_rows:,} texts from {n_corpora} corpora")
