"""
DuckDB database for the web app.

Loads arc corpus scores (arc_fiction.csv, arc_poetry.csv, etc.) into a
local DuckDB, JOINed with LLTK's metadata DB for year/author/title.

Arc CSVs use _id (canonical LLTK ID) and source_corpus columns.
No ID normalization needed.
"""

import os
import shutil
import tempfile
import threading

import duckdb

from ..config import PATH_DATA, SCORES_DIR


SCORES_DB_FILENAME = "scores.duckdb"
SCORES_VERSION = "v8-raw"
LLTK_DB_PATH = os.path.expanduser("~/lltk_data/data/metadb.duckdb")
LLTK_MATCHES_DB_PATH = os.path.expanduser("~/lltk_data/data/metadb_matches.duckdb")

_local = threading.local()
_lltk_snapshot_path = None


def _scores_db_path():
    return os.path.join(PATH_DATA, SCORES_DB_FILENAME)


def _scores_dir():
    return os.path.join(SCORES_DIR, SCORES_VERSION)


def get_connection():
    """Get a per-thread DuckDB connection with scores + LLTK metadata."""
    conn = getattr(_local, 'conn', None)
    if conn is None:
        conn = _build_connection()
        _local.conn = conn
    return conn


def _attach_lltk_snapshot(conn):
    """Copy LLTK's DB to a temp file and attach it (avoids write-lock)."""
    global _lltk_snapshot_path
    if _lltk_snapshot_path is None:
        print("[app] LLTK DB locked — copying snapshot...", flush=True)
        tmpdir = tempfile.mkdtemp(prefix="lltk_ro_app_")
        _lltk_snapshot_path = os.path.join(tmpdir, "metadb.duckdb")
        shutil.copy2(LLTK_DB_PATH, _lltk_snapshot_path)
    try:
        conn.execute(f"ATTACH '{_lltk_snapshot_path}' AS lltk (READ_ONLY)")
    except duckdb.BinderException:
        pass
    return conn


def _build_connection():
    """Create a DuckDB connection with scores + metadata."""
    db_path = _scores_db_path()
    conn = duckdb.connect(db_path, read_only=True)

    if os.path.exists(LLTK_DB_PATH):
        try:
            conn.execute(f"ATTACH '{LLTK_DB_PATH}' AS lltk (READ_ONLY)")
        except duckdb.BinderException:
            pass  # already attached
        except duckdb.IOException:
            # DB locked by another process — use a snapshot copy
            conn = _attach_lltk_snapshot(conn)

    # Attach matches DB for match group lookups
    if os.path.exists(LLTK_MATCHES_DB_PATH):
        try:
            conn.execute(f"ATTACH '{LLTK_MATCHES_DB_PATH}' AS matchdb (READ_ONLY)")
        except duckdb.BinderException:
            pass  # already attached
        except duckdb.IOException:
            pass  # locked, skip

    # Create joined view: scores + LLTK metadata
    try:
        conn.execute("""
            CREATE OR REPLACE TEMP VIEW texts AS
            SELECT
                s._id,
                s.source_corpus AS corpus_name,
                s.arc_corpus,
                m.title,
                m.author,
                m.year,
                m.genre,
                m.genre_raw,
                m.genre_enriched_source,
                m.is_translated,
                m.n_words,
                m.author_norm,
                s.* EXCLUDE (_id, source_corpus, arc_corpus)
            FROM scores s
            LEFT JOIN lltk.texts m ON s._id = m._id
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

    scores_dir = _scores_dir()
    if not os.path.isdir(scores_dir):
        print(f"[app] No scores directory: {scores_dir}")
        return

    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = duckdb.connect(db_path)

    all_dfs = []
    from tqdm import tqdm

    # Load arc corpus CSVs (arc_fiction.csv, etc.) — these have _id and source_corpus
    arc_files = sorted(fn for fn in os.listdir(scores_dir)
                       if fn.startswith("arc_") and fn.endswith(".csv"))

    # Load regular corpus CSVs (for non-arc corpora)
    regular_files = sorted(fn for fn in os.listdir(scores_dir)
                           if fn.endswith(".csv") and not fn.startswith("arc_"))

    if arc_files:
        print(f"  Loading {len(arc_files)} arc corpus CSVs...")
        for fn in tqdm(arc_files, desc="  Arc corpora"):
            arc_name = fn.removesuffix(".csv")
            path = os.path.join(scores_dir, fn)
            try:
                df = pd.read_csv(path, dtype={"_id": str, "source_corpus": str})
            except Exception as e:
                print(f"    Skipping {fn}: {e}")
                continue
            df["arc_corpus"] = arc_name
            # Ensure _id and source_corpus exist
            if "_id" not in df.columns:
                continue
            if "source_corpus" not in df.columns:
                df["source_corpus"] = arc_name
            all_dfs.append(df)

    if regular_files and not arc_files:
        # Fallback: load regular CSVs if no arc CSVs exist yet
        print(f"  No arc CSVs found, loading {len(regular_files)} regular CSVs...")
        from .db_compat import load_regular_csvs
        all_dfs = load_regular_csvs(scores_dir, regular_files)

    if not all_dfs:
        print("[app] No score data found.")
        conn.close()
        return

    combined = pd.concat(all_dfs, ignore_index=True)
    conn.execute("CREATE TABLE scores AS SELECT * FROM combined")
    conn.execute("CREATE INDEX idx_scores_id ON scores (_id)")
    conn.execute("CREATE INDEX idx_scores_arc ON scores (arc_corpus)")
    conn.execute("CREATE INDEX idx_scores_source ON scores (source_corpus)")

    n_rows = conn.execute("SELECT COUNT(*) FROM scores").fetchone()[0]
    n_arcs = conn.execute("SELECT COUNT(DISTINCT arc_corpus) FROM scores").fetchone()[0]
    conn.close()

    # Reset thread-local connections so they pick up new DB
    _local.conn = None

    print(f"[app] Scores database ready: {n_rows:,} texts from {n_arcs} arc corpora")
