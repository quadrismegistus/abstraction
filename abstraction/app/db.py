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

    # Stale if the canonical scores.duckdb is newer
    try:
        from ..config import PATH_SCORES_DB
        if os.path.exists(PATH_SCORES_DB) and os.path.getmtime(PATH_SCORES_DB) > db_mtime:
            return True
    except Exception:
        pass

    # Also stale if any legacy CSV is newer
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

    print(f"[app] Building scores database...")

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

    # Arcs sourced from the new scores.duckdb (Phase 2 pipeline).
    # Configuration: arc_corpus → lang.
    NEW_PIPELINE_ARCS = {
        "arc_fiction":    {"lang": "en"},
        "arc_fiction_fr": {"lang": "fr"},
    }

    try:
        from ..aggregate import get_arc_scores
        from ..config import PATH_SCORES_DB
        has_scores_db = os.path.exists(PATH_SCORES_DB)
    except Exception as e:
        print(f"  New pipeline unavailable ({e}); falling back to CSVs only")
        has_scores_db = False

    new_arc_names = set()
    if has_scores_db:
        print(f"  Aggregating {len(NEW_PIPELINE_ARCS)} arcs from scores.duckdb...")
        for arc_name, cfg in NEW_PIPELINE_ARCS.items():
            try:
                df = get_arc_scores(
                    arc_name, lang=cfg["lang"],
                    dedup="within_lang_group",
                    cross_lang_arc=cfg["cross_lang_arc"],
                )
                df["arc_corpus"] = arc_name
                df["source_corpus"] = arc_name  # no per-text corpus here; aggregation is lossy
                all_dfs.append(df)
                new_arc_names.add(arc_name)
                print(f"    {arc_name}: {len(df):,} reps")
            except Exception as e:
                print(f"    {arc_name}: FAILED ({e})")

    # Remaining arcs: load legacy CSVs (arcs we haven't re-scored yet).
    arc_files = sorted(
        fn for fn in os.listdir(scores_dir)
        if fn.startswith("arc_") and fn.endswith(".csv")
        and fn.removesuffix(".csv") not in new_arc_names
    )
    regular_files = sorted(fn for fn in os.listdir(scores_dir)
                           if fn.endswith(".csv") and not fn.startswith("arc_"))

    if arc_files:
        print(f"  Loading {len(arc_files)} legacy arc corpus CSVs...")
        for fn in tqdm(arc_files, desc="  Legacy arcs"):
            arc_name = fn.removesuffix(".csv")
            path = os.path.join(scores_dir, fn)
            try:
                df = pd.read_csv(path, dtype={"_id": str, "source_corpus": str})
            except Exception as e:
                print(f"    Skipping {fn}: {e}")
                continue
            df["arc_corpus"] = arc_name
            if "_id" not in df.columns:
                continue
            if "source_corpus" not in df.columns:
                df["source_corpus"] = arc_name
            all_dfs.append(df)

    if regular_files and not arc_files and not all_dfs:
        print(f"  No arc data; loading {len(regular_files)} regular CSVs as fallback...")
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
