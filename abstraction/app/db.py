"""
SQLite database for pre-joined scores + metadata.

On first run (or when score CSVs are newer than the DB), loads all scored
corpora via load_all_scored(), writes to a single SQLite table with indexes.
Subsequent boots skip the build step.
"""

import os
import sqlite3

from ..config import PATH_DATA, SCORES_DIR


DB_FILENAME = "app.db"
SCORES_VERSION = "v8-raw"


def get_db_path():
    return os.path.join(PATH_DATA, DB_FILENAME)


def get_connection(db_path=None):
    if db_path is None:
        db_path = get_db_path()
    return sqlite3.connect(db_path)


def _scores_dir():
    return os.path.join(SCORES_DIR, SCORES_VERSION)


def _is_stale(db_path):
    """Check if DB is missing, empty, or older than score CSVs or analysis.py."""
    if not os.path.exists(db_path):
        return True
    if os.path.getsize(db_path) == 0:
        return True
    # Verify the texts table exists
    try:
        conn = sqlite3.connect(db_path)
        conn.execute("SELECT 1 FROM texts LIMIT 1")
        conn.close()
    except Exception:
        return True
    db_mtime = os.path.getmtime(db_path)

    # Check if analysis.py changed (genre constants, exclusions, etc.)
    analysis_py = os.path.join(os.path.dirname(os.path.dirname(__file__)), "analysis.py")
    if os.path.exists(analysis_py) and os.path.getmtime(analysis_py) > db_mtime:
        return True

    scores_dir = _scores_dir()
    if not os.path.isdir(scores_dir):
        return False
    for fn in os.listdir(scores_dir):
        if fn.endswith(".csv"):
            csv_mtime = os.path.getmtime(os.path.join(scores_dir, fn))
            if csv_mtime > db_mtime:
                return True
    return False


def init_db(db_path=None):
    """Build the SQLite database if it's stale or missing."""
    if db_path is None:
        db_path = get_db_path()

    if not _is_stale(db_path):
        print(f"[app] SQLite database is fresh: {db_path}")
        return

    print(f"[app] Building SQLite database from {SCORES_VERSION} scores...")

    # Import here to avoid circular imports and slow startup when DB is fresh
    from ..analysis import load_all_scored

    df = load_all_scored(version=SCORES_VERSION)
    if df.empty:
        print("[app] WARNING: No scored data found.")
        return

    # Ensure year is numeric
    if "year" in df.columns:
        import pandas as pd
        df["year"] = pd.to_numeric(df["year"], errors="coerce")

    # Keep only the columns we need: id, metadata, corpus_name, and norm scores
    keep_cols = ["id", "corpus_name", "year", "author", "title", "genre_harmonized"]
    norm_cols = [c for c in df.columns if c.startswith("Abs-Conc.")]
    keep_cols = [c for c in keep_cols if c in df.columns] + norm_cols
    # Deduplicate column names (some corpora have overlapping metadata fields)
    seen = set()
    unique_cols = []
    for c in keep_cols:
        if c not in seen:
            seen.add(c)
            unique_cols.append(c)
    df = df[unique_cols]

    # Write to SQLite
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    # Remove old DB before writing
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    df.to_sql("texts", conn, index=False, if_exists="replace")

    # Create indexes for fast filtering
    conn.execute("CREATE INDEX IF NOT EXISTS idx_year ON texts (year)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_corpus ON texts (corpus_name)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_genre ON texts (genre_harmonized)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_corpus_year ON texts (corpus_name, year)")
    conn.commit()

    n_rows = conn.execute("SELECT COUNT(*) FROM texts").fetchone()[0]
    n_corpora = conn.execute("SELECT COUNT(DISTINCT corpus_name) FROM texts").fetchone()[0]
    conn.close()

    print(f"[app] SQLite database ready: {n_rows:,} texts from {n_corpora} corpora")
