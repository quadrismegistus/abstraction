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
_lltk_snapshot_lock = threading.Lock()


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


def _db_attached(conn, alias: str) -> bool:
    """True if `alias` appears in the conn's attached-database list."""
    try:
        row = conn.execute(
            "SELECT 1 FROM duckdb_databases() WHERE database_name = ?", [alias]
        ).fetchone()
        return bool(row)
    except Exception:
        return False


def _attach_lltk_snapshot(conn):
    """Copy LLTK's DB to a temp file once per process and attach it.

    Used when the canonical file is already held by another conn in this
    process (DuckDB doesn't always allow a second same-process ATTACH of
    the same file, and silently succeeds-then-missing in some versions).
    Snapshot is a separate file so no lock conflict is possible.

    Thread-safe: multiple concurrent first-callers will serialize on the
    copy, and the global path is only published after copy completes, so
    threads that see the path know the file is fully written.
    """
    global _lltk_snapshot_path
    with _lltk_snapshot_lock:
        if _lltk_snapshot_path is None:
            print("[app] Snapshotting LLTK metadb for per-thread connections...", flush=True)
            tmpdir = tempfile.mkdtemp(prefix="lltk_ro_app_")
            target = os.path.join(tmpdir, "metadb.duckdb")
            shutil.copy2(LLTK_DB_PATH, target)
            # Only publish path AFTER copy completes, so racing readers never
            # attach a partially-written file.
            _lltk_snapshot_path = target
    if not _db_attached(conn, "lltk"):
        try:
            conn.execute(f"ATTACH '{_lltk_snapshot_path}' AS lltk (READ_ONLY)")
        except Exception as e:
            print(f"[app] snapshot attach failed: {e}", flush=True)
    return conn


def _build_connection():
    """Create a DuckDB connection with scores + metadata."""
    db_path = _scores_db_path()
    conn = duckdb.connect(db_path, read_only=True)

    # Attach LLTK metadb. Try the canonical file first; if it's locked (or if
    # the attach appears to succeed but the alias doesn't actually get
    # registered — a subtle DuckDB behaviour when the same file is already
    # attached by a different conn in the same process), fall back to a
    # snapshot copy.
    if os.path.exists(LLTK_DB_PATH):
        try:
            conn.execute(f"ATTACH '{LLTK_DB_PATH}' AS lltk (READ_ONLY)")
        except Exception:
            pass
        if not _db_attached(conn, "lltk"):
            _attach_lltk_snapshot(conn)

    # Attach matches DB for match group lookups
    if os.path.exists(LLTK_MATCHES_DB_PATH):
        try:
            conn.execute(f"ATTACH '{LLTK_MATCHES_DB_PATH}' AS matchdb (READ_ONLY)")
        except Exception:
            pass  # optional — routes that don't use it still work

    # Create joined views: one per dedup mode (scores → texts, scores_rep → texts_rep).
    # Routes pick which view to query based on the `dedup` query param.
    view_to_table = [("texts", "scores"), ("texts_rep", "scores_rep")]
    existing_tables = {
        r[0] for r in conn.execute(
            "SELECT table_name FROM information_schema.tables"
        ).fetchall()
    }
    for view_name, table_name in view_to_table:
        if table_name not in existing_tables:
            continue  # table hasn't been built yet (e.g. legacy-only install)
        try:
            conn.execute(f"""
                CREATE OR REPLACE TEMP VIEW {view_name} AS
                SELECT
                    s._id,
                    -- Per-rep source corpus from LLTK (ecco, chadwyck, ...).
                    -- Falls back to the arc-level label stored in the scores
                    -- table if LLTK doesn't have a match (shouldn't happen
                    -- for any rep we scored, but safe).
                    COALESCE(m.corpus, s.source_corpus) AS corpus_name,
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
                FROM {table_name} s
                LEFT JOIN lltk.texts m ON s._id = m._id
            """)
        except Exception as e:
            print(f"[app] Could not create {view_name} view: {e}")

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
    # All arcs now scored 1:1 in scores_en / scores_fr — aggregation happens
    # here at load time via get_arc_scores.
    NEW_PIPELINE_ARCS = {
        "arc_fiction":    {"lang": "en"},
        "arc_fiction_fr": {"lang": "fr"},
        "arc_poetry":     {"lang": "en"},
        "arc_biography":  {"lang": "en"},
        "arc_essays":     {"lang": "en"},
        "arc_periodical": {"lang": "en"},
        "arc_sermons":    {"lang": "en"},
    }

    try:
        from ..aggregate import get_arc_scores
        from ..config import PATH_SCORES_DB
        has_scores_db = os.path.exists(PATH_SCORES_DB)
    except Exception as e:
        print(f"  New pipeline unavailable ({e}); falling back to CSVs only")
        has_scores_db = False

    # Build both dedup modes so the UI can toggle between them without a rebuild.
    # within_lang_group → scores table (default, match-group averaged scores)
    # rep_only          → scores_rep table (each rep's own raw per-text score)
    DEDUP_TABLES = [
        ("scores", "within_lang_group"),
        ("scores_rep", "rep_only"),
    ]

    new_arc_names = set()
    arc_files = sorted(
        fn for fn in os.listdir(scores_dir)
        if fn.startswith("arc_") and fn.endswith(".csv")
    )
    regular_files = sorted(fn for fn in os.listdir(scores_dir)
                           if fn.endswith(".csv") and not fn.startswith("arc_"))

    for table_name, dedup in DEDUP_TABLES:
        if not has_scores_db:
            break
        print(f"\n  Building {table_name} (dedup={dedup})...")
        all_dfs: list = []
        for arc_name, cfg in NEW_PIPELINE_ARCS.items():
            try:
                df = get_arc_scores(
                    arc_name, lang=cfg["lang"],
                    dedup=dedup,
                )
                df["arc_corpus"] = arc_name
                df["source_corpus"] = arc_name
                all_dfs.append(df)
                if table_name == "scores":
                    new_arc_names.add(arc_name)
                print(f"    {arc_name}: {len(df):,} reps")
            except Exception as e:
                print(f"    {arc_name}: FAILED ({e})")

        # Legacy CSV fallback only applies to the default (within_lang_group) table —
        # legacy CSVs contained match-group averaged scores, not rep_only.
        if table_name == "scores":
            legacy_files = [
                fn for fn in arc_files
                if fn.removesuffix(".csv") not in new_arc_names
            ]
            if legacy_files:
                print(f"    Loading {len(legacy_files)} legacy arc CSVs...")
                for fn in tqdm(legacy_files, desc="    Legacy arcs"):
                    arc_name = fn.removesuffix(".csv")
                    path = os.path.join(scores_dir, fn)
                    try:
                        df = pd.read_csv(path, dtype={"_id": str, "source_corpus": str})
                    except Exception as e:
                        print(f"      Skipping {fn}: {e}")
                        continue
                    df["arc_corpus"] = arc_name
                    if "_id" not in df.columns:
                        continue
                    if "source_corpus" not in df.columns:
                        df["source_corpus"] = arc_name
                    all_dfs.append(df)
            if regular_files and not legacy_files and not all_dfs:
                print(f"    No arc data; loading {len(regular_files)} regular CSVs...")
                from .db_compat import load_regular_csvs
                all_dfs = load_regular_csvs(scores_dir, regular_files)

        if not all_dfs:
            print(f"    No data for {table_name}; skipping")
            continue

        combined = pd.concat(all_dfs, ignore_index=True)
        conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM combined")
        conn.execute(f"CREATE INDEX idx_{table_name}_id ON {table_name} (_id)")
        conn.execute(f"CREATE INDEX idx_{table_name}_arc ON {table_name} (arc_corpus)")
        conn.execute(f"CREATE INDEX idx_{table_name}_source ON {table_name} (source_corpus)")
        n_rows = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        print(f"    {table_name}: {n_rows:,} rows written")

    # Final count from default table
    try:
        n_rows = conn.execute("SELECT COUNT(*) FROM scores").fetchone()[0]
        n_arcs = conn.execute("SELECT COUNT(DISTINCT arc_corpus) FROM scores").fetchone()[0]
    except Exception:
        n_rows, n_arcs = 0, 0
    conn.close()

    # Reset thread-local connections so they pick up new DB
    _local.conn = None

    print(f"\n[app] Scores database ready: {n_rows:,} texts from {n_arcs} arc corpora (both dedup modes)")
