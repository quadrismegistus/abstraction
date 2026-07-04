"""Shift-share decomposition of abstractness changes between periods."""

from fastapi import APIRouter, Query
from pydantic import BaseModel

from ..db import get_connection
from ..validation import validate_col

router = APIRouter()

DEFAULT_COL = "Abs-Conc.Median.median"


class DecompRow(BaseModel):
    category: str
    share_early: float
    share_late: float
    mean_early: float
    mean_late: float
    n_early: int
    n_late: int
    composition_effect: float
    within_effect: float
    interaction: float
    total_effect: float


class DecompResult(BaseModel):
    decompose_by: str
    period_early: str
    period_late: str
    overall_mean_early: float
    overall_mean_late: float
    overall_change: float
    total_composition: float
    total_within: float
    total_interaction: float
    rows: list[DecompRow]


class MultiDecompResult(BaseModel):
    results: list[DecompResult]


def _lookup_narrative_form(df, conn):
    """Look up END narrative_form for texts via match groups.

    Returns a Series with values like 'Epistolary', 'First-person',
    'Third-person', or '(unknown)' for texts without END matches.
    """
    import json

    try:
        # Get narrative_form from END texts matched to our texts
        ids = df["_id"].tolist() if "_id" in df.columns else []
        if not ids:
            return "(unknown)"

        # Query: arc texts -> match groups -> END texts with narrative_form.
        # CH requires FINAL to dedupe ReplacingMergeTree reads, and FINAL
        # doesn't accept an inline alias — wrap in subqueries.
        nf_rows = conn.execute("""
            SELECT mg1._id AS arc_id, end_t.meta
            FROM (SELECT _id, group_id FROM lltk.match_groups FINAL) mg1
            INNER JOIN (SELECT _id, group_id FROM lltk.match_groups FINAL) mg2
              ON mg1.group_id = mg2.group_id
            INNER JOIN (SELECT _id, corpus, meta FROM lltk.texts FINAL) end_t
              ON mg2._id = end_t._id
            WHERE end_t.corpus = 'end'
              AND end_t.meta LIKE '%narrative_form%'
        """).fetchall()

        # Build lookup: arc_id -> primary narrative form
        nf_map = {}
        for arc_id, meta in nf_rows:
            if arc_id in nf_map:
                continue  # keep first match
            m = json.loads(meta)
            nf = m.get("narrative_form", "")
            primary = nf.split("|")[0].strip() if nf else ""
            if primary:
                nf_map[arc_id] = primary

        return df["_id"].map(nf_map).fillna("(unknown)")
    except Exception:
        return "(unknown)"


_GENDER_CORPORA = (
    "litlab", "ravengarside", "long_arc_prestige", "end",
    "chicago", "canon_fiction", "clmet", "gildedage",
)

# Cache gender lookup across requests (static data)
_gender_cache: dict[str, str] | None = None


def _build_gender_cache(conn) -> dict[str, str]:
    """Build arc_id -> gender map from source corpora via match groups."""
    import json

    global _gender_cache
    if _gender_cache is not None:
        return _gender_cache

    corpus_list = ", ".join(f"'{c}'" for c in _GENDER_CORPORA)

    # Get gender from source corpora meta JSON
    rows = conn.execute(f"""
        SELECT _id, meta FROM lltk.texts FINAL
        WHERE corpus IN ({corpus_list})
          AND (meta LIKE '%gender%' OR meta LIKE '%author_gender%')
    """).fetchall()

    source_gender = {}
    for _id, meta in rows:
        m = json.loads(meta)
        g = (m.get("author_gender") or m.get("gender") or "").strip().rstrip(".").lower()
        if g in ("m", "male", "man"):
            source_gender[_id] = "Male"
        elif g in ("f", "female", "woman"):
            source_gender[_id] = "Female"

    # Propagate via match groups (FINAL needed on ReplacingMergeTree tables)
    mg_rows = conn.execute(f"""
        SELECT mg1._id AS arc_id, mg2._id AS source_id
        FROM (SELECT _id, group_id FROM lltk.match_groups FINAL) mg1
        INNER JOIN (SELECT _id, group_id FROM lltk.match_groups FINAL) mg2
          ON mg1.group_id = mg2.group_id
        WHERE mg2._id IN (
            SELECT _id FROM lltk.texts FINAL WHERE corpus IN ({corpus_list})
        )
    """).fetchall()

    gender_map = {}
    for arc_id, source_id in mg_rows:
        if arc_id not in gender_map and source_id in source_gender:
            gender_map[arc_id] = source_gender[source_id]

    _gender_cache = gender_map
    return gender_map


def _lookup_author_gender(df, conn):
    """Look up author gender for texts via match groups to corpora with gender metadata."""
    try:
        gender_map = _build_gender_cache(conn)
        return df["_id"].map(gender_map).fillna("(unknown)")
    except Exception:
        return "(unknown)"


def _parse_genre_raw(genre_raw: str | None) -> str:
    """Extract the most salient genre label from genre_raw.

    genre_raw may contain pipes ("Romance | Fiction") and/or commas
    ("Romance, political; Allegory").  Split by both "|" and ",", take
    the first token that isn't just "Fiction", and strip whitespace.
    """
    if not genre_raw or genre_raw == "":
        return "(unknown)"
    import re
    parts = [p.strip() for p in re.split(r"[|,]", str(genre_raw))]
    for p in parts:
        if p and p != "Fiction" and p != "":
            return p
    return parts[0] if parts else "(unknown)"


def _decompose(df_early, df_late, cat_col, min_texts=5):
    """Run shift-share decomposition on two DataFrames with _score and cat_col."""
    import numpy as np

    if len(df_early) == 0 or len(df_late) == 0:
        return None

    overall_early = df_early["_score"].mean()
    overall_late = df_late["_score"].mean()
    overall_change = overall_late - overall_early

    n_early_total = len(df_early)
    n_late_total = len(df_late)

    def genre_stats(sub, total):
        g = sub.groupby(cat_col).agg(
            mean=("_score", "mean"),
            n=("_score", "count"),
        ).reset_index()
        g["share"] = g["n"] / total
        return g

    ge = genre_stats(df_early, n_early_total).rename(
        columns={"mean": "mean_early", "n": "n_early", "share": "share_early"})
    gl = genre_stats(df_late, n_late_total).rename(
        columns={"mean": "mean_late", "n": "n_late", "share": "share_late"})

    merged = ge.merge(gl, on=cat_col, how="outer").fillna(0)
    merged = merged[(merged["n_early"] >= min_texts) | (merged["n_late"] >= min_texts)]

    rows = []
    for _, r in merged.iterrows():
        d_share = r["share_late"] - r["share_early"]
        d_mean = r["mean_late"] - r["mean_early"]

        comp = d_share * r["mean_early"]
        within = r["share_early"] * d_mean
        interaction = d_share * d_mean

        rows.append(DecompRow(
            category=str(r[cat_col]),
            share_early=round(r["share_early"], 4),
            share_late=round(r["share_late"], 4),
            mean_early=round(r["mean_early"], 4),
            mean_late=round(r["mean_late"], 4),
            n_early=int(r["n_early"]),
            n_late=int(r["n_late"]),
            composition_effect=round(comp, 6),
            within_effect=round(within, 6),
            interaction=round(interaction, 6),
            total_effect=round(comp + within + interaction, 6),
        ))

    rows.sort(key=lambda r: abs(r.total_effect), reverse=True)

    return DecompResult(
        decompose_by=cat_col,
        period_early="",  # filled by caller
        period_late="",
        overall_mean_early=round(overall_early, 4),
        overall_mean_late=round(overall_late, 4),
        overall_change=round(overall_change, 4),
        total_composition=round(sum(r.composition_effect for r in rows), 4),
        total_within=round(sum(r.within_effect for r in rows), 4),
        total_interaction=round(sum(r.interaction for r in rows), 4),
        rows=rows,
    )


@router.get("/shift-share", response_model=MultiDecompResult)
def shift_share(
    col: str = DEFAULT_COL,
    genre: str = "arc_fiction",
    year_early_min: int = 1640,
    year_early_max: int = 1680,
    year_late_min: int = 1740,
    year_late_max: int = 1780,
    corpus: list[str] = Query(default=[]),
    invert: bool = True,
    period_matched: bool = True,
    min_texts: int = 5,
    is_translated: str | None = None,
    corpus_corrected: bool = False,
):
    """Shift-share decomposition by genre_raw, corpus, and is_translated.

    Returns three decompositions simultaneously:
    - By genre_raw: which specific genres drove the change?
    - By corpus: which corpora drove the change?
    - By is_translated: how much is translation vs original?
    """
    import pandas as pd
    import numpy as np
    from ...analysis import assign_period_score

    col = validate_col(col)
    conn = get_connection()

    is_arc = genre.startswith("arc_")
    filter_col = "arc_corpus" if is_arc else "genre"

    corpus_filter = ""
    corpus_params: list = []
    if corpus and len(corpus) > 0:
        placeholders = ",".join("?" for _ in corpus)
        corpus_filter = f" AND corpus_name IN ({placeholders})"
        corpus_params = list(corpus)

    translated_filter = ""
    if is_translated == "true":
        translated_filter = " AND is_translated = 1"
    elif is_translated == "false":
        translated_filter = " AND (is_translated IS NULL OR is_translated = 0)"

    if period_matched:
        col_parts = col.split(".")
        source = col_parts[1] if len(col_parts) >= 2 else "Median"
        score_cols = [r[0] for r in conn.execute("DESCRIBE scores").fetchall()
                      if r[0].startswith(f"Abs-Conc.{source}.")]
        col_sql = ", ".join(f'"{c}"' for c in score_cols)

        df = conn.execute(f"""
            SELECT _id, year, genre_raw, corpus_name, is_translated, n_words,
                   genre_enriched_source, author_norm, title, {col_sql}
            FROM texts
            WHERE {filter_col} = ? AND year IS NOT NULL
              AND ((year >= {year_early_min} AND year <= {year_early_max})
                OR (year >= {year_late_min} AND year <= {year_late_max}))
              {corpus_filter}{translated_filter}
        """, [genre, *corpus_params]).fetchdf()

        if len(df) == 0:
            from fastapi import HTTPException
            raise HTTPException(404, "No data")

        df = assign_period_score(df, source=source)
        df["_score"] = df["period_score"]
        df = df.dropna(subset=["_score"])
    else:
        df = conn.execute(f"""
            SELECT _id, year, genre_raw, corpus_name, is_translated, n_words,
                   genre_enriched_source, author_norm, title, "{col}" as _score
            FROM texts
            WHERE {filter_col} = ? AND year IS NOT NULL AND "{col}" IS NOT NULL
              AND ((year >= {year_early_min} AND year <= {year_early_max})
                OR (year >= {year_late_min} AND year <= {year_late_max}))
              {corpus_filter}{translated_filter}
        """, [genre, *corpus_params]).fetchdf()

    if len(df) == 0:
        from fastapi import HTTPException
        raise HTTPException(404, "No data")

    # Apply corpus bias correction before sign flip
    if corpus_corrected:
        from ...corpus_correction import load_all_corpus_bias
        bias = load_all_corpus_bias()
        if bias:
            coefficients = bias.get("coefficients", {})
            df["_score"] = df["_score"] - df["corpus_name"].map(coefficients).fillna(0.0)

    sign = -1.0 if invert else 1.0
    df["_score"] = df["_score"] * sign

    # Parse genre_raw
    df["_genre_raw"] = df["genre_raw"].apply(_parse_genre_raw)

    # Parse is_translated (CH returns Nullable(UInt8), i.e. 0/1/NA) to readable labels
    df["_translated"] = df["is_translated"].apply(
        lambda x: "Translated" if x == 1 else "Original")

    # Bin text length
    df["_length_bin"] = pd.cut(
        df["n_words"].fillna(0),
        bins=[0, 10_000, 30_000, 60_000, 100_000, float("inf")],
        labels=["<10K", "10-30K", "30-60K", "60-100K", ">100K"],
    ).astype(str)

    # Genre x length interaction (short vs long at 30K threshold)
    length_label = df["n_words"].fillna(0).apply(lambda w: "short" if w < 30_000 else "long")
    df["_genre_length"] = df["_genre_raw"] + " (" + length_label + ")"

    # Genre enrichment source (how was this text classified as fiction?)
    df["_genre_source"] = df["genre_enriched_source"].fillna("corpus")

    # Author productivity bins
    author_counts = df["author_norm"].value_counts()
    df["_author_productivity"] = df["author_norm"].map(author_counts).fillna(0).apply(
        lambda n: "prolific (10+)" if n >= 10 else "moderate (3-9)" if n >= 3 else "single (1-2)"
    )

    # Epistolary detection: genre_raw label OR title keywords
    def _detect_epistolary(row):
        gr = str(row.get("genre_raw") or "").lower()
        title = str(row.get("title") or "").lower()
        if "epistol" in gr or "epistol" in title:
            return "epistolary"
        if "letter" in gr:
            return "epistolary"
        if "letter" in title and ("novel" in gr or "fiction" in gr or gr == ""):
            return "epistolary"
        return "non-epistolary"

    df["_epistolary"] = df.apply(_detect_epistolary, axis=1)

    # Narrative form from END (Early Novels Database) via match groups
    df["_narrative_form"] = _lookup_narrative_form(df, conn)

    # Author gender from corpora with gender metadata, via match groups
    df["_author_gender"] = _lookup_author_gender(df, conn)

    # Split into early/late
    early = df[(df["year"] >= year_early_min) & (df["year"] <= year_early_max)]
    late = df[(df["year"] >= year_late_min) & (df["year"] <= year_late_max)]

    if len(early) == 0 or len(late) == 0:
        from fastapi import HTTPException
        raise HTTPException(404, "No data in one of the periods")

    period_early = f"{year_early_min}-{year_early_max}"
    period_late = f"{year_late_min}-{year_late_max}"

    results = []

    # Decompose by genre_raw
    r = _decompose(early, late, "_genre_raw", min_texts)
    if r:
        r.decompose_by = "genre_raw"
        r.period_early = period_early
        r.period_late = period_late
        results.append(r)

    # Decompose by corpus
    r = _decompose(early, late, "corpus_name", min_texts)
    if r:
        r.decompose_by = "corpus"
        r.period_early = period_early
        r.period_late = period_late
        results.append(r)

    # Decompose by is_translated
    r = _decompose(early, late, "_translated", min_texts)
    if r:
        r.decompose_by = "is_translated"
        r.period_early = period_early
        r.period_late = period_late
        results.append(r)

    # Decompose by text length
    r = _decompose(early, late, "_length_bin", min_texts)
    if r:
        r.decompose_by = "text_length"
        r.period_early = period_early
        r.period_late = period_late
        results.append(r)

    # Decompose by genre x length
    r = _decompose(early, late, "_genre_length", min_texts)
    if r:
        r.decompose_by = "genre_raw x length"
        r.period_early = period_early
        r.period_late = period_late
        results.append(r)

    # Decompose by epistolary vs non-epistolary
    r = _decompose(early, late, "_epistolary", min_texts)
    if r:
        r.decompose_by = "epistolary"
        r.period_early = period_early
        r.period_late = period_late
        results.append(r)

    # Decompose by narrative form (END-classified texts only)
    early_nf = early[early["_narrative_form"] != "(unknown)"]
    late_nf = late[late["_narrative_form"] != "(unknown)"]
    r = _decompose(early_nf, late_nf, "_narrative_form", min_texts)
    if r:
        r.decompose_by = "narrative_form (END)"
        r.period_early = period_early
        r.period_late = period_late
        results.append(r)

    # Decompose by genre enrichment source
    r = _decompose(early, late, "_genre_source", min_texts)
    if r:
        r.decompose_by = "genre_source"
        r.period_early = period_early
        r.period_late = period_late
        results.append(r)

    # Decompose by author productivity
    r = _decompose(early, late, "_author_productivity", min_texts)
    if r:
        r.decompose_by = "author_productivity"
        r.period_early = period_early
        r.period_late = period_late
        results.append(r)

    # Decompose by author gender (classified texts only)
    early_g = early[early["_author_gender"] != "(unknown)"]
    late_g = late[late["_author_gender"] != "(unknown)"]
    r = _decompose(early_g, late_g, "_author_gender", min_texts)
    if r:
        r.decompose_by = "author_gender"
        r.period_early = period_early
        r.period_late = period_late
        results.append(r)

    # Corpus-controlled genre decomposition:
    # Run genre_raw decomposition within each corpus present in both periods,
    # then weight-average across corpora. Isolates within-corpus genre shifts.
    r = _corpus_controlled_decompose(early, late, min_texts)
    if r:
        r.decompose_by = "genre_raw (corpus-controlled)"
        r.period_early = period_early
        r.period_late = period_late
        results.append(r)

    return MultiDecompResult(results=results)


def _corpus_controlled_decompose(early, late, min_texts=5):
    """Genre decomposition controlling for corpus composition.

    Only uses corpora present in both periods. Runs genre_raw decomposition
    within each such corpus, then averages the effects weighted by corpus size.
    """
    import numpy as np

    # Find corpora present in both periods with enough texts
    early_corpora = set(early["corpus_name"].unique())
    late_corpora = set(late["corpus_name"].unique())
    shared = early_corpora & late_corpora

    if not shared:
        return None

    # Filter to shared corpora
    e = early[early["corpus_name"].isin(shared)]
    l = late[late["corpus_name"].isin(shared)]

    if len(e) < 10 or len(l) < 10:
        return None

    # Weight each corpus by its average share across both periods
    e_total = len(e)
    l_total = len(l)

    # Accumulate weighted effects across corpora
    all_genres = set()
    corpus_results = {}

    for corpus in sorted(shared):
        ce = e[e["corpus_name"] == corpus]
        cl = l[l["corpus_name"] == corpus]
        if len(ce) < min_texts or len(cl) < min_texts:
            continue

        r = _decompose(ce, cl, "_genre_raw", min_texts=1)  # lower threshold within corpus
        if r is None:
            continue

        # Weight = average share of this corpus across both periods
        weight = (len(ce) / e_total + len(cl) / l_total) / 2
        corpus_results[corpus] = (r, weight, len(ce), len(cl))
        for row in r.rows:
            all_genres.add(row.category)

    if not corpus_results:
        return None

    # Aggregate: weighted sum of per-corpus effects
    genre_effects: dict[str, dict] = {}
    total_weight = sum(w for _, w, _, _ in corpus_results.values())

    for corpus, (r, weight, n_e, n_l) in corpus_results.items():
        w = weight / total_weight  # normalize weights
        for row in r.rows:
            g = row.category
            if g not in genre_effects:
                genre_effects[g] = {
                    "composition": 0, "within": 0, "interaction": 0,
                    "n_early": 0, "n_late": 0,
                    "share_early_sum": 0, "share_late_sum": 0,
                    "mean_early_sum": 0, "mean_late_sum": 0,
                    "w_sum": 0,
                }
            ge = genre_effects[g]
            ge["composition"] += row.composition_effect * w
            ge["within"] += row.within_effect * w
            ge["interaction"] += row.interaction * w
            ge["n_early"] += row.n_early
            ge["n_late"] += row.n_late
            ge["share_early_sum"] += row.share_early * w
            ge["share_late_sum"] += row.share_late * w
            ge["mean_early_sum"] += row.mean_early * w if row.n_early > 0 else 0
            ge["mean_late_sum"] += row.mean_late * w if row.n_late > 0 else 0
            ge["w_sum"] += w

    # Build rows
    rows = []
    for g, ge in genre_effects.items():
        ws = ge["w_sum"] or 1
        comp = ge["composition"]
        within = ge["within"]
        interaction = ge["interaction"]
        rows.append(DecompRow(
            category=g,
            share_early=round(ge["share_early_sum"], 4),
            share_late=round(ge["share_late_sum"], 4),
            mean_early=round(ge["mean_early_sum"] / ws, 4) if ge["n_early"] > 0 else 0,
            mean_late=round(ge["mean_late_sum"] / ws, 4) if ge["n_late"] > 0 else 0,
            n_early=ge["n_early"],
            n_late=ge["n_late"],
            composition_effect=round(comp, 6),
            within_effect=round(within, 6),
            interaction=round(interaction, 6),
            total_effect=round(comp + within + interaction, 6),
        ))

    rows.sort(key=lambda r: abs(r.total_effect), reverse=True)

    # Overall stats (from shared-corpus subset only)
    overall_early = e["_score"].mean()
    overall_late = l["_score"].mean()

    n_corpora = len(corpus_results)
    corpus_list = ", ".join(sorted(corpus_results.keys()))

    return DecompResult(
        decompose_by=f"genre_raw (corpus-controlled, {n_corpora} shared corpora: {corpus_list})",
        period_early="",
        period_late="",
        overall_mean_early=round(overall_early, 4),
        overall_mean_late=round(overall_late, 4),
        overall_change=round(overall_late - overall_early, 4),
        total_composition=round(sum(r.composition_effect for r in rows), 4),
        total_within=round(sum(r.within_effect for r in rows), 4),
        total_interaction=round(sum(r.interaction for r in rows), 4),
        rows=rows,
    )
