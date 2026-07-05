"""
Plotting functions for word norms and fiction-level abstraction trends.
Uses plotnine (ggplot2-style grammar of graphics).
"""

import os
import numpy as np
import pandas as pd
import plotnine as p9
from scipy.stats import zscore

from .config import PATH_FIGS, SOURCES_FOR_PLOTTING, BAD_SOURCES
from .corpus import load_corpus
from .norms import (
    format_norms_as_long, get_allnorms, NORM_SOURCE_ORDER,
)

# Print/export resolution for saved figures. Previously set globally via
# `p9.options.dpi = 300` at import time, which silently affected every
# plotnine figure (including interactive Jupyter previews) for the whole
# process. plotnine's `ggplot.save()` accepts `dpi=` directly, so each
# save call below passes it explicitly instead
# (AUDIT-2026-07-04.md §2.9/§5).
SAVE_DPI = 300

# ---------------------------------------------------------------------------
# Label mappings
# ---------------------------------------------------------------------------

SOURCE_LABELS = {
    "PAV-Conc": "Paivio (1968),\n'Concreteness'",
    "PAV-Imag": "Paivio (1968),\n'Imagery'",
    "MRC-Conc": "MRC (1987),\n'Concreteness'",
    "MRC-Imag": "MRC (1987),\n'Imagery'",
    "MT-Conc": "Brysbaert (2014),\n'Concreteness'",
    "LSN-Imag": "LSN (2017),\n'Visual'",
    "LSN-Hapt": "LSN (2017),\n'Haptic'",
    "Median": "(Empirical median)",
    "median": "(Historical median)",
    "orig": "(Empirical median)",
    "C16": "C16", "C17": "C17", "C18": "C18", "C19": "C19", "C20": "C20",
}

SOURCE_LABELS_FULL = {
    "PAV-Conc.orig": "Paivio (1968),\n'Concreteness'",
    "PAV-Imag.orig": "Paivio (1968),\n'Imagery'",
    "MRC-Conc.orig": "MRC (1987),\n'Concreteness'",
    "MRC-Imag.orig": "MRC (1987),\n'Imagery'",
    "MT-Conc.orig": "Brysbaert (2014),\n'Concreteness'",
    "LSN-Imag.orig": "LSN (2017),\n'Visual'",
    "LSN-Hapt.orig": "LSN (2017),\n'Haptic'",
    "Median.orig": "(Empirical median)",
    "Median.C20": "C20 corpus\n(COHA)",
    "Median.C19": "C19 corpus\n(COHA)",
    "Median.C18": "C18 corpus\n(ECCO-TCP)",
    "Median.C17": "C17 corpus\n(EEBO-TCP)",
    "Median.C16": "C16 corpus\n(EEBO-TCP)",
    "Median.median": "(Historical median)",
}

ABSCONC_COLORS = {"Concrete": "#f9b466", "Neither": "#c0c0c0", "Abstract": "#83b9d8"}
ABSCONC_SHAPES = {"Concrete": "s", "Neither": "x", "Abstract": "o"}

GENRE_COLORS = {
    "Allegory": "#33a02c", "Dialogue": "#1f78b4", "Epic": "#b2df8a",
    "Novel": "#a6cee3", "Novella": "#fb9a99", "Other": "#e31a1c",
    "Pastoral": "#fdbf6f", "Picaresque": "#ff7f00", "Romance": "#cab2d6",
    "Satire": "#6a3d9a", "Tale": "#94945a", "Unknown": "gray", "Verse": "#b15928",
}
GENRE_SHAPES = {
    "Allegory": "d", "Dialogue": "8", "Epic": "<", "Novel": "o",
    "Novella": "v", "Other": "h", "Pastoral": "D", "Picaresque": ">",
    "Romance": "s", "Satire": "x", "Tale": "+", "Unknown": ".", "Verse": "*",
}


# ---------------------------------------------------------------------------
# Word-norm plots
# ---------------------------------------------------------------------------

def plot_norms(dfnorms, words=None, sample_n=10, sample_spacer=0.5,
               only_source=None, ofn=None, title="", ylabel="",
               jitter=False, min_z=-2.5, max_z=2.5,
               source_order=None, label_rename=None,
               font_size=7, width=9, height=8, random_state=0):
    """Plot word positions along the abstract-concrete axis across norm sources.

    Parameters
    ----------
    random_state : int, optional
        Seed for the example-word sampling (`sample_n`). Fixed by default
        so regenerating a book figure reproduces the same example words
        (AUDIT-2026-07-04.md §2.9); pass `None` for non-deterministic
        sampling.
    """
    if source_order is None:
        source_order = NORM_SOURCE_ORDER
    if label_rename is None:
        label_rename = SOURCE_LABELS
    if words is None:
        words = set()

    dfnorms = dfnorms[~dfnorms["source"].isin(BAD_SOURCES)].copy()

    if only_source:
        dfnorms = dfnorms[dfnorms["source"].str.contains(only_source)]
        dfnorms["source"] = dfnorms["source"].str.replace(only_source + ".", "", regex=False)

    dfnorms["order"] = dfnorms["source"].apply(
        lambda x: source_order.index(x) if x in source_order else len(source_order)
    )
    dfnorms = dfnorms.sort_values(["order", "source"])

    # select sample words
    val_counts = dfnorms["word"].value_counts()
    max_count = val_counts.max()
    top_words = set(val_counts[val_counts == max_count].index)
    df_top = dfnorms[dfnorms["word"].isin(top_words)].groupby("word").mean(numeric_only=True)
    df_top["zgroup"] = df_top["z"].apply(lambda x: x // sample_spacer * sample_spacer)

    sample_words = set()
    if sample_n:
        sample_words = set(
            df_top.groupby("zgroup")
            .sample(n=1, replace=False, random_state=random_state)
            .sort_values("z").index
        )
    sample_words |= words

    df_s = dfnorms[dfnorms["word"].isin(sample_words)].copy()
    df_s = df_s.sort_values(["word", "order", "source"])
    df_s["source"] = pd.Categorical(
        df_s["source"],
        categories=list(reversed(sorted(
            dfnorms["source"].unique(),
            key=lambda x: source_order.index(x) if x in source_order else len(source_order),
        ))),
        ordered=True,
    )
    df_s["source_label"] = df_s["source"].astype(str).map(label_rename).fillna(df_s["source"].astype(str))
    df_s["z"] = df_s["z"].clip(min_z, max_z)

    # build plot
    p9.options.figure_size = (width, height)
    fig = (
        p9.ggplot(df_s, p9.aes(x="z", y="source_label", group="source_type", label="word"))
        + p9.theme_classic()
        + p9.geom_point(size=1, alpha=0)
        + p9.geom_path(p9.aes(group="word"), linetype="dashed", alpha=0.25)
    )

    median_keys = {"Median.orig", "Median.median"}
    for i, grp in enumerate([
        df_s[~df_s["source"].isin(median_keys)],
        df_s[df_s["source"].isin(median_keys)],
    ]):
        fig += p9.geom_text(
            data=grp, size=font_size,
            adjust_text={"expand_points": (0, 0)} if jitter else None,
            fontweight=600 if i else "normal",
        )

    fig += p9.xlim(min_z, max_z)
    fig += p9.geom_vline(p9.aes(xintercept=0), alpha=0.666)
    fig += p9.geom_vline(p9.aes(xintercept=1), alpha=0.25)
    fig += p9.geom_vline(p9.aes(xintercept=-1), alpha=0.25)
    fig += p9.ylab(ylabel or "Origin of semantic field")
    fig += p9.xlab("Measured concreteness (standardized score)")
    if title:
        fig += p9.labs(title=title)
    fig += p9.scale_fill_manual(ABSCONC_COLORS)
    fig += p9.scale_color_manual(ABSCONC_COLORS)
    fig += p9.scale_shape_manual(ABSCONC_SHAPES)

    if ofn:
        os.makedirs(os.path.dirname(ofn), exist_ok=True)
        fig.save(ofn, dpi=SAVE_DPI)
    return fig


def plot_allnorms(dfnorms=None, words=None, jitter=False, ofn=None, **kwargs):
    """Plot word norms across all empirical + historical sources."""
    if dfnorms is None:
        dfnorms = format_norms_as_long(get_allnorms())
    source_periods = list(SOURCE_LABELS_FULL.keys())
    dfnorms = dfnorms[dfnorms["source"].isin(source_periods)]
    return plot_norms(
        dfnorms, words=words, sample_n=None, only_source=None,
        source_order=source_periods, label_rename=SOURCE_LABELS_FULL,
        ylabel="Source of measurement", jitter=jitter, ofn=ofn, **kwargs,
    )


# ---------------------------------------------------------------------------
# Fiction-level plotting
# ---------------------------------------------------------------------------

# Year compression for pre-1600 texts
_CUTOFF = 1600
_SPCR = 40
_PREBREAK_CUTS = [1500, 1000, 0, -1000]
_PREBREAKS = [_CUTOFF - (_SPCR * (i + 1)) for i in range(len(_PREBREAK_CUTS))]
_BREAKS = [1600, 1700, 1800, 1900, 2000]


def _compress_year(y):
    if y >= _CUTOFF:
        return y
    for i, brk in enumerate(_PREBREAK_CUTS):
        if y > brk:
            brk0 = _PREBREAK_CUTS[i - 1] if i > 0 else _CUTOFF
            return (_CUTOFF - (_SPCR * (i + 1))) + ((y - brk) / (brk0 - brk)) * _SPCR
    return y


VALTYPE_LABELS = {
    "abs-conc": "<< More concrete words | More abstract words >>   ",
    "abs/conc": "Frequency of abstract words per 1 concrete word",
    "abs": "% Abstract Words",
    "conc": "% Concrete Words",
    "neither": "% words neither abstract nor concrete",
}


def load_data_for_plotting(corpus_name="CanonFiction", sources=None, periods=None):
    """Load count data + metadata, return a plotting-ready DataFrame."""
    from .config import COUNT_DIR
    if sources is None:
        sources = SOURCES_FOR_PLOTTING

    cdf = pd.read_feather(os.path.join(COUNT_DIR, f"data.absconc.{corpus_name}.v6.csv.ft"))
    if sources:
        cdf = cdf[cdf["source"].isin(sources)]
    if periods:
        cdf = cdf[cdf["period"].isin(periods)]
    cdf["abs/conc"] = cdf["num_abs"] / cdf["num_conc"]
    for key in ["abs", "conc", "neither"]:
        cdf[f"perc_{key}"] = cdf[f"num_{key}"] / cdf["num_total"]

    meta = load_corpus(corpus_name).metadata
    alldf = cdf.merge(meta, on="id", how="inner")
    alldf["major_genre"] = alldf["major_genre"].fillna("Unknown").replace("", "Unknown")
    alldf["year_orig"] = alldf["year"]
    alldf["year"] = alldf["year_orig"].apply(_compress_year)

    alldf = alldf[(alldf["canon_genre"] != "") | (alldf.get("corpus_source", "") != "")]
    alldf.loc[alldf["canon_genre"].str.strip() == "", "major_genre"] = "Unknown"

    return (
        alldf.groupby(["major_genre", "canon_genre", "author"])
        .mean(numeric_only=True)
        .reset_index()
        .sort_values("abs/conc")
    )


def plot_fiction(df, valtype="abs/conc", min_y=None, max_y=None,
                 color_by="major_genre", shape_by="major_genre",
                 label_by="canon_genre", jitter=False, smooth=True,
                 span=0.2, title="", highlights=None, standardize=False,
                 save_to=None, width=22.5, height=17.5, version="v2"):
    """Plot abstraction trends across the history of fiction."""
    df = df.copy()
    if highlights is None:
        highlights = []

    # compute value column
    val_map = {
        "abs/conc": lambda d: d["abs/conc"],
        "abs-conc": lambda d: d["num_abs"] - d["num_conc"],
        "conc-abs": lambda d: d["num_conc"] - d["num_abs"],
        "abs+conc": lambda d: d["num_abs"] + d["num_conc"],
        "abs": lambda d: d["perc_abs"] * 100,
        "conc": lambda d: d["perc_conc"] * 100,
        "neither": lambda d: d["perc_neither"] * 100,
    }
    if valtype not in val_map:
        raise ValueError(f"Unknown valtype: {valtype}")
    df["value"] = val_map[valtype](df)

    if min_y is not None:
        # Floor sub-threshold values at min_y, mirroring the max_y clip
        # below. Previously this "compressed" values toward min_y
        # (`min_y + (y - min_y) * spcr`) but the result could still land
        # below min_y for large outliers, and the subsequent `p9.ylim`
        # silently dropped those points from the plot entirely
        # (AUDIT-2026-07-04.md §5).
        df["value"] = df["value"].clip(lower=min_y)
    if max_y is not None:
        df["value"] = df["value"].clip(upper=max_y)

    if standardize:
        df["value"] = zscore(df["value"])

    p9.options.figure_size = (width, height)
    aes_args = {"x": "year", "y": "value"}
    if color_by:
        aes_args["color"] = color_by
    if shape_by:
        aes_args["shape"] = shape_by

    fig = p9.ggplot(df, p9.aes(**aes_args)) + p9.theme_classic()
    fig += p9.scale_color_manual(GENRE_COLORS)
    fig += p9.scale_shape_manual(GENRE_SHAPES)
    fig += p9.scale_x_continuous(
        breaks=_PREBREAKS + _BREAKS,
        labels=[
            (f"{x * -1} BC" if x < 0 else f"{x} AD") if x <= 0 else str(x)
            for x in _PREBREAK_CUTS + _BREAKS
        ],
    )
    fig += p9.geom_vline(xintercept=_BREAKS + _PREBREAKS, color="silver")

    # reference lines
    if valtype == "abs/conc":
        fig += p9.geom_hline(yintercept=1, color="gray")
    elif valtype == "abs-conc":
        fig += p9.geom_hline(yintercept=0, color="gray")

    if min_y is not None and max_y is not None:
        # coord_cartesian only clips the rendered viewport rather than
        # converting out-of-range values to NA before stats are computed
        # (as `p9.ylim`/`scale_y_continuous(limits=...)` do), so it can't
        # silently drop points from the geom_smooth fit below, and it
        # can't be silently overridden by the second y-scale added below
        # for valtype == "abs-conc" (AUDIT-2026-07-04.md §5).
        fig += p9.coord_cartesian(ylim=(min_y, max_y))

    fig += p9.geom_point(alpha=0.5, size=2)

    # labels
    if label_by:
        aesd = {"x": "year", "y": "value", "label": label_by, "guide": False}
        group_cols = [x for x in {shape_by, label_by, color_by} if x]
        dfq = (
            df[(df[label_by] != "") & (df.get(shape_by, "") != "" if shape_by else True)]
            .groupby(group_cols).median(numeric_only=True).reset_index()
        )
        fig += p9.geom_point(alpha=1, size=5, data=dfq)
        if highlights:
            dfl = dfq[~dfq[label_by].isin(highlights)]
            dfh = dfq[dfq[label_by].isin(highlights)]
        else:
            dfl, dfh = dfq, None
        fig += p9.geom_text(
            p9.aes(**aesd), inherit_aes=False, data=dfl,
            adjust_text={"expand_points": (0, 0)} if jitter else None,
        )
        if dfh is not None and len(dfh):
            fig += p9.geom_text(p9.aes(**aesd), fontweight="bold", color="black", data=dfh)

    fig += p9.ylab(VALTYPE_LABELS.get(valtype, valtype))
    fig += p9.xlab("Year")

    if valtype == "abs-conc":
        # Breaks only — the axis *range* is set once, above, via
        # coord_cartesian. A second scale_y_continuous(limits=...) here
        # used to silently replace those limits with different ones
        # (AUDIT-2026-07-04.md §5); coord and scale are independent
        # layers, so this can no longer contradict it.
        fig += p9.scale_y_continuous(breaks=list(range(-50, 51, 10)))

    if smooth:
        fig += p9.geom_smooth(
            p9.aes(x="year", y="value"), inherit_aes=False,
            span=span, se=True, method="loess", alpha=0.15, color="gray", data=df,
        )

    if save_to is True:
        save_to = os.path.join(
            PATH_FIGS,
            f'fig.fiction.{valtype.replace("/", "_")}'
            f'{"_clean" if jitter else ""}.{version}.png',
        )
    if save_to:
        os.makedirs(os.path.dirname(save_to), exist_ok=True)
        fig.save(save_to, dpi=SAVE_DPI)

    return fig


# ---------------------------------------------------------------------------
# Arc plots (adjusted scores from analysis.adjust_scores)
# ---------------------------------------------------------------------------


_YLABEL_CONC = "<< More abstract | More concrete >>"#\n(corpus-adjusted)"
_YLABEL_ABS = "<< More concrete | More abstract >>"#\n(corpus-adjusted)"


def arc_caption(combined_df, genres=None, wrap_by=160):
    """Generate a caption string summarizing texts per genre and corpus.

    Parameters
    ----------
    combined_df : DataFrame
        Combined scored DataFrame with genre_harmonized and corpus_name columns.
    genres : list, optional
        Genres to include. If None, uses all genres present.
    wrap_by : int, optional
        Line width for text wrapping. Default 160.

    Returns
    -------
    str
        Caption like "- Fiction: 45,231 texts = Hathi Englit (30,102) + Chadwyck (5,012) + ..."
    """
    import textwrap

    df = combined_df
    if genres is not None:
        df = df[df["genre_harmonized"].isin(genres)]

    parts = []
    for genre in sorted(df["genre_harmonized"].dropna().unique()):
        gdf = df[df["genre_harmonized"] == genre]
        corpus_counts = gdf["corpus_name"].value_counts()
        corpus_str = " + ".join(
            f"{name.replace('_', ' ').title()} ({count:,})" for name, count in corpus_counts.items()
        )
        opart = f"- {genre}: {len(gdf):,} texts = {corpus_str}"
        parts.append(opart)

    def _wrap(part):
        return textwrap.fill(part, width=wrap_by, break_long_words=False, break_on_hyphens=False)

    out = "\n\n".join(_wrap(part).replace("\n", "\n        ") for part in parts)
    return out


def plot_arc(adj_df, title="", show_raw=True, show_corpus=True,
             show_lines=False, show_se=True, show_fit=True,
             show_loess=False, loess_span=0.3,
             invert=False, ylabel=None,
             save_to=None, width=10, height=6):
    """Plot corpus-adjusted arc from adjust_scores() output.

    Parameters
    ----------
    adj_df : DataFrame
        Output of analysis.adjust_scores(). Must have columns:
        year, score, adjusted, fitted, and optionally corpus, fitted_se.
    show_raw : bool
        If True, show raw (unadjusted) corpus points in light color.
    show_corpus : bool
        If True and corpus column exists, color points by corpus.
    show_lines : bool
        If True, draw lines connecting points within each corpus.
    show_se : bool
        If True and fitted_se column exists, show ±1 SE ribbon around
        the fitted trend line.
    show_fit : bool
        If True (default), show the parametric fitted trend line.
    show_loess : bool
        If True, overlay a LOESS smooth on the adjusted points.
    loess_span : float
        LOESS span parameter (smaller = more wiggly).
    invert : bool
        If True, negate scores so that higher = more abstract.
    """
    df = adj_df.copy()
    if invert:
        for col in ("score", "adjusted", "fitted"):
            if col in df.columns:
                df[col] = -df[col]
    if ylabel is None:
        ylabel = _YLABEL_ABS if invert else _YLABEL_CONC
    has_corpus = "corpus" in df.columns

    p9.options.figure_size = (width, height)

    has_n = "n_texts" in df.columns

    # Main layer: adjusted points
    aes_kw = {"x": "year", "y": "adjusted"}
    if has_corpus and show_corpus:
        aes_kw["color"] = "corpus"
    if has_n:
        aes_kw["size"] = "n_texts"
    fig = p9.ggplot(df, p9.aes(**aes_kw))

    fig += p9.theme_classic()
    fig += p9.theme(legend_position="right" if has_corpus and show_corpus else "none")
    if has_n:
        fig += p9.scale_size_continuous(range=(1, 6), name="Texts")

    # Raw points (before adjustment) as faint background
    if show_raw and has_corpus:
        raw_aes = {"x": "year", "y": "score", "color": "corpus"}
        if has_n:
            raw_aes["size"] = "n_texts"
        fig += p9.geom_point(p9.aes(**raw_aes), alpha=0.15)
        if show_lines:
            fig += p9.geom_line(p9.aes(x="year", y="score", color="corpus",
                                       group="corpus"),
                                alpha=0.15, size=0.5)

    # Adjusted points
    fig += p9.geom_point(alpha=0.6)

    # Lines connecting adjusted points within each corpus
    if show_lines and has_corpus:
        fig += p9.geom_line(p9.aes(group="corpus"), alpha=0.4, size=0.5)

    # Fitted trend line (with optional SE ribbon)
    if show_fit:
        has_se = "fitted_se" in df.columns
        trend_cols = ["year", "fitted"]
        if has_se:
            trend_cols.append("fitted_se")
        trend = df[trend_cols].drop_duplicates().sort_values("year")
        if show_se and has_se:
            trend["_lo"] = trend["fitted"] - trend["fitted_se"]
            trend["_hi"] = trend["fitted"] + trend["fitted_se"]
            fig += p9.geom_ribbon(p9.aes(x="year", ymin="_lo", ymax="_hi"),
                                  data=trend, fill="black", alpha=0.15,
                                  inherit_aes=False)
        fig += p9.geom_line(p9.aes(x="year", y="fitted"),
                            data=trend, color="black", size=1.2,
                            inherit_aes=False)

    # LOESS smooth on adjusted points
    if show_loess:
        fig += p9.geom_smooth(p9.aes(x="year", y="adjusted"),
                              method="loess", span=loess_span,
                              color="red", se=True, alpha=0.15,
                              inherit_aes=False, data=df)

    fig += p9.xlab("Year")
    fig += p9.ylab(ylabel)
    if title:
        fig += p9.labs(title=title)

    if save_to:
        os.makedirs(os.path.dirname(save_to), exist_ok=True)
        fig.save(save_to, dpi=SAVE_DPI)

    return fig


def plot_arc_by_genre(combined_df, genres=None,
                      score_col="Abs-Conc.Median.median",
                      model="quadratic", show_raw=False,
                      show_lines=False, show_facet=True,
                      show_se=True, show_fit=True,
                      show_loess=False, loess_span=0.3,
                      invert=False,
                      ylabel=None, save_to=None,
                      ncol=2, width=14, height=None, **adjust_kw):
    """Plot adjusted arcs for multiple genres.

    Parameters
    ----------
    combined_df : DataFrame
        Combined scored DataFrame with genre_harmonized and corpus_name columns.
        Typically from analysis.load_all_scored().
    genres : list, optional
        Genres to include. If None, uses all genres with enough data.
    model : str
        "quadratic", "cubic", "quartic", or "piecewise" — passed to adjust_scores.
    show_raw : bool
        If True, show raw (unadjusted) corpus points in light color.
    show_lines : bool
        If True, draw lines connecting points within each corpus
        (per-genre when faceted, per-genre-per-corpus when combined).
    show_facet : bool
        If True (default), show each genre in a separate facet panel,
        colored by corpus. If False, plot all genres on one panel,
        colored by genre and shaped by corpus.
    show_fit : bool
        If True (default), show the parametric fitted trend line.
    show_loess : bool
        If True, overlay a LOESS smooth on the adjusted points.
    loess_span : float
        LOESS span parameter (smaller = more wiggly).
    invert : bool
        If True, negate scores so that higher = more abstract.
    ncol : int
        Number of columns in facet grid (only used when show_facet=True).
    """
    from .analysis import adjust_scores

    if genres is None:
        gcounts = combined_df["genre_harmonized"].value_counts()
        genres = gcounts[gcounts >= 30].index.tolist()

    panels = []
    for genre in genres:
        gdf = combined_df[combined_df["genre_harmonized"] == genre]
        if len(gdf) < 30:
            continue
        adj = adjust_scores(gdf, score_col=score_col, model=model, **adjust_kw)
        if len(adj) == 0:
            continue
        adj["genre"] = genre
        panels.append(adj)

    if not panels:
        return None

    df = pd.concat(panels, ignore_index=True)
    if invert:
        for col in ("score", "adjusted", "fitted"):
            if col in df.columns:
                df[col] = -df[col]
    if ylabel is None:
        ylabel = _YLABEL_ABS if invert else _YLABEL_CONC
    has_corpus = "corpus" in df.columns
    has_n = "n_texts" in df.columns

    # Interaction group for lines: unique per genre+corpus combination
    if has_corpus:
        df["_grp"] = df["genre"] + ":" + df["corpus"]

    if height is None:
        if show_facet:
            nrow = int(np.ceil(len(panels) / ncol))
            height = 4 * nrow
        else:
            height = 8

    p9.options.figure_size = (width, height)

    # Build base aesthetics
    n_corpora = df["corpus"].nunique() if has_corpus else 0
    use_shape = has_corpus and not show_facet and n_corpora <= 20

    aes_kw = {"x": "year", "y": "adjusted"}
    if show_facet:
        if has_corpus:
            aes_kw["color"] = "corpus"
    else:
        aes_kw["color"] = "genre"
        if use_shape:
            aes_kw["shape"] = "corpus"
        else:
            aes_kw["shape"] = "genre"
    if has_n:
        aes_kw["size"] = "n_texts"

    fig = p9.ggplot(df, p9.aes(**aes_kw))
    fig += p9.theme_classic()
    fig += p9.theme(legend_position="bottom",
                    strip_text=p9.element_text(size=11, weight="bold"))
    if has_n:
        fig += p9.scale_size_continuous(range=(0.5, 4), name="Texts")
    if use_shape:
        fig += p9.guides(shape=p9.guide_legend(ncol=4))

    # Raw points (before adjustment) as faint background
    if show_raw and has_corpus:
        raw_aes = {"x": "year", "y": "score"}
        if show_facet:
            raw_aes["color"] = "corpus"
        else:
            raw_aes["color"] = "genre"
            if use_shape:
                raw_aes["shape"] = "corpus"
        if has_n:
            raw_aes["size"] = "n_texts"
        fig += p9.geom_point(p9.aes(**raw_aes), alpha=0.15)
        if show_lines:
            raw_line_aes = {"x": "year", "y": "score", "group": "_grp"}
            if show_facet:
                raw_line_aes["color"] = "corpus"
            else:
                raw_line_aes["color"] = "genre"
            fig += p9.geom_line(p9.aes(**raw_line_aes), alpha=0.15, size=0.5)

    # Adjusted points
    fig += p9.geom_point(alpha=0.35)

    # Lines connecting adjusted points within each corpus (and genre)
    if show_lines and has_corpus:
        line_aes = {"group": "_grp"}
        if not show_facet:
            line_aes["color"] = "genre"
        fig += p9.geom_line(p9.aes(**line_aes), alpha=0.25, size=0.5)

    # Fitted trend per genre (with optional SE ribbon)
    if show_fit:
        has_se = "fitted_se" in df.columns
        trend_cols = ["year", "fitted", "genre"]
        if has_se:
            trend_cols.append("fitted_se")
        trend = df[trend_cols].drop_duplicates().sort_values("year")
        if show_se and has_se:
            trend["_lo"] = trend["fitted"] - trend["fitted_se"]
            trend["_hi"] = trend["fitted"] + trend["fitted_se"]
        if show_facet:
            if show_se and has_se:
                fig += p9.geom_ribbon(p9.aes(x="year", ymin="_lo", ymax="_hi"),
                                      data=trend, fill="black", alpha=0.15,
                                      inherit_aes=False)
            fig += p9.geom_line(p9.aes(x="year", y="fitted"),
                                data=trend, color="black", size=1,
                                inherit_aes=False)
        else:
            if show_se and has_se:
                fig += p9.geom_ribbon(p9.aes(x="year", ymin="_lo", ymax="_hi",
                                             fill="genre"),
                                      data=trend, alpha=0.15,
                                      inherit_aes=False)
            fig += p9.geom_line(p9.aes(x="year", y="fitted", color="genre", linetype="genre"),
                                data=trend, size=1,
                                inherit_aes=False)

    # LOESS smooth on adjusted points
    if show_loess:
        if show_facet:
            fig += p9.geom_smooth(p9.aes(x="year", y="adjusted"),
                                  method="loess", span=loess_span,
                                  color="red", se=True, alpha=0.15,
                                  inherit_aes=False, data=df)
        else:
            fig += p9.geom_smooth(p9.aes(x="year", y="adjusted", color="genre", linetype="genre"),
                                  method="loess", span=loess_span,
                                  se=True, alpha=0.15,
                                  inherit_aes=False, data=df)

    if show_facet:
        fig += p9.facet_wrap("genre", ncol=ncol, scales="free_y")

    fig += p9.xlab("Year")
    fig += p9.ylab(ylabel)

    if save_to:
        os.makedirs(os.path.dirname(save_to), exist_ok=True)
        fig.save(save_to, dpi=SAVE_DPI)

    return fig


# ---------------------------------------------------------------------------
# UMAP and cross-language line plots (matplotlib)
# ---------------------------------------------------------------------------

PERIOD_COLORS = {
    "pre-1600": "#440154",
    "C17a": "#443983",
    "C17b": "#31688e",
    "C18a": "#21918c",
    "C18b": "#5ec962",
    "post-1800": "#fde725",
}

PERIOD_BINS = [0, 1600, 1650, 1700, 1750, 1800, 2100]
PERIOD_LABELS = ["pre-1600", "C17a", "C17b", "C18a", "C18b", "post-1800"]

FORM_GENRE_COLORS = {
    "novel": "#1f77b4", "romance": "#ff7f0e", "allegory": "#2ca02c",
    "tale": "#d62728", "biography": "#9467bd", "novella": "#8c564b",
    "dialogue": "#e377c2", "satire": "#bcbd22", "(other)": "#bbbbbb",
}

LANG_COLORS = {"en": "#1f77b4", "fr": "#d62728", "de": "#9467bd"}


def _savefig(fig, figname, figdir=None):
    if figname is None:
        return
    # Default to the package's configured figures directory rather than a
    # cwd-relative "../figures" (AUDIT-2026-07-04.md §5), which only
    # resolves correctly if the caller's cwd happens to be one level
    # below the repo root.
    path = os.path.join(figdir or PATH_FIGS, figname) if not os.path.isabs(figname) else figname
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, bbox_inches="tight")


def scatter_binary(ax, df, col, title, *, xcol="umap_x", ycol="umap_y",
                   true_color="#d62728", false_color="#1f77b4"):
    """Two-colour scatter (true/false on ``col``) with counts in the legend."""
    from matplotlib.lines import Line2D
    vals = df[col].fillna(0).astype(int)
    for v, color, z in [(0, false_color, 1), (1, true_color, 2)]:
        m = vals == v
        ax.scatter(df.loc[m, xcol], df.loc[m, ycol], c=color, s=5, alpha=0.4, zorder=z)
    t, f = int((vals == 1).sum()), int((vals == 0).sum())
    legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=true_color, markersize=8, label=f"true ({t})"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=false_color, markersize=8, label=f"false ({f})"),
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=8)
    ax.set_title(title, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])


def facet_by_genre(df, genre_cols, color_col=None, *, norm=None, cmap="RdBu",
                   period_mode=False, figname=None, title="",
                   xcol="umap_x", ycol="umap_y", ncols=4, figdir=None,
                   period_colors=None):
    """Multi-panel UMAP facet — one panel per genre column.

    Each panel shows all points greyed out, then overlays points belonging to
    that genre coloured either by a continuous ``color_col`` or by ``period_bin``
    if ``period_mode=True``.
    """
    import matplotlib.pyplot as plt
    period_colors = period_colors or PERIOD_COLORS
    n = len(genre_cols)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))
    axes = np.atleast_2d(axes).flatten()
    i = -1
    for i, gcol in enumerate(genre_cols):
        ax = axes[i]
        ax.scatter(df[xcol], df[ycol], c="#e8e8e8", s=3, alpha=0.25, zorder=1)
        mask = df[gcol] == 1
        d = df[mask]
        gname = gcol.replace("genre_", "")
        if period_mode:
            for p, color in period_colors.items():
                dm = d[d["period_bin"] == p]
                if len(dm):
                    ax.scatter(dm[xcol], dm[ycol], c=color, s=8, alpha=0.7,
                               label=f"{p} ({len(dm)})")
            ax.legend(loc="upper right", fontsize=6, markerscale=1.5)
            title_extra = ""
        elif color_col is None:
            # No color aesthetic requested — plot this genre's points in a
            # single flat color instead of crashing on `d[None]`
            # (AUDIT-2026-07-04.md §5).
            ax.scatter(d[xcol], d[ycol], c="#1f77b4", s=8, alpha=0.7)
            title_extra = ""
        else:
            valid = d[color_col].notna()
            ax.scatter(
                d.loc[valid, xcol], d.loc[valid, ycol],
                c=d.loc[valid, color_col], cmap=cmap, norm=norm, s=8, alpha=0.7,
            )
            title_extra = f", mean={d[color_col].mean():.2f}" if d[color_col].notna().any() else ""
        ax.set_title(f"{gname} (n={len(d)}){title_extra}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(title, fontsize=13, y=1.01)
    plt.tight_layout()
    _savefig(fig, figname, figdir)
    return fig


def plot_umap_overview(feat, *, form_tags, annotation_cols=(), figname=None,
                       title=None, figdir=None, xcol="umap_x", ycol="umap_y"):
    """6-panel overview: primary form-genre / period / abstractness / up to 3 annotations.

    Expects ``feat`` to already have ``form_primary``, ``period_bin``, ``abs_score``
    and the UMAP coordinates.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    fig, axes = plt.subplots(2, 3, figsize=(17, 11))

    ax = axes[0, 0]
    order = ["(other)"] + [g.replace("genre_", "") for g in form_tags]
    for g in order:
        m = feat["form_primary"] == g
        if m.sum() == 0: continue
        z = 1 if g == "(other)" else 2
        ax.scatter(feat.loc[m, xcol], feat.loc[m, ycol],
                   c=FORM_GENRE_COLORS.get(g, "#999"), s=5, alpha=0.5,
                   label=f"{g} ({m.sum()})", zorder=z)
    ax.legend(loc="upper right", fontsize=7, markerscale=2)
    ax.set_title("Primary form-genre", fontsize=11); ax.set_xticks([]); ax.set_yticks([])

    ax = axes[0, 1]
    for p, color in PERIOD_COLORS.items():
        m = feat["period_bin"] == p
        if m.sum():
            ax.scatter(feat.loc[m, xcol], feat.loc[m, ycol], c=color, s=5,
                       alpha=0.55, label=f"{p} ({m.sum()})")
    ax.legend(loc="upper right", fontsize=7, markerscale=2)
    ax.set_title("Period", fontsize=11); ax.set_xticks([]); ax.set_yticks([])

    ax = axes[0, 2]
    valid = feat["abs_score"].notna()
    ax.scatter(feat.loc[~valid, xcol], feat.loc[~valid, ycol], c="#eee", s=4, alpha=0.3)
    if valid.sum() > 0:
        # quantile() on an empty subset returns NaN bounds, which would
        # otherwise crash/blank out Normalize (AUDIT-2026-07-04.md §5).
        vmin, vmax = feat.loc[valid, "abs_score"].quantile([0.05, 0.95])
        norm = Normalize(vmin=vmin, vmax=vmax)
        sc = ax.scatter(feat.loc[valid, xcol], feat.loc[valid, ycol],
                        c=feat.loc[valid, "abs_score"], cmap="RdBu", norm=norm, s=5, alpha=0.55)
        plt.colorbar(sc, ax=ax, shrink=0.7, label="abstractness (+ = abstract)")
    ax.set_title("Abstractness score", fontsize=11); ax.set_xticks([]); ax.set_yticks([])

    annotation_cols = list(annotation_cols)
    for i in range(3):
        col = annotation_cols[i] if i < len(annotation_cols) else None
        if col and col in feat.columns:
            scatter_binary(axes[1, i], feat, col, col, xcol=xcol, ycol=ycol)
        else:
            axes[1, i].set_visible(False)

    fig.suptitle(title or f"Passage-level UMAP of {len(feat):,} passages",
                 fontsize=14, y=0.995)
    plt.tight_layout()
    _savefig(fig, figname, figdir)
    return fig


def plot_text_umap_4panel(df, *, xcol, ycol, form_tags, figname=None,
                          title_suffix="", figdir=None):
    """4-panel text-level UMAP: language / primary genre / abstractness / period."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    ax = axes[0, 0]
    for lang, color in LANG_COLORS.items():
        m = df["lang"] == lang
        if m.sum() == 0: continue
        ax.scatter(df.loc[m, xcol], df.loc[m, ycol], c=color, s=5, alpha=0.5,
                   label=f"{lang} ({m.sum()})")
    ax.legend(fontsize=9, markerscale=3); ax.set_title("Language", fontsize=12)
    ax.set_xticks([]); ax.set_yticks([])

    ax = axes[0, 1]
    def _primary(row):
        for t in form_tags:
            if row.get(t, 0) == 1:
                return t.replace("genre_", "")
        return "(other)"
    g = df.apply(_primary, axis=1)
    for gg in ["(other)"] + [t.replace("genre_", "") for t in form_tags]:
        m = g == gg
        if m.sum() == 0: continue
        z = 1 if gg == "(other)" else 2
        ax.scatter(df.loc[m, xcol], df.loc[m, ycol],
                   c=FORM_GENRE_COLORS.get(gg, "#999"), s=5, alpha=0.5,
                   label=f"{gg} ({m.sum()})", zorder=z)
    ax.legend(fontsize=7, loc="upper right", markerscale=2.5)
    ax.set_title("Primary genre", fontsize=12); ax.set_xticks([]); ax.set_yticks([])

    ax = axes[1, 0]
    valid = df["abs_score"].notna()
    ax.scatter(df.loc[~valid, xcol], df.loc[~valid, ycol], c="#eee", s=4, alpha=0.3)
    if valid.sum() > 0:
        # quantile() on an empty subset returns NaN bounds, which would
        # otherwise crash/blank out Normalize (AUDIT-2026-07-04.md §5).
        vmin, vmax = df.loc[valid, "abs_score"].quantile([0.05, 0.95])
        norm = Normalize(vmin=vmin, vmax=vmax)
        sc = ax.scatter(df.loc[valid, xcol], df.loc[valid, ycol],
                        c=df.loc[valid, "abs_score"], cmap="RdBu", norm=norm, s=5, alpha=0.55)
        plt.colorbar(sc, ax=ax, shrink=0.7, label="abstractness")
    ax.set_title("Abstractness", fontsize=12); ax.set_xticks([]); ax.set_yticks([])

    ax = axes[1, 1]
    for p, color in PERIOD_COLORS.items():
        m = df["period_bin"] == p
        if m.sum() == 0: continue
        ax.scatter(df.loc[m, xcol], df.loc[m, ycol], c=color, s=5, alpha=0.5,
                   label=f"{p} ({m.sum()})")
    ax.legend(fontsize=7, loc="upper right", markerscale=2.5)
    ax.set_title("Period", fontsize=12); ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(f"Text-level UMAP ({len(df):,} texts, mean-pooled e5){title_suffix}",
                 fontsize=14, y=0.995)
    plt.tight_layout()
    _savefig(fig, figname, figdir)
    return fig


def plot_cross_language_lines(df, field_specs, *, title, figname=None,
                              figsize=None, y_label="% of passages",
                              x_offset=25, figdir=None):
    """Horizontal row of EN-vs-FR line plots, one panel per field.

    ``df`` is long-form with columns ``field, lang, period, pct`` (as produced
    by ``analysis.fetch_cross_language_fields``). ``field_specs`` is the same
    list of ``(parent, enum)`` tuples used to build it — drives panel order.
    """
    import matplotlib.pyplot as plt
    n = len(field_specs)
    figsize = figsize or (max(13, 4 * n), 3.8)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]
    for ax, (parent, enum) in zip(axes, field_specs):
        fname = f"{parent}__{enum}"
        sub = df[df["field"] == fname]
        for lang, color in [("en", "#1f77b4"), ("fr", "#d62728")]:
            d = sub[sub["lang"] == lang].sort_values("period")
            label = f"{lang.upper()} (n={int(d['n'].sum()):,})" if not d.empty else lang.upper()
            ax.plot(d["period"] + x_offset, d["pct"], "o-", color=color,
                    label=label, linewidth=2, markersize=6)
        ax.set_title(enum.replace("_", " "), fontsize=11)
        ax.set_xlabel("year (bin center)")
        ax.set_ylabel(y_label)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(title, fontsize=13, y=1.02)
    plt.tight_layout()
    _savefig(fig, figname, figdir)
    return fig


def add_period_bin(df, year_col="year", bins=None, labels=None):
    """Convenience: add a ``period_bin`` column based on integer year."""
    bins = bins or PERIOD_BINS
    labels = labels or PERIOD_LABELS
    out = df.copy()
    out["period_bin"] = pd.cut(out[year_col], bins=bins, labels=labels)
    return out
