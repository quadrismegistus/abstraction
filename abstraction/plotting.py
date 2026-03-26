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
    format_norms_as_long, get_orignorms, get_allnorms, NORM_SOURCE_ORDER,
    classify_word,
)

p9.options.dpi = 300

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
               font_size=7, width=9, height=8):
    """Plot word positions along the abstract-concrete axis across norm sources."""
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
            df_top.groupby("zgroup").sample(n=1, replace=False).sort_values("z").index
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
        fig.save(ofn)
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
                 save_to=None, width=22.5, height=17.5, version="v1"):
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
        spcr = 0.5
        df["value"] = df["value"].apply(lambda y: y if y > min_y else min_y + (y - min_y) * spcr)
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
        fig += p9.ylim(min_y, max_y)

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
        fig += p9.scale_y_continuous(
            breaks=list(range(-50, 51, 10)),
            limits=[min_y - 2 if min_y else None, max_y + 2 if max_y else None],
        )

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
        fig.save(save_to)

    return fig


# ---------------------------------------------------------------------------
# Arc plots (adjusted scores from analysis.adjust_scores)
# ---------------------------------------------------------------------------


def plot_arc(adj_df, title="", show_raw=True, show_corpus=True,
             ylabel="Abstractness − Concreteness (corpus-adjusted)",
             save_to=None, width=10, height=6):
    """Plot corpus-adjusted arc from adjust_scores() output.

    Parameters
    ----------
    adj_df : DataFrame
        Output of analysis.adjust_scores(). Must have columns:
        year, score, adjusted, fitted, and optionally corpus.
    show_raw : bool
        If True, show raw (unadjusted) corpus points in light color.
    show_corpus : bool
        If True and corpus column exists, color points by corpus.
    """
    df = adj_df.copy()
    has_corpus = "corpus" in df.columns

    p9.options.figure_size = (width, height)

    # Main layer: adjusted points
    if has_corpus and show_corpus:
        fig = p9.ggplot(df, p9.aes(x="year", y="adjusted", color="corpus"))
    else:
        fig = p9.ggplot(df, p9.aes(x="year", y="adjusted"))

    fig += p9.theme_classic()
    fig += p9.theme(legend_position="right" if has_corpus and show_corpus else "none")

    # Raw points (before adjustment) as faint background
    if show_raw and has_corpus:
        fig += p9.geom_point(p9.aes(x="year", y="score", color="corpus"),
                             alpha=0.15, size=1.5)

    # Adjusted points
    fig += p9.geom_point(alpha=0.6, size=2.5)

    # Fitted trend line
    trend = df[["year", "fitted"]].drop_duplicates().sort_values("year")
    fig += p9.geom_line(p9.aes(x="year", y="fitted"),
                        data=trend, color="black", size=1.2,
                        inherit_aes=False)

    fig += p9.xlab("Year")
    fig += p9.ylab(ylabel)
    if title:
        fig += p9.labs(title=title)

    if save_to:
        os.makedirs(os.path.dirname(save_to), exist_ok=True)
        fig.save(save_to)

    return fig


def plot_arc_by_genre(combined_df, genres=None,
                      score_col="Abs-Conc.Median.median",
                      model="quadratic", save_to=None,
                      ncol=2, width=14, height=None, **adjust_kw):
    """Plot adjusted arcs for multiple genres as faceted panels.

    Parameters
    ----------
    combined_df : DataFrame
        Combined scored DataFrame with genre_harmonized and corpus_name columns.
        Typically from analysis.load_all_scored().
    genres : list, optional
        Genres to include. If None, uses all genres with enough data.
    model : str
        "quadratic" or "piecewise" — passed to adjust_scores.
    ncol : int
        Number of columns in facet grid.
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
    has_corpus = "corpus" in df.columns

    if height is None:
        nrow = int(np.ceil(len(panels) / ncol))
        height = 4 * nrow

    p9.options.figure_size = (width, height)

    if has_corpus:
        fig = p9.ggplot(df, p9.aes(x="year", y="adjusted", color="corpus"))
    else:
        fig = p9.ggplot(df, p9.aes(x="year", y="adjusted"))

    fig += p9.theme_classic()
    fig += p9.theme(legend_position="bottom",
                    strip_text=p9.element_text(size=11, weight="bold"))
    fig += p9.geom_point(alpha=0.5, size=1.5)

    # Fitted trend per genre
    trend = df[["year", "fitted", "genre"]].drop_duplicates().sort_values("year")
    fig += p9.geom_line(p9.aes(x="year", y="fitted"),
                        data=trend, color="black", size=1,
                        inherit_aes=False)

    fig += p9.facet_wrap("genre", ncol=ncol, scales="free_y")
    fig += p9.xlab("Year")
    fig += p9.ylab("Abstractness − Concreteness\n(corpus-adjusted)")

    if save_to:
        os.makedirs(os.path.dirname(save_to), exist_ok=True)
        fig.save(save_to)

    return fig
