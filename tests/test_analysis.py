"""Tests for research-validity fixes in analysis.py (docs/AUDIT-2026-07-04.md
sections 2 and 5): per_feature_r2 centering, assign_period_score fallback
labeling, int(NaN) breakpoint guards, pct_abstract/pct_concrete boundary
convention, and report_arc's corpus-balanced magnitude reporting.

Norm caches (_NORM_DICTS, _NORMS_ARRAYS_CACHE) are cleared per-test by the
autouse fixture in tests/conftest.py; none of these tests touch them since
analysis.py's report/period/feature helpers don't read live norm data.
"""

import numpy as np
import pandas as pd
import pytest

import abstraction.analysis as analysis
from abstraction.analysis import (
    assign_period_score,
    pct_abstract,
    pct_concrete,
    report_arc,
    report_arc_counts,
    report_full,
)


# ---------------------------------------------------------------------------
# per_feature_r2 centering (audit §2.3, analysis.py per_feature_r2)
# ---------------------------------------------------------------------------

class TestPerFeatureR2Centering:
    """A no-intercept regression on uncentered v understates R^2 for
    high-prevalence binary features; per_feature_r2 must center v first.
    """

    def _make_feat(self, seed=0, n=400):
        rng = np.random.default_rng(seed)
        z = rng.normal(size=n)
        # 3-d "embedding" correlated with the underlying signal z.
        emb = np.column_stack([
            z + rng.normal(scale=0.5, size=n),
            -0.5 * z + rng.normal(scale=0.5, size=n),
            rng.normal(scale=0.5, size=n),
        ])
        # feat_rare (~10% prevalence) and feat_common (its exact complement,
        # ~90% prevalence) carry IDENTICAL information about z -- one is
        # just "not" the other -- so a correct method must score them
        # identically regardless of prevalence.
        thresh = np.quantile(z, 0.9)
        feat_rare = (z > thresh).astype(float)
        feat_common = 1.0 - feat_rare
        return pd.DataFrame({
            "embedding": list(emb),
            "feat_rare": feat_rare,
            "feat_common": feat_common,
        })

    def test_complementary_features_get_equal_r2_after_centering(self):
        feat = self._make_feat()
        groups = {"g": ["feat_rare", "feat_common"]}
        result = analysis.per_feature_r2(feat, groups, pca_components=2, top_n=10)
        r2 = dict(zip(result["feature"], result["R2"]))
        prevalence = dict(zip(result["feature"], result["prevalence"]))

        # Sanity: prevalence really differs (~0.1 vs ~0.9).
        assert prevalence["feat_common"] - prevalence["feat_rare"] > 0.5
        # Same underlying signal -> (nearly) the same R^2 once centered.
        assert r2["feat_rare"] == pytest.approx(r2["feat_common"], abs=1e-9)

    def test_prevalence_still_reported_correctly(self):
        """Centering v for the regression must not corrupt the reported
        'prevalence' column (the original, uncentered feature mean)."""
        feat = self._make_feat()
        groups = {"g": ["feat_rare", "feat_common"]}
        result = analysis.per_feature_r2(feat, groups, pca_components=2, top_n=10)
        prevalence = dict(zip(result["feature"], result["prevalence"]))
        assert prevalence["feat_rare"] == pytest.approx(0.1, abs=0.05)
        assert prevalence["feat_common"] == pytest.approx(0.9, abs=0.05)

    def test_uncentered_regression_would_disagree(self):
        """Demonstrates the bug this fix addresses: without centering v,
        the complementary pair does NOT get equal R^2, confirming the
        centering step (not some other factor) is what makes them agree.
        """
        from sklearn.decomposition import PCA

        feat = self._make_feat()
        X_raw = np.array(feat["embedding"].tolist(), dtype=np.float32)
        X_raw = X_raw - X_raw.mean(axis=0)
        pca = PCA(n_components=2, random_state=42)
        Xp = pca.fit_transform(X_raw)
        total_ss = (Xp ** 2).sum()

        def uncentered_r2(v):
            Y = v.reshape(-1, 1)
            beta = np.linalg.lstsq(Y, Xp, rcond=None)[0]
            return 1 - ((Xp - Y @ beta) ** 2).sum() / total_ss

        r2_rare = uncentered_r2(feat["feat_rare"].values.astype(float))
        r2_common = uncentered_r2(feat["feat_common"].values.astype(float))
        assert abs(r2_rare - r2_common) > 1e-6


# ---------------------------------------------------------------------------
# assign_period_score fallback labeling (audit §2.4)
# ---------------------------------------------------------------------------

class TestAssignPeriodScoreFallbackLabeling:
    """A row whose year falls in a century bin, but whose century column is
    NaN for that row, must be relabeled to the median fallback -- not left
    mislabeled with the century name it never actually used.
    """

    def test_row_level_nan_in_period_column_relabels_to_median(self):
        # C18 column exists (so the old code's `if col in df.columns` gate
        # passes) but is NaN for this specific row -- the exact scenario
        # test_pct_scoring.py's test_nan_score_falls_back_to_median covers
        # for period_score; this test additionally checks the labeling.
        df = pd.DataFrame({
            "year": [1750],
            "Abs-Conc.Median.C18": [np.nan],
            "Abs-Conc.Median.median": [0.99],
        })
        result = assign_period_score(df)
        assert result.iloc[0]["period_score"] == 0.99
        assert result.iloc[0]["norm_period"] == "median"
        assert result.iloc[0]["period_score_source"] == "median"

    def test_normal_period_row_labeled_as_period_source(self):
        df = pd.DataFrame({
            "year": [1750],
            "Abs-Conc.Median.C18": [0.42],
            "Abs-Conc.Median.median": [0.99],
        })
        result = assign_period_score(df)
        assert result.iloc[0]["norm_period"] == "C18"
        assert result.iloc[0]["period_score"] == 0.42
        assert result.iloc[0]["period_score_source"] == "period"

    def test_mixed_frame_labels_each_row_by_its_own_source(self):
        df = pd.DataFrame({
            "year": [1750, 1755, 1650],
            "Abs-Conc.Median.C18": [0.42, np.nan, np.nan],
            "Abs-Conc.Median.C17": [np.nan, np.nan, 0.11],
            "Abs-Conc.Median.median": [0.99, 0.99, 0.99],
        })
        result = assign_period_score(df)
        assert result.iloc[0]["norm_period"] == "C18"
        assert result.iloc[0]["period_score_source"] == "period"
        # Row 1 (1755): year is in the C18 bin, C18 column exists, but is
        # NaN for this row -- must fall back to median, not stay "C18".
        assert result.iloc[1]["norm_period"] == "median"
        assert result.iloc[1]["period_score"] == 0.99
        assert result.iloc[1]["period_score_source"] == "median"
        assert result.iloc[2]["norm_period"] == "C17"
        assert result.iloc[2]["period_score_source"] == "period"

    def test_missing_period_column_entirely_still_falls_back(self):
        """No regression on the case where the whole column is absent."""
        df = pd.DataFrame({
            "year": [1650],
            "Abs-Conc.Median.median": [0.99],
        })
        result = assign_period_score(df)
        assert result.iloc[0]["norm_period"] == "median"
        assert result.iloc[0]["period_score_source"] == "median"

    def test_added_column_does_not_break_existing_columns(self):
        """Backward compatibility: period_score/norm_period keep their
        original meaning for non-fallback rows; period_score_source is
        purely additive (app/routes/arc.py only reads the first two)."""
        df = pd.DataFrame({
            "year": [1750],
            "Abs-Conc.Median.C18": [0.42],
            "Abs-Conc.Median.median": [0.99],
        })
        result = assign_period_score(df)
        assert set(["period_score", "norm_period", "period_score_source"]) <= set(result.columns)


# ---------------------------------------------------------------------------
# pct_abstract / pct_concrete boundary convention (audit §2.2 / §5)
# ---------------------------------------------------------------------------

class TestPctBoundaryConvention:
    """Histogram bins from count_corpus_freqs are keyed by UPPER edge and
    half-open on the left ([e - 0.1, e)); a z-score exactly at the cutoff
    always lands in the bin keyed cutoff + 0.1. pct_abstract must fold that
    bin in (matching the live path's inclusive v <= cutoff); pct_concrete
    already does with no adjustment.
    """

    def test_pct_abstract_includes_bin_containing_exact_tie(self):
        rec = {"Abs-Conc.Median.median": {"-1.1": 5, "-1.0": 3, "-0.9": 4, "0.0": 10}}
        pa = pct_abstract(rec, cutoff=-1.0)
        # abstract = bins keyed <= -1.0 + 0.1 = -0.9 -> "-1.1","-1.0","-0.9"
        assert pa == pytest.approx((5 + 3 + 4) / (5 + 3 + 4 + 10))

    def test_pct_abstract_does_not_overreach_a_second_bin(self):
        rec = {"Abs-Conc.Median.median": {"-1.0": 3, "-0.9": 4, "-0.8": 100}}
        pa = pct_abstract(rec, cutoff=-1.0)
        # "-0.8" bin is [-0.9, -0.8), entirely above the cutoff -- must
        # not be folded in even though it's the next bin up.
        assert pa == pytest.approx((3 + 4) / (3 + 4 + 100))

    def test_pct_concrete_already_inclusive_at_exact_tie(self):
        # Bin keyed "1.1" is [1.0, 1.1) -- the interval containing z == 1.0
        # -- and is already selected by plain `e > cutoff`, no adjustment.
        rec = {"Abs-Conc.Median.median": {"1.0": 5, "1.1": 3, "1.2": 2}}
        pc = pct_concrete(rec, cutoff=1.0)
        assert pc == pytest.approx((3 + 2) / (5 + 3 + 2))

    def test_pct_abstract_and_pct_concrete_partition_disjointly(self):
        """Sanity check that the (necessarily approximate) abstract
        boundary fix doesn't start double-counting the concrete side."""
        rec = {"Abs-Conc.Median.median": {"-1.0": 3, "-0.9": 4, "0.0": 10, "1.0": 5, "1.1": 3}}
        pa = pct_abstract(rec, cutoff=-1.0)
        pc = pct_concrete(rec, cutoff=1.0)
        total = 3 + 4 + 10 + 5 + 3
        # abstract: "-1.0","-0.9" (<= -0.9); concrete: "1.1" (> 1.0)
        assert pa == pytest.approx((3 + 4) / total)
        assert pc == pytest.approx(3 / total)
        assert pa + pc < 1.0  # "0.0" and "1.0" bins are neither


# ---------------------------------------------------------------------------
# int(NaN) breakpoint guards (audit §5, analysis.py:815,1667,1799,1912,2198)
# ---------------------------------------------------------------------------

def _make_arc_scores_df(score_col="Abs-Conc.Median.median", genre="Fiction",
                         corpus="c1", n_per_decade=5, seed=0):
    """20 decades (1650..1840) of synthetic scored texts -- enough for the
    default (1650, 1850) piecewise search to succeed, but too coarse for a
    narrow custom search_range to find a breakpoint in.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for dec in range(1650, 1850, 10):
        trend = 0.001 * (dec - 1650)
        for _ in range(n_per_decade):
            rows.append({
                "year": dec + int(rng.integers(0, 10)),
                score_col: trend + rng.normal(scale=0.01),
                "corpus_name": corpus,
                "genre_harmonized": genre,
            })
    return pd.DataFrame(rows)


def _make_arc_counts_df(genre="Fiction", corpus="c1", n_per_decade=5, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for dec in range(1650, 1850, 10):
        for _ in range(n_per_decade):
            rows.append({
                "year": dec + int(rng.integers(0, 10)),
                "pct_abstract": 0.2 + 0.0005 * (dec - 1650) + rng.normal(scale=0.01),
                "pct_concrete": 0.3 - 0.0003 * (dec - 1650) + rng.normal(scale=0.01),
                "corpus_name": corpus,
                "genre_harmonized": genre,
            })
    return pd.DataFrame(rows)


class TestNaNBreakpointGuards:
    """fit_piecewise returns NaN for pw_break_year when no candidate
    breakpoint in the search grid has >=10 points on each side. Because
    adjust_scores' own internal piecewise fit always uses fit_piecewise's
    hardcoded default search_range (it doesn't forward the caller's
    search_range/search_step), a caller-supplied narrow search_range can
    make the row-level fit_piecewise call fail (NaN breakpoint) even while
    adjust_scores' default-range fit still succeeds -- so a report row can
    genuinely reach the int()-conversion sites with a NaN breakpoint.
    """

    def test_report_arc_narrow_search_range_does_not_crash(self, capsys):
        df = _make_arc_scores_df()
        result = report_arc(
            combined_df=df, genres=["Fiction"],
            search_range=(1652, 1658), search_step=10,
            print_result=True,
        )
        assert len(result) == 1
        assert np.isnan(result.iloc[0]["breakpoint"])
        out = capsys.readouterr().out
        assert "insufficient data" in out
        assert "Traceback" not in out

    def test_report_arc_counts_narrow_search_range_does_not_crash(self, capsys):
        df = _make_arc_counts_df()
        result = report_arc_counts(
            combined_df=df, genres=["Fiction"],
            search_range=(1652, 1658), search_step=10,
            print_result=True,
        )
        assert len(result) == 1
        assert np.isnan(result.iloc[0]["abstract_breakpoint"])
        out = capsys.readouterr().out
        assert "insufficient data" in out
        assert "Traceback" not in out

    def test_report_full_handles_nan_breakpoint_and_mismatched_decades(self, monkeypatch):
        """Directly exercise report_full's own int(NaN) guards (summary
        table, per-genre detail, prose) via controlled score/count rows,
        decoupled from whichever internal fit happens to produce NaN in
        practice. Also exercises the §2.1 decade-mislabeling fix: the score
        and count analyses are given DIFFERENT key decades, and the count
        (word-proportion) table/prose must use the count analysis's own
        decades, not the score analysis's.
        """
        score_row = {
            "genre": "Fiction", "breakpoint": np.nan,
            "start_decade": 1700, "peak_decade": 1750, "end_decade": 1800,
            "raw_start": 0.1, "raw_peak": 0.3, "raw_end": 0.05,
            "raw_start_pooled": 0.11, "raw_peak_pooled": 0.31, "raw_end_pooled": 0.06,
            "rise_sd": 1.0, "fall_sd": 1.0,
            "slope_before": np.nan, "slope_before_p": np.nan,
            "slope_after": np.nan, "slope_after_p": np.nan,
            "r2": np.nan, "n_texts_total": 100,
        }
        count_row = {
            "genre": "Fiction", "n_texts": 100,
            "abstract_breakpoint": np.nan,
            "abstract_start_decade": 1690, "abstract_peak_decade": 1760,
            "abstract_end_decade": 1810,
            "abstract_pct_start": 10.0, "abstract_pct_peak": 20.0, "abstract_pct_end": 8.0,
            "conc_at_abs_start": 30.0, "conc_at_abs_peak": 25.0, "conc_at_abs_end": 35.0,
            "abs_conc_ratio_start": 0.33, "abs_conc_ratio_peak": 0.8, "abs_conc_ratio_end": 0.23,
            "abstract_r2": np.nan, "concrete_r2": np.nan,
        }
        monkeypatch.setattr(analysis, "report_arc",
                             lambda **kw: pd.DataFrame([score_row]))
        monkeypatch.setattr(analysis, "report_arc_counts",
                             lambda **kw: pd.DataFrame([count_row]))

        md, merged = report_full(scores_df=pd.DataFrame({"x": [1]}),
                                  counts_df=pd.DataFrame({"x": [1]}),
                                  genres=["Fiction"])

        assert "Traceback" not in md
        assert len(merged) == 1

        # Count-based rows/prose use the count analysis's OWN decades...
        assert "1690s (start)" in md
        assert "1760s (peak)" in md
        assert "1810s (end)" in md
        # ...not the score analysis's decades (which remain correct for
        # the Scores section).
        assert "1700s (start)" not in md
        assert "1750s (peak)" not in md
        assert "1800s (end)" not in md
        assert "1700s: 0.1000" in md  # Scores summary still uses sr's decades


# ---------------------------------------------------------------------------
# report_arc corpus-balanced magnitude reporting (audit §2.5)
# ---------------------------------------------------------------------------

def offset_share(offset, n_big, n_small):
    """Expected pooled-mean pull toward the minority corpus's offset,
    matching report_arc's sign convention (dec = -mean(score))."""
    return offset * n_small / (n_big + n_small)


class TestReportArcCorpusBalancedMagnitudes:
    """Key decades are selected on corpus-balanced (decade, corpus) means;
    the reported raw_start/raw_peak/raw_end magnitudes must come from that
    SAME aggregation, not a text-pooled mean that a single large corpus
    can dominate.
    """

    def _make_imbalanced_df(self, offset=5.0, n_big=100, n_small=5, seed=0):
        rng = np.random.default_rng(seed)
        rows = []
        for dec in range(1650, 1850, 10):
            trend = 0.01 * (dec - 1650)
            for _ in range(n_big):
                rows.append({
                    "year": dec + int(rng.integers(0, 10)),
                    "Abs-Conc.Median.median": trend,
                    "corpus_name": "big", "genre_harmonized": "Fiction",
                })
            for _ in range(n_small):
                rows.append({
                    "year": dec + int(rng.integers(0, 10)),
                    "Abs-Conc.Median.median": trend + offset,
                    "corpus_name": "small", "genre_harmonized": "Fiction",
                })
        return pd.DataFrame(rows)

    def test_raw_magnitudes_use_corpus_balanced_aggregation(self):
        offset, n_big, n_small = 5.0, 100, 5
        df = self._make_imbalanced_df(offset=offset, n_big=n_big, n_small=n_small)
        result = report_arc(combined_df=df, genres=["Fiction"], print_result=False)
        row = result.iloc[0]

        # The dominant "big" corpus (100 vs 5 texts/decade) would pull a
        # naive text-pooled mean far from a corpus-balanced one whenever
        # the corpora disagree by a fixed offset -- confirm the two are
        # now genuinely different (the bug made them silently identical
        # in effect, since only the pooled figure was ever reported).
        assert abs(row["raw_start"] - row["raw_start_pooled"]) > 0.1
        assert abs(row["raw_peak"] - row["raw_peak_pooled"]) > 0.1
        assert abs(row["raw_end"] - row["raw_end_pooled"]) > 0.1

        # The corpus-balanced value should recover ~the shared trend
        # (roughly the average of the two corpora after removing the
        # additive corpus offset via fixed effects), close to 0 at the
        # start decade (1650) given trend(1650) == 0.
        assert row["raw_start"] == pytest.approx(0.0, abs=0.05)
        # The naive pooled mean, by contrast, is pulled toward the small
        # corpus's offset (weighted ~5/105 of the way there).
        assert row["raw_start_pooled"] == pytest.approx(-offset_share(offset, n_big, n_small), abs=0.05)

    def test_pooled_columns_still_present_and_labeled(self):
        """The text-pooled numbers are kept (not deleted), just relabeled
        so they're not mistaken for the corpus-balanced headline figures."""
        df = self._make_imbalanced_df()
        result = report_arc(combined_df=df, genres=["Fiction"], print_result=False)
        for col in ("raw_start_pooled", "raw_peak_pooled", "raw_end_pooled"):
            assert col in result.columns
