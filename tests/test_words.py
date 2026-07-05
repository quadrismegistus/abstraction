"""Unit tests for abstraction.words."""

import numpy as np
import pandas as pd
import pytest

from abstraction.words import (
    correlate_words_with_trend,
    word_contributions,
    word_score_shifts,
)


# ---------------------------------------------------------------------------
# Fixtures: fake decade × word frequency matrix and fake norms
# ---------------------------------------------------------------------------

FAKE_NORMS_DATA = {
    "virtue": -1.8,
    "reason": -1.5,
    "hope": -1.3,
    "stone": 2.0,
    "wall": 2.1,
    "hair": 1.8,
    "the": 0.0,
    "old": 0.3,
    "big": 0.5,
    "callous": 0.2,
    "role": -0.5,
}

FAKE_VECNORMS = {
    "Abs-Conc.Median.C17": {
        "virtue": -0.99, "reason": -0.80, "stone": 1.93,
        "wall": 1.80, "hope": -0.70, "hair": 1.50,
        "callous": 1.58, "role": 1.99,
    },
    "Abs-Conc.Median.C19": {
        "virtue": -1.65, "reason": -1.40, "stone": 1.07,
        "wall": 1.50, "hope": -1.10, "hair": 1.60,
        "callous": -1.23, "role": -0.83,
    },
    "Abs-Conc.Median.C16": {
        "virtue": -0.50, "reason": -0.60, "stone": 2.00,
        "wall": 1.90, "hope": -0.50, "hair": 1.40,
        "callous": 1.70, "role": 2.10,
    },
    "Abs-Conc.Median.C18": {
        "virtue": -1.20, "reason": -1.00, "stone": 1.50,
        "wall": 1.60, "hope": -0.90, "hair": 1.55,
        "callous": 0.50, "role": 0.60,
    },
    "Abs-Conc.Median.C20": {
        "virtue": -1.70, "reason": -1.50, "stone": 1.00,
        "wall": 1.40, "hope": -1.20, "hair": 1.65,
        "callous": -1.50, "role": -1.00,
    },
}


@pytest.fixture
def fake_decade_freqs():
    """A small decade × word frequency DataFrame with a clear trend.

    Decades 1700-1800: abstract words frequent, concrete words rare.
    Decades 1810-1900: abstract words decline, concrete words rise.
    """
    decades = list(range(1700, 1910, 10))
    n = len(decades)
    data = {}
    # Abstract words: high early, declining
    for w in ["virtue", "reason", "hope"]:
        data[w] = [1000 - i * 80 for i in range(n)]
    # Concrete words: low early, rising
    for w in ["stone", "wall", "hair"]:
        data[w] = [200 + i * 80 for i in range(n)]
    # Neutral: roughly flat, but with slight per-decade variation so the
    # words don't end up bit-for-bit identical to each other. Perfectly
    # identical (degenerate) trajectories are the scenario that used to
    # trigger a scipy "precision loss" RuntimeWarning out of zscore() when
    # min_total_freq filtered everything else away (see
    # test_min_total_freq_filters below and words.py:244/AUDIT-2026-07-04 §5).
    for i_w, w in enumerate(["the", "old", "big"]):
        data[w] = [5000 + ((i + i_w) % 3 - 1) * 15 for i in range(n)]
    return pd.DataFrame(data, index=decades)


@pytest.fixture
def patch_norms(monkeypatch):
    """Patch get_allnorms to return fake norms."""
    col = "Abs-Conc.Median.median"
    fake_df = pd.DataFrame({col: FAKE_NORMS_DATA})
    fake_df.index.name = "word"

    # Add vector norm columns
    for vcol, vals in FAKE_VECNORMS.items():
        for word in fake_df.index:
            if word not in vals:
                vals[word] = np.nan
        fake_df[vcol] = fake_df.index.map(vals)

    monkeypatch.setattr("abstraction.words.get_allnorms", lambda **kw: fake_df)


@pytest.fixture
def patch_norms_duplicated(monkeypatch):
    """Same fake norms as patch_norms, but with a duplicated index entry.

    Reproduces AUDIT-2026-07-04 §4.12/§5: production allnorms frames
    demonstrably carry duplicate word-index rows (scoring.py dedups
    defensively in nine places); correlate_words_with_trend() and
    word_contributions() were the two consumers that didn't.
    """
    col = "Abs-Conc.Median.median"
    fake_df = pd.DataFrame({col: FAKE_NORMS_DATA})
    fake_df.index.name = "word"

    for vcol, vals in FAKE_VECNORMS.items():
        for word in fake_df.index:
            if word not in vals:
                vals[word] = np.nan
        fake_df[vcol] = fake_df.index.map(vals)

    # Duplicate a row that's actually used by the test fixtures below.
    dup_df = pd.concat([fake_df, fake_df.loc[["stone"]]])
    monkeypatch.setattr("abstraction.words.get_allnorms", lambda **kw: dup_df)


# ---------------------------------------------------------------------------
# correlate_words_with_trend
# ---------------------------------------------------------------------------

class TestCorrelateWordsWithTrend:
    def test_returns_dataframe(self, fake_decade_freqs, patch_norms):
        result = correlate_words_with_trend(fake_decade_freqs, min_total_freq=0)
        assert isinstance(result, pd.DataFrame)
        assert "word" in result.columns
        assert "correlation" in result.columns
        assert "z_score" in result.columns
        assert "category" in result.columns

    def test_abstract_words_anticorrelate(self, fake_decade_freqs, patch_norms):
        """Abstract words decline as trend concretizes → negative correlation."""
        result = correlate_words_with_trend(fake_decade_freqs, min_total_freq=0)
        virtue = result[result["word"] == "virtue"].iloc[0]
        assert virtue["correlation"] < 0

    def test_concrete_words_correlate(self, fake_decade_freqs, patch_norms):
        """Concrete words rise as trend concretizes → positive correlation."""
        result = correlate_words_with_trend(fake_decade_freqs, min_total_freq=0)
        stone = result[result["word"] == "stone"].iloc[0]
        assert stone["correlation"] > 0

    def test_sorted_ascending(self, fake_decade_freqs, patch_norms):
        result = correlate_words_with_trend(fake_decade_freqs, min_total_freq=0)
        assert result["correlation"].iloc[0] <= result["correlation"].iloc[-1]

    def test_categories_assigned(self, fake_decade_freqs, patch_norms):
        result = correlate_words_with_trend(fake_decade_freqs, min_total_freq=0)
        virtue = result[result["word"] == "virtue"].iloc[0]
        stone = result[result["word"] == "stone"].iloc[0]
        the = result[result["word"] == "the"].iloc[0]
        assert virtue["category"] == "Abstract"
        assert stone["category"] == "Concrete"
        assert the["category"] == "Neither"

    def test_min_total_freq_filters(self, fake_decade_freqs, patch_norms):
        result_all = correlate_words_with_trend(fake_decade_freqs, min_total_freq=0)
        result_high = correlate_words_with_trend(fake_decade_freqs, min_total_freq=50000)
        assert len(result_high) < len(result_all)

    def test_pearson_method(self, fake_decade_freqs, patch_norms):
        result = correlate_words_with_trend(fake_decade_freqs, min_total_freq=0,
                                            method="pearson")
        assert len(result) > 0
        virtue = result[result["word"] == "virtue"].iloc[0]
        assert virtue["correlation"] < 0


# ---------------------------------------------------------------------------
# word_contributions
# ---------------------------------------------------------------------------

class TestWordContributions:
    def test_returns_dataframe(self, fake_decade_freqs, patch_norms):
        result = word_contributions(fake_decade_freqs,
                                    period_early=(1700, 1760),
                                    period_late=(1840, 1910),
                                    min_total_freq=0)
        assert isinstance(result, pd.DataFrame)
        assert "word" in result.columns
        assert "contribution" in result.columns
        assert "freq_change" in result.columns

    def test_abstract_decline_concretizes(self, fake_decade_freqs, patch_norms):
        """Abstract word declining → positive contribution (concretizing)."""
        result = word_contributions(fake_decade_freqs,
                                    period_early=(1700, 1760),
                                    period_late=(1840, 1910),
                                    min_total_freq=0)
        virtue = result[result["word"] == "virtue"].iloc[0]
        # virtue: negative z, frequency declined → freq_change negative, z negative
        # contribution = negative × negative = positive (concretizing)
        assert virtue["contribution"] > 0
        assert virtue["freq_change"] < 0

    def test_concrete_rise_concretizes(self, fake_decade_freqs, patch_norms):
        """Concrete word rising → positive contribution (concretizing)."""
        result = word_contributions(fake_decade_freqs,
                                    period_early=(1700, 1760),
                                    period_late=(1840, 1910),
                                    min_total_freq=0)
        stone = result[result["word"] == "stone"].iloc[0]
        assert stone["contribution"] > 0
        assert stone["freq_change"] > 0

    def test_sorted_descending(self, fake_decade_freqs, patch_norms):
        result = word_contributions(fake_decade_freqs,
                                    period_early=(1700, 1760),
                                    period_late=(1840, 1910),
                                    min_total_freq=0)
        assert result["contribution"].iloc[0] >= result["contribution"].iloc[-1]

    def test_bad_period_raises(self, fake_decade_freqs, patch_norms):
        with pytest.raises(ValueError, match="No decades"):
            word_contributions(fake_decade_freqs,
                               period_early=(1200, 1300),
                               period_late=(1840, 1910),
                               min_total_freq=0)


# ---------------------------------------------------------------------------
# word_score_shifts
# ---------------------------------------------------------------------------

class TestWordScoreShifts:
    def test_returns_dataframe(self, patch_norms):
        result = word_score_shifts(source="Median", period_early="C17",
                                   period_late="C19", min_periods=2)
        assert isinstance(result, pd.DataFrame)
        assert "word" in result.columns
        assert "score_shift" in result.columns
        assert "score_early" in result.columns
        assert "score_late" in result.columns

    def test_callous_became_abstract(self, patch_norms):
        """callous: +1.58 in C17 → -1.23 in C19, should have negative shift."""
        result = word_score_shifts(source="Median", period_early="C17",
                                   period_late="C19", min_periods=2)
        callous = result[result["word"] == "callous"].iloc[0]
        assert callous["score_shift"] < -2.0

    def test_sorted_descending_by_shift(self, patch_norms):
        result = word_score_shifts(source="Median", period_early="C17",
                                   period_late="C19", min_periods=2)
        assert result["score_shift"].iloc[0] >= result["score_shift"].iloc[-1]

    def test_trajectory_column(self, patch_norms):
        result = word_score_shifts(source="Median", period_early="C17",
                                   period_late="C19", min_periods=2)
        traj = result.iloc[0]["trajectory"]
        assert isinstance(traj, dict)
        assert "Abs-Conc.Median.C17" in traj

    def test_bad_column_raises(self, patch_norms):
        with pytest.raises(ValueError, match="Unknown column"):
            word_score_shifts(source="Median", period_early="C99",
                              period_late="C19")


# ---------------------------------------------------------------------------
# Regression tests for AUDIT-2026-07-04 §4.12 / §5 (words.py)
# ---------------------------------------------------------------------------

class TestDuplicateAllnormsIndex:
    """A duplicated word in allnorms used to crash or misalign these two
    functions (words.py ~192-213, ~286-318); scoring.py dedups defensively
    but these functions didn't.
    """

    def test_correlate_words_with_trend_survives_duplicates(
        self, fake_decade_freqs, patch_norms_duplicated
    ):
        result = correlate_words_with_trend(fake_decade_freqs, min_total_freq=0)
        assert isinstance(result, pd.DataFrame)
        # "stone" must appear exactly once despite the duplicated allnorms row.
        assert (result["word"] == "stone").sum() == 1

    def test_word_contributions_survives_duplicates(
        self, fake_decade_freqs, patch_norms_duplicated
    ):
        result = word_contributions(fake_decade_freqs,
                                    period_early=(1700, 1760),
                                    period_late=(1840, 1910),
                                    min_total_freq=0)
        assert isinstance(result, pd.DataFrame)
        assert (result["word"] == "stone").sum() == 1


class TestNaNZScoreHandling:
    """A word with an exactly-constant (here: always-zero) frequency
    trajectory makes np.corrcoef return NaN under method='pearson'
    (division by a zero std). That single NaN used to poison every other
    word's correlation_z via zscore()'s default nan_policy='propagate'.
    """

    def test_constant_word_does_not_poison_other_zscores(self, patch_norms):
        decades = list(range(1700, 1910, 10))
        n = len(decades)
        data = {
            "virtue": [1000 - i * 80 for i in range(n)],
            "stone": [200 + i * 80 for i in range(n)],
            # "role" is in FAKE_NORMS_DATA but never occurs in these texts,
            # so its proportion trajectory is exactly 0.0 every decade.
            "role": [0] * n,
        }
        df = pd.DataFrame(data, index=decades)

        result = correlate_words_with_trend(df, min_total_freq=0, method="pearson")

        role = result[result["word"] == "role"].iloc[0]
        others = result[result["word"] != "role"]
        assert np.isnan(role["correlation"])
        assert others["correlation_z"].notna().all()
        assert np.isfinite(others["correlation_z"]).all()


class TestCosineMeasuresComovement:
    """Cosine similarity of non-centered, all-positive frequency proportions
    mostly reflects shared *level*, not co-movement, even though the output
    column is named "correlation". correlate_words_with_trend() now
    mean-centers both vectors before the cosine so it behaves like Pearson's
    r (words.py ~216-227 / AUDIT-2026-07-04 §5).
    """

    def test_flat_word_is_near_zero_even_when_trend_is_always_positive(
        self, patch_norms
    ):
        decades = list(range(1700, 1910, 10))
        n = len(decades)
        # "big" (z=+0.5) dominates and rises; "role" (z=-0.5) is a small,
        # slowly-declining minority. The resulting weighted trend is always
        # positive (never crosses zero) but trends upward. "old" (z=+0.3) is
        # perfectly flat and has zero true relationship to that trend.
        data = {
            "role": [200 - i * 5 for i in range(n)],
            "big": [800 + i * 5 for i in range(n)],
            "old": [1000] * n,
        }
        df = pd.DataFrame(data, index=decades)

        # Sanity-check the premise: the trend never crosses zero.
        trend_col = "Abs-Conc.Median.median"
        allnorms = pd.DataFrame({trend_col: FAKE_NORMS_DATA})
        freq_df = df[sorted(data)]
        prop_df = freq_df.div(freq_df.sum(axis=1), axis=0)
        trend = prop_df.values @ allnorms.loc[prop_df.columns, trend_col].values
        assert (trend > 0).all()

        result = correlate_words_with_trend(df, min_total_freq=0)
        old = result[result["word"] == "old"].iloc[0]
        big = result[result["word"] == "big"].iloc[0]
        role = result[result["word"] == "role"].iloc[0]

        # Flat word: no real co-movement with the trend, regardless of the
        # trend's overall (always-positive) level.
        assert abs(old["correlation"]) < 1e-6
        # Rising concrete word tracks the (rising) trend closely...
        assert big["correlation"] > 0.9
        # ...and the declining word anti-correlates, even though its own
        # frequency values stay positive throughout.
        assert role["correlation"] < -0.9


# ---------------------------------------------------------------------------
# freq_change_pct: words absent in the early period (AUDIT-2026-07-04 §5)
# ---------------------------------------------------------------------------

class TestFreqChangePct:
    def test_absent_early_word_gives_nan_not_huge_pct(self, patch_norms):
        """A word with zero frequency in the early period used to produce
        ~1e12% via `clip(lower=1e-10)` on the denominator; it should now be
        NaN (undefined percent change from nothing).
        """
        decades = list(range(1700, 1910, 10))
        n = len(decades)
        data = {
            "virtue": [1000 - i * 80 for i in range(n)],
            "stone": [200 + i * 80 for i in range(n)],
            # "role" only appears in the late period.
            "role": [0 if year < 1840 else 500 for year in decades],
        }
        df = pd.DataFrame(data, index=decades)

        result = word_contributions(df,
                                    period_early=(1700, 1760),
                                    period_late=(1840, 1910),
                                    min_total_freq=0)
        role = result[result["word"] == "role"].iloc[0]
        assert role["freq_early"] == 0
        assert np.isnan(role["freq_change_pct"])
        # Every other word's pct should stay finite and sane (not blown up
        # by the fix).
        others = result[result["word"] != "role"]
        assert np.isfinite(others["freq_change_pct"]).all()
        assert (others["freq_change_pct"].abs() < 1000).all()
