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
    # Neutral: flat
    for w in ["the", "old", "big"]:
        data[w] = [5000] * n
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
