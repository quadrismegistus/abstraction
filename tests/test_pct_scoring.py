"""Tests for _pct_ frequency proportion scoring and period assignment."""

import numpy as np
import pandas as pd

from abstraction.scoring import _score_freqs_dict_allnorms
from abstraction.analysis import assign_period_score, CENTURY_BINS

# Norm caches (_NORM_DICTS, _NORMS_ARRAYS_CACHE) are cleared per-test by the
# autouse fixture in tests/conftest.py.


# ---------------------------------------------------------------------------
# _pct_ column generation
# ---------------------------------------------------------------------------

class TestPctColumns:
    """Verify _pct_abs/conc/absconc columns are computed correctly."""

    def _make_norms(self):
        """Norms with known abstract/concrete words across two periods."""
        return pd.DataFrame({
            "Abs-Conc.Median.median": {
                "virtue": -1.5,   # abstract (below -1.0 and -0.5)
                "justice": -0.8,  # abstract at 0.5 cutoff only
                "face": 0.3,      # neither
                "rock": 1.2,      # concrete (above 1.0 and 0.5)
                "stone": 0.7,     # concrete at 0.5 cutoff only
            },
            "Abs-Conc.Median.C18": {
                "virtue": -1.8,
                "justice": -1.2,
                "face": 0.1,
                "rock": 1.5,
                "stone": 0.9,
            },
        })

    def test_pct_columns_present(self):
        """Scoring produces _pct_ columns for each period in allnorms."""
        norms = self._make_norms()
        freqs = {"virtue": 1, "rock": 1}
        result = _score_freqs_dict_allnorms(freqs, norms)
        # Should have _pct_ for both median and C18
        assert "_pct_abs_10_median" in result
        assert "_pct_conc_10_median" in result
        assert "_pct_absconc_10_median" in result
        assert "_pct_abs_05_C18" in result
        assert "_pct_conc_05_C18" in result

    def test_pct_values_at_10_cutoff(self):
        """At ±1.0 cutoff: virtue is abstract, rock is concrete."""
        norms = self._make_norms()
        # Equal weights: 2 words, each count=1
        freqs = {"virtue": 1, "rock": 1}
        result = _score_freqs_dict_allnorms(freqs, norms)
        # virtue=-1.5 <= -1.0 → abstract; rock=1.2 >= 1.0 → concrete
        assert result["_pct_abs_10_median"] == 0.5   # 1/2
        assert result["_pct_conc_10_median"] == 0.5   # 1/2
        assert abs(result["_pct_absconc_10_median"]) < 1e-10  # 0.5 - 0.5 = 0

    def test_pct_values_at_05_cutoff(self):
        """At ±0.5 cutoff: more words qualify as abstract/concrete."""
        norms = self._make_norms()
        # All 5 words, equal count
        freqs = {"virtue": 1, "justice": 1, "face": 1, "rock": 1, "stone": 1}
        result = _score_freqs_dict_allnorms(freqs, norms)
        # At ±0.5: virtue(-1.5), justice(-0.8) are abstract; rock(1.2), stone(0.7) are concrete
        assert abs(result["_pct_abs_05_median"] - 2/5) < 1e-10
        assert abs(result["_pct_conc_05_median"] - 2/5) < 1e-10
        assert abs(result["_pct_absconc_05_median"]) < 1e-10

    def test_pct_weighted_by_frequency(self):
        """Proportions are frequency-weighted, not type-weighted."""
        norms = self._make_norms()
        # virtue appears 9 times, rock appears 1 time
        freqs = {"virtue": 9, "rock": 1}
        result = _score_freqs_dict_allnorms(freqs, norms)
        # At ±1.0: virtue is abstract (9/10), rock is concrete (1/10)
        assert abs(result["_pct_abs_10_median"] - 0.9) < 1e-10
        assert abs(result["_pct_conc_10_median"] - 0.1) < 1e-10
        assert abs(result["_pct_absconc_10_median"] - 0.8) < 1e-10

    def test_pct_different_periods(self):
        """Different period norms produce different _pct_ values."""
        norms = self._make_norms()
        # justice: -0.8 in median (not abstract at ±1.0), -1.2 in C18 (abstract at ±1.0)
        freqs = {"justice": 1, "face": 1}
        result = _score_freqs_dict_allnorms(freqs, norms)
        # median: justice=-0.8 not abstract at 1.0 cutoff
        assert result["_pct_abs_10_median"] == 0.0
        # C18: justice=-1.2 IS abstract at 1.0 cutoff
        assert result["_pct_abs_10_C18"] == 0.5

    def test_pct_all_abstract_text(self):
        """100% abstract text should have pct_abs=1.0, pct_conc=0.0."""
        norms = self._make_norms()
        freqs = {"virtue": 5}
        result = _score_freqs_dict_allnorms(freqs, norms)
        assert result["_pct_abs_10_median"] == 1.0
        assert result["_pct_conc_10_median"] == 0.0
        assert result["_pct_absconc_10_median"] == 1.0

    def test_pct_all_concrete_text(self):
        """100% concrete text should have pct_abs=0.0, pct_conc=1.0."""
        norms = self._make_norms()
        freqs = {"rock": 5}
        result = _score_freqs_dict_allnorms(freqs, norms)
        assert result["_pct_abs_10_median"] == 0.0
        assert result["_pct_conc_10_median"] == 1.0
        assert result["_pct_absconc_10_median"] == -1.0

    def test_pct_neutral_text(self):
        """Text with only 'neither' words should have pct_abs=0, pct_conc=0."""
        norms = self._make_norms()
        freqs = {"face": 10}  # face=0.3, neither at any cutoff
        result = _score_freqs_dict_allnorms(freqs, norms)
        assert result["_pct_abs_10_median"] == 0.0
        assert result["_pct_conc_10_median"] == 0.0
        assert result["_pct_absconc_10_median"] == 0.0

    def test_no_pct_for_non_median_prefix(self):
        """_pct_ columns only generated for Abs-Conc.Median.* columns."""
        norms = pd.DataFrame({
            "Abs-Conc.PAV-Conc.orig": {"rock": 1.5, "virtue": -1.5},
        })
        freqs = {"rock": 1, "virtue": 1}
        result = _score_freqs_dict_allnorms(freqs, norms)
        # Should have the score but no _pct_ columns
        assert "Abs-Conc.PAV-Conc.orig" in result
        pct_keys = [k for k in result if k.startswith("_pct_")]
        assert len(pct_keys) == 0


# ---------------------------------------------------------------------------
# assign_period_score
# ---------------------------------------------------------------------------

class TestAssignPeriodScore:
    """Verify year-to-century mapping for period-matched scoring."""

    def _make_df(self, years):
        """Create a DataFrame with scores for multiple periods."""
        data = {"year": years}
        for label in ["C16", "C17", "C18", "C19", "C20", "median"]:
            # Each period gets a distinct value so we can verify which was picked
            val = {"C16": 0.1, "C17": 0.2, "C18": 0.3, "C19": 0.4,
                   "C20": 0.5, "median": 0.99}[label]
            data[f"Abs-Conc.Median.{label}"] = [val] * len(years)
        return pd.DataFrame(data)

    def test_c17_text(self):
        df = self._make_df([1650])
        result = assign_period_score(df)
        assert result.iloc[0]["norm_period"] == "C17"
        assert result.iloc[0]["period_score"] == 0.2

    def test_c18_text(self):
        df = self._make_df([1750])
        result = assign_period_score(df)
        assert result.iloc[0]["norm_period"] == "C18"
        assert result.iloc[0]["period_score"] == 0.3

    def test_c19_text(self):
        df = self._make_df([1850])
        result = assign_period_score(df)
        assert result.iloc[0]["norm_period"] == "C19"
        assert result.iloc[0]["period_score"] == 0.4

    def test_boundary_year_1700(self):
        """1700 should be C18, not C17 (bins are [lo, hi))."""
        df = self._make_df([1700])
        result = assign_period_score(df)
        assert result.iloc[0]["norm_period"] == "C18"
        assert result.iloc[0]["period_score"] == 0.3

    def test_boundary_year_1600(self):
        """1600 should be C17."""
        df = self._make_df([1600])
        result = assign_period_score(df)
        assert result.iloc[0]["norm_period"] == "C17"

    def test_boundary_year_1500(self):
        """1500 should be C16."""
        df = self._make_df([1500])
        result = assign_period_score(df)
        assert result.iloc[0]["norm_period"] == "C16"
        assert result.iloc[0]["period_score"] == 0.1

    def test_year_2000_uses_c20(self):
        """C20 norms cover 1900-2100."""
        df = self._make_df([2000])
        result = assign_period_score(df)
        assert result.iloc[0]["norm_period"] == "C20"
        assert result.iloc[0]["period_score"] == 0.5

    def test_nan_year_gets_median(self):
        """Text with NaN year falls back to median."""
        df = self._make_df([np.nan])
        result = assign_period_score(df)
        assert result.iloc[0]["norm_period"] == "median"
        assert result.iloc[0]["period_score"] == 0.99

    def test_multiple_centuries(self):
        """Mixed years get correct period-specific scores."""
        df = self._make_df([1550, 1650, 1750, 1850, 1950])
        result = assign_period_score(df)
        expected = [
            ("C16", 0.1),
            ("C17", 0.2),
            ("C18", 0.3),
            ("C19", 0.4),
            ("C20", 0.5),
        ]
        for i, (period, score) in enumerate(expected):
            assert result.iloc[i]["norm_period"] == period
            assert result.iloc[i]["period_score"] == score

    def test_missing_period_column_falls_back(self):
        """If a period column is missing, fall back to median."""
        df = pd.DataFrame({
            "year": [1650],
            "Abs-Conc.Median.median": [0.99],
            # No C17 column
        })
        result = assign_period_score(df)
        assert result.iloc[0]["period_score"] == 0.99
        assert result.iloc[0]["norm_period"] == "median"

    def test_nan_score_falls_back_to_median(self):
        """If period score is NaN, fall back to median."""
        df = pd.DataFrame({
            "year": [1750],
            "Abs-Conc.Median.C18": [np.nan],
            "Abs-Conc.Median.median": [0.99],
        })
        result = assign_period_score(df)
        assert result.iloc[0]["period_score"] == 0.99

    def test_custom_source(self):
        """Works with non-Median sources."""
        df = pd.DataFrame({
            "year": [1750],
            "Abs-Conc.MRC-Conc.C18": [0.42],
            "Abs-Conc.MRC-Conc.median": [0.99],
        })
        result = assign_period_score(df, source="MRC-Conc")
        assert result.iloc[0]["period_score"] == 0.42
        assert result.iloc[0]["norm_period"] == "C18"
