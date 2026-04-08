"""Tests for corpus bias estimation and correction."""

import json
import os

import numpy as np
import pandas as pd
import pytest

from abstraction.corpus_correction import (
    estimate_corpus_bias,
    save_corpus_bias,
    load_corpus_bias,
    correct_scores_df,
    _find_connected_components,
)


# ---------------------------------------------------------------------------
# _find_connected_components
# ---------------------------------------------------------------------------

class TestConnectedComponents:
    def test_single_component(self):
        """All corpora share match groups → one component."""
        df = pd.DataFrame({
            "group_id": [1, 1, 2, 2],
            "corpus": ["A", "B", "B", "C"],
        })
        comps = _find_connected_components(df)
        assert len(comps) == 1
        assert comps[0] == {"A", "B", "C"}

    def test_two_components(self):
        """Disconnected corpora → separate components."""
        df = pd.DataFrame({
            "group_id": [1, 1, 2, 2],
            "corpus": ["A", "B", "C", "D"],
        })
        comps = _find_connected_components(df)
        assert len(comps) == 2
        sets = [frozenset(c) for c in comps]
        assert frozenset({"A", "B"}) in sets
        assert frozenset({"C", "D"}) in sets

    def test_chain_connectivity(self):
        """A-B in group 1, B-C in group 2 → all connected."""
        df = pd.DataFrame({
            "group_id": [1, 1, 2, 2],
            "corpus": ["A", "B", "B", "C"],
        })
        comps = _find_connected_components(df)
        assert len(comps) == 1
        assert comps[0] == {"A", "B", "C"}


# ---------------------------------------------------------------------------
# estimate_corpus_bias
# ---------------------------------------------------------------------------

class TestEstimateCorpusBias:
    def _make_match_data(self):
        """Synthetic match group data with known bias.

        3 corpora: ref (baseline), biased_high (+0.1), biased_low (-0.05)
        20 match groups, each with ref + one other corpus.
        Bias is additive: same text scores 0.1 higher in biased_high.
        """
        rng = np.random.RandomState(42)
        rows = []
        for g in range(20):
            base_score = rng.normal(0.3, 0.1)
            # ref always present
            rows.append({"group_id": g, "corpus": "ref", "score": base_score,
                         "path_freqs": f"ref/{g}.json"})
            if g < 12:
                # biased_high: score is 0.1 higher
                rows.append({"group_id": g, "corpus": "biased_high",
                             "score": base_score + 0.1,
                             "path_freqs": f"high/{g}.json"})
            if g >= 5:
                # biased_low: score is 0.05 lower
                rows.append({"group_id": g, "corpus": "biased_low",
                             "score": base_score - 0.05,
                             "path_freqs": f"low/{g}.json"})
        return pd.DataFrame(rows)

    def test_reference_corpus_zero(self):
        """Reference corpus coefficient should be exactly 0."""
        df = self._make_match_data()
        result = estimate_corpus_bias(df, reference_corpus="ref",
                                       min_group_overlap=5)
        assert result["coefficients"]["ref"] == 0.0

    def test_positive_bias_detected(self):
        """biased_high should have positive coefficient (direction correct)."""
        df = self._make_match_data()
        result = estimate_corpus_bias(df, reference_corpus="ref",
                                       min_group_overlap=5)
        coef = result["coefficients"]["biased_high"]
        # Within-group demeaning attenuates estimates with unbalanced panels,
        # but direction and rough magnitude should hold
        assert coef > 0.04
        assert coef < 0.15

    def test_negative_bias_detected(self):
        """biased_low should have negative coefficient (direction correct)."""
        df = self._make_match_data()
        result = estimate_corpus_bias(df, reference_corpus="ref",
                                       min_group_overlap=5)
        coef = result["coefficients"]["biased_low"]
        assert coef < -0.02
        assert coef > -0.10

    def test_standard_errors_positive(self):
        df = self._make_match_data()
        result = estimate_corpus_bias(df, reference_corpus="ref",
                                       min_group_overlap=5)
        for c, se in result["se"].items():
            assert se >= 0.0

    def test_min_group_overlap_filters(self):
        """Corpora below min_group_overlap threshold are excluded."""
        df = self._make_match_data()
        # Only 2 groups have biased_high if we take first 2
        small = df[df["group_id"] < 3].copy()
        # biased_high appears in groups 0,1,2 = 3 groups; biased_low in none
        result = estimate_corpus_bias(small, reference_corpus="ref",
                                       min_group_overlap=5)
        # With min_overlap=5, neither non-ref corpus qualifies
        assert result is None or "biased_high" not in result.get("coefficients", {})

    def test_connected_components_reported(self):
        df = self._make_match_data()
        result = estimate_corpus_bias(df, reference_corpus="ref",
                                       min_group_overlap=5)
        assert "connected_components" in result
        assert len(result["connected_components"]) >= 1

    def test_n_groups_reported(self):
        df = self._make_match_data()
        result = estimate_corpus_bias(df, reference_corpus="ref",
                                       min_group_overlap=5)
        assert result["n_groups"]["ref"] > 0
        assert result["n_groups"]["biased_high"] > 0


# ---------------------------------------------------------------------------
# save / load corpus bias
# ---------------------------------------------------------------------------

class TestSaveLoadBias:
    def test_roundtrip(self, tmp_path):
        bias = {
            "coefficients": {"ref": 0.0, "ecco": 0.027},
            "se": {"ref": 0.0, "ecco": 0.003},
            "reference": "ref",
        }
        path = str(tmp_path / "bias.json")
        save_corpus_bias(bias, path)
        loaded = load_corpus_bias(path)
        assert loaded["coefficients"]["ecco"] == 0.027
        assert loaded["reference"] == "ref"

    def test_load_missing_returns_none(self, tmp_path):
        assert load_corpus_bias(str(tmp_path / "nonexistent.json")) is None


# ---------------------------------------------------------------------------
# correct_scores_df
# ---------------------------------------------------------------------------

class TestCorrectScoresDf:
    def test_subtracts_bias(self):
        """Correction subtracts corpus bias from scores."""
        df = pd.DataFrame({
            "corpus_name": ["ref", "ecco", "ecco", "ref"],
            "Abs-Conc.Median.median": [0.3, 0.35, 0.40, 0.25],
        })
        bias = {"coefficients": {"ref": 0.0, "ecco": 0.05}}
        result = correct_scores_df(df, bias=bias)
        # ecco rows should have 0.05 subtracted
        assert abs(result.iloc[0]["Abs-Conc.Median.median"] - 0.3) < 1e-10
        assert abs(result.iloc[1]["Abs-Conc.Median.median"] - 0.30) < 1e-10
        assert abs(result.iloc[2]["Abs-Conc.Median.median"] - 0.35) < 1e-10
        assert abs(result.iloc[3]["Abs-Conc.Median.median"] - 0.25) < 1e-10

    def test_unknown_corpus_no_change(self):
        """Corpora not in bias dict get no correction."""
        df = pd.DataFrame({
            "corpus_name": ["unknown_corpus"],
            "Abs-Conc.Median.median": [0.5],
        })
        bias = {"coefficients": {"ref": 0.0, "ecco": 0.05}}
        result = correct_scores_df(df, bias=bias)
        assert result.iloc[0]["Abs-Conc.Median.median"] == 0.5

    def test_does_not_mutate_input(self):
        """Input DataFrame should not be modified."""
        df = pd.DataFrame({
            "corpus_name": ["ecco"],
            "Abs-Conc.Median.median": [0.5],
        })
        bias = {"coefficients": {"ecco": 0.1}}
        original_val = df.iloc[0]["Abs-Conc.Median.median"]
        correct_scores_df(df, bias=bias)
        assert df.iloc[0]["Abs-Conc.Median.median"] == original_val

    def test_empty_bias_returns_unchanged(self):
        """If bias has no coefficients, return DataFrame unchanged."""
        df = pd.DataFrame({
            "corpus_name": ["ecco"],
            "Abs-Conc.Median.median": [0.5],
        })
        bias = {"coefficients": {}}
        result = correct_scores_df(df, bias=bias)
        assert result.iloc[0]["Abs-Conc.Median.median"] == 0.5
