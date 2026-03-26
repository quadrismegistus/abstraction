import numpy as np
import pandas as pd
import pytest

from abstraction.scoring import score_freqs, score_words


class TestScoreFreqs:
    def _patch_norms(self, monkeypatch):
        """Patch get_norm_dict to return a small known dictionary."""
        fake = {"rock": 1.5, "virtue": -1.8, "justice": -1.3, "face": 0.2}
        monkeypatch.setattr(
            "abstraction.scoring._NORM_DICTS",
            {"Abs-Conc.Median.median": fake},
        )
        return fake

    def test_basic(self, monkeypatch):
        self._patch_norms(monkeypatch)
        score = score_freqs({"rock": 2, "virtue": 2})
        expected = (1.5 * 2 + -1.8 * 2) / 4
        assert abs(score - expected) < 1e-6

    def test_unknown_words_ignored(self, monkeypatch):
        self._patch_norms(monkeypatch)
        score = score_freqs({"rock": 1, "xyzzy": 100})
        assert abs(score - 1.5) < 1e-6

    def test_empty_freqs(self, monkeypatch):
        self._patch_norms(monkeypatch)
        assert np.isnan(score_freqs({}))

    def test_all_unknown(self, monkeypatch):
        self._patch_norms(monkeypatch)
        assert np.isnan(score_freqs({"xyzzy": 5, "qqq": 3}))

    def test_case_insensitive(self, monkeypatch):
        self._patch_norms(monkeypatch)
        score = score_freqs({"Rock": 1, "VIRTUE": 1})
        expected = (1.5 + -1.8) / 2
        assert abs(score - expected) < 1e-6


class TestScoreWords:
    def _patch_norms(self, monkeypatch):
        fake = {"rock": 1.5, "virtue": -1.8, "justice": -1.3, "face": 0.2}
        monkeypatch.setattr(
            "abstraction.scoring._NORM_DICTS",
            {"Abs-Conc.Median.median": fake},
        )

    def test_returns_dataframe(self, monkeypatch):
        self._patch_norms(monkeypatch)
        df = score_words("the rock of virtue")
        assert isinstance(df, pd.DataFrame)
        assert "word" in df.columns
        assert "score" in df.columns
        assert "position" in df.columns
        assert "is_abstract" in df.columns
        assert "is_concrete" in df.columns

    def test_known_words_scored(self, monkeypatch):
        self._patch_norms(monkeypatch)
        df = score_words("rock and virtue")
        rock = df[df["word"] == "rock"].iloc[0]
        assert rock["score"] == 1.5
        assert rock["is_concrete"] == True
        assert rock["is_abstract"] == False
        virtue = df[df["word"] == "virtue"].iloc[0]
        assert virtue["score"] == -1.8
        assert virtue["is_abstract"] == True

    def test_unknown_words_nan(self, monkeypatch):
        self._patch_norms(monkeypatch)
        df = score_words("the rock")
        the_row = df[df["word"] == "the"].iloc[0]
        assert np.isnan(the_row["score"])

    def test_empty_text(self, monkeypatch):
        self._patch_norms(monkeypatch)
        df = score_words("")
        assert len(df) == 0

    def test_positions_sequential(self, monkeypatch):
        self._patch_norms(monkeypatch)
        df = score_words("rock face virtue justice")
        assert list(df["position"]) == sorted(df["position"].tolist())
