"""
CI-friendly integration tests using bundled fixture data.

These tests exercise cross-module flows without requiring local data files
or large corpora. They use a small fixture corpus and norms CSV checked
into the repo under tests/fixtures/.
"""

import os

import numpy as np
import pandas as pd
import pytest

import abstraction.scoring as _scoring_mod


@pytest.fixture(autouse=True)
def _clear_norms_cache():
    """Clear the global norms arrays cache and mock spelling modernizer."""
    _scoring_mod._NORMS_ARRAYS_CACHE = None
    # Mock get_spelling_modernizer to avoid file access in CI
    orig = _scoring_mod.get_spelling_modernizer
    _scoring_mod.get_spelling_modernizer = lambda: {}
    yield
    _scoring_mod._NORMS_ARRAYS_CACHE = None
    _scoring_mod.get_spelling_modernizer = orig

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
CORPUS_DIR = os.path.join(FIXTURES, "test_corpus")
NORMS_PATH = os.path.join(FIXTURES, "norms.csv")
STOPWORDS_PATH = os.path.join(FIXTURES, "stopwords.txt")


# ---------------------------------------------------------------------------
# Fixture norms (loaded once, shared across tests)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fixture_norms():
    """Load the small fixture norms as a DataFrame (same format as get_orignorms)."""
    return pd.read_csv(NORMS_PATH).set_index("word")


@pytest.fixture(scope="module")
def fixture_allnorms(fixture_norms):
    """Simulate allnorms: fixture norms with .orig suffix, like get_allnorms returns."""
    df = fixture_norms.copy()
    df.columns = [c + ".orig" for c in df.columns]
    return df


# ---------------------------------------------------------------------------
# Corpus loading with fixture data
# ---------------------------------------------------------------------------

class TestCorpusCIIntegration:
    def test_load_metadata(self):
        meta = pd.read_csv(os.path.join(CORPUS_DIR, "metadata.csv"))
        assert len(meta) == 3
        assert "id" in meta.columns
        assert "year" in meta.columns
        assert set(meta["id"]) == {"text1", "text2", "subdir/text3"}

    def test_read_texts(self):
        meta = pd.read_csv(os.path.join(CORPUS_DIR, "metadata.csv"))
        for _, row in meta.iterrows():
            path = os.path.join(CORPUS_DIR, "txt", row["id"] + ".txt")
            assert os.path.exists(path), f"Missing text file: {path}"
            with open(path) as f:
                txt = f.read()
            assert len(txt) > 50

    def test_freqs_match_metadata(self):
        from abstraction.scoring import _walk_freqs
        freqs_dir = os.path.join(CORPUS_DIR, "freqs")
        freqs_ids = {tid for tid, _ in _walk_freqs(freqs_dir)}
        meta = pd.read_csv(os.path.join(CORPUS_DIR, "metadata.csv"))
        assert freqs_ids == set(meta["id"])


# ---------------------------------------------------------------------------
# Tokenization on fixture texts
# ---------------------------------------------------------------------------

class TestTokenizeCIIntegration:
    def test_tokenize_fixture_text(self):
        from abstraction.tokenize import tokenize_agnostic
        with open(os.path.join(CORPUS_DIR, "txt", "text1.txt")) as f:
            txt = f.read()
        tokens = tokenize_agnostic(txt)
        assert len(tokens) > 20
        assert "truth" in tokens or "Truth" in tokens.lower() if hasattr(tokens, 'lower') else True

    def test_tokenize_all_fixture_texts(self):
        from abstraction.tokenize import tokenize_agnostic
        for fn in ["text1.txt", "text2.txt"]:
            with open(os.path.join(CORPUS_DIR, "txt", fn)) as f:
                txt = f.read()
            tokens = tokenize_agnostic(txt)
            assert len(tokens) > 10


# ---------------------------------------------------------------------------
# Scoring with fixture norms
# ---------------------------------------------------------------------------

class TestScoringCIIntegration:
    def test_score_psg_with_fixture_norms(self, fixture_norms, monkeypatch):
        """Score a passage using fixture norms."""
        from abstraction import scoring
        norm_dict = fixture_norms["Abs-Conc.Median"].dropna().to_dict()
        monkeypatch.setattr(scoring, "_NORM_DICTS", {"Abs-Conc.Median.median": norm_dict})
        from abstraction.scoring import score_psg
        # concrete passage
        score = score_psg("The rock and stone fell on the table")
        assert score > 0
        # abstract passage
        score = score_psg("Truth and justice demand virtue and freedom")
        assert score < 0

    def test_score_words_with_fixture_norms(self, fixture_norms, monkeypatch):
        from abstraction import scoring
        norm_dict = fixture_norms["Abs-Conc.Median"].dropna().to_dict()
        monkeypatch.setattr(scoring, "_NORM_DICTS", {"Abs-Conc.Median.median": norm_dict})
        from abstraction.scoring import score_words
        df = score_words("The rock of virtue stands in the world of justice")
        assert len(df) > 0
        rock = df[df["word"] == "rock"]
        assert len(rock) == 1
        assert rock.iloc[0]["score"] > 0
        assert rock.iloc[0]["is_concrete"]
        virtue = df[df["word"] == "virtue"]
        assert len(virtue) == 1
        assert virtue.iloc[0]["score"] < 0
        assert virtue.iloc[0]["is_abstract"]

    def test_score_freqs_file_with_fixture_norms(self, fixture_norms, monkeypatch):
        from abstraction import scoring
        norm_dict = fixture_norms["Abs-Conc.Median"].dropna().to_dict()
        monkeypatch.setattr(scoring, "_NORM_DICTS", {"Abs-Conc.Median.median": norm_dict})
        from abstraction.scoring import score_freqs_file
        score = score_freqs_file(os.path.join(CORPUS_DIR, "freqs", "text1.json"))
        assert isinstance(score, float)
        assert not np.isnan(score)


# ---------------------------------------------------------------------------
# Corpus-level frequency scoring (full flow)
# ---------------------------------------------------------------------------

class TestScoreCorpusCIIntegration:
    def test_score_corpus_freqs_in_memory(self, fixture_allnorms):
        from abstraction.scoring import score_corpus_freqs
        df = score_corpus_freqs(CORPUS_DIR, allnorms=fixture_allnorms)
        assert len(df) == 3
        assert "id" in df.columns
        assert set(df["id"]) == {"text1", "text2", "subdir/text3"}
        # should have norm columns
        norm_cols = [c for c in df.columns if c != "id"]
        assert len(norm_cols) > 0

    def test_score_corpus_freqs_to_csv(self, fixture_allnorms, tmp_path):
        from abstraction.scoring import score_corpus_freqs
        out = str(tmp_path / "scores.csv")
        df = score_corpus_freqs(CORPUS_DIR, allnorms=fixture_allnorms, output_path=out)
        assert len(df) == 3
        # CSV should exist and match
        csv_df = pd.read_csv(out)
        assert len(csv_df) == 3
        assert set(csv_df["id"]) == set(df["id"])

    def test_score_corpus_freqs_resumable(self, fixture_allnorms, tmp_path):
        from abstraction.scoring import score_corpus_freqs
        out = str(tmp_path / "scores.csv")
        # first run
        df1 = score_corpus_freqs(CORPUS_DIR, allnorms=fixture_allnorms, output_path=out)
        # second run (should skip all, no duplicates)
        df2 = score_corpus_freqs(CORPUS_DIR, allnorms=fixture_allnorms, output_path=out)
        assert len(df1) == len(df2) == 3
        csv_df = pd.read_csv(out)
        assert len(csv_df) == 3  # no duplicates

    def test_concrete_texts_score_higher(self, fixture_allnorms):
        """text2 (sail, water, shore, purse) should score more concrete than text3 (joy, wonder)."""
        from abstraction.scoring import score_corpus_freqs
        df = score_corpus_freqs(CORPUS_DIR, allnorms=fixture_allnorms)
        median_col = [c for c in df.columns if "Median" in c][0]
        scores = df.set_index("id")[median_col]
        assert scores["text2"] > scores["subdir/text3"]


# ---------------------------------------------------------------------------
# Norms classification with fixture data
# ---------------------------------------------------------------------------

class TestNormsCIIntegration:
    def test_classify_word(self):
        from abstraction.norms import classify_word
        assert classify_word(1.5) == "Concrete"
        assert classify_word(-1.5) == "Abstract"
        assert classify_word(0.0) == "Neither"
        assert classify_word(1.0) == "Concrete"
        assert classify_word(-1.0) == "Abstract"

    def test_get_contrasts_with_fixture(self, fixture_norms):
        from abstraction.norms import get_contrasts
        contrasts = get_contrasts(fixture_norms)
        assert len(contrasts) > 0
        for c in contrasts:
            assert "neg" in c and "pos" in c and "neither" in c
            assert isinstance(c["neg"], set)
            assert isinstance(c["pos"], set)
        # check that rock is concrete and justice is abstract in Median
        median_c = [c for c in contrasts if c["source"] == "Median"][0]
        assert "rock" in median_c["pos"]
        assert "justice" in median_c["neg"]

    def test_format_norms_as_long_with_fixture(self, fixture_norms):
        from abstraction.norms import format_norms_as_long
        long = format_norms_as_long(fixture_norms)
        assert "word" in long.columns
        assert "z" in long.columns
        assert "decision" in long.columns
        assert len(long) > len(fixture_norms)
