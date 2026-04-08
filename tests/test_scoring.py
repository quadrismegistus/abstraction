import json
import os

import numpy as np
import pandas as pd
import pytest

from abstraction.scoring import (
    score_freqs,
    score_words,
    score_psg,
    _modernize_score,
    _modernize_word_list,
    _walk_freqs,
    _score_freqs_allnorms,
    _get_csv_columns,
    _load_done_ids,
    score_corpus_freqs,
)
import abstraction.scoring as _scoring_mod


@pytest.fixture(autouse=True)
def _clear_norms_cache():
    """Clear the global norms arrays cache between tests."""
    _scoring_mod._NORMS_ARRAYS_CACHE = None
    yield
    _scoring_mod._NORMS_ARRAYS_CACHE = None


class TestScoreFreqs:
    def _patch_norms(self, monkeypatch):
        """Patch get_norm_dict and spelling modernizer."""
        fake = {"rock": 1.5, "virtue": -1.8, "justice": -1.3, "face": 0.2}
        monkeypatch.setattr(
            "abstraction.scoring._NORM_DICTS",
            {"Abs-Conc.Median.median": fake},
        )
        monkeypatch.setattr("abstraction.scoring.get_spelling_modernizer", lambda: {})
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
        monkeypatch.setattr("abstraction.scoring.get_spelling_modernizer", lambda: {})

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


# ---------------------------------------------------------------------------
# Helpers for corpus-level scoring tests
# ---------------------------------------------------------------------------

def _make_fake_allnorms():
    """Return a small allnorms DataFrame indexed by word.

    Uses Abs-Conc.Median.* column naming to match production code,
    which generates _pct_ columns from this pattern.
    """
    return pd.DataFrame(
        {
            "Abs-Conc.Median.median": {"rock": 1.5, "virtue": -1.8, "face": 0.3},
            "Abs-Conc.Median.orig": {"rock": 1.4, "virtue": -1.7, "face": 0.2},
        }
    )


def _write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


# ---------------------------------------------------------------------------
# _walk_freqs
# ---------------------------------------------------------------------------


class TestWalkFreqs:
    def test_flat_files(self, tmp_path):
        freqs = tmp_path / "freqs"
        freqs.mkdir()
        (freqs / "text001.json").write_text("{}")
        (freqs / "text002.json").write_text("{}")
        result = dict(_walk_freqs(str(freqs)))
        assert set(result.keys()) == {"text001", "text002"}
        for v in result.values():
            assert v.endswith(".json")

    def test_nested_dirs_slash_ids(self, tmp_path):
        freqs = tmp_path / "freqs"
        sub = freqs / "subdir"
        sub.mkdir(parents=True)
        (sub / "abc.json").write_text("{}")
        result = dict(_walk_freqs(str(freqs)))
        assert "subdir/abc" in result

    def test_non_json_ignored(self, tmp_path):
        freqs = tmp_path / "freqs"
        freqs.mkdir()
        (freqs / "notes.txt").write_text("hello")
        (freqs / "data.csv").write_text("a,b")
        (freqs / "real.json").write_text("{}")
        result = dict(_walk_freqs(str(freqs)))
        assert list(result.keys()) == ["real"]

    def test_empty_dir(self, tmp_path):
        freqs = tmp_path / "freqs"
        freqs.mkdir()
        assert list(_walk_freqs(str(freqs))) == []


# ---------------------------------------------------------------------------
# _score_freqs_allnorms
# ---------------------------------------------------------------------------


class TestScoreFreqsAllnorms:
    def test_normal_case(self, tmp_path):
        path = tmp_path / "t.json"
        _write_json(str(path), {"rock": 2, "virtue": 3})
        allnorms = _make_fake_allnorms()
        scores = _score_freqs_allnorms(str(path), allnorms)
        expected_median = (1.5 * 2 + (-1.8) * 3) / 5
        expected_orig = (1.4 * 2 + (-1.7) * 3) / 5
        assert abs(scores["Abs-Conc.Median.median"] - expected_median) < 1e-6
        assert abs(scores["Abs-Conc.Median.orig"] - expected_orig) < 1e-6

    def test_empty_freqs(self, tmp_path):
        path = tmp_path / "empty.json"
        _write_json(str(path), {})
        allnorms = _make_fake_allnorms()
        assert _score_freqs_allnorms(str(path), allnorms) == {}

    def test_invalid_json(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("not json at all")
        allnorms = _make_fake_allnorms()
        assert _score_freqs_allnorms(str(path), allnorms) == {}

    def test_words_not_in_norms(self, tmp_path):
        path = tmp_path / "unknown.json"
        _write_json(str(path), {"xyzzy": 10, "qqq": 5})
        allnorms = _make_fake_allnorms()
        assert _score_freqs_allnorms(str(path), allnorms) == {}

    def test_mixed_known_unknown(self, tmp_path):
        path = tmp_path / "mix.json"
        _write_json(str(path), {"rock": 1, "xyzzy": 100})
        allnorms = _make_fake_allnorms()
        scores = _score_freqs_allnorms(str(path), allnorms)
        # only rock matched, count=1
        assert abs(scores["Abs-Conc.Median.median"] - 1.5) < 1e-6

    def test_case_insensitive(self, tmp_path):
        path = tmp_path / "upper.json"
        _write_json(str(path), {"ROCK": 1, "Virtue": 1})
        allnorms = _make_fake_allnorms()
        scores = _score_freqs_allnorms(str(path), allnorms)
        expected = (1.5 + (-1.8)) / 2
        assert abs(scores["Abs-Conc.Median.median"] - expected) < 1e-6


# ---------------------------------------------------------------------------
# _get_csv_columns
# ---------------------------------------------------------------------------


class TestGetCsvColumns:
    def test_id_first_then_pct_then_norms(self):
        allnorms = _make_fake_allnorms()
        cols = _get_csv_columns(allnorms)
        assert cols[0] == "id"
        # _pct_* columns come after id, before norm columns
        pct_cols = [c for c in cols if c.startswith("_pct_")]
        norm_cols = sorted(allnorms.columns.tolist())
        assert cols == ["id"] + pct_cols + norm_cols

    def test_single_column(self):
        allnorms = pd.DataFrame({"Z.Score": {"a": 1.0}})
        cols = _get_csv_columns(allnorms)
        assert cols[0] == "id"
        assert cols[-1] == "Z.Score"


# ---------------------------------------------------------------------------
# _load_done_ids
# ---------------------------------------------------------------------------


class TestLoadDoneIds:
    def test_existing_file(self, tmp_path):
        csv_path = tmp_path / "done.csv"
        csv_path.write_text("id,score\nabc,1.0\ndef,2.0\n")
        ids = _load_done_ids(str(csv_path))
        assert ids == {"abc", "def"}

    def test_missing_file(self, tmp_path):
        csv_path = tmp_path / "nonexistent.csv"
        assert _load_done_ids(str(csv_path)) == set()

    def test_empty_file(self, tmp_path):
        csv_path = tmp_path / "empty.csv"
        csv_path.write_text("")
        assert _load_done_ids(str(csv_path)) == set()

    def test_header_only(self, tmp_path):
        csv_path = tmp_path / "header.csv"
        csv_path.write_text("id,score\n")
        ids = _load_done_ids(str(csv_path))
        assert ids == set()


# ---------------------------------------------------------------------------
# score_corpus_freqs
# ---------------------------------------------------------------------------


class TestScoreCorpusFreqs:
    def _setup_corpus(self, tmp_path):
        """Create a fake corpus dir with freqs/ containing two JSON files."""
        corpus = tmp_path / "my_corpus"
        freqs = corpus / "freqs"
        freqs.mkdir(parents=True)
        _write_json(str(freqs / "text1.json"), {"rock": 2, "virtue": 3})
        _write_json(str(freqs / "text2.json"), {"face": 4})
        return str(corpus)

    def test_in_memory(self, tmp_path):
        corpus_dir = self._setup_corpus(tmp_path)
        allnorms = _make_fake_allnorms()
        df = score_corpus_freqs(corpus_dir, allnorms=allnorms)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "id" in df.columns
        assert set(df["id"]) == {"text1", "text2"}

    def test_with_output_path(self, tmp_path):
        corpus_dir = self._setup_corpus(tmp_path)
        allnorms = _make_fake_allnorms()
        out = str(tmp_path / "scores.csv")
        df = score_corpus_freqs(corpus_dir, allnorms=allnorms, output_path=out)
        assert os.path.exists(out)
        saved = pd.read_csv(out)
        assert len(saved) == 2
        assert set(saved["id"]) == {"text1", "text2"}

    def test_resumability_no_duplicates(self, tmp_path):
        corpus_dir = self._setup_corpus(tmp_path)
        allnorms = _make_fake_allnorms()
        out = str(tmp_path / "scores.csv")
        # first run
        score_corpus_freqs(corpus_dir, allnorms=allnorms, output_path=out)
        # second run (should skip existing IDs)
        score_corpus_freqs(corpus_dir, allnorms=allnorms, output_path=out)
        saved = pd.read_csv(out)
        assert len(saved) == 2  # no duplicates

    def test_no_freqs_dir(self, tmp_path):
        corpus_dir = str(tmp_path / "empty_corpus")
        os.makedirs(corpus_dir)
        allnorms = _make_fake_allnorms()
        df = score_corpus_freqs(corpus_dir, allnorms=allnorms)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_columns_match_allnorms(self, tmp_path):
        corpus_dir = self._setup_corpus(tmp_path)
        allnorms = _make_fake_allnorms()
        df = score_corpus_freqs(corpus_dir, allnorms=allnorms)
        expected_cols = _get_csv_columns(allnorms)
        assert list(df.columns) == expected_cols


# ---------------------------------------------------------------------------
# score_all_corpora
# ---------------------------------------------------------------------------


class TestScoreAllCorpora:
    """Test corpus-level scoring via score_corpus_freqs.

    Note: score_all_corpora() now uses LLTK and cannot be tested without
    the full LLTK setup. These tests exercise the underlying scoring
    functions directly.
    """

    def _setup_corpus(self, tmp_path, name="corpus_a"):
        freqs = tmp_path / name / "freqs"
        freqs.mkdir(parents=True)
        _write_json(str(freqs / "t1.json"), {"rock": 1})
        return str(tmp_path / name)

    def test_basic(self, tmp_path):
        corpus_dir = self._setup_corpus(tmp_path)
        allnorms = _make_fake_allnorms()
        out_dir = tmp_path / "scores" / "v8-raw"
        out_dir.mkdir(parents=True)
        out_path = str(out_dir / "corpus_a.csv")
        df = score_corpus_freqs(corpus_dir, allnorms=allnorms, output_path=out_path)
        assert len(df) == 1
        assert os.path.exists(out_path)

    def test_force_flag(self, tmp_path):
        corpus_dir = self._setup_corpus(tmp_path)
        allnorms = _make_fake_allnorms()
        out_path = str(tmp_path / "scores.csv")
        # first run
        score_corpus_freqs(corpus_dir, allnorms=allnorms, output_path=out_path)
        mtime1 = os.path.getmtime(out_path)
        # second run (resumable — should skip existing)
        import time; time.sleep(0.05)
        score_corpus_freqs(corpus_dir, allnorms=allnorms, output_path=out_path)
        # file unchanged since all IDs already scored
        df = pd.read_csv(out_path)
        assert len(df) == 1  # no duplicates

    def test_no_freqs_dir_empty(self, tmp_path):
        corpus_dir = str(tmp_path / "empty_corpus")
        os.makedirs(corpus_dir)
        allnorms = _make_fake_allnorms()
        df = score_corpus_freqs(corpus_dir, allnorms=allnorms)
        assert len(df) == 0


# ---------------------------------------------------------------------------
# Spelling modernization
# ---------------------------------------------------------------------------


class TestModernizeScore:
    def test_word_not_in_spelling_dict(self):
        """Words not in the spelling dict use their own score."""
        norm_dict = {"virtue": -1.8, "rock": 1.5}
        spelling_d = {"vertue": "virtue"}
        score, word = _modernize_score("virtue", norm_dict, spelling_d)
        assert score == -1.8
        assert word == "virtue"

    def test_modern_form_preferred(self):
        """Historical spelling maps to modern form's score."""
        norm_dict = {"virtue": -1.8, "rock": 1.5}
        spelling_d = {"vertue": "virtue"}
        score, word = _modernize_score("vertue", norm_dict, spelling_d)
        assert score == -1.8
        assert word == "virtue"

    def test_modern_preferred_over_raw(self):
        """When both old and modern are in norms, modern form wins."""
        norm_dict = {"vertue": -0.72, "virtue": -1.59}
        spelling_d = {"vertue": "virtue"}
        score, word = _modernize_score("vertue", norm_dict, spelling_d)
        assert score == -1.59
        assert word == "virtue"

    def test_no_match(self):
        norm_dict = {"virtue": -1.8}
        spelling_d = {"vertue": "virtue"}
        score, word = _modernize_score("xyzzy", norm_dict, spelling_d)
        assert score is None
        assert word is None

    def test_modernized_not_in_norms_falls_back(self):
        """If modern form not in norms, fall back to raw word."""
        norm_dict = {"vertue": -0.72}
        spelling_d = {"vertue": "vertew"}  # maps to something not in norms
        score, word = _modernize_score("vertue", norm_dict, spelling_d)
        assert score == -0.72
        assert word == "vertue"


class TestModernizeWordList:
    def test_mixed(self):
        norm_index = {"virtue", "rock", "face"}
        spelling_d = {"vertue": "virtue", "rocke": "rock"}
        words = ["vertue", "rock", "xyzzy"]
        result = _modernize_word_list(words, norm_index, spelling_d)
        assert result == ["virtue", "rock", "xyzzy"]

    def test_no_modernization_needed(self):
        norm_index = {"virtue", "rock"}
        spelling_d = {"vertue": "virtue"}
        words = ["virtue", "rock"]
        result = _modernize_word_list(words, norm_index, spelling_d)
        assert result == ["virtue", "rock"]


class TestModernizeIntegration:
    """Test that scoring functions use modernization end-to-end."""

    def _patch(self, monkeypatch):
        fake_norms = {"virtue": -1.8, "rock": 1.5}
        fake_spelling = {"vertue": "virtue", "rocke": "rock"}
        monkeypatch.setattr(
            "abstraction.scoring._NORM_DICTS",
            {"Abs-Conc.Median.median": fake_norms},
        )
        monkeypatch.setattr(
            "abstraction.scoring.get_spelling_modernizer",
            lambda: fake_spelling,
        )
        return fake_norms

    def test_score_psg_modernizes(self, monkeypatch):
        self._patch(monkeypatch)
        score = score_psg("vertue and rocke")
        expected = (-1.8 + 1.5) / 2
        assert abs(score - expected) < 1e-6

    def test_score_freqs_modernizes(self, monkeypatch):
        self._patch(monkeypatch)
        score = score_freqs({"vertue": 1, "rocke": 1})
        expected = (-1.8 + 1.5) / 2
        assert abs(score - expected) < 1e-6

    def test_score_words_modernizes(self, monkeypatch):
        self._patch(monkeypatch)
        df = score_words("vertue and rocke")
        vertue_row = df[df["word"] == "vertue"].iloc[0]
        assert vertue_row["score"] == -1.8
        assert vertue_row["is_abstract"] == True

    def test_score_freqs_allnorms_modernizes(self, tmp_path, monkeypatch):
        allnorms = pd.DataFrame(
            {"Abs-Conc.Median.median": {"virtue": -1.8, "rock": 1.5}}
        )
        spelling_d = {"vertue": "virtue"}
        path = tmp_path / "t.json"
        _write_json(str(path), {"vertue": 3})
        scores = _score_freqs_allnorms(str(path), allnorms, spelling_d)
        assert abs(scores["Abs-Conc.Median.median"] - (-1.8)) < 1e-6

    def test_modern_form_preferred_over_raw(self, monkeypatch):
        """When both raw and modern forms exist in norms, modern wins."""
        fake_norms = {"vertue": -0.72, "virtue": -1.59}
        fake_spelling = {"vertue": "virtue"}
        monkeypatch.setattr(
            "abstraction.scoring._NORM_DICTS",
            {"Abs-Conc.Median.median": fake_norms},
        )
        monkeypatch.setattr(
            "abstraction.scoring.get_spelling_modernizer",
            lambda: fake_spelling,
        )
        score = score_psg("vertue")
        assert abs(score - (-1.59)) < 1e-6
