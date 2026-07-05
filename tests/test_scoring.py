import json
import os
import sqlite3
import warnings

import numpy as np
import pandas as pd
import pytest

import abstraction.norms as norms_mod
import abstraction.scoring as scoring
from abstraction.norms import norms_version
from abstraction.scoring import (
    score_freqs,
    score_words,
    score_psg,
    _modernize_score,
    _modernize_word_list,
    _walk_freqs,
    _score_freqs_allnorms,
    _score_freqs_dict_allnorms,
    _get_csv_columns,
    _get_norms_arrays,
    _load_done_ids,
    _load_freqs_cache,
    _save_freqs_cache,
    _check_resume_header,
    get_norm_dict,
    score_corpus_freqs,
)
from tests.conftest import make_fake_allnorms as _make_fake_allnorms


class TestScoreFreqs:
    def _patch_norms(self, install_fake_norms):
        """Install fake norms via the shared conftest contract."""
        return install_fake_norms(
            {"rock": 1.5, "virtue": -1.8, "justice": -1.3, "face": 0.2}
        )

    def test_basic(self, install_fake_norms):
        self._patch_norms(install_fake_norms)
        score = score_freqs({"rock": 2, "virtue": 2})
        expected = (1.5 * 2 + -1.8 * 2) / 4
        assert abs(score - expected) < 1e-6

    def test_unknown_words_ignored(self, install_fake_norms):
        self._patch_norms(install_fake_norms)
        score = score_freqs({"rock": 1, "xyzzy": 100})
        assert abs(score - 1.5) < 1e-6

    def test_empty_freqs(self, install_fake_norms):
        self._patch_norms(install_fake_norms)
        assert np.isnan(score_freqs({}))

    def test_all_unknown(self, install_fake_norms):
        self._patch_norms(install_fake_norms)
        assert np.isnan(score_freqs({"xyzzy": 5, "qqq": 3}))

    def test_case_insensitive(self, install_fake_norms):
        self._patch_norms(install_fake_norms)
        score = score_freqs({"Rock": 1, "VIRTUE": 1})
        expected = (1.5 + -1.8) / 2
        assert abs(score - expected) < 1e-6


class TestScoreWords:
    def _patch_norms(self, install_fake_norms):
        install_fake_norms(
            {"rock": 1.5, "virtue": -1.8, "justice": -1.3, "face": 0.2}
        )

    def test_returns_dataframe(self, install_fake_norms):
        self._patch_norms(install_fake_norms)
        df = score_words("the rock of virtue")
        assert isinstance(df, pd.DataFrame)
        assert "word" in df.columns
        assert "score" in df.columns
        assert "position" in df.columns
        assert "is_abstract" in df.columns
        assert "is_concrete" in df.columns

    def test_known_words_scored(self, install_fake_norms):
        self._patch_norms(install_fake_norms)
        df = score_words("rock and virtue")
        rock = df[df["word"] == "rock"].iloc[0]
        assert rock["score"] == 1.5
        assert rock["is_concrete"] == True
        assert rock["is_abstract"] == False
        virtue = df[df["word"] == "virtue"].iloc[0]
        assert virtue["score"] == -1.8
        assert virtue["is_abstract"] == True

    def test_unknown_words_nan(self, install_fake_norms):
        self._patch_norms(install_fake_norms)
        df = score_words("the rock")
        the_row = df[df["word"] == "the"].iloc[0]
        assert np.isnan(the_row["score"])

    def test_empty_text(self, install_fake_norms):
        self._patch_norms(install_fake_norms)
        df = score_words("")
        assert len(df) == 0

    def test_positions_sequential(self, install_fake_norms):
        self._patch_norms(install_fake_norms)
        df = score_words("rock face virtue justice")
        assert list(df["position"]) == sorted(df["position"].tolist())


# ---------------------------------------------------------------------------
# Helpers for corpus-level scoring tests
# (_make_fake_allnorms is shared via tests/conftest.py)
# ---------------------------------------------------------------------------

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

    # NOTE: a former test_force_flag lived here but never exercised --force:
    # score_corpus_freqs has no force parameter (force semantics are
    # implemented at the CLI layer, covered by
    # tests/test_cli.py::TestScoreCorpus::test_force_deletes_and_rescores),
    # and its body duplicated TestScoreCorpusFreqs::test_resumability_no_duplicates.

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

    def _patch(self, install_fake_norms):
        return install_fake_norms(
            {"virtue": -1.8, "rock": 1.5},
            spelling={"vertue": "virtue", "rocke": "rock"},
        )

    def test_score_psg_modernizes(self, install_fake_norms):
        self._patch(install_fake_norms)
        score = score_psg("vertue and rocke")
        expected = (-1.8 + 1.5) / 2
        assert abs(score - expected) < 1e-6

    def test_score_freqs_modernizes(self, install_fake_norms):
        self._patch(install_fake_norms)
        score = score_freqs({"vertue": 1, "rocke": 1})
        expected = (-1.8 + 1.5) / 2
        assert abs(score - expected) < 1e-6

    def test_score_words_modernizes(self, install_fake_norms):
        self._patch(install_fake_norms)
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

    def test_modern_form_preferred_over_raw(self, install_fake_norms):
        """When both raw and modern forms exist in norms, modern wins."""
        install_fake_norms(
            {"vertue": -0.72, "virtue": -1.59},
            spelling={"vertue": "virtue"},
        )
        score = score_psg("vertue")
        assert abs(score - (-1.59)) < 1e-6


# ---------------------------------------------------------------------------
# norms_version (allnorms fingerprint; audit §4.1)
# ---------------------------------------------------------------------------


class TestNormsVersion:
    def _patch_allnorms_file(self, tmp_path, monkeypatch, content=b"fake allnorms"):
        p = tmp_path / "data.allnorms.pkl.gz"
        p.write_bytes(content)
        monkeypatch.setattr(norms_mod, "PATH_ALLNORMS", str(p))
        return p

    def test_stable_same_file(self, tmp_path, monkeypatch):
        self._patch_allnorms_file(tmp_path, monkeypatch)
        v1 = norms_version()
        v2 = norms_version()
        assert v1 == v2
        assert len(v1) == 12
        assert all(c in "0123456789abcdef" for c in v1)

    def test_touch_mtime_changes_version(self, tmp_path, monkeypatch):
        p = self._patch_allnorms_file(tmp_path, monkeypatch)
        v1 = norms_version()
        st = os.stat(p)
        os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 10_000_000_000))
        v2 = norms_version()
        assert v1 != v2

    def test_size_change_changes_version(self, tmp_path, monkeypatch):
        p = self._patch_allnorms_file(tmp_path, monkeypatch)
        v1 = norms_version()
        p.write_bytes(b"regenerated allnorms with different size")
        v2 = norms_version()
        assert v1 != v2

    def test_missing_file_is_deterministic(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            norms_mod, "PATH_ALLNORMS", str(tmp_path / "nonexistent.pkl.gz")
        )
        assert norms_version() == norms_version()

    def test_language_variants_differ(self, tmp_path, monkeypatch):
        import abstraction.config as config_mod
        en = tmp_path / "en.pkl.gz"
        fr = tmp_path / "fr.pkl.gz"
        en.write_bytes(b"english")
        fr.write_bytes(b"french norms")
        monkeypatch.setattr(norms_mod, "PATH_ALLNORMS", str(en))
        monkeypatch.setattr(config_mod, "PATH_ALLNORMS_FR", str(fr))
        assert norms_version("en") != norms_version("fr")

    def test_unknown_lang_raises(self):
        with pytest.raises(ValueError):
            norms_version("xx")


# ---------------------------------------------------------------------------
# freqs_cache.db norms-version filtering (audit §4.1)
# ---------------------------------------------------------------------------


class TestFreqsCacheVersioning:
    def _patch_db(self, tmp_path, monkeypatch):
        db = str(tmp_path / "freqs_cache.db")
        monkeypatch.setattr(scoring, "FREQS_CACHE_PATH", db)
        return db

    def test_write_under_A_read_under_B_misses(self, tmp_path, monkeypatch):
        self._patch_db(tmp_path, monkeypatch)
        _save_freqs_cache(
            [("corpus/t1.json", {"Abs-Conc.Median.median": 0.5})],
            modernize=False, norms_ver="verA",
        )
        hit = _load_freqs_cache(modernize=False, norms_ver="verA")
        assert hit == {"corpus/t1.json": {"Abs-Conc.Median.median": 0.5}}
        miss = _load_freqs_cache(modernize=False, norms_ver="verB")
        assert miss == {}

    def test_new_version_write_dominates(self, tmp_path, monkeypatch):
        """Re-scoring under new norms replaces the row (PK dominance), so the
        unversioned corpus_correction read sees exactly one, fresh row."""
        db = self._patch_db(tmp_path, monkeypatch)
        _save_freqs_cache([("k1", {"a": 1.0})], modernize=False, norms_ver="verA")
        _save_freqs_cache([("k1", {"a": 2.0})], modernize=False, norms_ver="verB")
        assert _load_freqs_cache(modernize=False, norms_ver="verB") == {"k1": {"a": 2.0}}
        assert _load_freqs_cache(modernize=False, norms_ver="verA") == {}
        # corpus_correction.load_match_group_scores's exact query still works
        conn = sqlite3.connect(db)
        rows = conn.execute(
            "SELECT freqs_key, scores_json FROM freqs_scores WHERE modernized = ?",
            (0,),
        ).fetchall()
        conn.close()
        assert rows == [("k1", json.dumps({"a": 2.0}))]

    def test_modernize_flag_still_separates(self, tmp_path, monkeypatch):
        self._patch_db(tmp_path, monkeypatch)
        _save_freqs_cache([("k1", {"a": 1.0})], modernize=False, norms_ver="v")
        _save_freqs_cache([("k1", {"a": 9.0})], modernize=True, norms_ver="v")
        assert _load_freqs_cache(modernize=False, norms_ver="v") == {"k1": {"a": 1.0}}
        assert _load_freqs_cache(modernize=True, norms_ver="v") == {"k1": {"a": 9.0}}

    def test_missing_db_returns_empty(self, tmp_path, monkeypatch):
        self._patch_db(tmp_path, monkeypatch)
        assert _load_freqs_cache(modernize=False, norms_ver="v") == {}

    def test_legacy_rows_migrated_as_current_version(self, tmp_path, monkeypatch):
        """Pre-versioning rows are stamped with the version current at
        migration time (they were computed with it), so they keep hitting."""
        db = self._patch_db(tmp_path, monkeypatch)
        # Deterministic "current" version via a fake allnorms file
        p = tmp_path / "allnorms.pkl.gz"
        p.write_bytes(b"current norms artifact")
        monkeypatch.setattr(norms_mod, "PATH_ALLNORMS", str(p))
        conn = sqlite3.connect(db)
        conn.execute("""
            CREATE TABLE freqs_scores (
                freqs_key TEXT NOT NULL,
                modernized INTEGER NOT NULL,
                scores_json TEXT NOT NULL,
                PRIMARY KEY (freqs_key, modernized)
            )
        """)
        conn.execute(
            "INSERT INTO freqs_scores VALUES (?, ?, ?)",
            ("legacy_key", 0, json.dumps({"a": 1.0})),
        )
        conn.commit()
        conn.close()
        # Default load (current version) migrates and serves the legacy row
        assert _load_freqs_cache(modernize=False) == {"legacy_key": {"a": 1.0}}
        # A different (e.g. regenerated) version misses it
        assert _load_freqs_cache(modernize=False, norms_ver="other") == {}
        # Unversioned read contract (corpus_correction) intact post-migration
        conn = sqlite3.connect(db)
        rows = conn.execute(
            "SELECT freqs_key, scores_json FROM freqs_scores WHERE modernized = ?",
            (0,),
        ).fetchall()
        conn.close()
        assert rows == [("legacy_key", json.dumps({"a": 1.0}))]


# ---------------------------------------------------------------------------
# _NORMS_ARRAYS_CACHE keyed by frame identity (audit §4.2)
# ---------------------------------------------------------------------------


class TestNormsArraysCacheKeying:
    def test_two_frames_use_distinct_arrays(self):
        f1 = pd.DataFrame({"Abs-Conc.Median.median": {"rock": 1.0}})
        f2 = pd.DataFrame({"Abs-Conc.Median.median": {"rock": -1.0}})
        _w1, v1, _c1 = _get_norms_arrays(f1)
        _w2, v2, _c2 = _get_norms_arrays(f2)
        assert v1[0][0] == 1.0
        assert v2[0][0] == -1.0

    def test_same_frame_returns_cached_arrays(self):
        f = pd.DataFrame({"Abs-Conc.Median.median": {"rock": 1.0}})
        a = _get_norms_arrays(f)
        b = _get_norms_arrays(f)
        assert a is b
        assert id(f) in scoring._NORMS_ARRAYS_CACHE

    def test_cross_language_contamination_regression(self):
        """Score English then French: French must be scored against the
        FRENCH frame's arrays. Under the old keyless cache, the second call
        silently reused the English arrays and 'pierre' scored as unknown."""
        en = pd.DataFrame({"Abs-Conc.Median.median": {"stone": 2.0}})
        fr = pd.DataFrame({"Abs-Conc.Median.median": {"pierre": -2.0}})
        s_en = _score_freqs_dict_allnorms({"stone": 1}, en)
        s_fr = _score_freqs_dict_allnorms({"pierre": 1}, fr)
        assert s_en["Abs-Conc.Median.median"] == 2.0
        assert s_fr["Abs-Conc.Median.median"] == -2.0


# ---------------------------------------------------------------------------
# CSV resume header check (audit §4.4)
# ---------------------------------------------------------------------------


class TestResumeHeaderCheck:
    def _setup_corpus(self, tmp_path):
        freqs = tmp_path / "my_corpus" / "freqs"
        freqs.mkdir(parents=True)
        _write_json(str(freqs / "text1.json"), {"rock": 2, "virtue": 3})
        _write_json(str(freqs / "text2.json"), {"face": 4})
        return str(tmp_path / "my_corpus")

    def test_resume_with_changed_norms_raises(self, tmp_path):
        corpus_dir = self._setup_corpus(tmp_path)
        out = str(tmp_path / "scores.csv")
        allnorms_v1 = _make_fake_allnorms()
        score_corpus_freqs(corpus_dir, allnorms=allnorms_v1, output_path=out)
        # Norms regenerated with an extra column → resume must refuse
        allnorms_v2 = allnorms_v1.copy()
        allnorms_v2["Abs-Conc.NEW.median"] = 0.0
        with pytest.raises(ValueError, match="--force"):
            score_corpus_freqs(corpus_dir, allnorms=allnorms_v2, output_path=out)
        # File untouched: still exactly the v1 rows
        saved = pd.read_csv(out)
        assert len(saved) == 2
        assert list(saved.columns) == _get_csv_columns(allnorms_v1)

    def test_resume_with_same_norms_ok(self, tmp_path):
        corpus_dir = self._setup_corpus(tmp_path)
        out = str(tmp_path / "scores.csv")
        allnorms = _make_fake_allnorms()
        score_corpus_freqs(corpus_dir, allnorms=allnorms, output_path=out)
        # Same column set resumes cleanly, no duplicates
        score_corpus_freqs(corpus_dir, allnorms=allnorms, output_path=out)
        assert len(pd.read_csv(out)) == 2

    def test_check_helper_direct(self, tmp_path):
        csv_path = tmp_path / "scores.csv"
        csv_path.write_text("id,colA,colB\nx,1,2\n")
        _check_resume_header(str(csv_path), ["id", "colA", "colB"])  # no raise
        with pytest.raises(ValueError, match="does not match"):
            _check_resume_header(str(csv_path), ["id", "colA", "colB", "colC"])
        with pytest.raises(ValueError, match="does not match"):
            # same length, different names/order also refuses
            _check_resume_header(str(csv_path), ["id", "colB", "colA"])

    def test_check_helper_empty_file_ok(self, tmp_path):
        csv_path = tmp_path / "empty.csv"
        csv_path.write_text("")
        _check_resume_header(str(csv_path), ["id", "colA"])  # no raise


# ---------------------------------------------------------------------------
# get_norm_dict non-EN → EN fallback: cached + warned (audit §4.3)
# ---------------------------------------------------------------------------


class TestGetNormDictFallback:
    def _patch_lang_loader(self, monkeypatch):
        """Fake per-language allnorms; counts loads per language."""
        calls = []
        en = pd.DataFrame({"Abs-Conc.Median.C18": {"rock": 1.5}})
        es = pd.DataFrame({"Abs-Conc.Median.median": {"roca": 2.0}})
        def fake_get(lang):
            calls.append(lang)
            return en if lang == "en" else es
        monkeypatch.setattr(scoring, "_get_allnorms_for_lang", fake_get)
        return calls

    def test_fallback_warns_and_caches_under_requested_key(self, monkeypatch):
        self._patch_lang_loader(monkeypatch)
        with pytest.warns(UserWarning, match="falling back to ENGLISH"):
            d = get_norm_dict("Abs-Conc.Median.C18", lang="es")
        assert d == {"rock": 1.5}
        # Cached under the REQUESTED key (and the en key it resolved through)
        assert ("Abs-Conc.Median.C18", "es") in scoring._NORM_DICTS
        assert ("Abs-Conc.Median.C18", "en") in scoring._NORM_DICTS

    def test_second_call_hits_cache_no_reread_no_warning(self, monkeypatch):
        calls = self._patch_lang_loader(monkeypatch)
        with pytest.warns(UserWarning):
            d1 = get_norm_dict("Abs-Conc.Median.C18", lang="es")
        assert calls == ["es", "en"]
        calls.clear()
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning would fail the test
            d2 = get_norm_dict("Abs-Conc.Median.C18", lang="es")
        assert d2 is d1
        assert calls == []  # no allnorms re-read of any language

    def test_column_present_in_lang_no_fallback(self, monkeypatch):
        calls = self._patch_lang_loader(monkeypatch)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            d = get_norm_dict("Abs-Conc.Median.median", lang="es")
        assert d == {"roca": 2.0}
        assert calls == ["es"]

    def test_missing_col_english_still_raises(self, monkeypatch):
        self._patch_lang_loader(monkeypatch)
        with pytest.raises(KeyError):
            get_norm_dict("No.Such.Column", lang="en")
