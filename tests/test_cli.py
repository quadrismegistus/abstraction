import os
import subprocess
import sys

import pandas as pd
import pytest

from abstraction.cli import main


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FIXTURE_CORPUS = os.path.join(
    os.path.dirname(__file__), "fixtures", "test_corpus"
)


def _make_fake_allnorms():
    """Return a small allnorms DataFrame indexed by word."""
    return pd.DataFrame(
        {
            "Conc.Brys": {"rock": 4.5, "virtue": 1.2, "face": 3.8, "truth": 3.0, "man": 4.0},
            "Imag.MRC": {"rock": 5.0, "virtue": 2.1, "face": 4.0, "truth": 2.5, "man": 4.5},
        }
    )


def _patch_scoring(monkeypatch, tmp_path):
    """Patch config paths and get_allnorms so CLI commands work without real data."""
    corpora_dir = str(tmp_path / "corpora")
    scores_dir = str(tmp_path / "scores")
    os.makedirs(corpora_dir, exist_ok=True)
    os.makedirs(scores_dir, exist_ok=True)

    monkeypatch.setattr("abstraction.config.PATH_CORPORA", corpora_dir)
    monkeypatch.setattr("abstraction.config.SCORES_DIR", scores_dir)
    # Patch at both import sites so lazy imports inside cli functions pick them up
    monkeypatch.setattr("abstraction.scoring.PATH_CORPORA", corpora_dir)
    monkeypatch.setattr("abstraction.scoring.SCORES_DIR", scores_dir)
    monkeypatch.setattr("abstraction.scoring.get_allnorms", _make_fake_allnorms)

    return corpora_dir, scores_dir


def _copy_fixture_corpus(dest_dir, name="test_corpus"):
    """Copy the fixture corpus into dest_dir/<name>."""
    import shutil
    target = os.path.join(dest_dir, name)
    shutil.copytree(FIXTURE_CORPUS, target)
    return target


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCLIHelp:
    def test_help_exits_zero(self):
        result = subprocess.run(
            [sys.executable, "-m", "abstraction.cli", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "usage" in result.stdout.lower()

    def test_no_subcommand_exits_one(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["abstraction"])
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1


class TestScoreCorpus:
    def test_scores_fixture_corpus(self, tmp_path, monkeypatch):
        corpora_dir, scores_dir = _patch_scoring(monkeypatch, tmp_path)
        _copy_fixture_corpus(corpora_dir)

        monkeypatch.setattr(
            "sys.argv", ["abstraction", "score-corpus", "test_corpus"]
        )
        main()

        csv_path = os.path.join(scores_dir, "v8-raw", "test_corpus.csv")
        assert os.path.exists(csv_path)
        df = pd.read_csv(csv_path)
        assert len(df) >= 1
        assert "id" in df.columns

    def test_nonexistent_corpus_exits_error(self, tmp_path, monkeypatch):
        corpora_dir, scores_dir = _patch_scoring(monkeypatch, tmp_path)

        monkeypatch.setattr(
            "sys.argv", ["abstraction", "score-corpus", "nonexistent"]
        )
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1

    def test_force_deletes_and_rescores(self, tmp_path, monkeypatch):
        corpora_dir, scores_dir = _patch_scoring(monkeypatch, tmp_path)
        _copy_fixture_corpus(corpora_dir)

        # First run (no --force)
        monkeypatch.setattr(
            "sys.argv", ["abstraction", "score-corpus", "test_corpus"]
        )
        main()

        csv_path = os.path.join(scores_dir, "v8-raw", "test_corpus.csv")
        assert os.path.exists(csv_path)
        mtime1 = os.path.getmtime(csv_path)
        row_count1 = len(pd.read_csv(csv_path))

        # Second run with --force
        import time
        time.sleep(0.05)
        monkeypatch.setattr(
            "sys.argv", ["abstraction", "score-corpus", "--force", "test_corpus"]
        )
        main()

        assert os.path.exists(csv_path)
        mtime2 = os.path.getmtime(csv_path)
        row_count2 = len(pd.read_csv(csv_path))
        # File was recreated
        assert mtime2 > mtime1
        # No duplicate rows
        assert row_count2 == row_count1


class TestScoreCorpora:
    def test_scores_all_corpora(self, tmp_path, monkeypatch):
        corpora_dir, scores_dir = _patch_scoring(monkeypatch, tmp_path)
        _copy_fixture_corpus(corpora_dir, name="corpus_a")
        _copy_fixture_corpus(corpora_dir, name="corpus_b")
        # corpus_c has no freqs — should be skipped
        os.makedirs(os.path.join(corpora_dir, "corpus_c"))

        # score_all_corpora uses default args bound at definition time,
        # so we replace the function the CLI will import
        from abstraction.scoring import score_all_corpora as real_fn
        calls = []

        def mock_score_all(force=False, modernize=False):
            calls.append(force)
            return real_fn(corpora_dir=corpora_dir, output_dir=scores_dir, force=force, modernize=modernize)

        monkeypatch.setattr("abstraction.scoring.score_all_corpora", mock_score_all)

        monkeypatch.setattr("sys.argv", ["abstraction", "score-corpora"])
        main()

        assert len(calls) == 1
        assert os.path.exists(os.path.join(scores_dir, "v8-raw", "corpus_a.csv"))
        assert os.path.exists(os.path.join(scores_dir, "v8-raw", "corpus_b.csv"))
        assert not os.path.exists(os.path.join(scores_dir, "v8-raw", "corpus_c.csv"))
