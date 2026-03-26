import os

import pandas as pd
import pytest

from abstraction.counting import (
    count_absconc,
    count_absconc_path,
    count_absconc_psg,
    _count_window,
    get_norms_for_counting,
)


# ---------------------------------------------------------------------------
# Fake norm contrasts
# ---------------------------------------------------------------------------

FAKE_CONTRASTS = [
    {
        "contrast": "Abs-Conc",
        "source": "Median",
        "period": "median",
        "neg": {"virtue", "justice", "freedom", "truth", "honor"},
        "pos": {"rock", "table", "river", "house", "tree"},
        "neither": {"face", "run", "large", "small", "world"},
    },
    {
        "contrast": "Abs-Conc",
        "source": "Other",
        "period": "other",
        "neg": {"virtue", "justice"},
        "pos": {"rock", "table"},
        "neither": {"face"},
    },
]


@pytest.fixture(autouse=True)
def _patch_counting_norms(monkeypatch):
    """Patch the norms cache so counting never touches real data files."""
    # Patch the module-level cache and SOURCES_FOR_COUNTING
    monkeypatch.setattr("abstraction.counting._NORM_CONTRASTS", FAKE_CONTRASTS)
    monkeypatch.setattr(
        "abstraction.counting.SOURCES_FOR_COUNTING", {"Median", "Other"}
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# A text with known words from the fake contrasts.
# Abstract (neg): virtue, justice, freedom, truth, honor
# Concrete (pos): rock, table, river, house, tree
# Neither: face, run, large, small, world
SAMPLE_TEXT = (
    "The virtue of justice is like a rock upon a table. "
    "Freedom and truth bring honor to the river and the house. "
    "A tree stands by the face of a large world. "
    "Run past the small rock and the table near the river."
)


# ---------------------------------------------------------------------------
# Tests for count_absconc
# ---------------------------------------------------------------------------


class TestCountAbsconc:
    def test_basic_counts(self):
        """count_absconc returns results with correct keys and consistent totals."""
        results = count_absconc(SAMPLE_TEXT, window_len=5)
        assert len(results) > 0
        for r in results:
            assert "num_abs" in r
            assert "num_conc" in r
            assert "num_neither" in r
            assert "num_total" in r
            assert r["num_total"] == r["num_abs"] + r["num_conc"] + r["num_neither"]

    def test_total_consistency(self):
        """num_total always equals sum of abs + conc + neither."""
        results = count_absconc(SAMPLE_TEXT, window_len=3)
        for r in results:
            assert r["num_total"] == r["num_abs"] + r["num_conc"] + r["num_neither"]

    def test_known_word_classification(self):
        """A tiny text where we know exactly what the counts should be."""
        # 5 recognized words: virtue(abs), rock(conc), table(conc), face(neither), justice(abs)
        txt = "virtue rock table face justice"
        results = count_absconc(txt, window_len=5, keep_last=True)
        # Filter to Median source only
        median_results = [r for r in results if r["source"] == "Median"]
        assert len(median_results) == 1
        r = median_results[0]
        assert r["num_abs"] == 2  # virtue, justice
        assert r["num_conc"] == 2  # rock, table
        assert r["num_neither"] == 1  # face
        assert r["num_total"] == 5

    def test_source_filter(self):
        """Filtering by source limits which contrasts appear."""
        results_median = count_absconc(SAMPLE_TEXT, window_len=5, sources={"Median"})
        results_other = count_absconc(SAMPLE_TEXT, window_len=5, sources={"Other"})
        sources_median = {r["source"] for r in results_median}
        sources_other = {r["source"] for r in results_other}
        assert sources_median == {"Median"}
        assert sources_other == {"Other"}

    def test_period_filter(self):
        """Filtering by period limits which contrasts appear."""
        results = count_absconc(SAMPLE_TEXT, window_len=5, periods={"median"})
        periods = {r["period"] for r in results}
        assert periods == {"median"}

    def test_empty_text(self):
        """Empty text produces no results."""
        results = count_absconc("", window_len=5)
        assert results == []

    def test_very_short_text_keep_last(self):
        """Short text with keep_last=True still produces results for recognized words."""
        results = count_absconc("rock", window_len=100, keep_last=True)
        median_results = [r for r in results if r["source"] == "Median"]
        assert len(median_results) == 1
        assert median_results[0]["num_conc"] == 1

    def test_very_short_text_no_keep_last(self):
        """Short text with keep_last=False and large window yields no results."""
        results = count_absconc("rock", window_len=100, keep_last=False)
        assert results == []

    def test_no_recognized_words(self):
        """Text with no words in any norm set yields empty results."""
        results = count_absconc("the and of but", window_len=5)
        assert results == []

    def test_slice_numbering(self):
        """Slice numbers are sequential starting from 1 within each contrast."""
        results = count_absconc(SAMPLE_TEXT, window_len=3, sources={"Median"})
        slices = [r["slice"] for r in results]
        assert slices == list(range(1, len(slices) + 1))

    def test_window_len_respected(self):
        """Each full window has exactly window_len recognized tokens."""
        window_len = 5
        results = count_absconc(SAMPLE_TEXT, window_len=window_len, keep_last=False)
        for r in results:
            assert r["num_tokens"] == window_len


# ---------------------------------------------------------------------------
# Tests for count_absconc_psg
# ---------------------------------------------------------------------------


class TestCountAbsconcPsg:
    def test_returns_dataframe(self):
        df = count_absconc_psg(SAMPLE_TEXT, sources={"Median"}, periods={"median"})
        assert isinstance(df, pd.DataFrame)

    def test_has_expected_columns(self):
        df = count_absconc_psg(SAMPLE_TEXT, sources={"Median"}, periods={"median"})
        assert "passage" in df.columns
        assert "abs-conc" in df.columns
        assert "num_abs" in df.columns
        assert "num_conc" in df.columns

    def test_abs_conc_computed(self):
        """abs-conc column equals num_abs - num_conc."""
        df = count_absconc_psg(SAMPLE_TEXT, sources={"Median"}, periods={"median"})
        if len(df):
            for _, row in df.iterrows():
                assert row["abs-conc"] == row["num_abs"] - row["num_conc"]

    def test_sorted_by_abs_conc(self):
        """Result is sorted by abs-conc ascending."""
        df = count_absconc_psg(SAMPLE_TEXT, sources={"Median"}, periods={"median"})
        if len(df) > 1:
            vals = df["abs-conc"].tolist()
            assert vals == sorted(vals)

    def test_empty_text_returns_empty_df(self):
        df = count_absconc_psg("", sources={"Median"}, periods={"median"})
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0


# ---------------------------------------------------------------------------
# Tests for count_absconc_path
# ---------------------------------------------------------------------------


class TestCountAbsconcPath:
    def test_reads_file(self, tmp_path):
        """count_absconc_path reads a text file and returns results with path key."""
        fpath = tmp_path / "sample.txt"
        fpath.write_text("virtue rock table face justice", encoding="utf-8")
        results = count_absconc_path(str(fpath), window_len=5)
        assert len(results) > 0
        for r in results:
            assert r["path"] == str(fpath)

    def test_prefers_uncompressed(self, tmp_path):
        """If a .gz path is given but the uncompressed version exists, use that."""
        fpath = tmp_path / "sample.txt"
        fpath.write_text("rock table river house tree", encoding="utf-8")
        gz_path = str(fpath) + ".gz"
        results = count_absconc_path(gz_path, window_len=5)
        assert len(results) > 0
        for r in results:
            assert r["path"] == str(fpath)

    def test_counts_match_direct(self, tmp_path):
        """Counts from file should match counts from direct text."""
        txt = "virtue rock face justice table"
        fpath = tmp_path / "test.txt"
        fpath.write_text(txt, encoding="utf-8")
        direct = count_absconc(txt, window_len=5, sources={"Median"})
        from_file = count_absconc_path(
            str(fpath), window_len=5, sources={"Median"}
        )
        # Strip path key for comparison
        for r in from_file:
            del r["path"]
        assert direct == from_file
