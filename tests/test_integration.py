"""
Integration tests using real data files and corpora.

These tests verify that modules work together end-to-end with actual
psycholinguistic norms, corpus texts, and pre-computed data.
"""

import numpy as np
import pandas as pd
import pytest

from abstraction.config import PATH_NORMS, PATH_VECNORMS, PATH_ALLNORMS, ZCUT
from abstraction.corpus import load_corpus
from abstraction.tokenize import tokenize, tokenize_agnostic, get_stopwords
from abstraction.norms import (
    get_orignorms, get_vecnorms, get_allnorms,
    get_origcontrasts, get_allcontrasts,
    get_origfields, format_norms_as_long, corr_norms,
    classify_word,
)
from abstraction.counting import count_absconc, count_absconc_psg
from abstraction.scoring import score_psg, score_freqs, score_words


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

class TestCorpusIntegration:
    def test_load_canon_fiction(self):
        corpus = load_corpus("CanonFiction")
        meta = corpus.metadata
        assert len(meta) > 100
        assert "id" in meta.columns
        assert "year" in meta.columns
        assert "author" in meta.columns

    def test_read_text(self):
        corpus = load_corpus("CanonFiction")
        text_id = corpus.metadata["id"].iloc[0]
        txt = corpus.read_text(text_id)
        assert isinstance(txt, str)
        assert len(txt) > 1000

    def test_text_paths(self):
        corpus = load_corpus("CanonFiction")
        paths = corpus.text_paths()
        assert len(paths) > 100
        tid, path = paths[0]
        assert tid == corpus.metadata["id"].iloc[0]
        assert path.endswith(".txt")


# ---------------------------------------------------------------------------
# Tokenize on real text
# ---------------------------------------------------------------------------

class TestTokenizeIntegration:
    @pytest.fixture
    def sample_text(self):
        corpus = load_corpus("CanonFiction")
        return corpus.read_text(corpus.metadata["id"].iloc[0])[:5000]

    def test_tokenize_produces_tokens(self, sample_text):
        tokens = tokenize(sample_text)
        assert len(tokens) > 100
        assert all(isinstance(t, str) for t in tokens)
        # lowercased by default
        assert all(t == t.lower() for t in tokens if t.isalpha())

    def test_tokenize_agnostic_produces_tokens(self, sample_text):
        tokens = tokenize_agnostic(sample_text)
        assert len(tokens) > 100

    def test_stopwords_loaded(self):
        sw = get_stopwords()
        assert isinstance(sw, set)
        assert len(sw) > 50
        assert "the" in sw
        assert "of" in sw


# ---------------------------------------------------------------------------
# Norms: loading and properties
# ---------------------------------------------------------------------------

class TestNormsIntegration:
    def test_orignorms_shape(self):
        df = get_orignorms()
        assert len(df) > 30_000
        assert "Abs-Conc.Median" in df.columns
        # should have standard norm sources
        for src in ["PAV-Conc", "MRC-Conc", "MT-Conc"]:
            assert f"Abs-Conc.{src}" in df.columns

    def test_orignorms_z_scored(self):
        df = get_orignorms()
        # z-scores should be roughly centered around 0
        median_col = df["Abs-Conc.Median"]
        assert abs(median_col.mean()) < 0.5
        assert median_col.std() > 0.5

    def test_orignorms_stopwords_removed(self):
        df = get_orignorms(remove_stopwords=True)
        sw = get_stopwords()
        overlap = set(df.index) & sw
        assert len(overlap) == 0

    def test_orignorms_stopwords_kept(self):
        df = get_orignorms(remove_stopwords=False)
        # "the" or common words should be present if in norms
        assert len(df) > len(get_orignorms(remove_stopwords=True))

    def test_vecnorms_shape(self):
        df = get_vecnorms()
        assert len(df) > 100_000
        # should have period columns
        periods = ["C16", "C17", "C18", "C19", "C20"]
        for p in periods:
            matching = [c for c in df.columns if c.endswith(f".{p}")]
            assert len(matching) > 0, f"No columns for period {p}"
        # should have median columns
        median_cols = [c for c in df.columns if c.endswith(".median")]
        assert len(median_cols) > 0

    def test_allnorms_combines_orig_and_vec(self):
        df = get_allnorms()
        assert len(df) > len(get_orignorms())  # vec adds words
        # should contain both orig and vec columns
        orig_cols = [c for c in df.columns if ".orig" in c]
        vec_cols = [c for c in df.columns if c.endswith(".C19")]
        assert len(orig_cols) > 0
        assert len(vec_cols) > 0

    def test_known_words_scores(self):
        """Check that well-known concrete/abstract words have expected polarity."""
        df = get_orignorms()
        median = df["Abs-Conc.Median"]
        # "rock" should be concrete (positive z)
        if "rock" in median.index:
            assert median["rock"] > 0
        # "justice" should be abstract (negative z)
        if "justice" in median.index:
            assert median["justice"] < 0


# ---------------------------------------------------------------------------
# Contrasts and fields
# ---------------------------------------------------------------------------

class TestContrastsIntegration:
    def test_origcontrasts_structure(self):
        contrasts = get_origcontrasts()
        assert len(contrasts) > 0
        for c in contrasts:
            assert "neg" in c and "pos" in c and "neither" in c
            assert isinstance(c["neg"], set)
            assert isinstance(c["pos"], set)
            assert len(c["neg"]) > 0
            assert len(c["pos"]) > 0
            assert c["contrast"] == "Abs-Conc"

    def test_origcontrasts_median(self):
        contrasts = get_origcontrasts()
        median_c = [c for c in contrasts if c["source"] == "Median"]
        assert len(median_c) == 1
        mc = median_c[0]
        # "rock" should be concrete (pos), "freedom" should be abstract (neg)
        assert "rock" in mc["pos"]
        assert "virtue" in mc["neg"]

    def test_origfields_keys(self):
        fields = get_origfields()
        assert isinstance(fields, dict)
        # should have Abs, Conc, Neither for each source
        assert any("Abs" in k for k in fields)
        assert any("Conc" in k for k in fields)
        assert any("Neither" in k for k in fields)

    def test_allcontrasts_more_than_orig(self):
        orig = get_origcontrasts()
        all_c = get_allcontrasts()
        assert len(all_c) > len(orig)


# ---------------------------------------------------------------------------
# Format norms as long (with real data)
# ---------------------------------------------------------------------------

class TestFormatNormsIntegration:
    def test_long_format(self):
        df = get_orignorms()
        long = format_norms_as_long(df)
        assert "word" in long.columns
        assert "z" in long.columns
        assert "decision" in long.columns
        assert "source" in long.columns
        assert len(long) > len(df)  # multiple sources per word

    def test_decisions_consistent(self):
        df = get_orignorms()
        long = format_norms_as_long(df)
        for _, row in long.sample(100, random_state=42).iterrows():
            expected = classify_word(row["z"])
            assert row["decision"] == expected


# ---------------------------------------------------------------------------
# Correlations
# ---------------------------------------------------------------------------

class TestCorrNormsIntegration:
    def test_corr_norms(self):
        df = get_orignorms()
        cordf = corr_norms(df)
        assert "value" in cordf.columns
        assert len(cordf) > 0
        # concreteness sources should correlate positively
        conc_pair = cordf[
            (cordf["index"].str.contains("Conc")) &
            (cordf["variable"].str.contains("Conc"))
        ]
        if len(conc_pair) > 0:
            assert conc_pair["value"].iloc[0] > 0


# ---------------------------------------------------------------------------
# Counting: sliding-window on real text
# ---------------------------------------------------------------------------

class TestCountingIntegration:
    @pytest.fixture
    def sample_text(self):
        corpus = load_corpus("CanonFiction")
        return corpus.read_text(corpus.metadata["id"].iloc[0])[:20_000]

    def test_count_absconc_returns_results(self, sample_text):
        results = count_absconc(sample_text)
        assert len(results) > 0
        r = results[0]
        assert "num_abs" in r
        assert "num_conc" in r
        assert "num_neither" in r
        assert "num_total" in r
        assert r["num_total"] == r["num_abs"] + r["num_conc"] + r["num_neither"]

    def test_count_window_sizes_consistent(self, sample_text):
        # filter to single source/period so windows don't interleave
        results = count_absconc(sample_text, window_len=100,
                                sources={"Median"}, periods={"median"})
        # all windows except possibly last should have 100 recognized tokens
        for r in results[:-1]:
            assert r["num_total"] == 100

    def test_count_absconc_psg(self, sample_text):
        df = count_absconc_psg(sample_text)
        assert isinstance(df, pd.DataFrame)
        if len(df) > 0:
            assert "passage" in df.columns
            assert "abs-conc" in df.columns
            assert "num_abs" in df.columns
            # should be sorted by abs-conc
            assert df["abs-conc"].is_monotonic_increasing

    def test_count_sources_filter(self, sample_text):
        results = count_absconc(sample_text, sources={"Median"}, periods={"median"})
        assert len(results) > 0
        for r in results:
            assert r["source"] == "Median"
            assert r["period"] == "median"


# ---------------------------------------------------------------------------
# Scoring: passage and word-level with real norms
# ---------------------------------------------------------------------------

class TestScoringIntegration:
    def test_score_psg_concrete(self):
        """A passage about physical objects should score positive (concrete)."""
        txt = "The heavy stone rock fell on the wooden table beside the metal chair."
        score = score_psg(txt)
        assert isinstance(score, float)
        assert not np.isnan(score)
        assert score > 0  # concrete words dominate

    def test_score_psg_abstract(self):
        """A passage about ideas should score negative (abstract)."""
        txt = "Justice and liberty require moral virtue and ethical wisdom."
        score = score_psg(txt)
        assert isinstance(score, float)
        assert not np.isnan(score)
        assert score < 0  # abstract words dominate

    def test_score_freqs_with_real_norms(self):
        score = score_freqs({"rock": 10, "stone": 5, "table": 3})
        assert isinstance(score, float)
        assert not np.isnan(score)
        assert score > 0  # all concrete

    def test_score_words_on_real_text(self):
        txt = "The rock of virtue stands in the garden of justice."
        df = score_words(txt)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        rock = df[df["word"] == "rock"]
        if len(rock) > 0:
            assert rock.iloc[0]["score"] > 0
            assert rock.iloc[0]["is_concrete"]
        virtue = df[df["word"] == "virtue"]
        if len(virtue) > 0:
            assert virtue.iloc[0]["score"] < 0
            assert virtue.iloc[0]["is_abstract"]

    def test_score_words_coverage(self):
        """Real norms should cover a decent fraction of common English words."""
        txt = "The man walked to the house and opened the door to find a book on the table."
        df = score_words(txt)
        scored = df["score"].notna().sum()
        total = len(df)
        assert scored / total > 0.3  # at least 30% of words have norms


# ---------------------------------------------------------------------------
# End-to-end pipeline: corpus → tokenize → count → score
# ---------------------------------------------------------------------------

class TestPipelineIntegration:
    def test_full_pipeline(self):
        """Load a text, tokenize, count, and score — full pipeline."""
        corpus = load_corpus("CanonFiction")
        meta = corpus.metadata
        # pick a text with a known year
        text_row = meta[meta["year"] > 1700].iloc[0]
        text_id = text_row["id"]
        txt = corpus.read_text(text_id)

        # tokenize
        tokens = tokenize(txt[:10_000])
        assert len(tokens) > 100

        # count
        results = count_absconc(txt[:10_000])
        assert len(results) > 0

        # score a passage from the text
        passage = " ".join(tokens[:50])
        score = score_psg(passage)
        assert isinstance(score, float)

    def test_multiple_texts_score_variation(self):
        """Different texts should produce different scores — not a constant."""
        corpus = load_corpus("CanonFiction")
        meta = corpus.metadata
        scores = []
        for _, row in meta.sample(5, random_state=42).iterrows():
            try:
                txt = corpus.read_text(row["id"])[:5000]
                s = score_psg(txt)
                if not np.isnan(s):
                    scores.append(s)
            except FileNotFoundError:
                continue
        assert len(scores) >= 2
        assert max(scores) != min(scores)  # not all identical
