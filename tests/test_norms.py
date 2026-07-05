import os

from abstraction.norms import classify_word, get_contrasts, format_norms_as_long
import pandas as pd
import numpy as np
import pytest

from abstraction.config import PATH_ALLNORMS_FR, PATH_ALLNORMS_DE, PATH_ALLNORMS_ES


class TestClassifyWord:
    def test_abstract(self):
        assert classify_word(-1.5) == "Abstract"
        assert classify_word(-1.0) == "Abstract"

    def test_concrete(self):
        assert classify_word(1.5) == "Concrete"
        assert classify_word(1.0) == "Concrete"

    def test_neither(self):
        assert classify_word(0.0) == "Neither"
        assert classify_word(0.5) == "Neither"
        assert classify_word(-0.5) == "Neither"
        assert classify_word(0.99) == "Neither"
        assert classify_word(-0.99) == "Neither"

    def test_custom_zcut(self):
        assert classify_word(0.5, zcut=0.5) == "Concrete"
        assert classify_word(-0.5, zcut=0.5) == "Abstract"
        assert classify_word(0.3, zcut=0.5) == "Neither"


class TestGetContrasts:
    def _make_norms(self):
        return pd.DataFrame({
            "Abs-Conc.Test": [2.0, -2.0, 0.0, 1.5, -0.5],
        }, index=["rock", "virtue", "table", "hammer", "idea"])

    def test_returns_list(self):
        contrasts = get_contrasts(self._make_norms())
        assert isinstance(contrasts, list)
        assert len(contrasts) == 1

    def test_classifies_words(self):
        contrasts = get_contrasts(self._make_norms())
        c = contrasts[0]
        assert "rock" in c["pos"]       # z=2.0, concrete
        assert "virtue" in c["neg"]     # z=-2.0, abstract
        assert "table" in c["neither"]  # z=0.0
        assert c["contrast"] == "Abs-Conc"
        assert c["source"] == "Test"

    def test_custom_zcut(self):
        contrasts = get_contrasts(self._make_norms(), zcut=0.3)
        c = contrasts[0]
        assert "idea" in c["neg"]  # z=-0.5, below -0.3

    def test_skips_columns_without_dash(self):
        # IC.* (information-content) columns have no "-" in the contrast
        # part (e.g. "IC.Median.median") and must not crash get_contrasts.
        df = pd.DataFrame({
            "Abs-Conc.Test": [2.0, -2.0, 0.0, 1.5, -0.5],
            "IC.Median.median": [3.1, 4.2, 5.0, 2.7, 6.6],
        }, index=["rock", "virtue", "table", "hammer", "idea"])
        contrasts = get_contrasts(df)
        assert len(contrasts) == 1
        assert contrasts[0]["contrast"] == "Abs-Conc"
        assert not any(c["contrast"] == "IC" for c in contrasts)


class TestFormatNormsAsLong:
    def test_output_shape(self):
        df = pd.DataFrame({
            "Abs-Conc.Test": [2.0, -2.0, 0.0],
        }, index=["rock", "virtue", "table"])
        long = format_norms_as_long(df)
        assert "word" in long.columns
        assert "z" in long.columns
        assert "decision" in long.columns
        assert "source" in long.columns
        assert len(long) == 3

    def test_decisions_match_classify(self):
        df = pd.DataFrame({
            "Abs-Conc.Test": [2.0, -2.0, 0.0],
        }, index=["rock", "virtue", "table"])
        long = format_norms_as_long(df)
        rock = long[long["word"] == "rock"].iloc[0]
        assert rock["decision"] == "Concrete"
        virtue = long[long["word"] == "virtue"].iloc[0]
        assert virtue["decision"] == "Abstract"


# ---------------------------------------------------------------------------
# Multilingual polarity smoke tests
#
# These load the REAL generated allnorms pickles (FR/DE/ES) through the
# public getters and check a handful of unambiguous words land on the right
# side of zero on "Abs-Conc.Median.median" (positive = concrete, negative =
# abstract -- the same convention as classify_word/get_contrasts above). A
# reverse-coding bug (e.g. forgetting to negate a source, or mixing up which
# pole is "positive" when combining sources -- see the Kanske reverse-coding
# in norms_de.py) would silently flip an entire language's arc; these tests
# exist to catch exactly that. They're marked `integration` (see conftest.py)
# and skip automatically when the local data/ volume isn't mounted.
# ---------------------------------------------------------------------------

_MEDIAN_COL = "Abs-Conc.Median.median"


def _assert_polarity(df, concrete_words, abstract_words, col=_MEDIAN_COL):
    for w in concrete_words:
        assert w in df.index, f"expected concrete word {w!r} missing from norms"
        v = df.loc[w, col]
        assert v > 0, f"{w!r} expected concrete-positive on {col}, got {v}"
    for w in abstract_words:
        assert w in df.index, f"expected abstract word {w!r} missing from norms"
        v = df.loc[w, col]
        assert v < 0, f"{w!r} expected abstract-negative on {col}, got {v}"


@pytest.mark.integration
@pytest.mark.skipif(
    not os.path.exists(PATH_ALLNORMS_FR),
    reason=f"French allnorms pickle not present at {PATH_ALLNORMS_FR}",
)
class TestFrenchNormsPolarity:
    def test_polarity(self):
        from abstraction.norms_fr import get_allnorms_fr
        df = get_allnorms_fr()
        _assert_polarity(df, ["pierre", "table"], ["vérité", "idée"])


@pytest.mark.integration
@pytest.mark.skipif(
    not os.path.exists(PATH_ALLNORMS_DE),
    reason=f"German allnorms pickle not present at {PATH_ALLNORMS_DE}",
)
class TestGermanNormsPolarity:
    def test_polarity(self):
        from abstraction.norms_de import get_allnorms_de
        df = get_allnorms_de()
        _assert_polarity(df, ["stein", "tisch"], ["wahrheit", "begriff"])


@pytest.mark.integration
@pytest.mark.skipif(
    not os.path.exists(PATH_ALLNORMS_ES),
    reason=f"Spanish allnorms pickle not present at {PATH_ALLNORMS_ES}",
)
class TestSpanishNormsPolarity:
    def test_polarity(self):
        from abstraction.norms_es import get_allnorms_es
        df = get_allnorms_es()
        _assert_polarity(df, ["piedra", "mesa"], ["verdad", "idea"])
