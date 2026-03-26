from abstraction.norms import classify_word, get_contrasts, format_norms_as_long
import pandas as pd
import numpy as np


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
