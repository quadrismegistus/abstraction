from abstraction.tokenize import (
    tokenize, tokenize_agnostic, _strip_punct,
)


class TestTokenizeAgnostic:
    def test_basic(self):
        tokens = tokenize_agnostic("hello world")
        assert "hello" in tokens
        assert "world" in tokens

    def test_punctuation_separate(self):
        tokens = tokenize_agnostic("hello, world!")
        assert "hello" in tokens
        assert "," in tokens
        assert "world" in tokens
        assert "!" in tokens

    def test_contractions(self):
        tokens = tokenize_agnostic("don't stop")
        assert "don't" in tokens
        assert "stop" in tokens

    def test_empty(self):
        assert tokenize_agnostic("") == []


class TestTokenize:
    def test_lowercases_by_default(self):
        tokens = tokenize("Hello World")
        words = [t for t in tokens if t[0].isalpha()]
        assert all(w == w.lower() for w in words)

    def test_no_lowercase(self):
        tokens = tokenize("Hello World", lower=False)
        words = [t for t in tokens if t[0].isalpha()]
        assert "Hello" in words

    def test_xml_entities(self):
        tokens = tokenize("good&mdash;bad")
        words = [t for t in tokens if t[0].isalpha()]
        assert "good" in words
        assert "bad" in words

    def test_long_s(self):
        tokens = tokenize("plea&longs;ure")
        words = [t for t in tokens if t[0].isalpha()]
        assert "pleasure" in words

    def test_unicode_dashes(self):
        tokens = tokenize("word\u2014another")
        words = [t for t in tokens if t[0].isalpha()]
        assert "word" in words
        assert "another" in words


class TestStripPunct:
    def test_no_punct(self):
        assert _strip_punct("hello") == ("", "hello", "")

    def test_trailing(self):
        assert _strip_punct("hello,") == ("", "hello", ",")

    def test_leading(self):
        assert _strip_punct("(hello") == ("(", "hello", "")

    def test_both(self):
        assert _strip_punct('"hello"') == ('"', "hello", '"')

    def test_empty(self):
        assert _strip_punct("") == ("", "", "")
