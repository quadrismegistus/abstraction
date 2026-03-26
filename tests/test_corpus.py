import os
import shutil

import pytest

from abstraction.corpus import _camel_to_snake, Corpus, load_corpus, pmap

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "test_corpus")


class TestCamelToSnake:
    def test_simple(self):
        assert _camel_to_snake("CanonFiction") == "canon_fiction"

    def test_single_word(self):
        assert _camel_to_snake("Novel") == "novel"

    def test_already_lower(self):
        assert _camel_to_snake("corpus") == "corpus"

    def test_multiple_caps(self):
        # consecutive uppercase letters don't get split individually
        assert _camel_to_snake("EEBOTcp") == "eebotcp"

    def test_three_words(self):
        assert _camel_to_snake("HathiEngLit") == "hathi_eng_lit"


class TestCorpus:
    def test_loads_metadata(self):
        c = Corpus("test_corpus", root=os.path.dirname(FIXTURE_PATH))
        assert len(c.metadata) == 3
        assert "id" in c.metadata.columns
        assert "author" in c.metadata.columns
        assert "title" in c.metadata.columns
        assert "year" in c.metadata.columns

    def test_text_path(self):
        c = Corpus("test_corpus", root=os.path.dirname(FIXTURE_PATH))
        expected = os.path.join(FIXTURE_PATH, "txt", "text1.txt")
        assert c.text_path("text1") == expected

    def test_read_text(self):
        c = Corpus("test_corpus", root=os.path.dirname(FIXTURE_PATH))
        text = c.read_text("text1")
        assert "truth universally acknowledged" in text

    def test_read_text_nested_id(self):
        c = Corpus("test_corpus", root=os.path.dirname(FIXTURE_PATH))
        text = c.read_text("subdir/text3")
        assert "Mrs Ramsay" in text

    def test_text_paths_returns_existing(self):
        c = Corpus("test_corpus", root=os.path.dirname(FIXTURE_PATH))
        paths = c.text_paths()
        assert len(paths) == 3
        for text_id, path in paths:
            assert os.path.exists(path)

    def test_text_path_nonexistent_id(self):
        c = Corpus("test_corpus", root=os.path.dirname(FIXTURE_PATH))
        path = c.text_path("nonexistent")
        # Should return a path without erroring, even if file doesn't exist
        assert path.endswith("nonexistent.txt")
        assert not os.path.exists(path)

    def test_load_corpus_camel_case(self, tmp_path, monkeypatch):
        # Create a snake_case directory from the fixture
        dest = tmp_path / "test_corpus"
        shutil.copytree(FIXTURE_PATH, dest)
        c = load_corpus("TestCorpus", root=str(tmp_path))
        assert c.id == "test_corpus"
        assert len(c.metadata) == 3


class TestPmap:
    def test_single_proc(self):
        results = pmap(lambda x: x * 2, [1, 2, 3], num_proc=1)
        assert sorted(results) == [2, 4, 6]

    def test_empty(self):
        results = pmap(lambda x: x, [], num_proc=1)
        assert results == []
