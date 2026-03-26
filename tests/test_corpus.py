from abstraction.corpus import _camel_to_snake, Corpus, pmap


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


class TestPmap:
    def test_single_proc(self):
        results = pmap(lambda x: x * 2, [1, 2, 3], num_proc=1)
        assert sorted(results) == [2, 4, 6]

    def test_empty(self):
        results = pmap(lambda x: x, [], num_proc=1)
        assert results == []
