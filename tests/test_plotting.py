from abstraction.plotting import _compress_year


class TestCompressYear:
    def test_modern_unchanged(self):
        assert _compress_year(1700) == 1700
        assert _compress_year(1850) == 1850
        assert _compress_year(2000) == 2000

    def test_1600_boundary(self):
        assert _compress_year(1600) == 1600

    def test_pre1600_compressed(self):
        y = _compress_year(1500)
        assert y < 1600
        assert y > _compress_year(1000)

    def test_ordering_preserved(self):
        years = [-500, 0, 500, 1000, 1500, 1600, 1800, 2000]
        compressed = [_compress_year(y) for y in years]
        assert compressed == sorted(compressed)
