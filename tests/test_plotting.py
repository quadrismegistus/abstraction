import os
import tempfile

import matplotlib
matplotlib.use("Agg")  # headless backend before any pyplot import happens

import numpy as np
import pandas as pd
import pytest

from abstraction.config import PATH_FIGS
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


# ---------------------------------------------------------------------------
# plot_norms: seeded sampling (AUDIT-2026-07-04.md §2.9)
# ---------------------------------------------------------------------------

def _make_dfnorms(n_words=30, seed=0):
    rng = np.random.RandomState(seed)
    words = [f"w{i}" for i in range(n_words)]
    rows = []
    for src in ["PAV-Conc", "MRC-Conc"]:
        for w in words:
            rows.append({
                "word": w, "z": rng.uniform(-2.5, 2.5),
                "source": src, "source_type": "Abstract",
            })
    return pd.DataFrame(rows)


class TestPlotNormsRandomState:
    def test_same_seed_same_sample(self):
        from abstraction.plotting import plot_norms

        df = _make_dfnorms()
        fig1 = plot_norms(df, sample_n=5, random_state=42)
        fig2 = plot_norms(df, sample_n=5, random_state=42)
        assert sorted(fig1.data["word"].unique()) == sorted(fig2.data["word"].unique())

    def test_different_seed_can_differ(self):
        from abstraction.plotting import plot_norms

        df = _make_dfnorms()
        fig1 = plot_norms(df, sample_n=5, random_state=1)
        fig2 = plot_norms(df, sample_n=5, random_state=2)
        assert sorted(fig1.data["word"].unique()) != sorted(fig2.data["word"].unique())


# ---------------------------------------------------------------------------
# plot_fiction: min_y floor no longer drops points (AUDIT-2026-07-04.md §5)
# ---------------------------------------------------------------------------

class TestPlotFictionMinYFloor:
    def test_sub_floor_outlier_is_floored_not_dropped(self):
        from abstraction.plotting import plot_fiction

        df = pd.DataFrame({
            "year": [1700, 1750, 1800, 1850],
            "num_abs": [1, 2, 3, 200],
            "num_conc": [5, 5, 200, 5],
        })
        fig = plot_fiction(
            df, valtype="abs-conc", min_y=-10, max_y=10,
            color_by=None, shape_by=None, label_by=None, smooth=False,
        )
        # Previously: value = 3 - 200 = -197, well below min_y, and the
        # buggy compression (`min_y + (y - min_y) * 0.5`) still left it
        # below min_y, where p9.ylim silently dropped the whole row.
        assert fig.data["value"].min() >= -10
        assert fig.data["value"].max() <= 10
        assert len(fig.data) == len(df)


# ---------------------------------------------------------------------------
# facet_by_genre: color_col=None must not KeyError (AUDIT-2026-07-04.md §5)
# ---------------------------------------------------------------------------

class TestFacetByGenreColorColNone:
    def test_no_color_col_does_not_crash(self):
        from abstraction.plotting import facet_by_genre

        df = pd.DataFrame({
            "umap_x": np.random.rand(10),
            "umap_y": np.random.rand(10),
            "genre_novel": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            "genre_romance": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        })
        fig = facet_by_genre(df, ["genre_novel", "genre_romance"], color_col=None)
        assert fig is not None


# ---------------------------------------------------------------------------
# _savefig: default figure directory (AUDIT-2026-07-04.md §5)
# ---------------------------------------------------------------------------

class TestSavefigDefaultDir:
    def test_defaults_to_path_figs_not_cwd_relative(self):
        import matplotlib.pyplot as plt
        from abstraction.plotting import _savefig

        fig = plt.figure()
        with tempfile.TemporaryDirectory() as d:
            old_cwd = os.getcwd()
            os.chdir(d)
            try:
                _savefig(fig, "test_savefig_default.png")
                expected = os.path.join(PATH_FIGS, "test_savefig_default.png")
                assert os.path.exists(expected)
            finally:
                os.chdir(old_cwd)
                if os.path.exists(expected):
                    os.remove(expected)
        plt.close(fig)


# ---------------------------------------------------------------------------
# UMAP plots: empty-valid-subset NaN color bounds (AUDIT-2026-07-04.md §5)
# ---------------------------------------------------------------------------

class TestUmapEmptyValidGuard:
    def _make_df(self, abs_score):
        return pd.DataFrame({
            "umap_x": np.random.rand(6),
            "umap_y": np.random.rand(6),
            "lang": ["en"] * 6,
            "abs_score": abs_score,
            "period_bin": ["C18a"] * 6,
        })

    def test_all_nan_abs_score_skips_colorbar(self):
        from abstraction.plotting import plot_text_umap_4panel

        df = self._make_df([np.nan] * 6)
        fig = plot_text_umap_4panel(df, xcol="umap_x", ycol="umap_y", form_tags=[])
        # No NaN-normalized colorbar axes should be appended when there's
        # nothing valid to color by.
        assert len(fig.axes) == 4

    def test_some_valid_abs_score_still_adds_colorbar(self):
        from abstraction.plotting import plot_text_umap_4panel

        df = self._make_df(np.random.rand(6))
        fig = plot_text_umap_4panel(df, xcol="umap_x", ycol="umap_y", form_tags=[])
        assert len(fig.axes) == 5
