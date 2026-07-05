"""Unit tests for abstraction.passages."""

import html as html_mod
import os
import tempfile

import numpy as np
import pytest

from abstraction.passages import (
    _word_style,
    _render_body,
    _tokenize_display,
    render_passage_html,
    render_comparison_html,
    save_passage_html,
)


# ---------------------------------------------------------------------------
# Fake norms for deterministic tests
# ---------------------------------------------------------------------------

FAKE_NORMS = {
    "virtue": -2.0,     # abstract
    "justice": -1.5,    # abstract
    "moral": -1.8,      # abstract
    "stone": 2.0,       # concrete
    "wall": 2.1,        # concrete
    "brick": 2.5,       # concrete
    "the": 0.0,         # neither
    "old": 0.3,         # neither
}


@pytest.fixture(autouse=True)
def patch_norms(install_fake_norms):
    """Install FAKE_NORMS via the shared conftest contract (tuple-keyed
    _NORM_DICTS cache), so passages.py's from-imported get_norm_dict sees it."""
    install_fake_norms(FAKE_NORMS)


# ---------------------------------------------------------------------------
# _word_style
# ---------------------------------------------------------------------------

class TestWordStyle:
    def test_nan_returns_unscored(self):
        css, cls = _word_style(np.nan)
        assert css is None
        assert cls == "unscored"

    def test_abstract_gets_border(self):
        css, cls = _word_style(-2.0)
        assert cls == "abstract"
        assert "outline:" in css
        assert "solid" in css

    def test_concrete_gets_bold_and_background(self):
        css, cls = _word_style(2.0)
        assert cls == "concrete"
        assert "font-weight:" in css
        assert "background:rgba" in css

    def test_zero_gets_light_border(self):
        css, cls = _word_style(0.0)
        assert css is not None
        assert "outline:" in css
        assert cls == "neither"

    def test_slightly_abstract_gets_border(self):
        css, cls = _word_style(-0.5)
        assert css is not None
        assert "outline:" in css
        assert cls == "neither"

    def test_slightly_concrete_gets_shading(self):
        css, cls = _word_style(0.5)
        assert css is not None
        assert "background:rgba" in css
        assert cls == "neither"

    def test_abstract_border_scales_with_z(self):
        css_mild, _ = _word_style(-0.5)
        css_extreme, _ = _word_style(-3.0)
        mild_px = int(css_mild.split("outline:")[1].split("px")[0])
        extreme_px = int(css_extreme.split("outline:")[1].split("px")[0])
        assert extreme_px > mild_px

    def test_concrete_weight_scales_with_z(self):
        css_mild, _ = _word_style(0.5)
        css_extreme, _ = _word_style(3.0)
        mild_weight = int(css_mild.split("font-weight:")[1].split(";")[0])
        extreme_weight = int(css_extreme.split("font-weight:")[1].split(";")[0])
        assert extreme_weight > mild_weight

    def test_concrete_alpha_scales_with_z(self):
        css_mild, _ = _word_style(0.5)
        css_extreme, _ = _word_style(3.0)
        mild_alpha = float(css_mild.split("rgba(0,0,0,")[1].split(")")[0])
        extreme_alpha = float(css_extreme.split("rgba(0,0,0,")[1].split(")")[0])
        assert extreme_alpha > mild_alpha

    def test_boundary_z_minus_one_is_abstract_inclusive(self):
        # norms.classify_word uses z <= -1 -> "Abstract" (inclusive); the
        # passage CSS class must agree (AUDIT-2026-07-04.md §2.11) instead
        # of the previous strict `z < -1.0`.
        _, cls = _word_style(-1.0)
        assert cls == "abstract"

    def test_boundary_z_plus_one_is_concrete_inclusive(self):
        # norms.classify_word uses z >= 1 -> "Concrete" (inclusive).
        _, cls = _word_style(1.0)
        assert cls == "concrete"


# ---------------------------------------------------------------------------
# _render_body
# ---------------------------------------------------------------------------

class TestRenderBody:
    def test_abstract_word_gets_border_span(self):
        html = _render_body("the virtue")
        assert 'class="w abstract"' in html
        assert "outline:" in html

    def test_concrete_word_gets_bold_span(self):
        html = _render_body("the stone wall")
        assert 'class="w concrete"' in html
        assert "font-weight:" in html

    def test_neither_word_gets_light_style(self):
        html = _render_body("old")
        assert 'class="w neither"' in html
        assert "style=" in html

    def test_unknown_word_no_style(self):
        html = _render_body("xyzzyplugh")
        assert 'class="w unscored"' in html
        assert "style=" not in html

    def test_punctuation_no_leading_space(self):
        html = _render_body("the wall, old")
        # comma should not have a space before it
        assert " ," not in html

    def test_hyphenated_words_no_space(self):
        html = _render_body("Bed-chamber")
        # No space between hyphen and following word
        assert "- <" not in html  # no "- <span..." gap
        assert "-<span" in html   # hyphen directly abuts next span

    def test_paragraph_break_creates_paragraphs(self):
        html = _render_body("the wall\n\nthe stone")
        assert "<p " in html
        assert html.count("<p ") == 2

    def test_single_newline_collapses_to_space(self):
        html = _render_body("the\nwall")
        # Single newline should not create a new paragraph
        assert html.count("<p ") == 1

    def test_html_escaping(self):
        # render_passage_html escapes title text
        html = render_passage_html("the wall", title="A & B <C>")
        assert "&amp;" in html
        assert "&lt;C&gt;" in html


# ---------------------------------------------------------------------------
# Punctuation losslessness (AUDIT-2026-07-04.md §2.8)
#
# tokenize.tokenize_agnostic()'s punctuation class is a closed whitelist
# with no entry for `:`, `"`, `(`, `)` (etc.), so re.findall silently drops
# them. passages.py must use a display-only tokenizer that never drops a
# character, while still scoring the exact same words the production
# scoring path would.
# ---------------------------------------------------------------------------

_PUNCT_SAMPLE = 'He said: "Truth is stone," isn\'t it? (softly) — she left.'


class TestTokenizeDisplayLossless:
    def test_round_trips_every_character(self):
        # Concatenating every token must reproduce the input exactly —
        # the strongest possible "nothing was dropped" guarantee.
        assert "".join(_tokenize_display(_PUNCT_SAMPLE)) == _PUNCT_SAMPLE

    def test_previously_dropped_characters_are_their_own_tokens(self):
        tokens = _tokenize_display('a:b "c" (d)')
        assert ":" in tokens
        assert '"' in tokens
        assert "(" in tokens
        assert ")" in tokens

    def test_word_tokens_match_production_tokenizer(self):
        # The word-matching alternative must stay identical to
        # tokenize.tokenize_agnostic's, so scoring lookups (which key off
        # these same lowercased word tokens) are unaffected by the display
        # fix.
        from abstraction.tokenize import tokenize_agnostic

        txt = "Bed-chamber said: \"hello\" (softly) it's fine."
        word_tokens = [t for t in _tokenize_display(txt) if t and t[0].isalpha()]
        prod_word_tokens = [t for t in tokenize_agnostic(txt) if t and t[0].isalpha()]
        assert word_tokens == prod_word_tokens


class TestRenderPassageHtmlPunctuationLossless:
    def test_all_punctuation_survives_and_known_word_still_scored(self):
        html = render_passage_html(_PUNCT_SAMPLE, show_legend=False, show_title=False)
        # Characters may be re-encoded as HTML entities (e.g. `"` ->
        # `&quot;`) by html.escape, which is faithful — decode before
        # checking so the assertion is about information loss, not about
        # literal-vs-entity spelling.
        decoded = html_mod.unescape(html)
        for ch in ':"()—' + "'":
            assert ch in decoded, f"{ch!r} did not survive into rendered HTML"
        # A known scored word ("stone", concrete per FAKE_NORMS) must
        # still get its span — the display fix must not break scoring.
        assert 'class="w concrete"' in html
        assert ">stone<" in html


# ---------------------------------------------------------------------------
# render_passage_html
# ---------------------------------------------------------------------------

class TestRenderPassageHtml:
    def test_returns_complete_html(self):
        html = render_passage_html("the stone wall")
        assert "<!DOCTYPE html>" in html
        assert "<html>" in html
        assert "</html>" in html

    def test_title_shown_by_default(self):
        html = render_passage_html("the stone", title="My Title")
        assert "My Title" in html

    def test_show_title_false(self):
        html = render_passage_html("the stone", title="My Title", show_title=False)
        assert "My Title" not in html

    def test_show_legend_true_by_default(self):
        html = render_passage_html("the stone")
        assert "abstract</span>" in html
        assert "concrete</span>" in html

    def test_legend_swatches_match_word_style_formula(self):
        # The legend must be generated from _word_style itself (not
        # hand-copied CSS literals) so it can never drift from how words
        # are actually rendered (AUDIT-2026-07-04.md §2.11 idea / §12.17).
        from abstraction.passages import _LEGEND_Z

        html = render_passage_html("the stone")
        for z in _LEGEND_Z.values():
            css, _ = _word_style(z)
            assert css in html

    def test_show_legend_false(self):
        html = render_passage_html("the stone", show_legend=False)
        # Legend div should not be present (CSS class name in stylesheet is fine)
        assert "abstract</span>" not in html

    def test_title_html_escaped(self):
        html = render_passage_html("the stone", title="<script>alert('x')</script>")
        assert "&lt;script&gt;" in html
        assert "<script>" not in html


# ---------------------------------------------------------------------------
# render_comparison_html
# ---------------------------------------------------------------------------

class TestRenderComparisonHtml:
    def test_two_columns(self):
        passages = [
            {"text": "the virtue", "title": "Abstract"},
            {"text": "the stone wall", "title": "Concrete"},
        ]
        html = render_comparison_html(passages)
        assert html.count("<td>") == 2

    def test_show_titles_false(self):
        passages = [{"text": "the virtue", "title": "Hidden Title"}]
        html = render_comparison_html(passages, show_titles=False)
        assert "Hidden Title" not in html

    def test_show_legend_false(self):
        passages = [{"text": "the stone"}]
        html = render_comparison_html(passages, show_legend=False)
        assert "abstract</span>" not in html

    def test_show_titles_true_by_default(self):
        passages = [{"text": "the virtue", "title": "Shown Title"}]
        html = render_comparison_html(passages)
        assert "Shown Title" in html


# ---------------------------------------------------------------------------
# save_passage_html
# ---------------------------------------------------------------------------

class TestSavePassageHtml:
    def test_writes_file(self):
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            result = save_passage_html("the stone wall", path, title="Test")
            assert result == path
            assert os.path.exists(path)
            content = open(path).read()
            assert "<!DOCTYPE html>" in content
            assert "stone" in content
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# save_passage_image (integration — only if playwright available)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestSavePassageImage:
    @pytest.fixture
    def has_playwright(self):
        try:
            from playwright.sync_api import sync_playwright
            return True
        except ImportError:
            pytest.skip("playwright not installed")

    def test_png_export(self, has_playwright):
        from abstraction.passages import save_passage_image
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            save_passage_image("the stone wall", path, dpi=96)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)

    def test_dpi_scales_output(self, has_playwright):
        from abstraction.passages import save_passage_image
        from PIL import Image
        paths = []
        try:
            for dpi in (96, 288):
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                    paths.append(f.name)
                save_passage_image("the stone wall", paths[-1], dpi=dpi, width=400)
            img_lo = Image.open(paths[0])
            img_hi = Image.open(paths[1])
            # 3x DPI should produce ~3x wider image
            assert img_hi.size[0] > img_lo.size[0] * 2
        finally:
            for p in paths:
                if os.path.exists(p):
                    os.unlink(p)
