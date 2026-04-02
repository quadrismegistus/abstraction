"""
Passage visualization with per-word abstract/concrete styling.

Grayscale-friendly design for print:
  - Abstract words: bordered (thicker border = more abstract)
  - Concrete words: bold (heavier weight = more concrete)
  - Neutral words: plain text
"""

import asyncio
import html as html_mod
import os
import textwrap

import numpy as np

from .scoring import score_words, get_norm_dict, _modernize_score
from .tokenize import tokenize_agnostic, get_spelling_modernizer


# ---------------------------------------------------------------------------
# Per-word HTML rendering
# ---------------------------------------------------------------------------

def _word_style(z, max_z=3.0):
    """Return inline CSS for a word given its z-score.

    Continuous scale — every scored word gets styled proportionally:
    - z < 0 (abstract): bordered rectangle, thicker with |z|
    - z > 0 (concrete): gray background shading, darker with z
    - z == 0: minimal styling (very thin border)
    - NaN (unrecognized): no styling
    """
    if np.isnan(z):
        return None, "unscored"

    if z <= 0:
        # Abstract: outline scales from 1px (z=0) to 4px (z=-max_z)
        # Using outline instead of border so it doesn't affect layout
        intensity = min(abs(z), max_z) / max_z  # 0..1
        border_px = 1 + round(intensity * 3)  # 1..4
        alpha = 0.15 + intensity * 0.70  # 0.15..0.85
        css = (f"outline:{border_px}px solid rgba(0,0,0,{alpha:.2f}); "
               f"outline-offset:0px; border-radius:2px")
        cls = "abstract" if z < -1.0 else "neither"
        return css, cls

    # z > 0: concrete shading scales from barely visible to bold+dark
    intensity = min(z, max_z) / max_z  # 0..1
    weight = 400 + round(intensity * 500)  # 400..900
    alpha = 0.04 + intensity * 0.36  # 0.04..0.40
    css = (f"font-weight:{weight}; "
           f"background:rgba(0,0,0,{alpha:.2f}); "
           f"border-radius:2px")
    cls = "concrete" if z > 1.0 else "neither"
    return css, cls


def _render_body(txt, col="Abs-Conc.Median.median", inline_styles=True, preserve_newlines=False):
    """Render passage text to styled HTML fragment (no wrapper).

    Paragraph breaks (blank lines) become indented new paragraphs
    rather than double line breaks.
    """
    scores = get_norm_dict(col)
    spelling_d = get_spelling_modernizer()

    # Split into paragraphs on blank lines, render each
    paragraphs = _split_paragraphs(txt, preserve_newlines=preserve_newlines)
    rendered = []
    for i, para in enumerate(paragraphs):
        body = _render_paragraph(para, scores, spelling_d, inline_styles=inline_styles)
        if i == 0:
            rendered.append(f'<p class="psg-para psg-first">{body}</p>')
        else:
            rendered.append(f'<p class="psg-para">{body}</p>')
    return "\n".join(rendered)


def _split_paragraphs(txt, preserve_newlines=False):
    """Split text into paragraphs on blank lines."""
    import re
    # Split on one or more blank lines (two+ consecutive newlines)
    paras = re.split(r'\n\s*\n', txt.strip())
    if preserve_newlines:
        # Keep single newlines as <br> (useful for verse)
        return [re.sub(r'\n', '<br>\n', p).strip() for p in paras if p.strip()]
    # Collapse remaining single newlines within paragraphs to spaces
    return [re.sub(r'\n', ' ', p).strip() for p in paras if p.strip()]


def _render_paragraph(txt, scores, spelling_d, inline_styles=True):
    """Render a single paragraph to styled HTML.

    Each scored word gets:
      - class="w {abstract|concrete|neither|unscored}"
      - data-z="{z:.2f}" (or "" if unscored)
      - Inline styles depending on mode

    Parameters
    ----------
    inline_styles : bool or str
        True  = grayscale inline CSS (for print/export)
        False = no inline styles (classes + data-z only)
        "color" = blue↔orange continuous color scale (for web)
    """
    tokens = tokenize_agnostic(txt)

    parts = []
    for tok in tokens:
        tok_lower = tok.lower()
        escaped = html_mod.escape(tok)

        if not tok or not tok[0].isalpha():
            parts.append((escaped, True))
            continue

        s, _ = _modernize_score(tok_lower, scores, spelling_d)
        z = s if s is not None else np.nan
        z_attr = f'{z:.2f}' if np.isfinite(z) else ''

        if inline_styles == "color":
            css = _word_color_style(z)
        elif inline_styles:
            css, _ = _word_style(z)
        else:
            css = None

        cls = _word_style(z)[1]  # always need the class name

        if css:
            parts.append((f'<span class="w {cls}" data-z="{z_attr}" style="{css}">{escaped}</span>', False))
        else:
            parts.append((f'<span class="w {cls}" data-z="{z_attr}">{escaped}</span>', False))

    chunks = []
    prev_is_hyphen = False
    for i, (html_str, is_punct) in enumerate(parts):
        if i > 0 and not is_punct and not prev_is_hyphen:
            chunks.append(" ")
        chunks.append(html_str)
        prev_is_hyphen = is_punct and html_str.strip() in ("-", "\u2013", "\u2014")
    return "".join(chunks)


def _word_color_style(z, max_z=3.0):
    """Return inline CSS for color mode (blue↔orange background scale).

    Background color encodes the z-score on a continuous scale:
    - z < 0 (abstract): blue background, more saturated/opaque with |z|
    - z > 0 (concrete): orange background, more saturated/opaque with z
    - z ≈ 0: very faint or no background
    - NaN: no styling
    """
    if np.isnan(z):
        return None

    intensity = min(abs(z), max_z) / max_z  # 0..1
    alpha = 0.08 + intensity * 0.42         # 0.08..0.50

    if z <= 0:
        # Abstract: blue background
        return (f"background:hsla(220,70%,55%,{alpha:.2f}); "
                f"border-radius:2px")

    # Concrete: orange background
    return (f"background:hsla(25,85%,55%,{alpha:.2f}); "
            f"border-radius:2px")


def render_passage_body(txt, col="Abs-Conc.Median.median", mode="color"):
    """Render passage text to an HTML fragment with data-z attributes.

    Parameters
    ----------
    mode : str
        "color" — blue↔orange continuous color scale (for web), preserves newlines as <br>
        "print" — grayscale outline+shading (for inline print preview)

    Returns a <div class="passage"> containing word spans with
    class, data-z, and mode-appropriate inline styles.
    """
    style_mode = "color" if mode == "color" else True
    preserve = mode == "color"
    body = _render_body(txt, col=col, inline_styles=style_mode, preserve_newlines=preserve)
    return f'<div class="passage">\n{body}\n</div>'


def render_passage_html(txt, col="Abs-Conc.Median.median",
                        title="", show_title=True, show_legend=True, font_size=14,
                        line_height=2.2, max_width=700):
    """Render a passage as an HTML string with per-word styling.

    Every scored word is styled on a continuous scale:
    - z < 0 (abstract): bordered rectangle, thicker/darker with |z|
    - z > 0 (concrete): gray background shading, darker/bolder with z
    - Unrecognized words: plain text (no styling)

    Parameters
    ----------
    txt : str
        The passage text.
    col : str
        Norm column to use for scoring.
    title : str
        Optional title/header above the passage.
    show_legend : bool
        Whether to include a legend explaining the styling.
    font_size : int
        Base font size in pixels.
    line_height : float
        Line height multiplier (needs room for borders).
    max_width : int
        Max width of the passage block in pixels.

    Returns
    -------
    str
        Complete HTML document string.
    """
    body = _render_body(txt, col=col)

    legend_html = ""
    if show_legend:
        legend_html = f"""
        <div style="margin-bottom:12px; font-size:{font_size - 2}px; color:#555;">
            <span style="outline:3px solid rgba(0,0,0,0.60); outline-offset:0px; border-radius:2px; padding:0 3px; margin-right:12px;">abstract</span>
            <span style="outline:1px solid rgba(0,0,0,0.15); outline-offset:0px; border-radius:2px; padding:0 3px; margin-right:12px;">slightly abstract</span>
            <span style="font-weight:500; background:rgba(0,0,0,0.08); border-radius:2px; padding:0 3px; margin-right:12px;">slightly concrete</span>
            <span style="font-weight:800; background:rgba(0,0,0,0.30); border-radius:2px; padding:0 3px; margin-right:12px;">concrete</span>
            <span style="color:#888; margin-left:4px;">plain = unscored</span>
        </div>"""

    title_html = ""
    if title and show_title:
        title_html = f'<h3 style="margin:0 0 8px 0; font-family:serif;">{html_mod.escape(title)}</h3>'

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<style>
body {{
    font-family: "Georgia", "Times New Roman", serif;
    font-size: {font_size}px;
    line-height: {line_height};
    max-width: {max_width}px;
    margin: 20px auto;
    color: #111;
}}
.w {{
    display: inline;
}}
.psg-para {{
    margin: 0;
    text-indent: 2em;
}}
.psg-first {{
    text-indent: 0;
}}
</style>
</head>
<body>
{title_html}
{legend_html}
<div class="passage">
{body}
</div>
</body>
</html>"""
    return html


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def display_passage(txt, col="Abs-Conc.Median.median", title="", **kwargs):
    """Display a styled passage in a Jupyter notebook.

    Renders the passage inline using IPython.display.HTML.
    """
    from IPython.display import HTML, display
    html = render_passage_html(txt, col=col, title=title, **kwargs)
    # Strip the outer html/head/body for inline display; keep style + content
    display(HTML(html))


def save_passage_html(txt, path, col="Abs-Conc.Median.median", title="", **kwargs):
    """Save a styled passage as a standalone HTML file."""
    html = render_passage_html(txt, col=col, title=title, **kwargs)
    with open(path, "w") as f:
        f.write(html)
    return path


async def _playwright_render_async(html, path, ext, width, scale):
    """Render HTML to image/PDF via Playwright async API (for Jupyter)."""
    from playwright.async_api import async_playwright
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(
            viewport={"width": width, "height": 100},
            device_scale_factor=scale,
        )
        await page.set_content(html)
        await page.wait_for_load_state("networkidle")
        height = await page.evaluate("document.body.scrollHeight")
        await page.set_viewport_size({"width": width, "height": height + 40})
        if ext == ".pdf":
            await page.pdf(path=path, width=f"{width}px")
        else:
            await page.screenshot(path=path, full_page=True)
        await browser.close()


def _playwright_render(html, path, ext, width, scale):
    """Render HTML to image/PDF via Playwright, using async API inside Jupyter."""
    try:
        asyncio.get_running_loop()
        in_loop = True
    except RuntimeError:
        in_loop = False

    if in_loop:
        import nest_asyncio
        nest_asyncio.apply()
        asyncio.get_event_loop().run_until_complete(
            _playwright_render_async(html, path, ext, width, scale)
        )
    else:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(
                viewport={"width": width, "height": 100},
                device_scale_factor=scale,
            )
            page.set_content(html)
            page.wait_for_load_state("networkidle")
            height = page.evaluate("document.body.scrollHeight")
            page.set_viewport_size({"width": width, "height": height + 40})
            if ext == ".pdf":
                page.pdf(path=path, width=f"{width}px")
            else:
                page.screenshot(path=path, full_page=True)
            browser.close()


def save_passage_image(txt, path, col="Abs-Conc.Median.median", title="",
                       width=800, dpi=300, **kwargs):
    """Save a styled passage as a PNG image.

    Parameters
    ----------
    txt : str
        The passage text.
    path : str
        Output file path (.png or .pdf).
    col : str
        Norm column to use for scoring.
    title : str
        Optional title/header above the passage.
    width : int
        Layout width in CSS pixels.
    dpi : int
        Output resolution. 300 = print quality (3x scale factor).
        Default 300.
    **kwargs
        Passed to render_passage_html.

    Requires either playwright or wkhtmltoimage to be installed.
    Tries playwright first, falls back to wkhtmltoimage.
    """
    html = render_passage_html(txt, col=col, title=title, **kwargs)

    ext = os.path.splitext(path)[1].lower()
    if ext not in (".png", ".pdf"):
        raise ValueError(f"Unsupported format {ext}; use .png or .pdf")

    scale = max(1, round(dpi / 96))  # 96 CSS px per inch baseline

    try:
        _playwright_render(html, path, ext, width, scale)
        return path
    except ImportError:
        pass

    # Fallback: wkhtmltoimage
    import subprocess
    html_tmp = path + ".tmp.html"
    with open(html_tmp, "w") as f:
        f.write(html)
    try:
        cmd = ["wkhtmltoimage", "--width", str(width), "--quality", "100",
               html_tmp, path]
        subprocess.run(cmd, check=True, capture_output=True)
    finally:
        if os.path.exists(html_tmp):
            os.remove(html_tmp)
    return path


# ---------------------------------------------------------------------------
# Comparison rendering
# ---------------------------------------------------------------------------

def render_comparison_html(passages, col="Abs-Conc.Median.median",
                           show_legend=True, show_titles=True,
                           font_size=13, line_height=2.2, col_width=380):
    """Render multiple passages side-by-side for comparison.

    Parameters
    ----------
    passages : list of dict
        Each dict has 'text' and optionally 'title'.
        Example: [{"text": "...", "title": "Austen, *Emma* (1815)"}]
    col : str
        Norm column.
    show_legend : bool
        Whether to show the abstract/concrete legend. Default True.
    show_titles : bool
        Whether to show per-passage titles. Default True.
    col_width : int
        Width of each passage column in pixels.

    Returns
    -------
    str
        HTML document with passages in a table row.
    """
    cells = []
    for psg in passages:
        txt = psg["text"]
        title = psg.get("title", "")
        body = _render_body(txt, col=col)
        title_html = f"<h4>{title}</h4>" if (title and show_titles) else ""
        cells.append(f"<td>{title_html}<div class='passage'>{body}</div></td>")

    table = "<table><tr>" + "".join(cells) + "</tr></table>"

    legend_html = ""
    if show_legend:
        legend_html = f"""<div class="legend">
    <span style="outline:3px solid rgba(0,0,0,0.60); outline-offset:0px; border-radius:2px; padding:0 3px;">abstract</span>&ensp;
    <span style="outline:1px solid rgba(0,0,0,0.15); outline-offset:0px; border-radius:2px; padding:0 3px;">slightly abstract</span>&ensp;
    <span style="font-weight:500; background:rgba(0,0,0,0.08); border-radius:2px; padding:0 3px;">slightly concrete</span>&ensp;
    <span style="font-weight:800; background:rgba(0,0,0,0.30); border-radius:2px; padding:0 3px;">concrete</span>&ensp;
    <span style="color:#888;">plain = unscored</span>
</div>"""

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<style>
body {{
    font-family: "Georgia", "Times New Roman", serif;
    font-size: {font_size}px;
    line-height: {line_height};
    color: #111;
    margin: 20px;
}}
table {{ border-collapse: collapse; }}
td {{
    width: {col_width}px;
    vertical-align: top;
    padding: 10px 20px;
    border-right: 1px solid #ccc;
}}
td:last-child {{ border-right: none; }}
h4 {{ margin: 0 0 8px 0; font-family: serif; }}
.psg-para {{ margin: 0; text-indent: 2em; }}
.psg-first {{ text-indent: 0; }}
.legend {{
    margin-bottom: 12px; font-size: {font_size - 2}px; color: #555;
}}
</style>
</head>
<body>
{legend_html}
{table}
</body>
</html>"""
    return html


def display_comparison(passages, **kwargs):
    """Display side-by-side passages in a Jupyter notebook."""
    from IPython.display import HTML, display
    html = render_comparison_html(passages, **kwargs)
    display(HTML(html))


def save_comparison_image(passages, path, width=900, dpi=300, **kwargs):
    """Save a side-by-side passage comparison as a PNG or PDF.

    Parameters
    ----------
    passages : list of dict
        Each dict has 'text' and optionally 'title'.
    path : str
        Output file path (.png or .pdf).
    width : int
        Layout width in CSS pixels.
    dpi : int
        Output resolution. Default 300 (print quality).
    **kwargs
        Passed to render_comparison_html.
    """
    html = render_comparison_html(passages, **kwargs)

    ext = os.path.splitext(path)[1].lower()
    if ext not in (".png", ".pdf"):
        raise ValueError(f"Unsupported format {ext}; use .png or .pdf")

    scale = max(1, round(dpi / 96))

    _playwright_render(html, path, ext, width, scale)
    return path
