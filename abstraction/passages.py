"""
Passage visualization with per-word abstract/concrete styling.

Grayscale-friendly design for print:
  - Abstract words: bordered (thicker border = more abstract)
  - Concrete words: bold (heavier weight = more concrete)
  - Neutral words: plain text
"""

import html as html_mod
import os
import textwrap

import numpy as np

from .scoring import score_words, get_norm_dict
from .tokenize import tokenize_agnostic


# ---------------------------------------------------------------------------
# Per-word HTML rendering
# ---------------------------------------------------------------------------

def _word_style(z, abs_cutoff=-1.0, conc_cutoff=1.0, max_z=3.0):
    """Return inline CSS for a word given its z-score.

    Abstract (z <= abs_cutoff): border whose width scales with |z|.
    Concrete (z >= conc_cutoff): font-weight that scales with z.
    Neither: no special styling.
    """
    if np.isnan(z):
        return None, "neither"

    if z <= abs_cutoff:
        # More abstract → thicker border.  Map z from [abs_cutoff .. -max_z] to [1px .. 4px]
        intensity = min(abs(z), max_z) / max_z  # 0..1
        border_px = 1 + round(intensity * 3)  # 1..4
        css = f"border:{border_px}px solid #555; border-radius:2px; padding:0 2px"
        return css, "abstract"

    if z >= conc_cutoff:
        # More concrete → bolder.  Map z from [conc_cutoff .. max_z] to [600 .. 900]
        intensity = min(z, max_z) / max_z  # 0..1
        weight = 600 + round(intensity * 300)  # 600..900
        css = f"font-weight:{weight}"
        return css, "concrete"

    return None, "neither"


def render_passage_html(txt, col="Abs-Conc.Median.median",
                        abs_cutoff=-1.0, conc_cutoff=1.0,
                        title="", show_legend=True, font_size=14,
                        line_height=2.2, max_width=700):
    """Render a passage as an HTML string with per-word styling.

    Parameters
    ----------
    txt : str
        The passage text.
    col : str
        Norm column to use for scoring.
    abs_cutoff : float
        Z-score threshold for abstract classification (words <= this).
    conc_cutoff : float
        Z-score threshold for concrete classification (words >= this).
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
    scores = get_norm_dict(col)
    tokens = tokenize_agnostic(txt)

    # Build styled tokens: list of (html_str, is_punct) pairs
    parts = []
    for tok in tokens:
        tok_lower = tok.lower()
        escaped = html_mod.escape(tok)

        # Punctuation / whitespace — pass through
        if not tok or not tok[0].isalpha():
            if tok == "\n":
                parts.append(("<br/>", True))
            else:
                parts.append((escaped, True))
            continue

        z = scores.get(tok_lower, np.nan)
        css, cls = _word_style(z, abs_cutoff, conc_cutoff)

        if css:
            parts.append((f'<span class="w {cls}" style="{css}">{escaped}</span>', False))
        else:
            parts.append((f'<span class="w {cls}">{escaped}</span>', False))

    # Join with spaces, but no space before punctuation
    chunks = []
    for i, (html_str, is_punct) in enumerate(parts):
        if i > 0 and not is_punct:
            chunks.append(" ")
        chunks.append(html_str)
    body = "".join(chunks)

    legend_html = ""
    if show_legend:
        legend_html = f"""
        <div style="margin-bottom:12px; font-size:{font_size - 2}px; color:#555;">
            <span style="border:2px solid #555; border-radius:2px; padding:0 3px; margin-right:8px;">abstract</span>
            <span style="font-weight:800; margin-right:8px;">concrete</span>
            <span style="color:#888;">plain = unscored or neither</span>
        </div>"""

    title_html = ""
    if title:
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
.abstract {{
    display: inline-block;
    margin: 1px 0;
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


def save_passage_image(txt, path, col="Abs-Conc.Median.median", title="",
                       width=800, **kwargs):
    """Save a styled passage as a PNG image.

    Requires either playwright or wkhtmltoimage to be installed.
    Tries playwright first, falls back to wkhtmltoimage.
    """
    html = render_passage_html(txt, col=col, title=title, **kwargs)

    ext = os.path.splitext(path)[1].lower()
    if ext not in (".png", ".pdf"):
        raise ValueError(f"Unsupported format {ext}; use .png or .pdf")

    # Try playwright (headless Chromium)
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": width, "height": 100})
            page.set_content(html)
            page.wait_for_load_state("networkidle")
            # Auto-height: measure content
            height = page.evaluate("document.body.scrollHeight")
            page.set_viewport_size({"width": width, "height": height + 40})
            if ext == ".pdf":
                page.pdf(path=path, width=f"{width}px")
            else:
                page.screenshot(path=path, full_page=True)
            browser.close()
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
                           abs_cutoff=-1.0, conc_cutoff=1.0,
                           font_size=13, line_height=2.2, col_width=380):
    """Render multiple passages side-by-side for comparison.

    Parameters
    ----------
    passages : list of dict
        Each dict has 'text' and optionally 'title'.
        Example: [{"text": "...", "title": "Austen, *Emma* (1815)"}]
    col : str
        Norm column.
    col_width : int
        Width of each passage column in pixels.

    Returns
    -------
    str
        HTML document with passages in a table row.
    """
    scores = get_norm_dict(col)

    cells = []
    for psg in passages:
        txt = psg["text"]
        title = psg.get("title", "")
        tokens = tokenize_agnostic(txt)

        parts = []
        for tok in tokens:
            tok_lower = tok.lower()
            escaped = html_mod.escape(tok)
            if not tok or not tok[0].isalpha():
                parts.append(("<br/>" if tok == "\n" else escaped, True))
                continue
            z = scores.get(tok_lower, np.nan)
            css, cls = _word_style(z, abs_cutoff, conc_cutoff)
            if css:
                parts.append((f'<span class="w {cls}" style="{css}">{escaped}</span>', False))
            else:
                parts.append((f'<span class="w {cls}">{escaped}</span>', False))

        chunks = []
        for i, (html_str, is_punct) in enumerate(parts):
            if i > 0 and not is_punct:
                chunks.append(" ")
            chunks.append(html_str)
        body = "".join(chunks)

        title_html = f"<h4>{title}</h4>" if title else ""
        cells.append(f"<td>{title_html}<div class='passage'>{body}</div></td>")

    table = "<table><tr>" + "".join(cells) + "</tr></table>"

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
.abstract {{ display: inline-block; margin: 1px 0; }}
.legend {{
    margin-bottom: 12px; font-size: {font_size - 2}px; color: #555;
}}
</style>
</head>
<body>
<div class="legend">
    <span style="border:2px solid #555; border-radius:2px; padding:0 3px;">abstract</span>&ensp;
    <span style="font-weight:800;">concrete</span>&ensp;
    <span style="color:#888;">plain = unscored / neither</span>
</div>
{table}
</body>
</html>"""
    return html


def display_comparison(passages, **kwargs):
    """Display side-by-side passages in a Jupyter notebook."""
    from IPython.display import HTML, display
    html = render_comparison_html(passages, **kwargs)
    display(HTML(html))
