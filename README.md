# Abstraction: A Literary History

Code and data for measuring abstract and concrete language across the history of English-language literature.

## The central finding

**Abstract language rose in English literature from the 1500s to a peak around 1700-1780, then fell steeply through the 20th century.** This pattern — a plateau followed by a cliff — appears across fiction, poetry, and periodical writing, confirmed across 34 corpora comprising ~1.27 million texts.

![The arc of abstraction across three genres (Fiction, Poetry, Periodical), 1600-2000. LOESS smoothing on corpus-adjusted decade averages, inverted so abstractness is up.](figures/arc-by-3genres-loess.inverted.v4.png)

*The arc of abstraction across three genres. LOESS (span=0.3) on corpus-adjusted decade averages. Fiction shows a long plateau of peak abstractness ~1720-1780, then a steep sustained decline. Poetry peaks and breaks earliest (~1705/1720). Periodicals peak later (~1812).*

### Cross-genre results (piecewise regression with corpus fixed effects)

| Genre | Texts | Breakpoint | Rise slope | Fall slope | R² (scores) | R² (abstract %) | R² (concrete %) |
|---|---:|---:|---|---|---:|---:|---:|
| Fiction | 161,148 | 1760 | +0.0013/dec *** | -0.0029/dec *** | 0.882 | 0.820 | 0.900 |
| Poetry | 244,790 | 1700 | +0.0021/dec *** | -0.0019/dec *** | 0.881 | 0.794 | 0.904 |
| Periodical | 157,561 | 1830 | +0.0013/dec *** | -0.0018/dec *** | 0.870 | 0.849 | 0.787 |

#### Fiction (n = 161,148)

**Scores** (continuous weighted-mean concreteness, inverted so abstractness is up):
- 1600s: 0.2046 → 1750s: 0.3743 → 1990s: -0.2490
- Rise: +0.67 SD | Fall: +2.46 SD
- Breakpoint: 1760 | R² = 0.882
- Rise slope: +0.0013/decade (p = 7.5e-11) ***
- Fall slope: -0.0029/decade (p = 3.6e-83) ***

**Word proportions** (abstract: z ≤ -1.0, concrete: z > 1.0):

| Phase | Abstract | Concrete | Abs/Conc ratio |
|---|---|---|---|
| 1600s (start) | 18.8% | 9.4% | 2.0:1 |
| 1750s (peak) | 26.4% | 7.9% | 3.4:1 |
| 1990s (end) | 10.4% | 24.2% | 0.4:1 (1:2.3 conc/abs) |
| **Rise** (1600s→1750s) | 1.4x | 1.2x decline | 1.7x |
| **Fall** (1750s→1990s) | 2.5x decline | 3.1x increase | 7.8x decline |
| **Net** (1600s→1990s) | 1.8x decline | 2.6x increase | 4.6x decline |

R² abstract = 0.820, R² concrete = 0.900

#### Poetry (n = 244,790)

**Scores** (continuous weighted-mean concreteness, inverted so abstractness is up):
- 1610s: -0.0418 → 1690s: 0.1606 → 1980s: -0.4291
- Rise: +0.60 SD | Fall: +1.74 SD
- Breakpoint: 1700 | R² = 0.881
- Rise slope: +0.0021/decade (p = 6.9e-12) ***
- Fall slope: -0.0019/decade (p = 6.3e-52) ***

**Word proportions** (abstract: z ≤ -1.0, concrete: z > 1.0):

| Phase | Abstract | Concrete | Abs/Conc ratio |
|---|---|---|---|
| 1610s (start) | 10.3% | 12.4% | 0.8:1 (1:1.2 conc/abs) |
| 1690s (peak) | 16.3% | 9.8% | 1.7:1 |
| 1980s (end) | 7.7% | 31.3% | 0.2:1 (1:4.0 conc/abs) |
| **Rise** (1610s→1690s) | 1.6x | 1.3x decline | 2.0x |
| **Fall** (1690s→1980s) | 2.1x decline | 3.2x increase | 6.8x decline |
| **Net** (1610s→1980s) | 1.3x decline | 2.5x increase | 3.4x decline |

R² abstract = 0.794, R² concrete = 0.904

#### Periodical (n = 157,561)

**Scores** (continuous weighted-mean concreteness, inverted so abstractness is up):
- 1640s: -0.1142 → 1840s: 0.1735 → 2000s: 0.0312
- Rise: +0.82 SD | Fall: +0.41 SD
- Breakpoint: 1830 | R² = 0.870
- Rise slope: +0.0013/decade (p = 8.6e-06) ***
- Fall slope: -0.0018/decade (p = 3.5e-11) ***

**Word proportions** (abstract: z ≤ -1.0, concrete: z > 1.0):

| Phase | Abstract | Concrete | Abs/Conc ratio |
|---|---|---|---|
| 1640s (start) | 11.8% | 19.4% | 0.6:1 (1:1.7 conc/abs) |
| 1840s (peak) | 19.7% | 10.7% | 1.8:1 |
| 2000s (end) | 16.4% | 16.0% | 1.0:1 |
| **Rise** (1640s→1840s) | 1.7x | 1.8x decline | 3.0x |
| **Fall** (1840s→2000s) | 1.2x decline | 1.5x increase | 1.8x decline |
| **Net** (1640s→2000s) | 1.4x increase | 1.2x decline | 1.7x increase |

R² abstract = 0.849, R² concrete = 0.787

#### Prose summary

**Fiction** (n = 161,148): Abstractness rises from the 1600s to a peak in the 1750s (+0.67 SD), then falls through the 1990s (+2.46 SD). At peak, fiction has 3.4:1 abstract-to-concrete words, up from 2.0:1 in the 1600s (1.7x). By the 1990s, the ratio inverts to 0.4:1 (1:2.3 conc/abs) — a 7.8x decline from peak. Piecewise breakpoint at 1760; rise slope = +0.0013/decade (p = 7.5e-11), fall slope = -0.0029/decade (p = 3.6e-83); R² = 0.882.

**Poetry** (n = 244,790): Abstractness rises from the 1610s to a peak in the 1690s (+0.60 SD), then falls through the 1980s (+1.74 SD). At peak, poetry has 1.7:1 abstract-to-concrete words, up from 0.8:1 (1:1.2 conc/abs) in the 1610s. By the 1980s, the ratio inverts to 0.2:1 (1:4.0 conc/abs) — a 6.8x decline from peak. Piecewise breakpoint at 1700; rise slope = +0.0021/decade (p = 6.9e-12), fall slope = -0.0019/decade (p = 6.3e-52); R² = 0.881.

**Periodical** (n = 157,561): Abstractness rises from the 1640s to a peak in the 1840s (+0.82 SD), then falls through the 2000s (+0.41 SD). At peak, periodical has 1.8:1 abstract-to-concrete words, up from 0.6:1 (1:1.7 conc/abs) in the 1640s. By the 2000s it falls to 1.0:1. Piecewise breakpoint at 1830; rise slope = +0.0013/decade (p = 8.6e-06), fall slope = -0.0018/decade (p = 3.5e-11); R² = 0.870.

*(Scores: continuous weighted-mean concreteness, inverted. Proportions: frequency-weighted, abstract z ≤ -1.0, concrete z > 1.0. All regressions include corpus fixed effects.)*

<details>
<summary>Robustness: raw vs modernized spelling comparison</summary>

Spelling modernization (mapping historical forms like "vertue" to "virtue" via MorphAdorner) has minimal effect on the overall findings. Breakpoints, peak decades, and R² values are stable across both approaches. The main difference is in early-period baseline scores, where modernization slightly raises abstractness (more words match the norm vocabulary). Results below use raw spelling as the primary analysis.

| | Fiction | Poetry | Periodical |
|---|---|---|---|
| Breakpoint (raw) | 1760 | 1700 | 1830 |
| Breakpoint (mod) | 1760 | 1760 | 1830 |
| R² scores (raw) | 0.882 | 0.881 | 0.870 |
| R² scores (mod) | 0.884 | 0.887 | 0.871 |
| Abs/Conc rise (raw) | 2.0→3.4 (1.7x) | 0.8→1.7 (2.0x) | 0.6→1.8 (3.0x) |
| Abs/Conc rise (mod) | 2.3→3.3 (1.4x) | 1.3→1.7 (1.4x) | 0.6→1.8 (3.1x) |

The only notable divergence is poetry's score-based breakpoint, which shifts from 1700 (raw) to 1760 (modernized). This occurs because modernization slightly flattens the early rise (+0.60 SD raw → +0.36 SD modernized), making the pre-break slope non-significant (p = 0.78). The count-based breakpoint remains at 1690 in both cases.

Generated with `abstraction report-full --compare`.

</details>

## What drove the shift?

Three word-level analyses identify which words are responsible.

### 1. Contribution decomposition

Which words' frequency changes mechanically drove the aggregate concreteness shift? Comparing 1700-1780 (abstract peak) to 1850-1950:

**Abstract words that declined** (removing abstraction):
| Word | Z-score | Freq change | Contribution |
|---|---|---|---|
| virtue | -1.59 | -90% | +2.38e-03 |
| reason | -1.57 | -62% | +2.40e-03 |
| passion | -1.27 | -77% | +2.19e-03 |
| favour | -1.48 | -83% | +1.89e-03 |
| opinion | -1.77 | -68% | +1.71e-03 |

**Concrete words that rose** (adding concreteness):
| Word | Z-score | Freq change | Contribution |
|---|---|---|---|
| white | +2.07 | +462% | +3.25e-03 |
| black | +1.62 | +265% | +1.80e-03 |
| hair | +1.86 | +276% | +1.52e-03 |
| blue | +2.05 | +681% | +1.32e-03 |
| window | +1.57 | +216% | +1.17e-03 |

**Counter-trend** (abstract words that *rose* despite the overall shift): *feeling* (+1064%), *fact* (+600%), *simply* (+2987%) — a new psychological/epistemic vocabulary replaced the old moral one.

Net: concretizing forces (+0.49) overwhelm abstracting (-0.10). All three word categories (Abstract, Concrete, Neither) net concretized.

### 2. Frequency correlation

Which words' frequency trajectories most closely track the overall concreteness trend? (Cosine similarity.)

**Tracking abstractness** (r ~ -0.93): *presumption, disposition, opinion, happiness, passion, favour, neglect, conceive, inclination* — the moral and psychological vocabulary of 18th-century fiction. 35 of the top 50 are classified Abstract.

**Tracking concreteness** (r ~ +0.20): *garage, bathroom, grabbed, cigarettes, elevator, sidewalk, kids* — modern material-world vocabulary. These appear only in 20th-century fiction.

The abstracting correlations are ~5x stronger than the concretizing ones: the decline of abstract vocabulary is more coherent than the rise of concrete vocabulary.

### 3. Semantic drift (vector norms)

Which words changed what they *mean* — independent of how often they appear? Using Word2Vec models trained on period-specific corpora (C17 vs C19):

- 60.5% of words drifted toward abstraction; only 39.5% toward concreteness
- Mean shift = -0.14 z-score (toward abstraction)
- *hollowness*: concrete in C17 (+1.67) → abstract in C19 (-1.56)
- *callous*: concrete (+1.58) → abstract (-1.23) — literal to metaphorical

Words themselves drifted abstract even as fiction chose more concrete words.

## Passage visualization

The package renders passages with per-word grayscale styling for print:
- **Abstract words**: bordered (thicker border = more abstract)
- **Concrete words**: bold with gray background (darker = more concrete)

![Passage comparison: Haywood (1724) vs Austen (1813)](figures/comparison_test.png)

*Left: Eliza Haywood, The Masqueraders (c. 1724) — dense with bordered abstract words (Scruples, Arguments, Virtue, Fear, Desire). Right: Jane Austen, Pride and Prejudice (1813) — mostly plain text with occasional concrete shading.*

![Single passage: Haywood with per-word styling](figures/passage_haywood.png)

```python
from abstraction import display_passage, display_comparison, save_passage_image

# Render in Jupyter
display_passage(text, title="Haywood, The Masqueraders (1724)")

# Side-by-side comparison
display_comparison([
    {"text": haywood_text, "title": "Haywood (1724)"},
    {"text": austen_text, "title": "Austen (1813)"},
])

# Save as 300 DPI PNG for print
save_passage_image(text, "output.png", dpi=300)
```

## Setup

Requires Python 3.9+. Recommended: Python 3.10 via pyenv.

```bash
cd abstraction
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For LLM features (text generation with caching):
```bash
pip install -e ".[llm]"
```

For passage image export:
```bash
pip install playwright
playwright install chromium
```

## Data layout

The package expects two data directories:

### Corpora (`~/lltk_data/corpora/`)

Each corpus is a directory containing metadata and either plain-text files or pre-computed word frequencies:

```
~/lltk_data/corpora/canon_fiction/
    metadata.csv          # must have 'id' column; typically also: author, title, year
    txt/                  # raw text files
        text_id_1.txt
    freqs/                # pre-computed word frequency JSONs
        text_id_1.json    # {"word": count, ...}
```

The path is configurable via `abstraction.config.PATH_CORPORA`. Corpus names use CamelCase in code (`CanonFiction`) and snake_case on disk (`canon_fiction`).

### Generated data (`./data/`)

Intermediate and output data lives in `data/` (not tracked in git):

```
data/
    fields/                # word norms, semantic field definitions
        sources/           # raw downloaded norm files
        data.allnorms.pkl  # cached combined norms DataFrame
    models/                # trained Word2Vec models by corpus/period/run
    counts/v2-raw/         # per-corpus JSONL bin counts (raw spelling, default)
    counts/v2/             # per-corpus JSONL bin counts (modernized spelling)
    scores/v8-raw/         # per-corpus CSVs of text-level norm scores (raw spelling, default)
    scores/v8/             # per-corpus CSVs of text-level norm scores (modernized spelling)
    psgs/                  # generated passage markdown files
    stash/                 # LLM response cache
    figures/               # saved plots
```

## Package modules

| Module | Purpose |
|--------|---------|
| `config.py` | Path constants and hyperparameters (`ZCUT`, `COUNT_WINDOW_LEN`, etc.) |
| `corpus.py` | `Corpus` class, `load_corpus()`, `pmap()`/`pmap_iter()` for parallelism |
| `tokenize.py` | Tokenization, spelling modernization, stopword filtering |
| `norms.py` | Load/combine psycholinguistic norms, classify words, z-score, semantic fields |
| `counting.py` | Sliding-window abstract/concrete word counting |
| `scoring.py` | Text/passage scoring, corpus-level frequency scoring, passage sampling |
| `models.py` | Word2Vec training, vector field computation, historical vector norms |
| `analysis.py` | Score loading, corpus adjustment, piecewise/polynomial arc fitting |
| `words.py` | Word-trend analysis: frequency correlation, contribution decomposition, score shifts |
| `passages.py` | Per-word HTML/PNG passage visualization for print |
| `plotting.py` | plotnine-based visualization (norm plots, arc plots, LOESS) |
| `llm.py` | LLM text generation with litellm + hashstash caching |
| `cli.py` | CLI entrypoint (`abstraction score-corpora`, `score-corpus`, etc.) |
| `utils.py` | DataFrame I/O, streaming CSV writer, z-scoring, HTML cleaning |

## CLI

```bash
# Score all corpora with freqs/ directories
abstraction score-corpora [--force] [--modernize]

# Score a single corpus
abstraction score-corpus canon_fiction [--force] [--modernize]

# Count abstract/concrete words per corpus
abstraction count-corpora [--norms Median] [--force] [--modernize]
abstraction count-corpus canon_fiction [--force] [--modernize]

# Report arc statistics
abstraction report-arc [--modernize]
abstraction report-arc-counts [--abs-cutoff -1.0] [--conc-cutoff 1.0] [--modernize]

# Combined report (scores + counts + prose)
abstraction report-full [-o report.md] [--csv results.csv] [--modernize]
abstraction report-full --compare [-o comparison.md]   # raw vs modernized side-by-side
```

## Notebooks

| Notebook | Description |
|----------|-------------|
| `01-word-norms` | Load and explore psycholinguistic norms, classify words, visualize across sources |
| `02-counting` | Score passages, sliding-window counting, compare abstract vs. concrete text |
| `03-fiction-trends` | Plot abstraction trends across fiction history by genre |
| `04-passages` | Find extreme passages, stratified sampling, per-book passage generation |
| `05-models` | Inspect Word2Vec models, explore word neighborhoods, vector norms pipeline |
| `06-corpus` | Explore the Corpus class, list available corpora, read and tokenize texts |
| `07-arc-analysis` | Cross-corpus arc fitting, LOESS vs parametric, genre comparisons |

## Tests

```bash
source .venv/bin/activate
python -m pytest tests/ -v
```

| Test file | Coverage |
|-----------|----------|
| `test_corpus.py` | camel_to_snake, pmap |
| `test_tokenize.py` | tokenize, tokenize_agnostic, strip_punct |
| `test_norms.py` | classify_word, get_contrasts, format_norms_as_long |
| `test_utils.py` | zfy, read/save_df, get_avgs_df, get_slices, cleanhtml, parse_json_str |
| `test_scoring.py` | score_freqs, score_words, corpus scoring pipeline |
| `test_plotting.py` | _compress_year |
| `test_passages.py` | _word_style scaling, render_body, HTML rendering, flags, PNG export |
| `test_words.py` | frequency correlation, contribution decomposition, score shifts |
| `test_integration.py` | End-to-end pipeline (requires local data) |

## Key concepts

**Word norms** are published psycholinguistic ratings of how concrete or abstract a word is:

| Abbreviation | Source | What it measures |
|-------------|--------|-----------------|
| `PAV-Conc` | Paivio et al. (1968) | Concreteness |
| `PAV-Imag` | Paivio et al. (1968) | Imagery |
| `MRC-Conc` | MRC Psycholinguistic Database (1987) | Concreteness |
| `MRC-Imag` | MRC (1987) | Imagery |
| `MT-Conc` | Brysbaert et al. (2014) | Concreteness (Mechanical Turk) |
| `LSN-Imag` | Lancaster Sensorimotor Norms (2017) | Visual strength |
| `LSN-Hapt` | Lancaster (2017) | Haptic strength |

**Semantic fields**: words classified by z-score threshold. With `ZCUT = 1.0`: z >= 1.0 = **Concrete**, z <= -1.0 = **Abstract**, rest = **Neither**.

**Vector norms**: historical concreteness scores from period-specific Word2Vec models (C16, C17, C18, C19, C20). Measures how concrete a word was *as used in that era*.

**Scoring** (continuous): weighted-mean concreteness per text across 56 norm columns. Used for trend fitting and regression.

**Counting** (proportions): sliding 100-word window, counting abstract/concrete/neither words. Used for human-readable ratios ("26% of words were abstract at the 1750s peak vs 10% in the 1990s").
