# Abstraction: A Literary History

Code and data for measuring abstract and concrete language across the history of English-language literature.

## The central finding

**Abstract language rose in English literature from the 1500s to a peak around 1700-1780, then fell steeply through the 20th century.** This pattern — a plateau followed by a cliff — appears across fiction, poetry, and periodical writing, confirmed across 34 corpora comprising ~1.27 million texts.

![The arc of abstraction across three genres (Fiction, Poetry, Periodical), 1600-2000. LOESS smoothing on corpus-adjusted decade averages, inverted so abstractness is up.](figures/arc-by-3genres-loess.inverted.v4.png)

*The arc of abstraction across three genres. LOESS (span=0.3) on corpus-adjusted decade averages. Fiction shows a long plateau of peak abstractness ~1720-1780, then a steep sustained decline. Poetry peaks and breaks earliest (~1705/1720). Periodicals peak later (~1812).*

### Cross-genre results (piecewise regression with corpus fixed effects)

| Genre | Texts | Peak | Break | Before break | After break | R² |
|---|---:|---:|---:|---|---|---:|
| Fiction | 45K | 1709 | 1760 | abstracting *** | concretizing *** | 0.84 |
| Poetry | 281K | 1705 | 1720 | abstracting *** | concretizing *** | 0.78 |
| Drama | 8K | 1750 | 1720 | abstracting *** | concretizing *** | 0.83 |
| Periodical | 192K | 1812 | 1840 | abstracting *** | concretizing *** | 0.93 |
| Sermon | 7K | 1766 | 1660 | abstracting *** | abstracting *** | 0.79 |
| Letters | 5K | 1742 | 1740 | abstracting * | concretizing * | 0.86 |
| Essay/Treatise | 4K | 1751 | 1770 | abstracting ** | concretizing ** | 0.47 |

Sermons never concretize. Nonfiction shows the inverse pattern (gets more abstract over time).

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
    counts/v2/             # per-corpus JSONL bin counts
    scores/v8/             # per-corpus CSVs of text-level norm scores
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
abstraction score-corpora [--force]

# Score a single corpus
abstraction score-corpus canon_fiction [--force]

# Count abstract/concrete words per corpus
abstraction count-corpora [--norms Median] [--force]
abstraction count-corpus canon_fiction [--force]

# Report arc statistics
abstraction report-arc
abstraction report-arc-counts [--abs-cutoff -1.0] [--conc-cutoff 1.0]
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
