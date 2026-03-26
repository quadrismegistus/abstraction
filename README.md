# Abstraction: A Literary History

Code and data for measuring abstract and concrete language across the history of English-language fiction.

## What this project does

This project asks a simple question with a complex answer: **how has the balance between abstract and concrete language changed across the history of fiction?**

It does this by:

1. **Loading psycholinguistic word norms** — published concreteness and imageability ratings from four major studies (Paivio 1968, MRC 1987, Brysbaert 2014, Lancaster Sensorimotor Norms 2017). These tell us, for tens of thousands of English words, how "concrete" (rock, elbow, face) or "abstract" (justice, virtue, anxiety) they are.

2. **Building historical word embeddings** — training Word2Vec models on period-specific corpora (EEBO-TCP for C16-C17, ECCO-TCP for C18, COHA for C19-C20) to measure how word concreteness has shifted over centuries. A word like "interest" or "passion" may have been more concrete in the 1600s than it is today.

3. **Counting across fiction** — sliding a 100-word window across every text in a large literary corpus (CanonFiction, ~1800 texts from antiquity to the present), counting how many words in each window are abstract, concrete, or neither.

4. **Plotting trends** — visualizing how abstraction varies by genre (Novel, Romance, Epic, Satire, etc.) and how it has changed over time, from Homer to the present.

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

## Data layout

The package expects two data directories:

### Corpora (`~/lltk_data/corpora/`)

Each corpus is a directory containing a metadata CSV and a folder of plain-text files:

```
~/lltk_data/corpora/canon_fiction/
    metadata.csv          # must have an 'id' column; typically also: author, title, year, major_genre
    txt/
        text_id_1.txt
        text_id_2.txt
        ...
```

The path is configurable via `abstraction.config.PATH_CORPORA`. Corpus names use CamelCase in code (`CanonFiction`) and snake_case on disk (`canon_fiction`).

### Generated data (`./data/`)

Intermediate and output data lives in `data/` (not tracked in git):

```
data/
    fields/                # word norms, semantic field definitions
        sources/           # raw downloaded norm files (Paivio PDF, MRC dict, etc.)
        data.wordnorms_orig.csv
        data.wordnorms_vec.csv
        data.allnorms.pkl
        stopwords.txt
        capslocked.CanonFiction.txt
        spelling_variants_from_morphadorner.txt
    models/                # trained Word2Vec models, organized by corpus/period/run
        eebo_tcp/
            1500-1600/
                run_01/model.bin
                ...
        ecco_tcp/
        coha/
    counts/                # passage-level abstract/concrete counts per corpus
    scores/                # per-corpus frequency-based norm scores (from score_all_corpora)
    psgs/                  # generated passage markdown files
    stash/                 # LLM response cache
    figures/               # saved plots
```

## Package modules

### `config.py`
All path constants and hyperparameters. Change `PATH_CORPORA` here if your corpora are elsewhere. Key parameters: `ZCUT = 1.0` (z-score threshold for abstract/concrete classification), `COUNT_WINDOW_LEN = 100` (sliding window size).

### `corpus.py`
The `Corpus` class replaces the old `lltk` library dependency. `load_corpus("CanonFiction")` loads metadata and provides `text_path()`, `read_text()`, and `text_paths()` methods. Also provides `pmap()` and `pmap_iter()` for parallelism via `concurrent.futures`.

### `tokenize.py`
Text tokenization with support for historical text: handles XML entities (`&longs;`, `&mdash;`), unicode normalization, optional spelling modernization (via MorphAdorner variant table), and stopword filtering.

### `norms.py`
The heart of the project. Loads concreteness ratings from four published sources, z-scores them, and classifies words into semantic fields (Abstract, Concrete, Neither). Also handles vector-based norms from historical Word2Vec models, combining empirical and historical measurements.

**Key functions:**
- `get_orignorms()` — load empirical norms from published studies
- `get_vecnorms()` — load norms derived from historical word embeddings
- `get_allnorms()` — combine both into a single DataFrame
- `get_origcontrasts()` / `get_allcontrasts()` — get word sets for each field (abstract words, concrete words, neither)
- `classify_word(z)` — classify a single z-score as Abstract/Concrete/Neither
- `format_norms_as_long()` — reshape norms into long format for plotting

### `counting.py`
Slides a window of N recognized words across a text, counting how many fall into each category.

**Key functions:**
- `count_absconc(txt)` — count abstract/concrete words in sliding windows
- `count_absconc_psg(txt)` — same, but includes HTML-marked passage text showing which words were classified
- `count_absconc_corpus("CanonFiction", num_proc=4)` — batch-count an entire corpus in parallel
- `score_psg(txt)` is in `scoring.py` for a simpler single-number score

### `models.py`
Training and using historical Word2Vec models.

**Pipeline (each step writes to disk, run once):**
1. `gen_skipgrams_corpus("eebo_tcp", ...)` — tokenize corpus texts into skipgram files by period
2. `gen_model(skipgram_path, num_runs=10)` — train Word2Vec models (multiple runs for stability)
3. `gen_vecnorms()` — project all words onto abstract/concrete field vectors in each period's model, producing historical concreteness scores

**Analysis:**
- `load_model(path)` — load a trained model
- `get_model_paths()` — find all models on disk
- `get_fieldvecs_in_model(model, contrasts)` — compute abstract/concrete direction vectors

### `scoring.py`
Higher-level analysis utilities.

- `score_psg(txt)` — quick single-number concreteness score for any text
- `score_freqs(freqs)` — score a `{word: count}` dict against one norm column
- `score_corpus_freqs(corpus_dir)` — score all `freqs/*.json` files in a corpus against all 56 norm columns (vectorized)
- `score_all_corpora()` — discover all corpora with `freqs/` folders, deduplicate shared paths, score and save per-corpus `.pkl` files to `data/scores/`
- `get_all_passages("CanonFiction")` — load precomputed passage scores with z-scores and year bins
- `sample_passages(df)` — stratified sampling across time periods and abstraction levels
- `gen_bookpassages("CanonFiction", text_id)` — generate scored passages for one text

### `plotting.py`
Visualization with plotnine (ggplot2-style).

- `plot_norms(dfnorms, words={...})` — plot words along the abstract-concrete axis across norm sources
- `plot_allnorms()` — convenience wrapper for all empirical + historical sources
- `plot_fiction(df, valtype="abs-conc")` — plot abstraction trends across fiction history, colored by genre
- `load_data_for_plotting()` — load and merge count data with corpus metadata

### `llm.py`
Optional LLM text generation with caching. `generate_text(prompt, model="gemini/gemini-2.5-pro")` calls litellm and caches responses in a HashStash pairtree. Requires `pip install -e ".[llm]"`.

### `utils.py`
Shared utilities: `read_df`/`save_df` (multi-format DataFrame I/O), `writegen` (streaming dict-to-CSV writer), `download_tqdm`, `zfy` (z-score a series), `sent_tokenize_exact`, `parse_json_str`.

### `cli.py`
Command-line interface. After `pip install -e .`:

```bash
# Score all corpora that have freqs/ folders (saves to data/scores/)
abstraction score-corpora

# Score a single corpus
abstraction score-corpus canon_fiction

# Re-score even if output already exists
abstraction score-corpora --force
```

## Notebooks

The `notebooks/` directory contains Jupyter notebooks that demonstrate the package:

| Notebook | Description |
|----------|-------------|
| `01-word-norms` | Load and explore psycholinguistic norms, classify words, visualize across sources |
| `02-counting` | Score passages, sliding-window counting, compare abstract vs. concrete text |
| `03-fiction-trends` | Plot abstraction trends across fiction history by genre |
| `04-passages` | Find extreme passages, stratified sampling, per-book passage generation |
| `05-models` | Inspect Word2Vec models, explore word neighborhoods, vector norms pipeline |
| `06-corpus` | Explore the Corpus class, list available corpora, read and tokenize texts |

## Key concepts

**Word norms** are published psycholinguistic ratings of how concrete or abstract a word is. We use four sources, each contributing a different operationalization:

| Abbreviation | Source | What it measures |
|-------------|--------|-----------------|
| `PAV-Conc` | Paivio et al. (1968) | Concreteness |
| `PAV-Imag` | Paivio et al. (1968) | Imagery |
| `MRC-Conc` | MRC Psycholinguistic Database (1987) | Concreteness |
| `MRC-Imag` | MRC (1987) | Imagery |
| `MT-Conc` | Brysbaert et al. (2014) | Concreteness (Mechanical Turk) |
| `LSN-Imag` | Lancaster Sensorimotor Norms (2017) | Visual strength |
| `LSN-Hapt` | Lancaster (2017) | Haptic strength |

**Semantic fields** are sets of words classified by z-score threshold. With the default `ZCUT = 1.0`: words with z >= 1.0 are **Concrete**, z <= -1.0 are **Abstract**, and the rest are **Neither**.

**Vector norms** extend this into historical time. By training Word2Vec on period-specific corpora, we can measure a word's concreteness *as it was used in that era*. The periods are labeled by century (C16, C17, C18, C19, C20). The median across periods gives a historical baseline.

**Counting** works by sliding a window of 100 recognized words (words that appear in the norm vocabulary) across a text. Each window produces counts of abstract, concrete, and neither words, plus an `abs-conc` score (abstract count minus concrete count). Positive = more abstract; negative = more concrete.
