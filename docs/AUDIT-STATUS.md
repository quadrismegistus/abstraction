# Audit remediation status & handoff

**Companion to:** `docs/AUDIT-2026-07-04.md` (the full audit — read it for detail; this file tracks what's done and what any future agent/developer should pick up next).
**Last updated:** 2026-07-05.

## Done (commits 5c79968..9c506bb, 2026-07-04)

| Commit | Fix | Audit ref |
|---|---|---|
| `43ec3a2` | ClickHouse server droppings deleted from repo root; `.gitignore` guards added | §8.1 |
| `42699c3` | Missing deps declared (clickhouse-connect, openpyxl, xlrd, matplotlib, statsmodels); `tests` package no longer ships; `viz`/`analysis` extras | §1.3, §7 |
| `244a706` | `get_contrasts` no longer crashes on `IC.*` columns — counting pipeline restored; regression test added | §1.1 |
| `02f940e` | Mechanical bugs: `--invert` flag, `read_df` dispatch, `download_tqdm` progress, `generate_json` dedup, dead `if False` code | §5 |
| `a9ad759` | Corpus-bias estimator: proper FE within-estimator (demeans X within groups), ref-component restriction, dof correction; exact-recovery tests | §1.2 |
| `41439cd` | Test suite: `tests/conftest.py` owns norm-mocking via `(col, lang)` tuple keys; markers; integration skip guards; 232 passed in ~5s | §1.4, §6 |
| `9c506bb` | Web app: col allow-list validation (`app/validation.py`), parameterized genre/corpus SQL, path-traversal containment, localhost CORS, CH timeouts, `/arc/print` race fix, payload caps | §3 |

## Open — priority order

### 1. ~~Regenerate corpus-bias coefficients~~ DONE 2026-07-05 (commit `2c2019f`)
All three coefficient files regenerated with the corrected FE estimator (disk freed, CH restarted). Old attenuated files kept as `*.json.bak-attenuated`.

- **EN** (ref `ecco_tcp`, 34 corpora, via freqs-cache loader): headline change — **ecco is now +0.072** (old file said +0.029). Validated against ground truth: the direct within-group paired difference over 600 ecco/ecco_tcp pairs is +0.078. evans_tcp −0.012 (old −0.044), eebo_tcp +0.032, internet_archive +0.070. The old values weren't just attenuated; contamination from the raw-X regression flipped some magnitudes.
- **FR** (ref `gallica_literary_fictions`, CH loader): artfl +0.005, french_pd_books +0.003 — French transcription biases are tiny.
- **DE** (ref `dta`, CH loader — file did not previously exist): german_fiction −0.098, de_corp −0.102, **german_pd +0.215** (large OCR bias; german_pd is already excluded from the arc).
- **ES**: not estimable — only 3 multi-corpus match groups (spanish_pd_books × impact_es).
- Regeneration is now reproducible: `abstraction estimate-corpus-bias --lang en|fr|de|es`. Non-EN loads from `abstraction.scores_{lang}` joined to `lltk.match_groups FINAL`, restricted to **native corpora** — scores_fr/de also contain foreign-language texts from `ecco`/`txtlab` etc., and a coefficient saved under those names would collide with the English ones in `load_all_corpus_bias()`'s merged dict. `estimate_corpus_bias` now raises if the reference corpus is absent (previously it silently re-based).
- **Follow-up for the author**: the corrected-arc view will visibly change (ecco's correction more than doubled, and ecco is a major C18 source) — re-check any figures/prose that used the corpus-corrected toggle, and restart the web app so it reloads the JSONs.

### 2. Cache invalidation family (§4.1–4.4)
Design: one `norms_version()` helper (hash of `PATH_ALLNORMS` mtime + column names) stamped into every cache:
- `freqs_cache.db` (`scoring.py` — keyed only by relpath+modernized; serves stale scores after norm regeneration)
- `_NORMS_ARRAYS_CACHE` (`scoring.py:384-402` — **keyless**: can serve English arrays for French scoring; the test suite clears it, production does not)
- `get_norm_dict` EN-fallback (`scoring.py:114-117` — silent, and uncached so it re-reads the 425MB pickle every call)
- Web app trajectory JSON cache (`app/routes/trajectory.py:19-34`)
- CSV resume header check (`scoring.py:932,1121,1495` — appends current-schema rows under stale headers)

### 3. Retire the dead DuckDB shadow (§5)
`abstraction score-missing` still writes to `scores.duckdb`, which nothing reads post-CH-cutover (2026-04-19). Rewire to `score_ids_ch` or delete the command; add `DeprecationWarning` to `scores_db.py` and drop its `__init__.py` re-exports. Also consider promoting `scripts/score_serverside_ch.py` (server-side INSERT…SELECT, much faster) into the CLI.

### 4. Research-validity pass before finalizing book numbers (§2)
Each is a one-file fix; see audit §2 for file:line detail:
- `report_full` labels count-based values with score-based decades (§2.1)
- Histogram vs live `pct_abstract` disagree (±3z truncation + boundary convention) (§2.2)
- `per_feature_r2`: center y before no-intercept lstsq (§2.3)
- `norm_period` mislabels median-fallback texts in FE analyses (§2.4)
- Schmidtke DE supplement z-scored on its own subpopulation (§2.6)
- EN vs FR/DE/ES `remove_stopwords` semantics differ (180K list vs ~200 NLTK) (§2.7)
- Passage HTML drops `:`, `"`, `(`, `)` — affects print figures (§2.8, fix in `tokenize.py:93` display path)
- Seed `plot_norms` sampling for reproducible figures (§2.9)
- CH/DuckDB scorers omit `_pct_abs/_pct_conc` columns the legacy path computed (§2.10)

### 5. Remaining audit backlog
- §5 crashes/bugs not yet fixed: `pmap` pickling (`train-skipgrams --workers`), `gen-vecnorms --workers` no-op, `int(NaN)` report crashes, plotting bugs (`min_y` compression, `facet_by_genre`, `_savefig` cwd), `words.py` NaN/cosine issues, `_biny` NaN→1400, `save_bookpassages` index-as-position, gzip-as-text counting (`counting.py:126-131`), `aggregate.py:219` LIKE wildcard.
- §9 docs: README documents 10/21 CLI commands, never mentions ClickHouse; CLAUDE.md module table omits `analysis.py`/`words.py`/`passages.py`/`norms_es.py`.
- §10 structure: `analysis.py` split proposal (arcfit/reports/loading_legacy/embeddings); consolidate the 4 numpy scorer copies; delete confirmed-dead functions (list in §10).
- §11 performance: push app aggregation into CH `GROUP BY`; fix `list_corpora` N+1.
- Push discipline: CI runs `pytest tests/ --ignore=tests/test_integration.py` on pushes to main (`.github/workflows/tests.yml`); keep it green.

## Operational notes (things not obvious from the code)

- **Disk**: `/Volumes/chambers` (hosts `~/lltk_data`, this repo's `data/` symlink, and the CH store) hit 100% full on 2026-07-04; space was freed 2026-07-05 (~247G headroom). If it fills again, reclaim candidates were: `huggingface/hub` LLM weights (re-downloadable); stale `lltk_data/data/metadb*.duckdb.tmp` dirs. ClickHouse itself legitimately uses ~412G.
- **Starting ClickHouse** (localhost:8123, user `lltk`/`lltk`): `clickhouse server --config-file=/Users/rj416/lltk_data/data/clickhouse-config/config.xml`. **Never** run bare `clickhouse server` from a repo directory — that's what created the droppings cleaned up in `43ec3a2` (`.gitignore` now guards against it).
- **Sibling repos**: `~/github/lltk` is imported via `sys.path.insert` hacks (`cli.py:71`, `scoring.py:522`, `aggregate.py:51`); it is not pip-installable from this repo's requirements. `~/github/largeliterarymodels` similarly for passage/embedding analyses.
- **Testing**: `pytest tests/ -m "not integration" -q` is the fast suite (~5s, no local data needed). Integration tests need the `data/` symlink target mounted and `~/lltk_data/corpora/canon_fiction/`; they skip cleanly otherwise. The norm-mocking contract lives in `tests/conftest.py` (`install_fake_norms`) — `get_norm_dict` caches by `(col, lang)` tuples; never patch `_NORM_DICTS` with string keys. Note `abstraction/__init__.py` re-exports a `tokenize()` function that shadows the `tokenize` submodule — use `importlib.import_module("abstraction.tokenize")` to reach the module.
- **For orchestrators running parallel subagents in this repo**: instruct them never to run `git stash` — one did during the 2026-07-04 remediation and temporarily reverted all concurrent agents' uncommitted work (recovered via its own pop, but the failure mode is real).
