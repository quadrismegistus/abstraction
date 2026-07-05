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

### 2. ~~Cache invalidation family (§4.1–4.4)~~ DONE 2026-07-05 (commit `dc5fa31`)
`norms_version(lang)` fingerprint (norms.py) now stamps: `freqs_cache.db` (versioned column; legacy rows stamped with the version current at migration; `corpus_correction`'s query contract preserved), `_NORMS_ARRAYS_CACHE` (keyed by allnorms identity — cross-language contamination fixed), `get_norm_dict` EN-fallback (warns once, caches under requested key), trajectory JSON cache (fingerprint in filename), CSV resume (refuses on header mismatch). 22 regression tests.

### 3. ~~Retire the dead DuckDB shadow (§5)~~ DONE 2026-07-05 (commits `7080d5c`, `dc5fa31`)
`score-missing` removed; `score_all_missing` deleted; `scores_db.py` deprecated (import + write warnings; read path kept for archaeology); package-root re-exports removed. Still open from this item: promoting `scripts/score_serverside_ch.py` into the CLI.

### 4. ~~Research-validity pass (§2)~~ DONE 2026-07-05 except §2.10 (commits `cc0abd8`, `b662fd9`, `45539f7`)
Fixed: report_full decade mislabeling; per_feature_r2 centering; norm_period median-fallback labeling (+`period_score_source` column); DE Schmidtke single-population z-scoring (**changes 18/6,682 DE words, 7 flip classification — regenerate `get_orignorms_de(force=True)` → `get_allnorms_de(force=True)` when convenient; vecnorm retraining not warranted**); FR/DE/ES stopword case-handling (zero effect on current stored data — forward-looking); lossless passage rendering (`:"()` survive; scoring byte-identical); seeded plot_norms; ±1.0 boundary inclusive + legend generated from `_word_style`; pct_abstract boundary-bin fold-in; report_arc corpus-balanced magnitudes; post-selection caveat lines.
**Still open — §2.10**: CH/DuckDB scorers omit the `_pct_abs/_pct_conc` frequency-proportion columns the legacy path computed; adding them means a CH schema migration for `scores_{en,fr,de,es}` (new columns + re-score or backfill).

### 5. ~~Backlog round (§5/§10/§11 leftovers)~~ DONE 2026-07-05 (commits `89b5a35`..`94ccc3a`)
Fixed: gzip-as-text counting; both `--workers` pickling crashes (module-level workers); `gen-vecnorms --workers` parallelized for real; `periodize` (crashed <1000 AND silently read the wrong digit for 3-digit years); `pmap` input-order results; `_biny` NaN years; `save_bookpassages` partial-stack drop; `aggregate` LIKE `_` wildcard (matched inside names like `ecco_tcp`) → `startsWith`; Memory temp-table leak; INNER-JOIN rep accounting (live: 42/86,469 arc reps have no scored group members); `words.py` cosine centered (an anti-correlated word previously scored +0.97), NaN-poisoned z-scores, `freq_change_pct` NaN for absent-early words, duplicate-index crashes; `adjust_scores`/`report_arc` now accept AND forward `search_range`/`search_step`; dead `report_piecewise`/`pct_in_range`/`RawCorpusInfo` deleted; ratio/star helpers consolidated (isfinite-guarded); dead app endpoints (`/arc/aggregated`, `/arc/by-corpus`) removed; `list_corpora` N+1 → one GROUP BY (output verified identical); frontend: API base centralized + env-overridable, adapter-static, real texts-page pagination, keyboard/ARIA on clickable rows; package-wide pyflakes clean; CI installs `[analysis]` extra.

### 6. Remaining (optional tier)
- **§2.10**: CH/DuckDB scorers omit `_pct_abs/_pct_conc` columns — needs a CH schema migration for `scores_{en,fr,de,es}` + backfill decision.
- ~~**DE norms regeneration**~~ DONE 2026-07-05: orignorms + allnorms.de regenerated with the Schmidtke fix (exactly 18 SCM-Imag words changed, as predicted; polarity tests pass; backups at `*.bak-preschmidtke`). `norms_version("de")` changed, so DE caches self-invalidate. NOT yet done: re-scoring `scores_de` in CH under the new allnorms (`abstraction score-ids <corpus> --lang de --force` or the server-side script) — effect is ~18 words/0.27% of vocabulary, negligible at aggregate level, so re-score whenever DE work next happens anyway.
- **`analysis.py` file split** (arcfit/reports/loading_legacy/embeddings) — deferred for notebook import compatibility; if done, keep re-exports in `analysis.py`.
- **Consolidate the 4 numpy scorer copies** in scoring.py (audit §10) — the audit's §4 cache-keying fix removed the sharpest risk; remaining value is maintainability.
- Legacy notebooks importing vanished modules: `ESTCCounts*`/`ESTCMatch` (`book_history`), `LLMPsgs2-Analyze-Pamela` (`abstraction.llm`, removed `c97eb17`) — port or archive.
- Possible dead-weight functions kept only because `__init__.py` re-exports them (no other callers): `fit_arc_corpus`, `fit_arc_by_genre`, `display_passage`, `aggregate_freqs_by_decade`, `load_aggregate_freqs`.
- Frontend pre-existing `npm run check` errors (14, all in untouched code: plotly typings, route param strictness).

## Operational notes (things not obvious from the code)

- **Disk**: `/Volumes/chambers` (hosts `~/lltk_data`, this repo's `data/` symlink, and the CH store) hit 100% full on 2026-07-04; space was freed 2026-07-05 (~247G headroom). If it fills again, reclaim candidates were: `huggingface/hub` LLM weights (re-downloadable); stale `lltk_data/data/metadb*.duckdb.tmp` dirs. ClickHouse itself legitimately uses ~412G.
- **Starting ClickHouse** (localhost:8123, user `lltk`/`lltk`): `clickhouse server --config-file=/Users/rj416/lltk_data/data/clickhouse-config/config.xml`. **Never** run bare `clickhouse server` from a repo directory — that's what created the droppings cleaned up in `43ec3a2` (`.gitignore` now guards against it).
- **Sibling repos**: `~/github/lltk` is imported via `sys.path.insert` hacks (`cli.py:71`, `scoring.py:522`, `aggregate.py:51`); it is not pip-installable from this repo's requirements. `~/github/largeliterarymodels` similarly for passage/embedding analyses.
- **Testing**: `pytest tests/ -m "not integration" -q` is the fast suite (~5s, no local data needed). Integration tests need the `data/` symlink target mounted and `~/lltk_data/corpora/canon_fiction/`; they skip cleanly otherwise. The norm-mocking contract lives in `tests/conftest.py` (`install_fake_norms`) — `get_norm_dict` caches by `(col, lang)` tuples; never patch `_NORM_DICTS` with string keys. Note `abstraction/__init__.py` re-exports a `tokenize()` function that shadows the `tokenize` submodule — use `importlib.import_module("abstraction.tokenize")` to reach the module.
- **For orchestrators running parallel subagents in this repo**: instruct them never to run `git stash` — one did during the 2026-07-04 remediation and temporarily reverted all concurrent agents' uncommitted work (recovered via its own pop, but the failure mode is real).
