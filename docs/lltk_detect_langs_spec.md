# Prompt for lltk-claude: add per-text language detection

## Context

The `abstraction` project needs reliable per-text language labels for every row in `metadb_freqs.duckdb`. Currently:

- Some corpus metadata has a `lang` column (e.g., ARTFL, txtlab, DTA) but coverage is uneven.
- Most English-dominated corpora (ecco, eebo_tcp, chadwyck, hathi_*, earlyprint, blbooks, gale_amfic, etc.) don't explicitly tag language — texts are assumed English but include French/Latin/German extracts.
- Multilingual corpora like `txtlab` and `blbooks` have scattered French/German/Latin texts that would get scored against the wrong norm dictionary if we rely on corpus-based language assumptions.

Downstream effect: `abstraction` is migrating to a per-language scoring DB (`scores_en`, `scores_fr`, `scores_de`). To decide which language's norms to score a text against, we need a per-text `lang` label.

## What to build

Add a `detect_langs` (or similar) command to LLTK that:

1. Populates a `lang` column on the `texts` table in `metadb.duckdb` (authoritative location), or adds a lookup table if you prefer.
2. For texts where metadata already has a `lang` (from corpus `load_metadata()`), use that as the truth.
3. For texts without a metadata `lang`, detect language from their freqs via stopword intersection:
   - For each supported language (en, fr, de, la, it, es, pt — whatever's useful), keep a list of ~100-200 high-frequency function words.
   - For each text, compute the fraction of total tokens (by freq weight) that belong to each language's list.
   - Assign to the language with highest coverage if it exceeds a threshold (e.g., ≥5% and ≥2× the second-place language).
   - Below threshold → `lang = 'unknown'` (e.g., Latin texts with mixed vernacular).

This is very cheap in SQL:
```sql
WITH lang_hits AS (
    SELECT tf._id,
           fw.lang,
           SUM(tf.freqs[fw.word]) AS hits
    FROM freqs_db.text_freqs tf, function_words fw
    WHERE fw.word IN (SELECT UNNEST(map_keys(tf.freqs)))
    GROUP BY tf._id, fw.lang
),
text_totals AS (
    SELECT _id, sum(cnt) AS total_tokens
    FROM (SELECT _id, unnest(map_values(freqs)) AS cnt FROM freqs_db.text_freqs)
    GROUP BY _id
)
SELECT lh._id,
       argmax(lh.lang, lh.hits) AS detected_lang,
       max(lh.hits)::DOUBLE / tt.total_tokens AS coverage
FROM lang_hits lh
JOIN text_totals tt ON lh._id = tt._id
GROUP BY lh._id, tt.total_tokens
```

Runtime should be minutes for the full 1.6M text freqs DB.

4. Flag any disagreements between metadata `lang` and freqs-detected `lang` — those are interesting (corpus metadata errors, mixed-language editions, translations miscategorized). Store as `lang_metadata` vs `lang_detected`, or just log them.

## Inputs

- Function word lists: NLTK has stopwords for en/fr/de/it/es/pt; scikit-learn and spaCy have others. Pick a clean set per language — maybe 100-150 most frequent function/closed-class words (articles, prepositions, pronouns, auxiliaries). Avoid content words so a short poem about "la rose" doesn't get misclassified.
- Existing metadata `lang` values to respect when present.

## Deliverables

1. A `lltk detect-langs` CLI that writes the final `lang` column to `texts` (or to a dedicated `text_langs` table).
2. Document language distribution of the final result: how many texts per lang, how many `unknown`, how many metadata/detection disagreements.
3. Keep the function-word lists checked in so the detection is reproducible.

## Useful cross-check

The `abstraction` project already knows the rough distribution it expects:
- ~1.6M English total
- ~20K French (gallica 15K + artfl 3.5K + txtlab ~400)
- ~3K German (dta)
- Some Latin/Italian/Spanish scattered in eebo_tcp, chadwyck, ecco

If the detector returns wildly different numbers, something's off. Report back.

## Nice-to-have

A secondary `lang_confidence` column (how much the top language dominates over the second), to let downstream code flag borderline cases.
