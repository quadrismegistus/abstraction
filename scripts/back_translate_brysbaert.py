"""Back-translation validation pass on Brysbaert DE translations.

Takes German primary translations (de_1) produced by TranslationTask and
back-translates to English, to verify that the forward translation preserved
the original English sense. Round-trip success rate validates the
translation layer; stratified by ambiguity spread, it also tells us whether
the top-3 spread signal is honestly calibrated.

Design (per lltk-claude's recommendation):
- Stratified 1000-row sample: 500 low-ambig (de_ambig=1) + 500 high-ambig (=3).
- Back-translate with claude-sonnet-4-6 (different model from forward pass
  to break same-model confirmation bias; cross-model agreement as bonus signal).
- Success metric: does sonnet's top-1 EN match the original Brysbaert word
  (case-insensitive, allowing morphological variants via a loose matcher).

Hypothesis: low-ambig should round-trip at >95%; high-ambig significantly
lower. If the gradient matches ambiguity rank, the spread signal is honest.
"""

import os
import sys
import re

import pandas as pd
from pydantic import BaseModel, Field

from largeliterarymodels.task import Task


FULL_CSV = os.path.join(
    os.path.dirname(__file__), "..", "data", "fields", "sources",
    "brysbaert_translations_full.csv",
)
OUT_CSV = os.path.join(
    os.path.dirname(__file__), "..", "data", "fields", "sources",
    "brysbaert_back_translation_validation.csv",
)

MODEL = "claude-sonnet-4-6"
N_PER_BUCKET = 500


class EnglishTranslation(BaseModel):
    en_1: str = Field(
        description="Primary English translation of the given German word. "
        "Use the most natural single-word equivalent. For nouns, give the "
        "base form (lowercase, no article). For verbs, bare infinitive (no 'to'). "
        "For adjectives, bare positive form."
    )
    en_2: str = Field(
        default="",
        description="Second-ranked English translation, or empty if none. "
        "Same formatting as en_1."
    )
    en_3: str = Field(
        default="",
        description="Third-ranked English translation, or empty. "
        "Only include for genuinely polysemous words."
    )
    sense_note: str = Field(
        default="",
        description="One brief sentence explaining the sense chosen for "
        "polysemous German words. Leave empty otherwise."
    )


SYSTEM_PROMPT = """You are a multilingual lexicographer translating German words into English for a validation pass on automated translation quality. These back-translations are compared against an original English source word; your job is to return the most natural single-word English equivalents for the given German input, ranked by likelihood.

## Your task in context

A prior translation pass mapped English words to German. We now feed those German words back through you and check whether the original English word appears in your top-3. Round-trip matches validate the forward translation; mismatches either signal a translation error or a genuinely polysemous word where the forward pass chose a different sense than you choose now. Both outcomes are informative, but your job is NOT to guess what the original English word was — translate the German word on its merits and let the downstream comparison handle matching.

## Your input and output

Input: one German word (sometimes inflected, sometimes a base form — translate whatever form is given).

Output: up to 3 English translations, ranked by likelihood of being the word a native speaker would choose for this German word. Plus a brief sense_note for polysemous or ambiguous cases.

## Formatting rules, strict

1. **Verbs**: return the bare English infinitive WITHOUT the 'to' particle. Example: 'run' not 'to run'. This includes modals and auxiliaries.

2. **Nouns**: return the singular base form, lowercase, no article. No 'the', 'a', or possessives. Example: 'table' not 'the table'; 'child' not 'Child'.

3. **Adjectives**: return the bare positive form (no comparative/superlative, no inflection). Example: 'tall' not 'taller'; 'beautiful' not 'most beautiful'.

4. **Adverbs**: standard English form. Example: 'quickly' not 'quicker'.

5. **Compound translations**: if the German word's best English equivalent is a compound or phrase, return the phrase as a single string but note that it's not a single-word equivalent in sense_note.

## Ranking and padding rules

1. **Do not pad the top-3.** If only one or two clear English translations exist for this German word, leave en_2 or en_3 empty. Padding with near-synonyms, register variants, or regional forms corrupts the downstream validation signal.

2. **Rank by primary-sense likelihood, not completeness.** en_1 is what a native English speaker would say first given this German word out of context. en_2 and en_3 cover distinct senses or genuinely plausible alternates.

3. **Genuinely polysemous German words get multiple candidates.** A word like 'Bank' legitimately has two English translations (bank/bench); both belong in your output. A word like 'Freiheit' unambiguously maps to 'freedom' (with 'liberty' as a register-neighbor); do not reach for a third translation.

## sense_note discipline

1. **Leave sense_note empty for unambiguous words.** Most German nouns and non-polysemous verbs map cleanly to one English word and need no explanation.

2. **Populate sense_note ONLY when you made a judgement between multiple genuine senses**, or when the German word is a homograph, a false-friend, or a word with strong register/domain variation across English candidates.

3. **Keep sense_notes to one brief sentence.** This field is for downstream debugging, not for showing your work in general.

## Edge cases

- **No English equivalent**: if the German word has no natural single-word English translation, return the closest paraphrase in en_1 and note the mismatch in sense_note.
- **Function words or bound morphemes fed in isolation**: return empty primaries and explain in sense_note. These are almost certainly errors in the input.
- **Archaic or domain-specific German**: use the contemporary standard English equivalent; note if the German is archaic/register-marked.
"""


EXAMPLES = [
    ("German word: Freiheit",
     EnglishTranslation(en_1="freedom", en_2="liberty", en_3="", sense_note="")),
    ("German word: laufen",
     EnglishTranslation(
         en_1="run", en_2="walk", en_3="go",
         sense_note="Primary is fast locomotion; en_2/en_3 cover everyday ambulation senses common in idiomatic German.",
     )),
    ("German word: Tisch",
     EnglishTranslation(en_1="table", en_2="", en_3="", sense_note="")),
    ("German word: Wille",
     EnglishTranslation(en_1="will", en_2="volition", en_3="",
                        sense_note="Primary is volitional 'will' (strength of resolve).")),
    ("German word: sollen",
     EnglishTranslation(
         en_1="should", en_2="ought", en_3="must",
         sense_note="Modal verb with deontic/normative reading. Primary is 'should' "
                    "(weak obligation); en_3 covers stronger-force usage. Do NOT return "
                    "'shall' — archaic and not the primary English modal here.",
     )),
    ("German word: Bank",
     EnglishTranslation(
         en_1="bank", en_2="bench", en_3="",
         sense_note="Homograph: financial institution (primary in modern usage) and "
                    "seating bench (traditional). Both are genuine senses; ranking "
                    "reflects frequency in general-text corpora.",
     )),
    ("German word: tragen",
     EnglishTranslation(
         en_1="carry", en_2="wear", en_3="bear",
         sense_note="Polysemous across physical (carry an object), sartorial (wear "
                    "clothing), and metaphorical (bear responsibility) senses — all "
                    "three are equally common single-word English equivalents.",
     )),
    ("German word: schön",
     EnglishTranslation(
         en_1="beautiful", en_2="pretty", en_3="fine",
         sense_note="Aesthetic quality spanning intensity register: 'beautiful' is the "
                    "canonical strong term; 'pretty' is the everyday weaker form; "
                    "'fine' covers the evaluative/approving use ('fine weather').",
     )),
    ("German word: sich_verb_reflexive_marker",
     EnglishTranslation(
         en_1="", en_2="", en_3="",
         sense_note="'Sich' is not a word in isolation — it is a reflexive pronoun that "
                    "only functions with a verb. No standalone English equivalent. If a "
                    "back-translation input looks like a reflexive marker or function "
                    "word in isolation, return empty primaries and note the mismatch.",
     )),
]


class BackTranslationTask(Task):
    name = "back_translate_de"
    schema = EnglishTranslation
    system_prompt = SYSTEM_PROMPT
    examples = EXAMPLES
    retries = 2
    temperature = 0.3
    max_tokens = 512


def format_prompt(de_word: str) -> str:
    return f"German word: {de_word}"


def normalize_en(s: str) -> str:
    """Loose normalizer for round-trip matching."""
    if not isinstance(s, str):
        return ""
    return re.sub(r"[^a-z]", "", s.lower().strip())


def round_trip_match(brys_en: str, back_candidates: list[str]) -> bool:
    """True if any back-translation candidate matches the original English word
    under loose normalization (case-insensitive, punctuation-stripped)."""
    brys_norm = normalize_en(brys_en)
    for cand in back_candidates:
        if normalize_en(cand) == brys_norm:
            return True
    return False


def main():
    df = pd.read_csv(FULL_CSV)
    # Only rows with a valid DE primary and Brysbaert English word
    df = df[df["de_1"].notna() & (df["de_1"] != "")]
    df = df[df["meta_word"].notna()]

    low = df[df["de_ambig"] == 1]
    high = df[df["de_ambig"] == 3]
    print(f"available: low-ambig={len(low):,}  high-ambig={len(high):,}", file=sys.stderr)

    n_low = min(N_PER_BUCKET, len(low))
    n_high = min(N_PER_BUCKET, len(high))
    sample = pd.concat([
        low.sample(n=n_low, random_state=0),
        high.sample(n=n_high, random_state=0),
    ]).reset_index(drop=True)

    print(f"sampling {len(sample):,} rows for back-translation", file=sys.stderr)
    print(f"model: {MODEL}", file=sys.stderr)

    prompts = [format_prompt(r["de_1"]) for _, r in sample.iterrows()]
    metadata_list = [
        {
            "de_word": r["de_1"],
            "brys_en": r["meta_word"],
            "brys_pos": r["meta_pos"],
            "de_ambig": int(r["de_ambig"]),
            "source": "back_validation",
        }
        for _, r in sample.iterrows()
    ]

    task = BackTranslationTask()
    task.map(prompts, metadata_list=metadata_list,
             model=MODEL, num_workers=8)

    out = task.df
    out = out[(out["model"] == MODEL) & (out["meta_source"] == "back_validation")].copy()

    # Round-trip check
    out["round_trip"] = out.apply(
        lambda r: round_trip_match(
            r["meta_brys_en"], [r["en_1"], r["en_2"], r["en_3"]]
        ),
        axis=1,
    )
    # Stricter: top-1 only
    out["round_trip_top1"] = out.apply(
        lambda r: normalize_en(r["meta_brys_en"]) == normalize_en(r["en_1"]),
        axis=1,
    )

    out.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_CSV} ({len(out):,} rows)", file=sys.stderr)

    # Summary
    print("\n===== BACK-TRANSLATION VALIDATION =====")
    for amb in (1, 3):
        sub = out[out["meta_de_ambig"] == amb]
        if not len(sub):
            continue
        rt_any = sub["round_trip"].mean()
        rt_top1 = sub["round_trip_top1"].mean()
        print(f"\nde_ambig={amb}  (N={len(sub):,})")
        print(f"  round-trip any-of-3: {100*rt_any:.1f}%")
        print(f"  round-trip top-1:    {100*rt_top1:.1f}%")

    print("\n----- Round-trip failures (high-ambig, top-1) -----")
    fails = out[(out["meta_de_ambig"] == 3) & (~out["round_trip_top1"])].head(15)
    print(fails[["meta_brys_en", "meta_brys_pos", "meta_de_word",
                 "en_1", "en_2", "en_3", "sense_note"]].to_string(index=False))


if __name__ == "__main__":
    main()
