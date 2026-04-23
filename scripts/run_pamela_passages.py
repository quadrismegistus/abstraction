"""Run PassageTask over Richardson's Pamela Vols 1-2 using chadwyck letter sections.

Uses 500-word max windows (short letters merged, long letters split).
Outputs ~1500-2000 annotated passages covering both volumes.

Usage:
    python scripts/run_pamela_passages.py [--dry-run] [--workers 4]
"""
import sys, os, argparse
sys.path.insert(0, os.path.expanduser('~/github/lltk'))
sys.path.insert(0, os.path.expanduser('~/github/largeliterarymodels'))

import lltk
from largeliterarymodels.tasks.classify_passage import (
    PassageTask, format_chapters, format_passages_from_text
)

# Pamela texts in chadwyck
PAMELA_TEXTS = [
    'Eighteenth-Century_Fiction/richards.04',  # Pamela Vol 1 (1741)
    'Eighteenth-Century_Fiction/richards.05',  # Pamela Vol 2 (1742)
]

# Use 500-word windows for fine-grained tracking of the narrative arc
MAX_WORDS = 500
MIN_WORDS = 150


def get_pamela_prompts():
    """Generate all passage prompts for Pamela Vols 1-2."""
    c = lltk.load('chadwyck')
    all_prompts = []

    for tid in PAMELA_TEXTS:
        t = c.text(tid)
        print(f"\n{tid}: {t.title} ({t.year})")

        # Use chapter-based segmentation with 500-word max
        prompts = format_chapters(
            t,
            min_words=MIN_WORDS,
            max_words=MAX_WORDS,
        )

        if not prompts:
            # Fallback to windowed
            print("  No chapter structure, falling back to windows")
            prompts = format_passages_from_text(
                text_obj=t,
                n_words=MAX_WORDS,
                n_passages=None,  # all windows
                strategy='even',
                use_chapters=False,
            )

        print(f"  {len(prompts)} passages")
        all_prompts.extend(prompts)

    return all_prompts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true',
                        help='Show passage count without running LLM')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel workers')
    parser.add_argument('--model', type=str, default=None,
                        help='LLM model override (default: task default)')
    args = parser.parse_args()

    prompts_and_meta = get_pamela_prompts()
    prompts = [p for p, m in prompts_and_meta]
    metas = [m for p, m in prompts_and_meta]

    print(f"\nTotal: {len(prompts)} passages")
    print(f"Word counts: min={min(m['n_words'] for m in metas)}, "
          f"max={max(m['n_words'] for m in metas)}, "
          f"mean={sum(m['n_words'] for m in metas) // len(metas)}")

    if args.dry_run:
        # Show sample prompts
        for i in [0, len(prompts)//4, len(prompts)//2, 3*len(prompts)//4, -1]:
            m = metas[i]
            print(f"\n--- Passage {m['passage_index']} ({m['section_id']}) "
                  f"chapter='{m['chapter_title'][:50]}' words={m['n_words']} ---")
            print(prompts[i][:200] + '...')
        return

    task = PassageTask()
    kwargs = dict(num_workers=args.workers, metadata_list=metas)
    if args.model:
        kwargs['model'] = args.model

    print(f"\nRunning PassageTask with {args.workers} workers...")
    results = task.map(prompts, **kwargs)

    # Summary
    n_ok = sum(1 for r in results if r is not None)
    print(f"\nDone: {n_ok}/{len(results)} passages annotated")

    # Show the results DataFrame
    df = task.df
    if df is not None and len(df):
        print(f"\nResults shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"\nAllegorical regime distribution:")
        print(df['allegorical_regime'].value_counts().to_string())
        print(f"\nScene type distribution:")
        print(df['scene_type'].value_counts().head(10).to_string())

        # Save to CSV
        out_dir = os.path.expanduser('~/Dropbox/Prof/Books/AbsLitHist/data/llm_annotations')
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, 'pamela_passages.csv')
        df.to_csv(csv_path, index=False)
        print(f"\nSaved to {csv_path}")

        # Also save locally
        local_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'llm_annotations')
        os.makedirs(local_dir, exist_ok=True)
        local_path = os.path.join(local_dir, 'pamela_passages.csv')
        df.to_csv(local_path, index=False)
        print(f"Saved to {local_path}")


if __name__ == '__main__':
    main()
