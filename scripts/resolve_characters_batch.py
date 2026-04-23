"""Run CharacterTask (LLM character resolution) on all chadwyck BookNLP-parsed texts
that don't yet have characters_resolved.json.

Saves results as characters_resolved.json in each text's booknlp dir.

Usage:
    python scripts/resolve_characters_batch.py [--workers 8] [--dry-run]
"""
import sys, os, json, glob, argparse

sys.path.insert(0, os.path.expanduser('~/github/lltk'))
sys.path.insert(0, os.path.expanduser('~/github/largeliterarymodels'))

import lltk
from largeliterarymodels.tasks.resolve_characters import CharacterTask


def format_roster_from_dir(booknlp_dir, title='', author='', year='', genre='',
                           min_count=10, max_chars=50):
    """Build character roster prompt directly from booknlp directory files.

    Only includes clusters with at least one proper noun mention.
    Omits agent/patient verbs (not useful for resolution).
    Common-only clusters (Sir, Madam, my Master) are skipped — they're
    ambiguous and downstream tasks only use named characters.
    """
    book_path = os.path.join(booknlp_dir, 'text.book')
    if not os.path.exists(book_path):
        return None, None

    with open(book_path) as f:
        book = json.load(f)

    # Only clusters with proper noun mentions
    chars = [c for c in book['characters']
             if c['count'] >= min_count and c['id'] != 0
             and c['mentions']['proper']]
    chars.sort(key=lambda c: -c['count'])
    chars = chars[:max_chars]

    if not chars:
        return None, None

    lines = []
    for c in chars:
        cid = f"C{c['id']:03d}"
        proper = [m['n'] for m in c['mentions']['proper'][:5]]
        common = [m['n'] for m in c['mentions']['common'][:3]]
        g = c.get('g')
        gender = g.get('argmax', '?') if g else '?'
        line = f"{cid} ({c['count']}x) proper={proper}"
        if common:
            line += f" common={common}"
        line += f" gender={gender}"
        lines.append(line)

    header = f'Text: "{title}" ({year})'
    if author:
        name = author.split(',')[0].strip() if ',' in author else author
        header += f" by {name}"
    if genre:
        header += f". Genre: {genre}"
    header += "."

    prompt = f"{header}\n\nCLUSTERS:\n" + "\n".join(lines)

    meta = {
        'title': title,
        'author': author,
        'year': int(year) if year else None,
        'genre': genre,
        'n_clusters': len(chars),
    }

    return prompt, meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--model', type=str, default='gemini-2.5-flash')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--force', action='store_true',
                        help='Re-resolve even if characters_resolved.json exists')
    args = parser.parse_args()

    # Find all BookNLP-parsed texts without resolved characters
    # Search across all corpora
    book_paths = glob.glob(os.path.expanduser(
        '~/lltk_data/corpora/*/booknlp/en_small/*/*/text.book'
    )) + glob.glob(os.path.expanduser(
        '~/lltk_data/corpora/*/booknlp/en_small/*/text.book'
    ))

    texts = []
    for bp in sorted(set(book_paths)):
        booknlp_dir = os.path.dirname(bp)
        resolved_path = os.path.join(booknlp_dir, 'characters_resolved.json')

        if os.path.exists(resolved_path) and not args.force:
            continue

        # Get corpus and text_id from path
        parts = bp.split('/corpora/')[1].split('/booknlp/')
        corpus = parts[0]
        text_id = parts[1].replace('en_small/', '').replace('/text.book', '')
        _id = f'_{corpus}/{text_id}'

        try:
            r = lltk.db.get(_id)
            year = r.get('year') if r else None
            title = r.get('title', '') if r else ''
            author = r.get('author', '') if r else ''
            genre = r.get('genre', '') if r else ''
        except:
            year, title, author, genre = None, '', '', ''

        texts.append({
            '_id': _id, 'corpus': corpus, 'text_id': text_id,
            'booknlp_dir': booknlp_dir,
            'year': year, 'title': title, 'author': author, 'genre': genre,
        })

    texts.sort(key=lambda t: t.get('year') or 0)
    print(f"{len(texts)} texts need character resolution\n")

    # Build prompts
    prompts = []
    metas = []
    dirs = []
    for t in texts:
        prompt, meta = format_roster_from_dir(
            t['booknlp_dir'], title=t['title'], author=t['author'],
            year=t['year'] or '', genre=t['genre']
        )
        if prompt is None:
            continue
        meta['_id'] = t['_id']
        meta['booknlp_dir'] = t['booknlp_dir']
        prompts.append(prompt)
        metas.append(meta)
        dirs.append(t['booknlp_dir'])
        print(f"  {t['year'] or '????'}  {t['title'][:50]:<50s}  {meta['n_clusters']} clusters")

    print(f"\nTotal: {len(prompts)} texts to resolve")

    if args.dry_run:
        # Show a sample
        if prompts:
            print(f"\n--- Sample prompt ---")
            print(prompts[0][:500])
        return

    # Run
    task = CharacterTask()
    results = task.map(prompts, model=args.model, num_workers=args.workers,
                       metadata_list=metas)

    n_ok = 0
    n_fail = 0
    for result, meta, booknlp_dir in zip(results, metas, dirs):
        if result is None:
            n_fail += 1
            print(f"  FAILED: {meta['_id']}")
            continue

        # Save as characters_resolved.json
        output = {
            '_id': meta['_id'],
            'characters': [r.model_dump() for r in result],
        }
        out_path = os.path.join(booknlp_dir, 'characters_resolved.json')
        with open(out_path, 'w') as f:
            json.dump(output, f, indent=2)

        n_chars = len([c for c in result if c.type == 'character'])
        tid = booknlp_dir.split('en_small/')[1]
        print(f"  Saved {n_chars} chars -> {tid}/characters_resolved.json")
        n_ok += 1

    print(f"\nDone: {n_ok} resolved, {n_fail} failed")


if __name__ == '__main__':
    main()
