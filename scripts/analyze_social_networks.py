"""Aggregate social network statistics across all parsed texts.

Computes per-text graph metrics across 4 network types (relation, event,
dialogue, composite), event/relation distributions, gender ratios, name
classifications, and residualized metrics controlling for cast size.

Owns the thesis-shaped pipeline (decade/form/mode breakdowns, residualization,
summary tables). Graph builders themselves live upstream in
largeliterarymodels.analysis.social_networks.

Usage:
    python scripts/analyze_social_networks.py
    python scripts/analyze_social_networks.py --parish-data /path/to/ncumb.txt
"""

import argparse
import os
import re
import sys
from collections import Counter, defaultdict

import networkx as nx
import numpy as np
import pandas as pd

import lltk
from lltk.annotate import iter_task_results
from largeliterarymodels.analysis.social_networks import (
    SocialNetwork, build_graph, build_directed_graph,
    build_dialogue_graph, build_event_graph,
)
from abstraction.scoring import score_psg


def char_desc_text(chars):
    """Pool every character's `descriptions` + `intro_text` into one string.

    Used as a per-text proxy for "how abstract is the language used to
    describe persons in this novel?" — distinct from text-level abstractness
    (which averages over narrative + dialogue + description alike).
    """
    parts = []
    for c in chars:
        parts.extend(c.get('descriptions') or [])
        if c.get('intro_text'):
            parts.append(c['intro_text'])
    return ' '.join(parts)

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
SN_DIR = os.path.join(DATA_DIR, 'social_networks')
LLTM_DATA_DIR = os.path.expanduser('~/github/largeliterarymodels/data')


_DEFAULT_DROP_BUCKETS = {'excluded', 'excluded_or_generic', 'unclassified'}


def load_taxonomy(path, drop_buckets=_DEFAULT_DROP_BUCKETS):
    """Load a YAML mapping bucket -> member strings.

    Format: top-level keys are bucket names; each value is a dict with a
    `members` list of raw strings. Returns (lookup, bucket_names) where
    lookup maps raw string -> bucket and bucket_names is the ordered list
    of buckets (with `drop_buckets` removed).
    """
    import yaml
    with open(path) as f:
        spec = yaml.safe_load(f)
    lookup = {}
    bucket_names = []
    for bucket, body in spec.items():
        if bucket in drop_buckets:
            continue
        bucket_names.append(bucket)
        for raw in (body.get('members') or []):
            lookup[raw.strip().lower()] = bucket
    return lookup, bucket_names


def bucket_pcts(raw_values, lookup, bucket_names, prefix):
    """Per-text % of items falling into each bucket.

    Denominator is the count that maps to a known bucket (exclusions and
    unknown raw strings are dropped from the base). Returns
    {f'{prefix}_{bucket}_pct': value, ...} for every bucket name.
    """
    counts = Counter()
    for v in raw_values:
        v = (v or '').strip().lower()
        if not v:
            continue
        b = lookup.get(v)
        if b is not None:
            counts[b] += 1
    total = sum(counts.values())
    return {
        f'{prefix}_{b}_pct': round(counts[b] / total * 100, 1) if total else 0.0
        for b in bucket_names
    }


def _default_parish_path():
    """Return the compiled names list if present, else fall back to ncumb sources.

    Preferred: data/names/names.txt (built by scripts/build_names_list.py from
    NE England parish + Cambridge Group reconstitution + Edinburgh registers).
    Falls back to data/names/ncumb.txt or lltm's copy.
    """
    candidates = [
        os.path.join(DATA_DIR, 'names', 'names.txt'),
        os.path.join(DATA_DIR, 'names', 'ncumb.txt'),
        os.path.join(DATA_DIR, 'ncumb.txt'),
        os.path.join(LLTM_DATA_DIR, 'ncumb.txt'),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]  # caller will report missing


def load_parish_names(path):
    """Load name set. Supports either the simple names.txt format
    (one lowercase name per line) or the original ncumb.txt CSV format.
    """
    names = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Simple list format: just a name per line
            if not line[0].isdigit() and ',' not in line:
                if not line.startswith('Names from') and not line.startswith('Created') \
                        and not line.startswith('For source') and not line.startswith('http'):
                    n = line.strip('"').strip().lower()
                    if n and len(n) >= 3:
                        names.add(n)
                continue
            # ncumb.txt CSV format
            if not line[0].isdigit():
                continue
            parts = line.split(',')
            if len(parts) >= 5:
                for idx in [3, 4]:
                    n = parts[idx].strip('"').strip()
                    if n:
                        names.add(n.lower())
    return names


def classify_name(name, parish_names=None):
    if not name:
        return 'unknown'
    if re.match(r'^(the |a |an )', name.lower()):
        return 'type'
    if re.match(r"^[A-Z][a-z]+'s ", name):
        return 'type'
    if parish_names:
        for word in name.split():
            if word.strip('.,').lower() in parish_names:
                return 'realistic'
    if re.search(r'(us|ia|issa|andra|ander|enes|oles|inda|etta|ina)$',
                 name.split()[0].lower()):
        return 'classical'
    return 'other'


def degree_gini(G):
    """Gini coefficient of degree distribution. 0=uniform, 1=star."""
    if len(G) < 2:
        return 0
    degrees = np.array(sorted([d for _, d in G.degree()]), dtype=float)
    n = len(degrees)
    if degrees.sum() == 0:
        return 0
    index = np.arange(1, n + 1)
    return float((2 * (index * degrees).sum() / (n * degrees.sum())) - (n + 1) / n)


def graph_stats(G, prefix=''):
    """Compute graph metrics. Returns dict with prefixed keys."""
    if G is None or len(G) == 0:
        return {}
    s = {}
    s['nodes'] = len(G)
    s['edges'] = G.number_of_edges()
    s['density'] = round(nx.density(G), 4)

    if isinstance(G, nx.DiGraph):
        s['reciprocity'] = round(nx.reciprocity(G), 4) if G.number_of_edges() > 0 else 0
        U = G.to_undirected()
    else:
        U = G

    s['components'] = nx.number_connected_components(U)
    largest_cc = max(nx.connected_components(U), key=len)
    s['largest_cc_frac'] = round(len(largest_cc) / len(G), 4)

    if len(U) >= 3:
        s['clustering'] = round(nx.average_clustering(U), 4)
        s['transitivity'] = round(nx.transitivity(U), 4)

    bc = nx.betweenness_centrality(U)
    s['max_betweenness'] = round(max(bc.values()), 4) if bc else 0
    s['mean_betweenness'] = round(sum(bc.values()) / len(bc), 4) if bc else 0
    s['centralization'] = round(
        max(bc.values()) - sum(bc.values()) / len(bc), 4) if len(bc) > 1 else 0

    dc = nx.degree_centrality(U)
    s['max_degree_cent'] = round(max(dc.values()), 4) if dc else 0

    s['degree_gini'] = round(degree_gini(U), 4)

    try:
        s['assortativity'] = round(nx.degree_assortativity_coefficient(U), 4)
    except (nx.NetworkXError, ValueError):
        s['assortativity'] = None

    if len(U) > 1 and nx.is_connected(U):
        s['diameter'] = nx.diameter(U)
        s['avg_path'] = round(nx.average_shortest_path_length(U), 2)
    else:
        s['diameter'] = None
        s['avg_path'] = None

    if prefix:
        s = {f'{prefix}_{k}': v for k, v in s.items()}
    return s


def event_macro_counts(events):
    verbs = Counter(e.get('what', '').lower() for e in events)
    categories = {
        'violence': ['killed', 'murdered', 'died', 'executed', 'fought', 'dueled',
                     'attacked', 'poisoned', 'wounded', 'assassinated', 'stabbed'],
        'courtship': ['married', 'courted', 'proposed', 'rejected', 'confessed love',
                      'fell in love', 'declared love', 'engaged', 'attracted_to'],
        'movement': ['arrived', 'departed', 'traveled', 'fled', 'returned', 'escaped',
                     'shipwrecked', 'embarked', 'landed', 'visited'],
        'deception': ['deceived', 'disguised', 'betrayed', 'plotted', 'conspired',
                      'forged', 'impersonated', 'feigned'],
        'legal': ['arrested', 'imprisoned', 'tried', 'convicted', 'sentenced',
                  'pardoned', 'acquitted', 'confessed', 'accused', 'executed'],
    }
    return {k: sum(verbs.get(v, 0) for v in vs) for k, vs in categories.items()}


def residualize(df, metrics, x_col='log_n_chars'):
    """Add residualized versions of metrics, controlling for cast size."""
    df[x_col] = np.log1p(df['n_chars'])
    for m in metrics:
        col = m if m in df.columns else None
        if col is None:
            continue
        valid = df[[x_col, col]].dropna()
        if len(valid) < 5:
            df[f'{col}_resid'] = np.nan
            continue
        from numpy.polynomial import polynomial as P
        coef = P.polyfit(valid[x_col], valid[col], 1)
        predicted = P.polyval(df[x_col], coef)
        df[f'{col}_resid'] = df[col] - predicted
    return df


def build_composite_graph(result):
    """Build composite graph from result dict."""
    sn = SocialNetwork(result)
    return sn.composite_graph()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parish-data', default=_default_parish_path())
    parser.add_argument('--relation-taxonomy',
                        default=os.path.join(SN_DIR, 'sn_relation_metacategories.yml'))
    parser.add_argument('--class-taxonomy',
                        default=os.path.join(SN_DIR, 'sn_class_metacategories.yml'))
    parser.add_argument('--out', default=os.path.join(SN_DIR, 'social_network_analysis.csv'))
    args = parser.parse_args()

    parish_names = None
    if os.path.exists(args.parish_data):
        parish_names = load_parish_names(args.parish_data)
        print(f"Parish register: {len(parish_names)} names from {args.parish_data}", file=sys.stderr)
    else:
        print(f"Parish data not found at {args.parish_data} — skipping realistic-name classification", file=sys.stderr)

    rel_lookup, rel_buckets = {}, []
    if os.path.exists(args.relation_taxonomy):
        rel_lookup, rel_buckets = load_taxonomy(args.relation_taxonomy)
        print(f"Relation taxonomy: {len(rel_buckets)} buckets, "
              f"{len(rel_lookup)} mapped strings, from {args.relation_taxonomy}",
              file=sys.stderr)
    else:
        print(f"Relation taxonomy not found at {args.relation_taxonomy} — "
              f"skipping rel_*_pct columns", file=sys.stderr)

    class_lookup, class_buckets = {}, []
    if os.path.exists(args.class_taxonomy):
        class_lookup, class_buckets = load_taxonomy(args.class_taxonomy)
        print(f"Class taxonomy: {len(class_buckets)} buckets, "
              f"{len(class_lookup)} mapped strings, from {args.class_taxonomy}",
              file=sys.stderr)
    else:
        print(f"Class taxonomy not found at {args.class_taxonomy} — "
              f"skipping class_*_pct columns", file=sys.stderr)

    # Pass 1: stream task results via lltk iterator (canonical ids,
    # latest-per-text by mtime). Filter to non-empty character lists.
    loaded = [(src, d) for src, d in iter_task_results('social_network')
              if d.get('characters')]
    print(f"Loaded {len(loaded)} non-empty social networks", file=sys.stderr)

    # Batch metadata + propagated genre tags via lltk.db primitives.
    text_ids = sorted({src for src, _ in loaded})
    meta_df = lltk.db.fetch_metadata(text_ids, columns=['title', 'year', 'author'])
    tags_df = lltk.db.genre_tags(text_ids, propagate=True)

    meta_lookup = meta_df.set_index('_id').to_dict('index')
    tags_lookup: dict = defaultdict(lambda: defaultdict(list))
    for _, r in tags_df.iterrows():
        tags_lookup[r['_id']][r['facet']].append(r['tag'])

    # Pass 2: per-text computation using batched lookups.
    rows = []
    for i, (src, d) in enumerate(loaded):
        chars = d['characters']
        n_chars = len(chars)
        n_events = len(d.get('events', []))
        n_rels = len(d.get('relations', []))
        n_dialogue = len(d.get('dialogue', []))
        n_passages = d.get('metadata', {}).get('n_passages', 0)

        genders = Counter(c.get('gender', '?') for c in chars)
        female_pct = round(genders.get('female', 0) / n_chars * 100, 1)

        name_counts = Counter()
        for c in chars:
            name_counts[classify_name(c['name'], parish_names)] += 1

        macros = event_macro_counts(d.get('events', []))
        macro_pcts = {f'{k}_pct': round(v / n_events * 100, 1) if n_events else 0
                      for k, v in macros.items()}

        rel_pcts = bucket_pcts(
            (r.get('type') for r in d.get('relations', [])),
            rel_lookup, rel_buckets, prefix='rel') if rel_buckets else {}

        class_pcts = bucket_pcts(
            (c.get('class') for c in chars),
            class_lookup, class_buckets, prefix='class') if class_buckets else {}

        desc_text = char_desc_text(chars)
        char_desc_conc = score_psg(desc_text) if desc_text.strip() else None
        n_desc_words = len(desc_text.split())

        all_stats = {}
        try:
            all_stats.update(graph_stats(build_composite_graph(d), prefix='comp'))
        except Exception:
            pass
        try:
            all_stats.update(graph_stats(build_directed_graph(d), prefix='rel'))
        except Exception:
            pass
        try:
            all_stats.update(graph_stats(build_event_graph(d), prefix='evt'))
        except Exception:
            pass
        try:
            all_stats.update(graph_stats(build_dialogue_graph(d), prefix='dial'))
        except Exception:
            pass

        m = meta_lookup.get(src)
        if not m or m.get('year') is None:
            continue
        title = str(m.get('title') or '?')[:50]
        author = str(m.get('author') or '?').split(',')[0][:25]
        year = int(m['year'])

        text_tags = tags_lookup.get(src, {})
        form_tags = '; '.join(sorted(text_tags.get('form', ['?'])))
        mode_tags = '; '.join(sorted(text_tags.get('mode', ['?'])))

        row = {
            'year': year, 'source': src, 'id': src.split('/')[-1],
            'title': title, 'author': author,
            'form': form_tags, 'mode': mode_tags,
            'n_passages': n_passages,
            'n_chars': n_chars, 'n_rels': n_rels,
            'n_events': n_events, 'n_dialogue': n_dialogue,
            'female_pct': female_pct,
            'realistic_pct': round(name_counts['realistic'] / n_chars * 100, 1),
            'classical_pct': round(name_counts['classical'] / n_chars * 100, 1),
            'type_pct': round(name_counts['type'] / n_chars * 100, 1),
            'char_desc_conc': char_desc_conc,
            'n_desc_words': n_desc_words,
            **macro_pcts,
            **rel_pcts,
            **class_pcts,
            **all_stats,
        }
        rows.append(row)
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(loaded)}]", file=sys.stderr)

    df = pd.DataFrame(rows).sort_values('year').reset_index(drop=True)

    # Residualize key metrics against log(n_chars)
    resid_metrics = [
        'comp_density', 'comp_clustering', 'comp_centralization',
        'comp_degree_gini', 'comp_reciprocity', 'comp_assortativity',
        'rel_density', 'rel_clustering', 'rel_reciprocity',
        'dial_reciprocity', 'evt_centralization',
    ]
    df = residualize(df, resid_metrics)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nSaved {len(df)} texts to {args.out}", file=sys.stderr)

    # === SUMMARY TABLES ===

    display_cols = [
        'n_chars', 'n_events',
        'female_pct', 'violence_pct', 'courtship_pct', 'movement_pct',
        'realistic_pct', 'classical_pct', 'type_pct',
        'comp_density', 'comp_clustering', 'comp_reciprocity',
        'comp_degree_gini', 'comp_assortativity',
        'comp_centralization',
        'rel_reciprocity', 'dial_reciprocity', 'evt_centralization',
        'comp_density_resid', 'comp_clustering_resid', 'comp_reciprocity_resid',
    ]
    short = {
        'n_chars': 'chars', 'n_events': 'evnts',
        'female_pct': '%fem', 'violence_pct': '%viol',
        'courtship_pct': '%crt', 'movement_pct': '%mov',
        'realistic_pct': '%real', 'classical_pct': '%clas', 'type_pct': '%type',
        'comp_density': 'c.dens', 'comp_clustering': 'c.clus',
        'comp_reciprocity': 'c.reci', 'comp_degree_gini': 'c.gini',
        'comp_assortativity': 'c.asor', 'comp_centralization': 'c.cntr',
        'rel_reciprocity': 'r.reci', 'dial_reciprocity': 'd.reci',
        'evt_centralization': 'e.cntr',
        'comp_density_resid': 'dens.r', 'comp_clustering_resid': 'clus.r',
        'comp_reciprocity_resid': 'reci.r',
    }

    def print_table(label, groups):
        cols = [c for c in display_cols if c in df.columns]
        header = f"{'':>22} {'n':>3}"
        for c in cols:
            header += f" {short.get(c, c[:6]):>6}"
        print(f"\n=== {label} ===")
        print(header)
        print("-" * len(header))
        for name, sub in groups:
            line = f"{str(name)[:22]:>22} {len(sub):3d}"
            for c in cols:
                val = sub[c].mean()
                if pd.notna(val):
                    if abs(val) >= 10:
                        line += f" {val:6.0f}"
                    elif abs(val) >= 1:
                        line += f" {val:6.1f}"
                    else:
                        line += f" {val:6.3f}"
                else:
                    line += f" {'':>6}"
            print(line)

    # By decade
    decade_groups = []
    min_decade = (df['year'].min() // 10) * 10
    max_decade = (df['year'].max() // 10) * 10 + 10
    for decade in range(min_decade, max_decade, 10):
        sub = df[(df['year'] >= decade) & (df['year'] < decade + 10)]
        if len(sub):
            decade_groups.append((f"{decade}s", sub))
    print_table(f"BY DECADE ({len(df)} texts)", decade_groups)

    # By form
    form_rows = []
    for _, r in df.iterrows():
        for f in str(r['form']).split('; '):
            form_rows.append({**r.to_dict(), 'form_tag': f.strip()})
    fdf = pd.DataFrame(form_rows)
    top_forms = fdf['form_tag'].value_counts().head(10).index
    print_table("BY FORM", [(f, fdf[fdf['form_tag'] == f]) for f in top_forms])

    # By mode
    mode_rows = []
    for _, r in df.iterrows():
        for m in str(r['mode']).split('; '):
            mode_rows.append({**r.to_dict(), 'mode_tag': m.strip()})
    mdf = pd.DataFrame(mode_rows)
    top_modes = mdf['mode_tag'].value_counts().head(10).index
    print_table("BY MODE", [(m, mdf[mdf['mode_tag'] == m]) for m in top_modes])

    # Correlations with n_chars
    print(f"\n=== CORRELATION WITH log(n_chars) ===")
    corr_cols = ['comp_density', 'comp_clustering', 'comp_reciprocity',
                 'comp_centralization', 'comp_degree_gini',
                 'rel_reciprocity', 'dial_reciprocity']
    for c in corr_cols:
        if c in df.columns:
            valid = df[['log_n_chars', c]].dropna()
            if len(valid) > 5:
                r = valid['log_n_chars'].corr(valid[c])
                print(f"  {short.get(c, c):>8}  r={r:+.3f}  (n={len(valid)})")


if __name__ == '__main__':
    main()
