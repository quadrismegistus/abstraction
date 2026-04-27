"""Compile vernacular first-names from data/names/* into one sorted, deduped list.

Sources (all under data/names/):
- ncumb.txt — Galbi NE England parish data, 1530-1830.
  CSV: id,"PARISH",year,"OriginalName","ModernizedName",sex
- 26ParishesReconstitutions_ALL_DATA.txt — Cambridge Group reconstitution, 1538-1899.
  TSV; forename columns:
    husbands_Forename, wives_Forename, children_Name,
    husband[fathers|mothers]_Forename, wife[fathers|mothers]_Forename
  Date proxy: marriages_MarDate (parents/spouses), children_BapDate (children).
- Edinburgh.xlsx — top names per year, 1838-2014.
  One row per rank, columns are integer years; cells like "Mary (10095)".

Usage: python scripts/build_names_list.py [--max-year 1900] [--out data/names/names.txt]

Default cutoff is 1900. Earlier cutoffs drop Edinburgh entirely; later cutoffs add
modern names which dilute the vernacular vocabulary for 17-18C fiction.
"""

import argparse
import os
import re
import sys

import pandas as pd

NAMES_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'names')
NAME_RE = re.compile(r"^[A-Za-zÀ-ÿ' \-]+$")


def _norm(name):
    """Lowercase, strip punctuation tails, take first word, validate.

    Rejects bare hyphen-prefixed scraps and very short tokens (which are
    too aggressive for substring-matching against character names).
    """
    if not isinstance(name, str):
        return None
    s = name.strip().strip('"').strip()
    if not s:
        return None
    s = s.split()[0]
    s = s.strip(",.;:'\"-")
    if len(s) < 3 or not NAME_RE.match(s):
        return None
    if s[0] in "-' ":
        return None
    return s.lower()


def _parse_year(s):
    """Pull year (4-digit) from a 'D-M-YYYY' or '... YYYY' style string."""
    if not isinstance(s, str):
        return None
    m = re.search(r'(\d{4})', s)
    return int(m.group(1)) if m else None


def load_ncumb(path, max_year):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or not line[0].isdigit():
                continue
            parts = line.split(',')
            if len(parts) < 5:
                continue
            try:
                year = int(parts[2])
            except (ValueError, IndexError):
                continue
            if year > max_year:
                continue
            # Field 4 is ModernizedName (Galbi's normalization). Field 3 is
            # OriginalName (raw register spelling), which adds OCR/historical
            # artifacts that hurt the substring-matching classifier.
            n = _norm(parts[4].strip('"'))
            if n:
                out.append(n)
    return out


def load_26parishes(path, max_year):
    """Stream the big TSV in chunks to keep memory low."""
    spouse_cols = [
        'husbands_Forename', 'wives_Forename',
        'husbandfathers_Forename', 'husbandmothers_Forename',
        'wifefathers_Forename', 'wifemothers_Forename',
    ]
    cols = ['marriages_MarDate', 'children_BapDate', 'children_Name'] + spouse_cols
    out = []
    for chunk in pd.read_csv(path, sep='\t', usecols=cols, dtype=str,
                             chunksize=50_000, low_memory=False):
        chunk['mar_year'] = chunk['marriages_MarDate'].map(_parse_year)
        chunk['bap_year'] = chunk['children_BapDate'].map(_parse_year)
        # Spouses + their parents: keyed by marriage year
        spouse_mask = chunk['mar_year'].between(1500, max_year, inclusive='both')
        for col in spouse_cols:
            for n in chunk.loc[spouse_mask, col].dropna().map(_norm):
                if n:
                    out.append(n)
        # Children: keyed by baptism year
        child_mask = chunk['bap_year'].between(1500, max_year, inclusive='both')
        for n in chunk.loc[child_mask, 'children_Name'].dropna().map(_norm):
            if n:
                out.append(n)
    return out


def load_edinburgh(path, max_year, top_n=100):
    """Load Edinburgh top-N names per year. The file has 3,385 ranks/year
    including extremely rare names; capping at top_n keeps the well-attested
    vernacular pool without adding noise."""
    df = pd.read_excel(path, sheet_name='Sheet1')
    year_cols = [c for c in df.columns if isinstance(c, int) and c <= max_year]
    out = []
    pat = re.compile(r'^([A-Za-z\- ]+?)\s*\(')
    for col in year_cols:
        for cell in df[col].head(top_n).dropna():
            if not isinstance(cell, str):
                continue
            m = pat.match(cell.strip())
            if not m:
                continue
            n = _norm(m.group(1))
            if n:
                out.append(n)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-year', type=int, default=2100,
                    help='Year cutoff on register data. Default 2100 = include all '
                         'available names. The cutoff has small effect (deltas <2pt) '
                         'on pre-1900 fiction but understates realistic-pct by 3-5pt '
                         'for 20C texts when capped at 1900.')
    ap.add_argument('--out', default=os.path.join(NAMES_DIR, 'names.txt'))
    ap.add_argument('--names-dir', default=NAMES_DIR)
    args = ap.parse_args()

    sources = [
        ('ncumb.txt', load_ncumb),
        ('26ParishesReconstitutions_ALL_DATA.txt', load_26parishes),
        ('Edinburgh.xlsx', load_edinburgh),
    ]
    all_names = set()
    per_source_unique = {}
    for fname, loader in sources:
        path = os.path.join(args.names_dir, fname)
        if not os.path.exists(path):
            print(f"[skip] {fname} not found at {path}", file=sys.stderr)
            continue
        print(f"[load] {fname}", file=sys.stderr)
        names = loader(path, args.max_year)
        unique = set(names)
        per_source_unique[fname] = (len(names), len(unique))
        all_names.update(unique)
        print(f"       {len(names):,} occurrences -> {len(unique):,} unique", file=sys.stderr)

    print(f"\nTotal unique names <= {args.max_year}: {len(all_names):,}", file=sys.stderr)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        for name in sorted(all_names):
            f.write(name + '\n')
    print(f"Wrote {args.out}", file=sys.stderr)


if __name__ == '__main__':
    main()
