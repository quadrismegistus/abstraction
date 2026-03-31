"""Text tokenization and spelling modernization."""

import os
import re

from .config import PATH_STOPWORDS, PATH_NAMES, PATH_SPELLING_D

# Character replacements for historical/XML text
_REPLACEMENTS = {
    "&hyphen;": "-",
    "&sblank;": "--",
    "&mdash;": " -- ",
    "&ndash;": " - ",
    "&longs;": "s",
    "|": "",
    "&ldquo;": "\u201c",
    "&rdquo;": "\u201d",
    "&lsquo;": "\u2018",
    "&rsquo;": "\u2019",
    "&indent;": "     ",
    "&amp;": "&",
}
_REPLACEMENTS_UNICODE = {
    0x2013: " -- ",
    0x2014: " -- ",
    0x201C: '"',
    0x201D: '"',
    0x2018: "'",
    0x2019: "'",
    0x2026: " ... ",
    0x00A0: " ",
}

# Caches
_STOPWORDS = None
_NAMES = None
_SPELLING_D = None


def get_stopwords(use_nltk=True):
    """Return the set of stopwords (function words).

    If use_nltk=True (default), uses NLTK's 198 English stopwords.
    If False, uses the project's stopwords.txt file.
    """
    global _STOPWORDS
    if _STOPWORDS is None:
        if use_nltk:
            from nltk.corpus import stopwords as nltk_stops
            _STOPWORDS = set(nltk_stops.words("english"))
        else:
            _STOPWORDS = set()
            if os.path.exists(PATH_STOPWORDS):
                with open(PATH_STOPWORDS) as f:
                    _STOPWORDS = {w.strip().lower() for w in f if w.strip()}
    return _STOPWORDS


def get_names():
    """Return the set of proper names (for filtering contrast vector seeds)."""
    global _NAMES
    if _NAMES is None:
        _NAMES = set()
        if os.path.exists(PATH_NAMES):
            with open(PATH_NAMES) as f:
                _NAMES = {w.strip().lower() for w in f if w.strip()}
    return _NAMES


def get_stopwords_and_names():
    """Return stopwords + names combined (for contrast vector seed filtering)."""
    return get_stopwords() | get_names()


def get_spelling_modernizer():
    global _SPELLING_D
    if _SPELLING_D is None:
        if not os.path.exists(PATH_SPELLING_D):
            _SPELLING_D = {}
        else:
            with open(PATH_SPELLING_D) as f:
                _SPELLING_D = dict(
                    ln.strip().split("\t", 1)
                    for ln in f
                    if ln.strip() and "\t" in ln and not ln.startswith("#")
                )
    return _SPELLING_D


def tokenize_agnostic(txt):
    """Simple regex tokenizer that keeps punctuation as separate tokens."""
    return re.findall(r"[\w']+|[.,!?; \-\u2014\u2013\n]", txt)


def tokenize(txt, lower=True, modernize=False):
    """Tokenize text with optional lowercasing and spelling modernization."""
    for k, v in _REPLACEMENTS_UNICODE.items():
        txt = txt.replace(chr(k), v)
    for k, v in _REPLACEMENTS.items():
        txt = txt.replace(k, v)
    if lower:
        txt = txt.lower()
    if modernize:
        txt = _modernize_spelling(txt, get_spelling_modernizer())
    return tokenize_agnostic(txt)


def tokenize_sentences(txt):
    import nltk
    return nltk.sent_tokenize(txt)


def _modernize_spelling(txt, spelling_d):
    lines = []
    for ln in txt.split("\n"):
        tokens = []
        for tok in ln.split(" "):
            pre, word, post = _strip_punct(tok)
            word = spelling_d.get(word, word)
            tokens.append(pre + word + post)
        lines.append(" ".join(tokens))
    return "\n".join(lines)


def _strip_punct(token):
    """Strip leading/trailing non-alphanumeric characters from a token."""
    pre = ""
    post = ""
    while token and not token[0].isalnum():
        pre += token[0]
        token = token[1:]
    while token and not token[-1].isalnum():
        post = token[-1] + post
        token = token[:-1]
    return pre, token, post
