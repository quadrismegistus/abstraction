"""Shared pytest configuration for the abstraction test suite.

This file owns the norm-mocking contract (see docs/AUDIT-2026-07-04.md
sections 1.4 and 6), so that a future change to scoring's cache layout
breaks exactly one file instead of four.

The contract
------------
``abstraction.scoring.get_norm_dict(col, lang)`` memoizes {word: score}
dicts in the module-global ``scoring._NORM_DICTS`` keyed by the TUPLE
``(col, lang)`` — never by a bare string. Tests install fakes by writing
tuple keys into that cache (via ``install_fake_norms`` below) rather than
monkeypatching ``get_norm_dict`` itself, because other modules (e.g.
``abstraction.passages``) bind ``get_norm_dict`` through from-imports at
import time; patching the function attribute on ``abstraction.scoring``
would not reach those already-bound references, whereas every caller
routes through the one real function that consults ``_NORM_DICTS``.

``scoring._NORMS_ARRAYS_CACHE`` memoizes numpy arrays in a dict keyed by
the IDENTITY of the allnorms DataFrame (``id(frame)`` guarded by a
weakref), so distinct frames — fake or real, English or French — get
distinct entries and cannot cross-contaminate. It is still cleared
between tests (fresh empty dict) so no arrays built by one test outlive
it. Both caches are reset by the autouse ``_reset_norm_caches`` fixture.
"""

import importlib

import pandas as pd
import pytest

import abstraction.scoring as scoring

# NB: the abstraction/__init__.py re-exports a tokenize() FUNCTION that
# shadows the tokenize submodule as a package attribute, so
# `import abstraction.tokenize as m` (and monkeypatch's string-target form
# "abstraction.tokenize.X") would resolve to the function. Fetch the real
# module object explicitly.
tokenize_mod = importlib.import_module("abstraction.tokenize")

DEFAULT_NORM_COL = "Abs-Conc.Median.median"


def pytest_configure(config):
    # Markers are registered here rather than in an ini file because
    # pyproject.toml is owned by another workstream.
    config.addinivalue_line(
        "markers",
        "integration: requires local data files and corpora "
        "(deselect with -m 'not integration')",
    )
    config.addinivalue_line(
        "markers",
        "slow: slow test, e.g. browser-based rendering "
        "(deselect with -m 'not slow')",
    )


@pytest.fixture(autouse=True)
def _reset_norm_caches(request):
    """Clear scoring.py's module-global norm caches around every test.

    Clears BOTH ``_NORM_DICTS`` (the {(col, lang): {word: score}} cache that
    ``get_norm_dict`` actually consults) and ``_NORMS_ARRAYS_CACHE`` (the
    frame-identity-keyed numpy-array memo), before and after each test, so
    no test can observe entries — fake or real — left behind by another test.
    """
    if request.node.get_closest_marker("integration"):
        # Integration tests intentionally run against the real allnorms and
        # share its cache within the session (reloading the 2.6M-row pickle
        # per test would add ~15s each). Unit-test isolation is preserved
        # because non-integration tests still clear the caches on entry.
        yield
        return
    scoring._NORM_DICTS = {}
    scoring._NORMS_ARRAYS_CACHE = {}
    yield
    scoring._NORM_DICTS = {}
    scoring._NORMS_ARRAYS_CACHE = {}


@pytest.fixture
def install_fake_norms(monkeypatch):
    """Install a fake {word: score} norm dict under the real cache contract.

    Returns an installer::

        def test_x(install_fake_norms):
            install_fake_norms({"rock": 1.5, "virtue": -1.8})
            assert score_psg("rock") > 0

    Parameters of the installer:

    norm_dict : dict
        {word: z-score} mapping served by ``get_norm_dict(col, lang)``.
    col, lang : str
        Cache key parts; default to the production defaults of
        ``get_norm_dict``. Call the installer more than once to stub
        several (col, lang) combinations.
    spelling : dict or None
        Fake spelling-modernization map. Defaults to {} so unit tests never
        read the real spelling file. Patched via ``tokenize._SPELLING_D``
        (the cache inside ``get_spelling_modernizer``) so every from-import
        of the function sees it.
    """
    def _install(norm_dict, col=DEFAULT_NORM_COL, lang="en", spelling=None):
        monkeypatch.setitem(scoring._NORM_DICTS, (col, lang), dict(norm_dict))
        monkeypatch.setattr(
            tokenize_mod, "_SPELLING_D", dict(spelling) if spelling else {}
        )
        return norm_dict

    return _install


def make_fake_allnorms(**_kwargs):
    """Small allnorms DataFrame indexed by word, shared across test modules.

    Uses the production ``Abs-Conc.Median.*`` column naming so code that
    derives ``_pct_`` columns from the pattern behaves as in production.
    Accepts (and ignores) keyword arguments so it can stand in directly for
    ``abstraction.norms.get_allnorms``.
    """
    return pd.DataFrame(
        {
            "Abs-Conc.Median.median": {
                "rock": 1.5, "virtue": -1.8, "face": 0.3,
                "truth": -1.2, "man": 0.5,
            },
            "Abs-Conc.Median.orig": {
                "rock": 1.4, "virtue": -1.7, "face": 0.2,
                "truth": -1.1, "man": 0.4,
            },
        }
    )
