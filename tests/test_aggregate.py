"""
Tests for abstraction/aggregate.py.

Most of the logic here is SQL-string construction and control flow around a
ClickHouse client, so the bulk of these tests use a FakeCH stand-in that
records the calls it receives (`command`, `query`, `query_df`, `insert`,
`close`) instead of talking to a real server — this covers the audit
backlog fixes for the LIKE-wildcard bug (#4), the Memory-table leak (#5),
and the INNER JOIN rep-accounting line (#6) without needing ClickHouse up.

A few tests marked `@pytest.mark.integration` DO exercise the real CH
server (localhost:8123) plus LLTK's `arc_fiction` corpus, since both are
available in this environment; they're the closest thing to an end-to-end
check that the SQL actually runs.
"""

import pandas as pd
import pytest

import abstraction.aggregate as aggregate


# ---------------------------------------------------------------------------
# FakeCH: minimal stand-in for a clickhouse_connect client
# ---------------------------------------------------------------------------


class FakeCH:
    def __init__(self, query_result_rows=None, query_df_result=None,
                 raise_on_insert=False, raise_on_command=False):
        self.commands = []
        self.queries = []
        self.query_dfs = []
        self.inserts = []
        self.closed = False
        self._query_result_rows = query_result_rows or []
        self._query_df_result = (
            query_df_result if query_df_result is not None else pd.DataFrame()
        )
        self._raise_on_insert = raise_on_insert
        self._raise_on_command = raise_on_command

    class _Result:
        def __init__(self, rows):
            self.result_rows = rows

    def command(self, sql):
        self.commands.append(sql)
        if self._raise_on_command:
            raise RuntimeError("command boom")

    def query(self, sql, parameters=None):
        self.queries.append((sql, parameters))
        return FakeCH._Result(self._query_result_rows)

    def insert(self, table, data, column_names=None, database=None):
        self.inserts.append((table, data, column_names, database))
        if self._raise_on_insert:
            raise RuntimeError("insert boom")

    def query_df(self, sql, parameters=None):
        self.query_dfs.append((sql, parameters))
        return self._query_df_result

    def close(self):
        self.closed = True


# ---------------------------------------------------------------------------
# _backtick
# ---------------------------------------------------------------------------


class TestBacktick:
    def test_wraps_plain_name(self):
        assert aggregate._backtick("Abs-Conc.Median.median") == "`Abs-Conc.Median.median`"

    def test_escapes_embedded_backtick(self):
        assert aggregate._backtick("weird`col") == "`weird``col`"


# ---------------------------------------------------------------------------
# get_corpus_scores: startsWith, not LIKE (audit backlog #4)
# ---------------------------------------------------------------------------


class TestGetCorpusScoresQueryConstruction:
    def _fake(self, corpus_ids):
        return FakeCH(
            query_result_rows=[("Abs-Conc.Median.median",)],
            query_df_result=pd.DataFrame(
                {"_id": corpus_ids, "Abs-Conc.Median.median": [0.1] * len(corpus_ids)}
            ),
        )

    def test_uses_startswith_not_like(self):
        fake = self._fake(["_ecco_tcp/1"])
        aggregate.get_corpus_scores("ecco_tcp", lang="en", ch_client=fake)
        sql, params = fake.query_dfs[-1]
        assert "startsWith(s._id, %(prefix)s)" in sql
        assert "LIKE" not in sql
        assert params == {"prefix": "_ecco_tcp/", "lang": "en"}

    def test_prefix_is_literal_not_a_like_pattern(self):
        """The old `f"_{corpus}/%"` LIKE pattern needed a trailing `%`
        wildcard; startsWith takes a literal prefix with no wildcard char."""
        fake = self._fake(["_gallica_literary_fictions/1"])
        aggregate.get_corpus_scores("gallica_literary_fictions", lang="fr", ch_client=fake)
        _sql, params = fake.query_dfs[-1]
        assert params["prefix"] == "_gallica_literary_fictions/"
        assert "%" not in params["prefix"]

    def test_corpus_name_with_underscore_not_treated_as_wildcard(self):
        """Regression for the actual bug: LIKE's `_` is a single-char
        wildcard, so `LIKE '_ecco_tcp/%'` would also match e.g.
        `Xecco_tcp/...` or `_eccoXtcp/...`. startsWith is a literal
        comparison, so the parameter alone (not the SQL text) fully
        determines matching — no escaping needed even though "ecco_tcp"
        itself contains a literal underscore."""
        fake = self._fake(["_ecco_tcp/1", "_ecco_tcp/2"])
        df = aggregate.get_corpus_scores("ecco_tcp", lang="en", ch_client=fake)
        assert len(df) == 2
        _sql, params = fake.query_dfs[-1]
        assert params["prefix"] == "_ecco_tcp/"

    def test_invalid_lang_raises_before_any_ch_call(self):
        fake = FakeCH()
        with pytest.raises(ValueError, match="lang must be one of"):
            aggregate.get_corpus_scores("ecco_tcp", lang="xx", ch_client=fake)
        assert fake.queries == []
        assert fake.query_dfs == []

    def test_external_client_not_closed(self):
        fake = self._fake(["_ecco_tcp/1"])
        aggregate.get_corpus_scores("ecco_tcp", lang="en", ch_client=fake)
        assert fake.closed is False


# ---------------------------------------------------------------------------
# get_arc_scores: Memory-table create/insert lifecycle (audit backlog #5)
# ---------------------------------------------------------------------------


class TestGetArcScoresTempTableLifecycle:
    def _patch_reps(self, monkeypatch, rep_ids):
        monkeypatch.setattr(aggregate, "_load_arc_reps", lambda arc: list(rep_ids))

    def _created_table_name(self, fake):
        for cmd in fake.commands:
            if cmd.startswith("CREATE TABLE"):
                # "CREATE TABLE abstraction._arc_reps_xxxx (`_id` String) ..."
                return cmd.split()[2]
        return None

    def test_happy_path_drops_table_on_success(self, monkeypatch):
        self._patch_reps(monkeypatch, ["_c/1", "_c/2"])
        fake = FakeCH(
            query_result_rows=[("Abs-Conc.Median.median",)],
            query_df_result=pd.DataFrame(
                {"_id": ["_c/1", "_c/2"], "Abs-Conc.Median.median": [0.1, 0.2]}
            ),
        )
        aggregate.get_arc_scores("arc_fiction", lang="en", dedup="rep_only", ch_client=fake)
        name = self._created_table_name(fake)
        assert name is not None
        drop_cmds = [c for c in fake.commands if c.startswith("DROP TABLE")]
        assert len(drop_cmds) == 1
        assert name in drop_cmds[0]

    def test_insert_failure_after_create_still_drops_table(self, monkeypatch):
        """Regression for the leak: previously `_insert_temp_reps` only
        returned the table name *after* a successful `ch.insert`, so if
        CREATE TABLE succeeded but insert raised, the caller's try/finally
        (which started only after the whole call returned) never ran and
        the Memory table leaked for the life of the CH session."""
        self._patch_reps(monkeypatch, ["_c/1", "_c/2"])
        fake = FakeCH(
            query_result_rows=[("Abs-Conc.Median.median",)],
            raise_on_insert=True,
        )
        with pytest.raises(RuntimeError, match="insert boom"):
            aggregate.get_arc_scores("arc_fiction", lang="en", ch_client=fake)
        name = self._created_table_name(fake)
        assert name is not None, "CREATE TABLE should have been issued"
        drop_cmds = [c for c in fake.commands if c.startswith("DROP TABLE")]
        assert len(drop_cmds) == 1
        assert name in drop_cmds[0]

    def test_create_failure_does_not_error_in_cleanup(self, monkeypatch):
        """If CREATE TABLE itself fails, cleanup (DROP TABLE IF EXISTS) is a
        no-op-safe call on a name that was never created — should not raise
        a second exception masking the original one."""
        self._patch_reps(monkeypatch, ["_c/1"])
        fake = FakeCH(
            query_result_rows=[("Abs-Conc.Median.median",)],
            raise_on_command=True,
        )
        with pytest.raises(RuntimeError, match="command boom"):
            aggregate.get_arc_scores("arc_fiction", lang="en", ch_client=fake)


# ---------------------------------------------------------------------------
# _report_rep_coverage / INNER JOIN accounting (audit backlog #6)
# ---------------------------------------------------------------------------


class TestReportRepCoverage:
    def test_no_output_when_counts_match(self, capsys):
        df = pd.DataFrame({"_id": ["a", "b"]})
        aggregate._report_rep_coverage(["a", "b"], df)
        assert capsys.readouterr().out == ""

    def test_prints_when_reps_dropped(self, capsys):
        df = pd.DataFrame({"_id": ["a"]})
        aggregate._report_rep_coverage(["a", "b", "c"], df)
        out = capsys.readouterr().out
        assert "3 reps requested" in out
        assert "1" in out and "returned" in out
        assert "2 dropped" in out

    def test_empty_result_df(self, capsys):
        df = pd.DataFrame({"_id": []})
        aggregate._report_rep_coverage(["a"], df)
        out = capsys.readouterr().out
        assert "1 reps requested, 0 returned" in out


class TestGetArcScoresAccountingIntegratedWithFake:
    def _patch_reps(self, monkeypatch, rep_ids):
        monkeypatch.setattr(aggregate, "_load_arc_reps", lambda arc: list(rep_ids))

    def test_rep_only_prints_when_join_drops_reps(self, monkeypatch, capsys):
        self._patch_reps(monkeypatch, ["_c/1", "_c/2", "_c/3"])
        fake = FakeCH(
            query_result_rows=[("Abs-Conc.Median.median",)],
            query_df_result=pd.DataFrame(
                {"_id": ["_c/1"], "Abs-Conc.Median.median": [0.2]}
            ),
        )
        df = aggregate.get_arc_scores(
            "arc_fiction", lang="en", dedup="rep_only", ch_client=fake
        )
        assert len(df) == 1
        out = capsys.readouterr().out
        assert "3 reps requested, 1 returned" in out

    def test_within_lang_group_silent_when_nothing_dropped(self, monkeypatch, capsys):
        self._patch_reps(monkeypatch, ["_c/1", "_c/2"])
        fake = FakeCH(
            query_result_rows=[("Abs-Conc.Median.median",)],
            query_df_result=pd.DataFrame(
                {
                    "_id": ["_c/1", "_c/2"],
                    "_n_versions": [1, 1],
                    "Abs-Conc.Median.median": [0.1, 0.2],
                }
            ),
        )
        aggregate.get_arc_scores(
            "arc_fiction", lang="en", dedup="within_lang_group", ch_client=fake
        )
        assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# Live-ClickHouse smoke tests (CH is up in this environment; these should
# complete in well under 2 minutes)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestGetArcScoresLiveCH:
    def test_arc_fiction_small_score_cols_smoke(self):
        df = aggregate.get_arc_scores(
            "arc_fiction",
            lang="en",
            score_cols=["Abs-Conc.Median.median"],
            dedup="within_lang_group",
        )
        assert isinstance(df, pd.DataFrame)
        assert "_id" in df.columns
        assert "Abs-Conc.Median.median" in df.columns
        assert "_n_versions" in df.columns
        assert len(df) > 1000


@pytest.mark.integration
class TestGetCorpusScoresLiveCH:
    def test_canon_fiction_prefix_matches_only_that_corpus(self):
        """`canon_fiction` contains an embedded underscore — exactly the
        character LIKE treats as a wildcard — so this is the real-world
        case the #4 fix targets. All returned _ids must genuinely belong
        to canon_fiction."""
        df = aggregate.get_corpus_scores(
            "canon_fiction", lang="en", score_cols=["Abs-Conc.Median.median"]
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 100
        assert all(_id.startswith("_canon_fiction/") for _id in df["_id"])
