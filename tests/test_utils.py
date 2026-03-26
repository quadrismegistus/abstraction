import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from abstraction.utils import (
    zfy, read_df, save_df, get_avgs_df, get_slices, cleanhtml,
    parse_json_str,
)


class TestZfy:
    def test_basic(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        z = zfy(s)
        assert abs(z.mean()) < 1e-10
        assert abs(z.std(ddof=0) - 1.0) < 1e-10

    def test_drops_nan(self):
        s = pd.Series([1.0, np.nan, 3.0])
        z = zfy(s)
        assert len(z) == 2

    def test_coerces_strings(self):
        s = pd.Series([1.0, "bad", 3.0])
        z = zfy(s)
        assert len(z) == 2


class TestReadSaveDf:
    def test_csv_roundtrip(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            save_df(df, path)
            df2 = read_df(path)
            assert list(df2["a"]) == [1, 2]
        finally:
            os.remove(path)

    def test_pkl_roundtrip(self):
        df = pd.DataFrame({"x": [10, 20]})
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            save_df(df, path)
            df2 = read_df(path)
            assert list(df2["x"]) == [10, 20]
        finally:
            os.remove(path)

    def test_feather_roundtrip(self):
        df = pd.DataFrame({"z": [5, 6]})
        with tempfile.NamedTemporaryFile(suffix=".ft", delete=False) as f:
            path = f.name
        try:
            save_df(df, path)
            df2 = read_df(path)
            assert list(df2["z"]) == [5, 6]
        finally:
            os.remove(path)

    def test_unknown_format_raises(self):
        with pytest.raises(ValueError):
            read_df("data.xyz")
        with pytest.raises(ValueError):
            save_df(pd.DataFrame(), "/tmp/test_abstraction_data.xyz")


class TestGetAvgsDf:
    def test_basic_aggregation(self):
        df = pd.DataFrame({
            "genre": ["Novel"] * 4 + ["Epic"] * 4,
            "corpus": ["A"] * 8,
            "decade": [1800] * 4 + [1900] * 4,
            "score": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "num_texts": [1] * 8,
        })
        result = get_avgs_df(df, gby=["genre", "decade"], y="score")
        assert "mean" in result.columns
        assert "stderr" in result.columns
        assert "count" in result.columns
        assert len(result) == 2  # Novel/1800, Epic/1900

    def test_standardizes(self):
        df = pd.DataFrame({
            "genre": ["A"] * 10,
            "decade": [2000] * 10,
            "score": list(range(10)),
            "num_texts": [1] * 10,
        })
        result = get_avgs_df(df, gby=["genre", "decade"], y="score")
        # after standardization, mean of one group is ~0
        assert abs(result["mean"].iloc[0]) < 1e-10

    def test_min_texts_filter(self):
        df = pd.DataFrame({
            "genre": ["A"] * 3 + ["B"] * 1,
            "decade": [2000] * 4,
            "score": [1.0, 2.0, 3.0, 4.0],
            "num_texts": [1, 1, 1, 1],
        })
        result = get_avgs_df(df, gby=["genre"], y="score", min_texts=2)
        genres = result.index.get_level_values("genre").unique()
        assert "A" in genres
        assert "B" not in genres


class TestGetSlices:
    def test_by_length(self):
        slices = get_slices([1, 2, 3, 4, 5], slice_length=2)
        assert slices == [[1, 2], [3, 4], [5]]

    def test_by_length_no_runts(self):
        slices = get_slices([1, 2, 3, 4, 5], slice_length=2, keep_runts=False)
        assert slices == [[1, 2], [3, 4]]

    def test_by_count(self):
        slices = get_slices(list(range(10)), num_slices=3)
        assert len(slices) >= 3

    def test_no_args(self):
        assert get_slices([1, 2, 3]) == [[1, 2, 3]]


class TestCleanhtml:
    def test_strips_tags(self):
        assert cleanhtml("<b>hello</b>") == "hello"
        assert cleanhtml("<i><b>word</b></i> and text") == "word and text"

    def test_no_tags(self):
        assert cleanhtml("plain text") == "plain text"


class TestParseJsonStr:
    def test_plain_json(self):
        assert parse_json_str('{"key": "value"}') == {"key": "value"}

    def test_fenced_json(self):
        result = parse_json_str('```json\n{"a": 1}\n```')
        assert result == {"a": 1}

    def test_bad_json_returns_none(self):
        assert parse_json_str("not json at all") is None
