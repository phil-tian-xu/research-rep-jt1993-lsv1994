"""Offline checks for cache reuse, provenance, streaming, and connection ownership."""

import json
import os
import tempfile
import unittest
from contextlib import chdir
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

from utils.jt1993_data_download import download_jt1993_data, get_wrds_connection
from utils.path_info import root_path


def sample_frame(dataset, start="1962-07", end="1989-12"):
    """Small synthetic data spanning each requested calendar month."""
    dates = pd.period_range(start, end, freq="M").to_timestamp("M")
    if dataset == "market_returns":
        return pd.DataFrame({"month": dates, "vwretd": 0.01, "ewretd": 0.02})
    if dataset == "daily_returns":
        return pd.DataFrame({
            "permno": 10001, "date": dates, "ret": 0.01, "exchcd": 1, "shrcd": 10,
        })
    if dataset == "monthly_returns":
        return pd.DataFrame({
            "permno": 10001, "month": dates, "n_daily_rows": 20,
            "n_valid_ret": 20, "ret": 0.01,
        })
    return pd.DataFrame({
        "permno": 10001, "month": dates, "prc": 25.0, "shrout": 1000.0,
        "exchcd": 1, "shrcd": 10,
    })


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(
            prefix=".test_jt1993_", dir=Path(root_path),
        )
        self.addCleanup(self.temporary.cleanup)
        self.cache_dir = Path(self.temporary.name)
        self.request = {
            "start": "1962-07-01", "end": "1989-12-31", "cache_dir": self.cache_dir,
        }

    def test_download_then_reuse_all_caches_without_authentication(self):
        names = ("daily_returns", "monthly_returns", "monthly_characteristics", "market_returns")
        connection = Mock()
        connection.raw_sql.side_effect = [iter([sample_frame(name)]) for name in names]

        with patch("utils.jt1993_data_download.get_wrds_connection", return_value=connection) as connect:
            paths = download_jt1993_data(**self.request, username="example_user")
            connect.assert_called_once_with("example_user")
        self.assertEqual(connection.raw_sql.call_count, 4)
        connection.close.assert_called_once_with()
        for name, path in paths.items():
            self.assertEqual(len(pd.read_parquet(path)), 330)
            metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["rows"], 330)
            self.assertEqual(metadata["params"]["start_date"], "1962-07-01")
            self.assertEqual(metadata["exchange_codes"], None if name == "market_returns" else [1, 2])

        with patch("utils.jt1993_data_download.get_wrds_connection") as connect:
            self.assertEqual(download_jt1993_data(**self.request), paths)
            connect.assert_not_called()

    def test_streams_multiple_chunks_and_keeps_caller_connection_open(self):
        frame = sample_frame("daily_returns")
        connection = Mock()
        connection.raw_sql.return_value = iter([frame.iloc[:100], frame.iloc[100:]])
        paths = download_jt1993_data(
            **self.request, datasets=("daily_returns",), connection=connection, chunk_size=100,
        )
        result = pd.read_parquet(paths["daily_returns"])
        self.assertEqual(len(result), len(frame))
        self.assertEqual(result["ret"].tolist(), frame["ret"].tolist())
        kwargs = connection.raw_sql.call_args.kwargs
        self.assertTrue(kwargs["return_iter"])
        self.assertEqual(kwargs["chunksize"], 100)
        connection.close.assert_not_called()

    def test_only_missing_dataset_is_queried(self):
        first = Mock()
        first.raw_sql.return_value = iter([sample_frame("market_returns")])
        download_jt1993_data(**self.request, datasets=("market_returns",), connection=first)
        second = Mock()
        second.raw_sql.return_value = iter([sample_frame("monthly_returns")])
        download_jt1993_data(
            **self.request, datasets=("market_returns", "monthly_returns"), connection=second,
        )
        second.raw_sql.assert_called_once()
        self.assertIn("from crsp.dsf", second.raw_sql.call_args.args[0])

    def test_mismatched_cache_is_rejected_before_connecting(self):
        connection = Mock()
        connection.raw_sql.return_value = iter([sample_frame("market_returns")])
        paths = download_jt1993_data(**self.request, datasets=("market_returns",), connection=connection)
        sidecar = paths["market_returns"].with_suffix(".json")
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        metadata["signature"] = "different query or universe"
        sidecar.write_text(json.dumps(metadata), encoding="utf-8")
        with patch("utils.jt1993_data_download.get_wrds_connection") as connect:
            with self.assertRaisesRegex(ValueError, "specification differs"):
                download_jt1993_data(**self.request, datasets=("market_returns",))
            connect.assert_not_called()

    def test_unlabelled_cache_is_not_silently_reused(self):
        sample_frame("monthly_returns").to_parquet(self.cache_dir / "monthly_returns.parquet")
        with patch("utils.jt1993_data_download.get_wrds_connection") as connect:
            with self.assertRaisesRegex(ValueError, "Missing metadata"):
                download_jt1993_data(**self.request, datasets=("monthly_returns",))
            connect.assert_not_called()

    def test_failed_refresh_preserves_previous_cache_and_closes_connection(self):
        first = Mock()
        first.raw_sql.return_value = iter([sample_frame("market_returns")])
        paths = download_jt1993_data(**self.request, datasets=("market_returns",), connection=first)
        path = paths["market_returns"]
        before = path.read_bytes()
        metadata_before = path.with_suffix(".json").read_bytes()

        def interrupted_query():
            yield sample_frame("market_returns").iloc[:20]
            raise RuntimeError("interrupted download")

        failed = Mock()
        failed.raw_sql.return_value = interrupted_query()
        with patch("utils.jt1993_data_download.get_wrds_connection", return_value=failed):
            with self.assertRaisesRegex(RuntimeError, "interrupted download"):
                download_jt1993_data(**self.request, datasets=("market_returns",), refresh=True)
        failed.close.assert_called_once_with()
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(path.with_suffix(".json").read_bytes(), metadata_before)
        self.assertEqual(sorted(p.name for p in self.cache_dir.iterdir()), ["market_returns.json", "market_returns.parquet"])

    def test_duplicate_keys_across_chunks_are_rejected(self):
        frame = sample_frame("daily_returns")
        connection = Mock()
        connection.raw_sql.return_value = iter([frame.iloc[:100], frame.iloc[99:]])
        with self.assertRaisesRegex(ValueError, "across chunks"):
            download_jt1993_data(**self.request, datasets=("daily_returns",), connection=connection)
        self.assertEqual(list(self.cache_dir.iterdir()), [])

    def test_missing_month_is_rejected_without_publishing_cache(self):
        connection = Mock()
        connection.raw_sql.return_value = iter([sample_frame("monthly_returns").iloc[:-1]])
        with self.assertRaisesRegex(ValueError, "Missing calendar months"):
            download_jt1993_data(**self.request, datasets=("monthly_returns",), connection=connection)
        self.assertEqual(list(self.cache_dir.iterdir()), [])

    def test_empty_query_is_rejected_without_publishing_cache(self):
        connection = Mock()
        connection.raw_sql.return_value = iter([])
        with self.assertRaisesRegex(ValueError, "no observations"):
            download_jt1993_data(**self.request, datasets=("daily_returns",), connection=connection)
        self.assertEqual(list(self.cache_dir.iterdir()), [])

    def test_invalid_request_fails_before_authentication(self):
        with patch("utils.jt1993_data_download.get_wrds_connection") as connect:
            for kwargs in [
                {"datasets": ("unknown",)}, {"datasets": ()},
                {"datasets": ("daily_returns", "daily_returns")}, {"chunk_size": 0},
            ]:
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    download_jt1993_data(**self.request, **kwargs)
            connect.assert_not_called()

    def test_requested_dates_reach_query_metadata_and_default_cache_directory(self):
        connection = Mock()
        connection.raw_sql.side_effect = [
            iter([sample_frame("market_returns", "2000-01", "2000-02")]),
            iter([sample_frame("market_returns", "2001-01", "2001-02")]),
        ]
        with patch("utils.jt1993_data_download.DEFAULT_CACHE_DIR", self.cache_dir):
            first = download_jt1993_data(
                start="2000-01-01", end="2000-02-29", universe="nyse_amex",
                datasets=("market_returns",), connection=connection,
            )["market_returns"]
            second = download_jt1993_data(
                start="2001-01-01", end="2001-02-28", universe="nyse_amex",
                datasets=("market_returns",), connection=connection,
            )["market_returns"]
        self.assertEqual(first.parent.name, "nyse_amex_20000101_20000229")
        self.assertEqual(second.parent.name, "nyse_amex_20010101_20010228")
        params = connection.raw_sql.call_args_list[0].kwargs["params"]
        self.assertEqual(params, {"start_date": "2000-01-01", "end_date": "2000-02-29"})
        metadata = json.loads(first.with_suffix(".json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["params"], params)
        self.assertEqual(metadata["universe"], "nyse_amex")
        self.assertEqual(metadata["rows"], 2)

    def test_changed_dates_cannot_reuse_an_explicit_cache_directory(self):
        connection = Mock()
        connection.raw_sql.return_value = iter([sample_frame("market_returns")])
        download_jt1993_data(**self.request, datasets=("market_returns",), connection=connection)
        changed_request = {**self.request, "start": "1963-01-01"}
        with patch("utils.jt1993_data_download.get_wrds_connection") as connect:
            with self.assertRaisesRegex(ValueError, "specification differs"):
                download_jt1993_data(**changed_request, datasets=("market_returns",))
            connect.assert_not_called()

    def test_relative_cache_uses_project_root_instead_of_working_directory(self):
        connection = Mock()
        connection.raw_sql.return_value = iter([sample_frame("market_returns")])
        other_directory = self.cache_dir / "other_working_directory"
        other_directory.mkdir()
        request = {**self.request, "cache_dir": "relative_cache"}
        with patch("utils.jt1993_data_download.root_path", self.cache_dir):
            with chdir(other_directory):
                paths = download_jt1993_data(
                    **request, datasets=("market_returns",), connection=connection,
                )
        self.assertEqual(
            paths["market_returns"], self.cache_dir / "relative_cache" / "market_returns.parquet",
        )
        self.assertTrue(paths["market_returns"].exists())
        self.assertFalse((other_directory / "relative_cache").exists())

    def test_invalid_dates_and_partial_monthly_windows_fail_before_authentication(self):
        with patch("utils.jt1993_data_download.get_wrds_connection") as connect:
            for changes in [
                {"start": "invalid"}, {"end": "1989-02-30"},
                {"start": "1990-01-01"}, {"start": "1962-07-02"},
                {"end": "1989-12-30"},
            ]:
                request = {**self.request, **changes}
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    download_jt1993_data(**request)
            connect.assert_not_called()

    def test_unknown_universe_fails_before_authentication(self):
        with patch("utils.jt1993_data_download.get_wrds_connection") as connect:
            for universe in ["unknown", "", "nasdaq_only"]:
                with self.subTest(universe=universe), self.assertRaisesRegex(ValueError, "universe"):
                    download_jt1993_data(**self.request, universe=universe)
            connect.assert_not_called()

    def test_exchange_universes_filter_all_stock_datasets_and_cache_separately(self):
        names = ("daily_returns", "monthly_returns", "monthly_characteristics", "market_returns")
        for universe, codes in [("nyse", [1]), ("nyse_amex", [1, 2]), ("nyse_amex_nasdaq", [1, 2, 3])]:
            with self.subTest(universe=universe):
                connection = Mock()
                connection.raw_sql.side_effect = [iter([sample_frame(name)]) for name in names]
                with patch("utils.jt1993_data_download.DEFAULT_CACHE_DIR", self.cache_dir):
                    paths = download_jt1993_data(
                        start="1962-07-01", end="1989-12-31", universe=universe, connection=connection,
                    )
                    for name, call in zip(names, connection.raw_sql.call_args_list):
                        metadata = json.loads(paths[name].with_suffix(".json").read_text(encoding="utf-8"))
                        self.assertEqual(metadata["universe"], universe)
                        self.assertEqual(metadata["exchange_codes"], None if name == "market_returns" else codes)
                        self.assertEqual(paths[name].parent.name, f"{universe}_19620701_19891231")
                        if name != "market_returns":
                            self.assertIn("b.exchcd in (" + ", ".join(map(str, codes)) + ")", call.args[0])
                            self.assertIn("b.shrcd in (10, 11)", call.args[0])
                            self.assertIn("b.namedt <= a.date", call.args[0])
                    with patch("utils.jt1993_data_download.get_wrds_connection") as connect:
                        self.assertEqual(download_jt1993_data(
                            start="1962-07-01", end="1989-12-31", universe=universe,
                        ), paths)
                        connect.assert_not_called()

    def test_ciz_normalizes_schema_and_keeps_source_and_universe_in_provenance(self):
        names = ("daily_returns", "monthly_returns", "monthly_characteristics", "market_returns")
        for universe, exchanges in [("nyse", "'N'"), ("nyse_amex_nasdaq", "'N', 'A', 'Q'")]:
            with self.subTest(universe=universe):
                connection = Mock()
                connection.raw_sql.side_effect = [iter([sample_frame(name, "2025-01", "2025-02")]) for name in names]
                with patch("utils.jt1993_data_download.DEFAULT_CACHE_DIR", self.cache_dir):
                    paths = download_jt1993_data(
                        start="2025-01-01", end="2025-02-28", universe=universe,
                        data_source="ciz", connection=connection,
                    )
                    sqls = [call.args[0] for call in connection.raw_sql.call_args_list]
                    self.assertIn("crsp.dsf_v2", sqls[0])
                    self.assertIn("a.dlyret as ret", sqls[0])
                    self.assertIn("a.dlyret as ret", sqls[1])
                    self.assertIn("when ret = -1 then 1", sqls[1])
                    self.assertIn("crsp.msf_v2", sqls[2])
                    self.assertIn("a.mthprc as prc, a.shrout", sqls[2])
                    for sql in sqls[:3]:
                        self.assertIn(f"a.primaryexch in ({exchanges})", sql)
                        for condition in ("sharetype = 'NS'", "securitytype = 'EQTY'", "securitysubtype = 'COM'",
                                          "usincflg = 'Y'", "issuertype in ('ACOR', 'CORP')",
                                          "conditionaltype = 'RW'", "tradingstatusflg = 'A'"):
                            self.assertIn(condition, sql)
                    self.assertIn("crsp.indmthseriesdata", sqls[3])
                    self.assertIn("1000200", sqls[3])
                    self.assertIn("1000201", sqls[3])
                    for path in paths.values():
                        metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
                        self.assertEqual(metadata["data_source"], "ciz")
                        self.assertEqual(metadata["universe"], universe)
                        self.assertTrue(metadata["delisting_returns_in_source"])
                        self.assertEqual(metadata["shares_outstanding_units"], "thousands")
                        self.assertEqual(path.parent.name, f"{universe}_ciz_20250101_20250228")
                    self.assertTrue(pd.read_parquet(paths["monthly_characteristics"])["shrout"].eq(1000).all())
                    self.assertEqual(set(pd.read_parquet(paths["monthly_returns"]).columns),
                                     {"permno", "month", "n_daily_rows", "n_valid_ret", "ret"})
                    with patch("utils.jt1993_data_download.get_wrds_connection") as connect:
                        self.assertEqual(download_jt1993_data(
                            start="2025-01-01", end="2025-02-28", universe=universe, data_source="ciz",
                        ), paths)
                        connect.assert_not_called()

    def test_different_universe_or_source_cannot_reuse_an_explicit_cache_directory(self):
        connection = Mock()
        connection.raw_sql.return_value = iter([sample_frame("market_returns")])
        download_jt1993_data(**self.request, datasets=("market_returns",), connection=connection)
        with patch("utils.jt1993_data_download.get_wrds_connection") as connect:
            for changes in ({"universe": "nyse"}, {"data_source": "ciz"}):
                with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "specification differs"):
                    download_jt1993_data(**self.request, datasets=("market_returns",), **changes)
            connect.assert_not_called()

    def test_legacy_recent_dates_and_unknown_source_fail_before_authentication(self):
        with patch("utils.jt1993_data_download.get_wrds_connection") as connect:
            with self.assertRaisesRegex(ValueError, "Use data_source='ciz'"):
                download_jt1993_data(start="1960-01-01", end="2025-12-31", cache_dir=self.cache_dir)
            with self.assertRaisesRegex(ValueError, "data_source"):
                download_jt1993_data(**self.request, data_source="unknown")
            connect.assert_not_called()

    def test_default_cache_directories_distinguish_legacy_and_ciz(self):
        connection = Mock()
        connection.raw_sql.side_effect = [iter([sample_frame("market_returns")]) for _ in range(2)]
        with patch("utils.jt1993_data_download.DEFAULT_CACHE_DIR", self.cache_dir):
            paths = [download_jt1993_data(
                start="1962-07-01", end="1989-12-31", universe="nyse",
                data_source=source, datasets=("market_returns",), connection=connection,
            )["market_returns"] for source in ("legacy", "ciz")]
        self.assertNotEqual(paths[0], paths[1])
        self.assertEqual(paths[0].parent.name, "nyse_19620701_19891231")
        self.assertEqual(paths[1].parent.name, "nyse_ciz_19620701_19891231")

    def test_daily_only_request_accepts_partial_month_dates(self):
        connection = Mock()
        connection.raw_sql.return_value = iter([pd.DataFrame({
            "permno": 10001, "date": pd.date_range("2000-01-03", "2000-01-04"),
            "ret": 0.01, "exchcd": 1, "shrcd": 10,
        })])
        paths = download_jt1993_data(
            start="2000-01-03", end="2000-01-04", cache_dir=self.cache_dir,
            datasets=("daily_returns",), connection=connection,
        )
        self.assertEqual(len(pd.read_parquet(paths["daily_returns"])), 2)
        self.assertEqual(
            connection.raw_sql.call_args.kwargs["params"],
            {"start_date": "2000-01-03", "end_date": "2000-01-04"},
        )

    def test_returned_dates_outside_the_requested_window_are_rejected(self):
        connection = Mock()
        connection.raw_sql.return_value = iter([sample_frame("market_returns")])
        with self.assertRaisesRegex(ValueError, "outside the requested sample"):
            download_jt1993_data(
                start="2000-01-01", end="2000-01-31", cache_dir=self.cache_dir,
                datasets=("market_returns",), connection=connection,
            )
        self.assertEqual(list(self.cache_dir.iterdir()), [])


class ConnectionTests(unittest.TestCase):
    def test_explicit_username_is_passed_without_a_password(self):
        with patch.dict(os.environ, {"WRDS_USERNAME": "environment_user"}):
            with patch("wrds.Connection") as connect:
                self.assertIs(get_wrds_connection("explicit_user"), connect.return_value)
                connect.assert_called_once_with(wrds_username="explicit_user")

    def test_environment_username_is_used(self):
        with patch.dict(os.environ, {"WRDS_USERNAME": "environment_user"}):
            with patch("wrds.Connection") as connect:
                get_wrds_connection()
                connect.assert_called_once_with(wrds_username="environment_user")

    def test_missing_username_fails_without_an_interactive_connection(self):
        with patch.dict(os.environ, {"WRDS_USERNAME": ""}):
            with patch("wrds.Connection") as connect:
                with self.assertRaisesRegex(ValueError, "WRDS_USERNAME"):
                    get_wrds_connection()
                connect.assert_not_called()


if __name__ == "__main__":
    unittest.main()
