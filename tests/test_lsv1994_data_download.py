"""Offline checks for LSV raw coverage, identifier integrity, and safe caches."""

import io
import json
import sqlite3
import tempfile
import unittest
from contextlib import chdir, redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

from data_prepare_lsv1994 import data_prepare_lsv1994
from utils.lsv1994_data_download import download_lsv1994_data
from utils.path_info import root_path


def sample_frames():
    """Two months, two accounting years, and open/dated CCM boundaries."""
    return {
        "crsp_monthly": pd.DataFrame({
            "permno": [10001, 10001], "date": ["1967-12-29", "1968-01-31"],
            "ret": [None, -0.2], "prc": [-10.0, 8.0], "shrout": [1000, 1000],
            "shrcd": [10, 10], "exchcd": [1, 3],
        }),
        "compustat_annual": pd.DataFrame({
            "gvkey": ["000001", "000001"], "datadate": ["1962-12-31", "1963-12-31"],
            "fyear": [None, 1963], "fyr": [12, 12], "popsrc": ["D", "D"],
            "curcd": [None, "USD"], "ceq": [-1.0, 2.0], "seq": [None, 2.1],
            "pstk": [None, 0.1], "pstkrv": [None, 0.1], "pstkl": [None, 0.1],
            "txditc": [None, 0.0], "ib": [-2.0, 1.0], "dp": [0.2, 0.3],
            "sale": [5.0, 6.0], "dvc": [None, 0.1],
        }),
        "ccm_links": pd.DataFrame({
            "gvkey": ["000001", "000001"], "permno": [10001, 10001],
            "linkdt": [None, "1963-01-01"], "linkenddt": [None, "1989-12-31"],
            "linktype": ["LC", "LU"], "linkprim": [None, "P"], "liid": [None, "01C"],
        }),
        "market_returns": pd.DataFrame({
            "date": ["1967-12-29", "1968-01-31"], "vwretd": [0.01, 0.02],
            "ewretd": [0.02, -0.03],
        }),
    }


class DownloadTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=Path(root_path), prefix=".test_lsv1994_")
        self.addCleanup(temporary.cleanup)
        self.cache = Path(temporary.name)
        self.request = {
            "start": "1967-12-01", "end": "1968-01-31",
            "comp_start": "1962-01-01", "comp_end": "1963-12-31", "cache_dir": self.cache,
        }
        self.frames = sample_frames()
        self.enterContext(redirect_stdout(io.StringIO()))

    def test_stream_round_trip_preserves_raw_values_and_reuses_without_login(self):
        connection = Mock()
        connection.raw_sql.side_effect = [
            iter([frame.iloc[:1], frame.iloc[1:]]) for frame in self.frames.values()
        ]
        with patch("utils.lsv1994_data_download.get_wrds_connection", return_value=connection) as connect:
            paths = download_lsv1994_data(**self.request, username="test_user", chunk_size=1)
            connect.assert_called_once_with("test_user")
        connection.close.assert_called_once_with()
        for name, path in paths.items():
            self.assertEqual(len(pd.read_parquet(path)), 2)
            info = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
            self.assertEqual(info["rows"], 2)
            self.assertFalse(info["exchange_and_share_filters_applied"])
            self.assertEqual(info["formation_exchange_codes"], [1, 2])
            self.assertIn("sql", info)
        stock = pd.read_parquet(paths["crsp_monthly"])
        self.assertEqual(stock["exchcd"].tolist(), [1, 3])
        self.assertEqual(stock["prc"].tolist(), [-10.0, 8.0])
        self.assertTrue(pd.isna(stock.loc[0, "ret"]))
        self.assertEqual(stock.loc[0, "date"], pd.Timestamp("1967-12-29"))
        comp = pd.read_parquet(paths["compustat_annual"])
        self.assertEqual(comp["gvkey"].tolist(), ["000001", "000001"])
        self.assertEqual(comp["ib"].tolist(), [-2.0, 1.0])
        self.assertTrue(pd.isna(comp.loc[0, "fyear"]))
        links = pd.read_parquet(paths["ccm_links"])
        self.assertTrue(pd.isna(links.loc[0, "linkdt"]))
        self.assertTrue(pd.isna(links.loc[0, "linkenddt"]))
        self.assertEqual(links.loc[1, "liid"], "01C")
        with patch("utils.lsv1994_data_download.get_wrds_connection") as connect:
            self.assertEqual(download_lsv1994_data(**self.request), paths)
            connect.assert_not_called()

    def test_queries_follow_reference_and_use_independent_accounting_dates(self):
        connection = Mock()
        connection.raw_sql.side_effect = [iter([frame]) for frame in self.frames.values()]
        download_lsv1994_data(**self.request, connection=connection)
        stock, comp, links, market = connection.raw_sql.call_args_list
        self.assertIn("from crsp.msf", stock.args[0])
        self.assertIn("left join crsp.msenames", stock.args[0])
        self.assertNotIn("exchcd in", stock.args[0])
        self.assertNotIn("shrcd in", stock.args[0])
        self.assertEqual(stock.kwargs["params"]["start_date"], "1967-12-01")
        self.assertEqual(comp.kwargs["params"], {"start_date": "1962-01-01", "end_date": "1963-12-31"})
        for clause in ("indfmt = 'INDL'", "datafmt = 'STD'", "consol = 'C'"):
            self.assertIn(clause, comp.args[0])
        self.assertIn("dvc", comp.args[0])
        self.assertIn("linktype in ('LC', 'LU', 'LS')", links.args[0])
        self.assertEqual(links.kwargs["params"], {})
        self.assertIn("ewretd", market.args[0])
        self.assertTrue(all(call.kwargs["return_iter"] for call in connection.raw_sql.call_args_list))
        connection.close.assert_not_called()

    def test_failed_refresh_preserves_cached_data_and_metadata(self):
        connection = Mock()
        connection.raw_sql.return_value = iter([self.frames["crsp_monthly"]])
        paths = download_lsv1994_data(**self.request, datasets=("crsp_monthly",), connection=connection)
        path = paths["crsp_monthly"]
        before = path.read_bytes(), path.with_suffix(".json").read_bytes()

        def failed_query():
            yield self.frames["crsp_monthly"].iloc[:1]
            raise RuntimeError("connection interrupted")

        failed = Mock()
        failed.raw_sql.return_value = failed_query()
        with patch("utils.lsv1994_data_download.get_wrds_connection", return_value=failed):
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                download_lsv1994_data(**self.request, datasets=("crsp_monthly",), refresh=True)
        failed.close.assert_called_once_with()
        self.assertEqual((path.read_bytes(), path.with_suffix(".json").read_bytes()), before)

    def test_ciz_maps_formation_eligibility_without_removing_holding_returns(self):
        sql_database = sqlite3.connect(":memory:")
        self.addCleanup(sql_database.close)
        sql_database.execute("attach database ':memory:' as crsp")
        pd.DataFrame([
            [10001, "2025-01-31", 0.1, 10, 1000, "N", "RW", "A", "NS", "EQTY", "COM", "Y", "ACOR"],
            [10001, "2025-02-28", -0.2, 8, 1000, "Q", "RW", "A", "NS", "EQTY", "COM", "Y", "CORP"],
            [10002, "2025-01-31", 0.1, 10, 1000, "A", "RW", "A", "NS", "EQTY", "COM", "Y", "CORP"],
            [10002, "2025-02-28", -0.2, 8, 1000, "N", "RW", "H", "NS", "EQTY", "COM", "Y", "CORP"],
            [10003, "2025-01-31", 0.1, 10, 1000, "N", "NW", "A", "NS", "EQTY", "COM", "Y", "CORP"],
            [10003, "2025-02-28", -0.2, 8, 1000, "N", "RW", "A", "AD", "EQTY", "COM", "Y", "CORP"],
            [10004, "2025-01-31", 0.1, 10, 1000, "N", "RW", "A", "NS", "EQTY", "COM", "N", "CORP"],
            [10004, "2025-02-28", -0.2, 8, 1000, "N", "RW", "A", "NS", "EQTY", "COM", "Y", "REIT"],
        ], columns=[
            "permno", "mthcaldt", "mthret", "mthprc", "shrout", "primaryexch",
            "conditionaltype", "tradingstatusflg", "sharetype", "securitytype",
            "securitysubtype", "usincflg", "issuertype",
        ]).to_sql("stock_input", sql_database, index=False)
        sql_database.execute("create table crsp.msf_v2 as select * from stock_input")
        sql_database.execute(
            "create table crsp.indmthseriesdata (mthcaldt text, indno integer, mthtotret real)"
        )
        sql_database.executemany("insert into crsp.indmthseriesdata values (?, ?, ?)", [
            ("2025-01-31", 1000200, 0.01), ("2025-01-31", 1000201, 0.02),
            ("2025-02-28", 1000200, -0.03), ("2025-02-28", 1000201, -0.04),
        ])

        def query(sql, *, params, **kwargs):
            # SQLite executes the actual CASE/aggregation expressions; only
            # PostgreSQL parameter syntax and date casts need translation.
            sql = sql.replace("::date", "").replace("%(start_date)s", ":start_date").replace("%(end_date)s", ":end_date")
            frame = pd.read_sql_query(sql, sql_database, params=params)
            return iter([frame.iloc[:2], frame.iloc[2:]])

        connection = Mock()
        connection.raw_sql.side_effect = query
        request = self.request | {
            "start": "2025-01-01", "end": "2025-02-28", "data_source": "ciz",
            "datasets": ("crsp_monthly", "market_returns"), "universe": "nyse_amex_nasdaq",
        }
        paths = download_lsv1994_data(**request, connection=connection)
        stock = pd.read_parquet(paths["crsp_monthly"])
        self.assertEqual(len(stock), 8)
        self.assertEqual(stock["shrcd"].fillna(-1).tolist(), [10, 11, 11, 11, 11, -1, -1, -1])
        self.assertEqual(stock["exchcd"].fillna(-1).tolist(), [1, 3, 2, -1, -1, 1, 1, 1])
        self.assertEqual(stock.loc[3, "ret"], -0.2)
        self.assertEqual(stock.loc[3, "tradingstatusflg"], "H")
        market = pd.read_parquet(paths["market_returns"])
        self.assertEqual(market["vwretd"].tolist(), [0.01, -0.03])
        self.assertEqual(market["ewretd"].tolist(), [0.02, -0.04])
        info = json.loads(paths["crsp_monthly"].with_suffix(".json").read_text())
        self.assertEqual(info["data_source"], "ciz")
        self.assertTrue(info["delisting_returns_in_source"])
        self.assertFalse(info["delisting_return_adjustment"])
        self.assertFalse(info["exchange_and_share_filters_applied"])
        with patch("utils.lsv1994_data_download.get_wrds_connection") as connect:
            self.assertEqual(download_lsv1994_data(**request), paths)
            connect.assert_not_called()

    def test_missing_market_index_leg_is_rejected(self):
        connection = Mock()
        connection.raw_sql.return_value = iter([self.frames["market_returns"].assign(ewretd=None)])
        with self.assertRaisesRegex(ValueError, "market index return"):
            download_lsv1994_data(**self.request, datasets=("market_returns",), connection=connection)
        self.assertFalse((self.cache / "market_returns.parquet").exists())

    def test_missing_calendar_coverage_and_cross_chunk_duplicates_are_rejected(self):
        for name, chunks, message in [
            ("crsp_monthly", [self.frames["crsp_monthly"].iloc[:1]], "Missing calendar periods"),
            ("compustat_annual", [self.frames["compustat_annual"].iloc[:1]], "Missing calendar periods"),
            ("crsp_monthly", [self.frames["crsp_monthly"], self.frames["crsp_monthly"].iloc[1:]], "across chunks"),
        ]:
            with self.subTest(name=name, message=message):
                connection = Mock()
                connection.raw_sql.return_value = iter(chunks)
                with self.assertRaisesRegex(ValueError, message):
                    download_lsv1994_data(**self.request, datasets=(name,), connection=connection)
                self.assertFalse((self.cache / f"{name}.parquet").exists())

    def test_invalid_dates_and_options_fail_before_connecting(self):
        for changes in [
            {"start": "not-a-date"}, {"start": "1967-12-02"}, {"end": "1968-01-30"},
            {"start": "1969-01-01"}, {"comp_end": "1960-12-31"},
            {"datasets": ()}, {"datasets": ("daily_returns",)},
            {"datasets": ("ccm_links", "ccm_links")}, {"universe": "other"},
            {"chunk_size": 0}, {"chunk_size": True},
            {"data_source": "other"}, {"end": "2025-12-31"},
        ]:
            with self.subTest(changes=changes), patch("utils.lsv1994_data_download.get_wrds_connection") as connect:
                with self.assertRaises(ValueError):
                    download_lsv1994_data(**(self.request | changes))
                connect.assert_not_called()

    def test_changed_specification_and_unlabelled_cache_are_rejected(self):
        connection = Mock()
        connection.raw_sql.return_value = iter([self.frames["crsp_monthly"]])
        paths = download_lsv1994_data(**self.request, datasets=("crsp_monthly",), connection=connection)
        with patch("utils.lsv1994_data_download.get_wrds_connection") as connect:
            with self.assertRaisesRegex(ValueError, "specification differs"):
                download_lsv1994_data(**self.request, datasets=("crsp_monthly",), universe="nyse")
            paths["crsp_monthly"].with_suffix(".json").unlink()
            with self.assertRaisesRegex(ValueError, "Missing metadata"):
                download_lsv1994_data(**self.request, datasets=("crsp_monthly",))
            connect.assert_not_called()

    def test_relative_cache_is_anchored_to_project(self):
        connection = Mock()
        connection.raw_sql.return_value = iter([self.frames["ccm_links"]])
        request = self.request | {"cache_dir": self.cache.relative_to(Path(root_path)), "datasets": ("ccm_links",)}
        with chdir(self.cache):
            paths = download_lsv1994_data(**request, connection=connection)
        self.assertEqual(paths["ccm_links"].parent, self.cache)


class PreparationTests(unittest.TestCase):
    def test_default_prepares_all_original_tables_and_reference_only_is_available(self):
        for settings, datasets in [
            ({}, ("crsp_monthly", "compustat_annual", "ccm_links", "market_returns")),
            ({"tables": (1,), "start": "1967-12-01", "comp_end": "1988-12-31"},
             ("crsp_monthly", "compustat_annual", "ccm_links")),
        ]:
            with patch("data_prepare_lsv1994.download_lsv1994_data", side_effect=RuntimeError("before I/O")) as download:
                with redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "before I/O"):
                    data_prepare_lsv1994(**settings)
                self.assertEqual(download.call_args.kwargs["datasets"], datasets)
                self.assertEqual(download.call_args.kwargs["end"], "1990-04-30")
        for tables in ((), (1, 1), (3,)):
            with self.assertRaises(ValueError):
                data_prepare_lsv1994(tables=tables)

    def test_preparation_prints_downloaded_coverage(self):
        with tempfile.TemporaryDirectory(dir=Path(root_path), prefix=".test_lsv1994_") as temporary:
            paths = {}
            for name in ("crsp_monthly", "compustat_annual", "ccm_links"):
                path = Path(temporary) / f"{name}.parquet"
                path.with_suffix(".json").write_text(json.dumps({
                    "rows": 2, "first_date": "1962-12-31", "last_date": "1990-04-30",
                }), encoding="utf-8")
                paths[name] = path
            output = io.StringIO()
            with patch("data_prepare_lsv1994.download_lsv1994_data", return_value=paths):
                with redirect_stdout(output):
                    self.assertEqual(data_prepare_lsv1994(tables=(1,)), paths)
            self.assertIn("finished successfully", output.getvalue())
            self.assertIn("Raw folder for LSV analysis", output.getvalue())

    def test_task2_requests_shared_legacy_history_through_2024(self):
        import runpy

        with patch("data_prepare_lsv1994.download_lsv1994_data", side_effect=RuntimeError("before I/O")) as download:
            with redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "before I/O"):
                runpy.run_path(str(Path(root_path) / "data_prepare_lsv1994_task2.py"), run_name="__main__")
        settings = download.call_args.kwargs
        self.assertEqual(settings["start"], "1957-01-01")
        self.assertEqual(settings["comp_start"], "1957-01-01")
        self.assertEqual(settings["end"], "2024-12-31")
        self.assertEqual(settings["comp_end"], "2024-12-31")
        self.assertEqual(settings["data_source"], "legacy")
        self.assertEqual(settings["universe"], "nyse_amex_nasdaq")


if __name__ == "__main__":
    unittest.main()
