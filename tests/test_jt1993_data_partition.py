"""Offline subsample boundaries, warm-up, provenance, and analysis compatibility."""

import io
import json
from contextlib import chdir, redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pyarrow.parquet as pq

from core.jt1993.data_partition import partition_jt1993_data
from core.jt1993.replication.table1 import run_table1
from utils.path_info import root_path


class PartitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        days = pd.bdate_range("1962-07-01", "1989-12-31")
        time = np.arange(len(days))
        daily = pd.concat([
            pd.DataFrame({"permno": 10000 + i, "date": days,
                          "ret": 0.00005 * (i - 9.5) * np.cos(time / 150) + 0.0002 * np.sin(time / 11),
                          "exchcd": 1, "shrcd": 10}) for i in range(20)
        ], ignore_index=True)
        daily.loc[(daily["permno"] == 10000) & daily["date"].eq("1968-07-02"), "ret"] = np.nan
        monthly = daily.assign(month=daily["date"] + pd.offsets.MonthEnd(0), loggross=np.log1p(daily["ret"]))
        monthly = monthly.groupby(["permno", "month"], as_index=False).agg(
            loggross=("loggross", "sum"), n_daily_rows=("date", "size"), n_valid_ret=("ret", "count"),
        )
        monthly["ret"] = np.expm1(monthly.pop("loggross"))
        characteristics = monthly[["permno", "month"]].assign(prc=25.0, shrout=1000.0, exchcd=1, shrcd=10)
        months = pd.date_range("1962-07-31", "1989-12-31", freq="ME")
        market = pd.DataFrame({"month": months, "vwretd": 0.02 * np.sin(np.arange(len(months))), "ewretd": 0.01})
        cls.frames = {"monthly_returns": monthly, "daily_returns": daily,
                      "monthly_characteristics": characteristics, "market_returns": market}

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix=".test_jt1993_", dir=Path(root_path))
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.raw = self.root / "source"
        self.raw.mkdir()
        self.output = self.root / "subsamples"
        for name, frame in self.frames.items():
            frame.to_parquet(self.raw / f"{name}.parquet", index=False)
            metadata = {"dataset": name, "rows": len(frame), "universe": "nyse_amex", "data_source": "legacy",
                        "return_units": "decimal", "signature": "original-download-signature", "sql": "source query"}
            (self.raw / f"{name}.json").write_text(json.dumps(metadata), encoding="utf-8")
        self.args = {"raw_data_dir": self.raw, "run_start_date": "1965-01-01",
                     "split_date": "1975-01-01", "run_end_date": "1989-12-31", "output_dir": self.output}

    def test_boundaries_preserve_values_schema_order_and_source_files(self):
        original_bytes = {path.name: path.read_bytes() for path in self.raw.iterdir()}
        with redirect_stdout(io.StringIO()):
            paths = partition_jt1993_data(**self.args, batch_size=997)
        self.assertEqual(set(paths), {"period1", "period2"})
        for label, history_start, report_start, report_end, months in (
            ("period1", "1963-01-01", "1965-01-01", "1974-12-31", 120),
            ("period2", "1973-01-01", "1975-01-01", "1989-12-31", 180),
        ):
            folder = paths[label]
            info = json.loads((folder / "partition_info.json").read_text(encoding="utf-8"))
            self.assertEqual(info["history_start_date"], history_start)
            self.assertEqual(info["history_months"], 24)
            self.assertEqual(info["main_parameters"], {
                "raw_data_dir": str(folder), "run_start_date": report_start, "run_end_date": report_end,
            })
            self.assertEqual(len(pd.period_range(report_start, report_end, freq="M")), months)
            for name, frame in self.frames.items():
                date_column = "date" if name == "daily_returns" else "month"
                expected = frame.loc[frame[date_column].between(history_start, report_end)].reset_index(drop=True)
                actual = pd.read_parquet(folder / f"{name}.parquet")
                assert_frame_equal(actual, expected)
                self.assertEqual(pq.read_schema(folder / f"{name}.parquet"), pq.read_schema(self.raw / f"{name}.parquet"))
                metadata = json.loads((folder / f"{name}.json").read_text(encoding="utf-8"))
                self.assertEqual(metadata["rows"], len(actual))
                self.assertEqual(metadata["source_rows"], len(frame))
                self.assertEqual(metadata["universe"], "nyse_amex")
                self.assertEqual(metadata["source_download_metadata"]["signature"], "original-download-signature")
                self.assertNotIn("signature", metadata)  # A derived subset is not an authenticated download cache.
                self.assertEqual(metadata["last_date"], str(actual[date_column].max().date()))
        for name, content in original_bytes.items():
            self.assertEqual((self.raw / name).read_bytes(), content)
        self.assertFalse(list(self.output.glob(".partition_*")))

    def test_table1_results_match_using_full_source_or_each_partition(self):
        with redirect_stdout(io.StringIO()), patch("utils.jt1993_data_download.get_wrds_connection") as connect:
            paths = partition_jt1993_data(**self.args)
            for label, start, end, months in (
                ("period1", "1965-01-01", "1974-12-31", 120),
                ("period2", "1975-01-01", "1989-12-31", 180),
            ):
                expected_summary, expected_returns = run_table1(start, end, self.raw, self.root / f"full_{label}")
                actual_summary, actual_returns = run_table1(start, end, paths[label], self.root / f"subset_{label}")
                assert_frame_equal(actual_summary, expected_summary, check_exact=False, rtol=1e-10, atol=1e-10)
                assert_frame_equal(actual_returns, expected_returns, check_exact=False, rtol=1e-10, atol=1e-10)
                self.assertTrue(actual_summary["n"].eq(months).all())
            connect.assert_not_called()

    def test_relative_paths_use_path_info_instead_of_current_working_directory(self):
        cwd = self.root / "another_directory"
        cwd.mkdir()
        request = {**self.args, "raw_data_dir": self.raw.relative_to(Path(root_path)),
                   "output_dir": self.output.relative_to(Path(root_path))}
        with chdir(cwd), redirect_stdout(io.StringIO()):
            paths = partition_jt1993_data(**request)
        self.assertTrue(all(folder.parent == self.output for folder in paths.values()))
        self.assertTrue(all((folder / "monthly_returns.parquet").is_file() for folder in paths.values()))

    def test_missing_optional_inputs_are_not_created(self):
        for name in ("monthly_characteristics", "market_returns"):
            (self.raw / f"{name}.parquet").unlink()
            (self.raw / f"{name}.json").unlink()
        with redirect_stdout(io.StringIO()):
            paths = partition_jt1993_data(**self.args)
        for folder in paths.values():
            self.assertEqual({path.stem for path in folder.glob("*.parquet")}, {"monthly_returns", "daily_returns"})

    def test_missing_warmup_or_report_month_publishes_neither_period(self):
        source = self.frames["monthly_returns"]
        for label, frame in (
            ("warmup", source.loc[source["month"].ge("1965-01-01")]),
            ("report", source.loc[source["month"].ne("1978-04-30")]),
        ):
            frame.to_parquet(self.raw / "monthly_returns.parquet", index=False)
            output = self.output / label
            with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "missing calendar months"):
                partition_jt1993_data(**{**self.args, "output_dir": output})
            self.assertEqual(list(output.iterdir()), [])

    def test_missing_dates_are_rejected_instead_of_silently_dropped(self):
        monthly = self.frames["monthly_returns"].copy()
        monthly.loc[0, "month"] = pd.NaT
        monthly.to_parquet(self.raw / "monthly_returns.parquet", index=False)
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "Missing dates"):
            partition_jt1993_data(**self.args)
        self.assertEqual(list(self.output.iterdir()), [])

    def test_invalid_boundaries_source_output_and_existing_targets_are_rejected(self):
        for changes in (
            {"run_start_date": "1965-01-02"}, {"split_date": "1975-01-02"},
            {"split_date": "1965-01-01"}, {"split_date": "1990-01-01"},
            {"run_start_date": "1965-01-01 12:00"}, {"run_start_date": "1965-01-01T00:00:00+08:00"},
            {"run_end_date": "1989-12-30"}, {"batch_size": 0}, {"batch_size": True},
            {"output_dir": self.raw}, {"output_dir": self.raw / "subsets"},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                partition_jt1993_data(**{**self.args, **changes})
        with redirect_stdout(io.StringIO()):
            paths = partition_jt1993_data(**self.args)
        before = (paths["period1"] / "monthly_returns.parquet").read_bytes()
        with self.assertRaisesRegex(FileExistsError, "already exists"):
            partition_jt1993_data(**self.args)
        self.assertEqual((paths["period1"] / "monthly_returns.parquet").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
