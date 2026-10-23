"""Offline checks of the Table I preparation-to-analysis handoff."""

import io
import json
import tempfile
import unittest
from contextlib import chdir, redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

from data_prepare_jt1993 import data_prepare_jt1993
from core.jt1993.replication.table1 import run_table1
from utils.path_info import root_path
from tests.test_jt1993_data_download import sample_frame


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix=".test_jt1993_", dir=Path(root_path))
        self.addCleanup(temporary.cleanup)
        self.cache_dir = Path(temporary.name)
        months = pd.date_range("1962-07-31", "1989-12-31", freq="ME")
        self.monthly = pd.concat([
            pd.DataFrame({
                "permno": 10001 + i, "month": months, "ret": 0.001 * (i + 1),
                "n_daily_rows": 20, "n_valid_ret": 20,
            }) for i in range(20)
        ], ignore_index=True)
        self.daily = self.monthly[["permno", "month", "ret"]].rename(columns={"month": "date"})
        self.daily = self.daily.assign(exchcd=1, shrcd=10)

    def test_prepare_table_i_then_run_two_local_windows_without_authentication(self):
        connection = Mock()
        connection.raw_sql.side_effect = [iter([self.monthly]), iter([self.daily])]
        with (
            patch("utils.jt1993_data_download.DEFAULT_CACHE_DIR", self.cache_dir),
            patch("utils.jt1993_data_download.get_wrds_connection", return_value=connection),
            redirect_stdout(io.StringIO()),
        ):
            paths = data_prepare_jt1993("1962-07-01", "1989-12-31")

        self.assertEqual(set(paths), {"monthly_returns", "daily_returns"})
        self.assertEqual(connection.raw_sql.call_count, 2)
        connection.close.assert_called_once_with()
        before = {name: path.read_bytes() for name, path in paths.items()}
        relative_folder = paths["monthly_returns"].parent.relative_to(Path(root_path))
        output = io.StringIO()
        with (
            patch("utils.jt1993_data_download.get_wrds_connection") as connect,
            chdir(self.cache_dir),
            redirect_stdout(output),
        ):
            summary, returns = run_table1(
                "1965-01-01", "1989-12-31", relative_folder, self.cache_dir / "first_run",
            )
            run_table1(
                "1980-01-01", "1989-12-31", relative_folder, self.cache_dir / "second_run",
            )
            connect.assert_not_called()

        self.assertEqual(len(summary), 96)
        self.assertTrue(summary["n"].eq(300).all())
        self.assertEqual(len(returns), 32 * 300)
        self.assertTrue(returns["month"].between("1965-01-31", "1989-12-31").all())
        run_info = json.loads((self.cache_dir / "first_run" / "run_info.json").read_text(encoding="utf-8"))
        self.assertEqual(run_info["raw_inputs"]["monthly_returns"]["rows"], len(self.monthly))
        self.assertEqual(run_info["raw_inputs"]["monthly_returns"]["download_metadata"]["universe"], "nyse_amex")
        self.assertEqual(run_info["settings"]["panel_b_gap_trading_days"], 5)
        self.assertTrue(run_info["reference_comparison_applied"])
        for filename in run_info["outputs"]:
            self.assertTrue((self.cache_dir / "first_run" / filename).is_file())
        second_info = json.loads((self.cache_dir / "second_run" / "run_info.json").read_text(encoding="utf-8"))
        self.assertFalse(second_info["reference_comparison_applied"])
        # The earlier ranking history survives both subsample runs.
        self.assertEqual(output.getvalue().count("Raw months: 1962-07 to 1989-12"), 2)
        self.assertIn("300 months", output.getvalue())
        self.assertIn("120 months", output.getvalue())
        for name, path in paths.items():
            self.assertEqual(path.read_bytes(), before[name])

    def test_missing_daily_input_requires_preparation_without_connecting(self):
        self.monthly.to_parquet(self.cache_dir / "monthly_returns.parquet", index=False)
        with patch("utils.jt1993_data_download.get_wrds_connection") as connect:
            with self.assertRaisesRegex(FileNotFoundError, "data_prepare_jt1993.py first"):
                run_table1("1965-01-01", "1989-12-31", self.cache_dir)
            connect.assert_not_called()

    def test_prepare_task2_passes_universe_source_and_all_required_inputs(self):
        names = ("monthly_returns", "daily_returns", "monthly_characteristics", "market_returns")
        connection = Mock()
        connection.raw_sql.side_effect = [iter([sample_frame(name, "2025-01", "2025-02")]) for name in names]
        with (
            patch("utils.jt1993_data_download.DEFAULT_CACHE_DIR", self.cache_dir),
            patch("utils.jt1993_data_download.get_wrds_connection", return_value=connection),
            redirect_stdout(io.StringIO()),
        ):
            paths = data_prepare_jt1993(
                "2025-01-01", "2025-02-28", universe="nyse_amex_nasdaq",
                tables=(1, 2, 7), data_source="ciz",
            )
        self.assertEqual(set(paths), set(names))
        self.assertEqual(connection.raw_sql.call_count, 4)
        connection.close.assert_called_once_with()
        for path in paths.values():
            metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["data_source"], "ciz")
            self.assertEqual(metadata["universe"], "nyse_amex_nasdaq")
            self.assertEqual(path.parent.name, "nyse_amex_nasdaq_ciz_20250101_20250228")

    def test_local_data_must_cover_every_reported_month(self):
        monthly = self.monthly.loc[self.monthly["month"].ne(pd.Timestamp("1970-01-31"))]
        monthly.to_parquet(self.cache_dir / "monthly_returns.parquet", index=False)
        self.daily.to_parquet(self.cache_dir / "daily_returns.parquet", index=False)
        with self.assertRaisesRegex(ValueError, "1970-01"):
            run_table1("1965-01-01", "1989-12-31", self.cache_dir)


if __name__ == "__main__":
    unittest.main()
