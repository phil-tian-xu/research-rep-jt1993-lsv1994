"""Offline checks of the Table II/VII drafts, timing, inference, and reporters."""

import csv
import io
import json
from contextlib import chdir, redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from core.jt1993.replication.table2 import build_table2_cohorts, run_table2, summarize_table2
from core.jt1993.replication.table7 import newey_west_mean, run_table7, summarize_table7
from core.jt1993.reporter.table2 import report_table2, table2_rows
from core.jt1993.reporter.table7 import report_table7, table7_rows
from data_prepare_jt1993 import data_prepare_jt1993
from utils.path_info import root_path


class Table2Tests(unittest.TestCase):
    def test_size_uses_preformation_price_units_and_does_not_filter_returns(self):
        formation = 2000 * 12 + 2
        rank = pd.DataFrame({"permno": [1, 2], "formation_id": formation, "momr": 1})
        monthly = pd.DataFrame({
            "permno": [1, 2, 1, 2], "month_id": [formation, formation, formation + 1, formation + 1],
            "ret": [0.1, 0.3, 0.2, np.nan],
        })
        characteristics = pd.DataFrame({
            "permno": [1, 2, 1, 2],
            "month": pd.to_datetime(["2000-01-31"] * 2 + ["2000-02-29"] * 2),
            "prc": [-20.0, np.nan, 9999.0, 9999.0], "shrout": 2000.0,
        })
        cohort = build_table2_cohorts(rank, monthly, characteristics).set_index("age")
        self.assertAlmostEqual(cohort.loc[0, "cohort_cap"], 40.0)
        self.assertAlmostEqual(cohort.loc[0, "cohort_ret"], 0.2)
        self.assertEqual(cohort.loc[0, "n_with_ret"], 2)
        self.assertEqual(cohort.loc[0, "n_with_cap"], 1)
        self.assertAlmostEqual(cohort.loc[1, "cohort_cap"], 40.0)
        self.assertAlmostEqual(cohort.loc[1, "cohort_ret"], 0.2)

    def test_beta_is_ols_with_intercept_and_spread_is_directly_estimated(self):
        months = pd.date_range("2000-01-31", periods=6, freq="ME")
        ids = months.year * 12 + months.month
        x = np.array([0.02, -0.01, 0.03, -0.04, 0.01, 0.0])
        portfolios = pd.concat([
            pd.DataFrame({"hold_id": ids, "momr": i, "ret": i * 0.001 + i * 0.2 * x,
                          "market_cap_millions": i * 100.0}) for i in range(1, 11)
        ])
        market = pd.DataFrame({"month": months, "vwretd": x})
        summary, _ = summarize_table2(portfolios, market)
        np.testing.assert_allclose(summary.iloc[:10]["beta"], np.arange(1, 11) * 0.2)
        np.testing.assert_allclose(summary.iloc[:10]["intercept"], np.arange(1, 11) * 0.001)
        self.assertAlmostEqual(summary.iloc[-1]["beta"], 1.8)
        self.assertTrue(pd.isna(summary.iloc[-1]["average_market_cap_millions"]))
        with self.assertRaisesRegex(ValueError, "complete, nonconstant"):
            summarize_table2(portfolios, market.iloc[1:])
        with self.assertRaisesRegex(ValueError, "duplicate months"):
            summarize_table2(portfolios, pd.concat([market, market.iloc[:1]]))


class Table7Tests(unittest.TestCase):
    def test_newey_west_matches_bartlett_covariance_matrix(self):
        x = np.array([0.01, -0.03, 0.04, 0.02, -0.01, 0.005])
        residual = x - x.mean()
        for lags in (0, 1, 3):
            distance = np.abs(np.arange(len(x))[:, None] - np.arange(len(x)))
            weights = np.maximum(1 - distance / (lags + 1), 0)
            expected_se = np.sqrt(residual @ weights @ residual / (len(x) * (len(x) - 1)))
            mean, se, t = newey_west_mean(x, lags)
            self.assertAlmostEqual(mean, x.mean())
            self.assertAlmostEqual(se, expected_se)
            self.assertAlmostEqual(t, x.mean() / expected_se)
        self.assertAlmostEqual(newey_west_mean(x, 0)[1], x.std(ddof=1) / np.sqrt(len(x)))
        for invalid in (-1, len(x), 1.5, True):
            with self.assertRaises(ValueError):
                newey_west_mean(x, invalid)
        with self.assertRaisesRegex(ValueError, "finite, consecutive"):
            newey_west_mean([0.1, np.nan, 0.2], 0)

    def test_event_returns_are_single_cohorts_and_cumulative_sums(self):
        formations = np.arange(2000 * 12 + 1, 2000 * 12 + 41)
        cohort = pd.DataFrame([
            {"formation_id": formation, "hold_id": formation + age, "age": age,
             "momr": decile, "cohort_ret": (0.01 if decile == 1 else 0.01 + (i + 1) * 0.0001 + age * 0.00001)}
            for i, formation in enumerate(formations) for age in range(36) for decile in (1, 10)
        ])
        summary, event = summarize_table7(cohort, formations)
        mean_first = np.arange(1, 41).mean() * 0.0001
        self.assertAlmostEqual(summary.iloc[0]["monthly_return"], mean_first)
        self.assertAlmostEqual(summary.iloc[35]["monthly_return"], mean_first + 35 * 0.00001)
        self.assertAlmostEqual(summary.iloc[35]["cumulative_return"], 36 * mean_first + sum(range(36)) * 0.00001)
        self.assertTrue(summary["n_cohorts"].eq(40).all())
        self.assertEqual(summary["hac_lags"].to_list(), list(range(36)))
        self.assertEqual(len(event), 40 * 36)
        first_cohort = event.loc[event["formation_id"].eq(formations[0])]
        np.testing.assert_allclose(first_cohort["cumulative_return"], first_cohort["monthly_return"].cumsum())
        missing = cohort.drop(cohort.index[-1])
        with self.assertRaisesRegex(ValueError, "Missing event-month"):
            summarize_table7(missing, formations)  # No future-based dropping of a bad cohort.


class TableDraftWorkflowTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix=".test_jt1993_", dir=Path(root_path))
        self.addCleanup(temporary.cleanup)
        self.raw = Path(temporary.name)
        months = pd.date_range("1969-01-31", "1977-12-31", freq="ME")
        x = 0.002 * np.sin(np.arange(len(months)))
        self.monthly = pd.concat([
            pd.DataFrame({"permno": 10000 + i, "month": months,
                          "ret": (i + 1) * 0.001 + (0.5 + i * 0.02) * x}) for i in range(20)
        ], ignore_index=True)
        self.characteristics = pd.concat([
            pd.DataFrame({"permno": 10000 + i, "month": months,
                          "prc": -(10 + i + np.arange(len(months)) * 0.01), "shrout": 1000.0})
            for i in range(20)
        ], ignore_index=True)
        self.market = pd.DataFrame({"month": months, "vwretd": x})
        self.monthly.to_parquet(self.raw / "monthly_returns.parquet", index=False)
        self.before = (self.raw / "monthly_returns.parquet").read_bytes()

    def test_table2_local_workflow_and_paper_layout(self):
        self.characteristics.to_parquet(self.raw / "monthly_characteristics.parquet", index=False)
        self.market.to_parquet(self.raw / "market_returns.parquet", index=False)
        relative = self.raw.relative_to(Path(root_path))
        with patch("utils.jt1993_data_download.get_wrds_connection") as connect, redirect_stdout(io.StringIO()), chdir(self.raw):
            summary, portfolios = run_table2("1970-01-01", "1977-12-31", relative, relative / "table2")
            files = report_table2(relative / "table2")
            connect.assert_not_called()
        self.assertEqual(len(portfolios), 96 * 10)
        self.assertTrue(portfolios["n_cohorts"].eq(6).all())
        self.assertTrue(summary.iloc[:10]["n_months"].eq(96).all())
        np.testing.assert_allclose(summary.iloc[:10]["beta"], 0.5 + np.arange(0.5, 20, 2) * 0.02, atol=1e-12)
        self.assertAlmostEqual(summary.iloc[-1]["beta"], 0.36)
        # Jan 1970's cohorts formed Aug 1969-Jan 1970, using July-Dec 1969 size.
        jan_p1 = portfolios.loc[portfolios["month"].eq("1970-01-31") & portfolios["momr"].eq(1)].iloc[0]
        self.assertAlmostEqual(jan_p1["market_cap_millions"], 10.5 + np.arange(6, 12).mean() * 0.01)
        with files["csv"].open(encoding="utf-8", newline="") as stream:
            rows = list(csv.reader(stream))
        self.assertEqual(rows, table2_rows(summary))
        self.assertEqual(len(rows), 12)
        self.assertTrue(all(len(row) == 3 for row in rows))
        self.assertEqual(rows[0], ["", "Beta", "Average Market Capitalization"])
        self.assertEqual(rows[-1], ["P10-P1", "0.36", "—"])
        self.assertTrue(files["html"].is_file())
        self.assertEqual((self.raw / "monthly_returns.parquet").read_bytes(), self.before)

    def test_table7_local_workflow_balanced_boundaries_and_paper_layout(self):
        with patch("utils.jt1993_data_download.get_wrds_connection") as connect, redirect_stdout(io.StringIO()):
            summary, event = run_table7("1970-01-01", "1977-12-31", self.raw, self.raw / "table7")
            files = report_table7(self.raw / "table7")
            connect.assert_not_called()
        self.assertTrue(summary["n_cohorts"].eq(61).all())
        self.assertEqual(len(event), 61 * 36)
        self.assertEqual(event["formation_month"].max(), pd.Timestamp("1975-01-31"))
        self.assertEqual(event["holding_month"].max(), pd.Timestamp("1977-12-31"))
        info = json.loads((self.raw / "table7" / "run_info.json").read_text(encoding="utf-8"))
        self.assertEqual(info["formation_end"], "1975-01-31")
        self.assertEqual(info["n_cohorts"], 61)
        with files["csv"].open(encoding="utf-8", newline="") as stream:
            rows = list(csv.reader(stream))
        self.assertEqual(rows, table7_rows(summary))
        self.assertEqual(len(rows), 25)
        self.assertTrue(all(len(row) == 9 for row in rows))
        self.assertEqual(rows[0], ["t", "Monthly Return", "Cumulative Return"] * 3)
        self.assertEqual(rows[1][::3], ["1", "13", "25"])
        self.assertEqual(rows[-2][::3], ["12", "24", "36"])
        self.assertEqual(rows[2][::3], ["", "", ""])
        self.assertTrue(rows[2][1].startswith("("))
        self.assertTrue(files["html"].is_file())
        self.assertEqual((self.raw / "monthly_returns.parquet").read_bytes(), self.before)

    def test_missing_table2_inputs_require_preparation(self):
        with patch("utils.jt1993_data_download.get_wrds_connection") as connect:
            with self.assertRaisesRegex(FileNotFoundError, "tables=\\(2,\\)"):
                run_table2("1970-01-01", "1977-12-31", self.raw, self.raw / "table2")
            connect.assert_not_called()
        self.assertFalse((self.raw / "table2").exists())

    def test_missing_history_or_small_followup_window_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Too few complete"):
            run_table7("1976-01-01", "1977-12-31", self.raw)
        self.monthly.loc[self.monthly["month"].ne("1969-07-31")].to_parquet(
            self.raw / "monthly_returns.parquet", index=False,
        )
        with self.assertRaisesRegex(ValueError, "1969-07"):
            run_table7("1970-01-01", "1977-12-31", self.raw)


class PreparationAndLayoutTests(unittest.TestCase):
    def test_prepare_selects_only_inputs_required_by_requested_tables(self):
        for tables, expected in [
            ((2,), ("monthly_returns", "monthly_characteristics", "market_returns")),
            ((7,), ("monthly_returns",)),
            ((1, 2, 7), ("monthly_returns", "daily_returns", "monthly_characteristics", "market_returns")),
        ]:
            with patch("data_prepare_jt1993.download_jt1993_data", side_effect=RuntimeError("stop before I/O")) as download:
                with redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "stop before I/O"):
                    data_prepare_jt1993("1962-07-01", "1989-12-31", tables=tables)
                self.assertEqual(download.call_args.kwargs["datasets"], expected)
        for invalid in ((), (2, 2), (3,)):
            with self.assertRaises(ValueError):
                data_prepare_jt1993("1962-07-01", "1989-12-31", tables=invalid)

    def test_reporters_reject_missing_or_duplicate_paper_rows(self):
        for portfolios in (["P1"], [*(f"P{i}" for i in range(1, 11)), "P10"]):
            with self.assertRaisesRegex(ValueError, "exactly P1-P10"):
                table2_rows(pd.DataFrame({"portfolio": portfolios}))
        for months in (range(1, 36), [*range(1, 36), 35]):
            with self.assertRaisesRegex(ValueError, "exactly event months"):
                table7_rows(pd.DataFrame({"event_month": months}))


if __name__ == "__main__":
    unittest.main()
