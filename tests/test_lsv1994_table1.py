"""Economic timing, missing-return, and workflow checks for LSV Table I."""

from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

from core.lsv1994.replication._formation import (
    ntile_series, prepare_accounts, prepare_crsp, build_formation, make_constituents,
)
from core.lsv1994.replication._holding import annualize, build_holding_returns
from core.lsv1994.replication.table1 import run_table1, summarize_table1, validate_cohort_coverage
from core.lsv1994.reporter.table1 import report_table1
from utils.path_info import root_path


def synthetic_inputs():
    months = pd.date_range("1967-12-31", "1973-04-30", freq="ME")
    crsp = pd.concat([pd.DataFrame({"permno": 10000 + i, "date": months,
                                    "ret": .001 * i, "prc": 10 + i,
                                    "shrout": 1000, "shrcd": 10, "exchcd": 1})
                      for i in range(1, 21)], ignore_index=True)
    accounts = pd.DataFrame([
        {"gvkey": f"{i:06d}", "datadate": pd.Timestamp(y, 12, 31), "ceq": i ** 1.5,
         "seq": i ** 1.5, "pstk": 0., "pstkrv": 0., "pstkl": 0., "txditc": 0.,
         "ib": .1 * i ** 1.3, "dp": .05 * i, "sale": 100 * (1 + .01 * i) ** (y - 1962)}
        for i in range(1, 21) for y in range(1962, 1972)])
    links = pd.DataFrame({"gvkey": [f"{i:06d}" for i in range(1, 21)],
                          "permno": range(10001, 10021), "linkdt": pd.NaT,
                          "linkenddt": pd.NaT, "linktype": "LC", "linkprim": "P"})
    return crsp, accounts, links


class FormationTests(unittest.TestCase):
    def test_ties_stay_together_and_missing_signals_are_unranked(self):
        values = pd.Series([1., 1., 1., *range(2, 19), np.nan])
        groups = ntile_series(values)
        self.assertEqual(groups.iloc[:3].nunique(), 1)
        self.assertTrue(pd.isna(groups.iloc[-1]))
        self.assertEqual(groups.iloc[-2], 10)
        self.assertTrue(ntile_series(pd.Series(range(9))).isna().all())

    def test_future_accounts_do_not_change_prior_formations(self):
        crsp, accounts, links = synthetic_inputs()
        crsp = prepare_crsp(crsp)
        original = build_formation(crsp, prepare_accounts(accounts), links, 1968, 1972)
        future = accounts.loc[accounts["datadate"].dt.year.eq(1971)].copy()
        future["datadate"] = pd.Timestamp("2025-12-31")
        future[["ceq", "ib", "sale"]] = 1e12
        extended = build_formation(crsp, prepare_accounts(pd.concat([accounts, future])), links, 1968, 1972)
        pd.testing.assert_frame_equal(original, extended)
        first = original.loc[original["formation_year"].eq(1968)].iloc[0]
        prior = accounts.loc[accounts["gvkey"].eq(first.gvkey) & accounts["datadate"].dt.year.eq(1967)].iloc[0]
        self.assertAlmostEqual(first.bm, prior.ceq / first.me_apr)

    def test_sales_growth_does_not_bridge_missing_years(self):
        _, accounts, _ = synthetic_inputs()
        accounts = accounts.loc[~(accounts["gvkey"].eq("000001") & accounts["datadate"].dt.year.eq(1965))]
        prepared = prepare_accounts(accounts)
        self.assertTrue(pd.isna(prepared.loc[prepared["gvkey"].eq("000001")
                                             & prepared["acct_year"].eq(1966), "sales_growth"].iloc[0]))

    def test_positive_cp_ep_restrictions_do_not_remove_bm_members(self):
        formation = pd.DataFrame({"formation_year": 1968, "permno": range(20),
                                   "bm": np.arange(20) + .1, "cp": np.arange(20) - 2.,
                                   "ep": np.arange(20) - 3., "gs": np.arange(20) / 20})
        members = make_constituents(formation)
        self.assertEqual(len(members.loc[members["strategy"].eq("B/M")]), 20)
        self.assertEqual(len(members.loc[members["strategy"].eq("C/P")]), 17)
        self.assertEqual(len(members.loc[members["strategy"].eq("E/P")]), 16)

    def test_currency_sensitivity_is_explicit_and_separate_from_reference(self):
        _, accounts, _ = synthetic_inputs()
        accounts["curcd"] = np.where(accounts["gvkey"].eq("000001"), "CAD", "USD")
        baseline = prepare_accounts(accounts)
        restricted = prepare_accounts(accounts, accounting_currency="USD")
        self.assertEqual(baseline["gvkey"].nunique(), 20)
        self.assertEqual(restricted["gvkey"].nunique(), 19)
        self.assertTrue(restricted["curcd"].eq("USD").all())

    def test_invalid_returns_and_duplicate_months_are_rejected(self):
        crsp, _, _ = synthetic_inputs()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            prepare_crsp(pd.concat([crsp, crsp.iloc[[0]]]))
        crsp.loc[0, "ret"] = -1.1
        with self.assertRaisesRegex(ValueError, "Invalid decimal"):
            prepare_crsp(crsp)


class HoldingTests(unittest.TestCase):
    def test_disappearance_replacement_and_internal_missing_are_distinct(self):
        months = pd.DataFrame({"start_year": 1968, "ym": pd.period_range("1968-05", periods=12, freq="M"),
                                "month_no": range(1, 13)})
        stock_year = pd.DataFrame({"permno": [1, 2, 3], "start_year": 1968, "size_decile": 1})
        monthly = pd.concat([
            pd.DataFrame({"permno": 1, "start_year": 1968, "ym": months["ym"].iloc[:1], "ret": .1, "row_flag": 1}),
            pd.DataFrame({"permno": 2, "start_year": 1968, "ym": months["ym"], "ret": .1, "row_flag": 1}),
            pd.DataFrame({"permno": 3, "start_year": 1968, "ym": months["ym"].iloc[:1], "ret": -1., "row_flag": 1}),
        ], ignore_index=True)
        monthly.loc[monthly["permno"].eq(2) & monthly["ym"].eq(pd.Period("1968-07")), "ret"] = np.nan
        size_month = months[["start_year", "ym"]].assign(size_decile=1, size_ret=.02)
        result = annualize(stock_year, months, monthly, size_month).set_index("permno")
        self.assertAlmostEqual(result.loc[1, "annual_ret"], 1.1 * 1.02 ** 11 - 1)
        self.assertEqual(result.loc[1, "replacement_months"], 11)
        self.assertTrue(pd.isna(result.loc[2, "annual_ret"]))
        self.assertEqual(result.loc[2, "replacement_months"], 0)
        self.assertEqual(result.loc[2, "internal_missing"], 1)
        self.assertAlmostEqual(result.loc[3, "annual_ret"], -1.)

    def test_exchange_change_does_not_remove_original_holding_returns(self):
        raw, accounts, links = synthetic_inputs()
        first = prepare_crsp(raw)
        formation = build_formation(first, prepare_accounts(accounts), links, 1968, 1972)
        members = make_constituents(formation)
        changed = raw.copy()
        changed.loc[changed["date"].between("1968-05-01", "1969-03-31"), "exchcd"] = 3
        baseline, _, _ = build_holding_returns(first, members, 1968, "1973-04-30")
        alternate, _, _ = build_holding_returns(prepare_crsp(changed), members, 1968, "1973-04-30")
        cols = ["permno", "start_year", "annual_ret"]
        pd.testing.assert_frame_equal(baseline[cols], alternate[cols])

    def test_survivors_are_reweighted_without_future_survival_filter_at_formation(self):
        raw, accounts, links = synthetic_inputs()
        raw = raw.loc[~(raw["permno"].eq(10001) & raw["date"].gt("1968-12-31"))]
        crsp = prepare_crsp(raw)
        members = make_constituents(build_formation(crsp, prepare_accounts(accounts), links, 1968, 1968))
        first_decile = members.loc[members["strategy"].eq("B/M") & members["portfolio"].eq(1)]
        self.assertIn(10001, first_decile["permno"].tolist())
        annual, yearly, _ = build_holding_returns(crsp, members, 1968, "1973-04-30")
        self.assertFalse(((annual["permno"] == 10001) & (annual["start_year"] == 1969)).any())
        second = yearly.loc[yearly["strategy"].eq("B/M") & yearly["portfolio"].eq(1) & yearly["hold_year"].eq(2)].iloc[0]
        self.assertEqual(second.n_valid, 1)
        self.assertAlmostEqual(second.ret, 1.002 ** 12 - 1)

    def test_missing_calendar_month_and_partial_return_year_are_rejected(self):
        raw, accounts, links = synthetic_inputs()
        crsp = prepare_crsp(raw)
        members = make_constituents(build_formation(crsp, prepare_accounts(accounts), links, 1968, 1972))
        with self.assertRaisesRegex(ValueError, "missing required calendar"):
            build_holding_returns(crsp.loc[crsp["ym"].ne(pd.Period("1969-06"))], members, 1968, "1973-04-30")
        with self.assertRaisesRegex(ValueError, "April month-end"):
            build_holding_returns(crsp, members, 1968, "1973-03-31")

    def test_summary_compounds_horizon_means_in_paper_order(self):
        yearly = pd.DataFrame([(s, y, d, h, h / 100, h / 100 - .01)
                                for s in ("B/M", "C/P", "E/P", "GS") for y in (1968, 1969)
                                for d in range(1, 11) for h in range(1, 6)],
                               columns=["strategy", "formation_year", "portfolio", "hold_year", "ret", "saar"])
        summary = summarize_table1(yearly).set_index(["strategy", "stat", "decile"])
        self.assertAlmostEqual(summary.loc[("B/M", "AR", 1), "return"], .03)
        self.assertAlmostEqual(summary.loc[("B/M", "CR5", 1), "return"], np.prod(1 + np.arange(1, 6) / 100) - 1)
        self.assertAlmostEqual(summary.loc[("GS", "SAAR", 10), "return"], .02)


class WorkflowTests(unittest.TestCase):
    def test_end_to_end_constant_returns_raw_preservation_and_saved_only_reporting(self):
        raw, accounts, links = synthetic_inputs()
        with tempfile.TemporaryDirectory(prefix=".test_lsv_table1_", dir=root_path) as directory:
            folder = Path(directory)
            raw_dir, result_dir = folder / "raw", folder / "results"
            raw_dir.mkdir()
            for name, frame in (("crsp_monthly", raw), ("compustat_annual", accounts), ("ccm_links", links)):
                frame.to_parquet(raw_dir / f"{name}.parquet", index=False)
            before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in raw_dir.iterdir()}
            with redirect_stdout(io.StringIO()):
                summary, yearly = run_table1(1968, 1972, "1973-04-30", raw_dir, result_dir)
                paths = report_table1(result_dir)
            self.assertEqual(len(summary), 320)
            self.assertTrue(all(p.is_file() for p in paths.values()))
            coverage = validate_cohort_coverage(yearly, 1968, 1972, "1973-04-30")
            self.assertEqual(coverage.iloc[:5]["n_cohorts"].tolist(), [5, 4, 3, 2, 1])
            members = pd.read_parquet(result_dir / "portfolio_constituents.parquet")
            # Constant individual returns have an independently known annual
            # buy-and-hold value, distinct from a monthly-rebalanced portfolio.
            expected = members.assign(known_return=(1 + (members["permno"] - 10000) / 1000) ** 12 - 1)
            expected = expected.groupby(["strategy", "formation_year", "portfolio"])["known_return"].mean()
            r1 = yearly.loc[yearly["hold_year"].eq(1)].set_index(["strategy", "formation_year", "portfolio"])["ret"]
            np.testing.assert_allclose(r1.sort_index(), expected.sort_index(), atol=1e-12)
            self.assertEqual(before, {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in raw_dir.iterdir()})
            html = paths["html"].read_text(encoding="utf-8")
            self.assertIn("data:image/png;base64,", html)
            self.assertIn("Panel D", html)
            # Reporting succeeds with raw inputs unavailable.
            for path in raw_dir.iterdir():
                path.unlink()
            with redirect_stdout(io.StringIO()):
                report_table1(result_dir)
            # A stale or edited table cannot silently enter a report.
            (result_dir / "table1_summary.csv").write_text("corrupted", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Saved output changed"):
                report_table1(result_dir)


if __name__ == "__main__":
    unittest.main()
