"""Universe batches, calendar boundaries, and incomplete accounting follow-up."""

from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

from core.lsv1994.runs import plan_run, batch_main
from core.lsv1994.replication._formation import prepare_crsp
from core.lsv1994.replication.table5 import build_table5
from core.lsv1994.replication.table6 import build_figure2, build_table6
from core.lsv1994.reporter.tables import validate_summary, report_table
from utils.path_info import root_path


def write_sidecars(directory, *, first_account="1957-01-31", last_account="2024-12-31",
                   first_stock="1957-01-31", last_stock="2024-12-31", last_index="2024-12-31"):
    metadata = {
        "crsp_monthly": {"first_date":first_stock,"last_date":last_stock,"data_source":"legacy",
                         "exchange_and_share_filters_applied":False},
        "compustat_annual": {"first_date":first_account,"last_date":last_account},
        "market_returns": {"first_date":first_stock,"last_date":last_index},
        "ccm_links": {},
    }
    for name, info in metadata.items():
        (directory/f"{name}.parquet").touch()
        (directory/f"{name}.json").write_text(json.dumps(info),encoding="utf-8")


def growth_inputs():
    n = 8
    accounts = pd.DataFrame([
        {"gvkey":str(i),"acct_year":year,"earn":10*1.05**(year-1962),
         "cf":15*1.04**(year-1962),"sale":100*1.03**(year-1962)}
        for i in range(n) for year in range(1962,1976)])
    formation = pd.concat([
        pd.DataFrame({"permno":range(n),"gvkey":[str(i) for i in range(n)],"formation_year":year,
                      "me_apr":100.,"ep":.1,"cp":.15,"bm":.5,"sale":100.,"dvc":2.})
        for year in range(1968,1972)],ignore_index=True)
    deciles = pd.concat([
        pd.DataFrame({"permno":range(4),"formation_year":year,"strategy":"B/M","portfolio":[1,1,10,10]})
        for year in range(1968,1972)],ignore_index=True)
    pairs = pd.concat([
        pd.DataFrame({"permno":range(4,8),"formation_year":year,"strategy":"C/P-GS","portfolio":[3,3,7,7]})
        for year in range(1968,1972)],ignore_index=True)
    monthly = pd.DataFrame([(i,m,.01) for i in range(n) for m in pd.period_range("1965-05","1971-04",freq="M")],
                           columns=["permno","ym","ret"])
    caps = pd.DataFrame([(i,y,100.) for i in range(n) for y in range(1963,1976)],columns=["permno","year","me"])
    return formation,deciles,pairs,accounts,monthly,caps


class RunPlanTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix=".test_lsv_runs_",dir=root_path)
        self.addCleanup(self.temporary.cleanup)
        self.raw = Path(self.temporary.name)
        write_sidecars(self.raw)

    def test_requested_and_available_dates_are_distinct(self):
        plan = plan_run(self.raw,1962,2025,"nyse")
        self.assertEqual((plan["formation_start"],plan["formation_end"],plan["return_end"]),(1963,2023,"2024-04-30"))
        self.assertEqual(plan["requested_sample_start"],"1962-01-01")
        self.assertEqual(plan["requested_sample_end"],"2025-12-31")
        self.assertEqual(plan["last_complete_future_formation"],2020)
        self.assertEqual(len(plan["coverage_adjustments"]),3)
        with self.assertRaisesRegex(ValueError,"Incomplete requested coverage"):
            plan_run(self.raw,1962,2025,"nyse",strict_coverage=True)

    def test_subsample_returns_and_accounting_stay_inside_calendar_window(self):
        early = plan_run(self.raw,1963,1989,"nyse_amex",strict_coverage=True)
        recent = plan_run(self.raw,1990,2025,"nyse_amex")
        self.assertEqual((early["formation_start"],early["formation_end"]),(1963,1988))
        self.assertEqual((early["return_end"],early["outcome_end_year"]),("1989-04-30",1989))
        self.assertEqual(early["last_complete_future_formation"],1985)
        self.assertEqual(recent["first_holding_month"],"1990-05")
        self.assertLess(early["last_holding_month"],recent["first_holding_month"])

    def test_new_cache_enables_1962_and_2025_without_changing_code(self):
        write_sidecars(self.raw,first_account="1956-01-31",first_stock="1956-01-31",
                       last_stock="2025-12-31",last_index="2025-12-31")
        plan = plan_run(self.raw,1962,2025,"nyse_amex_nasdaq",strict_coverage=True)
        self.assertEqual((plan["formation_start"],plan["formation_end"],plan["return_end"]),(1962,2024,"2025-04-30"))
        self.assertEqual(plan["coverage_adjustments"],[])

    def test_market_and_accounting_endpoints_both_limit_full_years(self):
        write_sidecars(self.raw,last_index="2023-03-31")
        self.assertEqual(plan_run(self.raw,1990,2025,"nyse")["return_end"],"2022-04-30")
        write_sidecars(self.raw,last_account="2020-12-31")
        self.assertEqual(plan_run(self.raw,1990,2025,"nyse")["return_end"],"2022-04-30")

    def test_missing_files_filtered_cache_and_short_inference_are_rejected(self):
        with self.assertRaisesRegex(ValueError,"nine formation"):
            plan_run(self.raw,1990,1998,"nyse")
        stock_info = self.raw/"crsp_monthly.json"
        info = json.loads(stock_info.read_text(encoding="utf-8"))
        info["exchange_and_share_filters_applied"] = True
        stock_info.write_text(json.dumps(info),encoding="utf-8")
        with self.assertRaisesRegex(ValueError,"broad raw extract"):
            plan_run(self.raw,1962,2025,"nyse")
        (self.raw/"market_returns.parquet").unlink()
        with self.assertRaises(FileNotFoundError):
            plan_run(self.raw,1962,2025,"nyse")

    def test_batches_cover_requested_universes_and_never_build_submission(self):
        for question,universes in [(2,["nyse","nyse_amex_nasdaq"]),(3,["nyse_amex","nyse_amex"])]:
            with patch("sys.argv",["batch","--raw-data-dir",str(self.raw)]), patch("core.lsv1994.runs.run_lsv") as runner, redirect_stdout(io.StringIO()):
                batch_main(question)
            self.assertEqual([call.kwargs["universe"] for call in runner.call_args_list],universes)
            self.assertEqual(len({call.kwargs["run_tag"] for call in runner.call_args_list}),2)
            for call in runner.call_args_list:
                self.assertFalse(call.kwargs["build_submission"])
                self.assertTrue(call.kwargs["allow_partial_outcomes"])
                self.assertIn("requested_sample_end",call.kwargs["run_plan"])

    def test_dry_run_and_strict_rejection_do_not_start_any_case(self):
        for options in [["--dry-run"],["--strict-coverage"]]:
            with patch("sys.argv",["batch","--raw-data-dir",str(self.raw),*options]), patch("core.lsv1994.runs.run_lsv") as runner, redirect_stdout(io.StringIO()):
                if options == ["--strict-coverage"]:
                    with self.assertRaises(ValueError):
                        batch_main(2)
                else:
                    batch_main(2)
                runner.assert_not_called()

    def test_universe_eligibility_is_at_formation(self):
        frame = pd.DataFrame({"permno":[1,2,3,4],"date":pd.Timestamp("1990-04-30"),
                              "ret":.01,"prc":10.,"shrout":100.,"shrcd":[10,11,10,12],"exchcd":[1,2,3,1]})
        for universe,count in [("nyse",1),("nyse_amex",2),("nyse_amex_nasdaq",3)]:
            result = prepare_crsp(frame,universe)
            self.assertEqual(result.eligible.sum(),count)
            self.assertEqual(len(result),4)  # Later exchange changes must not discard holding data.


class OutcomeCoverageTests(unittest.TestCase):
    def test_partial_future_uses_same_eligible_cohorts_for_every_transition(self):
        args = growth_inputs()
        summary,_,growth,cohorts,_ = build_table5(*args,1968,1971,allow_partial_outcomes=True,outcome_end_year=1973)
        validate_summary(summary,5)
        self.assertTrue(summary.loc[summary.section.ne("C"),"n_cohorts"].eq(4).all())
        self.assertTrue(summary.loc[summary.section.eq("C"),"n_cohorts"].eq(2).all())
        self.assertTrue(growth.loc[growth.relative_end.gt(0),"n_cohorts"].eq(2).all())
        self.assertEqual(set(cohorts.loc[cohorts.relative_end.gt(0),"formation_year"]),{1968,1969})
        np.testing.assert_allclose(summary.loc[summary.metric.eq("AEG(0,5)"),"value"],.05,atol=1e-12)

    def test_later_outcomes_cannot_affect_bounded_subsample(self):
        args = list(growth_inputs())
        before = build_table5(*args,1968,1971,allow_partial_outcomes=True,outcome_end_year=1973)[0]
        args[3] = args[3].copy()
        args[3].loc[args[3].acct_year.gt(1973),["earn","cf","sale"]] *= 1000
        args[5] = args[5].copy()
        args[5].loc[args[5].year.gt(1973),"me"] *= 1000
        after = build_table5(*args,1968,1971,allow_partial_outcomes=True,outcome_end_year=1973)[0]
        pd.testing.assert_frame_equal(before,after)

    def test_no_future_cohorts_are_explicitly_unavailable(self):
        summary,*_ = build_table5(*growth_inputs(),1968,1971,allow_partial_outcomes=True,outcome_end_year=1971)
        future = summary.loc[summary.section.eq("C")]
        self.assertEqual(len(future),24)
        self.assertTrue(future.value.isna().all())
        self.assertTrue(future.n_cohorts.eq(0).all())
        self.assertTrue(future.status.eq("five-year accounting outcomes unavailable").all())
        validate_summary(summary,5)

    def test_strict_original_outcome_requirement_is_preserved(self):
        with self.assertRaisesRegex(ValueError,"April market equity"):
            build_table5(*growth_inputs(),1968,1971,outcome_end_year=1973)

    def test_entire_missing_transition_is_a_coverage_error(self):
        args = list(growth_inputs())
        args[3] = args[3].loc[args[3].acct_year.ne(1970)]
        with self.assertRaisesRegex(ValueError,"accounting growth cohorts"):
            build_table5(*args,1968,1971,allow_partial_outcomes=True,outcome_end_year=1973)


class ExtendedFigureTests(unittest.TestCase):
    def test_recent_recession_overlap_uses_full_may_april_window(self):
        years = [2001,2007,2009,2019,2020]
        cohorts = pd.DataFrame({"strategy":"C/P-GS","horizon":1,"formation_year":years,"spread":.1})
        dates = pd.date_range("2001-05-31","2021-04-30",freq="ME")
        panel = build_figure2(cohorts,pd.DataFrame({"date":dates,"ewretd":-.01}))
        self.assertEqual(panel.recession_overlap.tolist(),[True,True,True,True,False])
        self.assertTrue(panel.market_decline.all())
        np.testing.assert_allclose(panel.ew_market_return,.99**12-1)

    def test_unverified_chronology_endpoint_is_rejected(self):
        cohorts = pd.DataFrame({"strategy":["C/P-GS"],"horizon":[1],"formation_year":[2026],"spread":[.1]})
        dates = pd.date_range("2026-05-31","2027-04-30",freq="ME")
        with self.assertRaisesRegex(ValueError,"verified recession chronology"):
            build_figure2(cohorts,pd.DataFrame({"date":dates,"ewretd":.01}))

    def test_saved_report_contains_figure_without_submission_authoring(self):
        rows = []
        for strategy,ports in [("C/P tails",[1,2]),("C/P-GS",[3,7]),("B/M tails",[1,2]),("B/M",[1,10])]:
            for year in range(2000,2011):
                for horizon in range(1,min(5,2011-year)+1):
                    rows.extend([(strategy,year,ports[0],horizon,.1+.001*(year-2000)),
                                 (strategy,year,ports[1],horizon,.2+.001*(year-2000))])
        yearly = pd.DataFrame(rows,columns=["strategy","formation_year","portfolio","hold_year","ret"])
        summary,cohorts = build_table6(yearly,2000,2010,"2011-04-30")
        market = pd.DataFrame({"date":pd.date_range("2000-05-31","2011-04-30",freq="ME"),"ewretd":.01})
        figure = build_figure2(cohorts,market)
        with tempfile.TemporaryDirectory(prefix=".test_lsv_report_",dir=root_path) as directory:
            folder = Path(directory)
            outputs = {"table6_summary.csv":summary,"table6_cohort_returns.csv":cohorts,"figure2_data.csv":figure}
            for name,data in outputs.items():
                data.to_csv(folder/name,index=False)
            info = {"paper":"lsv1994","table":6,"universe":"nyse","formation_start":2000,
                    "formation_end":2010,"return_end":"2011-04-30","settings":{},
                    "run_plan":{"requested_sample_end":"2011-12-31"},
                    "output_sha256":{name:hashlib.sha256((folder/name).read_bytes()).hexdigest() for name in outputs}}
            (folder/"run_info.json").write_text(json.dumps(info),encoding="utf-8")
            html = report_table(6,folder).read_text(encoding="utf-8")
            self.assertIn("data:image/png;base64,",html)
            self.assertIn("Requested sample and effective coverage",html)
            self.assertIn("nyse | April formations 2000-2010",html)
            self.assertTrue((folder/"reports/figure2.png").is_file())


if __name__ == "__main__":
    unittest.main()
