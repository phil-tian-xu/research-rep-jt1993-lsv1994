"""Economic checks for the added LSV tables; no WRDS or large datasets."""

from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import re

import numpy as np
import pandas as pd

from core.lsv1994.replication.table2 import make_pair_constituents, summarize_table2
from core.lsv1994.replication.table4 import build_table4
from core.lsv1994.replication.table5 import aggregate_growth, build_table5
from core.lsv1994.replication.table6 import hh_mean_test, make_tail_constituents, build_table6, build_figure2
from core.lsv1994.reporter.tables import validate_summary
from core.submission.lsv1994 import generate_submission, load_results, load_extended_cases, EXTENSION_CASES
from utils.path_info import root_path


class DoubleSortTests(unittest.TestCase):
    def test_independent_common_sample_sorts_cover_all_nine_intersections(self):
        grid=pd.DataFrame([(a,b) for a in range(1,11) for b in range(1,11)],columns=["a","b"])
        formation=grid.assign(permno=np.arange(100),formation_year=1968,cp=grid.a,ep=grid.a+grid.b/100,
                              bm=grid.a,gs=grid.b)
        members=make_pair_constituents(formation)
        cp=members.loc[members.strategy.eq("C/P-GS")].merge(formation,on=["permno","formation_year"])
        self.assertEqual(cp.portfolio.nunique(),9)
        self.assertTrue(cp.groupby("cp").portfolio.apply(lambda s: ((s-1)//3+1).nunique()).eq(1).all())
        formation.loc[0,"cp"]=-1
        formation.loc[1,"gs"]=np.nan
        changed=make_pair_constituents(formation)
        self.assertFalse(changed.loc[changed.strategy.eq("C/P-GS"),"permno"].isin([0,1]).any())

    def test_table2_horizon_averaging_has_known_result(self):
        rows=[]
        for strategy in ["C/P-GS","E/P-GS","B/M-GS","E/P-B/M","B/M-C/P"]:
            for y in range(1968,1973):
                for h in range(1,6):
                    if y+h-1>1972:continue
                    for p in range(1,10):
                        rows.append((strategy,y,p,h,.01*p+.001*h,.01*p+.001*h-.02))
        yearly=pd.DataFrame(rows,columns=["strategy","formation_year","portfolio","hold_year","ret","saar"])
        summary=summarize_table2(yearly,1968,1972,"1973-04-30")
        self.assertEqual(len(summary),360)
        row=summary.loc[summary.strategy.eq("C/P-GS")&summary.stat.eq("AR")&summary.group1.eq(1)&summary.group2.eq(1)].iloc[0]
        self.assertAlmostEqual(row["return"],.013)
        with self.assertRaisesRegex(ValueError,"incomplete"):
            summarize_table2(yearly.iloc[1:],1968,1972,"1973-04-30")


class RegressionTests(unittest.TestCase):
    def test_annual_ols_and_time_series_inference_recover_known_coefficients(self):
        rng=np.random.default_rng(7)
        formations,returns=[],[]
        for k,y in enumerate(range(1968,1974)):
            n=200
            frame=pd.DataFrame({"permno":range(n),"formation_year":y,"gs":rng.uniform(size=n),
                                "bm":rng.normal(size=n),"me_apr":np.exp(rng.normal(5,1,n)),
                                "cp":rng.normal(.1,.2,n),"ep":rng.normal(.05,.15,n)})
            beta=.03+.002*k
            ret=.1+.02*frame.gs+.01*frame.bm-.005*np.log(frame.me_apr)+beta*frame.cp.clip(lower=0)-.04*frame.cp.lt(0)
            returns.append(pd.DataFrame({"permno":range(n),"start_year":y,"annual_ret":ret}))
            formations.append(frame)
        summary,coefs=build_table4(pd.concat(formations),pd.concat(returns))
        row=summary.loc[summary.model.eq(8)&summary.term.eq("cp_positive")].iloc[0]
        expected=.03+.002*np.arange(6)
        self.assertAlmostEqual(row["mean"],expected.mean(),places=12)
        self.assertAlmostEqual(row["se"],expected.std(ddof=1)/np.sqrt(6),places=12)
        self.assertTrue(summary.n_years.eq(6).all())
        self.assertEqual(len(coefs),32*6)

    def test_absent_negative_category_leaves_dummy_unestimated_and_preserves_slopes(self):
        rng=np.random.default_rng(12)
        formations,returns=[],[]
        for k,year in enumerate(range(1963,1966)):
            n=200
            frame=pd.DataFrame({"permno":range(n),"formation_year":year,"gs":rng.uniform(size=n),
                                "bm":rng.normal(size=n),"me_apr":np.exp(rng.normal(5,1,n)),
                                "cp":rng.uniform(.01,.5,n) if k==0 else rng.normal(.1,.2,n),
                                "ep":rng.normal(.05,.15,n)})
            beta=.03+.002*k
            ret=.1+.02*frame.gs+.01*frame.bm-.005*np.log(frame.me_apr)+beta*frame.cp.clip(lower=0)-.04*frame.cp.lt(0)
            formations.append(frame)
            returns.append(pd.DataFrame({"permno":range(n),"start_year":year,"annual_ret":ret}))
        summary,coefs=build_table4(pd.concat(formations),pd.concat(returns))
        first=coefs.loc[coefs.model.eq(8)&coefs.formation_year.eq(1963)].set_index("term")
        self.assertTrue(pd.isna(first.loc["cp_negative","coefficient"]))
        self.assertEqual(first.loc["cp_negative","status"],"no negative-ratio observations")
        for term,value in [("intercept",.1),("gs",.02),("bm",.01),("size",-.005),("cp_positive",.03)]:
            self.assertAlmostEqual(first.loc[term,"coefficient"],value,places=12)
        model8=summary.loc[summary.model.eq(8)].set_index("term")
        self.assertEqual(model8.loc["cp_negative","n_years"],2)
        self.assertEqual(model8.loc["cp_negative","n_unidentified_years"],1)
        self.assertEqual(model8.loc["cp_positive","n_years"],3)
        self.assertAlmostEqual(model8.loc["cp_negative","mean"],-.04,places=12)
        self.assertAlmostEqual(model8.loc["cp_positive","mean"],.032,places=12)
        # Genuine collinearity beyond the absent negative category remains an error.
        collinear=pd.concat(formations)
        collinear["bm"]=collinear["gs"]
        with self.assertRaisesRegex(ValueError,"unidentified regressors"):
            build_table4(collinear,pd.concat(returns))

    def test_missing_ratio_is_not_encoded_as_zero(self):
        rng=np.random.default_rng(4)
        frames=[]
        for y in [1968,1969]:
            frame=pd.DataFrame({"permno":range(100),"formation_year":y,"gs":rng.uniform(size=100),
                                "bm":rng.uniform(size=100),"me_apr":np.exp(rng.normal(5,1,100)),
                                "cp":rng.normal(0,.2,100),"ep":rng.normal(0,.2,100)})
            frame.loc[0,"ep"]=np.nan
            frames.append(frame)
        formations=pd.concat(frames)
        annual=formations[["permno","formation_year"]].rename(columns={"formation_year":"start_year"}).assign(annual_ret=rng.normal(size=200))
        _,coefs=build_table4(formations,annual)
        self.assertTrue(coefs.loc[coefs.model.eq(9),"n_stocks"].eq(99).all())
        self.assertTrue(coefs.loc[coefs.model.eq(8),"n_stocks"].eq(100).all())


class GrowthTests(unittest.TestCase):
    def test_growth_uses_ratio_of_average_cohort_levels(self):
        paired=pd.DataFrame({"classification":"B/M","leg":"Value","relative_end":0,"variable":"earn",
                             "formation_year":[1968,1968,1969],"permno":[1,2,3],
                             "base_per_dollar":[-1.,3.,10.],"next_per_dollar":[1.,5.,20.]})
        pooled,cohort=aggregate_growth(paired)
        self.assertAlmostEqual(pooled.iloc[0].gross_growth,11.5/5.5)
        self.assertEqual(cohort.n_pairs.tolist(),[2,1])
        self.assertNotAlmostEqual(pooled.iloc[0].gross_growth,2.5)

    def test_negative_aggregate_level_is_not_given_a_fake_growth_rate(self):
        paired=pd.DataFrame({"classification":"B/M","leg":"Value","relative_end":0,"variable":"earn",
                             "formation_year":[1968,1968],"permno":[1,2],
                             "base_per_dollar":[-1.,-3.],"next_per_dollar":[1.,5.]})
        pooled,_=aggregate_growth(paired)
        self.assertTrue(pd.isna(pooled.iloc[0].gross_growth))

    def test_table5_outcomes_are_separate_and_market_equity_buffer_is_required(self):
        n=8
        accounts=pd.DataFrame([{"gvkey":str(i),"acct_year":y,"earn":10*(1.05**(y-1962)),
                                "cf":15*(1.04**(y-1962)),"sale":100*(1.03**(y-1962))}
                               for i in range(n) for y in range(1962,1973)])
        formation=pd.DataFrame({"permno":range(n),"gvkey":[str(i) for i in range(n)],"formation_year":1968,
                                "me_apr":100.,"ep":.1,"cp":.15,"bm":.5,"sale":100.,"dvc":2.})
        deciles=pd.DataFrame({"permno":range(4),"formation_year":1968,"strategy":"B/M","portfolio":[1,1,10,10]})
        pairs=pd.DataFrame({"permno":range(4,8),"formation_year":1968,"strategy":"C/P-GS","portfolio":[3,3,7,7]})
        caps=pd.DataFrame([(i,y,100.) for i in range(n) for y in range(1963,1973)],columns=["permno","year","me"])
        monthly=pd.DataFrame([(i,m,.01) for i in range(n) for m in pd.period_range("1965-05","1968-04",freq="M")],
                             columns=["permno","ym","ret"])
        summary,*_=build_table5(formation,deciles,pairs,accounts,monthly,caps,1968,1968)
        self.assertEqual(len(summary),64)
        value=summary.loc[summary.metric.eq("AEG(0,5)"),"value"]
        np.testing.assert_allclose(value,.05,atol=1e-12)
        past=summary.loc[summary.metric.eq("RETURN(-3,0)"),"value"]
        np.testing.assert_allclose(past,1.01**36-1,atol=1e-12)
        future=accounts.copy()
        future.loc[future.acct_year.ge(1968),"earn"]*=2
        changed,*_=build_table5(formation,deciles,pairs,future,monthly,caps,1968,1968)
        pd.testing.assert_frame_equal(summary.loc[summary.section.eq("A")].reset_index(drop=True),
                                      changed.loc[changed.section.eq("A")].reset_index(drop=True))
        self.assertFalse(np.allclose(value,changed.loc[changed.metric.eq("AEG(0,5)"),"value"]))
        with self.assertRaisesRegex(ValueError,"April market equity"):
            build_table5(formation,deciles,pairs,accounts,monthly,caps.loc[caps.year.lt(1972)],1968,1968)


class InferenceTests(unittest.TestCase):
    def test_hh_zero_lag_equals_conventional_mean_test(self):
        values=np.array([.1,.2,-.1,.4,.3,.15])
        result=hh_mean_test(values,0)
        self.assertAlmostEqual(result["se"],values.std(ddof=1)/np.sqrt(len(values)))

    def test_hh_uses_uniform_lag_weights_and_exposes_negative_variance(self):
        values=np.array([.1,.2,-.1,.4,.3,.15])
        residual=values-values.mean()
        variance=(residual@residual+2*(residual[1:]@residual[:-1]))/(6*5)
        self.assertAlmostEqual(hh_mean_test(values,1)["se"],np.sqrt(variance))
        result=hh_mean_test([1.,-1.,1.,-1.,1.,-1.],1)
        self.assertTrue(pd.isna(result["t"]))
        self.assertEqual(result["status"],"nonpositive HH variance")

    def test_cumulative_spread_compounds_legs_and_respects_cutoff(self):
        rows=[]
        for strategy,ports in [("C/P tails",[1,2]),("C/P-GS",[3,7]),("B/M tails",[1,2]),("B/M",[1,10])]:
            for y in range(1968,1979):
                for h in range(1,6):
                    if y+h-1>1978:continue
                    rows.extend([(strategy,y,ports[0],h,.1+.001*(y-1968)),(strategy,y,ports[1],h,.2+.001*(y-1968))])
        yearly=pd.DataFrame(rows,columns=["strategy","formation_year","portfolio","hold_year","ret"])
        summary,cohorts=build_table6(yearly,1968,1978,"1979-04-30")
        first=cohorts.loc[cohorts.strategy.eq("C/P-GS")&cohorts.formation_year.eq(1968)&cohorts.horizon.eq(5)].iloc[0]
        self.assertAlmostEqual(first.spread,1.2**5-1.1**5)
        self.assertNotAlmostEqual(first.spread,1.1**5-1)
        self.assertEqual(summary.loc[summary.strategy.eq("C/P-GS"),"n_cohorts"].tolist(),[11,9,7])

    def test_figure2_uses_equal_weighted_index_and_matching_holding_window(self):
        cohorts=pd.DataFrame({"strategy":["C/P-GS"],"horizon":[1],"formation_year":[1969],"spread":[.1]})
        months=pd.date_range("1969-05-31","1970-04-30",freq="ME")
        market=pd.DataFrame({"date":months,"ewretd":-.01,"vwretd":.01})
        result=build_figure2(cohorts,market)
        self.assertAlmostEqual(result.iloc[0].ew_market_return,.99**12-1)
        self.assertTrue(result.iloc[0].market_decline)
        self.assertTrue(result.iloc[0].recession_overlap)
        with self.assertRaisesRegex(ValueError,"twelve"):
            build_figure2(cohorts,market.iloc[1:])


class SubmissionTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory(prefix=".test_lsv_q21_",dir=root_path)
        self.addCleanup(self.temporary.cleanup)
        self.directory=Path(self.temporary.name)
        self.result_dirs={t:self.directory/f"table{t}" for t in (1,2,4,5,6)}
        paper_dir=Path(root_path)/"references/lsv1994/paper_tables"
        paper_cohorts=pd.read_csv(paper_dir/"table6_cohorts.csv")
        extra=paper_cohorts.loc[paper_cohorts.strategy.eq("B/M tails")].assign(strategy="B/M")
        cohorts=pd.concat([paper_cohorts,extra],ignore_index=True)
        for table,path in self.result_dirs.items():
            path.mkdir()
            frame=pd.read_csv(paper_dir/f"table{table}.csv")
            if table==4:frame["se"]=(frame["mean"]/frame["t"]).abs()
            if table==5:frame["status"]="ok"
            if table==6:
                frame=pd.concat([frame,frame.loc[frame.strategy.eq("B/M tails")].assign(strategy="B/M")],ignore_index=True)
                frame["n_cohorts"]=frame.horizon.map({1:22,3:20,5:18})
                frame["status"]="ok"
            files={f"table{table}_summary.csv":frame}
            if table==6:
                files["table6_cohort_returns.csv"]=cohorts
                files["figure2_data.csv"]=paper_cohorts.loc[paper_cohorts.strategy.eq("C/P-GS")&paper_cohorts.horizon.eq(1)].assign(recession_overlap=False,market_decline=False)
            for name,data in files.items():data.to_csv(path/name,index=False)
            metadata={"paper":"lsv1994","table":table,"formation_start":1968,"formation_end":1989,
                      "return_end":"1990-04-30","universe":"nyse_amex","book_equity":"ceq","accounting_currency":None,
                      "raw_inputs":{"crsp_monthly":{"path":"example/raw/crsp_monthly.parquet"}},
                      "diagnostics":{"formation_account_currencies":{"USD":100}},
                      "table1_foundation":str(self.result_dirs[1]),
                      "output_sha256":{name:hashlib.sha256((path/name).read_bytes()).hexdigest() for name in files}}
            (path/"run_info.json").write_text(json.dumps(metadata),encoding="utf-8")

    def make_extended_cases(self):
        """Small saved-output fixtures; raw inputs deliberately do not exist."""
        configurations={"nyse":(1968,1989,1990,0,1993),"nyse_amex_nasdaq":(1968,1989,1990,0,1993),
                        "early":(1968,1988,1989,0,1989),"recent":(1990,2011,2012,22,2012)}
        cases={}
        for key,(start,end,cutoff,shift,outcome) in configurations.items():
            question,_,universe,requested_start,requested_end=EXTENSION_CASES[key]
            cases[key]={table:self.directory/key/f"table{table}" for table in self.result_dirs}
            plan={"question":question,"universe":universe,"requested_sample_start":requested_start,
                  "requested_sample_end":requested_end,"formation_start":start,"formation_end":end,
                  "return_end":f"{cutoff}-04-30","outcome_end_year":outcome}
            for table,folder in cases[key].items():
                folder.mkdir(parents=True)
                info=json.loads((self.result_dirs[table]/"run_info.json").read_text(encoding="utf-8"))
                frame=pd.read_csv(self.result_dirs[table]/f"table{table}_summary.csv")
                info.update(formation_start=start,formation_end=end,return_end=plan["return_end"],universe=universe,
                            table1_foundation=str(cases[key][1]),run_plan=plan,settings={})
                files={f"table{table}_summary.csv":frame}
                if table==5:
                    info["settings"]["outcome_end_year"]=outcome
                    frame["n_cohorts"]=np.where(frame.section.eq("C"),min(end,outcome-4)-start+1,end-start+1)
                if table==6:
                    frame["n_cohorts"]=cutoff-frame.horizon-start+1
                    frame["se"]=(frame["mean"]/frame["t"]).abs()
                    cohorts=pd.read_csv(self.result_dirs[6]/"table6_cohort_returns.csv")
                    cohorts["formation_year"]+=shift
                    cohorts=cohorts.loc[cohorts.formation_year.add(cohorts.horizon).le(cutoff)]
                    figure=cohorts.loc[cohorts.strategy.eq("C/P-GS")&cohorts.horizon.eq(1)].assign(recession_overlap=False,market_decline=False)
                    files.update({"table6_cohort_returns.csv":cohorts,"figure2_data.csv":figure})
                for name,data in files.items():
                    data.to_csv(folder/name,index=False)
                info["output_sha256"]={name:hashlib.sha256((folder/name).read_bytes()).hexdigest() for name in files}
                (folder/"run_info.json").write_text(json.dumps(info),encoding="utf-8")
        return cases

    def test_full_submission_embeds_all_cases_and_figures_without_raw_access(self):
        cases=self.make_extended_cases()
        with patch("pandas.read_parquet",side_effect=AssertionError("Submission must not load raw data")):
            output=generate_submission(self.result_dirs,extension_dirs=cases,output_path=self.directory/"full.html")
        html=output.read_text(encoding="utf-8")
        self.assertEqual(html.count("data:image/png;base64,"),15)
        self.assertEqual(html.count('class="table-panel"'),79)
        self.assertEqual(html.count('class="case-select"'),2)
        self.assertEqual(html.count('class="case-panel"'),4)
        ids=re.findall(r'\bid="([^" ]+)"',html)
        self.assertEqual(len(ids),len(set(ids)))
        for anchor in ["coverage","exchange-samples","subperiods","filters"]:
            self.assertIn('id="'+anchor+'"',html)
        self.assertIn("not fully covered",html)
        self.assertIn("No formal test",html)
        self.assertIn(".case-panel[hidden] { display: block; }",html)
        self.assertNotIn('src="http',html)
        self.assertNotIn("$extension",html)
        manifest=json.loads(output.with_name("full_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(set(manifest["cases"]),{"q21",*EXTENSION_CASES})
        self.assertEqual(manifest["submission_sha256"],hashlib.sha256(output.read_bytes()).hexdigest())
        self.assertTrue(manifest["saved_results_only"])

    def test_missing_extended_case_is_rejected(self):
        with self.assertRaisesRegex(ValueError,"both Q2-2"):
            load_extended_cases({})

    def test_extended_followup_cannot_cross_subsample_boundary(self):
        cases=self.make_extended_cases()
        for folder in cases["early"].values():
            info=json.loads((folder/"run_info.json").read_text(encoding="utf-8"))
            info["run_plan"]["outcome_end_year"]=1990
            (folder/"run_info.json").write_text(json.dumps(info),encoding="utf-8")
        with self.assertRaisesRegex(ValueError,"outcomes exceed"):
            load_extended_cases(cases)

    def test_different_vintage_across_extended_cases_is_rejected(self):
        cases=self.make_extended_cases()
        for folder in cases["nyse_amex_nasdaq"].values():
            info=json.loads((folder/"run_info.json").read_text(encoding="utf-8"))
            info["raw_inputs"]["crsp_monthly"]["modified_ns"]=123
            (folder/"run_info.json").write_text(json.dumps(info),encoding="utf-8")
        with self.assertRaisesRegex(ValueError,"same raw-data vintage"):
            load_extended_cases(cases)

    def test_incorrect_future_cohort_count_is_rejected(self):
        cases=self.make_extended_cases()
        folder=cases["recent"][5]
        frame=pd.read_csv(folder/"table5_summary.csv")
        frame.loc[frame.section.eq("C"),"n_cohorts"]=22
        frame.to_csv(folder/"table5_summary.csv",index=False)
        info=json.loads((folder/"run_info.json").read_text(encoding="utf-8"))
        info["output_sha256"]["table5_summary.csv"]=hashlib.sha256((folder/"table5_summary.csv").read_bytes()).hexdigest()
        (folder/"run_info.json").write_text(json.dumps(info),encoding="utf-8")
        with self.assertRaisesRegex(ValueError,"follow-up cohort counts"):
            load_extended_cases(cases)

    def test_figure_with_shifted_year_labels_is_rejected(self):
        folder=self.result_dirs[6]
        frame=pd.read_csv(folder/"figure2_data.csv")
        frame["formation_year"]+=1
        frame.to_csv(folder/"figure2_data.csv",index=False)
        info=json.loads((folder/"run_info.json").read_text(encoding="utf-8"))
        info["output_sha256"]["figure2_data.csv"]=hashlib.sha256((folder/"figure2_data.csv").read_bytes()).hexdigest()
        (folder/"run_info.json").write_text(json.dumps(info),encoding="utf-8")
        with self.assertRaisesRegex(ValueError,"Figure 2 disagrees"):
            load_results(self.result_dirs)

    def test_submission_is_complete_and_offline_without_raw_files(self):
        output=generate_submission(self.result_dirs,output_path=self.directory/"submission.html")
        html=output.read_text(encoding="utf-8")
        self.assertEqual(html.count("data:image/png;base64,"),5)
        for anchor in ["table-i","table-ii","table-iv","table-v","table-vi","filters"]:
            self.assertIn('id="'+anchor+'"',html)
        self.assertIn("@media print",html)
        self.assertIn("Published paper",html)
        self.assertNotIn('src="http',html)
        self.assertNotIn('href="http',html)
        self.assertNotIn("$table",html)

    def test_missing_table_and_mixed_samples_are_rejected(self):
        with self.assertRaisesRegex(ValueError,"requires"):
            load_results({t:p for t,p in self.result_dirs.items() if t!=4})
        path=self.result_dirs[4]/"run_info.json"
        info=json.loads(path.read_text(encoding="utf-8"))
        info["return_end"]="1991-04-30"
        path.write_text(json.dumps(info),encoding="utf-8")
        with self.assertRaisesRegex(ValueError,"different samples"):
            load_results(self.result_dirs)

    def test_same_path_with_a_different_data_vintage_is_rejected(self):
        path=self.result_dirs[4]/"run_info.json"
        info=json.loads(path.read_text(encoding="utf-8"))
        info["raw_inputs"]["crsp_monthly"]["modified_ns"]=123
        path.write_text(json.dumps(info),encoding="utf-8")
        with self.assertRaisesRegex(ValueError,"data vintages"):
            load_results(self.result_dirs)

    def test_modified_saved_output_is_rejected(self):
        path=self.result_dirs[5]/"table5_summary.csv"
        path.write_text("changed",encoding="utf-8")
        with self.assertRaisesRegex(ValueError,"Saved output changed"):
            load_results(self.result_dirs)


if __name__=="__main__":
    unittest.main()
