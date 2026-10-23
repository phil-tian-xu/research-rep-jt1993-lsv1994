"""One LSV run and explicit calendar-window plans for Q2-2/Q2-3 batches."""

import argparse
import json
from pathlib import Path

import pandas as pd

from utils.path_info import root_path, data_path, results_path
from .replication._formation import UNIVERSES
from .replication.table1 import run_table1
from .replication.workflow import run_extensions
from .reporter.table1 import report_table1
from .reporter.tables import report_table


DEFAULT_EXTENDED_RAW = Path(data_path)/"raw/lsv1994/nyse_amex_nasdaq_19570101_20241231_comp19570101_20241231"


def plan_run(raw_data_dir, sample_start_year, sample_end_year, universe, *, strict_coverage=False):
    """Plan full May-April holding years wholly inside the requested calendar sample.

    Warm-up data may precede the sample. Returns and future accounting outcomes
    may not exceed its endpoint. Available-data adjustments are printed/saved;
    strict mode rejects missing history or endpoint coverage instead.
    """
    if universe not in UNIVERSES or sample_start_year >= sample_end_year:
        raise ValueError("Choose a supported universe and an ordered multi-year sample.")
    raw = Path(raw_data_dir)
    if not raw.is_absolute():
        raw = Path(root_path)/raw
    metadata = {}
    for dataset in ("crsp_monthly", "compustat_annual", "market_returns", "ccm_links"):
        parquet = raw/f"{dataset}.parquet"
        sidecar = parquet.with_suffix(".json")
        if not parquet.is_file() or not sidecar.is_file():
            raise FileNotFoundError(f"Missing {parquet} or its coverage sidecar; prepare the extended LSV cache first.")
        metadata[dataset] = json.loads(sidecar.read_text(encoding="utf-8"))
    stocks, accounts, market = (metadata[name] for name in ("crsp_monthly","compustat_annual","market_returns"))
    if stocks.get("exchange_and_share_filters_applied", False):
        raise ValueError("Batch runs require the broad raw extract; apply exchange filters at formation.")
    first_stock, last_stock = pd.Timestamp(stocks["first_date"]), pd.Timestamp(stocks["last_date"])
    first_account, last_account = pd.Timestamp(accounts["first_date"]), pd.Timestamp(accounts["last_date"])
    first_index, last_index = pd.Timestamp(market["first_date"]), pd.Timestamp(market["last_date"])
    requested_end = pd.Timestamp(sample_end_year,12,31)
    first_cap_april = first_stock.year + int(first_stock.month > 4)
    earliest = max(sample_start_year, first_account.year+6, first_stock.year+1,
                   first_cap_april+5, first_index.year+int(first_index.month>5))
    available_end = min(requested_end, last_stock, last_index,
                        pd.Timestamp(last_account.year+2,4,30))
    holding_end_year = available_end.year-int(available_end.month<4)
    holding_end = pd.Timestamp(holding_end_year,4,30)
    latest = holding_end_year-1
    gaps = []
    if earliest > sample_start_year:
        gaps.append(f"First common formation is {earliest}; requested {sample_start_year}. GS needs accounts from {sample_start_year-6}.")
    if last_stock < requested_end:
        gaps.append(f"CRSP ends {last_stock.date()}, before requested {requested_end.date()}.")
    if last_index < requested_end:
        gaps.append(f"CRSP index ends {last_index.date()}, before requested {requested_end.date()}.")
    if last_account.year < sample_end_year-2:
        gaps.append(f"Accounts end in {last_account.year}; the latest full holding year needs preformation accounts through {sample_end_year-2}.")
    if strict_coverage and gaps:
        raise ValueError("Incomplete requested coverage: " + " ".join(gaps))
    if latest-earliest < 8:
        raise ValueError("Need at least nine formation cohorts for five-year HH inference with four lags.")
    source = stocks.get("data_source","unknown")
    outcome_year = min(sample_end_year,last_account.year,
                       last_stock.year-int(last_stock.month<4))
    return {
        "requested_sample_start":f"{sample_start_year}-01-01",
        "requested_sample_end":str(requested_end.date()), "universe":universe,
        "raw_data_dir":str(raw), "data_source":source,
        "formation_start":earliest, "formation_end":latest, "return_end":str(holding_end.date()),
        "first_holding_month":f"{earliest}-05", "last_holding_month":f"{holding_end_year}-04",
        "outcome_end_year":outcome_year,
        "last_complete_future_formation":min(latest,outcome_year-4),
        "warmup_accounts_start":earliest-6, "warmup_market_start":f"{earliest-5}-04",
        "coverage_adjustments":gaps, "strict_coverage":strict_coverage,
        "annual_window_policy":"Only complete May-April years inside the requested calendar window; no partial-year annualization.",
        "outcome_policy":"Future accounting growth uses the same fully observed five-year cohorts at every transition, within the sample endpoint.",
    }


def run_lsv(formation_start, formation_end, return_end, raw_data_dir, *, universe="nyse_amex",
            tables=(1,2,4,5,6), book_equity="ceq", accounting_currency=None,
            outcome_market_data_dir=None, allow_partial_outcomes=False, outcome_end_year=None,
            run_replication=True, show_plots=False, build_submission=False,
            run_tag=None, run_plan=None, output_root=None):
    """Run one case; submission authoring is off unless explicitly selected."""
    raw = Path(raw_data_dir)
    if not raw.is_absolute():
        raw=Path(root_path)/raw
    tables=(tables,) if isinstance(tables,int) else tuple(tables)
    if not tables or len(set(tables))!=len(tables) or set(tables)-{1,2,4,5,6}:
        raise ValueError("Select unique table numbers from 1, 2, 4, 5, 6.")
    if run_tag is None:
        run_tag=f"{raw.name}_{formation_start}_{formation_end}_to{pd.Timestamp(return_end):%Y%m}_{universe}_{book_equity}"
        if accounting_currency is not None:
            run_tag+=f"_currency{accounting_currency}"
    output_root=Path(output_root) if output_root is not None else Path(results_path)/"lsv1994/replication"
    directories={table:output_root/f"table{table}"/run_tag for table in (1,2,4,5,6)}
    if run_replication:
        if 1 in tables:
            run_table1(formation_start,formation_end,return_end,raw,directories[1],universe=universe,
                       book_equity=book_equity,accounting_currency=accounting_currency)
        foundation_path=directories[1]/"run_info.json"
        if not foundation_path.is_file():
            raise FileNotFoundError("Run Table I for this case first; the other tables reuse its formation panel.")
        foundation=json.loads(foundation_path.read_text(encoding="utf-8"))
        expected=(formation_start,formation_end,str(pd.Timestamp(return_end).date()),universe,book_equity,accounting_currency)
        actual=tuple(foundation[key] for key in ("formation_start","formation_end","return_end","universe","book_equity","accounting_currency"))
        if actual!=expected or Path(foundation["raw_inputs"]["crsp_monthly"]["path"]).parent.resolve()!=raw.resolve():
            raise ValueError("Saved Table I foundation does not match this run's sample/settings.")
        if run_plan is not None:
            foundation["run_plan"]=run_plan
            foundation_path.write_text(json.dumps(foundation,indent=2)+"\n",encoding="utf-8")
        if set(tables)-{1}:
            run_extensions(directories[1],{t:directories[t] for t in tables if t!=1},
                           outcome_market_data_dir=outcome_market_data_dir,
                           allow_partial_outcomes=allow_partial_outcomes,outcome_end_year=outcome_end_year)
    for table in tables:
        if table==1:
            report_table1(directories[1],show=show_plots)
        else:
            report_table(table,directories[table])
    if build_submission:
        # Original-sample authoring remains available through an explicit opt-in.
        from core.submission.lsv1994 import generate_submission
        output=Path(results_path)/"lsv1994/submission"/f"submission_q21{'_currency'+accounting_currency if accounting_currency else ''}.html"
        print(f"Submission saved: {generate_submission(directories,output_path=output)}",flush=True)
    return directories


def batch_main(question):
    parser=argparse.ArgumentParser(description=f"LSV HW1 Q2-{question}: separate run outputs; no submission changes.")
    parser.add_argument("--raw-data-dir",type=Path,default=DEFAULT_EXTENDED_RAW)
    parser.add_argument("--end-year",type=int,default=2025)
    parser.add_argument("--universes",nargs="+",choices=list(UNIVERSES),
                        default=["nyse","nyse_amex_nasdaq"] if question==2 else ["nyse_amex"])
    parser.add_argument("--strict-coverage",action="store_true",help="Reject missing history/endpoints instead of explicitly using available dates.")
    parser.add_argument("--dry-run",action="store_true",help="Print plans without writing or running any portfolios.")
    if question==2:
        parser.add_argument("--start-year",type=int,default=1962)
    args=parser.parse_args()
    if len(set(args.universes)) != len(args.universes):
        parser.error("Select each universe only once.")
    if question not in (2,3) or (question==3 and args.end_year<=1990):
        parser.error("Q2-3's recent sample must extend beyond 1990.")
    periods=[("full",args.start_year,args.end_year)] if question==2 else [("early",1963,1989),("recent",1990,args.end_year)]
    # Plan every case before writing results, so strict coverage fails up front.
    plans=[]
    for label,start,end in periods:
        for universe in args.universes:
            plan=plan_run(args.raw_data_dir,start,end,universe,strict_coverage=args.strict_coverage)
            plan["question"]=f"Q2-{question}"
            plan["period_label"]=label
            plan["run_tag"]=(f"q2{question}_{label}_{universe}_{start}_{end}_{plan['data_source']}"
                             f"_f{plan['formation_start']}_{plan['formation_end']}_r{plan['return_end'][:7].replace('-','')}_ceq")
            plans.append(plan)
    for plan in plans:
        print(json.dumps(plan,indent=2),flush=True)
        if args.dry_run:
            continue
        run_lsv(plan["formation_start"],plan["formation_end"],plan["return_end"],args.raw_data_dir,
                universe=plan["universe"],outcome_market_data_dir=args.raw_data_dir,
                allow_partial_outcomes=True,outcome_end_year=plan["outcome_end_year"],
                run_tag=plan["run_tag"],run_plan=plan,build_submission=False)
