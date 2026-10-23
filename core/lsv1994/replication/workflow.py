"""Extend the verified Table I foundation to all remaining Q2-1 outputs."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from utils.path_info import root_path, data_path
from ._formation import prepare_crsp, prepare_accounts
from ._holding import build_holding_returns
from .table2 import make_pair_constituents, summarize_table2
from .table4 import build_table4
from .table5 import build_table5
from .table6 import make_tail_constituents, build_table6, build_figure2, hh_mean_test


def run_extensions(table1_dir, table_dirs, *, outcome_market_data_dir=None,
                   allow_partial_outcomes=False, outcome_end_year=None):
    """Read the verified formation panel once; write separate table results.

    The Table I foundation is an explicit dependency even when Table I itself
    is not selected for recalculation. Later accounting/market equity is read
    only inside the Table V branch and never extends holding-return windows.
    """
    table1_dir = Path(table1_dir)
    selected = set(table_dirs)
    if selected - {2, 4, 5, 6}:
        raise ValueError("Extension tables must be selected from 2, 4, 5, 6.")
    info = json.loads((table1_dir / "run_info.json").read_text(encoding="utf-8"))
    for name, digest in info["output_sha256"].items():
        if hashlib.sha256((table1_dir / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Table I foundation has changed: {name}")
    raw_dir = Path(info["raw_inputs"]["crsp_monthly"]["path"]).parent
    start, end = info["formation_start"], info["formation_end"]
    cutoff = pd.Timestamp(info["return_end"])
    formation = pd.read_parquet(table1_dir / "formation_panel.parquet")
    deciles = pd.read_parquet(table1_dir / "portfolio_constituents.parquet")
    pairs = make_pair_constituents(formation)
    tails = make_tail_constituents(deciles)
    all_members = formation[["permno", "formation_year"]].assign(strategy="All", portfolio=1)
    members = pd.concat([pairs, tails, all_members], ignore_index=True)
    crsp = prepare_crsp(pd.read_parquet(raw_dir / "crsp_monthly.parquet", filters=[
        ("date", ">=", pd.Timestamp(f"{start-1}-12-01")), ("date", "<=", cutoff)]), info["universe"])
    annual, yearly, sizes = build_holding_returns(crsp, members, start, cutoff)
    # Verify common stock-year returns do not depend on which sorts requested them.
    reference = pd.read_parquet(table1_dir / "annual_stock_returns.parquet")
    overlap = reference.merge(annual, on=["permno", "start_year"], validate="one_to_one", suffixes=("_reference", "_new"))
    if len(overlap) != len(reference) or not np.allclose(overlap["annual_ret_reference"], overlap["annual_ret_new"],
                                                       atol=1e-12, rtol=1e-12, equal_nan=True):
        raise ValueError("Shared annual stock returns disagree with the verified Table I foundation.")
    shared_dir = Path(data_path) / "processed" / "lsv1994" / table1_dir.name
    shared_dir.mkdir(parents=True, exist_ok=True)
    pairs.to_parquet(shared_dir / "pair_constituents.parquet", index=False)
    annual.to_parquet(shared_dir / "annual_stock_returns.parquet", index=False)
    yearly.to_parquet(shared_dir / "portfolio_year_returns.parquet", index=False)
    formation_counts = members.groupby(["strategy", "formation_year", "portfolio"], as_index=False).agg(n_stocks=("permno", "size"))
    formation_counts.to_csv(shared_dir / "formation_counts.csv", index=False)
    tables = {}
    details = {}
    input_additions = {}
    if 2 in selected:
        print("Calculating Table 2...", flush=True)
        tables[2] = summarize_table2(yearly, start, end, cutoff)
        details[2] = {"portfolio_year_returns.csv": yearly.loc[yearly["strategy"].isin(pairs["strategy"].unique())],
                      "formation_counts.csv": formation_counts.loc[formation_counts["strategy"].isin(pairs["strategy"].unique())]}
    if 4 in selected:
        print("Calculating Table 4...", flush=True)
        tables[4], coefficients = build_table4(formation, annual)
        details[4] = {"cross_sectional_coefficients.csv": coefficients}
    if 5 in selected:
        print("Calculating Table 5...", flush=True)
        future_max_year = min(end+4, outcome_end_year) if outcome_end_year is not None else end+4
        accounts = prepare_accounts(pd.read_parquet(raw_dir / "compustat_annual.parquet", filters=[
            ("datadate", ">=", pd.Timestamp(f"{start-6}-01-01")),
            ("datadate", "<=", pd.Timestamp(f"{future_max_year}-12-31"))]), info["book_equity"], info["accounting_currency"])
        # Earlier rows are solely for Table V historical growth and past returns.
        historical = prepare_crsp(pd.read_parquet(raw_dir / "crsp_monthly.parquet", filters=[
            ("date", ">=", pd.Timestamp(f"{start-5}-04-01")), ("date", "<=", cutoff)]), info["universe"])
        cap = historical.loc[historical["month"].eq(4), ["permno", "year", "me"]]
        outcome_dir = Path(outcome_market_data_dir) if outcome_market_data_dir is not None else raw_dir
        if not outcome_dir.is_absolute():
            outcome_dir = Path(root_path) / outcome_dir
        extra_path = outcome_dir / "crsp_monthly.parquet"
        extra_info_path = extra_path.with_suffix(".json")
        extra_info = json.loads(extra_info_path.read_text(encoding="utf-8")) if extra_info_path.is_file() else {}
        original_source = (info["raw_inputs"]["crsp_monthly"]["download_metadata"] or {}).get("data_source")
        if extra_info.get("data_source", original_source) != original_source:
            raise ValueError("Table V outcome market equity must use the same CRSP source as the original sample.")
        future_cap = pd.read_parquet(extra_path, columns=["permno", "date", "prc", "shrout"], filters=[
            ("date", ">", cutoff), ("date", "<=", pd.Timestamp(f"{future_max_year}-04-30"))])
        future_cap["date"] = pd.to_datetime(future_cap["date"])
        future_cap = future_cap.loc[future_cap["date"].dt.month.eq(4)].copy()
        future_cap["year"] = future_cap["date"].dt.year
        future_cap["me"] = future_cap["prc"].astype(float).abs() * future_cap["shrout"].astype(float) / 1000
        cap = pd.concat([cap, future_cap[["permno", "year", "me"]]], ignore_index=True)
        cap = cap.loc[cap["year"].le(future_max_year)].copy()
        if cap.duplicated(["permno", "year"]).any():
            raise ValueError("Duplicate Table V April market equity.")
        tables[5], fundamentals, growth, growth_cohorts, past = build_table5(
            formation, deciles, pairs, accounts, historical, cap, start, end,
            allow_partial_outcomes=allow_partial_outcomes, outcome_end_year=outcome_end_year)
        details[5] = {"fundamentals_by_cohort.csv": fundamentals, "growth_transitions.csv": growth,
                      "growth_levels_by_cohort.csv": growth_cohorts, "past_returns_by_cohort.csv": past}
        input_additions[5] = {"outcome_market_equity": {"path": str(extra_path), "download_metadata": extra_info,
                                                       "requested_last_outcome_april": future_max_year,
                                                       "available_last_outcome_april": int(cap["year"].max()),
                                                       "last_complete_future_formation": min(end, int(cap["year"].max())-4, int(accounts["acct_year"].max())-4)}}
    if 6 in selected:
        print("Calculating Table 6...", flush=True)
        base_yearly = pd.read_csv(table1_dir / "portfolio_year_returns.csv")
        tables[6], cohorts = build_table6(pd.concat([yearly, base_yearly], ignore_index=True), start, end, cutoff)
        market_path = raw_dir / "market_returns.parquet"
        if not market_path.is_file():
            raise FileNotFoundError("Figure 2 requires market_returns.parquet from data_prepare_lsv1994.py.")
        figure_data = build_figure2(cohorts, pd.read_parquet(market_path))
        details[6] = {"table6_cohort_returns.csv": cohorts, "figure2_data.csv": figure_data}
        recession_path = Path(root_path)/"references/lsv1994/recessions.csv"
        input_additions[6] = {"recession_chronology": {
            "path": str(recession_path), "sha256": hashlib.sha256(recession_path.read_bytes()).hexdigest(),
            "metadata": json.loads(recession_path.with_suffix(".json").read_text(encoding="utf-8"))}}
        if (start,end,str(cutoff.date())) == (1968,1989,"1990-04-30"):
            paper_dir=Path(root_path)/"references/lsv1994/paper_tables"
            printed=pd.read_csv(paper_dir/"table6_cohorts.csv")
            reported=pd.read_csv(paper_dir/"table6.csv").set_index(["strategy","horizon"])
            audit=[]
            for (strategy,horizon),cell in printed.groupby(["strategy","horizon"]):
                estimate=hh_mean_test(cell.sort_values("formation_year")["spread"],horizon-1)
                audit.append({"strategy":strategy,"horizon":horizon,"t_on_rounded_paper_returns":estimate["t"],
                              "published_t":reported.loc[(strategy,horizon),"t"]})
            details[6]["published_inference_audit.csv"]=pd.DataFrame(audit)
    methods = {
        2: {"sorts": "independent pair-common-sample SAS deciles collapsed 1-3/4-7/8-10",
            "panels": ["C/P-GS", "E/P-GS", "B/M-GS", "E/P-B/M", "B/M-C/P"]},
        4: {"dependent_variable": "one-year raw stock buy-and-hold return", "models": 9,
            "inference": "time-series standard deviation of annual cross-sectional coefficients / sqrt(n)",
            "negative_ratios": "positive-part regressor plus negative dummy where specified; missing stays missing",
            "outlier_treatment": "none",
            "unidentified_zero_dummy": "omit an all-zero negative-ratio dummy from that annual fit; retain other slopes; its coefficient stays NaN and is excluded only from its own time-series mean/inference; year counts saved"},
        5: {"relative_accounting_year_zero": "formation year t-1",
            "growth": "paired within-cohort per-dollar levels; average cohort levels before ratios; geometric mean of transition growth",
            "ownership": "1/market equity at April start of each relative-year transition; same ownership for both accounting amounts",
            "missing_outcomes": "pairwise common accounting observations; no effect on portfolio formation",
            "allow_partial_outcomes": allow_partial_outcomes, "outcome_end_year": outcome_end_year,
            "future_cohorts": "same calendar-eligible cohorts across every future-growth transition; recent unobserved outcomes excluded",
            "past_returns": "mean of complete 36-month stock buy-and-hold returns, May t-3 through April t",
            "nonpositive_aggregate_levels": "undefined geometric growth; retained with status and displayed as a dash"},
        6: {"cumulative_returns": "compound each cohort leg before subtracting; annual re-equal-weighting of original survivors",
            "tails": "pool stocks in deciles 1/2 and 9/10 with equal stock weights",
            "bm_definition": "main: heading/text 9/10 minus 1/2; caption 10-minus-1 saved as sensitivity",
            "inference": "Hansen-Hodrick rectangular kernel, lags h-1, n/(n-1) correction",
            "figure2_recession": "May-April window overlaps versioned NBER peak-to-trough dates, peak inclusive",
            "figure2_decline": "CRSP equally weighted index May-April buy-and-hold return is negative"},
    }
    for table, frame in tables.items():
        directory = Path(table_dirs[table])
        directory.mkdir(parents=True, exist_ok=True)
        outputs = {f"table{table}_summary.csv": frame, **details[table]}
        for name, data in outputs.items():
            data.to_csv(directory / name, index=False)
        metadata = {key: info[key] for key in ["paper", "formation_start", "formation_end", "return_end", "universe",
                                              "book_equity", "accounting_currency", "raw_inputs", "diagnostics", "limitations"]}
        metadata.update({"table": table, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                         "table1_foundation": str(table1_dir), "shared_dir": str(shared_dir),
                         "run_plan": info.get("run_plan"),
                         "settings": methods[table], "additional_inputs": input_additions.get(table, {}),
                         "validation_status": "tested implementation of paper definitions; no supplied program for this table",
                         "outputs": list(outputs),
                         "output_sha256": {name: hashlib.sha256((directory/name).read_bytes()).hexdigest() for name in outputs}})
        (directory / "run_info.json").write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf-8")
        print(f"Table {table}: {len(frame)} summary rows saved to {directory}", flush=True)
    return tables
