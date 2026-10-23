"""Table I Panels A-D, with local inputs and separately frozen benchmarks."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from utils.path_info import root_path, results_path
from ._formation import SIGNALS, prepare_crsp, prepare_accounts, build_formation, make_constituents
from ._holding import build_holding_returns


STATS = ["R1", "R2", "R3", "R4", "R5", "AR", "CR5", "SAAR"]
BENCHMARK = Path(root_path) / "references" / "lsv1994_table1_reference_results.json"


def summarize_table1(portfolio_year):
    """Average cohorts first, then average/compound the five horizon means."""
    rows = []
    for strategy in SIGNALS:
        x = portfolio_year.loc[portfolio_year["strategy"].eq(strategy)]
        raw = x.groupby(["portfolio", "hold_year"])["ret"].mean().unstack().reindex(
            index=range(1, 11), columns=range(1, 6))
        adj = x.groupby(["portfolio", "hold_year"])["saar"].mean().unstack().reindex(
            index=range(1, 11), columns=range(1, 6))
        if raw.isna().any().any() or adj.isna().any().any():
            raise ValueError(f"Incomplete decile/horizon coverage for {strategy}.")
        panel = raw.T.copy()
        panel.index = [f"R{h}" for h in range(1, 6)]
        panel.loc["AR"] = raw.mean(axis=1)
        panel.loc["CR5"] = (1 + raw).prod(axis=1, min_count=5) - 1
        panel.loc["SAAR"] = adj.mean(axis=1)
        for stat in STATS:
            for decile in range(1, 11):
                rows.append({"strategy": strategy, "stat": stat, "decile": decile,
                             "return": float(panel.loc[stat, decile])})
    return pd.DataFrame(rows)


def validate_cohort_coverage(portfolio_year, formation_start, formation_end, return_end):
    last_year = pd.Timestamp(return_end).year - 1
    expected = pd.DataFrame([
        (s, d, h, max(0, min(formation_end, last_year - h + 1) - formation_start + 1))
        for s in SIGNALS for d in range(1, 11) for h in range(1, 6)
    ], columns=["strategy", "portfolio", "hold_year", "expected_cohorts"])
    actual = portfolio_year.groupby(["strategy", "portfolio", "hold_year"], as_index=False).agg(
        n_cohorts=("ret", "count"), n_size_adj_cohorts=("saar", "count"))
    coverage = expected.merge(actual, on=["strategy", "portfolio", "hold_year"], how="left",
                              validate="one_to_one").fillna(0)
    if not coverage["expected_cohorts"].gt(0).all():
        raise ValueError("Table I requires at least one complete five-year formation cohort.")
    if not (coverage["n_cohorts"].eq(coverage["expected_cohorts"])
            & coverage["n_size_adj_cohorts"].eq(coverage["expected_cohorts"])).all():
        raise ValueError("Missing deciles or incomplete cohort/horizon coverage; inspect formation eligibility.")
    return coverage


def run_table1(formation_start=1968, formation_end=1989, return_end="1990-04-30",
               raw_data_dir=None, output_dir=None, *, universe="nyse_amex", book_equity="ceq", accounting_currency=None):
    """Run all four Table I sorts without a WRDS connection or raw-data writes.

    Bounds describe formation years and the last holding month separately.
    Extra raw history/outcomes are filtered before signal construction.
    """
    end = pd.Timestamp(return_end)
    if formation_start > formation_end or formation_end >= end.year or end.month != 4:
        raise ValueError("Formation years must be ordered and end before the April return cutoff.")
    if raw_data_dir is None:
        raw_data_dir = Path(root_path) / "data/raw/lsv1994/nyse_amex_19621201_19900430_comp19620101_19931231"
    raw_data_dir = Path(raw_data_dir)
    if not raw_data_dir.is_absolute():
        raw_data_dir = Path(root_path) / raw_data_dir
    provenance = {}
    for name in ("crsp_monthly", "compustat_annual", "ccm_links"):
        path = raw_data_dir / f"{name}.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}; run data_prepare_lsv1994.py first.")
        sidecar = path.with_suffix(".json")
        metadata = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.is_file() else None
        provenance[name] = {"path": str(path), "bytes": path.stat().st_size,
                            "modified_ns": path.stat().st_mtime_ns, "download_metadata": metadata}
    print(f"LSV Table I | {universe} | April formations {formation_start}-{formation_end} | returns through {end.date()}", flush=True)
    crsp = pd.read_parquet(raw_data_dir / "crsp_monthly.parquet", filters=[
        ("date", ">=", pd.Timestamp(f"{formation_start-1}-12-01")), ("date", "<=", end)])
    accounts = pd.read_parquet(raw_data_dir / "compustat_annual.parquet", filters=[
        ("datadate", ">=", pd.Timestamp(f"{formation_start-6}-01-01")),
        ("datadate", "<=", pd.Timestamp(f"{formation_end-1}-12-31"))])
    duplicate_accounts = int(pd.to_datetime(accounts["datadate"]).dt.year.to_frame("year")
                             .assign(gvkey=accounts["gvkey"]).duplicated(["gvkey", "year"]).sum())
    crsp = prepare_crsp(crsp, universe)
    accounts = prepare_accounts(accounts, book_equity, accounting_currency)
    links = pd.read_parquet(raw_data_dir / "ccm_links.parquet")
    formation = build_formation(crsp, accounts, links, formation_start, formation_end)
    members = make_constituents(formation)
    print(f"Linked April stock-years: {len(formation):,}; decile memberships: {len(members):,}", flush=True)
    annual_stock, portfolio_year, size_annual = build_holding_returns(crsp, members, formation_start, end)
    coverage = validate_cohort_coverage(portfolio_year, formation_start, formation_end, end)
    summary = summarize_table1(portfolio_year)
    counts = members.groupby(["strategy", "formation_year", "portfolio"], as_index=False).agg(
        n_stocks=("permno", "size"))
    source_info = provenance["crsp_monthly"]["download_metadata"] or {}
    source = source_info.get("data_source", "unknown")
    compare = (formation_start == 1968 and formation_end == 1989 and end == pd.Timestamp("1990-04-30")
               and universe == "nyse_amex" and book_equity == "ceq" and source == "legacy"
               and accounting_currency is None)
    if compare:
        benchmark = pd.DataFrame(json.loads(BENCHMARK.read_text(encoding="utf-8"))["rows"])
        summary = summary.merge(benchmark, on=["strategy", "stat", "decile"], validate="one_to_one")
        summary["difference_paper_pp"] = (summary["return"] - summary["paper"]) * 100
        summary["difference_notebook_pp"] = (summary["return"] - summary["notebook_rounded"]) * 100
    tag = f"{raw_data_dir.name}_{formation_start}_{formation_end}_to{end:%Y%m}_{universe}_{book_equity}"
    if accounting_currency is not None:
        tag += f"_currency{accounting_currency}"
    output_dir = Path(output_dir) if output_dir is not None else Path(results_path) / "lsv1994/replication/table1" / tag
    if not output_dir.is_absolute():
        output_dir = Path(root_path) / output_dir
    # Never allow an output folder to overwrite cached source files.
    if output_dir.resolve() == raw_data_dir.resolve() or raw_data_dir.resolve() in output_dir.resolve().parents:
        raise ValueError("Output must be outside the raw-data folder.")
    output_dir.mkdir(parents=True, exist_ok=True)
    formation.to_parquet(output_dir / "formation_panel.parquet", index=False)
    members.to_parquet(output_dir / "portfolio_constituents.parquet", index=False)
    annual_stock.to_parquet(output_dir / "annual_stock_returns.parquet", index=False)
    portfolio_year.to_csv(output_dir / "portfolio_year_returns.csv", index=False)
    size_annual.to_csv(output_dir / "size_benchmark_returns.csv", index=False)
    counts.to_csv(output_dir / "formation_counts.csv", index=False)
    coverage.to_csv(output_dir / "horizon_coverage.csv", index=False)
    summary.to_csv(output_dir / "table1_summary.csv", index=False)
    diagnostics = {
        "linked_stock_years": len(formation), "constituent_memberships": len(members),
        "accounting_calendar_year_duplicates_resolved": duplicate_accounts,
        "annual_stock_years": len(annual_stock),
        "valid_annual_stock_returns": int(annual_stock["annual_ret"].notna().sum()),
        "valid_size_adjusted_stock_returns": int(annual_stock["size_adj_ret"].notna().sum()),
        "stock_years_with_size_replacement": int(annual_stock["replacement_months"].gt(0).sum()),
        "internal_missing_monthly_returns": int(annual_stock["internal_missing"].sum()),
        "portfolio_cells_with_fewer_size_adjusted_returns": int(portfolio_year["n_size_adj"].lt(portfolio_year["n_valid"]).sum()),
        "formation_account_currencies": formation["curcd"].fillna("missing").value_counts().to_dict() if "curcd" in formation else {},
    }
    info = {
        "paper": "lsv1994", "table": 1, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "formation_start": formation_start, "formation_end": formation_end, "return_end": str(end.date()),
        "universe": universe, "book_equity": book_equity, "accounting_currency": accounting_currency,
        "raw_inputs": provenance,
        "diagnostics": diagnostics, "paper_comparison_applied": compare,
        "settings": {"share_codes": [10, 11], "rank_method": "SAS average ranks",
                     "gs_rank_mode": "percentile", "positive_cp_ep_only": True,
                     "accounting_lag": "calendar datadate year t-1 for April t",
                     "size_assignment": "previous December; reassign each holding year",
                     "holding": "May-April buy-and-hold, annual re-equal-weighting of surviving original stocks",
                     "disappearance": "size-decile replacement after final CRSP row within holding year",
                     "missing_returns": "no internal filling; exclude incomplete annual stock returns",
                     "delisting_adjustment": source_info.get("delisting_return_adjustment", "unknown"),
                     "units": "decimal returns; market equity in USD millions"},
        "limitations": [
            "Agreement with the supplied notebook does not imply exact agreement with the 1994 paper.",
            "Current CRSP/Compustat/CCM vintage; prior-year accounts are a timing convention, not point-in-time data.",
            "Reference defaults: CEQ book equity, percentile GS ranks, no winsorization or currency conversion.",
            "CRSP missing rows after the last observed row proxy disappearance; no separate legacy delisting-return adjustment.",
            "Stocks without December size may have a raw annual return but no size-adjusted return.",
            "Non-USD accounting values are not converted in the notebook baseline; use accounting_currency=USD for a separate sensitivity.",
            "Historical notebook benchmark is rounded to three decimals; full-precision parity is checked by replay."
        ],
        "reference_notebook_sha256": json.loads(BENCHMARK.read_text(encoding="utf-8"))["notebook_sha256"],
        "outputs": ["formation_panel.parquet", "portfolio_constituents.parquet", "annual_stock_returns.parquet",
                    "portfolio_year_returns.csv", "size_benchmark_returns.csv", "formation_counts.csv",
                    "horizon_coverage.csv", "table1_summary.csv"],
    }
    info["output_sha256"] = {name: hashlib.sha256((output_dir / name).read_bytes()).hexdigest()
                              for name in info["outputs"]}
    (output_dir / "run_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print("Cohorts by horizon: " + ", ".join(str(int(x)) for x in coverage.iloc[:5]["n_cohorts"]), flush=True)
    print(f"Saved Table I results: {output_dir}", flush=True)
    return summary, portfolio_year
