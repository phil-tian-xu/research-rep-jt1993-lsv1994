"""Draft replication of Table II: post-ranking betas and average firm size.

Reuse Table I Panel A's six-month signals, SAS-style deciles, and six equally
weighted active cohorts. Estimate post-ranking beta by monthly OLS with an
intercept against CRSP vwretd (raw returns, no risk-free subtraction).

The paper does not fully specify beta frequency or size timing/aggregation.
Draft convention: monthly betas; size measured at the month end BEFORE each
cohort forms and held fixed for that cohort. Average available sizes among
stocks with holding returns, then equally across six cohorts and report months.
Missing size never removes a stock from return/ranking calculations.
Price is absolute dollars; CRSP shrout is thousands; size is USD millions.
These conventions need assessment against the paper, unlike the verified
Table I notebook replication. No WRDS connection is made here.
"""

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from utils.path_info import results_path, root_path
from ._monthly import load_monthly_sample, month_ends
from .table1 import make_panel_a_ranks


def build_table2_cohorts(rank, monthly, characteristics):
    """Keep all ten deciles and attach only pre-formation market capitalization."""
    size = characteristics[["permno", "month", "prc", "shrout"]].copy()
    size["month"] = pd.to_datetime(size["month"]) + pd.offsets.MonthEnd(0)
    if size[["permno", "month"]].isna().any().any() or size.duplicated(["permno", "month"]).any():
        raise ValueError("Missing or duplicate PERMNO-month keys in monthly characteristics.")
    size["formation_id"] = size["month"].dt.year * 12 + size["month"].dt.month + 1
    price, shares = size["prc"].astype(float).abs(), size["shrout"].astype(float)
    size["formation_cap"] = (price * shares / 1000).where(
        np.isfinite(price) & np.isfinite(shares) & price.gt(0) & shares.gt(0),
    )
    members = rank[["permno", "formation_id", "momr"]].merge(
        size[["permno", "formation_id", "formation_cap"]],
        on=["permno", "formation_id"], how="left", validate="many_to_one",
    )
    expanded = members.merge(pd.DataFrame({"age": np.arange(6, dtype="int8")}), how="cross")
    expanded["hold_id"] = expanded["formation_id"] + expanded["age"]
    expanded = expanded.merge(
        monthly[["permno", "month_id", "ret"]].rename(columns={"month_id": "hold_id"}),
        on=["permno", "hold_id"], how="left", validate="many_to_one",
    )
    expanded["holding_cap"] = expanded["formation_cap"].where(expanded["ret"].notna())
    return expanded.groupby(["formation_id", "hold_id", "age", "momr"], as_index=False).agg(
        cohort_ret=("ret", "mean"), cohort_cap=("holding_cap", "mean"),
        n_constituents=("permno", "size"), n_with_ret=("ret", "count"), n_with_cap=("holding_cap", "count"),
    )


def summarize_table2(portfolios, market):
    """Estimate all ten post-ranking OLS slopes and the P10-P1 slope directly."""
    returns = portfolios.pivot(index="hold_id", columns="momr", values="ret").reindex(columns=range(1, 11))
    caps = portfolios.pivot(index="hold_id", columns="momr", values="market_cap_millions").reindex(columns=range(1, 11))
    if returns.isna().any().any() or not np.isfinite(returns.to_numpy()).all():
        raise ValueError("Table II needs all six active cohorts in every decile/month.")
    market = market[["month", "vwretd"]].copy()
    market["month"] = pd.to_datetime(market["month"]) + pd.offsets.MonthEnd(0)
    market["hold_id"] = market["month"].dt.year * 12 + market["month"].dt.month
    if market["hold_id"].isna().any() or market["hold_id"].duplicated().any():
        raise ValueError("Missing or duplicate months in CRSP market returns.")
    x = market.set_index("hold_id")["vwretd"].reindex(returns.index).astype(float)
    if len(x) < 3 or not np.isfinite(x).all() or x.var(ddof=1) <= 0:
        raise ValueError("Need at least three complete, nonconstant monthly market returns.")
    returns["P10-P1"] = returns[10] - returns[1]
    rows = []
    for decile in [*range(1, 11), "P10-P1"]:
        y = returns[decile]
        beta = ((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum()
        cap = caps[decile] if isinstance(decile, int) else pd.Series(dtype=float)
        rows.append({
            "portfolio": f"P{decile}" if isinstance(decile, int) else decile,
            "beta": beta, "average_market_cap_millions": cap.mean(),
            "n_months": len(y), "n_size_months": int(cap.count()),
            "mean_return": y.mean(), "intercept": y.mean() - beta * x.mean(),
        })
    return pd.DataFrame(rows), returns


def run_table2(run_start_date: str, run_end_date: str, raw_data_dir: str | Path,
               output_dir: str | Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read local returns/characteristics/market extracts and save Table II inputs."""
    monthly, start, end, raw_data_dir = load_monthly_sample(
        run_start_date, run_end_date, raw_data_dir, history_months=11,
    )
    for name in ("monthly_characteristics", "market_returns"):
        if not (raw_data_dir / f"{name}.parquet").is_file():
            raise FileNotFoundError(f"Missing {name}.parquet. Run data_prepare_jt1993.py with tables=(2,).")
    characteristics = pd.read_parquet(raw_data_dir / "monthly_characteristics.parquet")
    market = pd.read_parquet(raw_data_dir / "market_returns.parquet", columns=["month", "vwretd"])
    required_size_months = pd.period_range(start.to_period("M") - 6, end.to_period("M") - 1, freq="M")
    available_size_months = pd.to_datetime(characteristics["month"]).dt.to_period("M").unique()
    if len(required_size_months.difference(available_size_months)):
        raise ValueError("Monthly characteristics do not cover every required pre-formation month.")
    start_id, end_id = start.year * 12 + start.month, end.year * 12 + end.month
    rank = make_panel_a_ranks(monthly, J=6, formation_start=start_id - 5, formation_end=end_id)
    cohort = build_table2_cohorts(rank, monthly, characteristics)
    selected = cohort.loc[cohort["hold_id"].between(start_id, end_id)]
    portfolios = selected.groupby(["hold_id", "momr"], as_index=False).agg(
        ret=("cohort_ret", "mean"), market_cap_millions=("cohort_cap", "mean"),
        n_cohorts=("cohort_ret", "count"), n_size_cohorts=("cohort_cap", "count"),
    )
    portfolios.loc[portfolios["n_cohorts"].ne(6), "ret"] = np.nan
    portfolios.loc[portfolios["n_size_cohorts"].ne(6), "market_cap_millions"] = np.nan
    if len(portfolios) != len(pd.period_range(start, end, freq="M")) * 10:
        raise ValueError("Missing decile/month cells for Table II.")
    summary, _ = summarize_table2(portfolios, market)
    if not summary.iloc[:10]["n_size_months"].eq(summary.iloc[:10]["n_months"]).all():
        raise ValueError("Incomplete six-cohort size coverage for one or more deciles/months.")
    portfolios["month"] = month_ends(portfolios["hold_id"])
    output_dir = Path(output_dir) if output_dir is not None else (
        Path(results_path) / "jt1993" / "replication" / "table2" / f"{raw_data_dir.name}_{start:%Y%m}_{end:%Y%m}"
    )
    if not output_dir.is_absolute():
        output_dir = Path(root_path) / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "table2_summary.csv", index=False)
    portfolios.to_csv(output_dir / "table2_monthly_portfolios.csv", index=False)
    cohort.to_csv(output_dir / "table2_cohort_returns.csv", index=False)
    rank.groupby(["formation_id", "momr"], as_index=False).agg(
        n_stocks=("permno", "size"), mean_signal=("signal", "mean"),
    ).to_csv(output_dir / "table2_formation_counts.csv", index=False)
    info = {
        "paper": "jt1993", "table": 2, "status": "draft; not numerically validated against the paper",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_start_date": start.date().isoformat(), "run_end_date": end.date().isoformat(),
        "raw_data_dir": str(raw_data_dir), "J": 6, "K": 6, "panel": "A",
        "beta_method": "monthly OLS with intercept: raw decile return on CRSP vwretd",
        "size_method": "pre-formation month-end abs(prc)*shrout/1000, USD millions; fixed within cohort; average stocks, cohorts, months",
        "missing_sizes": "excluded only from size means; stocks remain in rankings and return portfolios",
        "boundary_convention": "include five formation cohorts before the first reported return month",
        "delisting_returns_added": False,
        "paper_ambiguities": "beta estimation frequency and size timing/aggregation are explicit draft conventions",
    }
    (output_dir / "run_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(f"Table II numeric results: {output_dir}")
    return summary, portfolios
