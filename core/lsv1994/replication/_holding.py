"""May-April buy-and-hold returns and previous-December size benchmarks."""

import numpy as np
import pandas as pd

from ._formation import ntile_series


def annualize(stock_year, months, monthly, size_month):
    """Fill only months after the last CRSP row, never internal missing returns.

    Size returns are monthly equal-weighted returns of the April size members,
    as in the reference. Annual size benchmarks average stock buy-and-hold
    returns; compounding the monthly equal-weighted index would change weights.
    """
    grid = (stock_year.merge(months, on="start_year", validate="many_to_many")
            .merge(monthly[["permno", "start_year", "ym", "ret", "row_flag"]],
                   on=["permno", "start_year", "ym"], how="left", validate="many_to_one")
            .merge(size_month, on=["start_year", "size_decile", "ym"],
                   how="left", validate="many_to_one"))
    last = (grid.loc[grid["row_flag"].eq(1)].groupby(["permno", "start_year"], as_index=False)
            .agg(last_obs=("month_no", "max")))
    grid = grid.merge(last, on=["permno", "start_year"], how="left", validate="many_to_one")
    grid["last_obs"] = grid["last_obs"].fillna(0)
    post = grid["month_no"].gt(grid["last_obs"])
    grid["ret_filled"] = grid["ret"].where(~post, grid["size_ret"])
    grid["replacement"] = post & grid["ret_filled"].notna()
    grid["internal_missing"] = ~post & grid["ret"].isna()
    grid["gross"] = 1 + grid["ret_filled"]
    out = grid.groupby(["permno", "start_year", "size_decile"], dropna=False, as_index=False).agg(
        n_months=("ret_filled", "count"), gross=("gross", "prod"),
        replacement_months=("replacement", "sum"), internal_missing=("internal_missing", "sum"))
    out["annual_ret"] = (out["gross"] - 1).where(out["n_months"].eq(12))
    return out.drop(columns="gross")


def build_holding_returns(crsp, constituents, formation_start, return_end):
    endpoint = pd.Timestamp(return_end)
    if endpoint.month != 4 or endpoint != endpoint + pd.offsets.MonthEnd(0):
        raise ValueError("Use an April month-end return endpoint for complete May-April years.")
    last_year = endpoint.year - 1
    required = pd.period_range(f"{formation_start - 1}-12", endpoint, freq="M")
    missing = required.difference(crsp["ym"].unique())
    if len(missing):
        raise ValueError(f"CRSP is missing required calendar months: {missing.tolist()}")
    december = crsp.loc[crsp["month"].eq(12) & crsp["eligible"]
                        & crsp["year"].between(formation_start - 1, last_year - 1),
                        ["permno", "year", "me"]].copy()
    december["start_year"] = december["year"] + 1
    december["size_decile"] = december.groupby("start_year")["me"].transform(ntile_series)
    size_assign = december[["permno", "start_year", "size_decile"]].dropna().copy()
    size_assign["size_decile"] = size_assign["size_decile"].astype("int8")
    april = crsp.loc[crsp["month"].eq(4) & crsp["year"].between(formation_start, last_year)]
    alive = april[["permno", "year"]].rename(columns={"year": "start_year"})
    size_start = april.loc[april["eligible"], ["permno", "year"]].rename(columns={"year": "start_year"})
    size_members = size_start.merge(size_assign, on=["permno", "start_year"], validate="one_to_one")
    monthly = crsp.loc[crsp["ym"].between(pd.Period(f"{formation_start}-05", "M"),
                                         endpoint.to_period("M")), ["permno", "ym", "ret"]].copy()
    monthly["start_year"] = np.where(monthly["ym"].dt.month >= 5,
                                      monthly["ym"].dt.year, monthly["ym"].dt.year - 1)
    monthly["row_flag"] = 1
    size_month = (monthly.merge(size_members, on=["permno", "start_year"], validate="many_to_one")
                  .dropna(subset=["ret"]).groupby(["start_year", "size_decile", "ym"], as_index=False)
                  .agg(size_ret=("ret", "mean")))
    months = pd.DataFrame([(y, m, j) for y in range(formation_start, last_year + 1)
                           for j, m in enumerate(pd.period_range(f"{y}-05", f"{y+1}-04", freq="M"), 1)],
                          columns=["start_year", "ym", "month_no"])
    size_stock = annualize(size_members, months, monthly, size_month)
    size_annual = (size_stock.dropna(subset=["annual_ret"])
                   .groupby(["start_year", "size_decile"], as_index=False)
                   .agg(size_ret_annual=("annual_ret", "mean"), n_stocks=("annual_ret", "count")))
    needed = constituents.merge(pd.DataFrame({"hold_year": range(1, 6)}), how="cross")
    needed["start_year"] = needed["formation_year"] + needed["hold_year"] - 1
    needed = needed.loc[needed["start_year"].le(last_year)].copy()
    stock_year = (needed[["permno", "start_year"]].drop_duplicates()
                  .merge(alive, on=["permno", "start_year"], validate="one_to_one")
                  .merge(size_assign, on=["permno", "start_year"], how="left", validate="one_to_one"))
    annual_stock = annualize(stock_year, months, monthly, size_month)
    annual_stock = annual_stock.merge(size_annual[["start_year", "size_decile", "size_ret_annual"]],
                                      on=["start_year", "size_decile"], how="left", validate="many_to_one")
    annual_stock["size_adj_ret"] = annual_stock["annual_ret"] - annual_stock["size_ret_annual"]
    returns = needed.merge(annual_stock, on=["permno", "start_year"], how="left", validate="many_to_one")
    portfolio_year = returns.groupby(["strategy", "formation_year", "portfolio", "hold_year"], as_index=False).agg(
        ret=("annual_ret", "mean"), saar=("size_adj_ret", "mean"),
        n_valid=("annual_ret", "count"), n_size_adj=("size_adj_ret", "count"),
        n_constituents=("permno", "nunique"))
    if portfolio_year["ret"].isna().any():
        raise ValueError("One or more portfolio/cohort/horizon cells have no valid annual returns.")
    return annual_stock, portfolio_year, size_annual
