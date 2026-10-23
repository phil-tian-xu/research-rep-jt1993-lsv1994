"""Table VI: cohort cumulative value-minus-glamour returns and HH inference."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from utils.path_info import root_path


DEFINITIONS = {"C/P tails": (1, 2), "C/P-GS": (3, 7), "B/M tails": (1, 2), "B/M": (1, 10)}


def make_tail_constituents(members):
    tails = members.loc[members["strategy"].isin(["C/P", "B/M"])
                        & members["portfolio"].isin([1, 2, 9, 10])].copy()
    tails["strategy"] += " tails"
    tails["portfolio"] = np.where(tails["portfolio"].le(2), 1, 2)
    return tails


def hh_mean_test(values, lags):
    """Uniform-kernel Hansen-Hodrick, n/(n-1) correction; no Bartlett weights.

    A negative HH long-run variance is not silently clipped or replaced by
    Newey-West. Return an undefined standard error with an explicit status.
    """
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or len(values) <= max(1, lags):
        raise ValueError("HH requires complete consecutive observations and more observations than lags.")
    n = len(values)
    residual = values - values.mean()
    meat = residual @ residual
    for lag in range(1, lags+1):
        meat += 2 * (residual[lag:] @ residual[:-lag])
    variance = meat / (n * (n-1))
    se = np.sqrt(variance) if variance > 0 else np.nan
    return {"mean": values.mean(), "se": se, "t": values.mean()/se if se > 0 else np.nan,
            "n_cohorts": n, "hh_lags": lags, "status": "ok" if variance > 0 else "nonpositive HH variance"}


def build_table6(yearly, formation_start, formation_end, return_end):
    rows = []
    last_year = pd.Timestamp(return_end).year-1
    for strategy, (glamour, value) in DEFINITIONS.items():
        panel = yearly.loc[yearly["strategy"].eq(strategy)]
        for year in range(formation_start, formation_end+1):
            cohort = panel.loc[panel["formation_year"].eq(year)].pivot(index="hold_year", columns="portfolio", values="ret")
            for horizon in (1, 3, 5):
                if year + horizon - 1 > last_year:
                    continue
                cell = cohort.reindex(index=range(1, horizon+1), columns=[glamour, value])
                if cell.isna().any().any():
                    raise ValueError(f"Table VI incomplete {strategy}, {year}, {horizon}-year cohort.")
                cumulative = (1 + cell).prod(min_count=horizon)-1
                rows.append({"strategy": strategy, "formation_year": year, "horizon": horizon,
                             "glamour": cumulative[glamour], "value": cumulative[value],
                             "spread": cumulative[value]-cumulative[glamour]})
    cohorts = pd.DataFrame(rows)
    summary = []
    for (strategy, horizon), cell in cohorts.groupby(["strategy", "horizon"], sort=False):
        cell = cell.sort_values("formation_year")
        expected = max(0, min(formation_end, last_year-horizon+1)-formation_start+1)
        if len(cell) != expected or (len(cell)>1 and not cell["formation_year"].diff().dropna().eq(1).all()):
            raise ValueError("Table VI inference requires consecutive complete formation cohorts.")
        summary.append({"strategy": strategy, "horizon": horizon,
                        **hh_mean_test(cell["spread"], horizon-1)})
    return pd.DataFrame(summary), cohorts


def build_figure2(cohorts, market):
    """R: May-April window overlaps NBER recession; D: EW index BH return < 0."""
    panel = cohorts.loc[cohorts["strategy"].eq("C/P-GS") & cohorts["horizon"].eq(1)].copy()
    market = market[["date", "ewretd"]].copy()
    market["ym"] = pd.to_datetime(market["date"]).dt.to_period("M")
    if market["ym"].duplicated().any():
        raise ValueError("Duplicate CRSP index month.")
    returns = market.set_index("ym")["ewretd"].astype(float)
    recession_dir = Path(root_path)/"references/lsv1994"
    chronology = pd.read_csv(recession_dir/"recessions.csv")
    chronology_info = json.loads((recession_dir/"recessions.json").read_text(encoding="utf-8"))
    recessions = [(pd.Period(a,"M"),pd.Period(b,"M")) for a,b in chronology.itertuples(index=False,name=None)]
    if len(panel):
        first_month = pd.Period(f"{int(panel['formation_year'].min())}-05","M")
        last_month = pd.Period(f"{int(panel['formation_year'].max())+1}-04","M")
        if first_month < pd.Period(chronology_info["coverage_start"],"M") or last_month > pd.Period(chronology_info["coverage_end"],"M"):
            raise ValueError("Figure 2 extends beyond the verified recession chronology; update its snapshot first.")
    values, flags = [], []
    for year in panel["formation_year"]:
        months = pd.period_range(f"{year}-05", f"{year+1}-04", freq="M")
        sample = returns.reindex(months)
        if sample.isna().any() or not np.isfinite(sample).all() or sample.lt(-1).any():
            raise ValueError(f"Figure 2 requires all twelve valid index returns for May {year}-April {year+1}.")
        values.append((1 + sample).prod()-1)
        flags.append(any(months[0] <= end and months[-1] >= start for start,end in recessions))
    panel["ew_market_return"] = values
    panel["recession_overlap"] = flags
    panel["market_decline"] = panel["ew_market_return"].lt(0)
    return panel
