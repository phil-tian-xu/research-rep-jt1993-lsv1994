"""Table V: portfolio fundamentals and growth of aggregate accounting amounts.

Accounting relative year 0 is t-1 for an April t formation. Each growth
transition fixes ownership at 1/ME measured at the April starting that relative
year. Paired firm observations are aggregated within a cohort, then averaged
across cohorts, before growth is calculated (paper pp.1545-1546).
"""

import numpy as np
import pandas as pd


METRICS = ["E/P", "C/P", "S/P", "D/P", "B/M", "SIZE",
           "AEG(-5,0)", "ACG(-5,0)", "ASG(-5,0)", "RETURN(-3,0)",
           "AEG(0,5)", "ACG(0,5)", "ASG(0,5)", "AEG(2,5)", "ACG(2,5)", "ASG(2,5)"]


def select_extremes(deciles, pairs):
    bm = deciles.loc[deciles["strategy"].eq("B/M") & deciles["portfolio"].isin([1, 10])].copy()
    bm["leg"] = np.where(bm["portfolio"].eq(1), "Glamour", "Value")
    cp = pairs.loc[pairs["strategy"].eq("C/P-GS") & pairs["portfolio"].isin([3, 7])].copy()
    cp["leg"] = np.where(cp["portfolio"].eq(3), "Glamour", "Value")
    return pd.concat([bm, cp], ignore_index=True).rename(columns={"strategy": "classification"})


def aggregate_growth(paired):
    """Ratio of cross-cohort average portfolio levels, not mean firm growth."""
    cohort = paired.groupby(["classification", "leg", "formation_year", "relative_end", "variable"], as_index=False).agg(
        base_level=("base_per_dollar", "mean"), next_level=("next_per_dollar", "mean"), n_pairs=("permno", "size"))
    pooled = cohort.groupby(["classification", "leg", "relative_end", "variable"], as_index=False).agg(
        base_level=("base_level", "mean"), next_level=("next_level", "mean"), n_cohorts=("formation_year", "nunique"),
        min_pairs=("n_pairs", "min"), max_pairs=("n_pairs", "max"))
    valid = pooled["base_level"].gt(0) & pooled["next_level"].gt(0)
    pooled["gross_growth"] = (pooled["next_level"] / pooled["base_level"]).where(valid)
    return pooled, cohort


def build_table5(formation, deciles, pairs, accounts, monthly, april_caps, formation_start, formation_end,
                 *, allow_partial_outcomes=False, outcome_end_year=None):
    if accounts["acct_year"].min() > formation_start-6:
        raise ValueError("Table V requires accounting history from formation_start-6.")
    observed_end = min(int(accounts["acct_year"].max()), int(april_caps["year"].max()))
    if outcome_end_year is not None:
        observed_end = min(observed_end, outcome_end_year)
    if not allow_partial_outcomes and observed_end < formation_end+4:
        raise ValueError("Table V requires April market equity and accounts through formation_end+4 for growth outcomes.")
    # Calendar follow-up eligibility never affects formation membership or returns.
    future_end = min(formation_end, observed_end-4)
    n_future = max(0, future_end-formation_start+1)
    required_aprils = set(range(formation_start-5, formation_end+1))
    if n_future:
        required_aprils.update(range(formation_start, future_end+5))
    if not required_aprils.issubset(set(april_caps["year"])):
        raise ValueError("Table V requires April market equity for every selected growth transition.")
    members = select_extremes(deciles, pairs)
    data = members.merge(formation, on=["permno", "formation_year"], validate="many_to_one")
    data["E/P"], data["C/P"], data["B/M"] = data["ep"], data["cp"], data["bm"]
    data["S/P"] = data["sale"] / data["me_apr"]
    if "dvc" not in data:
        raise ValueError("Table V requires Compustat dvc; rerun its data preparation with Table V inputs.")
    data["D/P"] = pd.to_numeric(data["dvc"], errors="raise").astype(float) / data["me_apr"]
    data["SIZE"] = data["me_apr"]
    fund = data.melt(id_vars=["classification", "leg", "formation_year", "permno"],
                     value_vars=METRICS[:6], var_name="metric", value_name="value").dropna(subset=["value"])
    fund_cohort = fund.groupby(["classification", "leg", "formation_year", "metric"], as_index=False).agg(
        value=("value", "mean"), n_stocks=("permno", "size"))
    summary = fund_cohort.groupby(["classification", "leg", "metric"], as_index=False).agg(
        value=("value", "mean"), n_cohorts=("formation_year", "nunique"))
    summary["section"] = "A"
    summary["status"] = "ok"
    base = data[["classification", "leg", "formation_year", "permno", "gvkey"]]
    transitions = base.merge(pd.DataFrame({"relative_end": range(-4, 6)}), how="cross")
    transitions = transitions.loc[transitions["relative_end"].le(0) | transitions["formation_year"].le(future_end)].copy()
    transitions["base_year"] = transitions["formation_year"] + transitions["relative_end"] - 2
    # Ownership at the April that begins the base relative accounting year.
    transitions["cap_year"] = transitions["formation_year"] + transitions["relative_end"] - 1
    cap = april_caps[["permno", "year", "me"]].rename(columns={"year": "cap_year"})
    transitions = transitions.merge(cap, on=["permno", "cap_year"], how="left", validate="many_to_one")
    account_cols = ["gvkey", "acct_year", "earn", "cf", "sale"]
    prev = accounts[account_cols].rename(columns={"acct_year": "base_year", "earn": "base_earn", "cf": "base_cf", "sale": "base_sale"})
    following = accounts[account_cols].copy()
    following["base_year"] = following["acct_year"]-1
    following = following.drop(columns="acct_year").rename(columns={"earn": "next_earn", "cf": "next_cf", "sale": "next_sale"})
    transitions = (transitions.merge(prev, on=["gvkey", "base_year"], how="left", validate="many_to_one")
                   .merge(following, on=["gvkey", "base_year"], how="left", validate="many_to_one"))
    paired = []
    for variable in ("earn", "cf", "sale"):
        cell = transitions.loc[transitions["me"].gt(0)
                               & transitions[f"base_{variable}"].notna()
                               & transitions[f"next_{variable}"].notna()].copy()
        cell["variable"] = variable
        cell["base_per_dollar"] = cell[f"base_{variable}"] / cell["me"]
        cell["next_per_dollar"] = cell[f"next_{variable}"] / cell["me"]
        paired.append(cell[["classification", "leg", "formation_year", "relative_end", "variable", "permno",
                            "base_per_dollar", "next_per_dollar"]])
    growth, growth_cohort = aggregate_growth(pd.concat(paired, ignore_index=True))
    n_years = formation_end-formation_start+1
    expected_growth_cohorts = np.where(growth["relative_end"].le(0), n_years, n_future)
    expected_growth_keys = {(s,l,r,v) for s in ("B/M","C/P-GS") for l in ("Glamour","Value")
                            for r in (range(-4,6) if n_future else range(-4,1)) for v in ("earn","cf","sale")}
    growth_keys = set(growth[["classification","leg","relative_end","variable"]].itertuples(index=False,name=None))
    if growth_keys != expected_growth_keys or not growth["n_cohorts"].eq(expected_growth_cohorts).all():
        raise ValueError("Table V lacks a complete set of accounting growth cohorts; inspect outcome coverage.")
    growth_rows = []
    for (classification, leg, variable), cell in growth.groupby(["classification", "leg", "variable"]):
        cell = cell.set_index("relative_end")
        for begin, end in [(-5, 0), (0, 5), (2, 5)]:
            values = cell["gross_growth"].reindex(range(begin+1, end+1))
            expected_cohorts = n_years if begin < 0 else n_future
            valid = expected_cohorts > 0 and values.notna().all() and values.gt(0).all()
            prefix = {"earn": "AEG", "cf": "ACG", "sale": "ASG"}[variable]
            growth_rows.append({"classification": classification, "leg": leg, "metric": f"{prefix}({begin},{end})",
                                "value": values.prod() ** (1/(end-begin))-1 if valid else np.nan,
                                "n_cohorts": expected_cohorts, "section": "B" if begin < 0 else "C",
                                "status": ("ok" if valid else "five-year accounting outcomes unavailable" if expected_cohorts == 0
                                           else "nonpositive aggregate accounting level")})
    # Three-year preformation stock buy-and-hold returns, with no zero filling.
    history = base.merge(pd.DataFrame({"offset": range(36)}), how="cross")
    history["ym"] = pd.PeriodIndex(history["formation_year"].astype(str)+"-04", freq="M") - history["offset"].to_numpy()
    history = history.merge(monthly[["permno", "ym", "ret"]], on=["permno", "ym"], how="left", validate="many_to_one")
    history["gross"] = 1+history["ret"]
    stock_past = history.groupby(["classification", "leg", "formation_year", "permno"], as_index=False).agg(
        gross=("gross", "prod"), n_months=("ret", "count"))
    stock_past["past_return"] = (stock_past["gross"]-1).where(stock_past["n_months"].eq(36))
    past_cohort = stock_past.groupby(["classification", "leg", "formation_year"], as_index=False).agg(
        value=("past_return", "mean"), n_stocks=("past_return", "count"))
    past = past_cohort.groupby(["classification", "leg"], as_index=False).agg(
        value=("value", "mean"), n_cohorts=("value", "count"))
    past["metric"], past["section"], past["status"] = "RETURN(-3,0)", "B", "ok"
    summary = pd.concat([summary, pd.DataFrame(growth_rows), past], ignore_index=True)
    expected_summary_cohorts = np.where(summary["section"].eq("C"), n_future, n_years)
    if len(summary) != 64 or not summary["n_cohorts"].eq(expected_summary_cohorts).all():
        raise ValueError("Table V is missing required fundamentals or past-return cohorts.")
    return summary, fund_cohort, growth, growth_cohort, past_cohort
