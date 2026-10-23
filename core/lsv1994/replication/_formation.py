"""April formations following the supplied WRDS Table I notebook.

Year t uses accounts with a calendar datadate in t-1. Future accounts never
enter formation signals or eligibility. See ../README.md for conventions.
"""

import numpy as np
import pandas as pd


SIGNALS = {"B/M": "bm", "C/P": "cp", "E/P": "ep", "GS": "gs"}
UNIVERSES = {"nyse": (1,), "nyse_amex": (1, 2), "nyse_amex_nasdaq": (1, 2, 3)}


def ntile_series(values, n=10):
    """Ascending SAS PROC RANK groups, keeping average-rank ties together."""
    out = pd.Series(pd.NA, index=values.index, dtype="Int64")
    valid = values.dropna()
    if len(valid) >= n:
        ranks = valid.rank(method="average")
        out.loc[valid.index] = (np.floor(ranks * n / (len(valid) + 1)) + 1).clip(1, n).astype(int)
    return out


def prepare_crsp(frame, universe="nyse_amex"):
    if universe not in UNIVERSES:
        raise ValueError(f"Unknown universe: {universe}")
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    for col in ("permno", "ret", "prc", "shrout", "shrcd", "exchcd"):
        frame[col] = pd.to_numeric(frame[col], errors="raise").astype(float)
    if frame[["permno", "date"]].isna().any().any():
        raise ValueError("Missing CRSP observation key.")
    if frame["permno"].mod(1).ne(0).any():
        raise ValueError("Noninteger CRSP PERMNO.")
    frame["permno"] = frame["permno"].astype("int64")
    frame["ym"] = frame["date"].dt.to_period("M")
    if frame.duplicated(["permno", "ym"]).any():
        raise ValueError("Duplicate CRSP PERMNO-month observations.")
    if frame["ret"].lt(-1).any() or np.isinf(frame["ret"]).any():
        raise ValueError("Invalid decimal CRSP return (less than -1 or infinite).")
    frame["year"] = frame["date"].dt.year
    frame["month"] = frame["date"].dt.month
    frame["me"] = (frame["prc"].abs() * frame["shrout"] / 1000).replace([np.inf, -np.inf], np.nan)
    frame["eligible"] = (frame["shrcd"].isin([10, 11])
                         & frame["exchcd"].isin(UNIVERSES[universe]) & frame["me"].gt(0))
    return frame.sort_values(["permno", "date"]).reset_index(drop=True)


def prepare_accounts(frame, book_equity="ceq", accounting_currency=None):
    """Resolve calendar-year records and calculate only backward sales growth."""
    frame = frame.copy()
    if accounting_currency is not None:
        if "curcd" not in frame:
            raise ValueError("Currency restriction requires the Compustat curcd field.")
        frame = frame.loc[frame["curcd"].eq(accounting_currency)].copy()
    frame["gvkey"] = frame["gvkey"].astype("string")
    frame["datadate"] = pd.to_datetime(frame["datadate"], errors="raise")
    for col in ("ceq", "seq", "pstk", "pstkrv", "pstkl", "txditc", "ib", "dp", "sale"):
        frame[col] = pd.to_numeric(frame[col], errors="raise").astype(float)
    frame = frame.dropna(subset=["gvkey", "datadate"])
    frame["acct_year"] = frame["datadate"].dt.year
    # The notebook keeps the last record in each calendar year. Stable source
    # ordering makes same-date ties reproducible; retained source fields permit
    # inspection of population/currency choices instead of hiding them.
    frame = (frame.sort_values(["gvkey", "acct_year", "datadate"], kind="stable")
             .drop_duplicates(["gvkey", "acct_year"], keep="last").reset_index(drop=True))
    preferred = frame["pstkrv"].combine_first(frame["pstkl"]).combine_first(frame["pstk"])
    seq = frame["seq"].combine_first(frame["ceq"] + frame["pstk"].fillna(0))
    book = {"ceq": frame["ceq"], "seq": seq,
            "ff": seq + frame["txditc"].fillna(0) - preferred.fillna(0)}
    if book_equity not in book:
        raise ValueError("book_equity must be 'ceq', 'seq', or 'ff'.")
    frame["be"] = book[book_equity]
    frame["earn"] = frame["ib"]
    frame["cf"] = frame["ib"] + frame["dp"]
    grouped = frame.groupby("gvkey", sort=False)
    lag_sale = grouped["sale"].shift()
    consecutive = frame["acct_year"].sub(grouped["acct_year"].shift()).eq(1)
    frame["sales_growth"] = (frame["sale"] / lag_sale - 1).where(
        consecutive & frame["sale"].gt(0) & lag_sale.gt(0))
    return frame


def build_formation(crsp, accounts, links, formation_start=1968, formation_end=1989):
    links = links.copy()
    links["gvkey"] = links["gvkey"].astype("string")
    links["permno"] = pd.to_numeric(links["permno"], errors="raise")
    links["linkdt"] = pd.to_datetime(links["linkdt"]).fillna(pd.Timestamp("1900-01-01"))
    links["linkenddt"] = pd.to_datetime(links["linkenddt"]).fillna(pd.Timestamp("2099-12-31"))
    links = links.loc[links["linktype"].isin(["LC", "LU", "LS"])]
    acct = accounts.loc[accounts["acct_year"].between(formation_start - 1, formation_end - 1)].copy()
    acct["formation_year"] = acct["acct_year"] + 1
    acct["formation_date"] = pd.to_datetime(acct["formation_year"].astype(str) + "-04-30")
    linked = acct.merge(links, on="gvkey", how="inner")
    linked = linked.loc[linked["permno"].notna() & linked["formation_date"].ge(linked["linkdt"])
                        & linked["formation_date"].le(linked["linkenddt"])].copy()
    linked["permno"] = linked["permno"].astype("int64")
    linked["linkprim_rank"] = linked["linkprim"].map({"P": 0, "C": 1}).fillna(9)
    linked["linktype_rank"] = linked["linktype"].map({"LC": 0, "LU": 1, "LS": 2}).fillna(9)
    linked = (linked.sort_values(["permno", "formation_year", "linkprim_rank", "linktype_rank", "datadate"],
                                ascending=[True, True, True, True, False], kind="stable")
              .drop_duplicates(["permno", "formation_year"], keep="first"))
    april = crsp.loc[crsp["month"].eq(4) & crsp["eligible"],
                     ["permno", "year", "me", "shrcd", "exchcd"]].rename(
                         columns={"year": "formation_year", "me": "me_apr"})
    formation = linked.merge(april, on=["permno", "formation_year"], validate="one_to_one")
    for signal, numerator in (("bm", "be"), ("cp", "cf"), ("ep", "earn")):
        formation[signal] = (formation[numerator] / formation["me_apr"]).replace([np.inf, -np.inf], np.nan)
    base = formation[["gvkey", "permno", "formation_year"]]
    hist = base.merge(pd.DataFrame({"lag": np.arange(1, 6)}), how="cross")
    hist["acct_year"] = hist["formation_year"] - hist["lag"]
    hist = hist.merge(accounts[["gvkey", "acct_year", "sales_growth"]],
                      on=["gvkey", "acct_year"], how="left", validate="many_to_one")
    hist["annual_rank"] = hist.groupby(["formation_year", "lag"])["sales_growth"].rank(
        method="average", pct=True)
    hist["weighted_rank"] = hist["annual_rank"] * (6 - hist["lag"])
    gs = hist.groupby(["gvkey", "permno", "formation_year"], as_index=False).agg(
        n_gs=("annual_rank", "count"), gs_num=("weighted_rank", "sum"))
    gs["gs"] = (gs["gs_num"] / 15).where(gs["n_gs"].eq(5))
    formation = formation.merge(gs[["gvkey", "permno", "formation_year", "gs"]],
                                on=["gvkey", "permno", "formation_year"], validate="one_to_one")
    missing = set(range(formation_start, formation_end + 1)) - set(formation["formation_year"])
    if missing:
        raise ValueError(f"No eligible linked April formations in years {sorted(missing)}.")
    return formation


def make_constituents(formation):
    """Independent ascending deciles; positive C/P and E/P sorts only."""
    keys = ["formation_year", "permno"]
    members = []
    for strategy, signal in SIGNALS.items():
        x = formation[keys + [signal]].dropna().copy()
        if signal in ("cp", "ep"):
            x = x.loc[x[signal].gt(0)].copy()
        x["portfolio"] = x.groupby("formation_year")[signal].transform(ntile_series)
        x["strategy"] = strategy
        members.append(x[keys + ["strategy", "portfolio"]].dropna())
    result = pd.concat(members, ignore_index=True)
    result["portfolio"] = result["portfolio"].astype("int8")
    if result.duplicated(keys + ["strategy"]).any():
        raise ValueError("Duplicate constituent within a strategy/formation.")
    return result
