"""JT1993 Table I, using local files from data_prepare_jt1993.py.

Paper: Section I and Table I, pp. 68-70. Match the supplied reference notebook:
NYSE/AMEX common shares; daily-compounded monthly returns; equal weights and
monthly rebalancing within each cohort; equal weights across K active cohorts.
Panel A uses J complete calendar months before formation. Panel B shifts the
signal back five trading days, keeping monthly holding returns unchanged.

SAS-style average-rank ties, five trading days as one week, and missing-return
handling follow references/jt1993_table1_wrds_replication.ipynb. These are
implementation conventions, not details fully specified in the paper.
Delisting returns are not added. Missing holding returns are excluded from the
cohort mean; a strategy return requires all K cohorts to have a valid return.
Inference follows the reference: mean / (sample std / sqrt(n)), without HAC.

Run through main.py or call run_table1(); no WRDS connection is made.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from utils.path_info import results_path, root_path


def momentum_deciles(signal):
    """Assign ascending SAS-style deciles, preserving ties and missing signals."""
    decile = pd.Series(pd.NA, index=signal.index, dtype="Int64")
    valid = signal.dropna()
    if len(valid) >= 10:
        groups = np.floor(valid.rank(method="average") * 10 / (len(valid) + 1)) + 1
        decile.loc[valid.index] = groups.clip(1, 10).astype("int64")
    return decile


def make_panel_a_ranks(monthly, J, formation_start, formation_end):
    """Use only consecutive, valid J-month returns ending before formation."""
    x = monthly[["permno", "month_id", "ret"]].sort_values(
        ["permno", "month_id"], ignore_index=True,
    )
    ret = x["ret"].astype(float)
    valid = np.isfinite(ret) & ret.ge(-1)
    # Treat -100% separately so log(0) cannot contaminate later rolling windows.
    x["loggross"] = np.log1p(ret.where(valid & ret.gt(-1), 0)).where(valid)
    x["minus_one"] = (valid & ret.eq(-1)).astype("int8")
    groups = x.groupby("permno", sort=False)
    log_signal = groups["loggross"].rolling(J, min_periods=J).sum().droplevel(0)
    zero_count = groups["minus_one"].rolling(J, min_periods=J).sum().droplevel(0)
    first_month = groups["month_id"].shift(J - 1)
    x["signal"] = np.expm1(log_signal)
    x.loc[zero_count.gt(0) & log_signal.notna(), "signal"] = -1.0
    x.loc[x["month_id"].sub(first_month).ne(J - 1), "signal"] = np.nan
    x["formation_id"] = x["month_id"] + 1
    rank = x.loc[
        x["signal"].notna() & x["formation_id"].between(formation_start, formation_end),
        ["permno", "formation_id", "signal"],
    ].copy()
    rank["momr"] = rank.groupby("formation_id")["signal"].transform(momentum_deciles)
    return rank.dropna(subset=["momr"]).astype({"momr": "int8"})


def panel_b_windows(trading_days, J, formation_start, formation_end, gap_days=5):
    """Reference timing: signal in (end minus J calendar months, end], then gap.

    end is the trading day immediately before the five skipped trading days
    preceding the first trading day of the holding month. The trading calendar
    is the union of dates in the filtered CRSP daily panel, as in the notebook.
    """
    calendar = pd.DataFrame({"date": pd.to_datetime(trading_days)})
    calendar["month_id"] = calendar["date"].dt.year * 12 + calendar["date"].dt.month
    first_trade = calendar.groupby("month_id")["date"].min()
    rows = []
    for formation_id in range(formation_start, formation_end + 1):
        if formation_id not in first_trade.index:
            raise ValueError(f"No daily trading calendar for formation month {formation_id}.")
        hold_start = first_trade.loc[formation_id]
        position = np.searchsorted(trading_days, hold_start.to_datetime64())
        if position <= gap_days:
            raise ValueError("Insufficient daily history for the Panel B gap.")
        signal_end = pd.Timestamp(trading_days[position - gap_days - 1])
        rows.append({
            "formation_id": formation_id, "hold_start": hold_start,
            "signal_start_boundary": signal_end - pd.DateOffset(months=J),
            "signal_end": signal_end,
            "skipped_days": ", ".join(
                pd.to_datetime(trading_days[position - gap_days:position]).strftime("%Y-%m-%d")
            ),
        })
    return pd.DataFrame(rows)


def make_panel_b_ranks(daily, windows):
    """Compound daily signals by stock; daily must be sorted by PERMNO/date.

    Match the reference eligibility rule: a stock's observed history must span
    both boundaries and contain at least one valid return inside the window.
    Missing/invalid daily returns do not contribute; an all-missing signal is
    ineligible, while any -100% return forces the signal to -100%.
    """
    starts = windows["signal_start_boundary"].to_numpy(dtype="datetime64[ns]")
    ends = windows["signal_end"].to_numpy(dtype="datetime64[ns]")
    formation = windows["formation_id"].to_numpy(dtype=int)
    pieces = []
    for permno, stock in daily.groupby("permno", sort=False):
        dates = stock["date"].to_numpy(dtype="datetime64[ns]")
        ret = stock["ret"].to_numpy(dtype=float)
        valid = np.isfinite(ret) & (ret >= -1)
        logret = np.zeros(len(ret), dtype=float)
        positive_gross = valid & (ret > -1)
        logret[positive_gross] = np.log1p(ret[positive_gross])
        cumlog = np.r_[0.0, np.cumsum(logret)]
        cumzero = np.r_[0, np.cumsum(valid & (ret == -1))]
        cumvalid = np.r_[0, np.cumsum(valid)]
        left = np.searchsorted(dates, starts, side="right")
        right = np.searchsorted(dates, ends, side="right")
        spans = (
            (dates[0] <= starts) & (dates[-1] >= ends) & (right > left)
            & ((cumvalid[right] - cumvalid[left]) > 0)
        )
        signal = np.expm1(cumlog[right] - cumlog[left])
        signal[(cumzero[right] - cumzero[left]) > 0] = -1.0
        keep = spans & np.isfinite(signal)
        if keep.any():
            pieces.append(pd.DataFrame({
                "permno": int(permno), "formation_id": formation[keep], "signal": signal[keep],
            }))
    if not pieces:
        raise ValueError("No eligible Panel B signals; check raw history and sample filters.")
    rank = pd.concat(pieces, ignore_index=True)
    rank["momr"] = rank.groupby("formation_id")["signal"].transform(momentum_deciles)
    return rank.dropna(subset=["momr"]).astype({"momr": "int8"})


def build_cohort_returns(rank, monthly, max_k=12):
    """Rebalance each fixed formation cohort to equal stock weights every month."""
    base = rank.loc[rank["momr"].isin([1, 10]), ["permno", "formation_id", "momr"]]
    expanded = base.merge(pd.DataFrame({"age": np.arange(max_k, dtype="int8")}), how="cross")
    expanded["hold_id"] = expanded["formation_id"] + expanded["age"]
    hold = monthly[["permno", "month_id", "ret"]].rename(columns={"month_id": "hold_id"})
    merged = expanded.merge(hold, on=["permno", "hold_id"], how="left", validate="many_to_one")
    return merged.groupby(["formation_id", "hold_id", "age", "momr"], as_index=False).agg(
        cohort_ret=("ret", "mean"), n_constituents=("permno", "size"),
        n_with_ret=("ret", "count"),
    )


def summarize_strategy(cohort, K, start_id, end_id):
    """Average K cohort returns per month, then calculate Table I means and t's."""
    selected = cohort.loc[cohort["age"].lt(K) & cohort["hold_id"].between(start_id, end_id)]
    portfolio = selected.groupby(["hold_id", "momr"], as_index=False).agg(
        ret=("cohort_ret", "mean"), n_cohorts=("cohort_ret", "count"),
        n_active=("formation_id", "nunique"),
    )
    portfolio.loc[portfolio["n_cohorts"].lt(K), "ret"] = np.nan
    ts = portfolio.pivot(index="hold_id", columns="momr", values="ret").reindex(
        index=range(start_id, end_id + 1), columns=[1, 10],
    ).rename(columns={1: "loser", 10: "winner"})
    ts["buy_sell"] = ts["winner"] - ts["loser"]
    n = ts.count()
    standard_error = ts.std(ddof=1) / np.sqrt(n)
    stats = pd.DataFrame({
        "mean": ts.mean(), "t": ts.mean() / standard_error.where(standard_error.gt(0)), "n": n,
    })
    counts = portfolio.pivot(index="hold_id", columns="momr", values="n_cohorts").reindex(
        index=ts.index, columns=[1, 10],
    ).fillna(0)
    return stats, ts, counts


def run_table1(
    run_start_date: str,
    run_end_date: str,
    raw_data_dir: str | Path,
    output_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run both panels and save numeric results for a separate reporter.

    Start at a calendar month start and end at a calendar month end. The raw
    folder selects the downloaded universe and must include earlier ranking
    and overlapping-cohort history. Relative paths use utils.path_info.root_path.
    Return the summary and monthly strategy returns, both in decimal units.
    Frozen reference/paper comparisons apply only to January 1965-December 1989.
    The default output folder is results/jt1993/replication/table1/<sample_dates>/.
    CSV files hold summary statistics, monthly returns, cohort returns, formation
    counts, and Panel B timing. run_info.json records inputs, units, and settings.
    """
    start, end = pd.Timestamp(run_start_date), pd.Timestamp(run_end_date)
    if not start <= end or start.day != 1 or not end.is_month_end:
        raise ValueError("Require a month-start run_start_date <= a month-end run_end_date.")

    raw_data_dir = Path(raw_data_dir)
    if not raw_data_dir.is_absolute():
        raw_data_dir = Path(root_path) / raw_data_dir
    monthly_path = raw_data_dir / "monthly_returns.parquet"
    daily_path = raw_data_dir / "daily_returns.parquet"
    for path in (monthly_path, daily_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {path}. Run data_prepare_jt1993.py first, "
                "then set raw_data_dir to its output folder."
            )

    # 1. Load raw returns; never trim away pre-sample ranking/cohort history.
    monthly = pd.read_parquet(monthly_path, columns=["permno", "month", "ret"])
    monthly["permno"] = monthly["permno"].astype("int32")
    monthly["month"] = pd.to_datetime(monthly["month"]) + pd.offsets.MonthEnd(0)
    monthly["ret"] = monthly["ret"].astype(float)
    monthly.loc[~np.isfinite(monthly["ret"]) | monthly["ret"].lt(-1), "ret"] = np.nan
    if monthly[["permno", "month"]].isna().any().any() or monthly.duplicated(["permno", "month"]).any():
        raise ValueError("Missing or duplicate PERMNO-month keys in monthly input.")
    monthly = monthly.sort_values(["permno", "month"], ignore_index=True)
    monthly["month_id"] = monthly["month"].dt.year * 12 + monthly["month"].dt.month
    expected_months = pd.period_range(start, end, freq="M")
    missing = expected_months.difference(monthly["month"].dt.to_period("M").unique())
    if len(missing):
        raise ValueError(f"Raw data do not cover the reported months: {list(map(str, missing))}")
    reported = monthly.loc[monthly["month"].between(start, end)]

    print(f"JT1993 | returns {start:%Y-%m} to {end:%Y-%m}")
    print(f"Raw folder: {raw_data_dir}")
    print(f"Monthly input: {len(monthly):,} rows, {monthly['permno'].nunique():,} stocks")
    print(f"Raw months: {monthly['month'].min():%Y-%m} to {monthly['month'].max():%Y-%m}")
    print(f"Reported window: {len(reported):,} stock-months, {len(expected_months):,} months")
    daily = pd.read_parquet(daily_path, columns=["permno", "date", "ret"])
    daily["permno"] = daily["permno"].astype("int32")
    daily["date"] = pd.to_datetime(daily["date"])
    daily["ret"] = daily["ret"].astype(float)
    if daily[["permno", "date"]].isna().any().any() or daily.duplicated(["permno", "date"]).any():
        raise ValueError("Missing or duplicate PERMNO-date keys in daily input.")
    daily = daily.sort_values(["permno", "date"], ignore_index=True)
    trading_days = np.sort(daily["date"].unique().astype("datetime64[ns]"))
    print(f"Daily input: {len(daily):,} rows")

    # 2. Include K-1 pre-sample formation cohorts, even for the first reported month.
    JS = KS = (3, 6, 9, 12)
    start_id, end_id = start.year * 12 + start.month, end.year * 12 + end.month
    formation_start = start_id - (max(KS) - 1)
    history_start = start.to_period("M") - (max(JS) + max(KS) - 1)
    history_months = pd.period_range(history_start, end, freq="M")
    missing_history = history_months.difference(monthly["month"].dt.to_period("M").unique())
    if len(missing_history):
        raise ValueError(f"Insufficient monthly ranking/cohort history: {list(map(str, missing_history))}")

    summary_rows, monthly_series, cohort_frames, rank_audits, timing_frames = [], [], [], [], []
    for panel in ("A", "B"):
        for J in JS:
            if panel == "A":
                rank = make_panel_a_ranks(monthly, J, formation_start, end_id)
            else:
                windows = panel_b_windows(trading_days, J, formation_start, end_id)
                if daily["date"].min() > windows["signal_start_boundary"].min():
                    raise ValueError("Insufficient daily history for the earliest Panel B signal.")
                timing_frames.append(windows.assign(J=J))
                rank = make_panel_b_ranks(daily, windows)
            if rank.empty:
                raise ValueError(f"No decile portfolios for Panel {panel}, J={J}; need at least 10 stocks.")
            print(f"Panel {panel}, J={J}: {len(rank):,} ranked stock-formations", flush=True)
            rank_audits.append(rank.groupby(["formation_id", "momr"], as_index=False).agg(
                n_stocks=("permno", "size"), mean_signal=("signal", "mean"),
            ).assign(panel=panel, J=J))
            cohort = build_cohort_returns(rank, monthly, max_k=max(KS))
            cohort_frames.append(cohort.assign(panel=panel, J=J))
            for K in KS:
                stats, ts, counts = summarize_strategy(cohort, K, start_id, end_id)
                if not stats["n"].eq(len(expected_months)).all():
                    raise ValueError(f"Incomplete K-cohort returns for Panel {panel}, J={J}, K={K}.")
                for leg in ("loser", "winner", "buy_sell"):
                    summary_rows.append({
                        "panel": panel, "J": J, "K": K, "leg": leg,
                        "mean": stats.loc[leg, "mean"], "t": stats.loc[leg, "t"],
                        "n": int(stats.loc[leg, "n"]), "min_valid_cohorts": int(counts.min().min()),
                    })
                ts["n_loser_cohorts"], ts["n_winner_cohorts"] = counts[1], counts[10]
                ids = ts.index.to_numpy()
                ts["month"] = (pd.to_datetime({
                    "year": (ids - 1) // 12, "month": (ids - 1) % 12 + 1, "day": 1,
                }) + pd.offsets.MonthEnd(0)).to_numpy()
                monthly_series.append(ts.rename_axis("month_id").reset_index().assign(panel=panel, J=J, K=K))

    # 3. Compare with frozen outputs of the supplied notebook, never rerun its WRDS cells.
    summary = pd.DataFrame(summary_rows)
    strategy_returns = pd.concat(monthly_series, ignore_index=True)
    if start == pd.Timestamp("1965-01-01") and end == pd.Timestamp("1989-12-31"):
        reference = pd.read_csv(Path(root_path) / "references" / "jt1993_table1_reference_results.csv")
        summary = summary.merge(reference, on=["panel", "J", "K", "leg"], how="left", validate="one_to_one")
        if summary["reference_n"].isna().any():
            raise ValueError("Missing strategy rows in the frozen reference results.")
        summary["reference_diff_bp"] = (summary["mean"] - summary["reference_mean"]) * 10_000
        summary["reference_t_diff"] = summary["t"] - summary["reference_t"]
        summary["paper_diff_bp"] = (summary["mean"] - summary["paper_mean"]) * 10_000
    else:
        print("Reference/paper comparison skipped: their reported sample is 1965-1989.")
    for panel in ("A", "B"):
        table = summary.loc[summary["panel"].eq(panel)]
        print(f"\nTable I Panel {panel}: decimal mean monthly return (t-statistic)")
        display = table.assign(value=table.apply(lambda r: f"{r['mean']:.6f} ({r['t']:.3f})", axis=1))
        print(display.pivot(index=["J", "leg"], columns="K", values="value").reindex(
            pd.MultiIndex.from_product([JS, ("loser", "winner", "buy_sell")], names=["J", "leg"]),
        ).to_string())
        if "reference_diff_bp" in table:
            print(f"Max absolute return difference vs reference: {table['reference_diff_bp'].abs().max():.3g} bp/month")
            print(f"Max absolute t-stat difference vs reference: {table['reference_t_diff'].abs().max():.3g}")
            wl = table.loc[table["leg"].eq("buy_sell")]
            print(f"Mean absolute W-L difference vs paper: {wl['paper_diff_bp'].abs().mean():.4f} bp/month")

    # 4. Keep returns and compact diagnostics for checking the initial replication.
    output_dir = Path(output_dir) if output_dir is not None else (
        Path(results_path) / "jt1993" / "replication" / "table1"
        / f"{raw_data_dir.name}_{start:%Y%m}_{end:%Y%m}"
    )
    if not output_dir.is_absolute():
        output_dir = Path(root_path) / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "table1_summary.csv", index=False)
    strategy_returns.to_csv(output_dir / "table1_monthly_returns.csv", index=False)
    pd.concat(cohort_frames, ignore_index=True).to_csv(output_dir / "table1_cohort_returns.csv", index=False)
    pd.concat(rank_audits, ignore_index=True).to_csv(output_dir / "table1_formation_counts.csv", index=False)
    pd.concat(timing_frames, ignore_index=True).to_csv(output_dir / "panel_b_timing.csv", index=False)
    monthly_coverage = reported.groupby("month").agg(
        n_stocks=("permno", "nunique"), n_returns=("ret", "count"), mean_ret=("ret", "mean"),
    )
    monthly_coverage["missing_rate"] = 1 - monthly_coverage["n_returns"] / monthly_coverage["n_stocks"]
    monthly_coverage.to_csv(output_dir / "table1_monthly_coverage.csv")

    raw_inputs = {}
    for name, path, frame, date_column in (
        ("monthly_returns", monthly_path, monthly, "month"),
        ("daily_returns", daily_path, daily, "date"),
    ):
        sidecar = path.with_suffix(".json")
        raw_inputs[name] = {
            "path": str(path), "rows": len(frame),
            "first_date": frame[date_column].min().date().isoformat(),
            "last_date": frame[date_column].max().date().isoformat(),
            "download_metadata": json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.is_file() else None,
        }
    run_info = {
        "paper": "jt1993", "table": 1,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_start_date": start.date().isoformat(), "run_end_date": end.date().isoformat(),
        "raw_inputs": raw_inputs,
        "settings": {
            "ranking_months": list(JS), "holding_months": list(KS), "panels": ["A", "B"],
            "panel_b_gap_trading_days": 5, "decile_method": "SAS average ranks",
            "stock_weights": "equal, rebalanced monthly within each fixed cohort",
            "cohort_weights": "equal across K active cohorts; require K valid cohort returns",
            "missing_holding_returns": "excluded from cohort mean, never filled with zero",
            "delisting_returns_added": False,
            "t_statistic": "mean / (sample standard deviation / sqrt(n)); no HAC",
        },
        "return_units": "decimal monthly returns; difference columns ending in _bp are basis points",
        "month_id_convention": "year * 12 + month, with January=1",
        "reference_comparison_applied": "reference_mean" in summary.columns,
        "outputs": {
            "table1_summary.csv": "96 rows: means, t-statistics, n, and available reference/paper comparisons",
            "table1_monthly_returns.csv": "monthly loser, winner, buy_sell and valid cohort counts for each panel/J/K",
            "table1_cohort_returns.csv": "cohort means and constituent/valid-return counts, including pre-sample cohorts",
            "table1_formation_counts.csv": "stock counts and average signals by panel/J/formation/decile",
            "panel_b_timing.csv": "daily signal boundaries, skipped dates, and holding start by J/formation",
            "table1_monthly_coverage.csv": "reported stock-month coverage, valid-return counts, missing rate, and equal-weighted sample mean",
        },
        "versions": {"numpy": np.__version__, "pandas": pd.__version__},
    }
    (output_dir / "run_info.json").write_text(
        json.dumps(run_info, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    print(f"\nResults saved: {output_dir}")
    return summary, strategy_returns
