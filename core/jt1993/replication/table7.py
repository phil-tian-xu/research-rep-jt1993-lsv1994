"""Draft Table VII: six-month signals and 36 event months of zero-cost returns.

Use Panel A formation, fixed constituents, and monthly equal-weight rebalancing.
Each event return is a SINGLE cohort's P10-P1, not the K-cohort calendar-time
strategy of Table I. Cumulative returns sum monthly W-L, without compounding.

Boundary convention (not fully specified in the paper): formations start at
run_start_date and stop 35 months before run_end_date. Thus every formation has
36 calendar months of follow-up, all within the requested data window. The
1965-1989 window gives Jan 1965-Jan 1987 formations (265 cohorts). This is a calendar cutoff,
not a requirement that individual stocks survive or have 36 valid returns.

Monthly t-statistics are conventional. Cumulative t-statistics use Newey-West,
Bartlett weights, default lag h-1 at horizon h, and n/(n-1) correction. The
paper specifies Newey-West but not its bandwidth or finite-sample adjustment.
These draft choices must be assessed before claiming a Table VII replication
match. No future data or WRDS connection is used.
"""

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from utils.path_info import results_path, root_path
from ._monthly import load_monthly_sample, month_ends
from .table1 import build_cohort_returns, make_panel_a_ranks


def newey_west_mean(values, lags):
    """Return mean, standard error, t; observations must be consecutive cohorts.

    Intercept-only Newey-West variance with Bartlett kernel and n/(n-1)
    correction. Missing observations are rejected so gaps are not compressed.
    With zero lags this equals the conventional sample-standard-deviation SE.
    """
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("Inference requires at least two finite, consecutive cohort observations.")
    if not isinstance(lags, int) or isinstance(lags, bool) or not 0 <= lags < len(x):
        raise ValueError("Newey-West lags must be an integer >=0 and smaller than the cohort count.")
    mean = float(x.mean())
    residual = x - mean
    meat = float(residual @ residual)
    for lag in range(1, lags + 1):
        meat += 2 * (1 - lag / (lags + 1)) * float(residual[lag:] @ residual[:-lag])
    se = np.sqrt(max(meat, 0) / (len(x) * (len(x) - 1)))
    return mean, float(se), mean / se if se > 0 else np.nan


def summarize_table7(cohort, formation_ids, hac_lags=None):
    """Summarize a balanced formation-by-event-month panel without survivor filtering."""
    event = cohort.pivot(index=["formation_id", "hold_id", "age"], columns="momr", values="cohort_ret")
    event = event.reindex(columns=[1, 10]).rename(columns={1: "loser", 10: "winner"}).reset_index()
    event["event_month"] = event["age"] + 1
    event["monthly_return"] = event["winner"] - event["loser"]
    panel = event.pivot(index="formation_id", columns="event_month", values="monthly_return").reindex(
        index=formation_ids, columns=range(1, 37),
    )
    if panel.isna().any().any() or not np.isfinite(panel.to_numpy()).all():
        raise ValueError("Missing event-month returns: each calendar-selected cohort needs both portfolio legs in all 36 months.")
    cumulative = panel.cumsum(axis=1)
    rows = []
    for horizon in range(1, 37):
        monthly_mean, monthly_se, monthly_t = newey_west_mean(panel[horizon], 0)
        lags = horizon - 1 if hac_lags is None else hac_lags
        cumulative_mean, cumulative_se, cumulative_t = newey_west_mean(cumulative[horizon], lags)
        rows.append({
            "event_month": horizon, "monthly_return": monthly_mean, "monthly_t": monthly_t,
            "cumulative_return": cumulative_mean, "cumulative_t": cumulative_t,
            "monthly_se": monthly_se, "cumulative_se": cumulative_se,
            "n_cohorts": len(panel), "hac_lags": lags,
        })
    event = event.sort_values(["formation_id", "event_month"], ignore_index=True)
    event["cumulative_return"] = event.groupby("formation_id")["monthly_return"].cumsum()
    event["formation_month"] = month_ends(event["formation_id"])
    event["holding_month"] = month_ends(event["hold_id"])
    return pd.DataFrame(rows), event


def run_table7(run_start_date: str, run_end_date: str, raw_data_dir: str | Path,
               output_dir: str | Path | None = None, *, hac_lags: int | None = None
               ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the balanced-cohort draft using only local monthly return extracts.

    run_start_date is the first formation month; run_end_date is the final
    allowed holding-return month. At least 36 follow-up months plus enough
    monthly formations for the specified HAC bandwidth are needed.
    """
    monthly, start, end, raw_data_dir = load_monthly_sample(
        run_start_date, run_end_date, raw_data_dir, history_months=6,
    )
    start_id, end_id = start.year * 12 + start.month, end.year * 12 + end.month
    formation_end = end_id - 35
    formation_ids = np.arange(start_id, formation_end + 1)
    max_lags = 35 if hac_lags is None else hac_lags
    if (not isinstance(max_lags, int) or isinstance(max_lags, bool) or max_lags < 0
            or len(formation_ids) < max(2, max_lags + 1)):
        raise ValueError("Too few complete 36-month cohorts for the requested Newey-West lags.")
    rank = make_panel_a_ranks(monthly, J=6, formation_start=start_id, formation_end=formation_end)
    cohort = build_cohort_returns(rank, monthly, max_k=36)
    summary, event = summarize_table7(cohort, formation_ids, hac_lags=hac_lags)
    output_dir = Path(output_dir) if output_dir is not None else (
        Path(results_path) / "jt1993" / "replication" / "table7" / f"{raw_data_dir.name}_{start:%Y%m}_{end:%Y%m}"
    )
    if not output_dir.is_absolute():
        output_dir = Path(root_path) / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "table7_summary.csv", index=False)
    event.to_csv(output_dir / "table7_event_returns.csv", index=False)
    cohort.to_csv(output_dir / "table7_cohort_returns.csv", index=False)
    rank.groupby(["formation_id", "momr"], as_index=False).agg(
        n_stocks=("permno", "size"), mean_signal=("signal", "mean"),
    ).to_csv(output_dir / "table7_formation_counts.csv", index=False)
    info = {
        "paper": "jt1993", "table": 7, "status": "draft; not numerically validated against the paper",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_start_date": start.date().isoformat(), "run_end_date": end.date().isoformat(),
        "formation_start": str(pd.Timestamp(month_ends([start_id])[0]).date()),
        "formation_end": str(pd.Timestamp(month_ends([formation_end])[0]).date()),
        "n_cohorts": len(formation_ids), "raw_data_dir": str(raw_data_dir), "J": 6, "event_months": 36,
        "sample_policy": "balanced calendar cohorts; latest formation is 35 months before final return month",
        "stock_weights": "fixed cohort constituents, equal weights rebalanced monthly; missing stock returns excluded",
        "cumulative_method": "sum monthly P10-P1 returns within each formation cohort, then average cohorts",
        "monthly_inference": "conventional mean / (sample std / sqrt(n))",
        "cumulative_inference": "Newey-West Bartlett, n/(n-1) correction",
        "hac_lags": "horizon minus one" if hac_lags is None else hac_lags,
        "delisting_returns_added": False,
        "paper_ambiguities": "complete-cohort boundary rule, HAC bandwidth and finite-sample correction are explicit draft conventions",
        "return_units": "decimal; cumulative returns are additive",
    }
    (output_dir / "run_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(f"Table VII numeric results: {output_dir}; {len(formation_ids)} complete 36-month cohorts")
    return summary, event
