"""Local monthly-return loading shared by the Table II and VII drafts."""

from pathlib import Path

import numpy as np
import pandas as pd

from utils.path_info import root_path


def load_monthly_sample(start_date, end_date, raw_data_dir, history_months):
    """Validate a complete reported window and its required calendar history."""
    start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
    if not start <= end or start.day != 1 or not end.is_month_end:
        raise ValueError("Require a month-start run_start_date <= a month-end run_end_date.")
    raw_data_dir = Path(raw_data_dir)
    if not raw_data_dir.is_absolute():
        raw_data_dir = Path(root_path) / raw_data_dir
    path = raw_data_dir / "monthly_returns.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Run data_prepare_jt1993.py first.")
    monthly = pd.read_parquet(path, columns=["permno", "month", "ret"])
    monthly["month"] = pd.to_datetime(monthly["month"]) + pd.offsets.MonthEnd(0)
    if monthly[["permno", "month"]].isna().any().any() or monthly.duplicated(["permno", "month"]).any():
        raise ValueError("Missing or duplicate PERMNO-month keys in monthly returns.")
    monthly["permno"] = monthly["permno"].astype("int32")
    monthly["ret"] = monthly["ret"].astype(float)
    monthly.loc[~np.isfinite(monthly["ret"]) | monthly["ret"].lt(-1), "ret"] = np.nan
    monthly["month_id"] = monthly["month"].dt.year * 12 + monthly["month"].dt.month
    expected = pd.period_range(start.to_period("M") - history_months, end, freq="M")
    missing = expected.difference(monthly["month"].dt.to_period("M").unique())
    if len(missing):
        raise ValueError(f"Missing required monthly sample/history: {list(map(str, missing))}")
    return monthly.sort_values(["permno", "month_id"], ignore_index=True), start, end, raw_data_dir


def month_ends(month_ids):
    """Convert year*12+month identifiers to an array of calendar month ends."""
    ids = np.asarray(month_ids, dtype=int)
    return (pd.to_datetime({"year": (ids - 1) // 12, "month": (ids - 1) % 12 + 1, "day": 1})
            + pd.offsets.MonthEnd(0)).to_numpy()
