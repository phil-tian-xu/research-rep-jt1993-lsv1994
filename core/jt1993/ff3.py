"""Q1-5: regress a saved Table I winner-minus-loser strategy on monthly FF3."""

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

from utils.inference import ols_newey_west
from utils.path_info import results_path, root_path


def run_ff3(
    run_start_date: str,
    run_end_date: str,
    table1_result_dir: str | Path,
    factor_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    panel: str = "A",
    J: int = 6,
    K: int = 6,
    hac_lags: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run a monthly FF3 regression using saved returns and local factor data.

    Dates are inclusive calendar month-start/month-end boundaries inside the
    saved Table I reporting window. Select exactly one panel/J/K strategy.
    Regress buy_sell directly on a constant, Mkt-RF, SMB, and HML: RF cancels
    between the winner and loser excess returns and must not be subtracted again.
    All inputs must be in decimal units. Missing/duplicate months or nonfinite
    observations cause an error rather than an implicit sample change.

    Newey-West uses Bartlett weights, n/(n-4) correction, and asymptotic normal
    two-sided p-values/95% intervals. Five lags is an explicit baseline choice;
    change hac_lags to assess sensitivity. No download or CRSP calculation runs.
    Save full-precision coefficients, aligned monthly inputs/fits, covariance,
    and run_info.json under results/jt1993/ff3/<source/panel/J/K/HAC tag>/.
    """
    start, end = pd.Timestamp(run_start_date), pd.Timestamp(run_end_date)
    if (any(pd.isna(value) or value.tzinfo is not None or value != value.normalize() for value in (start, end))
            or start > end or start.day != 1 or not end.is_month_end):
        raise ValueError("Require calendar month-start start <= month-end end, without timezones.")
    if panel not in ("A", "B") or any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in (J, K)):
        raise ValueError("Select panel A/B and positive integer J and K.")
    table1_result_dir, factor_path = Path(table1_result_dir), Path(factor_path)
    if not table1_result_dir.is_absolute():
        table1_result_dir = Path(root_path) / table1_result_dir
    if not factor_path.is_absolute():
        factor_path = Path(root_path) / factor_path
    table1_result_dir, factor_path = table1_result_dir.resolve(), factor_path.resolve()
    returns_path = table1_result_dir / "table1_monthly_returns.csv"
    source_info = json.loads((table1_result_dir / "run_info.json").read_text(encoding="utf-8"))
    if source_info.get("paper") != "jt1993" or source_info.get("table") != 1:
        raise ValueError("The source must be a saved JT1993 Table I run.")
    if not source_info.get("return_units", "").startswith("decimal"):
        raise ValueError("Saved Table I metadata must declare decimal return units.")
    if not pd.Timestamp(source_info["run_start_date"]) <= start <= end <= pd.Timestamp(source_info["run_end_date"]):
        raise ValueError("Requested dates extend beyond the saved Table I reporting window.")
    if not factor_path.is_file() or not factor_path.with_suffix(".json").is_file():
        raise FileNotFoundError("Missing local FF3 factors or metadata. Run data_prepare_ff3.py first.")
    factor_info = json.loads(factor_path.with_suffix(".json").read_text(encoding="utf-8"))
    if factor_info.get("frequency") != "monthly" or factor_info.get("return_units") != "decimal":
        raise ValueError("FF3 metadata must declare monthly frequency and decimal return units.")
    if factor_info.get("parquet_sha256") != sha256(factor_path.read_bytes()).hexdigest():
        raise ValueError("FF3 data checksum disagrees with its metadata; prepare the factors again.")

    columns = ["month", "panel", "J", "K", "loser", "winner", "buy_sell"]
    returns = pd.read_csv(returns_path, usecols=columns, parse_dates=["month"])
    selected = returns.loc[returns["panel"].eq(panel) & returns["J"].eq(J) & returns["K"].eq(K)].copy()
    if selected["month"].isna().any():
        raise ValueError("Missing dates in the selected momentum returns.")
    selected["month"] = selected["month"].dt.to_period("M").dt.to_timestamp("M")
    selected = selected.loc[selected["month"].between(start, end)].sort_values("month").reset_index(drop=True)
    expected = pd.date_range(start, end, freq="ME")
    if selected["month"].duplicated().any() or not pd.DatetimeIndex(selected["month"]).equals(expected):
        raise ValueError("Selected strategy has duplicate or missing calendar months in the requested window.")
    if not np.isfinite(selected[["loser", "winner", "buy_sell"]].to_numpy()).all():
        raise ValueError("Momentum returns must be finite in every requested month.")
    if not np.allclose(selected["buy_sell"], selected["winner"] - selected["loser"], rtol=1e-10, atol=1e-12):
        raise ValueError("Saved spread returns disagree with winner minus loser.")
    factors = pd.read_parquet(factor_path, columns=["month", "mkt_rf", "smb", "hml", "rf"])
    if factors["month"].isna().any():
        raise ValueError("Missing dates in FF3 factors.")
    factors["month"] = pd.to_datetime(factors["month"]).dt.to_period("M").dt.to_timestamp("M")
    if factors["month"].duplicated().any():
        raise ValueError("Duplicate calendar months in FF3 factors.")
    monthly = selected[["month", "loser", "winner", "buy_sell"]].merge(
        factors, on="month", how="left", validate="one_to_one", sort=True,
    )
    complete = np.isfinite(monthly[["mkt_rf", "smb", "hml"]].to_numpy()).all(axis=1)
    if not complete.all():
        missing = monthly.loc[~complete, "month"].dt.strftime("%Y-%m").tolist()
        raise ValueError(f"Missing/nonfinite monthly FF3 observations: {missing}. No months were dropped.")
    fitted = ols_newey_west(monthly["buy_sell"], monthly[["mkt_rf", "smb", "hml"]], lags=hac_lags)
    terms = ["alpha", "mkt_rf", "smb", "hml"]
    coefficients = pd.DataFrame({"term": terms, **{name: fitted[name] for name in (
        "coefficient", "hac_se", "t_stat", "p_value", "ci95_lower", "ci95_upper",
    )}})
    monthly["fitted_return"], monthly["residual"] = fitted["fitted"], fitted["residual"]
    download_info = source_info["raw_inputs"]["monthly_returns"].get("download_metadata") or {}
    info = {
        "paper": "jt1993", "question": "Q1-5", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_start_date": str(start.date()), "run_end_date": str(end.date()),
        "universe": download_info.get("universe", "unknown"), "panel": panel, "J": J, "K": K,
        "table1_result_dir": str(table1_result_dir), "returns_file": str(returns_path),
        "returns_sha256": sha256(returns_path.read_bytes()).hexdigest(),
        "factor_file": str(factor_path), "factor_metadata": factor_info,
        "model": "buy_sell = alpha + beta_mkt*mkt_rf + beta_smb*smb + beta_hml*hml + residual",
        "risk_free_subtracted_from_spread": False, "return_units": "decimal monthly returns",
        "n_obs": fitted["n_obs"], "df_resid": fitted["df_resid"],
        "r_squared": fitted["r_squared"], "adjusted_r_squared": fitted["adjusted_r_squared"],
        "hac_lags": hac_lags, "hac_kernel": "Bartlett", "covariance_correction": "n/(n-4)",
        "inference": "two-sided asymptotic normal p-values and 95% confidence intervals",
        "alpha_positive_and_significant_5pct": bool(fitted["coefficient"][0] > 0 and fitted["p_value"][0] < 0.05),
    }
    tag = f"{table1_result_dir.name}_{start:%Y%m}_{end:%Y%m}_panel{panel}_J{J}_K{K}_HAC{hac_lags}"
    output_dir = Path(output_dir) if output_dir is not None else Path(results_path) / "jt1993" / "ff3" / tag
    if not output_dir.is_absolute():
        output_dir = Path(root_path) / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    coefficients.to_csv(output_dir / "ff3_coefficients.csv", index=False)
    monthly.to_csv(output_dir / "ff3_monthly_regression.csv", index=False)
    pd.DataFrame(fitted["covariance"], index=pd.Index(terms, name="term"), columns=terms).to_csv(output_dir / "ff3_hac_covariance.csv")
    (output_dir / "run_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(f"FF3 | {info['universe']} | Panel {panel}, J={J}, K={K} | {len(monthly)} months | HAC {hac_lags}", flush=True)
    print(coefficients.to_string(index=False), flush=True)
    print(f"R-squared: {fitted['r_squared']:.4f}; numeric results: {output_dir}", flush=True)
    return coefficients, monthly
