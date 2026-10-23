"""Q1-5: a plain coefficient table from saved FF3 results."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from utils.path_info import root_path
from ._table import write_display_table


def report_ff3(result_dir: str | Path):
    """Write reports/ff3.csv and ff3.html; never download or estimate anything."""
    result_dir = Path(result_dir)
    if not result_dir.is_absolute():
        result_dir = Path(root_path) / result_dir
    frame = pd.read_csv(result_dir / "ff3_coefficients.csv")
    info = json.loads((result_dir / "run_info.json").read_text(encoding="utf-8"))
    fields = ["coefficient", "hac_se", "t_stat", "p_value", "ci95_lower", "ci95_upper"]
    if frame["term"].duplicated().any() or set(frame["term"]) != {"alpha", "mkt_rf", "smb", "hml"}:
        raise ValueError("FF3 report requires exactly alpha and the three factor coefficients.")
    if not np.isfinite(frame[fields].to_numpy()).all():
        raise ValueError("FF3 coefficient estimates and inference must be finite.")
    values = frame.set_index("term")
    rows = [["Term", "Estimate", "HAC SE", "HAC t-stat", "p-value", "95% CI lower", "95% CI upper"]]
    for term, label in (("alpha", "Alpha (% per month)"), ("mkt_rf", "Mkt-RF"), ("smb", "SMB"), ("hml", "HML")):
        record = values.loc[term]
        scale = 100 if term == "alpha" else 1
        rows.append([label, f"{record['coefficient'] * scale:.4f}", f"{record['hac_se'] * scale:.4f}",
                     f"{record['t_stat']:.3f}", f"{record['p_value']:.4f}",
                     f"{record['ci95_lower'] * scale:.4f}", f"{record['ci95_upper'] * scale:.4f}"])
    alpha = values.loc["alpha"]
    conclusion = ("Alpha is positive and significant at the 5% level."
                  if alpha["coefficient"] > 0 and alpha["p_value"] < 0.05
                  else "Alpha is not both positive and significant at the 5% level.")
    notes = (f"{info['universe']}; {info['run_start_date']} to {info['run_end_date']}; "
             f"Panel {info['panel']}, J={info['J']}, K={info['K']}; N={info['n_obs']}; R-squared={info['r_squared']:.4f}. "
             f"Newey-West: {info['hac_lags']} lags, Bartlett kernel, n/(n-4) correction; two-sided normal inference. "
             "Alpha and its SE/CI are in percent per month; factor loadings are dimensionless. "
             "RF cancels between the two portfolio legs. " + conclusion)
    print(conclusion, flush=True)
    return write_display_table(rows, result_dir / "reports" / "ff3", "Q1-5 — Fama-French Three-Factor Regression", notes)
