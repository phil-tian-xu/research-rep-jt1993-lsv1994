"""Table II reporter: P1-P10/P10-P1 rows, beta and average capitalization."""

import json
from pathlib import Path

import pandas as pd

from utils.path_info import root_path
from .._table import write_display_table


def table2_rows(summary):
    """Use calculated values, with the same column/row labels as Table II."""
    order = [*(f"P{i}" for i in range(1, 11)), "P10-P1"]
    if summary["portfolio"].duplicated().any() or set(summary["portfolio"]) != set(order):
        raise ValueError("Table II needs exactly P1-P10 and P10-P1.")
    values = summary.set_index("portfolio")
    rows = [["", "Beta", "Average Market Capitalization"]]
    for portfolio in order:
        record = values.loc[portfolio]
        beta = f"{record['beta']:.2f}" if pd.notna(record["beta"]) else "NA"
        size = (f"{record['average_market_cap_millions']:.2f}"
                if pd.notna(record["average_market_cap_millions"]) else "NA")
        rows.append([portfolio, beta, "—" if portfolio == "P10-P1" else size])
    return rows


def report_table2(result_dir: str | Path, *, output_dir: str | Path | None = None):
    """Read saved Table II summary; write reports/table2.csv and table2.html."""
    result_dir = Path(result_dir)
    if not result_dir.is_absolute():
        result_dir = Path(root_path) / result_dir
    summary = pd.read_csv(result_dir / "table2_summary.csv")
    info = json.loads((result_dir / "run_info.json").read_text(encoding="utf-8"))
    output_dir = Path(output_dir) if output_dir is not None else result_dir / "reports"
    if not output_dir.is_absolute():
        output_dir = Path(root_path) / output_dir
    notes = (f"{info['run_start_date']} to {info['run_end_date']}. Size in USD millions. "
             "Draft: monthly OLS betas; pre-formation capitalization, fixed within each cohort.")
    return write_display_table(table2_rows(summary), output_dir / "table2",
                               "Table II — Betas and Market Capitalization of Relative Strength Portfolios", notes)
