"""Table VII reporter: event months 1-12, 13-24, 25-36 in three column blocks."""

import json
from pathlib import Path

import pandas as pd

from utils.path_info import root_path
from .._table import write_display_table


def table7_rows(summary):
    """Return paper-format means/t-stat rows, using calculated values only."""
    if summary["event_month"].duplicated().any() or set(summary["event_month"]) != set(range(1, 37)):
        raise ValueError("Table VII needs exactly event months 1 through 36.")
    values = summary.set_index("event_month")
    rows = [["t", "Monthly Return", "Cumulative Return"] * 3]
    for first_month in range(1, 13):
        means, tstats = [], []
        for month in (first_month, first_month + 12, first_month + 24):
            record = values.loc[month]
            means.extend([str(month), f"{record['monthly_return']:.4f}", f"{record['cumulative_return']:.4f}"])
            tstats.extend(["", f"({record['monthly_t']:.2f})" if pd.notna(record["monthly_t"]) else "(NA)",
                           f"({record['cumulative_t']:.2f})" if pd.notna(record["cumulative_t"]) else "(NA)"])
        rows.extend([means, tstats])
    return rows


def report_table7(result_dir: str | Path, *, output_dir: str | Path | None = None):
    """Read saved Table VII summary; write reports/table7.csv and table7.html."""
    result_dir = Path(result_dir)
    if not result_dir.is_absolute():
        result_dir = Path(root_path) / result_dir
    summary = pd.read_csv(result_dir / "table7_summary.csv")
    info = json.loads((result_dir / "run_info.json").read_text(encoding="utf-8"))
    output_dir = Path(output_dir) if output_dir is not None else result_dir / "reports"
    if not output_dir.is_absolute():
        output_dir = Path(root_path) / output_dir
    notes = (f"Draft: formations {info['formation_start']} to {info['formation_end']}; "
             f"{info['n_cohorts']} complete cohorts, returns through {info['run_end_date']}. "
             f"Monthly returns in decimals; cumulative returns are sums. T-statistics in parentheses; "
             f"cumulative Newey-West lags: {info['hac_lags']}.")
    return write_display_table(table7_rows(summary), output_dir / "table7",
                               "Table VII — Performance of Relative Strength Portfolios in Event Time", notes)
