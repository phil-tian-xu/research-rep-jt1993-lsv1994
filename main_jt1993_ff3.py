"""Q1-5 entry point. Run data_prepare_ff3.py first; edit selections below."""

from pathlib import Path

from core.jt1993.ff3 import run_ff3
from core.jt1993.reporter.ff3 import report_ff3
from utils.path_info import data_path, results_path


if __name__ == "__main__":
    run_start_date = "1965-01-01"
    run_end_date = "1989-12-31"
    table1_run_tag = "nyse_amex_19620701_19891231_196501_198912"
    run_panel = "A"  # "B" for the five-trading-day-gap strategy.
    formation_months = 6
    holding_months = 6
    run_hac_lags = 5  # Change to 12 for a separate sensitivity run.
    run_regression = True  # False regenerates the report from saved FF3 results.

    table1_result_dir = Path(results_path) / "jt1993" / "replication" / "table1" / table1_run_tag
    factor_path = Path(data_path) / "raw" / "ff3" / "ff3_monthly.parquet"
    run_tag = (f"{table1_run_tag}_{run_start_date[:7].replace('-', '')}_{run_end_date[:7].replace('-', '')}"
               f"_panel{run_panel}_J{formation_months}_K{holding_months}_HAC{run_hac_lags}")
    result_dir = Path(results_path) / "jt1993" / "ff3" / run_tag
    if run_regression:
        run_ff3(
            run_start_date=run_start_date, run_end_date=run_end_date,
            table1_result_dir=table1_result_dir, factor_path=factor_path, output_dir=result_dir,
            panel=run_panel, J=formation_months, K=holding_months, hac_lags=run_hac_lags,
        )
    report_ff3(result_dir)
