"""HW1 Q1 Task 3: run Tables I, II, and VII for both NYSE/AMEX subperiods.

Use the existing local partitions, including their past formation history.
Reporting periods are 1965-1989 and 1990-2024. Table VII uses the existing
complete-36-month-cohort convention inside each period. No data are downloaded.
Results use the same table/sample/date folders as main.py.
"""

from pathlib import Path
from time import perf_counter

from core.jt1993.replication.table1 import run_table1
from core.jt1993.replication.table2 import run_table2
from core.jt1993.replication.table7 import run_table7
from core.jt1993.reporter.table1 import report_table1
from core.jt1993.reporter.table2 import report_table2
from core.jt1993.reporter.table7 import report_table7
from utils.path_info import data_path, results_path


if __name__ == "__main__":
    run_tables = (1, 2, 7)
    run_periods = (
        ("1965-01-01", "1989-12-31", "nyse_amex_19600101_20241231_196501_198912"),
        ("1990-01-01", "2024-12-31", "nyse_amex_19600101_20241231_199001_202412"),
    )
    subsample_dir = Path(data_path) / "processed" / "jt1993" / "subsamples"
    replications = {
        1: (run_table1, report_table1),
        2: (run_table2, report_table2),
        7: (run_table7, report_table7),
    }

    # Check both samples before starting the batch.
    for _, _, folder in run_periods:
        for name in ("monthly_returns", "daily_returns", "monthly_characteristics", "market_returns"):
            path = subsample_dir / folder / f"{name}.parquet"
            if not path.is_file():
                raise FileNotFoundError(f"Missing local input: {path}")

    started = perf_counter()
    completed = 0
    total = len(run_periods) * len(run_tables)
    for run_start_date, run_end_date, folder in run_periods:
        raw_data_dir = subsample_dir / folder
        run_tag = f"{folder}_{run_start_date[:7].replace('-', '')}_{run_end_date[:7].replace('-', '')}"
        for table in run_tables:
            result_dir = Path(results_path) / "jt1993" / "replication" / f"table{table}" / run_tag
            replicate, reporter = replications[table]
            print(f"\n[{completed + 1}/{total}] Table {table}: {run_start_date} to {run_end_date}", flush=True)
            replicate(
                run_start_date=run_start_date,
                run_end_date=run_end_date,
                raw_data_dir=raw_data_dir,
                output_dir=result_dir,
            )
            if table == 1:
                reporter(result_dir, show=False)
            else:
                reporter(result_dir)
            completed += 1
            print(f"[{completed}/{total}] Complete: {result_dir / 'reports' / f'table{table}.csv'}", flush=True)

    print(f"\nAll {completed} replications and reports completed in {perf_counter() - started:.1f} seconds.", flush=True)
