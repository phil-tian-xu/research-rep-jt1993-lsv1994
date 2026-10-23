"""Run Tables I, II, and VII for Task 2's universes over 1965-1989 only.

Use the existing NYSE and NYSE/AMEX/Nasdaq partitions, including 24 months
of past formation history. The frozen Table I reference remains the original
NYSE/AMEX benchmark, so differences from it are expected for these universes.
This batch uses local data and the existing replication/reporting conventions.
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
    run_universes = ("nyse", "nyse_amex_nasdaq")
    run_start_date = "1965-01-01"
    run_end_date = "1989-12-31"
    period_tag = f"{run_start_date[:7].replace('-', '')}_{run_end_date[:7].replace('-', '')}"
    subsample_dir = Path(data_path) / "processed" / "jt1993" / "subsamples"
    raw_data_dirs = {
        universe: subsample_dir / f"{universe}_19600101_20241231_{period_tag}"
        for universe in run_universes
    }
    replications = {
        1: (run_table1, report_table1),
        2: (run_table2, report_table2),
        7: (run_table7, report_table7),
    }

    # Check both samples before starting the batch.
    for raw_data_dir in raw_data_dirs.values():
        for name in ("monthly_returns", "daily_returns", "monthly_characteristics", "market_returns"):
            path = raw_data_dir / f"{name}.parquet"
            if not path.is_file():
                raise FileNotFoundError(f"Missing local input: {path}")

    started = perf_counter()
    completed = 0
    total = len(run_universes) * len(run_tables)
    for universe, raw_data_dir in raw_data_dirs.items():
        run_tag = f"{raw_data_dir.name}_{period_tag}"
        for table in run_tables:
            result_dir = Path(results_path) / "jt1993" / "replication" / f"table{table}" / run_tag
            replicate, reporter = replications[table]
            print(f"\n[{completed + 1}/{total}] {universe}, Table {table}: {run_start_date} to {run_end_date}", flush=True)
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

    print(f"\nAll {completed} universe replications and reports completed in {perf_counter() - started:.1f} seconds.", flush=True)
