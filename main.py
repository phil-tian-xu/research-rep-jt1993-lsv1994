"""Project entry point; prepare raw data with data_prepare_jt1993.py first."""

from pathlib import Path

from core.jt1993.replication.table1 import run_table1
from core.jt1993.replication.table2 import run_table2
from core.jt1993.replication.table7 import run_table7
from core.jt1993.reporter.table1 import report_table1
from core.jt1993.reporter.table2 import report_table2
from core.jt1993.reporter.table7 import report_table7
from utils.path_info import data_path, results_path


if __name__ == "__main__":
    replication_table = (1, 2, 7)# One table: 1, 2, or 7. All tables: (1, 2, 7).
    run_start_date = "1965-01-01"  # Change dates for each subsample run.
    run_end_date = "1989-12-31"
    # For Table VII, start is the first formation; end is the last holding month.
    # Only formations with all 36 calendar holding months inside this window are used.
    raw_data_dir = (
        Path(data_path) / "raw" / "jt1993" / "nyse_amex_19620701_19891231"
    )
    run_replication = True  # Set False to regenerate the table/report from saved results only.
    show_plots = False     # Set True for interactive Matplotlib windows.
    run_tag = f"{raw_data_dir.name}_{run_start_date[:7].replace('-', '')}_{run_end_date[:7].replace('-', '')}"
    tables = (replication_table,) if isinstance(replication_table, int) else replication_table
    if not tables or any(table not in (1, 2, 7) for table in tables):
        raise ValueError("Select replication_table = 1, 2, 7, or a tuple such as (1, 2, 7).")

    for table in tables:
        print(f"Table {table}: {run_start_date} to {run_end_date}", flush=True)
        result_dir = Path(results_path) / "jt1993" / "replication" / f"table{table}" / run_tag
        if table == 1:
            replicate, reporter = run_table1, report_table1
        elif table == 2:
            replicate, reporter = run_table2, report_table2
        else:
            replicate, reporter = run_table7, report_table7

        if run_replication:
            replicate(
                run_start_date=run_start_date,
                run_end_date=run_end_date,
                raw_data_dir=raw_data_dir,
                output_dir=result_dir,
            )
        if table == 1:
            reporter(result_dir, show=show_plots)
        else:
            reporter(result_dir)
