"""Q1 Task 3: partition existing data before running main.py twice.

Split a completed full sample into the paper's 1965-1989 reporting period
and the remaining 1990-2024 period.
The output folders include warm-up history; use the reporting dates printed
below or saved in each partition_info.json when selecting a subset in main.py.
This script only creates derived subsets; it does not download or run analysis.
"""

from pathlib import Path

from core.jt1993.data_partition import partition_jt1993_data
from utils.path_info import data_path


if __name__ == "__main__":
    run_sample = "nyse_amex"  # "nyse", "nyse_amex", or "nyse_amex_nasdaq".
    raw_data_dir = Path(data_path) / "raw" / "jt1993" / f"{run_sample}_19600101_20241231"
    run_start_date = "1965-01-01"
    split_date = "1990-01-01"  # Original paper period ends in December 1989.
    run_end_date = "2024-12-31"  # Latest complete local snapshot.

    partition_jt1993_data(
        raw_data_dir=raw_data_dir,
        run_start_date=run_start_date,
        split_date=split_date,
        run_end_date=run_end_date,
    )
