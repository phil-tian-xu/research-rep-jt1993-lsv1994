"""HW1 Q1 Task 2: prepare the two exchange samples, running one at a time.

Select run_sample="nyse" and run, then select "nyse_amex_nasdaq" and run again.
Each sample/source/date combination has its own cached raw-data folder.
January 1960 supplies warm-up history for analysis starting January 1962.
The defaults prepare Table I only, using legacy CRSP through December 2024.
The fixed endpoint avoids a WRDS coverage lookup. This file downloads data only.

Both monthly and daily return extracts use crsp.dsf, with crsp.msenames supplying
historical security classifications, as in the reference replication.
"""

from data_prepare_jt1993 import data_prepare_jt1993


if __name__ == "__main__":
    data_start_date = "1960-01-01"  # Warm-up data; report results from 1962-01-01.
    data_end_date = "2024-12-31"  # Fixed extended-sample endpoint; no coverage lookup.
    run_sample = "nyse"           # "nyse" or "nyse_amex_nasdaq"; run each separately.
    refresh_data = False
    run_tables = (1,)             # Table I: monthly and daily returns only.
    run_data_source = "legacy"    # Both return extracts use crsp.dsf, through 2024.

    data_prepare_jt1993(
        start=data_start_date,
        end=data_end_date,
        universe=run_sample,
        refresh=refresh_data,
        tables=run_tables,
        data_source=run_data_source,
    )
