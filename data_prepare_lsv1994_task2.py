"""HW1 Q2 Task 2: prepare extended LSV inputs for the exchange samples.

Select run_sample="nyse", "nyse_amex", or "nyse_amex_nasdaq" below and run.
All selections share one broad legacy raw extract. Valid cached inputs are
reused and only missing datasets are queried. Eligibility belongs at April formation:
NYSE uses exchcd=1, NYSE/AMEX uses (1, 2), and NYSE/AMEX/Nasdaq uses (1, 2, 3).
The NYSE/AMEX option extends the original paper's stock universe through 2024.
All samples use common shares (shrcd 10/11). Retain original constituents even
after an exchange change, so the raw extract is not filtered here.

The defaults use legacy CRSP and Compustat from 1957 through December 2024.
Legacy CRSP covers this endpoint; CIZ is not needed. The earlier history supports
the 1963-2024 comparison in Task 3. For five-year growth signals in 1962, change BOTH
start dates to 1956-01-01; that wider history requires a new data request.

This file prepares data and prints the selected formation filters. It does
not construct portfolios or execute the LSV tables. Tables I, II, IV, V, VI
and Figure 2 share the four raw inputs below. Figure 2's versioned NBER
chronology is stored under references/lsv1994/recessions.csv. After preparation,
use main_lsv_universes.py or main_lsv_subperiods.py to run the analyses.
To cover the requested 2025 endpoint, prepare a new CIZ cache and pass its
folder through --raw-data-dir; existing 2024 inputs remain unchanged.

Run from the project directory:
    uv run python data_prepare_lsv1994_task2.py
"""

from data_prepare_lsv1994 import data_prepare_lsv1994


if __name__ == "__main__":
    data_start_date = "1957-01-01"       # Warm-up history for formations from 1963.
    data_end_date = "2024-12-31"         # Fixed extended-sample endpoint.
    accounting_start_date = "1957-01-01"
    accounting_end_date = "2024-12-31"   # April t signals use t-1 accounts.
    run_sample = "nyse_amex"             # "nyse", "nyse_amex", or "nyse_amex_nasdaq".
    run_tables = (1, 2, 4, 5, 6)         # Table VI also requests Figure 2's market index.
    run_data_source = "legacy"
    refresh_data = False

    exchange_codes = {"nyse": (1,), "nyse_amex": (1, 2), "nyse_amex_nasdaq": (1, 2, 3)}
    if run_sample not in exchange_codes:
        raise ValueError("Select run_sample='nyse', 'nyse_amex', or 'nyse_amex_nasdaq'.")
    formation_exchange_codes = exchange_codes[run_sample]
    print(f"HW1 Q2 Task 2 | selected analysis sample: {run_sample}", flush=True)
    print(f"April formation filters: exchcd in {formation_exchange_codes}; shrcd in (10, 11).",
          flush=True)
    print("Preparing the shared broad raw inputs for this analysis sample...", flush=True)

    paths = data_prepare_lsv1994(
        start=data_start_date,
        end=data_end_date,
        comp_start=accounting_start_date,
        comp_end=accounting_end_date,
        universe="nyse_amex_nasdaq",     # Shared raw cache for all three analysis universes.
        tables=run_tables,
        data_source=run_data_source,
        refresh=refresh_data,
    )
    print(f"For the LSV analysis, select universe={run_sample!r} and "
          f"raw_data_dir={str(paths['crsp_monthly'].parent)!r}.", flush=True)
    print("Task 2 data preparation finished successfully.", flush=True)
