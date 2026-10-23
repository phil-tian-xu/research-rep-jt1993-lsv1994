"""HW1 Q2 Task 1: prepare the original LSV1994 sample with legacy CRSP.

Run this file for the original NYSE/AMEX replication: CRSP December 1962-April
1990 and Compustat 1962-1993. The latter endpoint provides an accounting outcome
buffer, without extending the original return sample. Earlier CRSP history and
dividends support Table V; market indexes support Figure 2's decline indicator.

For the extended NYSE-only and NYSE/AMEX/Nasdaq samples through December 2024,
run data_prepare_lsv1994_task2.py. Both entry points use legacy CRSP.
Signals must use prior-year accounts. Later accounting rows are outcomes only;
recent cohorts have incomplete five-year outcomes and must be handled explicitly
in the analysis, without requiring future availability at portfolio formation.
"""

import json
from pathlib import Path

import pandas as pd

from utils.lsv1994_data_download import download_lsv1994_data


def data_prepare_lsv1994(
    start: str = "1962-12-01",
    end: str = "1990-04-30",
    universe: str = "nyse_amex",
    refresh: bool = False,
    *,
    comp_start: str = "1962-01-01",
    comp_end: str = "1993-12-31",
    tables: tuple[int, ...] = (1, 2, 4, 5, 6),
    data_source: str = "legacy",
    username: str | None = None,
) -> dict[str, Path]:
    """Cache raw inputs and print coverage; do not construct portfolios.

    All requested tables need crsp_monthly, compustat_annual and ccm_links.
    Table VI also requests market_returns for the associated Figure 2.
    To match only the reference Table I download, set tables=(1,),
    start='1967-12-01' and comp_end='1988-12-31'. No daily data are needed.
    Use data_source="ciz" for an end date after December 2024. Compustat's
    endpoint is independent of the CRSP endpoint; extend both for modern signals.

    Files and SQL/coverage JSON sidecars are saved under data/raw/lsv1994/.
    Valid caches are reused unless refresh=True. Credentials use WRDS_USERNAME
    or username plus PostgreSQL's saved password, as in the momentum workflow.
    """
    if not tables or len(set(tables)) != len(tables) or set(tables) - {1, 2, 4, 5, 6}:
        raise ValueError("Select unique table numbers from (1, 2, 4, 5, 6).")
    datasets = ["crsp_monthly", "compustat_annual", "ccm_links"]
    if 6 in tables:
        datasets.append("market_returns")
    print(f"LSV1994 tables {tables} | {universe} | {data_source} | CRSP {start} to {end} | "
          f"Compustat {comp_start} to {comp_end}", flush=True)
    print("Raw CRSP includes all exchanges; apply sample eligibility at April formation.", flush=True)
    paths = download_lsv1994_data(
        start=start, end=end, comp_start=comp_start, comp_end=comp_end,
        universe=universe, datasets=tuple(datasets), refresh=refresh, username=username,
        data_source=data_source,
    )
    summary = []
    for name, path in paths.items():
        info = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        summary.append({
            "dataset": name, "rows": info["rows"],
            "first_date": info["first_date"], "last_date": info["last_date"],
        })
    print(pd.DataFrame(summary).to_string(index=False), flush=True)
    print("CCM coverage above describes link START dates; open link boundaries are retained.", flush=True)
    print(f"Raw folder for LSV analysis: {paths['crsp_monthly'].parent}", flush=True)
    print("LSV1994 data preparation finished successfully.", flush=True)
    return paths


if __name__ == "__main__":
    data_start_date = "1962-12-01"       # Historical market equity/past returns for Table V.
    data_end_date = "1990-04-30"         # Original holding-return endpoint.
    accounting_start_date = "1962-01-01"  # Sales history for the first 1968 formation.
    accounting_end_date = "1993-12-31"    # Original accounting outcome buffer only.
    run_sample = "nyse_amex"
    run_data_source = "legacy"
    run_tables = (1, 2, 4, 5, 6)  # Table VI includes the Figure 2 index input.
    refresh_data = False

    data_prepare_lsv1994(
        start=data_start_date,
        end=data_end_date,
        comp_start=accounting_start_date,
        comp_end=accounting_end_date,
        universe=run_sample,
        tables=run_tables,
        data_source=run_data_source,
        refresh=refresh_data,
    )
