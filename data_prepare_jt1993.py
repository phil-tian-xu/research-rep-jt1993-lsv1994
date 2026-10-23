"""Download the raw inputs for selected JT1993 tables; edit settings below.

Run this before main.py. The original reference uses raw data from
July 1962 through December 1989 for reported returns starting in January 1965.
The earlier history supports ranking and portfolios formed before the sample.
The settings below prepare Table I only, using legacy CRSP through December 2024.
"""

import json
from datetime import date
from pathlib import Path

import pandas as pd

from utils.jt1993_data_download import download_jt1993_data, get_latest_jt1993_end_date, get_wrds_connection


def data_prepare_jt1993(
    start: str,
    end: str,
    universe: str = "nyse_amex",
    refresh: bool = False,
    *,
    tables: tuple[int, ...] = (1,),
    data_source: str = "legacy",
) -> dict[str, Path]:
    """Download/reuse the inputs for Tables I, II and/or VII and print coverage.

    start/end are inclusive raw-data dates, at month start/end respectively.
    end="latest" checks WRDS for the latest complete month shared by the selected
    inputs, excluding the current calendar month. It uses one connection for
    discovery and download, and concrete resolved dates in cache paths/metadata.
    An explicit ISO end date preserves a fixed replication window.
    universe selects "nyse", "nyse_amex", or "nyse_amex_nasdaq", retaining
    US common shares (share codes 10/11 or their CIZ equivalents).
    data_source="legacy" preserves the reference download; "ciz" supports
    recent data. Legacy ends in December 2024. CIZ has delisting returns in
    the source; we do not add another delisting adjustment. Shares are thousands.

    monthly_returns compounds crsp.dsf/dsf_v2 returns on WRDS for Panel A ranking and
    both panels' holding returns. daily_returns is needed for Panel B's five
    trading-day gap. We do not add separate delisting returns to either source.
    tables=(2,) additionally selects monthly_characteristics (prices/shares)
    and market_returns (CRSP vwretd). Table VII needs monthly_returns only.
    tables=(1, 2, 7) selects all four files. The default tables=(1,) preserves
    the original two-file preparation.

    Files and SQL/coverage JSON sidecars go under utils.path_info.data_path at
    raw/jt1993/<universe>_<start YYYYMMDD>_<end YYYYMMDD>/. Valid caches are reused
    unless refresh=True. CIZ folders add "_ciz" after the universe name.
    Set WRDS_USERNAME and save the WRDS password once in
    PostgreSQL's password file. "latest" needs a connection to discover coverage;
    explicit dated requests only connect if a requested cache is missing.
    """
    if not tables or len(set(tables)) != len(tables) or set(tables) - {1, 2, 7}:
        raise ValueError("Select unique table numbers from (1, 2, 7).")
    datasets = ["monthly_returns"]
    if 1 in tables:
        datasets.append("daily_returns")
    if 2 in tables:
        datasets.extend(["monthly_characteristics", "market_returns"])
    print(f"JT1993 tables {tables} inputs | {universe} | {data_source} | raw dates {start} to {end}", flush=True)
    connection = None
    try:
        if end == "latest":
            if universe not in ("nyse", "nyse_amex", "nyse_amex_nasdaq"):
                raise ValueError("Select universe='nyse', 'nyse_amex', or 'nyse_amex_nasdaq'.")
            if data_source not in ("legacy", "ciz"):
                raise ValueError("Select data_source='legacy' or 'ciz'.")
            try:
                start_date = date.fromisoformat(start)
            except (TypeError, ValueError) as error:
                raise ValueError("start must be an ISO date string (YYYY-MM-DD).") from error
            if start_date.day != 1:
                raise ValueError("Monthly datasets require start at month start.")
            print("Connecting to WRDS to check coverage; its 'Done' message marks setup only.", flush=True)
            connection = get_wrds_connection()
            end = get_latest_jt1993_end_date(connection, data_source=data_source, datasets=tuple(datasets))
            if date.fromisoformat(end) < start_date:
                raise ValueError(f"Latest available end date {end} precedes requested start {start}.")
        paths = download_jt1993_data(
            start=start,
            end=end,
            universe=universe,
            datasets=tuple(datasets),
            refresh=refresh,
            data_source=data_source,
            connection=connection,
        )
    finally:
        if connection is not None:
            print("Closing WRDS connection...", flush=True)
            connection.close()

    summary = []
    for name, path in paths.items():
        info = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        summary.append({
            "dataset": name, "rows": info["rows"],
            "start": info["first_date"], "end": info["last_date"],
        })
    print(pd.DataFrame(summary).to_string(index=False), flush=True)
    print(f"Raw folder for main.py: {paths['monthly_returns'].parent}", flush=True)
    print("JT1993 data preparation finished successfully.", flush=True)
    return paths


if __name__ == "__main__":
    data_start_date = "1960-01-01"  # Warm-up history for reported results starting in 1962.
    data_end_date = "2024-12-31"  # Fixed extended-sample endpoint; no coverage lookup.
    run_sample = "nyse_amex"
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
