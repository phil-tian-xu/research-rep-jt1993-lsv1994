"""Cached WRDS downloads for the Jegadeesh-Titman (1993) replication.

The stock universe is NYSE, NYSE/AMEX, or NYSE/AMEX/Nasdaq common shares,
classified using the information valid on each observation date.
Pass the requested raw-data start and end dates to download_jt1993_data().
For the original replication, use July 1962-December 1989 and select reported
strategy returns from January 1965 onward in the main analysis script.

Monthly returns are compounded from CRSP daily returns, as in the reference
notebook. They are a server-derived extract, not native CRSP monthly returns.
Delisting returns are not added. Characteristic prices are in dollars and
shares outstanding in thousands; returns are decimals. All monthly dates are
calendar month ends. Legacy data preserve the original replication. CIZ data
support recent samples, using WRDS's common-stock/exchange mapping and the same
output columns. CIZ returns can include delisting returns in the source; no
additional delisting adjustment is applied. Shares outstanding stay in thousands.

Downloads return paths, so loading the entire daily panel is optional. Each
Parquet file has a JSON sidecar recording its SQL, filters, coverage, and vintage.
The data_prepare_jt1993.py script selects only monthly_returns and daily_returns
for Table I; main.py reads these local files without downloading.
"""

import hashlib
import json
import os
from datetime import date, datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from utils.path_info import data_path, root_path


DEFAULT_CACHE_DIR = Path(data_path) / "raw" / "jt1993"
_UNIVERSES = {"nyse": (1,), "nyse_amex": (1, 2), "nyse_amex_nasdaq": (1, 2, 3)}
_PRIMARY_EXCHANGES = {1: "N", 2: "A", 3: "Q"}

_MONTH_END = "(date_trunc('month', a.date) + interval '1 month - 1 day')::date"
_NAME_JOIN = """
inner join crsp.msenames as b
    on a.permno = b.permno
   and b.namedt <= a.date
   and a.date <= coalesce(b.nameendt, date '9999-12-31')
"""
_STOCK_FILTER = """
where a.date between %(start_date)s::date and %(end_date)s::date
  and b.exchcd in ({exchange_codes})
  and b.shrcd in (10, 11)
"""

# Each entry contains SQL, its observation date, and its unique key.
_DATASETS = {
    "daily_returns": (
        f"""
select a.permno, a.date, a.ret, b.exchcd, b.shrcd
from crsp.dsf as a
{_NAME_JOIN}
{_STOCK_FILTER}
order by a.permno, a.date
""",
        "date", ["permno", "date"],
    ),
    "monthly_returns": (
        f"""
with daily as (
    select a.permno, {_MONTH_END} as month, a.ret
    from crsp.dsf as a
    {_NAME_JOIN}
    {_STOCK_FILTER}
)
select permno, month, count(*) as n_daily_rows,
    sum(case when ret is not null and ret >= -1 then 1 else 0 end) as n_valid_ret,
    case
        when sum(case when ret is not null and ret >= -1 then 1 else 0 end) = 0
            then null
        when sum(case when ret = -1 then 1 else 0 end) > 0 then -1
        else exp(sum(case when ret > -1 then ln(1 + ret) else null end)) - 1
    end as ret
from daily
group by permno, month
order by permno, month
""",
        "month", ["permno", "month"],
    ),
    "monthly_characteristics": (
        f"""
select a.permno, {_MONTH_END} as month,
       a.prc, a.shrout, b.exchcd, b.shrcd
from crsp.msf as a
{_NAME_JOIN}
{_STOCK_FILTER}
order by a.permno, month
""",
        "month", ["permno", "month"],
    ),
    "market_returns": (
        f"""
select {_MONTH_END} as month, a.vwretd, a.ewretd
from crsp.msi as a
where a.date between %(start_date)s::date and %(end_date)s::date
order by month
""",
        "month", ["month"],
    ),
}


# Mapping: https://wrds-www.wharton.upenn.edu/pages/wrds-research/macros/crsp-ciztosiz-macro/
# CIZ shares are in thousands: CRSP's Flat File Format 2.0 guide, ShrOut definition.
_CIZ_FILTER = """
  and a.primaryexch in ({primary_exchanges})
  and a.conditionaltype = 'RW'
  and a.tradingstatusflg = 'A'
  and a.sharetype = 'NS'
  and a.securitytype = 'EQTY'
  and a.securitysubtype = 'COM'
  and a.usincflg = 'Y'
  and a.issuertype in ('ACOR', 'CORP')
"""
_CIZ_EXCHANGE_CODE = "case a.primaryexch when 'N' then 1 when 'A' then 2 when 'Q' then 3 end"
_CIZ_SHARE_CODE = "case a.issuertype when 'ACOR' then 10 else 11 end"
_CIZ_MONTH_END = "(date_trunc('month', a.mthcaldt) + interval '1 month - 1 day')::date"
_CIZ_DATASETS = {
    "daily_returns": (
        f"""
select a.permno, a.dlycaldt as date, a.dlyret as ret,
       {_CIZ_EXCHANGE_CODE} as exchcd, {_CIZ_SHARE_CODE} as shrcd
from crsp.dsf_v2 as a
where a.dlycaldt between %(start_date)s::date and %(end_date)s::date
{_CIZ_FILTER}
order by a.permno, a.dlycaldt
""",
        "date", ["permno", "date"],
    ),
    "monthly_returns": (
        f"""
with daily as (
    select a.permno,
        (date_trunc('month', a.dlycaldt) + interval '1 month - 1 day')::date as month,
        a.dlyret as ret
    from crsp.dsf_v2 as a
    where a.dlycaldt between %(start_date)s::date and %(end_date)s::date
    {_CIZ_FILTER}
)
select permno, month, count(*) as n_daily_rows,
    sum(case when ret is not null and ret >= -1 then 1 else 0 end) as n_valid_ret,
    case
        when sum(case when ret is not null and ret >= -1 then 1 else 0 end) = 0
            then null
        when sum(case when ret = -1 then 1 else 0 end) > 0 then -1
        else exp(sum(case when ret > -1 then ln(1 + ret) else null end)) - 1
    end as ret
from daily
group by permno, month
order by permno, month
""",
        "month", ["permno", "month"],
    ),
    "monthly_characteristics": (
        f"""
select a.permno, {_CIZ_MONTH_END} as month, a.mthprc as prc, a.shrout,
       {_CIZ_EXCHANGE_CODE} as exchcd, {_CIZ_SHARE_CODE} as shrcd
from crsp.msf_v2 as a
where a.mthcaldt between %(start_date)s::date and %(end_date)s::date
{_CIZ_FILTER}
order by a.permno, month
""",
        "month", ["permno", "month"],
    ),
    "market_returns": (
        f"""
select {_CIZ_MONTH_END} as month,
       max(a.mthtotret) filter (where a.indno = 1000200) as vwretd,
       max(a.mthtotret) filter (where a.indno = 1000201) as ewretd
from crsp.indmthseriesdata as a
where a.mthcaldt between %(start_date)s::date and %(end_date)s::date
  and a.indno in (1000200, 1000201)
group by month
order by month
""",
        "month", ["month"],
    ),
}


def get_wrds_connection(username: str | None = None):
    """Connect with an explicit username and PostgreSQL's saved password.

    Supply a username or set WRDS_USERNAME in the kernel's environment. On
    Windows, libpq normally reads %APPDATA%/postgresql/pgpass.conf. WRDS may
    prompt if the saved entry is missing/invalid, and may require Duo approval.
    Call create_pgpass_file() only for one-time credential setup or updates.
    """
    username = (username or os.environ.get("WRDS_USERNAME", "")).strip()
    if not username:
        raise ValueError("Pass username=... or set the WRDS_USERNAME environment variable.")

    import wrds

    return wrds.Connection(wrds_username=username)


def get_latest_jt1993_end_date(connection, *, data_source="ciz", datasets=None, as_of=None) -> str:
    """Find the latest published, completed month shared by the requested inputs.

    Monthly stock publication dates anchor stock extracts. The anchor must not
    exceed daily coverage when returns are requested, or market-index coverage
    when market returns are requested. This avoids treating a partial daily
    month as complete, including when the last trading day is before month end.
    The current calendar month is excluded using Asia/Shanghai's date. as_of
    optionally fixes that date for a reproducible check. No connection is opened
    or closed here; the caller reuses its connection for the subsequent download.
    """
    if data_source not in ("legacy", "ciz"):
        raise ValueError("Select data_source='legacy' or 'ciz'.")
    names = tuple(_DATASETS) if datasets is None else tuple(datasets)
    if not names or len(set(names)) != len(names) or set(names) - _DATASETS.keys():
        raise ValueError(f"Select unique dataset names from {tuple(_DATASETS)}.")
    today = pd.Timestamp(as_of).date() if as_of is not None else pd.Timestamp.now(tz="Asia/Shanghai").date()
    cutoff = (pd.Timestamp(today).replace(day=1) - pd.Timedelta(days=1)).date()
    if data_source == "legacy":
        cutoff = min(cutoff, date(2024, 12, 31))

    ciz = data_source == "ciz"
    monthly_table, monthly_date = ("crsp.msf_v2", "mthcaldt") if ciz else ("crsp.msf", "date")
    daily_table, daily_date = ("crsp.dsf_v2", "dlycaldt") if ciz else ("crsp.dsf", "date")
    index_table, index_date = ("crsp.indmthseriesdata", "mthcaldt") if ciz else ("crsp.msi", "date")
    stock_inputs = set(names) - {"market_returns"}
    anchor_table, anchor_date = (monthly_table, monthly_date) if stock_inputs else (index_table, index_date)
    conditions = [f"a.{anchor_date} <= %(calendar_end)s::date"]
    if {"monthly_returns", "daily_returns"} & set(names):
        conditions.append(
            f"a.{anchor_date} <= (select max({daily_date}) from {daily_table} "
            f"where {daily_date} <= %(calendar_end)s::date)"
        )
    if "market_returns" in names:
        if ciz:
            if not stock_inputs:
                conditions.extend(["a.indno = 1000200", "a.mthtotret is not null"])
            for index_id in (1000200, 1000201):
                conditions.append(
                    f"a.{anchor_date} <= (select max({index_date}) from {index_table} "
                    f"where indno = {index_id} and mthtotret is not null "
                    f"and {index_date} <= %(calendar_end)s::date)"
                )
        else:
            conditions.append(
                f"a.{anchor_date} <= (select max(date) from crsp.msi "
                "where vwretd is not null and ewretd is not null "
                "and date <= %(calendar_end)s::date)"
            )
    sql = f"select max(a.{anchor_date}) as last_date from {anchor_table} as a\nwhere " + "\n  and ".join(conditions)
    print(f"Checking WRDS for the latest complete {data_source} month (calendar cutoff {cutoff})...", flush=True)
    available = connection.raw_sql(sql, params={"calendar_end": cutoff.isoformat()}, date_cols=["last_date"])
    if len(available) != 1 or pd.isna(available["last_date"].iloc[0]):
        raise ValueError("No common completed month is available for the requested WRDS inputs.")
    resolved = pd.Timestamp(available["last_date"].iloc[0]) + pd.offsets.MonthEnd(0)
    if resolved.date() > cutoff:
        raise ValueError("WRDS returned an availability date beyond the completed-month cutoff.")
    print(f"Resolved latest available end date: {resolved:%Y-%m-%d}", flush=True)
    return resolved.date().isoformat()


def _write_query(connection, sql, params, date_column, key_columns, path, chunk_size):
    """Stream an ordered query to a complete, validated Parquet file."""
    started = perf_counter()
    writer = None
    row_count = 0
    previous_key = None
    months_seen = set()
    first_date = last_date = None

    try:
        print(
            f"[{path.stem}] Querying WRDS; waiting for the first batch "
            f"(up to {chunk_size:,} rows)...",
            flush=True,
        )
        chunks = connection.raw_sql(
            sql, params=params.copy(), date_cols=[date_column],
            chunksize=chunk_size, return_iter=True,
        )
        for batch, frame in enumerate(chunks, start=1):
            if frame.empty:
                continue
            if row_count == 0:
                print(
                    f"[{path.stem}] First batch received after {perf_counter() - started:.1f}s; "
                    "validating and writing...",
                    flush=True,
                )
            frame = frame.copy()
            frame[date_column] = pd.to_datetime(frame[date_column], errors="raise")
            for column in frame.columns:
                if column in {"permno", "exchcd", "shrcd", "n_daily_rows", "n_valid_ret"}:
                    frame[column] = pd.to_numeric(frame[column], errors="raise").astype("Int64")
                elif column != date_column:
                    frame[column] = pd.to_numeric(frame[column], errors="raise").astype("float64")

            if frame[key_columns].isna().any().any():
                raise ValueError(f"Missing observation key in {path.name}.")
            keys = pd.MultiIndex.from_frame(frame[key_columns])
            if not keys.is_unique or not keys.is_monotonic_increasing:
                raise ValueError(f"Duplicate or unordered observation keys in {path.name}.")
            if previous_key is not None and keys[0] <= previous_key:
                raise ValueError(f"Duplicate or unordered keys across chunks in {path.name}.")
            previous_key = keys[-1]

            dates = frame[date_column]
            if not dates.between(params["start_date"], params["end_date"]).all():
                raise ValueError(f"Dates outside the requested sample in {path.name}.")
            months_seen.update(dates.dt.to_period("M").unique())
            first_date = dates.min() if first_date is None else min(first_date, dates.min())
            last_date = dates.max() if last_date is None else max(last_date, dates.max())

            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(path, table.schema, compression="zstd")
            writer.write_table(table)
            row_count += len(frame)
            print(
                f"[{path.stem}] Batch {batch}: {len(frame):,} rows written; "
                f"{row_count:,} total ({perf_counter() - started:.1f}s elapsed).",
                flush=True,
            )
    finally:
        if writer is not None:
            writer.close()

    if not row_count:
        raise ValueError(f"WRDS returned no observations for {path.name}.")
    expected = pd.period_range(params["start_date"], params["end_date"], freq="M")
    missing = sorted(set(expected) - months_seen)
    if missing:
        raise ValueError(f"Missing calendar months in {path.name}: {list(map(str, missing))}")

    return {
        "rows": row_count,
        "columns": table.column_names,
        "first_date": first_date.date().isoformat(),
        "last_date": last_date.date().isoformat(),
    }


def download_jt1993_data(
    start: str,
    end: str,
    cache_dir: str | Path | None = None,
    *,
    universe: str = "nyse_amex",
    data_source: str = "legacy",
    datasets: tuple[str, ...] | None = None,
    username: str | None = None,
    connection=None,
    refresh: bool = False,
    chunk_size: int = 250_000,
) -> dict[str, Path]:
    """Download JT1993 inputs for inclusive start/end dates, or reuse cached files.

    start, end
        Raw-data dates as ISO strings (YYYY-MM-DD), for example "1962-07-01"
        and "1989-12-31" for the original replication. Monthly datasets require
        start to be the first day and end the last day of a calendar month so
        a partial daily window is not reported as a complete monthly return.
        A daily-only request may use arbitrary dates. Portfolio warm-up and
        reported return dates remain choices in the main analysis script.
    universe
        Exchange universe applied using the historical classification valid on
        each stock observation date: "nyse" (1), "nyse_amex" (1/2), or
        "nyse_amex_nasdaq" (1/2/3). Keep common-share codes 10/11. CIZ uses
        the equivalent security flags, active regular-way trading, and
        primary exchanges N/A/Q, mapped to the same numeric output codes.
    data_source
        "legacy" (default) preserves reference data and caches; available
        through December 2024. "ciz" uses dsf_v2/msf_v2 for recent samples.
        Both derive monthly returns by compounding filtered daily returns.
        CIZ source returns can include delisting returns; no extra adjustment
        is added. CIZ index data use CRSP's broad VW/EW indexes 1000200/1000201
        from indmthseriesdata. Access depends on your WRDS subscription.
    cache_dir
        Cache directory if supplied; relative paths are based on root_path
        from utils.path_info, independent of the working directory. Otherwise use
        data/raw/jt1993/<universe>_<YYYYMMDD start>_<YYYYMMDD end>/ for legacy;
        CIZ adds "_ciz" after the universe to keep the two sources separate.

    Dataset names: daily_returns, monthly_returns, monthly_characteristics,
    market_returns. None selects all four. Monthly returns come from dsf;
    msf/msf_v2 are used only for prices/shares. Market indices are CRSP's published
    broad-market series, not an index rebuilt from the filtered stock universe.

    A connection is created only if a requested file is missing or refresh=True.
    An internally created connection is closed; a caller-supplied one is not.
    Cache metadata must match the query specification. Existing legacy caches
    are not automatically imported because they lack the new provenance/schema.
    Downloads use temporary files, preserving any previous cache on failure.
    Console messages report cache reuse, query waits, written batches, and completion.

    Example::

        paths = download_jt1993_data(
            start="1962-07-01", end="1989-12-31", universe="nyse_amex",
        )
    """
    names = tuple(_DATASETS) if datasets is None else tuple(datasets)
    if not names or len(set(names)) != len(names) or set(names) - _DATASETS.keys():
        raise ValueError(f"Select unique dataset names from {tuple(_DATASETS)}.")
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    if universe not in _UNIVERSES:
        raise ValueError(f"Select universe from {tuple(_UNIVERSES)}.")
    if data_source not in ("legacy", "ciz"):
        raise ValueError("Select data_source='legacy' or 'ciz'.")
    try:
        start_date = date.fromisoformat(start)
        end_date = date.fromisoformat(end)
    except (TypeError, ValueError) as error:
        raise ValueError("start and end must be ISO date strings (YYYY-MM-DD).") from error
    if start_date > end_date:
        raise ValueError("start must be on or before end.")
    if data_source == "legacy" and end_date > date(2024, 12, 31):
        raise ValueError("Legacy CRSP ends at 2024-12-31. Use data_source='ciz' for later dates.")
    if any(_DATASETS[name][1] == "month" for name in names):
        if start_date.day != 1 or not pd.Timestamp(end_date).is_month_end:
            raise ValueError("Monthly datasets require start at month start and end at month end.")
    params = {"start_date": start_date.isoformat(), "end_date": end_date.isoformat()}

    sample_name = universe if data_source == "legacy" else f"{universe}_ciz"
    cache_dir = (
        DEFAULT_CACHE_DIR / f"{sample_name}_{start_date:%Y%m%d}_{end_date:%Y%m%d}"
        if cache_dir is None else Path(cache_dir)
    )
    if not cache_dir.is_absolute():
        cache_dir = Path(root_path) / cache_dir
    paths = {name: cache_dir / f"{name}.parquet" for name in names}
    pending = []
    dataset_specs = _DATASETS if data_source == "legacy" else _CIZ_DATASETS
    exchange_codes = _UNIVERSES[universe]
    for name, path in paths.items():
        sql, date_column, key_columns = dataset_specs[name]
        # Only fixed, internal exchange codes are inserted; dates remain SQL parameters.
        sql = sql.format(
            exchange_codes=", ".join(map(str, exchange_codes)),
            primary_exchanges=", ".join(f"'{_PRIMARY_EXCHANGES[code]}'" for code in exchange_codes),
        )
        specification = {
            "cache_version": 2, "sql": sql, "params": params, "universe": universe,
            "date_column": date_column, "key_columns": key_columns,
        }
        if data_source == "ciz":
            specification["data_source"] = data_source
        signature = hashlib.sha256(
            json.dumps(specification, sort_keys=True).encode("utf-8")
        ).hexdigest()
        if path.exists() and not refresh:
            sidecar = path.with_suffix(".json")
            if not sidecar.exists():
                raise ValueError(f"Missing metadata for {path}. Use a new cache directory or refresh=True.")
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
            if metadata.get("signature") != signature:
                raise ValueError(f"Cache specification differs for {path}. Use a new directory or refresh=True.")
            parquet_metadata = pq.read_metadata(path)
            if parquet_metadata.num_rows != metadata.get("rows"):
                raise ValueError(f"Cache row count differs from metadata for {path}.")
            print(f"[{name}] Using cached file: {parquet_metadata.num_rows:,} rows.", flush=True)
        else:
            pending.append((name, path, specification, signature))

    if not pending:
        print("All requested datasets are cached; nothing to download.", flush=True)
        return paths

    print("Pending downloads: " + ", ".join(item[0] for item in pending), flush=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    owns_connection = connection is None
    if owns_connection:
        print("Connecting to WRDS; its 'Done' message marks setup only, not download completion.", flush=True)
        connection = get_wrds_connection(username)
    print("WRDS connection ready. Starting downloads.", flush=True)
    try:
        for number, (name, path, specification, signature) in enumerate(pending, start=1):
            started = perf_counter()
            print(f"[{number}/{len(pending)}] Preparing {name}...", flush=True)
            with TemporaryDirectory(prefix=f".{name}_", dir=cache_dir) as temporary:
                temporary_path = Path(temporary) / path.name
                details = _write_query(
                    connection, specification["sql"], params, specification["date_column"],
                    specification["key_columns"], temporary_path, chunk_size,
                )
                metadata = {
                    "dataset": name, "signature": signature, **specification, **details,
                    "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
                    "exchange_codes": list(exchange_codes) if name != "market_returns" else None,
                    "share_codes": [10, 11] if name != "market_returns" else None,
                    "return_units": "decimal", "delisting_return_adjustment": False,
                    "data_source": data_source,
                    "delisting_returns_in_source": data_source == "ciz",
                    "shares_outstanding_units": "thousands",
                }
                if data_source == "ciz" and name == "market_returns":
                    metadata["market_index_ids"] = {"vwretd": 1000200, "ewretd": 1000201}
                temporary_metadata = temporary_path.with_suffix(".json")
                temporary_metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
                temporary_path.replace(path)
                temporary_metadata.replace(path.with_suffix(".json"))
            print(
                f"[{name}] Complete: {details['rows']:,} validated rows, "
                f"{details['first_date']} to {details['last_date']} "
                f"({perf_counter() - started:.1f}s elapsed). Saved to {path}",
                flush=True,
            )
    except KeyboardInterrupt:
        print("\nDownload interrupted. Previously completed datasets remain cached.", flush=True)
        raise
    except Exception:
        print("Download failed. Previously completed datasets remain cached; see the traceback for details.", flush=True)
        raise
    finally:
        if owns_connection:
            print("Closing WRDS connection...", flush=True)
            connection.close()
    print("All requested JT1993 datasets are ready.", flush=True)
    return paths
