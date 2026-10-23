"""Cached WRDS extracts for Lakonishok, Shleifer and Vishny (1994).

The queries follow references/lsv1994_table1_wrds_replication.ipynb.
CRSP observations retain their source dates and are NOT filtered by exchange
or share code: apply NYSE/AMEX common-share eligibility at April formation,
then follow the original constituents even if their exchange changes.
No returns are filled, compounded, or adjusted for delistings here. Legacy
inputs reproduce the reference; CIZ monthly returns support 2025 onward and
already incorporate delisting returns in the source.

Compared with the Table I reference, Compustat also includes common dividends
(dvc) for Table V, and the optional CRSP index supplies Figure 2's market
decline indicator. Portfolio construction belongs in the replication modules.
"""

import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from utils.jt1993_data_download import get_wrds_connection
from utils.path_info import data_path, root_path


DEFAULT_CACHE_DIR = Path(data_path) / "raw" / "lsv1994"
_UNIVERSES = {"nyse": (1,), "nyse_amex": (1, 2), "nyse_amex_nasdaq": (1, 2, 3)}

# SQL, all date columns, and a source-level observation key. Preserve Compustat
# population variants; selection of one fiscal/calendar-year record is analysis.
_DATASETS = {
    "crsp_monthly": (
        """
select a.permno, a.date, a.ret, a.prc, a.shrout, b.shrcd, b.exchcd
from crsp.msf as a
left join crsp.msenames as b
    on a.permno = b.permno
   and b.namedt <= a.date
   and a.date <= coalesce(b.nameendt, date '9999-12-31')
where a.date between %(start_date)s::date and %(end_date)s::date
order by a.permno, a.date
""",
        ("date",), ("permno", "date"),
    ),
    "compustat_annual": (
        """
select gvkey, datadate, fyear, fyr, popsrc, curcd,
       ceq, seq, pstk, pstkrv, pstkl, txditc, ib, dp, sale, dvc
from comp.funda
where indfmt = 'INDL'
  and datafmt = 'STD'
  and consol = 'C'
  and datadate between %(start_date)s::date and %(end_date)s::date
order by gvkey, datadate, popsrc
""",
        ("datadate",), ("gvkey", "datadate", "popsrc"),
    ),
    "ccm_links": (
        """
select gvkey, lpermno as permno, linkdt, linkenddt, linktype, linkprim, liid
from crsp.ccmxpf_linktable
where lpermno is not null
  and linktype in ('LC', 'LU', 'LS')
order by gvkey, lpermno, linkdt nulls first, linkenddt nulls first,
         linktype, linkprim nulls first, liid nulls first
""",
        ("linkdt", "linkenddt"),
        ("gvkey", "permno", "linkdt", "linkenddt", "linktype", "linkprim", "liid"),
    ),
    "market_returns": (
        """
select date, vwretd, ewretd
from crsp.msi
where date between %(start_date)s::date and %(end_date)s::date
order by date
""",
        ("date",), ("date",),
    ),
}
# CIZ mapping follows WRDS's CRSP CIZtoSIZ macro. Only the common-share and
# exchange codes needed for formation eligibility are mapped; other codes are
# null. Keep source flags and ALL stock rows so holding returns remain available
# after a constituent changes exchange, share type, or trading status.
# https://wrds-www.wharton.upenn.edu/pages/wrds-research/macros/crsp-ciztosiz-macro/
_CIZ_DATASETS = {
    "crsp_monthly": (
        """
select a.permno, a.mthcaldt as date, a.mthret as ret,
       a.mthprc as prc, a.shrout,
       case when a.sharetype = 'NS' and a.securitytype = 'EQTY'
                  and a.securitysubtype = 'COM' and a.usincflg = 'Y'
            then case a.issuertype when 'ACOR' then 10 when 'CORP' then 11 end
       end as shrcd,
       case when a.conditionaltype = 'RW' and a.tradingstatusflg = 'A'
            then case a.primaryexch when 'N' then 1 when 'A' then 2 when 'Q' then 3 end
       end as exchcd,
       a.primaryexch, a.conditionaltype, a.tradingstatusflg,
       a.sharetype, a.securitytype, a.securitysubtype, a.usincflg, a.issuertype
from crsp.msf_v2 as a
where a.mthcaldt between %(start_date)s::date and %(end_date)s::date
order by a.permno, a.mthcaldt
""",
        ("date",), ("permno", "date"),
    ),
    "market_returns": (
        """
select a.mthcaldt as date,
       max(a.mthtotret) filter (where a.indno = 1000200) as vwretd,
       max(a.mthtotret) filter (where a.indno = 1000201) as ewretd
from crsp.indmthseriesdata as a
where a.mthcaldt between %(start_date)s::date and %(end_date)s::date
  and a.indno in (1000200, 1000201)
group by a.mthcaldt
order by a.mthcaldt
""",
        ("date",), ("date",),
    ),
}
_STRING_COLUMNS = {
    "gvkey", "popsrc", "curcd", "linktype", "linkprim", "liid",
    "primaryexch", "conditionaltype", "tradingstatusflg", "sharetype",
    "securitytype", "securitysubtype", "usincflg", "issuertype",
}
_INTEGER_COLUMNS = {"permno", "fyear", "fyr", "shrcd", "exchcd"}


def _write_query(connection, name, specification, path, chunk_size):
    """Stream a raw extract, checking keys, schema, and calendar coverage."""
    date_columns = specification["date_columns"]
    key_columns = specification["key_columns"]
    params = specification["params"]
    writer = None
    row_count = 0
    previous_key = None
    periods_seen = set()
    first_date = last_date = None
    schema = None
    started = perf_counter()
    print(f"[{name}] Querying WRDS; waiting for the first batch...", flush=True)
    try:
        chunks = connection.raw_sql(
            specification["sql"], params=params.copy(), date_cols=list(date_columns),
            chunksize=chunk_size, return_iter=True,
        )
        for batch, frame in enumerate(chunks, start=1):
            if frame.empty:
                continue
            frame = frame.copy()
            for column in frame.columns:
                if column in date_columns:
                    frame[column] = pd.to_datetime(frame[column], errors="raise")
                elif column in _STRING_COLUMNS:
                    frame[column] = frame[column].astype("string")
                else:
                    dtype = "Int64" if column in _INTEGER_COLUMNS else "float64"
                    frame[column] = pd.to_numeric(frame[column], errors="raise").astype(dtype)

            required = ("gvkey", "permno", "linktype") if name == "ccm_links" else key_columns
            if frame[list(required)].isna().any().any():
                raise ValueError(f"Missing observation key in {path.name}.")
            if name == "market_returns" and frame[["vwretd", "ewretd"]].isna().any().any():
                raise ValueError("Missing value-weighted or equal-weighted market index return.")
            # Open CCM boundaries remain null in the saved raw data. Convert
            # dates only in the validation keys to match SQL NULLS FIRST order.
            key_frame = frame[list(key_columns)].copy()
            if name == "ccm_links":
                for column in date_columns:
                    key_frame[column] = key_frame[column].dt.strftime("%Y-%m-%d").fillna("")
                for column in ("linkprim", "liid"):
                    key_frame[column] = key_frame[column].fillna("")
                bounded = frame["linkdt"].notna() & frame["linkenddt"].notna()
                if (frame.loc[bounded, "linkenddt"] < frame.loc[bounded, "linkdt"]).any():
                    raise ValueError("CCM link end precedes link start.")
            keys = pd.MultiIndex.from_frame(key_frame)
            if not keys.is_unique or not keys.is_monotonic_increasing:
                raise ValueError(f"Duplicate or unordered observation keys in {path.name}.")
            if previous_key is not None and keys[0] <= previous_key:
                raise ValueError(f"Duplicate or unordered keys across chunks in {path.name}.")
            previous_key = keys[-1]

            dates = frame[date_columns[0]].dropna()
            if params:
                if not dates.between(params["start_date"], params["end_date"]).all():
                    raise ValueError(f"Dates outside the requested range in {path.name}.")
                frequency = "Y" if name == "compustat_annual" else "M"
                periods_seen.update(dates.dt.to_period(frequency).unique())
            if not dates.empty:
                first_date = dates.min() if first_date is None else min(first_date, dates.min())
                last_date = dates.max() if last_date is None else max(last_date, dates.max())

            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                schema = table.schema
                writer = pq.ParquetWriter(path, schema, compression="zstd")
            writer.write_table(table)
            row_count += len(frame)
            print(f"[{name}] Batch {batch}: {row_count:,} rows written "
                  f"({perf_counter() - started:.1f}s elapsed).", flush=True)
    finally:
        if writer is not None:
            writer.close()
    if not row_count:
        raise ValueError(f"WRDS returned no observations for {path.name}.")
    if params:
        frequency = "Y" if name == "compustat_annual" else "M"
        expected = pd.period_range(params["start_date"], params["end_date"], freq=frequency)
        missing = sorted(set(expected) - periods_seen)
        if missing:
            raise ValueError(f"Missing calendar periods in {path.name}: {list(map(str, missing))}")
    return {
        "rows": row_count, "columns": schema.names,
        "first_date": first_date.date().isoformat() if first_date is not None else None,
        "last_date": last_date.date().isoformat() if last_date is not None else None,
    }


def download_lsv1994_data(
    start: str,
    end: str,
    comp_start: str,
    comp_end: str,
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
    """Download/reuse raw monthly CRSP, annual Compustat, CCM, and index data.

    start/end are inclusive CRSP dates, requiring complete calendar months.
    comp_start/comp_end are independent, inclusive Compustat datadate limits.
    To reproduce only the supplied Table I notebook, use CRSP 1967-12-01 to
    1990-04-30 and Compustat 1962-01-01 to 1988-12-31. The preparation script
    requests wider histories for Table V; additional rows do not change the
    replication's formation sample or outcome cutoff.

    universe records the intended APRIL formation restriction, not an SQL
    restriction on holding returns. Raw strings, missing values, negative
    accounting values, all LC/LU/LS links, and open link dates are preserved.
    In particular, do not require positive earnings/cash flow at download time:
    Table IV also needs negative observations. Compustat population/currency
    variants are retained for explicit selection in the analysis.

    data_source="legacy" preserves the original files through December 2024.
    data_source="ciz" uses native CIZ monthly stock returns, prices, and indexes
    for the entire requested history. CIZ folders add "_ciz" after the universe
    so legacy and CIZ returns are not mixed. CIZ source returns incorporate
    delisting returns; no extra adjustment is applied. Mapped shrcd/exchcd only
    identify eligible common shares and active regular-way NYSE/AMEX/Nasdaq
    listings. Source classification flags are also retained for diagnostics.

    Each Parquet has a JSON sidecar with SQL, dates, units, and provenance.
    A relative cache_dir is anchored to the project, independent of cwd.
    Valid caches require no WRDS login. Failed queries preserve older files;
    only connections opened here are closed here. Set WRDS_USERNAME and use
    the PostgreSQL password file, as for the momentum downloader.
    """
    names = tuple(_DATASETS) if datasets is None else tuple(datasets)
    if not names or len(set(names)) != len(names) or set(names) - _DATASETS.keys():
        raise ValueError(f"Select unique dataset names from {tuple(_DATASETS)}.")
    if universe not in _UNIVERSES:
        raise ValueError(f"Select universe from {tuple(_UNIVERSES)}.")
    if data_source not in ("legacy", "ciz"):
        raise ValueError("Select data_source='legacy' or 'ciz'.")
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    try:
        start_date, end_date, comp_start_date, comp_end_date = (
            date.fromisoformat(value) for value in (start, end, comp_start, comp_end)
        )
    except (TypeError, ValueError) as error:
        raise ValueError("Dates must be ISO strings (YYYY-MM-DD).") from error
    if start_date > end_date or comp_start_date > comp_end_date:
        raise ValueError("Each start date must be on or before its end date.")
    if start_date.day != 1 or not pd.Timestamp(end_date).is_month_end:
        raise ValueError("CRSP dates require start at month start and end at month end.")
    if data_source == "legacy" and end_date > date(2024, 12, 31) and set(names) & {"crsp_monthly", "market_returns"}:
        raise ValueError("Legacy CRSP ends at 2024-12-31. Use data_source='ciz' for later dates.")
    crsp_params = {"start_date": start_date.isoformat(), "end_date": end_date.isoformat()}
    comp_params = {"start_date": comp_start_date.isoformat(), "end_date": comp_end_date.isoformat()}
    sample_name = universe if data_source == "legacy" else f"{universe}_ciz"
    cache_dir = (
        DEFAULT_CACHE_DIR / (f"{sample_name}_{start_date:%Y%m%d}_{end_date:%Y%m%d}"
                             f"_comp{comp_start_date:%Y%m%d}_{comp_end_date:%Y%m%d}")
        if cache_dir is None else Path(cache_dir)
    )
    if not cache_dir.is_absolute():
        cache_dir = Path(root_path) / cache_dir
    paths = {name: cache_dir / f"{name}.parquet" for name in names}
    pending = []
    for name, path in paths.items():
        sql, date_columns, key_columns = (
            _CIZ_DATASETS[name] if data_source == "ciz" and name in _CIZ_DATASETS else _DATASETS[name]
        )
        params = comp_params if name == "compustat_annual" else crsp_params
        if name == "ccm_links":
            params = {}
        specification = {
            "cache_version": 1, "sql": sql, "params": params, "universe": universe,
            "date_columns": date_columns, "key_columns": key_columns,
        }
        if data_source == "ciz" and name in _CIZ_DATASETS:
            specification["data_source"] = data_source
        signature = hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()
        if path.exists() and not refresh:
            sidecar = path.with_suffix(".json")
            if not sidecar.is_file():
                raise ValueError(f"Missing metadata for {path}. Use a new directory or refresh=True.")
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
            if metadata.get("signature") != signature:
                raise ValueError(f"Cache specification differs for {path}. Use a new directory or refresh=True.")
            parquet_metadata = pq.read_metadata(path)
            if (parquet_metadata.num_rows != metadata.get("rows")
                    or parquet_metadata.schema.names != metadata.get("columns")):
                raise ValueError(f"Cache schema or row count differs from metadata for {path}.")
            print(f"[{name}] Using cached file: {parquet_metadata.num_rows:,} rows.", flush=True)
        else:
            pending.append((name, path, specification, signature))
    if not pending:
        print("All requested LSV1994 datasets are cached; nothing to download.", flush=True)
        return paths

    print("Pending downloads: " + ", ".join(item[0] for item in pending), flush=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    owns_connection = connection is None
    if owns_connection:
        print("Connecting to WRDS; setup completion does not mean downloads are finished.", flush=True)
        connection = get_wrds_connection(username)
    try:
        for name, path, specification, signature in pending:
            with TemporaryDirectory(prefix=f".{name}_", dir=cache_dir) as temporary:
                temporary_path = Path(temporary) / path.name
                details = _write_query(connection, name, specification, temporary_path, chunk_size)
                metadata = {
                    "dataset": name, "signature": signature, **specification, **details,
                    "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
                    "reference_notebook": "references/lsv1994_table1_wrds_replication.ipynb",
                    "data_source": data_source if name in _CIZ_DATASETS else ("compustat" if name == "compustat_annual" else "ccm"),
                    "formation_exchange_codes": list(_UNIVERSES[universe]),
                    "formation_share_codes": [10, 11],
                    "exchange_and_share_filters_applied": False,
                    "return_units": "decimal", "delisting_return_adjustment": False,
                    "delisting_returns_in_source": data_source == "ciz" and name == "crsp_monthly",
                    "shares_outstanding_units": "thousands",
                    "compustat_monetary_units": "millions of reported currency; curcd retained",
                    "market_equity_formula_usd_millions": "abs(prc) * shrout / 1000",
                    "accounting_timing": "April t formation uses prior-year accounts; later rows are outcomes only",
                }
                temporary_metadata = temporary_path.with_suffix(".json")
                temporary_metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
                temporary_path.replace(path)
                temporary_metadata.replace(path.with_suffix(".json"))
            print(f"[{name}] Complete: {details['rows']:,} validated rows. Saved to {path}", flush=True)
    finally:
        if owns_connection:
            connection.close()
    print("All requested LSV1994 datasets are ready.", flush=True)
    return paths
