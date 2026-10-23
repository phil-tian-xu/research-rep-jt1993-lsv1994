"""Prepare two local subsamples before running the existing JT1993 analysis.

Reporting periods do not overlap. Each subset also contains 24 months of past
history for Table I's longest J=12/K=12 strategy and Panel B's five-day gap.
That history is for formation only: use the saved reporting dates in main.py.
No observations after a period's end are copied, and source files are never
modified. Table VII keeps its existing complete-36-month-cohort convention.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from utils.path_info import data_path, root_path


def _partition_parquet(source, periods, folders, batch_size):
    """Read a source file once, preserving columns, values, and row order."""
    date_column = "date" if source.stem == "daily_returns" else "month"
    stats = [{"rows": 0, "first_date": None, "last_date": None, "months": set()} for _ in periods]
    bounds = [
        (pa.scalar(period["history_start"].to_pydatetime(), type=pa.timestamp("ns")),
         pa.scalar((period["end"] + pd.DateOffset(days=1)).to_pydatetime(), type=pa.timestamp("ns")))
        for period in periods
    ]
    with pq.ParquetFile(source) as parquet:
        schema = parquet.schema_arrow
        if date_column not in schema.names:
            raise ValueError(f"Missing {date_column} column in {source}.")
        source_rows = parquet.metadata.num_rows
        writers = []
        try:
            for folder in folders:
                writers.append(pq.ParquetWriter(folder / source.name, schema, compression="zstd"))
            rows_read = 0
            for number, batch in enumerate(parquet.iter_batches(batch_size=batch_size), start=1):
                dates = pc.cast(batch.column(date_column), pa.timestamp("ns"))
                if dates.null_count:
                    raise ValueError(f"Missing dates in {source}; partitioning would silently lose rows.")
                for i, (lower, upper) in enumerate(bounds):
                    mask = pc.and_(pc.greater_equal(dates, lower), pc.less(dates, upper))
                    selected = batch.filter(mask)
                    if selected.num_rows == 0:
                        continue
                    writers[i].write_table(pa.Table.from_batches([selected], schema=schema))
                    selected_dates = dates.filter(mask)
                    extent = pc.min_max(selected_dates).as_py()
                    first, last = extent["min"].date(), extent["max"].date()
                    record = stats[i]
                    record["rows"] += selected.num_rows
                    record["first_date"] = first if record["first_date"] is None else min(record["first_date"], first)
                    record["last_date"] = last if record["last_date"] is None else max(record["last_date"], last)
                    month_ids = pc.add(pc.multiply(pc.year(selected_dates), 12), pc.month(selected_dates))
                    record["months"].update(pc.unique(month_ids).to_pylist())
                rows_read += batch.num_rows
                if number % 10 == 0:
                    print(f"[{source.stem}] Read {rows_read:,}/{source_rows:,} source rows.", flush=True)
        finally:
            for writer in writers:
                writer.close()

    for period, record in zip(periods, stats):
        first_id = period["history_start"].year * 12 + period["history_start"].month
        last_id = period["end"].year * 12 + period["end"].month
        missing = sorted(set(range(first_id, last_id + 1)) - record.pop("months"))
        if missing:
            labels = [f"{(value - 1) // 12:04d}-{(value - 1) % 12 + 1:02d}" for value in missing]
            raise ValueError(f"{source.name}: missing calendar months for {period['label']} (including warm-up): {labels}")
        record["first_date"] = record["first_date"].isoformat()
        record["last_date"] = record["last_date"].isoformat()
        record["source_rows"] = source_rows
        record["columns"] = schema.names
    return stats


def partition_jt1993_data(
    raw_data_dir: str | Path,
    run_start_date: str,
    split_date: str,
    run_end_date: str,
    output_dir: str | Path | None = None,
    *,
    batch_size: int = 250_000,
) -> dict[str, Path]:
    """Write period1=[start, split) and period2=[split, end], plus past history.

    Start and split must be month starts; end must be a month end. For the
    original paper, choose 1965-01-01, 1975-01-01, and 1989-12-31. Every file
    retains 24 months before its reporting start, and must cover every calendar
    month through its end. Thus the second subset contains 1973-1974 formation
    history, but its reported returns start in 1975.

    Copy monthly_returns and whichever of daily_returns, monthly_characteristics,
    and market_returns already exist. Missing optional datasets are not created.
    Stream each Parquet source once, routing rows to both subsets. All source
    columns and values survive; there is no stock filtering or return adjustment.

    Relative paths use utils.path_info.root_path. Default outputs are under
    data/processed/jt1993/subsamples/<source name>_<report start>_<report end>/.
    Each folder has partition_info.json with the three parameters for main.py.
    Dataset sidecars record actual subset coverage and retain source provenance.
    These are derived files, not download caches. Existing targets are refused.
    No WRDS connection, analysis, or reporter is run here.
    """
    start, split, end = map(pd.Timestamp, (run_start_date, split_date, run_end_date))
    if any(pd.isna(value) or value.tzinfo is not None or value != value.normalize() for value in (start, split, end)):
        raise ValueError("Partition dates must be calendar dates without a timezone or time of day.")
    if not start < split <= end or start.day != 1 or split.day != 1 or not end.is_month_end:
        raise ValueError("Require month-start start < month-start split <= month-end end.")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    raw_data_dir = Path(raw_data_dir)
    if not raw_data_dir.is_absolute():
        raw_data_dir = Path(root_path) / raw_data_dir
    raw_data_dir = raw_data_dir.resolve()
    if not (raw_data_dir / "monthly_returns.parquet").is_file():
        raise FileNotFoundError(f"Missing monthly_returns.parquet in {raw_data_dir}. Prepare the source data first.")
    output_dir = Path(output_dir) if output_dir is not None else Path(data_path) / "processed" / "jt1993" / "subsamples"
    if not output_dir.is_absolute():
        output_dir = Path(root_path) / output_dir
    output_dir = output_dir.resolve()
    if output_dir == raw_data_dir or output_dir.is_relative_to(raw_data_dir):
        raise ValueError("Place derived partitions outside the source raw-data folder.")

    periods = []
    for label, period_start, period_end in (("period1", start, split - pd.DateOffset(days=1)), ("period2", split, end)):
        target = (output_dir / f"{raw_data_dir.name}_{period_start:%Y%m}_{period_end:%Y%m}").resolve()
        if not target.is_relative_to(output_dir):
            raise ValueError("Partition target must stay inside output_dir.")
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"Partition already exists: {target}. Choose a new output_dir to regenerate.")
        periods.append({"label": label, "start": period_start, "end": period_end,
                        "history_start": period_start - pd.DateOffset(months=24), "target": target, "datasets": {}})
    names = ("monthly_returns", "daily_returns", "monthly_characteristics", "market_returns")
    sources = [raw_data_dir / f"{name}.parquet" for name in names if (raw_data_dir / f"{name}.parquet").is_file()]
    output_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".partition_", dir=output_dir) as temporary:
        staging_root = Path(temporary).resolve()
        folders = [staging_root / period["label"] for period in periods]
        for folder in folders:
            folder.mkdir()
        for source in sources:
            print(f"Partitioning {source.name}...", flush=True)
            sidecar = source.with_suffix(".json")
            source_info = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.is_file() else None
            records = _partition_parquet(source, periods, folders, batch_size)
            for period, folder, record in zip(periods, folders, records):
                info = {
                    "dataset": source.stem, "kind": "subsample",
                    "params": {"start_date": str(period["history_start"].date()), "end_date": str(period["end"].date())},
                    "run_start_date": str(period["start"].date()), "run_end_date": str(period["end"].date()),
                    "history_months": 24, **record,
                    "source_file": str(source), "source_download_metadata": source_info,
                }
                for key in ("universe", "data_source", "exchange_codes", "share_codes", "return_units",
                            "shares_outstanding_units", "delisting_return_adjustment", "delisting_returns_in_source"):
                    if source_info and key in source_info:
                        info[key] = source_info[key]
                (folder / f"{source.stem}.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
                period["datasets"][source.stem] = record
                print(f"  {period['label']}: {record['rows']:,} rows, {record['first_date']} to {record['last_date']}", flush=True)
        for period, folder in zip(periods, folders):
            info = {
                "paper": "jt1993", "source_raw_data_dir": str(raw_data_dir),
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "split_date": str(split.date()), "history_months": 24,
                "history_start_date": str(period["history_start"].date()),
                "history_policy": "past formation history only; report dates below exclude warm-up",
                "main_parameters": {"raw_data_dir": str(period["target"]),
                                    "run_start_date": str(period["start"].date()), "run_end_date": str(period["end"].date())},
                "datasets": period["datasets"],
            }
            (folder / "partition_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
        # Both subsets have been validated before either is published.
        for period, folder in zip(periods, folders):
            if not folder.resolve().is_relative_to(staging_root) or not period["target"].is_relative_to(output_dir):
                raise ValueError("Partition paths must stay inside their staging/output directories.")
            folder.rename(period["target"])

    for period in periods:
        print(f"\n{period['label']} settings for main.py:", flush=True)
        print(f"  raw_data_dir = {period['target']}", flush=True)
        print(f"  run_start_date = {period['start']:%Y-%m-%d}", flush=True)
        print(f"  run_end_date = {period['end']:%Y-%m-%d}", flush=True)
    return {period["label"]: period["target"] for period in periods}
