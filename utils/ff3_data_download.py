"""Download the public monthly US FF3 factors, independently of CRSP/WRDS."""

import csv
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO, StringIO
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from urllib.request import Request, urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd

from utils.path_info import data_path, root_path


FF3_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip"
FACTOR_COLUMNS = ["mkt_rf", "smb", "hml", "rf"]


def parse_ff3_monthly(archive: bytes) -> pd.DataFrame:
    """Read only the monthly CSV block; convert percent returns to decimals.

    Annual rows and descriptive text are excluded. Source missing-value codes
    -99.99 and -999 become NaN, never zero. Dates become calendar month ends.
    Duplicate, invalid, or missing calendar months are rejected.
    """
    with ZipFile(BytesIO(archive)) as zipped:
        files = [name for name in zipped.namelist() if name.lower().endswith(".csv")]
        if len(files) != 1:
            raise ValueError("Expected exactly one CSV in the French FF3 archive.")
        text = zipped.read(files[0]).decode("utf-8-sig")
    rows, header_seen = [], False
    for raw in csv.reader(StringIO(text)):
        row = [value.strip() for value in raw]
        if row == ["", "Mkt-RF", "SMB", "HML", "RF"]:
            header_seen = True
            continue
        if not header_seen:
            continue
        if row and re.fullmatch(r"\d{6}", row[0]):
            if len(row) != 5:
                raise ValueError(f"Malformed monthly FF3 row: {row}")
            rows.append(row)
        elif rows:
            break  # Monthly block has ended; do not read the annual block.
    if not rows:
        raise ValueError("No monthly FF3 observations found after the expected CSV header.")
    frame = pd.DataFrame(rows, columns=["month", *FACTOR_COLUMNS])
    frame["month"] = pd.to_datetime(frame["month"], format="%Y%m") + pd.offsets.MonthEnd(0)
    periods = pd.PeriodIndex(frame["month"], freq="M")
    expected = pd.period_range(periods.min(), periods.max(), freq="M")
    if not periods.equals(expected):
        raise ValueError("Monthly FF3 dates must be unique, ordered, and cover consecutive calendar months.")
    values = frame[FACTOR_COLUMNS].apply(pd.to_numeric, errors="raise")
    if np.isinf(values.to_numpy()).any():
        raise ValueError("Infinite factor returns in the FF3 source.")
    frame[FACTOR_COLUMNS] = values.replace([-99.99, -999.0], np.nan) / 100
    return frame


def download_ff3_data(refresh: bool = False, *, data_dir: str | Path | None = None) -> Path:
    """Cache the full monthly US FF3 series and its original ZIP; return Parquet.

    The series is small, so download all available months once. The analysis
    script selects its reporting dates. No WRDS account or password is needed.
    Default: data/raw/ff3/ff3_monthly.parquet, with source ZIP and JSON provenance.
    The JSON records decimal units, coverage, retrieval time, and ZIP checksum.
    refresh=True explicitly replaces the cached snapshot with the current file;
    the publisher may revise historical observations between data releases.
    Relative data_dir paths are anchored to utils.path_info.root_path.
    """
    folder = Path(data_dir) if data_dir is not None else Path(data_path) / "raw" / "ff3"
    if not folder.is_absolute():
        folder = Path(root_path) / folder
    folder = folder.resolve()
    parquet = folder / "ff3_monthly.parquet"
    archive_path = folder / "F-F_Research_Data_Factors_CSV.zip"
    metadata_path = parquet.with_suffix(".json")
    paths = (parquet, archive_path, metadata_path)
    if not refresh and any(path.exists() for path in paths):
        if not all(path.is_file() for path in paths):
            raise ValueError("Incomplete FF3 cache. Run data_prepare_ff3.py with refresh_data=True.")
        info = json.loads(metadata_path.read_text(encoding="utf-8"))
        cached = pd.read_parquet(parquet)
        if (info.get("source_url") != FF3_URL or info.get("return_units") != "decimal"
                or info.get("frequency") != "monthly" or info.get("rows") != len(cached)
                or cached.columns.tolist() != ["month", *FACTOR_COLUMNS]
                or info.get("source_sha256") != sha256(archive_path.read_bytes()).hexdigest()
                or info.get("parquet_sha256") != sha256(parquet.read_bytes()).hexdigest()):
            raise ValueError("FF3 cache/provenance mismatch. Set refresh_data=True to download again.")
        print(f"FF3 cache hit: {parquet} | {info['first_month']} to {info['last_month']}", flush=True)
        return parquet

    print(f"Downloading monthly US FF3 factors: {FF3_URL}", flush=True)
    request = Request(FF3_URL, headers={"User-Agent": "JT1993-replication/1.0"})
    with urlopen(request, timeout=60) as response:
        archive = response.read()
    frame = parse_ff3_monthly(archive)
    folder.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".ff3_", dir=folder) as temporary:
        staging = Path(temporary)
        staged_parquet = staging / parquet.name
        frame.to_parquet(staged_parquet, index=False)
        info = {
            "dataset": "US Fama-French three factors and risk-free rate", "frequency": "monthly",
            "source_url": FF3_URL, "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_sha256": sha256(archive).hexdigest(),
            "parquet_sha256": sha256(staged_parquet.read_bytes()).hexdigest(),
            "source_units": "percent", "return_units": "decimal", "annual_rows_included": False,
            "first_month": str(frame["month"].min().date()), "last_month": str(frame["month"].max().date()),
            "rows": len(frame), "columns": frame.columns.tolist(),
            "missing_values": frame[FACTOR_COLUMNS].isna().sum().to_dict(),
        }
        (staging / archive_path.name).write_bytes(archive)
        (staging / metadata_path.name).write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
        for target in (archive_path, parquet, metadata_path):
            (staging / target.name).replace(target)
    print(f"FF3 preparation complete: {len(frame):,} months, {info['first_month']} to {info['last_month']}", flush=True)
    print(f"Factor file for main_jt1993_ff3.py: {parquet}", flush=True)
    return parquet
