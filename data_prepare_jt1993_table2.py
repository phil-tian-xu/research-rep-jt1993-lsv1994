"""Download only the additional JT1993 Table II inputs; edit settings below.

The defaults match the existing 1960-2024 legacy return folders and prepare
all three exchange samples sequentially. Only monthly_characteristics
(CRSP monthly prices, shares, exchange/share codes) and market_returns
(CRSP value-/equal-weighted indexes) are requested. No daily or monthly
stock-return extract is queried or rewritten.

Valid cached files are reused, so rerunning resumes with missing files only.
The original 1962-1989 sample already has these inputs. To target a different
folder, match its dates, universe, and data_source exactly in the settings.
Credentials use WRDS_USERNAME and the saved PostgreSQL password as usual.
"""

import json

import pandas as pd

from utils.jt1993_data_download import download_jt1993_data


if __name__ == "__main__":
    data_start_date = "1960-01-01"  # Match the existing return folders, including warm-up history.
    data_end_date = "2024-12-31"
    run_data_source = "legacy"
    run_samples = ("nyse", "nyse_amex", "nyse_amex_nasdaq")  # Keep only the samples you need.

    for run_sample in run_samples:
        print(f"JT1993 Table II additional inputs | {run_sample} | {run_data_source} | "
              f"{data_start_date} to {data_end_date}", flush=True)
        paths = download_jt1993_data(
            start=data_start_date,
            end=data_end_date,
            universe=run_sample,
            data_source=run_data_source,
            datasets=("monthly_characteristics", "market_returns"),
            refresh=False,
        )
        summary = []
        for name, path in paths.items():
            info = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
            summary.append({
                "dataset": name, "rows": info["rows"],
                "start": info["first_date"], "end": info["last_date"],
            })
        print(pd.DataFrame(summary).to_string(index=False), flush=True)
        print(f"Raw folder for main.py: {paths['monthly_characteristics'].parent}", flush=True)

    print("JT1993 Table II additional-input preparation finished successfully.", flush=True)
