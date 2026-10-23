"""Generate the complete current JT1993 submission from saved results.

Run: .venv/Scripts/python.exe -B submission_jt1993.py
Select the saved original-sample, exchange, subperiod, and FF3 runs below.
This script does not download data or rerun portfolio calculations.
"""

from pathlib import Path

from core.submission.jt1993 import generate_submission
from utils.path_info import root_path, results_path


if __name__ == "__main__":
    run_tags = {
        "original": "nyse_amex_19620701_19891231_196501_198912",
        "NYSE": "nyse_19600101_20241231_196501_198912_196501_198912",
        "NYSE/AMEX": "nyse_amex_19600101_20241231_196501_198912_196501_198912",
        "NYSE/AMEX/Nasdaq": "nyse_amex_nasdaq_19600101_20241231_196501_198912_196501_198912",
        "recent": "nyse_amex_19600101_20241231_199001_202412_199001_202412",
    }
    runs = {
        label: {table: Path(results_path) / "jt1993" / "replication" / f"table{table}" / tag
                for table in (1, 2, 7)}
        for label, tag in run_tags.items()
    }
    paper_dir = Path(root_path) / "references" / "jt1993" / "paper_tables"
    output_path = Path(results_path) / "jt1993" / "submission" / "submission.html"
    ff3_result_dir = (
        Path(results_path) / "jt1993" / "ff3"
        / f"{run_tags['original']}_196501_198912_panelA_J6_K6_HAC5"
    )

    submission = generate_submission(
        runs["original"], paper_dir=paper_dir, output_path=output_path,
        exchange_runs={label: runs[label] for label in ("NYSE", "NYSE/AMEX", "NYSE/AMEX/Nasdaq")},
        subperiod_runs={"1965–1989": runs["NYSE/AMEX"], "1990–2024": runs["recent"]},
        ff3_result_dir=ff3_result_dir,
    )
    print(f"Submission saved: {submission}")
