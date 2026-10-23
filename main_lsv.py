"""Run an LSV case; Q2-2/Q2-3 batches have separate entry points."""

from pathlib import Path

from core.lsv1994.runs import run_lsv
from utils.path_info import data_path


if __name__ == "__main__":
    replication_table = (1, 2, 4, 5, 6)  # One table or a tuple; Table VI includes Figure 2.
    formation_start = 1968
    formation_end = 1989
    return_end = "1990-04-30"
    run_sample = "nyse_amex"
    book_equity = "ceq"
    accounting_currency = None  # Reference convention; "USD" excludes non-USD accounts.
    raw_data_dir = (Path(data_path) / "raw" / "lsv1994"
                    / "nyse_amex_19621201_19900430_comp19620101_19931231")
    # Later market equity is used ONLY for Table V accounting-growth outcomes.
    outcome_market_data_dir = (Path(data_path) / "raw" / "lsv1994"
                               / "nyse_amex_19621201_20241231_comp19620101_19931231")
    run_replication = True
    show_plots = False
    build_submission = False  # Original-only preview; assemble all cases with submission_lsv1994.py.
    run_lsv(formation_start,formation_end,return_end,raw_data_dir,
            universe=run_sample,tables=replication_table,book_equity=book_equity,
            accounting_currency=accounting_currency,outcome_market_data_dir=outcome_market_data_dir,
            run_replication=run_replication,show_plots=show_plots,build_submission=build_submission)
