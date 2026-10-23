"""Independently replay the supplied Table I notebook on local cached inputs.

Run: python -m tests.validate_lsv1994_reference RAW_FOLDER RESULT_FOLDER
The notebook is a user-supplied reference. No WRDS connection is constructed;
only calculation cells 6,7,9,11,13 and its ntile helper are executed.
"""

import argparse
import ast
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.path_info import root_path


def validate_reference(raw_dir, result_dir):
    raw_dir, result_dir = Path(raw_dir), Path(result_dir)
    notebook_path = Path(root_path) / "references/lsv1994_table1_wrds_replication.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    names = ("crsp_monthly", "compustat_annual", "ccm_links")
    before = {name: hashlib.sha256((raw_dir / f"{name}.parquet").read_bytes()).hexdigest() for name in names}
    env = {"np": np, "pd": pd, "plt": plt, "display": lambda *args, **kwargs: None,
           "FORMATION_START": 1968, "FORMATION_END": 1989, "RETURN_END_YEAR": 1989,
           "HOLD_YEARS": [1, 2, 3, 4, 5], "COMMON_SHARES": [10, 11],
           "BOOK_EQUITY": "ceq", "DECILE_METHOD": "sas", "GS_RANK_MODE": "percentile",
           "CRSP_START": "1967-12-01", "CRSP_END": "1990-04-30",
           "COMP_START": "1962-01-01", "COMP_END": "1988-12-31",
           "MSF": "crsp.msf", "MSENAMES": "crsp.msenames", "FUNDA": "comp.funda",
           "CCM": "crsp.ccmxpf_linktable"}

    def local_load(name, sql, date_cols=None):
        if name.startswith("crsp_monthly"):
            frame = pd.read_parquet(raw_dir / "crsp_monthly.parquet")
            return frame.loc[pd.to_datetime(frame["date"]).between("1967-12-01", "1990-04-30")].copy()
        if name.startswith("compustat"):
            frame = pd.read_parquet(raw_dir / "compustat_annual.parquet")
            return frame.loc[pd.to_datetime(frame["datadate"]).between("1962-01-01", "1988-12-31")].copy()
        if name.startswith("ccm"):
            return pd.read_parquet(raw_dir / "ccm_links.parquet")
        raise ValueError(f"Unexpected reference input {name}")

    env["load_or_query"] = local_load
    helper_ast = ast.parse("".join(notebook["cells"][4]["source"]))
    helper_ast.body = [node for node in helper_ast.body if isinstance(node, ast.FunctionDef) and node.name == "ntile_series"]
    with redirect_stdout(io.StringIO()):
        exec(compile(helper_ast, str(notebook_path), "exec"), env)
        for cell in (6, 7, 9, 11, 13):
            exec(compile("".join(notebook["cells"][cell]["source"]), f"{notebook_path}:cell{cell}", "exec"), env)
    plt.close("all")
    actual_formation = pd.read_parquet(result_dir / "formation_panel.parquet")
    cols = ["permno", "formation_year", "me_apr", "bm", "cp", "ep", "gs"]
    pd.testing.assert_frame_equal(actual_formation[cols].sort_values(cols[:2]).reset_index(drop=True),
                                  env["formation"][cols].sort_values(cols[:2]).reset_index(drop=True),
                                  check_dtype=False, atol=1e-12, rtol=1e-12)
    actual_members = pd.read_parquet(result_dir / "portfolio_constituents.parquet").rename(columns={"portfolio": "decile"})
    cols = ["strategy", "formation_year", "permno", "decile"]
    pd.testing.assert_frame_equal(actual_members[cols].sort_values(cols).reset_index(drop=True),
                                  env["constituents"][cols].sort_values(cols).reset_index(drop=True), check_dtype=False)
    actual_annual = pd.read_parquet(result_dir / "annual_stock_returns.parquet")
    cols = ["permno", "start_year", "annual_ret", "size_adj_ret", "replacement_months", "n_months"]
    pd.testing.assert_frame_equal(actual_annual[cols].sort_values(cols[:2]).reset_index(drop=True),
                                  env["annual_stock"][cols].sort_values(cols[:2]).reset_index(drop=True),
                                  check_dtype=False, atol=1e-12, rtol=1e-12)
    actual_yearly = pd.read_csv(result_dir / "portfolio_year_returns.csv").rename(columns={"portfolio": "decile"})
    cols = ["strategy", "formation_year", "decile", "hold_year", "ret", "saar", "n_valid", "n_size_adj", "n_constituents"]
    pd.testing.assert_frame_equal(actual_yearly[cols].sort_values(cols[:4]).reset_index(drop=True),
                                  env["portfolio_year"][cols].sort_values(cols[:4]).reset_index(drop=True),
                                  check_dtype=False, atol=1e-12, rtol=1e-12)
    summary = pd.read_csv(result_dir / "table1_summary.csv")
    # 'return' is a Python keyword, so use explicit columns rather than namedtuple attributes.
    expected = np.array([env["table1"][s].loc[stat, d] for s, stat, d in
                         summary[["strategy", "stat", "decile"]].itertuples(index=False, name=None)])
    np.testing.assert_allclose(summary["return"], expected, atol=1e-12, rtol=1e-12)
    after = {name: hashlib.sha256((raw_dir / f"{name}.parquet").read_bytes()).hexdigest() for name in names}
    if before != after:
        raise AssertionError("Raw input changed during validation.")
    record = {"passed": True, "cells_compared": len(summary),
              "maximum_absolute_return_difference": float(np.max(np.abs(summary["return"] - expected))),
              "formation_rows_compared": len(actual_formation), "memberships_compared": len(actual_members),
              "annual_stock_years_compared": len(actual_annual), "cohort_returns_compared": len(actual_yearly),
              "notebook_sha256": hashlib.sha256(notebook_path.read_bytes()).hexdigest(),
              "summary_sha256": hashlib.sha256((result_dir / "table1_summary.csv").read_bytes()).hexdigest(),
              "raw_sha256": before, "raw_inputs_unchanged": True}
    (result_dir / "reference_validation.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_dir", type=Path)
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate_reference(args.raw_dir, args.result_dir), indent=2))
