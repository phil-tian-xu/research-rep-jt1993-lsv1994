"""Table I reporter: the paper-format results table first, followed by plots.

Call report_table1(result_dir) after replication. It reads saved CSV/JSON files,
checks their consistency, and saves table1.csv/table1.html under
<result_dir>/reports/. The main report displays the table before the plots.
It never loads raw CRSP data or reruns portfolio logic.
Use show=True for Matplotlib windows when running with an interactive backend.
"""

from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.path_info import root_path
from .notebook import plot_notebook
from .overview import plot_overview
from .tables import write_table1


def _load_results(result_dir):
    """Reject incomplete or mixed run files before plotting their comparisons."""
    required = ("table1_summary.csv", "table1_monthly_returns.csv", "table1_formation_counts.csv",
                "table1_monthly_coverage.csv", "run_info.json")
    for name in required:
        if not (result_dir / name).is_file():
            raise FileNotFoundError(f"Missing {result_dir / name}. Run main.py with run_replication=True first.")
    info = json.loads((result_dir / "run_info.json").read_text(encoding="utf-8"))
    summary = pd.read_csv(result_dir / "table1_summary.csv")
    monthly = pd.read_csv(result_dir / "table1_monthly_returns.csv", parse_dates=["month"])
    formations = pd.read_csv(result_dir / "table1_formation_counts.csv")
    coverage = pd.read_csv(result_dir / "table1_monthly_coverage.csv", parse_dates=["month"])
    keys, legs = ["panel", "J", "K"], ["loser", "winner", "buy_sell"]
    expected_keys = pd.MultiIndex.from_product(
        [info["settings"]["panels"], info["settings"]["ranking_months"], info["settings"]["holding_months"], legs],
        names=keys + ["leg"],
    )
    actual_keys = pd.MultiIndex.from_frame(summary[keys + ["leg"]])
    if not actual_keys.is_unique or set(actual_keys) != set(expected_keys):
        raise ValueError("Summary must contain each panel/J/K/leg exactly once.")
    expected_months = pd.date_range(info["run_start_date"], info["run_end_date"], freq="ME")
    counts = monthly.groupby(keys).size().reindex(expected_keys.droplevel("leg").unique())
    if (monthly.duplicated(keys + ["month"]).any() or not monthly["month"].isin(expected_months).all()
            or not counts.eq(len(expected_months)).all() or not summary["n"].eq(len(expected_months)).all()):
        raise ValueError("Monthly strategy dates/counts disagree with the saved run window.")
    if not np.isfinite(monthly[legs].to_numpy()).all():
        raise ValueError("Monthly strategy returns must be complete and finite.")
    means = monthly.groupby(keys)[legs].mean().stack()
    means.index.names = keys + ["leg"]
    saved_means = summary.set_index(keys + ["leg"])["mean"].reindex(means.index)
    if not np.allclose(means, saved_means, atol=1e-12, rtol=1e-10):
        raise ValueError("Summary means disagree with monthly returns; check for mixed runs or units.")
    if coverage["month"].duplicated().any() or set(coverage["month"]) != set(expected_months):
        raise ValueError("Monthly stock coverage does not match the saved run window.")
    if formations.duplicated(["panel", "J", "formation_id", "momr"]).any() or not formations["n_stocks"].gt(0).all():
        raise ValueError("Formation counts must have unique keys and positive stock counts.")
    return (summary, monthly.sort_values(keys + ["month"]), formations,
            coverage.sort_values("month"), info)


def report_table1(result_dir: str | Path, *, output_dir: str | Path | None = None, show: bool = False) -> dict[str, Path]:
    """Write the display-ready Table I, then plots; return all report paths.

    Paths are anchored to utils.path_info.root_path when relative. Notebook
    figures retain their original decimal/percent/basis-point conventions;
    overview figures label all return axes in percent or percentage points.
    Paper comparisons use only benchmark columns already saved by replication.
    """
    result_dir = Path(result_dir)
    if not result_dir.is_absolute():
        result_dir = Path(root_path) / result_dir
    output_dir = Path(output_dir) if output_dir is not None else result_dir / "reports"
    if not output_dir.is_absolute():
        output_dir = Path(root_path) / output_dir
    summary, monthly, formations, coverage, info = _load_results(result_dir)
    metadata = info["raw_inputs"]["monthly_returns"].get("download_metadata") or {}
    sample_label = {
        "nyse_amex": "NYSE/AMEX Common Stocks", "nyse": "NYSE Common Stocks",
        "nyse_amex_nasdaq": "NYSE/AMEX/Nasdaq Common Stocks",
    }.get(metadata.get("universe"), "Sample Stocks")
    caption = f"{sample_label} | {info['run_start_date'][:7]} to {info['run_end_date'][:7]} | {int(summary['n'].min())} months"
    table_paths, table_markup = write_table1(summary, output_dir, caption)
    print(f"Table I results CSV: {table_paths['table1_csv']}")
    print(f"Table I display: {table_paths['table1_html']}")
    sets = {
        "notebook": plot_notebook(summary, monthly, formations, coverage, output_dir / "notebook", sample_label, show),
        "overview": plot_overview(summary, monthly, output_dir / "overview", caption, show),
    }
    sections = []
    for name, plots in sets.items():
        cards = []
        for title, path in plots.items():
            relative = escape(path.relative_to(output_dir).as_posix(), quote=True)
            label = escape(title.replace("_", " ").title())
            cards.append(f'<figure><a href="{relative}"><img loading="lazy" src="{relative}" alt="{label}"></a><figcaption>{label}</figcaption></figure>')
        heading = "1. Reference notebook plots" if name == "notebook" else "2. Overview plots"
        sections.append(f'<section id="{name}"><h2>{heading}</h2><div class="gallery {name}">{"".join(cards)}</div></section>')
    html = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>JT1993 Table I replication results</title><style>
body{{font:16px/1.5 system-ui,sans-serif;color:#24323e;background:#f5f7f9;margin:0;padding:28px}}
main{{max-width:1450px;margin:auto}}h1,h2{{line-height:1.2}}nav a{{margin-right:24px;color:#176b9b}}
.gallery{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,500px),1fr));gap:18px}}
.overview{{grid-template-columns:1fr}}figure{{background:white;margin:0;padding:10px;border:1px solid #dce1e6;border-radius:6px}}
img{{display:block;width:100%;height:auto}}figcaption{{padding:8px;font-size:14px;color:#52606d}}section{{margin-top:36px}}
</style><main><h1>JT1993 Table I</h1><p>{escape(caption)}</p>
<nav><a href="#table1">Table I results</a><a href="table1.csv" download>Download Table I CSV</a>
<a href="#notebook">Reference notebook plots</a><a href="#overview">Overview plots</a></nav>
<section id="table1">{table_markup}<p><a href="table1.csv" download>Download Table I CSV</a> | <a href="table1.html">Open table only</a></p></section>
<p>Click a figure to open the full PNG. Paper comparisons appear only when the saved run includes the paper benchmark.</p>
{"".join(sections)}</main></html>'''
    gallery = output_dir / "index.html"
    gallery.write_text(html, encoding="utf-8")
    report_info = {
        "result_dir": str(result_dir), "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "matplotlib_version": matplotlib.__version__, "caption": caption,
        "paper_comparison_available": "paper_mean" in summary.columns,
        "tables": {name: path.relative_to(output_dir).as_posix() for name, path in table_paths.items()},
        "figures": {name: [path.relative_to(output_dir).as_posix() for path in plots.values()] for name, plots in sets.items()},
        "notes": {
            "table1": "Primary output: calculated means (four decimals) and parenthesized t-statistics (two decimals); panels side by side, J/K=3,6,9,12, Sell/Buy/Buy-sell in paper order.",
            "notebook": "Original plot designs and units, including independent heatmap scales and 12-month rolling means (minimum six observations).",
            "overview": "Shared scales across panels; paper/notebook/replication point comparisons; additive cumulative W-L and trailing 36-month means with pointwise conventional standard-error bands.",
            "ranking_signal": "Formation-decile means are weighted by stock counts to reconstruct the individual stock-formation average.",
        },
    }
    (output_dir / "report_info.json").write_text(json.dumps(report_info, indent=2) + "\n", encoding="utf-8")
    paths = {f"{name}/{title}": path for name, plots in sets.items() for title, path in plots.items()}
    print(f"Table I reporter: {len(paths)} figures saved; gallery: {gallery}")
    if show:
        plt.show()
    return {**table_paths, **paths, "index": gallery}
