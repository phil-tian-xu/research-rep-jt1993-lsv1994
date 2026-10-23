"""Build Q1-1 through Q1-5 from saved replication/factor results and sample notes.

Reads saved summaries and their existing metadata, never raw CRSP data or WRDS.
The only output is a self-contained HTML file. Published benchmarks are loaded
independently of any paper/reference columns already attached to a run.
"""

from html import escape
import json
from pathlib import Path
from string import Template

from matplotlib.figure import Figure
import numpy as np
import pandas as pd

from core.jt1993.reporter.table1.tables import table1_rows
from core.jt1993.reporter.table2 import table2_rows
from core.jt1993.reporter.table7 import table7_rows
from utils.path_info import root_path
from ._html import html_table as _html_table, figure_html as _figure_html
from .comparisons import render_case_section
from .ff3 import render_ff3_section


ROOT = Path(root_path)
PAPER_DIR = ROOT / "references" / "jt1993" / "paper_tables"
KEYS = {1: ["panel", "J", "K", "leg"], 2: ["portfolio"], 7: ["event_month"]}
VALUES = {
    1: ["mean", "t"],
    2: ["beta", "average_market_cap_millions"],
    7: ["monthly_return", "monthly_t", "cumulative_return", "cumulative_t"],
}
BLUE, GRAY = "#16766d", "#687180"


def _validate_table(frame, table):
    required = KEYS[table] + VALUES[table]
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"Table {table}: missing columns {sorted(missing)}.")
    # Reuse the existing reporters' exact row/strategy coverage checks.
    {1: table1_rows, 2: table2_rows, 7: table7_rows}[table](frame)
    numeric = frame[VALUES[table]].to_numpy(dtype=float, copy=True)
    if table == 2:
        spread = frame["portfolio"].eq("P10-P1").to_numpy()
        if not frame.loc[spread, "average_market_cap_millions"].isna().all():
            raise ValueError("Table II: spread capitalization must be absent, not zero.")
        numeric[spread, 1] = 0  # Only this published dash may be nonnumeric.
    if not np.isfinite(numeric).all():
        raise ValueError(f"Table {table}: means, t-statistics, betas, and sizes must be finite.")


def load_paper_tables(paper_dir=PAPER_DIR):
    """Load all published cells at their printed precision, without recomputing them."""
    paper_dir = Path(paper_dir)
    if not paper_dir.is_absolute():
        paper_dir = ROOT / paper_dir
    tables = {table: pd.read_csv(paper_dir / f"table{table}.csv") for table in KEYS}
    for table, frame in tables.items():
        _validate_table(frame, table)
    return tables


def load_results(result_dirs, *, original_sample=True):
    """Reject incomplete or incompatible tables within each sample/period.

    Paper comparisons require the original NYSE/AMEX 1965-1989 sample.
    Existing run_info.json files are input checks, not submission deliverables.
    """
    if set(result_dirs) != set(KEYS):
        raise ValueError("Each sample requires result directories for tables 1, 2, and 7.")
    tables, info = {}, {}
    for table, directory in result_dirs.items():
        directory = Path(directory)
        if not directory.is_absolute():
            directory = ROOT / directory
        tables[table] = pd.read_csv(directory / f"table{table}_summary.csv")
        _validate_table(tables[table], table)
        info[table] = json.loads((directory / "run_info.json").read_text(encoding="utf-8"))
        metadata = info[table]
        if metadata.get("paper") != "jt1993" or metadata.get("table") != table:
            raise ValueError(f"Table {table}: incompatible paper/table identity.")
        if original_sample and (metadata.get("run_start_date") != "1965-01-01"
                                or metadata.get("run_end_date") != "1989-12-31"):
            raise ValueError(f"Table {table}: the original-sample submission requires JT1993, 1965-1989.")
    windows = {(metadata["run_start_date"], metadata["run_end_date"]) for metadata in info.values()}
    if len(windows) != 1:
        raise ValueError("Tables I, II, and VII must use the same reporting dates.")
    start, end = map(pd.Timestamp, windows.pop())
    if start.day != 1 or not end.is_month_end or start > end:
        raise ValueError("Reporting dates must span complete calendar months.")
    n_months = len(pd.period_range(start, end, freq="M"))
    monthly_input = info[1]["raw_inputs"]["monthly_returns"]
    universe = monthly_input["download_metadata"]["universe"]
    if universe not in ("nyse", "nyse_amex", "nyse_amex_nasdaq"):
        raise ValueError(f"Unsupported exchange universe: {universe}.")
    if original_sample and universe != "nyse_amex":
        raise ValueError("The original paper comparison requires the NYSE/AMEX universe.")
    raw_directory = Path(monthly_input["path"]).parent
    for table in (2, 7):
        if Path(info[table]["raw_data_dir"]) != raw_directory:
            raise ValueError(f"Table {table}: raw sample differs from Table I.")
    if not tables[1]["n"].eq(n_months).all() or not tables[2]["n_months"].eq(n_months).all():
        raise ValueError(f"This reporting window requires {n_months} calendar return months.")
    legs = tables[1].pivot(index=["panel", "J", "K"], columns="leg", values="mean")
    if not np.allclose(legs["buy_sell"], legs["winner"] - legs["loser"], rtol=0, atol=1e-12):
        raise ValueError("Table I: spread means disagree with winner minus loser.")
    event = tables[7].sort_values("event_month")
    if not np.allclose(event["cumulative_return"], event["monthly_return"].cumsum(), rtol=0, atol=1e-12):
        raise ValueError("Table VII: replication cumulative returns must be additive.")
    first = pd.Period(info[7]["formation_start"], freq="M")
    last = pd.Period(info[7]["formation_end"], freq="M")
    n_cohorts = last.ordinal - first.ordinal + 1
    if (first != start.to_period("M") or last + 35 != end.to_period("M")
            or n_cohorts < 2 or n_cohorts != info[7]["n_cohorts"] or not event["n_cohorts"].eq(n_cohorts).all()):
        raise ValueError("Table VII: inconsistent complete-cohort dates/counts.")
    return tables, info


def compare_tables(results, paper):
    """Join by economic identifiers; differences are replication minus paper."""
    comparisons = {}
    for table in KEYS:
        _validate_table(results[table], table)
        _validate_table(paper[table], table)
        columns = KEYS[table] + VALUES[table]
        joined = results[table][columns].merge(
            paper[table][columns], on=KEYS[table], how="outer", validate="one_to_one",
            suffixes=("_replication", "_paper"), indicator=True,
        )
        if not joined["_merge"].eq("both").all():
            raise ValueError(f"Table {table}: replication/paper keys do not match.")
        joined = joined.drop(columns="_merge")
        for column in VALUES[table]:
            difference = joined[f"{column}_replication"] - joined[f"{column}_paper"]
            is_return = column in ("mean", "monthly_return", "cumulative_return")
            joined[column + ("_diff_bp" if is_return else "_diff")] = difference * (10_000 if is_return else 1)
        comparisons[table] = joined
    return comparisons


def _table1_html(results, paper):
    mine, original = table1_rows(results)[2:], table1_rows(paper)[2:]
    columns = [0, 1, 3, 4, 5, 6, 8, 9, 10, 11]
    standalone = _html_table(
        ["J", "Portfolio", "3", "6", "9", "12", "3", "6", "9", "12"],
        [[row[i] for i in columns] for row in mine], "Table I. My replication",
        groups=[("", 2), ("Panel A: K months", 4), ("Panel B: K months", 4)], t_rows=True,
    )
    comparisons = []
    for panel, offset in (("A", 3), ("B", 8)):
        rows = [left[:2] + left[offset:offset + 4] + right[offset:offset + 4]
                for left, right in zip(mine, original)]
        comparisons.append(_html_table(
            ["J", "Portfolio", "3", "6", "9", "12", "3", "6", "9", "12"],
            rows, f"Table I, Panel {panel}. My replication versus original paper",
            groups=[("", 2), ("My replication: K months", 4), ("Original paper: K months", 4)], t_rows=True,
        ))
    return standalone, "".join(comparisons)


def _table2_html(results, comparison):
    display = table2_rows(results)
    standalone = _html_table(display[0], display[1:], "Table II. My replication")
    values = comparison.set_index("portfolio")
    rows = []
    for portfolio in [*(f"P{i}" for i in range(1, 11)), "P10-P1"]:
        row = values.loc[portfolio]
        cells = [portfolio]
        for source in ("replication", "paper"):
            cells.extend([f"{row[f'beta_{source}']:.2f}",
                          "—" if portfolio == "P10-P1" else f"{row[f'average_market_cap_millions_{source}']:.2f}"])
        cells.extend([f"{row['beta_diff']:+.2f}", "—" if portfolio == "P10-P1"
                      else f"{row['average_market_cap_millions_diff']:+.2f}"])
        rows.append(cells)
    paired = _html_table(
        ["Portfolio", "Beta", "Size ($m)", "Beta", "Size ($m)", "Beta", "Size ($m)"],
        rows, "Table II. My replication versus original paper",
        groups=[("", 1), ("My replication", 2), ("Original paper", 2), ("Difference", 2)],
    )
    return standalone, paired


def _table7_html(results, comparison):
    rows = table7_rows(results)
    standalone = _html_table(rows[0], rows[1:], "Table VII. My replication", t_rows=True)
    rows = []
    for _, row in comparison.sort_values("event_month").iterrows():
        means, tstats = [str(int(row["event_month"]))], [""]
        for source in ("replication", "paper"):
            means.extend([f"{row[f'monthly_return_{source}']:.4f}", f"{row[f'cumulative_return_{source}']:.4f}"])
            tstats.extend([f"({row[f'monthly_t_{source}']:.2f})", f"({row[f'cumulative_t_{source}']:.2f})"])
        means.extend([f"{row['monthly_return_diff_bp']:+.2f}", f"{row['cumulative_return_diff_bp']:+.2f}"])
        rows.extend([means, tstats + ["", ""]])
    paired = _html_table(
        ["Event month", "Monthly", "Cumulative", "Monthly", "Cumulative", "Monthly (bp)", "Cumulative (bp)"],
        rows, "Table VII. My replication versus original paper",
        groups=[("", 1), ("My replication", 2), ("Original paper", 2), ("Difference", 2)], t_rows=True,
    )
    return standalone, paired


def _table1_figure(comparison):
    wl = comparison.loc[comparison["leg"].eq("buy_sell")]
    figure = Figure(figsize=(13.2, 7), layout="constrained")
    axes = figure.subplots(2, 3)
    maximum = max(wl["mean_replication"].max(), wl["mean_paper"].max()) * 100
    difference_bound = max(wl["mean_diff_bp"].abs().max(), 0.01)
    for col, (field, scale, title) in enumerate([
        ("mean_replication", 100, "My replication (% per month)"),
        ("mean_paper", 100, "Original paper (% per month)"),
        ("mean_diff_bp", 1, "Replication minus paper (bp/month)"),
    ]):
        low, high = (-difference_bound, difference_bound) if col == 2 else (0, maximum)
        for i, panel in enumerate(("A", "B")):
            ax = axes[i, col]
            matrix = wl.loc[wl["panel"].eq(panel)].pivot(index="J", columns="K", values=field).sort_index().sort_index(axis=1) * scale
            im = ax.imshow(matrix, cmap="RdBu_r" if col == 2 else "YlGnBu", vmin=low, vmax=high, aspect="auto")
            ax.set_xticks(range(4), labels=matrix.columns)
            ax.set_yticks(range(4), labels=matrix.index)
            ax.set(xlabel="Holding months K", ylabel=f"Panel {panel}\nFormation months J")
            if i == 0:
                ax.set_title(title, fontsize=11, pad=12)
            for r in range(4):
                for c in range(4):
                    value = matrix.iloc[r, c]
                    red, green, blue, _ = im.cmap(im.norm(value))
                    color = "white" if 0.21 * red + 0.72 * green + 0.07 * blue < 0.5 else "#182b33"
                    ax.text(c, r, f"{value:+.1f}" if col == 2 else f"{value:.2f}", ha="center", va="center", color=color, fontsize=10)
        figure.colorbar(im, ax=list(axes[:, col]), location="bottom", shrink=0.9, pad=0.07)
    return _figure_html(figure, "Buy–Sell returns across all 32 strategies. Color scales are shared across panels; one basis point is 0.0001 in decimal returns.")


def _table2_figure(comparison):
    values = comparison.set_index("portfolio").reindex([f"P{i}" for i in range(1, 11)])
    figure = Figure(figsize=(12.8, 4.3), layout="constrained")
    axes = figure.subplots(1, 2)
    for ax, field, title, ylabel in zip(axes, ["beta", "average_market_cap_millions"],
                                       ["Post-ranking market beta", "Average market capitalization"],
                                       ["Beta", "USD millions"]):
        for source, label, color, style in [("replication", "My replication", BLUE, "-"), ("paper", "Original paper", GRAY, "--")]:
            ax.plot(range(1, 11), values[f"{field}_{source}"], marker="o", markersize=4, color=color, linestyle=style, label=label)
        ax.set(title=title, ylabel=ylabel, xlabel="Past-return decile (P1 = losers; P10 = winners)")
        ax.set_xticks(range(1, 11))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.2)
        ax.legend(frameon=False)
    return _figure_html(figure, "Beta and size profiles for the ten (6,6) portfolios. Replication beta frequency and size timing are draft conventions.")


def _table7_figure(comparison):
    values = comparison.sort_values("event_month")
    figure = Figure(figsize=(12.8, 4.3), layout="constrained")
    axes = figure.subplots(1, 2)
    for ax, field, title, ylabel in zip(axes, ["monthly_return", "cumulative_return"],
                                       ["Monthly event-time returns", "Cumulative event-time returns"],
                                       ["Monthly W-L (%)", "Cumulative W-L (percentage points)"]):
        for source, label, color, style in [("replication", "My replication", BLUE, "-"), ("paper", "Original paper", GRAY, "--")]:
            ax.plot(values["event_month"], values[f"{field}_{source}"] * 100, color=color, linestyle=style, linewidth=1.8, label=label)
        ax.axhline(0, color="#a4aab0", linewidth=0.8)
        ax.axvline(12, color="#a4aab0", linewidth=0.8, linestyle=":")
        ax.set(title=title, ylabel=ylabel, xlabel="Months after formation", xlim=(1, 36))
        ax.set_xticks([1, 6, 12, 18, 24, 30, 36])
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.2)
        ax.legend(frameon=False)
    return _figure_html(figure, "Six-month-ranked winner-minus-loser cohorts followed for 36 months. Cumulative returns are additive, not compounded; the vertical line marks month 12.")


def generate_submission(result_dirs: dict[int, str | Path], *, paper_dir: str | Path = PAPER_DIR,
                        output_path: str | Path | None = None, exchange_runs=None, subperiod_runs=None,
                        ff3_result_dir: str | Path | None = None) -> Path:
    """Generate the replication, selected sample comparisons, and optional Q1-5.

    Run submission_jt1993.py to select existing runs. No replication, downloads,
    edits to source results, or separate metadata output occur during this build.
    """
    results, info = load_results(result_dirs)
    paper = load_paper_tables(paper_dir)
    comparisons = compare_tables(results, paper)
    exchange_cases = {label: load_results(dirs, original_sample=False)
                      for label, dirs in (exchange_runs or {}).items()}
    subperiod_cases = {label: load_results(dirs, original_sample=False)
                       for label, dirs in (subperiod_runs or {}).items()}
    exchange_section = render_case_section(exchange_cases, "exchanges") if exchange_cases else ""
    subperiod_section = render_case_section(subperiod_cases, "subperiods") if subperiod_cases else ""
    ff3_section = render_ff3_section(ff3_result_dir, result_dirs[1], info[1]) if ff3_result_dir is not None else ""
    question_scope = "Q1-1 & Q1-4"
    if exchange_cases and subperiod_cases:
        question_scope = "Q1-1–4"
    elif exchange_cases:
        question_scope = "Q1-1–2 & Q1-4"
    elif subperiod_cases:
        question_scope = "Q1-1 & Q1-3–4"
    if ff3_section:
        question_scope = "Q1-1–5" if exchange_cases and subperiod_cases else question_scope + " & Q1-5"
    extension_nav = ('<a href="#exchange-samples">Q1-2 · Exchanges</a>' if exchange_cases else "")
    extension_nav += ('<a href="#subperiods">Q1-3 · Subperiods</a>' if subperiod_cases else "")
    t1, c1 = _table1_html(results[1], paper[1])
    t2, c2 = _table2_html(results[2], comparisons[2])
    t7, c7 = _table7_html(results[7], comparisons[7])
    wl = comparisons[1].query("panel == 'A' and J == 6 and K == 6 and leg == 'buy_sell'").iloc[0]
    beta = comparisons[2].set_index("portfolio").loc["P10-P1"]
    event = comparisons[7].set_index("event_month")
    spread = comparisons[1].loc[comparisons[1]["leg"].eq("buy_sell")]
    significant = ((spread["mean_replication"] > 0) & (spread["t_replication"].abs() >= 1.96)).sum()
    values = {
        "question_scope": question_scope, "extension_nav": extension_nav,
        "exchange_section": exchange_section, "subperiod_section": subperiod_section,
        "ff3_section": ff3_section,
        "ff3_nav": '<a href="#ff3-regression">Q1-5 · FF3 alpha</a>' if ff3_section else "",
        "style": (Path(__file__).with_name("submission.css")).read_text(encoding="utf-8"),
        "table1": t1, "comparison1": c1, "figure1": _table1_figure(comparisons[1]),
        "table2": t2, "comparison2": c2, "figure2": _table2_figure(comparisons[2]),
        "table7": t7, "comparison7": c7, "figure7": _table7_figure(comparisons[7]),
        "wl_mean": f"{wl['mean_replication'] * 100:.2f}%",
        "wl_paper": f"{wl['mean_paper'] * 100:.2f}%",
        "wl_gap": f"{wl['mean_diff_bp']:+.2f}",
        "beta": f"{beta['beta_replication']:.2f}", "beta_paper": f"{beta['beta_paper']:.2f}",
        "month12": f"{event.loc[12, 'cumulative_return_replication'] * 100:.2f}%",
        "month12_paper": f"{event.loc[12, 'cumulative_return_paper'] * 100:.2f}%",
        "month36": f"{event.loc[36, 'cumulative_return_replication'] * 100:.2f}%",
        "month36_paper": f"{event.loc[36, 'cumulative_return_paper'] * 100:.2f}%",
        "n_positive": str(int(spread['mean_replication'].gt(0).sum())),
        "n_significant": str(int(significant)),
        "mean_abs_gap": f"{spread['mean_diff_bp'].abs().mean():.2f}",
        "n_cohorts": str(int(info[7]["n_cohorts"])),
        "formation_end": escape(pd.Timestamp(info[7]["formation_end"]).strftime("%B %Y")),
        "hac_lags": escape(str(info[7]["hac_lags"])),
    }
    template = Template(Path(__file__).with_name("jt1993_template.html").read_text(encoding="utf-8"))
    html = template.substitute(values)
    output_path = Path(output_path) if output_path is not None else ROOT / "results/jt1993/submission/submission.html"
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    if output_path.suffix.lower() != ".html":
        raise ValueError("The submission output must be an .html file.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path
