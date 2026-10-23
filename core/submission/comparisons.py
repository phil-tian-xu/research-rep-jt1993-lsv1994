"""Exchange and subperiod sections from already validated numerical runs."""

from html import escape
import json

from matplotlib.figure import Figure
import numpy as np
import pandas as pd

from core.jt1993.reporter.table1.tables import table1_rows
from core.jt1993.reporter.table2 import table2_rows
from core.jt1993.reporter.table7 import table7_rows
from ._html import html_table, figure_html


COLORS = ("#16766d", "#687180", "#b47a32")


def _validate_cases(cases, kind):
    if len(cases) < 2 or (kind == "subperiods" and len(cases) != 2):
        raise ValueError("Exchange comparisons need at least two samples; subperiod comparisons need two periods.")
    windows, universes, methods = [], [], []
    for _, info in cases.values():
        metadata = info[1]["raw_inputs"]["monthly_returns"]["download_metadata"]
        windows.append((info[1]["run_start_date"], info[1]["run_end_date"]))
        universes.append(metadata["universe"])
        methods.append(json.dumps({
            "source": metadata.get("data_source", "legacy"),
            "source_delisting": metadata.get("delisting_returns_in_source", False),
            "table1": info[1].get("settings"), "beta": info[2].get("beta_method"),
            "size": info[2].get("size_method"), "weights": info[7].get("stock_weights"),
            "cumulative": info[7].get("cumulative_method"),
            "inference": info[7].get("cumulative_inference"), "hac_lags": info[7].get("hac_lags"),
        }, sort_keys=True))
    if len(set(methods)) != 1:
        raise ValueError("Compared runs must use the same data source and portfolio/inference conventions.")
    if kind == "exchanges":
        if len(set(windows)) != 1 or len(set(universes)) != len(cases):
            raise ValueError("Exchange comparisons require the same reporting dates and distinct universes.")
    elif kind == "subperiods":
        if len(set(universes)) != 1:
            raise ValueError("Subperiod comparisons require the same exchange universe.")
        if pd.Period(windows[0][1], freq="M") + 1 != pd.Period(windows[1][0], freq="M"):
            raise ValueError("Subperiods must be chronological, nonoverlapping, and adjacent.")
    else:
        raise ValueError(f"Unknown comparison: {kind}.")


def _comparison_tables(cases, section_id, kind):
    indexed = {label: {table: frame.set_index(
        ["panel", "J", "K", "leg"] if table == 1 else "portfolio" if table == 2 else "event_month"
    ) for table, frame in tables.items()} for label, (tables, _) in cases.items()}
    labels = list(cases)
    rows = []
    for panel in ("A", "B"):
        for J in (3, 6, 9, 12):
            for K in (3, 6, 9, 12):
                values = [indexed[label][1].loc[(panel, J, K, "buy_sell")] for label in labels]
                means = [panel, J, K] + [f"{row['mean']:.4f}" for row in values]
                tstats = ["", "", ""] + [f"({row['t']:.2f})" for row in values]
                if kind == "subperiods":
                    means.append(f"{(values[1]['mean'] - values[0]['mean']) * 10_000:+.2f}")
                    tstats.append("")
                rows.extend([means, tstats])
    headers = ["Panel", "J", "K"] + labels + (["Recent − early (bp)"] if kind == "subperiods" else [])
    table1 = html_table(headers, rows, f"{section_id} · Table I. Buy–Sell comparison", t_rows=True)
    rows = []
    for portfolio in [*(f"P{i}" for i in range(1, 11)), "P10-P1"]:
        row = [portfolio]
        for label in labels:
            record = indexed[label][2].loc[portfolio]
            row.extend([f"{record['beta']:.2f}", "—" if portfolio == "P10-P1"
                        else f"{record['average_market_cap_millions']:.2f}"])
        rows.append(row)
    table2 = html_table(["Portfolio"] + ["Beta", "Size ($m)"] * len(labels), rows,
                        f"{section_id} · Table II. Beta and size comparison",
                        groups=[("", 1)] + [(label, 2) for label in labels])
    rows = []
    for month in range(1, 37):
        means, tstats = [month], [""]
        for label in labels:
            record = indexed[label][7].loc[month]
            means.extend([f"{record['monthly_return']:.4f}", f"{record['cumulative_return']:.4f}"])
            tstats.extend([f"({record['monthly_t']:.2f})", f"({record['cumulative_t']:.2f})"])
        rows.extend([means, tstats])
    table7 = html_table(["Event month"] + ["Monthly", "Cumulative"] * len(labels), rows,
                        f"{section_id} · Table VII. Event-time comparison",
                        groups=[("", 1)] + [(label, 2) for label in labels], t_rows=True)
    return table1, table2, table7


def _return_figure(cases):
    figure = Figure(figsize=(4.4 * len(cases), 7), layout="constrained")
    axes = figure.subplots(2, len(cases), squeeze=False)
    spreads = {label: tables[1].loc[tables[1]["leg"].eq("buy_sell")] for label, (tables, _) in cases.items()}
    all_returns = np.concatenate([frame["mean"].to_numpy() * 100 for frame in spreads.values()])
    bound = max(float(np.abs(all_returns).max()), 0.01)
    low = -bound if all_returns.min() < 0 else 0
    for column, (label, frame) in enumerate(spreads.items()):
        for row, panel in enumerate(("A", "B")):
            ax = axes[row, column]
            matrix = frame.loc[frame["panel"].eq(panel)].pivot(index="J", columns="K", values="mean").sort_index().sort_index(axis=1) * 100
            im = ax.imshow(matrix, vmin=low, vmax=bound, cmap="RdBu_r" if low < 0 else "YlGnBu", aspect="auto")
            ax.set_xticks(range(4), labels=matrix.columns)
            ax.set_yticks(range(4), labels=matrix.index)
            ax.set(xlabel="Holding months K", ylabel=f"Panel {panel}\nFormation months J")
            if row == 0:
                ax.set_title(label, fontsize=11)
            for i in range(4):
                for j in range(4):
                    value = matrix.iloc[i, j]
                    red, green, blue, _ = im.cmap(im.norm(value))
                    color = "white" if 0.21 * red + 0.72 * green + 0.07 * blue < 0.5 else "#203337"
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=color, fontsize=10)
    figure.colorbar(im, ax=list(axes.flat), location="bottom", shrink=0.7, pad=0.06, label="Mean monthly Buy–Sell return (%)")
    return figure_html(figure, "All 32 momentum strategies, with a common return scale across samples and both panels.")


def _characteristic_figure(cases):
    figure = Figure(figsize=(12.8, 4.3), layout="constrained")
    axes = figure.subplots(1, 2)
    for ax, field, title, ylabel in zip(axes, ["beta", "average_market_cap_millions"],
                                       ["Post-ranking beta", "Average market capitalization"], ["Beta", "Nominal USD millions"]):
        for i, (label, (tables, _)) in enumerate(cases.items()):
            values = tables[2].set_index("portfolio").reindex([f"P{j}" for j in range(1, 11)])
            ax.plot(range(1, 11), values[field], color=COLORS[i % len(COLORS)], marker="o", markersize=3, label=label)
        ax.set(title=title, ylabel=ylabel, xlabel="Past-return decile (P1 = losers; P10 = winners)")
        ax.set_xticks(range(1, 11))
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=9)
    return figure_html(figure, "Beta and size profiles under the same draft construction conventions. Size is nominal; levels across periods also reflect changes in prices and the firm population.")


def _event_figure(cases):
    figure = Figure(figsize=(12.8, 4.3), layout="constrained")
    axes = figure.subplots(1, 2)
    for ax, field, title, ylabel in zip(axes, ["monthly_return", "cumulative_return"],
                                       ["Monthly event-time returns", "Cumulative event-time returns"],
                                       ["Monthly W-L (%)", "Cumulative W-L (percentage points)"]):
        for i, (label, (tables, _)) in enumerate(cases.items()):
            values = tables[7].sort_values("event_month")
            ax.plot(values["event_month"], values[field] * 100, color=COLORS[i % len(COLORS)], label=label)
        ax.axhline(0, color="#a4aab0", linewidth=0.8)
        ax.axvline(12, color="#a4aab0", linewidth=0.8, linestyle=":")
        ax.set(title=title, ylabel=ylabel, xlabel="Months after formation", xlim=(1, 36))
        ax.set_xticks([1, 6, 12, 18, 24, 30, 36])
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=9)
    return figure_html(figure, "Six-month-ranked cohorts followed for 36 months within each saved window. Cumulative returns sum monthly spreads; they are not compounded.")


def _complete_tables(cases, section_id):
    options, panels = [], []
    for i, (label, (tables, info)) in enumerate(cases.items()):
        case_id = f"{section_id}-case-{i}"
        options.append(f'<option value="{case_id}">{escape(label)}</option>')
        prefix = f"Q1-{2 if section_id == 'exchange-samples' else 3} · {label}"
        t1 = table1_rows(tables[1])
        columns = [0, 1, 3, 4, 5, 6, 8, 9, 10, 11]
        display1 = html_table(["J", "Portfolio", "3", "6", "9", "12", "3", "6", "9", "12"],
                              [[row[j] for j in columns] for row in t1[2:]], prefix + " · Complete Table I",
                              groups=[("", 2), ("Panel A: K months", 4), ("Panel B: K months", 4)], t_rows=True)
        t2, t7 = table2_rows(tables[2]), table7_rows(tables[7])
        display2 = html_table(t2[0], t2[1:], prefix + " · Complete Table II")
        display7 = html_table(t7[0], t7[1:], prefix + " · Complete Table VII", t_rows=True)
        start, end = pd.Timestamp(info[1]["run_start_date"]), pd.Timestamp(info[1]["run_end_date"])
        first, last = pd.Timestamp(info[7]["formation_start"]), pd.Timestamp(info[7]["formation_end"])
        hidden = " hidden" if i else ""
        panels.append(f'<article class="case-panel" id="{case_id}"{hidden}><h4>{escape(label)}</h4>'
                      f'<p class="table-note">Reported window: {start:%B %Y}–{end:%B %Y}. '
                      f'{int(tables[1]["n"].iloc[0])} return months. Table VII: {info[7]["n_cohorts"]} '
                      f'complete cohorts formed {first:%B %Y}–{last:%B %Y}.</p>'
                      f'{display1}{display2}{display7}</article>')
    return (f'<details class="full-results" open><summary>Complete Tables I, II, and VII for each sample</summary>'
            f'<div class="case-picker"><label for="{section_id}-select">Show results for</label>'
            f'<select class="case-select" id="{section_id}-select" data-group="{section_id}-cases">'
            + "".join(options) + f'</select></div><div id="{section_id}-cases">'
            + "".join(panels) + '</div></details>')


def _findings(cases, kind):
    pieces = []
    for panel in ("A", "B"):
        estimates = []
        for label, (tables, _) in cases.items():
            row = tables[1].set_index(["panel", "J", "K", "leg"]).loc[(panel, 6, 6, "buy_sell")]
            estimates.append(f"{escape(label)}: {row['mean'] * 100:.2f}% (t = {row['t']:.2f})")
        pieces.append(f"(6,6), Panel {panel} — " + "; ".join(estimates) + ".")
    if kind == "subperiods":
        early, recent = [tables[1] for tables, _ in cases.values()]
        counts = [int(((frame["leg"] == "buy_sell") & (frame["mean"] > 0) & (frame["t"].abs() >= 1.96)).sum())
                  for frame in (early, recent)]
        pieces.append(f"Positive returns with conventional |t| ≥ 1.96 occur in {counts[0]} of 32 strategies "
                      f"in the early period and {counts[1]} in the recent period. These are unadjusted individual tests.")
        recent66 = recent.loc[recent["leg"].eq("buy_sell") & recent["J"].eq(6) & recent["K"].eq(6)]
        if recent66["mean"].gt(0).all() and recent66["t"].abs().lt(1.96).all():
            pieces.append("The recent (6,6) spread remains positive in both panels, but neither estimate meets the conventional two-sided 5% threshold.")
        pieces.append("Differences in means are descriptive; no formal test of the change between periods is reported.")
    return "".join(f"<p>{piece}</p>" for piece in pieces)


def render_case_section(cases, kind):
    """Render all tables and graphs without borrowing paper benchmarks for new samples."""
    _validate_cases(cases, kind)
    section_id = "exchange-samples" if kind == "exchanges" else "subperiods"
    label = "Q1-2 / Exchange samples" if kind == "exchanges" else "Q1-3 / Subperiods"
    title = "Momentum across exchange samples" if kind == "exchanges" else "Does momentum persist in the recent period?"
    windows = []
    for name, (_, info) in cases.items():
        start, end = pd.Timestamp(info[1]["run_start_date"]), pd.Timestamp(info[1]["run_end_date"])
        universe = info[1]["raw_inputs"]["monthly_returns"]["download_metadata"]["universe"]
        universe_label = {"nyse": "NYSE", "nyse_amex": "NYSE/AMEX", "nyse_amex_nasdaq": "NYSE/AMEX/Nasdaq"}[universe]
        display_name = escape(name) if name == universe_label else f"{escape(name)} ({universe_label})"
        windows.append(f"{display_name}: {start:%B %Y}–{end:%B %Y}")
    coverage = "; ".join(windows) + "."
    if kind == "exchanges":
        scope = ("These are matched-window exchange comparisons. The homework requests 1962–present; "
                 "the saved exchange runs included here cover the dates above, rather than that full span.")
    else:
        scope = ("These are the two saved reporting periods for the same exchange universe. Only the dates "
                 "above are included; the requested 1962–present span is not assumed from raw-data coverage.")
    tables = _comparison_tables(cases, "Q1-2" if kind == "exchanges" else "Q1-3", kind)
    figures = (_return_figure(cases), _characteristic_figure(cases), _event_figure(cases))
    body = []
    for heading, table, figure in zip(["Table I · Buy–Sell returns", "Table II · Beta and size", "Table VII · Event-time returns"], tables, figures):
        body.append(f"<h3>{heading}</h3>{table}{figure}")
    return (f'<section id="{section_id}"><div class="section-label">{label}</div><h2>{title}</h2>'
            f'<p>{coverage}</p><p class="table-note">{scope}</p>'
            f'<div class="finding">{_findings(cases, kind)}</div>'
            '<p class="table-note">Returns in tables are decimals, with t-statistics in parentheses. '
            'Table II size is in nominal USD millions. Tables II and VII retain the draft beta/size, '
            'complete-cohort endpoint, and inference conventions described in Q1-1. '
            'Only Q1-1 compares with published paper numbers.</p>'
            + "".join(body) + _complete_tables(cases, section_id) + "</section>")
