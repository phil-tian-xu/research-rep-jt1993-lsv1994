"""Compact views of all Table I strategies and the (6,6) return history.

Use common scales across panels, separate percent returns from basis-point
errors, and distinguish the paper from the saved notebook reference. Historical
cumulative W-L is an additive sum of monthly zero-cost returns. Rolling bands
use conventional monthly standard errors and describe overlapping windows.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from ._plots import save_figure, strategy_matrix


def _strategy_overview(summary, caption):
    columns = [("mean", 100, "Mean W-L return (%)", ".2f"),
               ("t", 1, "Conventional t-statistic", ".2f")]
    if "paper_mean" in summary:
        summary = summary.assign(paper_difference=(summary["mean"] - summary["paper_mean"]) * 10_000)
        columns.append(("paper_difference", 1, "Replication minus paper (bp/month)", "+.1f"))
    fig, axes = plt.subplots(2, len(columns), figsize=(5.1 * len(columns), 8.3), layout="constrained")
    for column, (field, multiplier, label, value_fmt) in enumerate(columns):
        values = summary.loc[summary["leg"].eq("buy_sell"), field].to_numpy(dtype=float) * multiplier
        finite = values[np.isfinite(values)]
        bound = max(float(np.max(np.abs(finite))) if finite.size else 0.0, 0.01)
        diverging = field == "paper_difference" or (finite.size and finite.min() < 0)
        cmap, low = ("RdBu_r", -bound) if diverging else ("YlGnBu", 0)
        for row, panel in enumerate(("A", "B")):
            ax = axes[row, column]
            matrix = strategy_matrix(summary, panel, field) * multiplier
            image = ax.imshow(matrix, cmap=cmap, vmin=low, vmax=bound, aspect="auto")
            ax.set_xticks(range(len(matrix.columns)), labels=matrix.columns)
            ax.set_yticks(range(len(matrix.index)), labels=matrix.index)
            ax.set(xlabel="Holding months K", ylabel=f"Panel {panel}\nRanking months J")
            if row == 0:
                ax.set_title(label, fontsize=11, pad=10)
            for i in range(len(matrix.index)):
                for j in range(len(matrix.columns)):
                    value = matrix.iloc[i, j]
                    if np.isfinite(value):
                        red, green, blue, _ = image.cmap(image.norm(value))
                        text_color = "white" if 0.21 * red + 0.72 * green + 0.07 * blue < 0.5 else "#17212b"
                        ax.text(j, i, format(value, value_fmt), ha="center", va="center", color=text_color, fontsize=11)
                        if field == "t" and abs(value) >= 1.96:
                            ax.add_patch(Rectangle((j - 0.46, i - 0.46), 0.92, 0.92,
                                                   fill=False, edgecolor="#202020", linewidth=1.2))
                    else:
                        ax.text(j, i, "NA", ha="center", va="center")
            ax.set_xticks(np.arange(len(matrix.columns) + 1) - 0.5, minor=True)
            ax.set_yticks(np.arange(len(matrix.index) + 1) - 0.5, minor=True)
            ax.grid(which="minor", color="white", linewidth=1)
            ax.tick_params(which="minor", bottom=False, left=False)
        fig.colorbar(image, ax=list(axes[:, column]), location="bottom", shrink=0.9, pad=0.06)
    title = "returns, inference and replication errors" if "paper_mean" in summary else "returns and inference"
    fig.suptitle(f"JT1993 Table I | {title}\n{caption}", fontsize=15)
    fig.supxlabel("Common color scales across panels. Outlined t cells: |t| >= 1.96 (conventional, unadjusted).", fontsize=9)
    return fig


def _portfolio_comparison(summary, caption):
    fig, axes = plt.subplots(2, 3, figsize=(15, 11.2), sharex=True, sharey=True)
    order = [(j, k) for j in sorted(summary["J"].unique()) for k in sorted(summary["K"].unique())]
    labels = [f"{j} / {k}" for j, k in order]
    positions = np.arange(len(order))
    bounds = [summary["mean"]]
    for field in ("paper_mean", "reference_mean"):
        if field in summary:
            bounds.append(summary[field])
    low = min(0, min(values.min() for values in bounds) * 100) - 0.04
    high = max(0, max(values.max() for values in bounds) * 100) + 0.08
    for row, panel in enumerate(("A", "B")):
        for column, (leg, title) in enumerate((("loser", "Sell (P1)"), ("winner", "Buy (P10)"), ("buy_sell", "Buy minus Sell"))):
            ax = axes[row, column]
            selected = summary.loc[summary["panel"].eq(panel) & summary["leg"].eq(leg)].set_index(["J", "K"]).reindex(order)
            for group in range(len(sorted(summary["J"].unique()))):
                if group % 2 == 0:
                    size = len(sorted(summary["K"].unique()))
                    ax.axhspan(group * size - 0.5, (group + 1) * size - 0.5, color="#f1f4f7", zorder=0)
            if "paper_mean" in selected:
                ax.hlines(positions, selected["paper_mean"] * 100, selected["mean"] * 100, color="#aab2bb", linewidth=1.3)
                ax.scatter(selected["paper_mean"] * 100, positions, marker="D", s=28, color="#4b5563", label="Paper", zorder=3)
            if "reference_mean" in selected:
                ax.scatter(selected["reference_mean"] * 100, positions, s=88, facecolors="none",
                           edgecolors="#d98418", linewidths=1.3, label="Notebook reference", zorder=4)
            ax.scatter(selected["mean"] * 100, positions, s=25, color="#176b9b", label="Our replication", zorder=5)
            ax.set_yticks(positions, labels=labels)
            ax.set_xlim(low, high)
            ax.grid(axis="x", color="#dce1e6", linewidth=0.6)
            ax.axvline(0, color="#8c969f", linewidth=0.8)
            ax.spines[["top", "right", "left"]].set_visible(False)
            if row == 0:
                ax.set_title(title, fontsize=12)
            if column == 0:
                ax.set_ylabel(f"Panel {panel}\nRanking J / holding K (months)")
    axes[0, 0].invert_yaxis()
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle(f"Table I | every Sell, Buy and Buy-Sell return\n{caption}", fontsize=15, y=0.99)
    fig.legend(handles, legend_labels, ncol=len(handles), loc="upper center", bbox_to_anchor=(0.5, 0.935), frameon=False)
    xlabel = "Mean monthly return (%)"
    if "reference_mean" in summary:
        xlabel += " | Hollow reference rings overlap the replication when results match."
    fig.supxlabel(xlabel, fontsize=10, y=0.012)
    fig.tight_layout(rect=(0, 0.035, 1, 0.95))
    return fig


def _six_by_six_history(monthly, caption):
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 8), sharex=True)
    checkpoint = monthly.loc[monthly["J"].eq(6) & monthly["K"].eq(6)]
    window = min(36, int(checkpoint.groupby("panel").size().min()))
    for panel, color, label in (("A", "#176b9b", "Panel A: no gap"), ("B", "#c77b14", "Panel B: five trading-day gap")):
        selected = checkpoint.loc[checkpoint["panel"].eq(panel)].sort_values("month")
        dates, returns = selected["month"], selected["buy_sell"]
        axes[0].plot(dates, returns.cumsum() * 100, color=color, linewidth=1.7, label=label)
        rolling = returns.rolling(window, min_periods=window)
        mean = rolling.mean() * 100
        band = 1.96 * rolling.std(ddof=1) / np.sqrt(window) * 100
        axes[1].plot(dates, mean, color=color, linewidth=1.6, label=label)
        axes[1].fill_between(dates, mean - band, mean + band, color=color, alpha=0.12, linewidth=0)
    axes[0].set(title="Cumulative W-L: additive sum of monthly zero-cost returns", ylabel="Cumulative W-L\n(percentage points)")
    axes[1].set(title=f"Trailing {window}-month mean W-L with pointwise +/-1.96 SE bands",
                ylabel="Mean monthly W-L (%)", xlabel="Calendar month")
    for ax in axes:
        ax.axhline(0, color="#8c969f", linewidth=0.8, linestyle="--")
        ax.grid(axis="y", color="#dce1e6", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, loc="upper left", fontsize=9)
    axes[1].set_xlim(checkpoint["month"].min(), checkpoint["month"].max())
    fig.suptitle(f"The (6,6) strategy | realized history and changing average returns\n{caption}", fontsize=14, y=0.99)
    fig.text(0.5, 0.013, "Bands use monthly sample standard deviations / sqrt(window); no HAC. Rolling windows overlap.",
             ha="center", fontsize=9, color="#52606d")
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    return fig


def plot_overview(summary, monthly, output_dir, caption, show=False):
    """Save three views: strategy grids, all-portfolio comparisons, and history."""
    paths = {}
    with plt.rc_context({"font.size": 10, "axes.titleweight": "bold", "axes.labelcolor": "#26323d"}):
        for name, figure in (
            ("strategy_overview", _strategy_overview(summary, caption)),
            ("portfolio_comparison", _portfolio_comparison(summary, caption)),
            ("six_by_six_history", _six_by_six_history(monthly, caption)),
        ):
            paths[name] = save_figure(figure, output_dir, name, show)
    return paths
