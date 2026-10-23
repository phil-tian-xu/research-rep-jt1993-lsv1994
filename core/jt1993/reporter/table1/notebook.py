"""Reproduce the eleven plots in the original Table I reference notebook.

Keep its plot sizes, default colors, individual heatmap scales, rolling-window
conventions, and units for a direct visual comparison. Plotting reads saved
results only. Paper-comparison plots are omitted outside the original sample.
"""

import matplotlib.pyplot as plt
import numpy as np

from ._plots import save_figure, strategy_matrix


def _heatmap(data, title, value_fmt):
    values = data.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    image = ax.imshow(values, aspect="auto")
    ax.set_xticks(np.arange(len(data.columns)), labels=data.columns)
    ax.set_yticks(np.arange(len(data.index)), labels=data.index)
    ax.set(xlabel="K", ylabel="J", title=title)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            if np.isfinite(values[row, column]):
                ax.text(column, row, format(values[row, column], value_fmt), ha="center", va="center")
    fig.colorbar(image, ax=ax, shrink=0.82)
    fig.tight_layout()
    return fig


def plot_notebook(summary, monthly, formations, coverage, output_dir, sample_label, show=False):
    """Save notebook-style coverage, signal, time-series, heatmap and bar plots."""
    paths = {}
    with plt.rc_context(plt.rcParamsDefault):
        fig, ax = plt.subplots(figsize=(10, 4.6))
        ax.plot(coverage["month"], coverage["n_stocks"], linewidth=1.4)
        ax.set(title=f"{sample_label} in the Monthly Sample", xlabel="Calendar month", ylabel="Number of stocks")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        paths["sample_coverage"] = save_figure(fig, output_dir, "sample_coverage", show)

        fig, ax = plt.subplots(figsize=(10, 4.6))
        ax.plot(coverage["month"], coverage["mean_ret"], linewidth=0.9, label="Monthly mean")
        ax.plot(coverage["month"], coverage["mean_ret"].rolling(12, min_periods=6).mean(),
                linewidth=1.5, label="12-month moving average")
        ax.axhline(0, linewidth=0.8, linestyle="--")
        ax.set(title="Equal-Weighted Mean Return of Sample Stocks", xlabel="Calendar month", ylabel="Return")
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        paths["sample_mean_return"] = save_figure(fig, output_dir, "sample_mean_return", show)

        selected = formations.loc[formations["panel"].eq("A") & formations["J"].eq(6)].copy()
        # Weight each formation-decile mean by its stock count, matching the
        # notebook's mean across all individual stock-formation signals.
        selected["signal_sum"] = selected["mean_signal"] * selected["n_stocks"]
        totals = selected.groupby("momr")[["signal_sum", "n_stocks"]].sum()
        signals = (totals["signal_sum"] / totals["n_stocks"]).reindex(range(1, 11))
        fig, ax = plt.subplots(figsize=(8.5, 4.6))
        ax.bar(signals.index.astype(str), signals.values)
        ax.set(title="Average Six-Month Ranking Return by Formation Decile",
               xlabel="Momentum decile: P1 losers to P10 winners", ylabel="Average ranking return")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        paths["ranking_signal_deciles"] = save_figure(fig, output_dir, "ranking_signal_deciles", show)

        checkpoint = monthly.loc[monthly["panel"].eq("A") & monthly["J"].eq(6) & monthly["K"].eq(6)]
        fig, ax = plt.subplots(figsize=(10, 4.8))
        ax.plot(checkpoint["month"], checkpoint["buy_sell"], linewidth=0.8, label="Monthly W-L")
        ax.plot(checkpoint["month"], checkpoint["buy_sell"].rolling(12, min_periods=6).mean(),
                linewidth=1.5, label="12-month moving average")
        ax.axhline(0, linewidth=0.8, linestyle="--")
        ax.set(title="Panel A: J=6, K=6 Monthly Momentum Return", xlabel="Calendar month", ylabel="Return per dollar long")
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        paths["panel_a_6x6_history"] = save_figure(fig, output_dir, "panel_a_6x6_history", show)

        for panel in ("A", "B"):
            data = strategy_matrix(summary, panel, "mean") * 100
            fig = _heatmap(data, f"Panel {panel} Replicated W-L Return (%)", ".2f")
            name = f"panel_{panel.lower()}_wl_return"
            paths[name] = save_figure(fig, output_dir, name, show)
            if "paper_mean" in summary:
                difference = (strategy_matrix(summary, panel, "mean")
                              - strategy_matrix(summary, panel, "paper_mean")) * 10_000
                fig = _heatmap(difference, f"Panel {panel}: Replication Minus Paper (bp/month)", ".1f")
                name = f"panel_{panel.lower()}_paper_difference"
                paths[name] = save_figure(fig, output_dir, name, show)

        gap_effect = (strategy_matrix(summary, "B", "mean") - strategy_matrix(summary, "A", "mean")) * 10_000
        fig = _heatmap(gap_effect, "Effect of the Skipped Week: Panel B Minus Panel A (bp/month)", ".1f")
        paths["skipped_week_effect"] = save_figure(fig, output_dir, "skipped_week_effect", show)

        if "paper_mean" in summary:
            for panel in ("A", "B"):
                selected = summary.loc[summary["panel"].eq(panel) & summary["leg"].eq("buy_sell")].sort_values(["J", "K"])
                labels = [f"J{j}/K{k}" for j, k in zip(selected["J"], selected["K"])]
                positions, width = np.arange(len(selected)), 0.38
                fig, ax = plt.subplots(figsize=(12, 5.0))
                ax.bar(positions - width / 2, selected["paper_mean"] * 100, width, label="Paper")
                ax.bar(positions + width / 2, selected["mean"] * 100, width, label="Replication")
                ax.set_xticks(positions, labels=labels, rotation=45, ha="right")
                ax.set(ylabel="Mean monthly W-L return (%)", title=f"Panel {panel}: Paper versus Replication")
                ax.legend(frameon=False)
                ax.spines[["top", "right"]].set_visible(False)
                fig.tight_layout()
                name = f"panel_{panel.lower()}_paper_comparison"
                paths[name] = save_figure(fig, output_dir, name, show)
    return paths
