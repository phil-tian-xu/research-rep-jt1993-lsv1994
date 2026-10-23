"""Shared figure saving and strategy-grid ordering for the two plot sets."""

from pathlib import Path

import matplotlib.pyplot as plt


def save_figure(figure, output_dir, name, show=False):
    """Save a PNG; retain its figure only when interactive display is requested."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.png"
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    if not show:
        plt.close(figure)
    return path


def strategy_matrix(summary, panel, column):
    """Put J on rows and K on columns, both in ascending month order."""
    selected = summary.loc[summary["panel"].eq(panel) & summary["leg"].eq("buy_sell")]
    return selected.pivot(index="J", columns="K", values=column).reindex(
        index=sorted(summary["J"].unique()), columns=sorted(summary["K"].unique()),
    )
