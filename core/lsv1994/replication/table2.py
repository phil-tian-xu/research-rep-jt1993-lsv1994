"""Table II: independent 30/40/30 sorts and nine-portfolio return panels."""

import numpy as np
import pandas as pd

from ._formation import ntile_series
from .table1 import STATS


PAIRS = {"C/P-GS": ("cp", "gs"), "E/P-GS": ("ep", "gs"),
         "B/M-GS": ("bm", "gs"), "E/P-B/M": ("ep", "bm"), "B/M-C/P": ("bm", "cp")}


def make_pair_constituents(formation):
    keys = ["formation_year", "permno"]
    panels = []
    for strategy, signals in PAIRS.items():
        frame = formation[keys + list(signals)].dropna().copy()
        for signal in signals:
            if signal in ("cp", "ep"):
                frame = frame.loc[frame[signal].gt(0)].copy()
        # Both independent ranks use the pair's common eligible sample.
        # Collapse SAS deciles 1-3/4-7/8-10, preserving the Table I tie rule.
        for i, signal in enumerate(signals, 1):
            decile = frame.groupby("formation_year")[signal].transform(ntile_series)
            frame[f"group{i}"] = pd.Series(np.select([decile.le(3).fillna(False).to_numpy(dtype=bool), decile.le(7).fillna(False).to_numpy(dtype=bool)],
                                                      [1, 2], default=3), index=frame.index).where(decile.notna())
        frame = frame.dropna(subset=["group1", "group2"])
        frame["portfolio"] = ((frame["group1"] - 1) * 3 + frame["group2"]).astype(int)
        frame["strategy"] = strategy
        panels.append(frame[keys + ["strategy", "portfolio"]])
    return pd.concat(panels, ignore_index=True)


def summarize_table2(yearly, formation_start, formation_end, return_end):
    rows = []
    last_year = pd.Timestamp(return_end).year - 1
    for strategy in PAIRS:
        frame = yearly.loc[yearly["strategy"].eq(strategy)]
        for portfolio in range(1, 10):
            cell = frame.loc[frame["portfolio"].eq(portfolio)]
            counts = cell.groupby("hold_year")[["ret", "saar"]].count().reindex(range(1, 6), fill_value=0)
            expected = [max(0, min(formation_end, last_year - h + 1) - formation_start + 1) for h in range(1, 6)]
            if min(expected) < 1 or not counts.eq(expected, axis=0).all().all():
                raise ValueError(f"Table II: incomplete {strategy}, portfolio {portfolio} cohort/horizon coverage.")
            raw = cell.groupby("hold_year")["ret"].mean().reindex(range(1, 6))
            adjusted = cell.groupby("hold_year")["saar"].mean().reindex(range(1, 6))
            values = [*raw, raw.mean(), (1 + raw).prod(min_count=5) - 1, adjusted.mean()]
            for stat, value in zip(STATS, values):
                rows.append({"strategy": strategy, "stat": stat, "group1": (portfolio-1)//3+1,
                             "group2": (portfolio-1)%3+1, "return": value})
    return pd.DataFrame(rows)
