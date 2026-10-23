"""Small examples for timing, compounding, and overlapping portfolio weights."""

import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from core.jt1993.replication.table1 import (
    build_cohort_returns,
    make_panel_a_ranks,
    make_panel_b_ranks,
    momentum_deciles,
    panel_b_windows,
    summarize_strategy,
)


class RankingTests(unittest.TestCase):
    def test_ties_stay_together_and_small_cross_sections_are_ineligible(self):
        signals = pd.Series([0.0] * 10 + [0.1] * 10 + [np.nan])
        deciles = momentum_deciles(signals)
        self.assertTrue(deciles.iloc[:10].eq(3).all())
        self.assertTrue(deciles.iloc[10:20].eq(8).all())
        self.assertTrue(pd.isna(deciles.iloc[20]))
        self.assertTrue(momentum_deciles(pd.Series(range(9))).isna().all())

    def test_monthly_signals_need_consecutive_valid_history_and_exclude_future_returns(self):
        monthly = pd.concat([
            pd.DataFrame({"permno": i, "month_id": range(1, 7), "ret": 0.001 * (i + 1)})
            for i in range(20)
        ], ignore_index=True)
        monthly.loc[(monthly.permno == 0) & (monthly.month_id == 2), "ret"] = -1.0
        monthly = monthly.loc[~((monthly.permno == 1) & (monthly.month_id == 2))].copy()
        monthly.loc[(monthly.permno == 2) & (monthly.month_id == 3), "ret"] = np.nan
        rank = make_panel_a_ranks(monthly, J=3, formation_start=4, formation_end=7)
        first = rank.loc[rank.formation_id.eq(4)].set_index("permno")
        self.assertEqual(first.loc[0, "signal"], -1.0)
        self.assertNotIn(1, first.index)  # An absent month is not a zero return.
        self.assertNotIn(2, first.index)
        self.assertAlmostEqual(first.loc[3, "signal"], (1.004 ** 3) - 1)
        self.assertIn(1, rank.loc[rank.formation_id.eq(6), "permno"].to_list())
        self.assertAlmostEqual(
            rank.loc[(rank.permno == 0) & (rank.formation_id == 6), "signal"].iloc[0],
            1.001 ** 3 - 1,
        )  # -100% does not poison windows after it has rolled out.
        changed = monthly.copy()
        changed.loc[changed.month_id.ge(4), "ret"] = 0.9
        future_rank = make_panel_a_ranks(changed, 3, 4, 4).set_index("permno")
        assert_frame_equal(first, future_rank)

    def test_daily_signal_excludes_boundary_gap_and_future_returns(self):
        days = pd.bdate_range("2019-11-01", "2020-02-28").to_numpy()
        formation_id = 2020 * 12 + 2
        windows = panel_b_windows(days, 1, formation_id, formation_id)
        window = windows.iloc[0]
        self.assertEqual(window.hold_start, pd.Timestamp("2020-02-03"))
        self.assertEqual(window.signal_end, pd.Timestamp("2020-01-24"))
        self.assertEqual(window.signal_start_boundary, pd.Timestamp("2019-12-24"))
        self.assertEqual(window.skipped_days.split(", "), [
            "2020-01-27", "2020-01-28", "2020-01-29", "2020-01-30", "2020-01-31",
        ])
        daily = pd.concat([
            pd.DataFrame({"permno": i, "date": days, "ret": 0.0001 * (i + 1)})
            for i in range(20)
        ], ignore_index=True)
        rank = make_panel_b_ranks(daily, windows).set_index("permno")
        n = ((days > window.signal_start_boundary) & (days <= window.signal_end)).sum()
        self.assertAlmostEqual(rank.loc[0, "signal"], 1.0001 ** n - 1)
        changed = daily.copy()
        outside = changed.date.le(window.signal_start_boundary) | changed.date.gt(window.signal_end)
        changed.loc[outside, "ret"] = 0.9
        assert_frame_equal(rank, make_panel_b_ranks(changed, windows).set_index("permno"))

        inside = daily.date.gt(window.signal_start_boundary) & daily.date.le(window.signal_end)
        daily.loc[(daily.permno == 0) & daily.date.eq(window.signal_end), "ret"] = -1.0
        daily.loc[(daily.permno == 1) & inside, "ret"] = np.nan
        daily = daily.loc[~((daily.permno == 2) & daily.date.le(window.signal_start_boundary))]
        eligible = make_panel_b_ranks(daily, windows).set_index("permno")
        self.assertEqual(eligible.loc[0, "signal"], -1.0)
        self.assertNotIn(1, eligible.index)  # All missing inside the signal window.
        self.assertNotIn(2, eligible.index)  # Listing history does not span the start.


class HoldingReturnTests(unittest.TestCase):
    def setUp(self):
        self.rank = pd.DataFrame({
            "permno": [1, 2, 3, 4, 5], "formation_id": [10, 10, 11, 10, 11],
            "momr": [1, 1, 1, 10, 10],
        })
        self.monthly = pd.DataFrame({
            "permno": [1, 2, 4, 1, 2, 3, 4, 5],
            "month_id": [10, 10, 10, 11, 11, 11, 11, 11],
            "ret": [0.9, 0.1, 0.2, 0.1, 0.3, 0.8, 0.6, 0.2],
        })

    def test_rebalance_stocks_within_cohorts_then_equal_weight_cohorts(self):
        cohort = build_cohort_returns(self.rank, self.monthly, max_k=2)
        old_loser = cohort.loc[
            cohort.formation_id.eq(10) & cohort.hold_id.eq(11) & cohort.momr.eq(1)
        ].iloc[0]
        self.assertAlmostEqual(old_loser.cohort_ret, 0.2)  # No buy-and-hold drift.
        _, ts, counts = summarize_strategy(cohort, 2, 11, 11)
        self.assertAlmostEqual(ts.loc[11, "loser"], 0.5)  # mean(0.2, 0.8), not pooled stocks.
        self.assertAlmostEqual(ts.loc[11, "winner"], 0.4)
        self.assertAlmostEqual(ts.loc[11, "buy_sell"], -0.1)
        self.assertTrue(counts.loc[11].eq(2).all())

    def test_missing_stock_returns_are_excluded_but_missing_cohorts_are_not_zero(self):
        monthly = self.monthly.copy()
        monthly.loc[(monthly.permno == 1) & (monthly.month_id == 11), "ret"] = np.nan
        cohort = build_cohort_returns(self.rank, monthly, max_k=2)
        _, ts, _ = summarize_strategy(cohort, 2, 11, 11)
        self.assertAlmostEqual(ts.loc[11, "loser"], (0.3 + 0.8) / 2)
        monthly.loc[monthly.permno.eq(3), "ret"] = np.nan
        cohort = build_cohort_returns(self.rank, monthly, max_k=2)
        stats, ts, counts = summarize_strategy(cohort, 2, 11, 11)
        self.assertTrue(pd.isna(ts.loc[11, "loser"]))
        self.assertTrue(pd.isna(ts.loc[11, "buy_sell"]))
        self.assertEqual(stats.loc["buy_sell", "n"], 0)
        self.assertEqual(counts.loc[11, 1], 1)

    def test_conventional_t_statistic_uses_monthly_portfolio_returns(self):
        cohort = pd.DataFrame({
            "formation_id": [1, 2, 3] * 2, "hold_id": [1, 2, 3] * 2, "age": 0,
            "momr": [1] * 3 + [10] * 3, "cohort_ret": [0.0] * 3 + [0.01, 0.02, -0.01],
        })
        stats, _, _ = summarize_strategy(cohort, 1, 1, 3)
        returns = np.array([0.01, 0.02, -0.01])
        self.assertAlmostEqual(stats.loc["buy_sell", "mean"], returns.mean())
        self.assertAlmostEqual(
            stats.loc["buy_sell", "t"], returns.mean() / (returns.std(ddof=1) / np.sqrt(3)),
        )
        self.assertEqual(stats.loc["buy_sell", "n"], 3)
        self.assertTrue(pd.isna(stats.loc["loser", "t"]))


if __name__ == "__main__":
    unittest.main()
