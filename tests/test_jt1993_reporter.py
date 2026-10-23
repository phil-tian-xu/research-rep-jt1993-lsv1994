"""Check plot units/weighting, stale-output detection, and report-only rendering."""

import csv
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core.jt1993.reporter.table1 import _load_results, report_table1
from core.jt1993.reporter.table1.notebook import plot_notebook
from core.jt1993.reporter.table1.overview import _six_by_six_history, _strategy_overview
from core.jt1993.reporter.table1.tables import table1_rows, write_table1
from utils.path_info import root_path


class ReporterTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix=".test_jt1993_", dir=Path(root_path))
        self.addCleanup(temporary.cleanup)
        self.addCleanup(plt.close, "all")
        self.result_dir = Path(temporary.name)
        dates = pd.date_range("1965-01-31", "1989-12-31", freq="ME")
        wave = np.sin(np.arange(len(dates))) * 0.01
        rows, series = [], []
        for panel in ("A", "B"):
            for J in (3, 6, 9, 12):
                for K in (3, 6, 9, 12):
                    loser = 0.004 + J * 0.0001 + wave
                    spread = 0.002 + K * 0.0003 + (panel == "B") * 0.0002 + wave * 0.5
                    frame = pd.DataFrame({
                        "month": dates, "loser": loser, "winner": loser + spread,
                        "buy_sell": spread, "panel": panel, "J": J, "K": K,
                    })
                    series.append(frame)
                    for leg in ("loser", "winner", "buy_sell"):
                        mean = frame[leg].mean()
                        t_stat = mean / (frame[leg].std(ddof=1) / np.sqrt(len(frame)))
                        rows.append({
                            "panel": panel, "J": J, "K": K, "leg": leg,
                            "mean": mean, "t": t_stat, "n": len(frame),
                            "reference_mean": mean, "reference_t": t_stat,
                            "paper_mean": mean + 0.0001, "paper_t": t_stat + 0.1,
                        })
        self.summary = pd.DataFrame(rows)
        self.monthly = pd.concat(series, ignore_index=True)
        self.formations = pd.DataFrame([
            {"panel": "A", "J": 6, "formation_id": 23570 + i, "momr": decile,
             "n_stocks": (5, 15)[i], "mean_signal": decile * (0.1, 0.2)[i]}
            for i in range(2) for decile in range(1, 11)
        ])
        self.coverage = pd.DataFrame({
            "month": dates, "n_stocks": 100, "n_returns": 99,
            "mean_ret": 0.005 + wave, "missing_rate": 0.01,
        })
        for name, frame in (
            ("table1_summary.csv", self.summary), ("table1_monthly_returns.csv", self.monthly),
            ("table1_formation_counts.csv", self.formations), ("table1_monthly_coverage.csv", self.coverage),
        ):
            frame.to_csv(self.result_dir / name, index=False)
        info = {
            "paper": "jt1993", "table": 1, "run_start_date": "1965-01-01", "run_end_date": "1989-12-31",
            "settings": {"panels": ["A", "B"], "ranking_months": [3, 6, 9, 12], "holding_months": [3, 6, 9, 12]},
            "raw_inputs": {"monthly_returns": {"download_metadata": {"universe": "nyse_amex"}}},
        }
        (self.result_dir / "run_info.json").write_text(json.dumps(info), encoding="utf-8")

    def test_notebook_units_gap_effect_and_stock_weighted_signal(self):
        figures = {}

        def capture(figure, output_dir, name, show=False):
            figures[name] = figure
            plt.close(figure)
            return Path(output_dir) / f"{name}.png"

        with patch("core.jt1993.reporter.table1.notebook.save_figure", side_effect=capture):
            paths = plot_notebook(self.summary, self.monthly, self.formations, self.coverage,
                                  self.result_dir, "Synthetic sample")
        self.assertEqual(len(paths), 11)
        expected = self.summary.query("panel == 'A' and leg == 'buy_sell'").pivot(index="J", columns="K", values="mean") * 100
        np.testing.assert_allclose(figures["panel_a_wl_return"].axes[0].images[0].get_array(), expected)
        np.testing.assert_allclose(figures["panel_a_paper_difference"].axes[0].images[0].get_array(), -1)
        np.testing.assert_allclose(figures["skipped_week_effect"].axes[0].images[0].get_array(), 2)
        heights = [bar.get_height() for bar in figures["ranking_signal_deciles"].axes[0].patches]
        np.testing.assert_allclose(heights, np.arange(1, 11) * 0.175)

    def test_table1_has_paper_layout_and_uses_calculated_values(self):
        summary = self.summary.sample(frac=1, random_state=7).copy()
        summary[["paper_mean", "paper_t", "reference_mean", "reference_t"]] = 99
        rows = table1_rows(summary)
        self.assertEqual(len(rows), 26)  # Two headers, then twelve return/t-stat pairs.
        self.assertTrue(all(len(row) == 12 for row in rows))
        self.assertEqual(rows[0], ["", "", "Panel A", "", "", "", "", "Panel B", "", "", "", ""])
        self.assertEqual(rows[1], ["J", "", "K =", "3", "6", "9", "12", "K =", "3", "6", "9", "12"])
        self.assertEqual([(row[0], row[1]) for row in rows[2::2]], [
            ("3", "Sell"), ("3", "Buy"), ("3", "Buy-sell"),
            ("6", "Sell"), ("6", "Buy"), ("6", "Buy-sell"),
            ("9", "Sell"), ("9", "Buy"), ("9", "Buy-sell"),
            ("12", "Sell"), ("12", "Buy"), ("12", "Buy-sell"),
        ])
        source = self.summary.set_index(["panel", "J", "K", "leg"])
        for pair, (J, leg) in enumerate((j, leg) for j in (3, 6, 9, 12) for leg in ("loser", "winner", "buy_sell")):
            means, tstats = rows[2 + pair * 2:4 + pair * 2]
            self.assertEqual(tstats[:3], ["", "", ""])
            self.assertEqual(means[2], "")
            self.assertEqual(means[7], "")
            for panel, columns in (("A", range(3, 7)), ("B", range(8, 12))):
                for K, column in zip((3, 6, 9, 12), columns):
                    record = source.loc[(panel, J, K, leg)]
                    self.assertEqual(means[column], f"{record['mean']:.4f}")
                    self.assertEqual(tstats[column], f"({record['t']:.2f})")

    def test_table1_rejects_missing_or_duplicate_portfolio_cells(self):
        for summary in (self.summary.iloc[1:], pd.concat([self.summary, self.summary.iloc[:1]])):
            with self.subTest(rows=len(summary)):
                with self.assertRaisesRegex(ValueError, "exactly one"):
                    table1_rows(summary)

    def test_table1_csv_round_trip_and_plain_html_display(self):
        summary = self.summary.copy()
        summary.loc[0, ["mean", "t"]] = [-0.012345, -2.346]
        summary.loc[1, "t"] = np.nan
        paths, markup = write_table1(summary, self.result_dir / "tables", "Synthetic sample")
        with paths["table1_csv"].open(newline="", encoding="utf-8") as stream:
            rows = list(csv.reader(stream))
        self.assertEqual(rows, table1_rows(summary))
        self.assertEqual(rows[2][3], "-0.0123")
        self.assertEqual(rows[3][3], "(-2.35)")
        self.assertEqual(rows[5][3], "(NA)")
        self.assertEqual(markup.count('class="mean-row"'), 12)
        self.assertEqual(markup.count('class="t-row"'), 12)
        self.assertIn('<th colspan="5">Panel A</th><th colspan="5">Panel B</th>', markup)
        self.assertIn(markup, paths["table1_html"].read_text(encoding="utf-8"))

    def test_overview_common_scales_and_additive_history(self):
        fig = _strategy_overview(self.summary, "Synthetic sample")
        self.assertEqual(fig.axes[0].images[0].get_clim(), fig.axes[3].images[0].get_clim())
        self.assertEqual(fig.axes[2].images[0].get_clim(), fig.axes[5].images[0].get_clim())
        low, high = fig.axes[2].images[0].get_clim()
        self.assertAlmostEqual(low, -high)
        history = _six_by_six_history(self.monthly, "Synthetic sample")
        returns = self.monthly.query("panel == 'A' and J == 6 and K == 6")["buy_sell"]
        np.testing.assert_allclose(history.axes[0].lines[0].get_ydata(), returns.cumsum() * 100)
        np.testing.assert_allclose(history.axes[1].lines[0].get_ydata(), returns.rolling(36).mean() * 100, equal_nan=True)

    def test_mixed_summary_or_missing_month_is_rejected_before_plotting(self):
        changed = self.summary.copy()
        changed.loc[0, "mean"] += 0.1
        changed.to_csv(self.result_dir / "table1_summary.csv", index=False)
        with self.assertRaisesRegex(ValueError, "Summary means disagree"):
            _load_results(self.result_dir)
        self.summary.to_csv(self.result_dir / "table1_summary.csv", index=False)
        self.monthly.iloc[1:].to_csv(self.result_dir / "table1_monthly_returns.csv", index=False)
        with self.assertRaisesRegex(ValueError, "dates/counts disagree"):
            _load_results(self.result_dir)

    def test_without_paper_benchmark_the_notebook_comparison_plots_are_omitted(self):
        summary = self.summary.drop(columns=["paper_mean", "paper_t", "reference_mean", "reference_t"])

        def discard(figure, output_dir, name, show=False):
            plt.close(figure)
            return Path(output_dir) / f"{name}.png"

        with patch("core.jt1993.reporter.table1.notebook.save_figure", side_effect=discard):
            paths = plot_notebook(summary, self.monthly, self.formations, self.coverage,
                                  self.result_dir, "Synthetic sample")
        self.assertEqual(len(paths), 7)
        self.assertFalse(any("paper" in name for name in paths))
        fig = _strategy_overview(summary, "Synthetic sample")
        self.assertEqual(len(fig.axes), 6)  # Two panels x two grids, plus two colorbars.

    def test_report_only_renders_both_sets_and_preserves_saved_inputs(self):
        before = {path.name: path.read_bytes() for path in self.result_dir.iterdir()}
        with (
            patch("core.jt1993.replication.table1.run_table1", side_effect=AssertionError("No replication")),
            patch("utils.jt1993_data_download.get_wrds_connection", side_effect=AssertionError("No WRDS")),
        ):
            paths = report_table1(self.result_dir)
        images = [path for path in paths.values() if path.suffix == ".png"]
        self.assertEqual(len(images), 14)
        for path in images:
            self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            self.assertGreater(path.stat().st_size, 1000)
        gallery = paths["index"].read_text(encoding="utf-8")
        self.assertTrue(paths["table1_csv"].is_file())
        self.assertTrue(paths["table1_html"].is_file())
        self.assertLess(gallery.index('id="table1"'), gallery.index('id="notebook"'))
        self.assertEqual(len(re.findall(r'<img ', gallery)), 14)
        for src in re.findall(r'src="([^"]+)"', gallery):
            self.assertTrue((paths["index"].parent / src).is_file())
        report_info = json.loads((paths["index"].parent / "report_info.json").read_text(encoding="utf-8"))
        self.assertEqual(len(report_info["figures"]["notebook"]), 11)
        self.assertEqual(len(report_info["figures"]["overview"]), 3)
        self.assertEqual(report_info["tables"]["table1_csv"], "table1.csv")
        for name, contents in before.items():
            self.assertEqual((self.result_dir / name).read_bytes(), contents)
        self.assertEqual(plt.get_fignums(), [])


if __name__ == "__main__":
    unittest.main()
