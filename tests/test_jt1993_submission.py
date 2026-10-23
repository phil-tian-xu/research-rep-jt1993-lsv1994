"""Check published benchmarks, comparison alignment/units, and portable assembly."""

import base64
from contextlib import chdir
from hashlib import sha256
import json
from math import erfc, sqrt
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from core.submission.jt1993 import (
    compare_tables, generate_submission, load_paper_tables, load_results,
)
from core.submission.comparisons import _comparison_tables, render_case_section
from core.submission.ff3 import render_ff3_section
from utils.path_info import root_path


class SubmissionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix=".test_submission_", dir=Path(root_path))
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.paper = load_paper_tables()
        self.results = {table: frame.copy() for table, frame in self.paper.items()}
        t1 = self.results[1]
        t1["mean"] = np.where(t1["leg"].eq("loser"), 0.005,
                              np.where(t1["leg"].eq("winner"), 0.015, 0.010))
        t1["t"], t1["n"] = 2.5, 300
        # These deliberately wrong attached columns must never supply paper values.
        t1["paper_mean"], t1["paper_t"], t1["reference_mean"] = 99, 99, 99
        t2 = self.results[2]
        t2["n_months"] = 300
        t2.loc[t2["portfolio"].ne("P10-P1"), "average_market_cap_millions"] += 10
        t7 = self.results[7]
        t7["monthly_return"] += 0.0001
        t7["cumulative_return"] = t7["monthly_return"].cumsum()
        t7["n_cohorts"] = 265
        self.result_dirs = {table: self.directory / f"table{table}" for table in self.results}
        raw = self.directory / "nonexistent_raw_data"
        for table, directory in self.result_dirs.items():
            directory.mkdir()
            self.results[table].to_csv(directory / f"table{table}_summary.csv", index=False)
            info = {"paper": "jt1993", "table": table, "run_start_date": "1965-01-01",
                    "run_end_date": "1989-12-31", "raw_data_dir": str(raw)}
            if table == 1:
                info["raw_inputs"] = {"monthly_returns": {
                    "path": str(raw / "monthly_returns.parquet"),
                    "download_metadata": {"universe": "nyse_amex"},
                }}
            if table == 7:
                info.update(formation_start="1965-01-31", formation_end="1987-01-31",
                            n_cohorts=265, hac_lags="horizon minus one")
            (directory / "run_info.json").write_text(json.dumps(info), encoding="utf-8")

    def save_case(self, name, universe, start="1965-01-01", end="1989-12-31", spread=0.01, t_stat=2.5):
        """Save a coherent synthetic sample, including its complete-cohort boundary."""
        directories = {}
        n_months = len(pd.period_range(start, end, freq="M"))
        raw = self.directory / name / "nonexistent_raw"
        for table, original in self.results.items():
            directory = self.directory / name / f"table{table}"
            directory.mkdir(parents=True)
            directories[table] = directory
            frame = original.copy()
            if table == 1:
                frame["n"] = n_months
                frame.loc[frame["leg"].eq("winner"), "mean"] = 0.005 + spread
                frame.loc[frame["leg"].eq("buy_sell"), ["mean", "t"]] = [spread, t_stat]
            elif table == 2:
                frame["n_months"] = n_months
            else:
                frame["n_cohorts"] = n_months - 35
            frame.to_csv(directory / f"table{table}_summary.csv", index=False)
            info = json.loads((self.result_dirs[table] / "run_info.json").read_text(encoding="utf-8"))
            info.update(run_start_date=start, run_end_date=end, raw_data_dir=str(raw))
            if table == 1:
                info["raw_inputs"]["monthly_returns"]["path"] = str(raw / "monthly_returns.parquet")
                info["raw_inputs"]["monthly_returns"]["download_metadata"]["universe"] = universe
            if table == 7:
                info.update(formation_start=str(pd.Timestamp(start).to_period("M").to_timestamp(how="end").date()),
                            formation_end=str((pd.Timestamp(end).to_period("M") - 35).to_timestamp(how="end").date()),
                            n_cohorts=n_months - 35)
            (directory / "run_info.json").write_text(json.dumps(info), encoding="utf-8")
        return directories

    def save_ff3(self, alpha=0.005):
        """Saved regression fixture with explicit units, inference, and source identity."""
        directory = self.directory / "ff3"
        directory.mkdir(exist_ok=True)
        wave = np.sin(np.arange(300))
        spread = 0.01 + (wave - wave.mean()) * 0.002
        returns_file = self.result_dirs[1] / "table1_monthly_returns.csv"
        pd.DataFrame({"month": pd.date_range("1965-01-31", "1989-12-31", freq="ME"),
                      "panel": "A", "J": 6, "K": 6, "loser": 0.005,
                      "winner": 0.005 + spread, "buy_sell": spread}).to_csv(returns_file, index=False)
        coefficient = np.array([alpha, -0.1, -0.2, -0.3])
        se = np.array([0.002, 0.15, 0.1, 0.12])
        t_stat = coefficient / se
        frame = pd.DataFrame({
            "term": ["alpha", "mkt_rf", "smb", "hml"], "coefficient": coefficient,
            "hac_se": se, "t_stat": t_stat,
            "p_value": [erfc(abs(value) / sqrt(2)) for value in t_stat],
            "ci95_lower": coefficient - 1.959963984540054 * se,
            "ci95_upper": coefficient + 1.959963984540054 * se,
        })
        frame.to_csv(directory / "ff3_coefficients.csv", index=False)
        info = {
            "paper": "jt1993", "question": "Q1-5", "universe": "nyse_amex",
            "run_start_date": "1965-01-01", "run_end_date": "1989-12-31", "panel": "A", "J": 6, "K": 6,
            "table1_result_dir": str(self.result_dirs[1]), "returns_sha256": sha256(returns_file.read_bytes()).hexdigest(),
            "return_units": "decimal monthly returns", "risk_free_subtracted_from_spread": False,
            "hac_kernel": "Bartlett", "hac_lags": 5, "covariance_correction": "n/(n-4)",
            "inference": "two-sided asymptotic normal p-values and 95% confidence intervals",
            "n_obs": 300, "df_resid": 296, "r_squared": 0.16,
            # The renderer must derive the conclusion from estimates, not this flag.
            "alpha_positive_and_significant_5pct": True,
        }
        (directory / "run_info.json").write_text(json.dumps(info), encoding="utf-8")
        return directory

    def test_published_values_coverage_and_rounding_are_preserved(self):
        self.assertEqual({table: len(frame) for table, frame in self.paper.items()}, {1: 96, 2: 11, 7: 36})
        t1 = self.paper[1].set_index(["panel", "J", "K", "leg"])
        self.assertEqual(t1.loc[("A", 6, 6, "buy_sell"), "mean"], 0.0095)
        self.assertEqual(t1.loc[("B", 12, 3, "buy_sell"), "t"], 4.28)
        t2 = self.paper[2].set_index("portfolio")
        self.assertEqual(t2.loc["P1", "average_market_cap_millions"], 208.24)
        self.assertEqual(t2.loc["P10-P1", "beta"], -0.08)
        self.assertTrue(pd.isna(t2.loc["P10-P1", "average_market_cap_millions"]))
        t7 = self.paper[7].set_index("event_month")
        self.assertEqual(t7.loc[12, "cumulative_return"], 0.0951)
        self.assertEqual(t7.loc[36, "cumulative_return"], 0.0406)
        self.assertEqual(t7.loc[3, "cumulative_return"], 0.0216)
        self.assertNotAlmostEqual(t7.loc[3, "cumulative_return"], t7.loc[:3, "monthly_return"].sum(), places=6)

    def test_comparisons_use_keys_independent_paper_values_and_correct_units(self):
        shuffled = {table: frame.sample(frac=1, random_state=table) for table, frame in self.results.items()}
        paper_before = {table: frame.copy() for table, frame in self.paper.items()}
        compared = compare_tables(shuffled, self.paper)
        t1 = compared[1].set_index(["panel", "J", "K", "leg"])
        self.assertAlmostEqual(t1.loc[("A", 6, 6, "buy_sell"), "mean_diff_bp"], 5)
        self.assertEqual(t1.loc[("A", 6, 6, "buy_sell"), "mean_paper"], 0.0095)
        t2 = compared[2].set_index("portfolio")
        self.assertAlmostEqual(t2.loc["P1", "average_market_cap_millions_diff"], 10)
        self.assertTrue(pd.isna(t2.loc["P10-P1", "average_market_cap_millions_diff"]))
        t7 = compared[7].set_index("event_month")
        self.assertAlmostEqual(t7.loc[1, "monthly_return_diff_bp"], 1)
        for table, before in paper_before.items():
            pd.testing.assert_frame_equal(self.paper[table], before)

    def test_missing_duplicate_and_nonfinite_cells_fail(self):
        for table in (1, 2, 7):
            for broken in (self.results[table].iloc[1:],
                           pd.concat([self.results[table], self.results[table].iloc[:1]])):
                with self.subTest(table=table, rows=len(broken)):
                    with self.assertRaises(ValueError):
                        compare_tables({**self.results, table: broken}, self.paper)
        broken = self.results[7].copy()
        broken.loc[0, "monthly_return"] = np.inf
        with self.assertRaisesRegex(ValueError, "finite"):
            compare_tables({**self.results, 7: broken}, self.paper)
        broken = self.paper[2].copy()
        broken.loc[broken["portfolio"].eq("P10-P1"), "average_market_cap_millions"] = 0
        with self.assertRaisesRegex(ValueError, "absent, not zero"):
            compare_tables(self.results, {**self.paper, 2: broken})

    def test_mixed_samples_and_corrupted_cumulative_results_fail(self):
        metadata_path = self.result_dirs[1] / "run_info.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["raw_inputs"]["monthly_returns"]["download_metadata"]["universe"] = "nyse"
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "NYSE/AMEX"):
            load_results(self.result_dirs)
        metadata["raw_inputs"]["monthly_returns"]["download_metadata"]["universe"] = "nyse_amex"
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        broken = self.results[7].copy()
        broken.loc[0, "cumulative_return"] += 0.01
        broken.to_csv(self.result_dirs[7] / "table7_summary.csv", index=False)
        with self.assertRaisesRegex(ValueError, "additive"):
            load_results(self.result_dirs)
        self.results[7].to_csv(self.result_dirs[7] / "table7_summary.csv", index=False)
        metadata_path = self.result_dirs[2] / "run_info.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["run_end_date"] = "2025-12-31"
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "1965-1989"):
            load_results(self.result_dirs)

    def test_one_offline_html_embeds_scrollable_tables_and_images_without_data_exports(self):
        before = {p: p.read_bytes() for directory in self.result_dirs.values() for p in directory.iterdir()}
        output = self.directory / "output" / "submission.html"
        relative_dirs = {table: directory.relative_to(Path(root_path)) for table, directory in self.result_dirs.items()}
        with (
            patch("core.jt1993.replication.table1.run_table1", side_effect=AssertionError("No replication")),
            patch("utils.jt1993_data_download.get_wrds_connection", side_effect=AssertionError("No WRDS")),
            chdir(self.directory),
        ):
            result = generate_submission(relative_dirs, output_path=output)
        self.assertEqual(result, output)
        html = output.read_text(encoding="utf-8")
        self.assertEqual(html.count("<table>"), 7)
        images = re.findall(r'src="data:image/png;base64,([^"]+)"', html)
        self.assertEqual(len(images), 3)
        for image in images:
            self.assertTrue(base64.b64decode(image).startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertNotRegex(html, r'src="(?:https?:|file:|\.\.?/)')
        self.assertNotRegex(html, r'href="(?:file:|\.\.?/)')
        self.assertNotIn(str(root_path), html)
        self.assertNotIn("run_info.json", html)
        self.assertNotIn("$comparison", html)
        self.assertNotIn('download="', html)
        self.assertNotIn("data:text/csv", html)
        self.assertNotIn("Reference and supporting data", html)
        self.assertNotIn('id="sources"', html)
        self.assertEqual(html.count('class="table-scroll"'), 7)
        self.assertEqual(html.count('class="table-expand"'), 7)
        self.assertEqual(list(output.parent.iterdir()), [output])
        for source, contents in before.items():
            self.assertEqual(source.read_bytes(), contents)

    def test_extended_loader_respects_actual_dates_universe_and_month_counts(self):
        recent = self.save_case("recent", "nyse_amex", "1990-01-01", "2024-12-31")
        tables, info = load_results(recent, original_sample=False)
        self.assertTrue(tables[1]["n"].eq(420).all())
        self.assertTrue(tables[7]["n_cohorts"].eq(385).all())
        self.assertEqual(info[7]["formation_end"], "2022-01-31")
        with self.assertRaisesRegex(ValueError, "1965-1989"):
            load_results(recent)
        nyse = self.save_case("nyse", "nyse")
        load_results(nyse, original_sample=False)
        with self.assertRaisesRegex(ValueError, "NYSE/AMEX"):
            load_results(nyse)
        broken = tables[1].copy()
        broken["n"] = 300
        broken.to_csv(recent[1] / "table1_summary.csv", index=False)
        with self.assertRaisesRegex(ValueError, "420"):
            load_results(recent, original_sample=False)

    def test_sample_comparisons_reject_wrong_windows_universes_or_period_gaps(self):
        early = load_results(self.result_dirs)
        recent = load_results(self.save_case("recent", "nyse_amex", "1990-01-01", "2024-12-31"), original_sample=False)
        nyse = load_results(self.save_case("nyse", "nyse"), original_sample=False)
        with self.assertRaisesRegex(ValueError, "same reporting dates"):
            render_case_section({"NYSE": nyse, "Recent": recent}, "exchanges")
        with self.assertRaisesRegex(ValueError, "same exchange universe"):
            render_case_section({"NYSE": nyse, "Recent": recent}, "subperiods")
        gap = load_results(self.save_case("gap", "nyse_amex", "1991-01-01", "2024-12-31"), original_sample=False)
        with self.assertRaisesRegex(ValueError, "adjacent"):
            render_case_section({"Early": early, "Late": gap}, "subperiods")
        recent[1][1]["raw_inputs"]["monthly_returns"]["download_metadata"]["data_source"] = "ciz"
        with self.assertRaisesRegex(ValueError, "same data source"):
            render_case_section({"Early": early, "Recent": recent}, "subperiods")

    def test_expanded_submission_contains_every_sample_and_descriptive_period_differences(self):
        nyse = self.save_case("nyse", "nyse")
        all_exchanges = self.save_case("all", "nyse_amex_nasdaq")
        recent = self.save_case("recent", "nyse_amex", "1990-01-01", "2024-12-31", spread=0.006, t_stat=1.4)
        cases = {"1965–1989": load_results(self.result_dirs),
                 "1990–2024": load_results(recent, original_sample=False)}
        paired = _comparison_tables(cases, "subperiods", "subperiods")[0]
        # 0.006 minus 0.010 is -40 bp; rows retain the exact saved t-statistics.
        self.assertEqual(paired.count("-40.00"), 32)
        self.assertEqual(paired.count("(1.40)"), 32)
        output = generate_submission(
            self.result_dirs, output_path=self.directory / "expanded.html",
            exchange_runs={"NYSE": nyse, "NYSE/AMEX": self.result_dirs, "NYSE/AMEX/Nasdaq": all_exchanges},
            subperiod_runs={"1965–1989": self.result_dirs, "1990–2024": recent},
        )
        html = output.read_text(encoding="utf-8")
        self.assertEqual(html.count("<table>"), 28)
        self.assertEqual(html.count('src="data:image/png;base64,'), 9)
        self.assertEqual(html.count('class="case-select"'), 2)
        self.assertIn('id="exchange-samples"', html)
        self.assertIn('id="subperiods"', html)
        self.assertIn("January 1990–December 2024", html)
        self.assertIn("385 complete cohorts", html)
        self.assertIn("neither estimate meets", html)
        self.assertIn("no formal test of the change", html)
        self.assertIn("rather than that full span", html)
        self.assertNotIn('id="overview"', html)
        self.assertNotIn('download="', html)
        ids = re.findall(r'\bid="([^"]+)"', html)
        self.assertEqual(len(ids), len(set(ids)))
        for controlled in re.findall(r'aria-controls="([^"]+)"', html):
            self.assertIn(controlled, ids)

    def test_q1_5_assembly_uses_saved_estimates_and_preserves_inputs(self):
        ff3 = self.save_ff3()
        before = {p: p.read_bytes() for folder in (*self.result_dirs.values(), ff3) for p in folder.iterdir()}
        relative_dirs = {table: folder.relative_to(Path(root_path)) for table, folder in self.result_dirs.items()}
        with (
            patch("core.jt1993.ff3.run_ff3", side_effect=AssertionError("No regression during assembly")),
            patch("utils.ff3_data_download.urlopen", side_effect=AssertionError("No factor download")),
            chdir(self.directory),
        ):
            output = generate_submission(relative_dirs, ff3_result_dir=ff3.relative_to(Path(root_path)),
                                         output_path=self.directory / "with_ff3.html")
        html = output.read_text(encoding="utf-8")
        self.assertIn('href="#ff3-regression"', html)
        self.assertIn('id="ff3-regression"', html)
        self.assertIn("Alpha (% per month)", html)
        self.assertIn("0.5000", html)
        self.assertIn("0.2000", html)
        self.assertIn("2.500", html)
        self.assertIn("0.0124", html)
        self.assertIn("positive and statistically significant alpha", html)
        self.assertIn("asymptotic normal inference", html)
        self.assertIn("risk-free rate cancels", html)
        self.assertIn("R² = 0.1600", html)
        self.assertEqual(html.count("<table>"), 8)
        self.assertEqual(html.count('src="data:image/png;base64,'), 3)
        self.assertNotIn("$ff3_", html)
        self.assertNotIn(str(root_path), html)
        self.assertEqual(before, {p: p.read_bytes() for p in before})

    def test_q1_5_rejects_mismatched_inputs_and_invalid_inference(self):
        ff3 = self.save_ff3()
        table1_info = json.loads((self.result_dirs[1] / "run_info.json").read_text(encoding="utf-8"))
        metadata_path = ff3 / "run_info.json"
        original = json.loads(metadata_path.read_text(encoding="utf-8"))
        for key, wrong in (("universe", "nyse"), ("run_end_date", "2024-12-31"), ("n_obs", 299),
                           ("return_units", "percent"), ("risk_free_subtracted_from_spread", True),
                           ("hac_lags", True), ("table1_result_dir", str(self.directory / "other")),
                           ("returns_sha256", "outdated")):
            metadata_path.write_text(json.dumps({**original, key: wrong}), encoding="utf-8")
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "Q1-5"):
                render_ff3_section(ff3, self.result_dirs[1], table1_info)
        metadata_path.write_text(json.dumps(original), encoding="utf-8")
        coefficients = pd.read_csv(ff3 / "ff3_coefficients.csv")
        for broken in (coefficients.iloc[1:], pd.concat([coefficients, coefficients.iloc[:1]]),
                       coefficients.assign(t_stat=999), coefficients.assign(p_value=-1),
                       coefficients.assign(coefficient=np.nan)):
            broken.to_csv(ff3 / "ff3_coefficients.csv", index=False)
            with self.assertRaisesRegex(ValueError, "Q1-5"):
                render_ff3_section(ff3, self.result_dirs[1], table1_info)

    def test_q1_5_conclusion_follows_sign_and_significance(self):
        table1_info = json.loads((self.result_dirs[1] / "run_info.json").read_text(encoding="utf-8"))
        ff3 = self.save_ff3(alpha=-0.005)
        html = render_ff3_section(ff3, self.result_dirs[1], table1_info)
        self.assertIn("does not provide evidence of a positive", html)
        self.assertNotIn("retains a positive", html)
        ff3 = self.save_ff3(alpha=0.001)
        html = render_ff3_section(ff3, self.result_dirs[1], table1_info)
        self.assertIn("not statistically significant at the 5% level", html)
        self.assertNotIn("retains a positive", html)


if __name__ == "__main__":
    unittest.main()
