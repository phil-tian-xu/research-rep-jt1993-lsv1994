"""Offline checks for FF3 parsing, cache provenance, regression, and month alignment."""

import csv
from contextlib import chdir, redirect_stdout
from hashlib import sha256
from io import BytesIO, StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zipfile import BadZipFile, ZipFile

import numpy as np
import pandas as pd

from core.jt1993.ff3 import run_ff3
from core.jt1993.replication.table7 import newey_west_mean
from core.jt1993.reporter.ff3 import report_ff3
from utils.ff3_data_download import download_ff3_data, parse_ff3_monthly
from utils.inference import ols_newey_west
from utils.path_info import root_path


def factor_archive(rows):
    """Small monthly/annual archive with the official CSV structure."""
    text = "Created for an offline test.\nMissing values are -99.99 or -999.\n,Mkt-RF,SMB,HML,RF\n"
    text += "\n".join(",".join(map(str, row)) for row in rows)
    text += "\n\n Annual Factors: January-December\n,Mkt-RF,SMB,HML,RF\n2000,99,88,77,66\n"
    output = BytesIO()
    with ZipFile(output, "w") as zipped:
        zipped.writestr("F-F_Research_Data_Factors.csv", text)
    return output.getvalue()


class InferenceTests(unittest.TestCase):
    def test_ols_and_hac_against_known_coefficients_and_dense_kernel(self):
        rng = np.random.default_rng(1993)
        x = rng.normal(size=(100, 3))
        design = np.column_stack([np.ones(100), x])
        beta = np.array([0.005, 0.2, -0.3, 0.4])
        noise = np.cumsum(rng.normal(size=100)) * 0.001
        noise -= design @ np.linalg.lstsq(design, noise, rcond=None)[0]
        y = design @ beta + noise
        bread = np.linalg.inv(design.T @ design)
        for lags in (0, 1, 5, 12):
            result = ols_newey_west(y, x, lags)
            np.testing.assert_allclose(result["coefficient"], beta, atol=1e-12)
            distance = np.abs(np.arange(100)[:, None] - np.arange(100))
            kernel = np.maximum(1 - distance / (lags + 1), 0)
            covariance = bread @ design.T @ (np.outer(noise, noise) * kernel) @ design @ bread * 100 / 96
            np.testing.assert_allclose(result["covariance"], covariance, rtol=1e-10, atol=1e-15)
            np.testing.assert_allclose(result["hac_se"], np.sqrt(np.diag(covariance)), rtol=1e-10)
            self.assertTrue(np.all(result["ci95_lower"] < result["coefficient"]))
            self.assertTrue(np.all(result["ci95_upper"] > result["coefficient"]))

    def test_intercept_only_agrees_with_existing_mean_inference(self):
        values = np.array([0.01, -0.03, 0.04, 0.02, -0.01, 0.005])
        for lags in (0, 1, 3):
            result = ols_newey_west(values, np.empty((6, 0)), lags)
            expected = newey_west_mean(values, lags)
            np.testing.assert_allclose([result["coefficient"][0], result["hac_se"][0], result["t_stat"][0]], expected)

    def test_bad_inputs_and_singular_design_are_rejected(self):
        y = np.arange(8, dtype=float)
        x = np.column_stack([np.sin(y), np.cos(y)])
        for lags in (-1, 8, 1.5, True):
            with self.subTest(lags=lags), self.assertRaises(ValueError):
                ols_newey_west(y, x, lags)
        for bad_y, bad_x in ((y[:2], x[:2]), (y, np.ones((8, 1))), (np.ones(8), x),
                             (np.r_[np.nan, y[1:]], x), (y[:, None], x), (y, x[:7])):
            with self.assertRaises(ValueError):
                ols_newey_west(bad_y, bad_x, 1)


class DownloadTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix=".test_ff3_", dir=Path(root_path))
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        self.rows = [[200001, 1.25, -2.0, 3.5, 0.4], [200002, -3.0, 0.0, -1.5, 0.3]]

    def test_monthly_block_units_dates_and_missing_codes(self):
        frame = parse_ff3_monthly(factor_archive(self.rows))
        self.assertEqual(frame["month"].tolist(), [pd.Timestamp("2000-01-31"), pd.Timestamp("2000-02-29")])
        np.testing.assert_allclose(frame.iloc[0, 1:].to_numpy(dtype=float), [0.0125, -0.02, 0.035, 0.004])
        self.assertEqual(len(frame), 2)  # Annual row is excluded.
        missing = parse_ff3_monthly(factor_archive([[200001, -99.99, -999, 0, 0.4]]))
        self.assertTrue(missing[["mkt_rf", "smb"]].isna().all().all())
        self.assertEqual(missing.loc[0, "hml"], 0)

    def test_bad_months_header_and_values_are_rejected(self):
        bad_rows = [self.rows + [self.rows[-1]], [self.rows[0], [200003, 1, 2, 3, 4]],
                    list(reversed(self.rows)), [[200013, 1, 2, 3, 4]], [[200001, 1, 2]], [[200001, "bad", 2, 3, 4]]]
        for rows in bad_rows:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                parse_ff3_monthly(factor_archive(rows))
        with self.assertRaises(ValueError):
            parse_ff3_monthly(factor_archive([]))

    def test_cache_is_reused_and_failed_refresh_preserves_snapshot(self):
        archive = factor_archive(self.rows)
        relative = self.folder.relative_to(Path(root_path))
        with patch("utils.ff3_data_download.urlopen", return_value=BytesIO(archive)) as connect, redirect_stdout(StringIO()), chdir(self.folder):
            path = download_ff3_data(data_dir=relative)
            connect.assert_called_once()
        before = {file.name: file.read_bytes() for file in self.folder.iterdir() if file.is_file()}
        metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["source_sha256"], sha256(archive).hexdigest())
        self.assertEqual(metadata["return_units"], "decimal")
        with patch("utils.ff3_data_download.urlopen", side_effect=AssertionError("Unexpected download")), redirect_stdout(StringIO()):
            self.assertEqual(download_ff3_data(data_dir=self.folder), path)
        with patch("utils.ff3_data_download.urlopen", return_value=BytesIO(b"not a ZIP")), redirect_stdout(StringIO()):
            with self.assertRaises(BadZipFile):
                download_ff3_data(refresh=True, data_dir=self.folder)
        self.assertEqual(before, {file.name: file.read_bytes() for file in self.folder.iterdir() if file.is_file()})
        changed = pd.read_parquet(path)
        changed.loc[0, "mkt_rf"] += 0.1
        changed.to_parquet(path, index=False)
        with self.assertRaisesRegex(ValueError, "cache/provenance mismatch"):
            download_ff3_data(data_dir=self.folder)


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix=".test_ff3_", dir=Path(root_path))
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        self.source = self.folder / "table1"
        self.source.mkdir()
        self.factor_path = self.folder / "ff3_monthly.parquet"
        rng = np.random.default_rng(1993)
        months = pd.date_range("2000-01-31", periods=120, freq="ME")
        factors = rng.normal(0, 0.025, size=(120, 3))
        design = np.column_stack([np.ones(120), factors])
        error = rng.normal(0, 0.01, size=120)
        error -= design @ np.linalg.lstsq(design, error, rcond=None)[0]
        self.beta = np.array([0.004, 0.2, -0.3, 0.4])
        spread = design @ self.beta + error
        self.returns = pd.DataFrame({"month": months, "panel": "A", "J": 6, "K": 6,
                                     "loser": 0.01, "winner": 0.01 + spread, "buy_sell": spread})
        panel_b = self.returns.assign(panel="B", winner=self.returns["winner"] + 0.002, buy_sell=spread + 0.002)
        self.returns = pd.concat([self.returns, panel_b], ignore_index=True)
        self.returns.to_csv(self.source / "table1_monthly_returns.csv", index=False)
        info = {"paper": "jt1993", "table": 1, "return_units": "decimal monthly returns",
                "run_start_date": "2000-01-01", "run_end_date": "2009-12-31",
                "raw_inputs": {"monthly_returns": {"download_metadata": {"universe": "nyse_amex"}}}}
        (self.source / "run_info.json").write_text(json.dumps(info), encoding="utf-8")
        self.factors = pd.DataFrame(factors, columns=["mkt_rf", "smb", "hml"])
        self.factors.insert(0, "month", months)
        self.factors["rf"] = 0.04  # A second RF subtraction would visibly distort alpha.
        self.write_factors(self.factors)

    def write_factors(self, frame, units="decimal"):
        frame.to_parquet(self.factor_path, index=False)
        info = {"frequency": "monthly", "return_units": units,
                "parquet_sha256": sha256(self.factor_path.read_bytes()).hexdigest()}
        self.factor_path.with_suffix(".json").write_text(json.dumps(info), encoding="utf-8")

    def run_model(self, **kwargs):
        return run_ff3("2000-01-01", "2009-12-31", self.source, self.factor_path,
                       self.folder / "output", **kwargs)

    def test_offline_regression_report_and_no_second_rf_subtraction(self):
        before = (self.source / "table1_monthly_returns.csv").read_bytes(), self.factor_path.read_bytes()
        with patch("utils.ff3_data_download.urlopen", side_effect=AssertionError("Main must stay offline")), redirect_stdout(StringIO()):
            coefficients, monthly = self.run_model()
            paths = report_ff3(self.folder / "output")
        np.testing.assert_allclose(coefficients["coefficient"], self.beta, atol=1e-12)
        self.assertEqual(len(monthly), 120)
        np.testing.assert_allclose(monthly["fitted_return"] + monthly["residual"], monthly["buy_sell"])
        info = json.loads((self.folder / "output" / "run_info.json").read_text(encoding="utf-8"))
        self.assertFalse(info["risk_free_subtracted_from_spread"])
        self.assertEqual(info["n_obs"], 120)
        with paths["csv"].open(encoding="utf-8", newline="") as stream:
            rows = list(csv.reader(stream))
        self.assertEqual(len(rows), 5)
        self.assertEqual(rows[1][:2], ["Alpha (% per month)", "0.4000"])
        self.assertTrue(paths["html"].is_file())
        self.assertEqual(before, ((self.source / "table1_monthly_returns.csv").read_bytes(), self.factor_path.read_bytes()))
        # RF is kept for audit, but it is not a regressor or a sample restriction.
        self.write_factors(self.factors.assign(rf=np.nan))
        with redirect_stdout(StringIO()):
            without_rf, _ = self.run_model(panel="B", hac_lags=12)
        np.testing.assert_allclose(without_rf["coefficient"], self.beta + [0.002, 0, 0, 0], atol=1e-12)

    def test_missing_duplicate_months_and_wrong_units_are_rejected(self):
        for bad in (self.factors.drop(index=20), pd.concat([self.factors, self.factors.iloc[:1]]),
                    self.factors.assign(smb=self.factors["smb"].mask(self.factors.index == 20))):
            self.write_factors(bad)
            with self.assertRaisesRegex(ValueError, "monthly FF3|Duplicate calendar"):
                self.run_model()
        self.write_factors(self.factors, units="percent")
        with self.assertRaisesRegex(ValueError, "decimal"):
            self.run_model()
        self.write_factors(self.factors)
        for bad in (self.returns.drop(index=20), pd.concat([self.returns, self.returns.iloc[:1]])):
            bad.to_csv(self.source / "table1_monthly_returns.csv", index=False)
            with self.assertRaisesRegex(ValueError, "duplicate or missing"):
                self.run_model()

    def test_spread_identity_dates_and_relative_paths(self):
        self.returns.assign(buy_sell=self.returns["buy_sell"] + 0.001).to_csv(self.source / "table1_monthly_returns.csv", index=False)
        with self.assertRaisesRegex(ValueError, "winner minus loser"):
            self.run_model()
        self.returns.to_csv(self.source / "table1_monthly_returns.csv", index=False)
        for start, end in (("1999-01-01", "2009-12-31"), ("2000-01-02", "2009-12-31"),
                           ("2000-01-01T01:00:00", "2009-12-31")):
            with self.assertRaises(ValueError):
                run_ff3(start, end, self.source, self.factor_path, self.folder / "bad")
        relative = self.folder.relative_to(Path(root_path))
        with chdir(self.folder), redirect_stdout(StringIO()):
            coefficients, monthly = run_ff3("2002-01-01", "2005-12-31", relative / "table1",
                                           relative / self.factor_path.name, relative / "subset")
        self.assertEqual(len(monthly), 48)
        self.assertEqual(len(coefficients), 4)


if __name__ == "__main__":
    unittest.main()
