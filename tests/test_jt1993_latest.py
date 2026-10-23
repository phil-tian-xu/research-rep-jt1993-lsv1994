"""Latest complete-month selection and connection reuse, without WRDS access."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

from data_prepare_jt1993 import data_prepare_jt1993
from tests.test_jt1993_data_download import sample_frame
from utils.jt1993_data_download import get_latest_jt1993_end_date
from utils.path_info import root_path


class LatestDateTests(unittest.TestCase):
    def setUp(self):
        self.connection = Mock()
        self.connection.raw_sql.return_value = pd.DataFrame({"last_date": [pd.Timestamp("2026-08-31")]})

    def test_latest_month_is_bounded_by_daily_monthly_and_both_index_sources(self):
        with redirect_stdout(io.StringIO()):
            end = get_latest_jt1993_end_date(self.connection, as_of="2026-10-22")
        self.assertEqual(end, "2026-08-31")  # Published coverage may lag the calendar.
        call = self.connection.raw_sql.call_args
        self.assertEqual(call.kwargs["params"], {"calendar_end": "2026-09-30"})
        for table in ("crsp.msf_v2", "crsp.dsf_v2", "crsp.indmthseriesdata"):
            self.assertIn(table, call.args[0])
        self.assertIn("a.mthcaldt <= (select max(dlycaldt)", call.args[0])
        self.assertIn("indno = 1000200 and mthtotret is not null", call.args[0])
        self.assertIn("indno = 1000201 and mthtotret is not null", call.args[0])
        self.connection.close.assert_not_called()

    def test_last_trading_day_is_normalized_to_calendar_month_end(self):
        self.connection.raw_sql.return_value = pd.DataFrame({"last_date": [pd.Timestamp("2025-08-29")]})
        with redirect_stdout(io.StringIO()):
            end = get_latest_jt1993_end_date(self.connection, as_of="2025-09-22")
        self.assertEqual(end, "2025-08-31")  # Friday publication; Sunday calendar end.
        self.assertEqual(self.connection.raw_sql.call_args.kwargs["params"]["calendar_end"], "2025-08-31")

    def test_first_day_of_month_and_leap_year_cutoff(self):
        self.connection.raw_sql.return_value = pd.DataFrame({"last_date": [pd.Timestamp("2024-02-29")]})
        with redirect_stdout(io.StringIO()):
            end = get_latest_jt1993_end_date(self.connection, as_of="2024-03-01")
        self.assertEqual(end, "2024-02-29")
        self.assertEqual(self.connection.raw_sql.call_args.kwargs["params"]["calendar_end"], "2024-02-29")

    def test_current_month_or_future_publication_is_rejected(self):
        self.connection.raw_sql.return_value = pd.DataFrame({"last_date": [pd.Timestamp("2026-10-01")]})
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "beyond the completed-month"):
            get_latest_jt1993_end_date(self.connection, as_of="2026-10-22")

    def test_legacy_discovery_uses_legacy_tables_and_caps_at_2024(self):
        self.connection.raw_sql.return_value = pd.DataFrame({"last_date": [pd.Timestamp("2024-12-31")]})
        with redirect_stdout(io.StringIO()):
            end = get_latest_jt1993_end_date(self.connection, data_source="legacy", as_of="2026-10-22")
        self.assertEqual(end, "2024-12-31")
        call = self.connection.raw_sql.call_args
        self.assertEqual(call.kwargs["params"]["calendar_end"], "2024-12-31")
        for table in ("crsp.msf", "crsp.dsf", "crsp.msi"):
            self.assertIn(table, call.args[0])
        self.assertNotIn("_v2", call.args[0])

    def test_requested_inputs_control_which_sources_are_required(self):
        with redirect_stdout(io.StringIO()):
            get_latest_jt1993_end_date(self.connection, datasets=("monthly_returns",), as_of="2026-10-22")
        sql = self.connection.raw_sql.call_args.args[0]
        self.assertIn("crsp.msf_v2", sql)
        self.assertIn("crsp.dsf_v2", sql)
        self.assertNotIn("indmthseriesdata", sql)
        with redirect_stdout(io.StringIO()):
            get_latest_jt1993_end_date(self.connection, datasets=("market_returns",), as_of="2026-10-22")
        sql = self.connection.raw_sql.call_args.args[0]
        self.assertIn("a.indno = 1000200", sql)
        self.assertIn("indno = 1000201", sql)
        self.assertNotIn("crsp.msf_v2", sql)
        self.assertNotIn("crsp.dsf_v2", sql)

    def test_no_common_publication_fails_clearly(self):
        for frame in (pd.DataFrame({"last_date": [pd.NaT]}), pd.DataFrame(columns=["last_date"])):
            self.connection.raw_sql.return_value = frame
            with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "No common completed month"):
                get_latest_jt1993_end_date(self.connection, as_of="2026-10-22")

    def test_invalid_discovery_settings_fail_before_querying(self):
        for kwargs in ({"data_source": "unknown"}, {"datasets": ()}, {"datasets": ("unknown",)}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                get_latest_jt1993_end_date(self.connection, **kwargs)
        self.connection.raw_sql.assert_not_called()


class LatestPreparationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix=".test_jt1993_", dir=Path(root_path))
        self.addCleanup(temporary.cleanup)
        self.cache_dir = Path(temporary.name)
        self.available = pd.DataFrame({"last_date": [pd.Timestamp("2025-02-28")]})

    def test_latest_preparation_reuses_one_connection_and_resolves_cache_dates(self):
        names = ("monthly_returns", "daily_returns", "monthly_characteristics", "market_returns")
        connection = Mock()
        connection.raw_sql.side_effect = [self.available] + [
            iter([sample_frame(name, "2025-01", "2025-02")]) for name in names
        ]
        with (
            patch("utils.jt1993_data_download.DEFAULT_CACHE_DIR", self.cache_dir),
            patch("data_prepare_jt1993.get_wrds_connection", return_value=connection) as connect,
            patch("utils.jt1993_data_download.get_wrds_connection") as second_connect,
            redirect_stdout(io.StringIO()),
        ):
            paths = data_prepare_jt1993(
                "2025-01-01", "latest", universe="nyse_amex", tables=(1, 2, 7), data_source="ciz",
            )
        connect.assert_called_once_with()
        second_connect.assert_not_called()
        connection.close.assert_called_once_with()
        self.assertEqual(connection.raw_sql.call_count, 5)
        for call in connection.raw_sql.call_args_list[1:]:
            self.assertEqual(call.kwargs["params"], {"start_date": "2025-01-01", "end_date": "2025-02-28"})
        for path in paths.values():
            metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["params"]["end_date"], "2025-02-28")
            self.assertEqual(path.parent.name, "nyse_amex_ciz_20250101_20250228")

        # A later check at the same vintage only needs the availability query.
        cached_bytes = {name: path.read_bytes() for name, path in paths.items()}
        next_connection = Mock()
        next_connection.raw_sql.return_value = self.available
        with (
            patch("utils.jt1993_data_download.DEFAULT_CACHE_DIR", self.cache_dir),
            patch("data_prepare_jt1993.get_wrds_connection", return_value=next_connection),
            redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(paths, data_prepare_jt1993(
                "2025-01-01", "latest", universe="nyse_amex", tables=(1, 2, 7), data_source="ciz",
            ))
        self.assertEqual(next_connection.raw_sql.call_count, 1)
        next_connection.close.assert_called_once_with()
        for name, path in paths.items():
            self.assertEqual(path.read_bytes(), cached_bytes[name])

    def test_availability_and_download_failures_close_the_connection(self):
        for responses in ([RuntimeError("availability failed")], [self.available, RuntimeError("download failed")]):
            connection = Mock()
            connection.raw_sql.side_effect = responses
            with (
                patch("utils.jt1993_data_download.DEFAULT_CACHE_DIR", self.cache_dir),
                patch("data_prepare_jt1993.get_wrds_connection", return_value=connection),
                redirect_stdout(io.StringIO()),
                self.assertRaisesRegex(RuntimeError, "failed"),
            ):
                data_prepare_jt1993("2025-01-01", "latest", data_source="ciz")
            connection.close.assert_called_once_with()
        self.assertEqual(list(self.cache_dir.rglob("*.parquet")), [])

    def test_available_end_before_start_fails_without_downloading(self):
        connection = Mock()
        connection.raw_sql.return_value = self.available
        with (
            patch("data_prepare_jt1993.get_wrds_connection", return_value=connection),
            redirect_stdout(io.StringIO()),
            self.assertRaisesRegex(ValueError, "precedes requested start"),
        ):
            data_prepare_jt1993("2026-01-01", "latest", data_source="ciz")
        connection.close.assert_called_once_with()
        self.assertEqual(connection.raw_sql.call_count, 1)

    def test_bad_latest_request_settings_fail_before_authentication(self):
        for changes in ({"start": "invalid"}, {"start": "2025-01-02"},
                        {"universe": "unknown"}, {"data_source": "unknown"}):
            request = {"start": "2025-01-01", "end": "latest", "data_source": "ciz", **changes}
            with patch("data_prepare_jt1993.get_wrds_connection") as connect, redirect_stdout(io.StringIO()):
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    data_prepare_jt1993(**request)
                connect.assert_not_called()


if __name__ == "__main__":
    unittest.main()
