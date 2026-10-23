"""Download/cache monthly US FF3 factors before running main_jt1993_ff3.py."""

from utils.ff3_data_download import download_ff3_data


if __name__ == "__main__":
    refresh_data = False  # Reuse the saved factor vintage; True downloads the current vintage.
    download_ff3_data(refresh=refresh_data)
