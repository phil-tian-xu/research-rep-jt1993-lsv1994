"""Assemble Q2-1–Q2-4, including all completed extended runs, from saved results."""

from pathlib import Path
from core.submission.lsv1994 import generate_submission
from utils.path_info import results_path


if __name__ == "__main__":
    run_tag = "nyse_amex_19621201_19900430_comp19620101_19931231_1968_1989_to199004_nyse_amex_ceq"
    result_dirs = {table: Path(results_path)/"lsv1994"/"replication"/f"table{table}"/run_tag
                   for table in (1,2,4,5,6)}
    extension_tags={
        "nyse":"q22_full_nyse_1962_2025_legacy_f1963_2023_r202404_ceq",
        "nyse_amex_nasdaq":"q22_full_nyse_amex_nasdaq_1962_2025_legacy_f1963_2023_r202404_ceq",
        "early":"q23_early_nyse_amex_1963_1989_legacy_f1963_1988_r198904_ceq",
        "recent":"q23_recent_nyse_amex_1990_2025_legacy_f1990_2023_r202404_ceq",
    }
    extension_dirs={key:{table:Path(results_path)/"lsv1994/replication"/f"table{table}"/tag
                         for table in (1,2,4,5,6)} for key,tag in extension_tags.items()}
    output_path=Path(results_path)/"lsv1994"/"submission"/"submission.html"
    print(f"Submission saved: {generate_submission(result_dirs,output_path=output_path,extension_dirs=extension_dirs)}")
