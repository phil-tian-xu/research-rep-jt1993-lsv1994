from os.path import dirname, abspath
from pathlib import Path

root_path = dirname(dirname(abspath(__file__)))
core_path = Path(root_path, "core")
utils_path = Path(root_path, "utils")
data_path = Path(root_path, 'data')
results_path = Path(root_path, 'results')
experiments_path = results_path / 'experiments'
experiments_archive_path = results_path / 'experiments_archive'
config_path = Path(root_path, 'config')
log_path = Path(root_path, 'log')

