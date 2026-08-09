"""Centralise repository-relative locations used by operational runners.

``FIGURE_DIR`` and ``TABLE_DIR`` are legacy result roots with no tracked
consumer beyond directory creation. They remain unchanged here pending the
dedicated path-policy migration to the development/formal results hierarchy.
"""

from pathlib import Path


# Resolve the root from this tracked module, independently of the shell CWD.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
ANNOTATION_DIR = DATA_DIR / "annotations"

EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
LOG_DIR = EXPERIMENTS_DIR / "logs"
OUTPUT_DIR = EXPERIMENTS_DIR / "outputs"

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURE_DIR = RESULTS_DIR / "figures"
TABLE_DIR = RESULTS_DIR / "tables"


def create_project_directories() -> None:
    """Create every currently declared data, experiment and result directory.

    This includes the two legacy result roots documented at module level; their
    removal is deferred because all operational entry points call this function.
    """
    for directory in [
        RAW_DATA_DIR,
        ANNOTATION_DIR,
        LOG_DIR,
        OUTPUT_DIR,
        FIGURE_DIR,
        TABLE_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)
