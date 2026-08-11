"""Centralise repository-relative locations used by operational runners.

Runner startup creates only the experiment directories that receive runtime
outputs. Result paths are selected by the relevant evaluation command rather
than created globally by operational runners.
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


def create_project_directories() -> None:
    """Create the experiment directories used by operational runners."""
    for directory in [
        LOG_DIR,
        OUTPUT_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)
