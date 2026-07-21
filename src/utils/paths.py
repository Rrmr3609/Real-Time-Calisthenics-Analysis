from pathlib import Path


# paths.py is inside: repository/src/utils/paths.py
# parents[0] = utils
# parents[1] = src
# parents[2] = repository root
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
    """Create generated-output directories when they do not exist."""
    for directory in [
        RAW_DATA_DIR,
        ANNOTATION_DIR,
        LOG_DIR,
        OUTPUT_DIR,
        FIGURE_DIR,
        TABLE_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)