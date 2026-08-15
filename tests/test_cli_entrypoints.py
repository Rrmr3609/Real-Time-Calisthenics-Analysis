import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ENTRYPOINTS = (
    "src/main.py",
    "src/run_live_enhanced.py",
    "src/run_video.py",
    "src/run_video_enhanced.py",
    "src/validate_dataset.py",
    "src/annotate_repetitions.py",
    "src/run_formal_evaluation.py",
)


@pytest.mark.parametrize("entrypoint", ENTRYPOINTS)
def test_cli_help_works_without_pythonpath(entrypoint):
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)

    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / entrypoint), "--help"],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
