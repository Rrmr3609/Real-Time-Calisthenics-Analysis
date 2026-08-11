import csv

import pytest

from utils.csv_logger import CSVLogger, ensure_output_paths_available

FIELDNAMES = ["clip_id", "value"]


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.reader(file))


def test_new_csv_contains_header_before_any_rows(tmp_path):
    output_path = tmp_path / "new.csv"

    logger = CSVLogger(
        output_path=str(output_path),
        fieldnames=FIELDNAMES,
    )
    logger.close()

    assert read_rows(output_path) == [FIELDNAMES]


def test_existing_csv_fails_without_overwrite_and_is_unchanged(
    tmp_path,
):
    output_path = tmp_path / "existing.csv"
    original_content = "clip_id,value\nold,1\n"
    output_path.write_text(original_content, encoding="utf-8")

    with pytest.raises(FileExistsError, match="--overwrite"):
        CSVLogger(
            output_path=str(output_path),
            fieldnames=FIELDNAMES,
        )

    assert output_path.read_text(encoding="utf-8") == original_content


def test_overwrite_replaces_existing_csv_instead_of_appending(
    tmp_path,
):
    output_path = tmp_path / "existing.csv"
    output_path.write_text(
        "clip_id,value\nold,1\n",
        encoding="utf-8",
    )

    logger = CSVLogger(
        output_path=str(output_path),
        fieldnames=FIELDNAMES,
        overwrite=True,
    )
    logger.write_row({"clip_id": "new", "value": 2})
    logger.close()

    assert read_rows(output_path) == [
        FIELDNAMES,
        ["new", "2"],
    ]


def test_zero_byte_csv_requires_overwrite_and_receives_header(
    tmp_path,
):
    output_path = tmp_path / "empty.csv"
    output_path.touch()

    with pytest.raises(FileExistsError, match="--overwrite"):
        CSVLogger(
            output_path=str(output_path),
            fieldnames=FIELDNAMES,
        )

    logger = CSVLogger(
        output_path=str(output_path),
        fieldnames=FIELDNAMES,
        overwrite=True,
    )
    logger.close()

    assert read_rows(output_path) == [FIELDNAMES]


def test_output_set_preflight_checks_every_path_before_creation(
    tmp_path,
):
    new_path = tmp_path / "new.csv"
    stale_path = tmp_path / "stale.csv"
    stale_path.write_text("stale", encoding="utf-8")

    with pytest.raises(FileExistsError, match="stale.csv"):
        ensure_output_paths_available(
            [new_path, stale_path],
            overwrite=False,
        )

    assert not new_path.exists()
    assert stale_path.read_text(encoding="utf-8") == "stale"


def test_close_is_idempotent(tmp_path):
    logger = CSVLogger(
        output_path=str(tmp_path / "output.csv"),
        fieldnames=FIELDNAMES,
    )

    logger.close()
    logger.close()

    assert logger.file.closed
