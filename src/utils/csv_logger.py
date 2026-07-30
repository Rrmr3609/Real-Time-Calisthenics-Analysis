import csv
from pathlib import Path
from typing import Iterable, Mapping


def ensure_output_paths_available(
    output_paths: Iterable[str | Path],
    overwrite: bool = False,
) -> None:
    """
    Fail before a run starts if any requested output already exists.

    Callers that manage more than one output should pass the complete set so
    one stale file cannot be combined with a newly created file.
    """
    if overwrite:
        return

    existing_paths = [
        Path(output_path)
        for output_path in output_paths
        if Path(output_path).exists()
    ]

    if not existing_paths:
        return

    formatted_paths = "\n".join(
        f"- {path}" for path in existing_paths
    )
    raise FileExistsError(
        "Output file already exists. Refusing to append or replace it:\n"
        f"{formatted_paths}\n"
        "Re-run with --overwrite to replace every requested output."
    )


def prepare_output_paths(
    output_paths: Iterable[str | Path],
    overwrite: bool = False,
) -> None:
    """
    Preflight one complete output set before processing starts.

    With explicit overwrite permission, every existing file in the set is
    removed together before any new logger or metadata writer is created.
    """
    paths = [Path(output_path) for output_path in output_paths]

    if len(paths) != len(set(paths)):
        raise ValueError(
            "Output paths for one run must be unique"
        )

    ensure_output_paths_available(
        paths,
        overwrite=overwrite,
    )

    if not overwrite:
        return

    non_files = [
        path
        for path in paths
        if path.exists() and not path.is_file()
    ]

    if non_files:
        formatted_paths = "\n".join(
            f"- {path}" for path in non_files
        )
        raise IsADirectoryError(
            "Cannot overwrite non-file output paths:\n"
            f"{formatted_paths}"
        )

    for path in paths:
        if path.exists():
            path.unlink()


class CSVLogger:
    def __init__(
        self,
        output_path: str,
        fieldnames,
        overwrite: bool = False,
    ):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = fieldnames
        self.file = None

        ensure_output_paths_available(
            [self.output_path],
            overwrite=overwrite,
        )

        mode = "w" if overwrite else "x"

        try:
            self.file = self.output_path.open(
                mode,
                newline="",
                encoding="utf-8",
            )
            self.writer = csv.DictWriter(
                self.file,
                fieldnames=self.fieldnames,
            )
            self.writer.writeheader()
            self.file.flush()
        except FileExistsError as error:
            raise FileExistsError(
                "Output file already exists. Refusing to append or replace "
                f"it: {self.output_path}. Re-run with --overwrite to replace "
                "the output."
            ) from error
        except Exception:
            if self.file is not None:
                self.file.close()
            raise

    def write_row(self, row: Mapping[str, object]) -> None:
        self.writer.writerow(row)
        self.file.flush()

    def close(self) -> None:
        if self.file is not None and not self.file.closed:
            self.file.close()
