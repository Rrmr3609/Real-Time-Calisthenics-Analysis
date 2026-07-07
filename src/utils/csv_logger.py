import csv
from pathlib import Path
from typing import Dict, Optional


class CSVLogger:
    def __init__(self, output_path: str, fieldnames):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = fieldnames

        file_exists = self.output_path.exists()

        self.file = self.output_path.open("a", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames)

        if not file_exists:
            self.writer.writeheader()

    def write_row(self, row: Dict[str, Optional[float]]) -> None:
        self.writer.writerow(row)
        self.file.flush()

    def close(self) -> None:
        self.file.close()