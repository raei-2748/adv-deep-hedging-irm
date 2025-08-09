import csv
from pathlib import Path
from typing import Dict


def append_run_registry(row: Dict, registry_path: str = "runs/registry.csv"):
    path = Path(registry_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


