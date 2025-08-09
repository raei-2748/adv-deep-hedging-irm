import hashlib
import json
from pathlib import Path
from typing import Dict

import pandas as pd


def md5_of_parquet(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.md5(data).hexdigest()


def save_with_hash(df: pd.DataFrame, out_path: str, card_path: str, meta: Dict):
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    file_hash = md5_of_parquet(out)

    card = {
        "file": str(out),
        "md5": file_hash,
        "rows": int(df.shape[0]),
        "cols": list(map(str, df.columns)),
    }
    card.update(meta)

    card_file = Path(card_path)
    card_file.parent.mkdir(parents=True, exist_ok=True)
    if card_file.exists():
        prev = json.loads(card_file.read_text())
        if isinstance(prev, list):
            prev.append(card)
            card_file.write_text(json.dumps(prev, indent=2))
        else:
            card_file.write_text(json.dumps([prev, card], indent=2))
    else:
        card_file.write_text(json.dumps([card], indent=2))


