import hashlib
import json
from pathlib import Path
from typing import Dict

import pandas as pd


def _md5_bytes(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def save_with_hash(df: pd.DataFrame, out_path: str, card_path: str, meta: Dict):
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Try Parquet first, fall back to CSV automatically
    file_used = out
    try:
        df.to_parquet(out)
    except Exception:
        file_used = out.with_suffix(".csv")
        df.to_csv(file_used, index=True)

    file_hash = _md5_bytes(file_used)

    card = {
        "file": str(file_used),
        "md5": file_hash,
        "rows": int(df.shape[0]),
        "cols": list(map(str, df.columns)),
    }
    card.update(meta)

    # Append to dataset_card.json (list of entries)
    card_file = Path(card_path)
    card_file.parent.mkdir(parents=True, exist_ok=True)
    if card_file.exists():
        prev = json.loads(card_file.read_text() or "[]")
        if isinstance(prev, list):
            prev.append(card)
            card_file.write_text(json.dumps(prev, indent=2))
        else:
            card_file.write_text(json.dumps([prev, card], indent=2))
    else:
        card_file.write_text(json.dumps([card], indent=2))


