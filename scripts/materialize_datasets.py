# scripts/materialize_datasets.py
#!/usr/bin/env python3
from pathlib import Path
from datetime import datetime

import pandas as pd
from omegaconf import OmegaConf

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from deephedge.data.dataloader import DataManager
from deephedge.utils.dataset import save_with_hash


def slice_by_dates(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)
    return df.loc[str(start):str(end)]


def main():
    cfg = OmegaConf.load("configs/base.yaml")
    dm = DataManager(start_date=cfg.data.start_date, end_date=cfg.data.end_date)

    # Build full dataset once
    market = dm.fetch_sp500_data(
        symbol=cfg.data.ticker,
        interval=cfg.data.interval,
        sequence_length=cfg.model.gan.sequence_length,
    )
    option = dm.calculate_option_prices(market)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(cfg.logging.log_dir) / ts
    card_path = run_dir / "dataset_card.json"

    # Train splits
    for span in cfg.data.splits.train:
        m = slice_by_dates(market, span.start, span.end)
        o = option.loc[m.index]
        save_with_hash(
            m,
            out_path=str(run_dir / f"train_market_{span.start}_{span.end}.parquet"),
            card_path=str(card_path),
            meta={"split": "train", "asset": cfg.data.ticker, "start": span.start, "end": span.end, "kind": "market"},
        )
        save_with_hash(
            o,
            out_path=str(run_dir / f"train_option_{span.start}_{span.end}.parquet"),
            card_path=str(card_path),
            meta={"split": "train", "asset": cfg.data.ticker, "start": span.start, "end": span.end, "kind": "option"},
        )

    # Valid splits
    for span in cfg.data.splits.valid:
        m = slice_by_dates(market, span.start, span.end)
        o = option.loc[m.index]
        save_with_hash(
            m,
            out_path=str(run_dir / f"valid_market_{span.start}_{span.end}.parquet"),
            card_path=str(card_path),
            meta={"split": "valid", "asset": cfg.data.ticker, "start": span.start, "end": span.end, "kind": "market"},
        )
        save_with_hash(
            o,
            out_path=str(run_dir / f"valid_option_{span.start}_{span.end}.parquet"),
            card_path=str(card_path),
            meta={"split": "valid", "asset": cfg.data.ticker, "start": span.start, "end": span.end, "kind": "option"},
        )

    # Test (crisis windows)
    for span in cfg.data.splits.test:
        m = slice_by_dates(market, span.start, span.end)
        o = option.loc[m.index]
        name = span.name
        save_with_hash(
            m,
            out_path=str(run_dir / f"test_{name}_market_{span.start}_{span.end}.parquet"),
            card_path=str(card_path),
            meta={"split": "test", "window": name, "asset": cfg.data.ticker, "start": span.start, "end": span.end, "kind": "market"},
        )
        save_with_hash(
            o,
            out_path=str(run_dir / f"test_{name}_option_{span.start}_{span.end}.parquet"),
            card_path=str(card_path),
            meta={"split": "test", "window": name, "asset": cfg.data.ticker, "start": span.start, "end": span.end, "kind": "option"},
        )

    print(f"Wrote dataset files and card: {card_path}")


if __name__ == "__main__":
    main()
