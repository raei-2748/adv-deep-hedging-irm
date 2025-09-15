"""
FIRM training loop wrapper.

Uses regime detection and an IRM penalty during training.
This is a lightweight scaffold intended to integrate with the
existing deep hedging components without modifying them.
"""
from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
import torch

from deephedge.data.dataloader import DataManager
from deephedge.utils.seed import set_seed
from deephedge.utils.logger import getLogger if hasattr(__import__('deephedge.utils.logger', fromlist=['getLogger']), 'getLogger') else None  # safe import

from firm.regime.detector import RegimeDetector
from firm.invariance.irm_loss import irm_penalty_from_env_losses


def _get_logger():
    try:
        from deephedge.utils.logger import logger as _logger

        return _logger
    except Exception:
        class _Dummy:
            def info(self, *a, **k):
                print(*a)

        return _Dummy()


logger = _get_logger()


def make_environments(market_df, regimes: np.ndarray) -> List[np.ndarray]:
    envs = []
    for regime in np.unique(regimes):
        idx = regimes == regime
        envs.append(market_df["Returns"].values[idx])
    return envs


def main(config: Dict):
    seed = int(config.get("experiment", {}).get("seed", 123))
    set_seed(seed)

    logger.info("[FIRM] Starting training with config:")
    logger.info(str(config))

    # 1) Data
    dm = DataManager(
        start_date=config.get("data", {}).get("start_date", "2015-01-01"),
        end_date=config.get("data", {}).get("end_date", "2025-01-01"),
    )
    market_df, option_df = dm.get_data(synthetic=False)

    # 2) Regime labels
    detector = RegimeDetector(method=config.get("regimes", {}).get("method", "volatility_percentile"))
    regimes = detector.detect(
        market_df,
        low_pct=config.get("regimes", {}).get("low_pct", 0.3),
        high_pct=config.get("regimes", {}).get("high_pct", 0.7),
    )

    # 3) Dummy model loss per environment (placeholder)
    # In real training, compute task loss per environment from predictions.
    env_series = make_environments(market_df, regimes)
    env_losses = [torch.tensor(np.var(s) + 1e-6, requires_grad=True) for s in env_series if len(s) > 0]

    # 4) IRM penalty
    penalty = irm_penalty_from_env_losses(env_losses)
    irm_lambda = float(config.get("training", {}).get("irm_lambda", 1.0))
    total_obj = sum(env_losses) + irm_lambda * penalty

    logger.info(f"[FIRM] Num envs={len(env_losses)} | IRM penalty={float(penalty.item()):.6f}")
    logger.info(f"[FIRM] Objective (dummy)={float(total_obj.item()):.6f}")

    return {
        "num_envs": len(env_losses),
        "irm_penalty": float(penalty.item()),
        "objective": float(total_obj.item()),
    }


if __name__ == "__main__":
    main({})

