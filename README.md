# Advanced Deep Hedging with FIRM

Adversarial deep-hedging framework combining an LSTM GAN for worst-case price paths with an actor–critic hedger to minimize 95% CVaR on option P&L. This repo now includes FIRM — a lightweight Framework for Invariant Risk Minimization — to improve robustness across market regimes.

What FIRM adds:
- Regime detection utilities (VIX/volatility-based) to label environments
- IRM penalty to encourage invariant predictors across regimes
- Per‑regime evaluation and simple analysis utilities

Key entry points:
- Baseline: `python -m src.deephedge.train`
- FIRM run: `python experiment_firm.py` (uses `configs/firm.yaml`)

## Quick start

```bash
pip install -e .[dev]
pytest -k "sanity or smoke" -q
python -m src.deephedge.train
```

## Experiment

Compare GAN-generated stress paths with geometric Brownian motion and evaluate 60-step episodes of an ATM call on the S&P 500, focusing on daily CVaR95 during crash periods. Next steps are to make the hedger identify real vs fake market patterns, improving robustness to regime changes like COVID via IRM.

### What has been done so far
- Uses a GAN (Generative Adversarial Networks) to generate worst-case (i.e., CVaR95) market scenarios
- Trains an Actor-Critic model to hedge options
- Tested on S&P 500 data including March 2020 COVID crash
- Calculated risk metrics (CVaR and P&L)
- Produced runs and results

## Core components

- **BlackScholesPricer** – pricing, Greeks, implied vol
- **DataManager** – market fetch + synthetic generation
- **LSTM GAN** – adversarial path generator and discriminator
- **ActorCriticHedger** – LSTM policy/value networks with CVaR reward
- **Hydra config & logging stack** – CLI overrides, W&B metrics, Matplotlib artifacts

FIRM-specific components:
- `src/firm/regime/detector.py` – simple VIX/volatility regime detector
- `src/firm/invariance/irm_loss.py` – IRM penalty implementation
- `src/deephedge/train_firm.py` – training loop with IRM penalty
- `src/deephedge/utils/regime_metrics.py` – per‑regime evaluation helpers

See `STRUCTURE.md` for the organized folder layout.

## Custom runs

```bash
python src/deephedge/train.py experiment.seed=123 training.gan.num_epochs=200
```

## Citation

Deep Hedging Experiment: GAN-Generated Worst-Case Paths vs Geometric Brownian Motion
