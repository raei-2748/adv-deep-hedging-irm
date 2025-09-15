# FIRM: Framework for Invariant Risk Minimalization

Adversarial deep-hedging framework combining an LSTM GAN for worst-case price paths with an actor–critic hedger to minimize 95% CVaR on option P&L

## Quick start

```bash
pip install -e .[dev]
pytest -k "sanity or smoke" -q
python -m src.deephedge.train
```

## Experiment

Compare GAN-generated stress paths with geometric Brownian motion.

Evaluate 60-step episodes of an ATM call on the S&P 500, focusing on daily CVaR95 during crash periods

## Core components

- **BlackScholesPricer** – pricing, Greeks, implied vol
- **DataManager** – market fetch + synthetic generation
- **LSTM GAN** – adversarial path generator and discriminator
- **ActorCriticHedger** – LSTM policy/value networks with CVaR reward
- **Hydra config & logging stack** – CLI overrides, W&B metrics, Matplotlib artifacts

## Custom runs

```bash
python src/deephedge/train.py experiment.seed=123 training.gan.num_epochs=200
```

## Citation

Deep Hedging Experiment: GAN-Generated Worst-Case Paths vs Geometric Brownian Motion
