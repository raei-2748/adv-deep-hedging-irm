## Experimental Protocol v1.0 (Frozen)

This document freezes the experimental protocol for deep hedging prior to hyper-parameter search.

1) Scope and Principles
- Purpose: Fix dataset, splits, metrics, baselines, costs, and statistical tests. All future models must use these.
- Non-negotiables: Any change requires a new version (v1.x) and re-running comparisons.
- Separation: Tuning uses only train/valid; all crisis windows are test-only.

2) Data & Splits
- Asset: SPY OHLCV via `yfinance` with 8-minute synthetic intraday constructed from daily GBM; later replace with real intraday once available. Option surface via Black–Scholes using realized volatility.
- Resolution: 8-minute bars, fixed.
- Windows (frozen):
  - Train: 2015-01-01→2017-12-31; 2019-02→2019-11; 2021-01→2022-12
  - Validation: 2018-01→2018-06; 2020-06→2020-12
  - Test (crisis): COVID-19 (2020-02-15→2020-04-30), Volmageddon (2018-01-15→2018-02-28), QT18 (2018-10-01→2018-12-31), SVB (2023-03), Hikes22 (2022-06→2022-07)
- Deterministic materialization: All datasets saved as Parquet with MD5 content hash recorded as `runs/<timestamp>/dataset_card.json`.
- Time alignment: Features at time t use only data ≤ t. Rolling vols use close-to-close over specified window.

3) Baselines
- C-static: BS delta hedger with constant vol.
- C-dyn: BS delta hedger with rolling realized vol (window=30 bars).
- RL-GBM: Actor-Critic trained on GBM.
- RL-GAN: Current adversarially trained hedger.
Each baseline is tied to a Git SHA and config snapshot. A smoke unit test must finish a 1-episode run with finite P&L.

4) Market frictions & accounting
- Transaction costs: proportional bps on notional traded: cost = |Δpos| × price × bps/10000. Defaults: {0, 2, 10} bps.
- Slippage/impact: off by default.
- Financing: rf fixed at 0 unless otherwise specified.
- Hedging frequency: every bar (8 minutes).
- P&L bookkeeping: PnL_t = ΔCash_t + ΔOptionValue_t + ΔHedgeValue_t − Costs_t. Positions roll bar to bar.

5) Metrics
- Primary: CVaR@95 of episode P&L distribution. VaR@95 = x_(k) with k=⌊0.05N⌋ on sorted ascending P&L; CVaR@95 = mean of first k values.
- Secondary: Mean P&L, Sharpe (excess over rf), Sortino, Max Drawdown, Turnover, Hedging error RMSE.
- Aggregation: Per-window and pooled; 95% bootstrap CIs (B=10,000).

6) Statistical testing & power
- H0: CVaR95(RL-GAN) − CVaR95(C-dyn) = 0 (per window). H1: Difference < 0 (one-sided).
- Test: Paired bootstrap over episode P&Ls; combine p-values via Fisher or apply Holm–Bonferroni across windows.
- Power: Target 10–20% improvement detection; estimate seeds/episodes to reach ≥80% power.

7) Reproducibility
- Seeds: single master seed, derive sub-seeds; log ledger.
- Environment: use `requirements.lock` and Docker image tag; log Python/CUDA/torch versions.
- Config immutability: save resolved Hydra config per run.

8) Sanity tests (pre-Phase 2)
- BS Greeks accuracy, P&L accounting sanity, CVaR correctness against known mixture, no leakage test.

9) Governance
- Tag: `protocol-v1.0`. Changes via `docs/protocol_change_request.md`, bump version.
- Run registry: CSV listing model_id, data_window, tc_bps, seed, commit_sha, docker_tag, metrics.

10) Exit criteria
- All sanity and smoke tests green; stable baselines across ≥10 seeds; statistical pipeline yields interpretable CIs; artifacts reproducible from clean checkout and Docker image.


