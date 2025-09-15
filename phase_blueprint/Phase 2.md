# Phase 2 – Hyper-Parameter Search & Sensitivity Analysis  
*(with Phase 1.5 Commitments)*

---

## 1. Purpose

Phase 2 is where we take the **locked experimental protocol from Phase 1.5** and explore the hyper-parameter space to optimise model performance. The goal is to find the best combination of learning and model settings that yield **stable, low-tail-risk performance** on the fixed dataset, without altering the evaluation framework established earlier.

---

## 2. Context from Phase 1.5 Commitments

Before Phase 2 starts, Phase 1.5 ensures that:
1. **Data Splits are Frozen**  
   - Training, validation, and test (crisis) windows are defined and encoded in YAML configs.  
   - Crisis windows (e.g., COVID-19, Dec-2018 QT) are *test-only* and never touched during tuning.

2. **Baselines are Established**  
   - C-static, C-dyn, RL-GBM, and RL-GAN baselines implemented and validated.  
   - All baselines run successfully on frozen splits with stable metrics.

3. **Metrics and Cost Models are Locked**  
   - Primary metric: CVaR@95 of final P&L.  
   - Secondary metrics: mean P&L, Sharpe, turnover, max drawdown.  
   - Transaction cost model fixed (bps + optional fixed cost).

4. **Reproducibility is Guaranteed**  
   - Seed control implemented.  
   - Unit tests for pricing, CVaR, and P&L accounting pass in CI.  
   - Smoke test runs green for all baselines.

**These commitments mean:** Phase 2 tuning will be comparable across runs, reproducible, and anchored to a trusted evaluation protocol.

---

## 3. Objectives of Phase 2

1. **Optimise Model Hyper-Parameters**  
   - Search for the best combination of parameters for:
     - Actor-Critic architecture (hidden size, LSTM layers)
     - Learning rate, entropy coefficient, gamma (discount)
     - GAN parameters (latent dim, adversarial loss weight λ)
     - Reward shaping parameters (CVaR α, Sharpe weighting)

2. **Assess Sensitivity**  
   - Determine how sensitive performance is to each parameter.
   - Identify stable regions in hyper-parameter space to avoid overfitting to noise.

3. **Select Candidate Models for Phase 3**  
   - Choose top-performing RL-GBM and RL-GAN variants to feed into Phase 3’s LLM signal integration.

---

## 4. Search Methodology

### 4.1 Optimisation Strategy
- **Search Tool**: Optuna (Bayesian optimisation with early stopping).
- **Objective Function**: Maximise negative CVaR@95 (minimise tail loss) on validation set.
- **Parallel Execution**: Use GPU farm / multi-core for multiple trials.

### 4.2 Search Space (Example)
```yaml
actor_critic:
  hidden_size: [64, 128, 256]
  lstm_layers: [1, 2]
  lr: loguniform(1e-5, 1e-3)
  entropy_coef: uniform(0.0, 0.05)
  gamma: uniform(0.90, 0.99)

gan:
  latent_dim: [16, 32, 64]
  adv_lambda: uniform(0.1, 1.0)

reward:
  cvar_alpha: [0.90, 0.95, 0.99]
  sharpe_weight: [0.0, 0.5]
```

---

## 5. Evaluation Framework

- **Validation Set**: Non-crisis validation window(s) from Phase 1.5.
- **Metric Logging**:
  - CVaR@95, mean P&L, Sharpe, turnover per trial.
  - Learning curves, P&L time series saved as plots.
- **Reproducibility**:
  - Hydra config snapshot for each trial.
  - Random seed logged and fixed across comparable runs.

---

## 6. Sensitivity Analysis

After best parameters are found:
1. Vary one parameter at a time (OAT analysis) around optimal value.
2. Produce tornado chart of metric change vs parameter change.
3. Identify robust ranges where small changes do not degrade performance significantly.

---

## 7. Deliverables

1. **Hyper-Parameter Search Report**  
   - Description of search space, optimisation method, and number of trials.
   - Tables of top 5 configurations for RL-GBM and RL-GAN.
   - Sensitivity analysis plots.

2. **Best Model Checkpoints**  
   - Saved PyTorch weights for top RL-GBM and RL-GAN models.
   - Associated Hydra configs frozen in `configs/final_models/`.

3. **Run Registry Update**  
   - Append best model run IDs, commit SHA, and metrics to registry CSV/W&B table.

4. **Phase 2 Summary Document**  
   - Ready for inclusion in main research paper.
   - Includes direct comparison to Phase 1 baselines using validation set.

---

## 8. Exit Criteria

Phase 2 is complete when:
- At least 100 trials have been run per model type (RL-GBM, RL-GAN).
- The top models achieve CVaR@95 improvement over their Phase 1 counterparts on validation set.
- Sensitivity analysis confirms chosen parameters are in stable regions.
- All best models and configs are reproducible via one command.

---

## 9. Transition to Phase 3

The top RL-GBM and RL-GAN configurations from Phase 2:
- Are **frozen** for use in Phase 3.
- Serve as the “base models” for adding LLM-driven causal features.
- Will be re-evaluated in Phase 4 on *test-only crisis periods* (COVID-19, QT 2018, etc.).