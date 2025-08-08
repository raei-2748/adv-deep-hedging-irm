# Phase 1

Parent item: Updated Next Step Discussions (https://www.notion.so/Updated-Next-Step-Discussions-24341e16dc04802fa04cf291df2033ee?pvs=21)
Status: In progress

### **Phase 1 – Robust Baselines & Ablations**

*(Objective: establish indisputable evidence that any performance gain later in the project comes from the new ideas—not from lucky seeds, lax cost assumptions, or cherry-picked metrics.)*

---

## 1. Scientific Objectives

| # | Objective | Rationale |
| --- | --- | --- |
| 1 | **Quantify the incremental value of deep RL versus classical delta hedging.** | Shows whether modern RL techniques truly add risk-reduction benefits beyond analytical formulas. |
| 2 | **Isolate the extra benefit of adversarial (GAN) training over standard GBM training.** | Prevents attributing GAN gains to RL architecture alone. |
| 3 | **Assess the sensitivity of results to transaction-cost modelling, hedge frequency, and reward shaping.** | Ensures robustness; regulatory reviewers will demand this. |

---

## 2. Model Tracks to Implement

| Track | Variant ID | Description | Key Hyper-parameters |
| --- | --- | --- | --- |
| **Classical** | **C-static** | Static Black-Scholes delta hedger (no parameter updates). | None |
|  | **C-dyn** | Dynamic delta hedger: uses rolling 30-min realised volatility and linear TC optimised position sizes. | Window length, TC slope |
| **Deep RL (GBM)** | **RL-GBM** | Actor-Critic hedger trained only on GBM paths. | hidden_size, lr, entropy_coef |
| **Deep RL + adversary** | **RL-GAN** | *Current flagship*: Actor-Critic trained on GAN worst-case paths. | All above + adv_λ |

*Optional Stretch:* RL with diffusion paths (**RL-Diffusion**) to test alternative adversary.

---

## 3. Experimental Matrix

| Factor | Levels |
| --- | --- |
| Transaction Costs | 0 bps, 2 bps, 10 bps |
| Hedge Frequency | 1 min, 8 min (baseline), 30 min |
| Reward Function | −CVaR95 (baseline), mean–variance, Sharpe surrogate |

*Total core runs = 4 model variants × 3 TC levels × 3 frequencies × 1 reward = 36 runs.*

Ablation runs (alternate reward) add 12 more → **48 runs**.

---

## 4. Metrics & Reporting

| Category | Metric | Calculation Notes |
| --- | --- | --- |
| Tail risk | **CVaR 95 & CVaR 99** | Historical (non-parametric) |
| Profitability | Mean P&L | Daily average |
| Risk-adj. | Sharpe, Sortino | Excess over Rf=0 |
| Liquidity | Turnover (shares traded / notional) | Adds realism |
| Stability | Max Drawdown, Std-Err across 10 seeds | Ensures repeatability |

All metrics **logged per episode** → aggregated to run-level CSV; visualised in seaborn bar plots with 95 % CI.

---

## 5. Implementation Tasks

| Task | File / Module | Owner | Done When |
| --- | --- | --- | --- |
| Refactor delta hedger into `models/hedge_delta.py` | code | You | ✓ standalone class |
| Add dynamic volatility estimator | `utils/vol.py` | You | returns σ_rolling |
| Integrate transaction-cost parameters into env | `envs/deep_hedging_env.py` | You | env accepts `tc_bps` |
| Hydra configs for each variant | `configs/model/*.yaml` | You | C-static etc. |
| `run_matrix.py` script to launch 48 jobs | new | You | loops over Hydra overrides |
| Metrics logger (`metrics.py`) | update | You | writes CSV per run |
| Aggregation notebook / script | `notebooks/phase1_analysis.ipynb` | You | plots & summary tables |

---

## 6. Reproducibility Requirements

1. **Hydra overrides list** stored in `runs/<timestamp>/overrides.txt`.
2. **Seed** recorded in config and output folder (handled in Phase 0.3).
3. Each run uploads to **Weights & Biases** project **deep_hedging/phase1** (or MLflow) with tag = variant ID.

---

## 7. Decision Gates

| Gate | Pass Criterion |
| --- | --- |
| G-1: RL viability | RL-GBM must beat C-static on CVaR 95 by ≥ 15 % at baseline TC. |
| G-2: GAN value-add | RL-GAN must beat RL-GBM on CVaR 95 by ≥ 10 % with no Sharpe penalty. |
| G-3: Robustness | Gains persist (same direction) in ≥ 80 % of ablative settings. |

Failing any gate means revisit architecture or adversary design before proceeding to Phase 2.

---

## 8. Deliverables

1. **Phase 1 Technical Report** (5–7 pp)
    - Methodology, tables, plots, interpretation against gates.
2. **Phase 1 Code Tag** (`phase1_complete`) in Git.
3. **Saved Checkpoints** for each best-performing variant (top CVaR95).

---

## 9. Time-boxed Timeline (≈ 3 weeks)

| Week | Milestone |
| --- | --- |
| 1 | Classical & RL-GBM implementations + smoke tests |
| 2 | RL-GAN training, env TC integration, run_matrix execution |
| 3 | Aggregation analysis, report writing, gate review meeting |

---

### **Key Take-away**

Phase 1 is your **baseline rocket booster**: build a bullet-proof comparison set, exhaustively test common criticisms (costs, frequency, reward), and lock in deterministic metrics. Only after clearing these statistical and engineering hurdles does it make sense to invest effort into hyperparameter sweeps (Phase 2) and LLM causal-signal integration (Phase 3).