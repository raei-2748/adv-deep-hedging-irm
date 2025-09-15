# FIRM Framework Structure

High-level organization of the repository:

- README.md: Project overview and quickstart
- LICENSE: Project license
- requirements.txt: Dependencies
- setup.py: Packaging shim (packages under `src/`)
- Makefile: Common tasks
- CLAUDE.md: Assistant guidance
- STRUCTURE.md: This document
- experiment.py: Baseline runner
- experiment_firm.py: FIRM runner

Configs
- configs/base.yaml: Baseline config
- configs/firm.yaml: FIRM config (IRM + regimes)
- configs/model/: Model variants including `irm.yaml`

Source
- src/deephedge/: Baseline code (kept intact)
  - data/regime_dataloader.py: Adds regime labels to data
  - train_firm.py: Training with IRM penalty
  - utils/regime_metrics.py: Per‑regime evaluation helpers
- src/firm/: FIRM framework additions
  - regime/detector.py, visualizer.py
  - invariance/irm_loss.py, feature_analyzer.py
  - framework/base_classes.py

Experiments
- experiments/baseline/: Pre‑FIRM results
- experiments/firm/week1/: Example runs grouped by IRM lambda
- experiments/comparisons/: Comparison plots

Tests
- tests/test_regime_detector.py: Regime detection tests
- tests/test_irm.py: IRM penalty tests

Data
- data/: CSV placeholders for SPY, VIX, and regime labels
