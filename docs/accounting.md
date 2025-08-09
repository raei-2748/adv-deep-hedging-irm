## Accounting and P&L Definitions (Protocol v1.0)

Let price S_t and option value V_t evolve on bar t. Hedge position H_t (in shares) is applied from t to t+1. Transaction costs apply on notional traded.

PnL_t = ΔV_t + ΔHedgeValue_t − Costs_t

Where:
- ΔV_t = V_{t+1} − V_t
- ΔHedgeValue_t = − H_t × (S_{t+1} − S_t)
- Costs_t = |H_{t+1} − H_t| × S_t × (bps/10000)

Positions roll bar to bar: H_{t+1} is set after observing state at t. Hedging frequency is every bar.


