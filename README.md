# ARES — Algorithmic Trading System

**A**utonomous **R**ealised-vol **E**dge **S**ystem

Multi-sleeve derivatives strategy combining Volatility Risk Premium capture,
HMM-based regime detection, and cross-asset momentum.

---

## Performance — Holdout 2025 (never seen during development)

| Metric | Value |
|--------|-------|
| **Sharpe Ratio** (RF-adjusted, Lo 2002) | **1.883** |
| **P-value** | **0.029** |
| **Max Drawdown** | **1.21 %** |
| **Ann. Return** (leveraged 3.5×) | **+14.0 %** |
| **Ann. Return** (unlevered) | **+4.0 %** |
| **Trades / year** | **37.7** |
| **Profit Factor (VRP sleeve)** | **2.61** |
| **Bootstrap CI 95 %** | [0.19 ; 2.47] |

> The holdout period (2025) was **locked before inspection** and evaluated
> in a single pass. Parameters were frozen on the training set (2015–2019)
> only. No contamination between train / validation / test sets.

---

## Strategy Overview

ARES combines three independent alpha sources:

### 1. VRP — Volatility Risk Premium
- Enter short when implied vol (IV proxy) exceeds realised vol (HAR-RV)
  by a z-score threshold
- Exit on theta decay, stop-loss, or signal reversal
- Long-vol hedge activated in high-vol regime (crisis protection)

### 2. HMM Regime Detection
- 3-state Gaussian HMM: `low_vol` / `mid_vol` / `high_vol`
- Trained on 5 features: RV22, vol momentum, log-RV, IV, term-structure slope
- Regime probability drives continuous position sizing (`regime_scale`)
- Correctly identifies bear markets (2022 rate cycle classified as `mid_vol`)

### 3. Cross-Asset Momentum + Carry
- Momentum: Jegadeesh-Titman 12-1 on ETF basket
- Carry: yield differential across fixed-income assets
- Risk-adjusted scoring with trend filter and stop-loss

### Meta-Allocator
Mean-variance optimisation with shrunk covariance matrix across sleeves.
VRP floor at 30 % — ensures the statistical edge is never crowded out.

---

## Validation Methodology

Strict protocol to prevent overfitting and data contamination:

```
Train      2015-01-01 → 2019-12-31   Parameter calibration
Val        2020-01-01 → 2021-12-31   Architecture selection
Test       2022-01-01 → 2024-12-31   ⚠️ Contaminated (seen twice)
Holdout    2025-01-01 → 2025-12-31   ✅ Single pass — true OOS
```

Key validation tools (in `validation/`):
- **Contamination checker** — verifies no data leakage across periods
- **Walk-forward engine** — strict single-pass OOS evaluation
- **Sharpe IC** — bootstrap confidence intervals via Lo (2002)
- **Lock files** — prevent re-running test set after first evaluation

---

## Architecture

```
ares/
├── config.py              # All tunable parameters — single source of truth
├── strategy_engine.py     # Main orchestrator (backtest / paper / live)
├── ai/
│   └── regime_engine.py   # HMM regime detection (3 states)
└── backtest/
    └── walkforward_ares.py  # Strict walk-forward validation

validation/
├── contamination_checker.py
├── diag_pf_trainonly.py
├── sweep_zscore_trainonly.py
├── sharpe_ic.py
└── walkforward_strict.py

main.py                    # CLI entry point
```

---

## Quick Start

### Backtest
```bash
python main.py --mode backtest --start 2015-01-01 --end 2024-12-31
```

### Walk-forward validation
```bash
python ares/backtest/walkforward_ares.py \
  --leverage-multiplier 3.5 \
  --log-level INFO
```

### Paper trading (real data, simulated execution)
```bash
python main.py --mode paper --poll 300
```

### Live trading (IBKR connection)
```bash
python main.py --mode live \
  --ibkr-host 127.0.0.1 \
  --ibkr-port 7497 \
  --client-id 1
```

---

## Key Parameters (`ares/config.py`)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `vrp_zscore_entry` | 0.80 | VRP entry threshold |
| `vrp_zscore_max` | 3.2 | Cap — avoids entering during active crisis |
| `hmm_n_states` | 3 | Low / Mid / High vol regimes |
| `kelly_fraction` | 0.28 | Fractional Kelly sizing |
| `strategy_leverage_multiplier` | 1.0 | Global sizing multiplier |
| `max_drawdown_pct` | 0.15 | Portfolio halt threshold |
| `backtest_deterministic` | True | Fixed seed — fully reproducible |

---

## Stack

```
Python 3.10+
pandas · numpy · scipy · sklearn
hmmlearn          # HMM regime detection
yfinance          # Market data
lightgbm          # Ensemble model (HAR-RV enhancement)
ibapi             # IBKR live execution (optional)
```

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/ares-trading-system
cd ares-trading-system
pip install -r requirements.txt
python main.py --mode backtest --start 2020-01-01 --end 2024-12-31
```

---

## Results by Period

| Period | Ann. Return | Sharpe | Max DD | Status |
|--------|-------------|--------|--------|--------|
| Train 2015–2019 | +10.25 % | 2.20 | 3.00 % | In-sample |
| Val 2020–2021 | +14.06 % | 2.34 | 2.46 % | Architecture selection |
| Holdout 2025 | **+14.0 %** | **1.883** | **1.21 %** | ✅ True OOS |
| COVID stress (Jan–Apr 2020) | +25.3 % | 4.83 | 1.10 % | Stress test |
| Bear market 2022 | +12.5 % | 1.64 | 4.25 % | Stress test |

---

## Reproducibility

All backtests are fully deterministic:
```python
backtest_deterministic = True
backtest_seed = 42
```

Run 7 and Run 8 on identical parameters produce identical NAV to the cent.

---

*Built independently — no formal graduate training in quantitative finance.*
*All mathematical frameworks (HMM, HAR-RV, Kelly, Bootstrap CI) implemented from scratch.*
