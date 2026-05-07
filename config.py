"""
ARES configuration — central place for all tunable constants.
Edit this file to change universe, capital, risk limits and cost assumptions.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import List


@dataclass
class UniverseConfig:
    """Tradeable assets and their IV-proxy fallback chains."""
    assets: List[str] = field(default_factory=lambda: ["SPY", "QQQ", "GLD", "TLT"])
    meta_assets: List[str] = field(default_factory=lambda: ["EFA", "EEM", "HYG", "LQD", "USO"])
    iv_proxies: dict = field(default_factory=lambda: {
        "SPY": ["^VIX"],
        "QQQ": ["^VXN", "^VIX"],
        "GLD": ["^GVZ", "^VIX"],
        "TLT": ["^VIX"],
    })
    benchmark: str = "SPY"


@dataclass
class CapitalConfig:
    paper_capital: float = 1_000_000.0
    max_position_pct: float = 0.20       # max 20% NAV per position
    max_gross_leverage: float = 2.0      # max gross notional / NAV
    min_trade_pnl_threshold: float = 50  # USD — ignore below this


@dataclass
class RiskConfig:
    max_drawdown_pct: float = 0.15       # halt if DD > 15%
    vol_target_ann: float = 0.12         # 12% annualised portfolio vol
    kelly_fraction: float = 0.28         # Kelly légèrement augmenté — VRP edge empirique > 0.25
    strategy_leverage_multiplier: float = 1.0  # global sizing multiplier for all sleeves
    max_position_pct: float = 0.20       # max 20% NAV per position
    max_gross_leverage: float = 2.0      # max gross notional / NAV
    max_delta_per_asset: float = 500.0   # SPY-equivalent shares
    max_vega_per_asset: float = 50_000   # USD vega
    max_corr_cluster: float = 0.70       # max intra-portfolio correlation
    lookback_dd: int = 63                # rolling window for live DD check (days)


@dataclass
class CostConfig:
    commission_per_contract: float = 0.65   # USD
    spread_vol_pts: float = 0.30            # IV spread in vol points
    delta_hedge_slippage_bps: float = 1.0   # bps on notional


@dataclass
class ModelConfig:
    """Walk-forward / training parameters."""
    train_window_days: int = 756            # ~3 years
    oos_start: str = "2020-01-01"
    refit_freq_days: int = 5
    har_min_obs: int = 120
    lgbm_n_estimators: int = 500
    lgbm_learning_rate: float = 0.02
    lgbm_max_depth: int = 5
    lgbm_num_leaves: int = 31
    lgbm_feature_fraction: float = 0.80
    lgbm_bagging_fraction: float = 0.80
    lgbm_bagging_freq: int = 5
    lstm_seq_len: int = 21
    lstm_hidden: int = 64
    lstm_epochs: int = 50
    lstm_batch: int = 32
    hmm_n_states: int = 3
    hmm_n_iter: int = 200
    vrp_zscore_entry: float = 0.80          # enter when |Z| > this (z=0.80 = top ~21% VRP observations)
    vrp_zscore_max: float = 3.2             # élargi — capture pics de VRP en début de crise avant retour
    vrp_zscore_exit: float = 0.0
    # Regime-aware VRP controls (modern market microstructure)
    vrp_min_vol_spread_short: float = 0.010  # IV - RV must exceed 1.0 vol points (élargi pour 9 actifs)
    vrp_min_vol_spread_long: float = 0.022   # RV - IV must exceed 2.2 vol points to long vol
    vrp_max_rv_percentile_short: float = 0.90  # élargi pour capturer plus de signaux
    vrp_max_holding_days: int = 21           # 1 mois complet de theta decay (augmenté)
    vrp_cooldown_days: int = 1
    vrp_allow_long_vol_hedge: bool = True    # activer long vol en high_vol regime
    vrp_stop_loss_notional_pct: float = 0.020  # 2% notional — plus de room pour la position
    vrp_take_profit_notional_pct: float = 0.018  # 1.8% TP — laisser courir les gagnants
    vrp_shock_exit_multiplier: float = 1.5
    vrp_trading_enabled: bool = True
    adaptive_learning_enabled: bool = True
    backtest_deterministic: bool = True
    backtest_seed: int = 42
    # Taux sans risque annualisé par période (Fed funds historiques)
    risk_free_rate_ann: float = 0.025   # défaut 2.5% (moyenne long terme)
    risk_free_rate_by_period: dict = field(default_factory=lambda: {
        "2015-01-01": 0.010,   # Fed funds ~0-1% 2015-2019
        "2020-01-01": 0.005,   # Near-zero post-COVID
        "2022-01-01": 0.045,   # Cycle de hausse FOMC 2022-2024
    })
    adaptive_persist_state_in_backtest: bool = False
    adaptive_recalibration_days: int = 21
    adaptive_min_trades: int = 30
    adaptive_lookback_trades: int = 150
    # Adaptive PF-first guardrails: prevent learner from degrading trade quality.
    adaptive_pf_floor: float = 1.20
    adaptive_win_rate_floor: float = 0.52
    adaptive_offensive_pf_threshold: float = 1.60
    adaptive_offensive_win_rate_threshold: float = 0.66
    adaptive_min_short_trades_for_offensive: int = 40
    adaptive_vrp_zscore_entry_floor: float = 0.75
    adaptive_vrp_min_vol_spread_short_floor: float = 0.008  # sync avec vrp_min_vol_spread_short=0.010
    # Momentum/carry réduits pour que le VRP devienne l'alpha dominant
    # Les sleeves synthétiques sont complètes en backtest mais lourdes en live (slippage)
    meta_momentum_position_pct: float = 0.07   # 7% NAV par jambe (augmenté de 5%)
    meta_carry_position_pct: float = 0.09      # 9% NAV par jambe (augmenté de 7%)
    meta_momentum_stop_loss_pct: float = 0.025 # stop serré
    meta_momentum_max_holding_days: int = 20
    meta_carry_max_holding_days: int = 35
    meta_carry_neutral_exit_days: int = 4


@dataclass
class AresConfig:
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    capital: CapitalConfig = field(default_factory=CapitalConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data_dir: str = "~/.ares/data"
    log_level: str = "INFO"


# Singleton default config
DEFAULT_CONFIG = AresConfig()
