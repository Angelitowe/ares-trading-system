"""
ARES Strategy Engine — Main Orchestrator.

This is the central coordinator of the entire ARES system.
It wires together:
  - Data layer (MarketData)
  - Core models (HAR-RV-CJ, GARCH, Jump detection)
  - Options module (SVI surface, Greeks, VRP signal)
  - Signals (term structure, dispersion, gold)
  - AI layer (LightGBM ensemble, Regime engine, Geo-risk)
  - Risk Manager (Kelly sizing, drawdown halt)
  - Order Manager (paper fill or IBKR live)
  - Dashboard (reporting)

Run modes
---------
  1. BACKTEST  — run on historical data, generate full performance report
  2. PAPER     — live data, paper-trade via OrderManager simulation
  3. LIVE      — live data + IBKR execution (requires connect())

Quick start (backtest):
  engine = StrategyEngine(config=AresConfig(), mode="backtest")
  engine.run(start="2020-01-01", end="2024-12-31")

Quick start (paper):
  engine = StrategyEngine(config=AresConfig(), mode="paper")
  engine.run_live()

Quick start (live):
  engine = StrategyEngine(config=AresConfig(), mode="live")
  engine.connect_ibkr(host="127.0.0.1", port=7497)
  engine.run_live()
"""
from __future__ import annotations

import logging
import os
import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import math
import numpy as np
import pandas as pd

from .config import AresConfig, DEFAULT_CONFIG
from .data.market_data import MarketData
from .core.rv_estimators import RVEstimators
from .core.har_rv_model import HARRVModel
from .core.jump_detector import JumpDetector
from .core.garch_models import GARCHModels
from .options.svi_surface import SVISurface
from .options.greeks_engine import GreeksEngine
from .options.options_chain import OptionsChain, OptionContract
from .options.vrp_calculator import VRPCalculator
from .signals.term_structure import TermStructureSignal
from .signals.dispersion_signal import DispersionSignal
from .signals.gold_signal import GoldSignal
from .ai.lgbm_forecaster import LGBMForecaster
from .ai.regime_engine import RegimeEngine
from .ai.geo_risk import GeoRiskSignal
from .ai.adaptive_alpha import AdaptiveAlphaLearner, TradeFeedback
from .ai.meta_strategy import MetaStrategy
from .risk.risk_manager import RiskManager
from .execution.order_manager import OrderManager
from .execution.ibkr_connector import IBKRConnector
from .monitoring.dashboard import Dashboard
from world_monitor_feed import WorldMonitorFeed

logger = logging.getLogger(__name__)


class StrategyEngine:
    """
    ARES main strategy engine.

    Parameters
    ----------
    config : AresConfig — central configuration
    mode   : "backtest" | "paper" | "live"
    """

    def __init__(
        self,
        config: AresConfig = None,
        mode: str = "paper",
        log_level: str = "INFO",
    ) -> None:
        self.cfg = config or DEFAULT_CONFIG
        self.mode = mode.lower()
        assert self.mode in ("backtest", "paper", "live"), f"Unknown mode: {mode}"

        logging.basicConfig(
            level=getattr(logging, log_level.upper(), logging.INFO),
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )

        # --- Components ---
        self.data = MarketData()
        self.rv = RVEstimators()
        self.har = HARRVModel(min_obs=self.cfg.model.har_min_obs)
        self.jumps = JumpDetector()
        self.garch = GARCHModels()
        self.svi = SVISurface()
        self.greeks = GreeksEngine()
        self.option_chain = OptionsChain()
        self.vrp_calc = VRPCalculator(
            zscore_window=63,
            entry_z=self.cfg.model.vrp_zscore_entry,
            exit_z=self.cfg.model.vrp_zscore_exit,
        )
        self.term_struct = TermStructureSignal()
        self.dispersion = DispersionSignal()
        self.gold_sig = GoldSignal()
        self.lgbm = LGBMForecaster(
            n_estimators=self.cfg.model.lgbm_n_estimators,
            learning_rate=self.cfg.model.lgbm_learning_rate,
            max_depth=self.cfg.model.lgbm_max_depth,
        )
        self.regime = RegimeEngine(
            n_states=self.cfg.model.hmm_n_states,
            n_iter=self.cfg.model.hmm_n_iter,
        )
        self.geo_risk = GeoRiskSignal()
        self.risk = RiskManager(self.cfg.risk, self.cfg.capital.paper_capital)
        self.ibkr: Optional[IBKRConnector] = None
        self.orders = OrderManager(
            connector=self.ibkr,
            cost_config=self.cfg.cost,
            paper=(mode != "live"),
        )
        self.dashboard = Dashboard()

        # State
        self._nav_history: List[float] = [self.cfg.capital.paper_capital]
        self._current_signals: Dict[str, dict] = {}
        self._regime_state: str = "low_vol"
        self._regime_prob_high: float = 0.0
        self._geo_score: float = 0.0
        self._news_geo_score: float = 0.0
        self._lgbm_fitted: bool = False
        self._regime_fitted: bool = False
        self._macro_proxy_data: Dict[str, pd.Series] = {}
        self._world_monitor: Optional[WorldMonitorFeed] = None
        # Active position tracker for daily mark-to-market in backtest
        # {asset: {direction, contracts, notional, entry_iv, days_held, mtm_pnl}}
        self._active_vol_positions: Dict[str, dict] = {}
        self._entry_cooldown: Dict[str, int] = {}
        self._step_count: int = 0
        self._last_meta_vrp_weight: float = 1.0  # previous day meta VRP weight
        self._strategy_pnl_by_asset: Dict[str, float] = {}
        self._closed_vrp_trades: List[dict] = []
        self.alpha_learner = AdaptiveAlphaLearner(
            enabled=self.cfg.model.adaptive_learning_enabled,
            persist_state=(self.mode != "backtest") or self.cfg.model.adaptive_persist_state_in_backtest,
        )
        self.meta_strategy = MetaStrategy(
            momentum_position_pct=self.cfg.model.meta_momentum_position_pct * self.cfg.risk.strategy_leverage_multiplier,
            carry_position_pct=self.cfg.model.meta_carry_position_pct * self.cfg.risk.strategy_leverage_multiplier,
            momentum_stop_loss_pct=self.cfg.model.meta_momentum_stop_loss_pct,
            momentum_max_holding_days=self.cfg.model.meta_momentum_max_holding_days,
            carry_max_holding_days=self.cfg.model.meta_carry_max_holding_days,
            carry_neutral_exit_days=self.cfg.model.meta_carry_neutral_exit_days,
            enable_momentum=True,
            enable_carry=True,
        )
        if self.mode in ("paper", "live") and os.getenv("ARES_ENABLE_WORLD_MONITOR", "1") != "0":
            self._world_monitor = WorldMonitorFeed()

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect_ibkr(
        self,
        host: str = "127.0.0.1",
        port: int = 7497,
        client_id: int = 1,
        live: bool = False,
    ) -> "StrategyEngine":
        """
        Connect to IBKR TWS / Gateway.
        Must be called before run_live() in live mode.
        """
        self.ibkr = IBKRConnector(host=host, port=port, client_id=client_id, live=live)
        self.ibkr.connect()
        self.orders.connector = self.ibkr
        self.orders.paper = not live
        logger.info("IBKR connected: %s:%d [%s]", host, port, "LIVE" if live else "PAPER")
        return self

    # ------------------------------------------------------------------
    # Backtest
    # ------------------------------------------------------------------

    def _reset_backtest_state(self) -> None:
        self._nav_history = [self.cfg.capital.paper_capital]
        self._current_signals = {}
        self._regime_state = "low_vol"
        self._regime_prob_high = 0.0
        self._geo_score = 0.0
        self._news_geo_score = 0.0
        self._lgbm_fitted = False
        self._regime_fitted = False
        self._macro_proxy_data = {}
        self._active_vol_positions = {}
        self._entry_cooldown = {}
        self._step_count = 0
        self._last_meta_vrp_weight = 1.0
        self._strategy_pnl_by_asset = {}
        self._closed_vrp_trades = []
        self.risk = RiskManager(self.cfg.risk, self.cfg.capital.paper_capital)
        self.orders = OrderManager(
            connector=self.ibkr,
            cost_config=self.cfg.cost,
            paper=(self.mode != "live"),
        )
        self.meta_strategy = MetaStrategy(
            momentum_position_pct=self.cfg.model.meta_momentum_position_pct * self.cfg.risk.strategy_leverage_multiplier,
            carry_position_pct=self.cfg.model.meta_carry_position_pct * self.cfg.risk.strategy_leverage_multiplier,
            momentum_stop_loss_pct=self.cfg.model.meta_momentum_stop_loss_pct,
            momentum_max_holding_days=self.cfg.model.meta_momentum_max_holding_days,
            carry_max_holding_days=self.cfg.model.meta_carry_max_holding_days,
            carry_neutral_exit_days=self.cfg.model.meta_carry_neutral_exit_days,
            enable_momentum=True,
            enable_carry=True,
        )

    def run(
        self,
        start: str = "2020-01-01",
        end: Optional[str] = None,
        verbose: bool = True,
    ) -> Dashboard:
        """
        Run ARES backtest over history.
        Returns Dashboard with performance report.
        """
        if end is None:
            end = datetime.today().strftime("%Y-%m-%d")

        self._reset_backtest_state()

        all_assets = list(dict.fromkeys(self.cfg.universe.assets + self.cfg.universe.meta_assets))
        logger.info("ARES BACKTEST: %s → %s  assets=%s", start, end, all_assets)

        if self.cfg.model.backtest_deterministic:
            seed = int(self.cfg.model.backtest_seed)
            np.random.seed(seed)
            random.seed(seed)
            cache_dir = os.path.expanduser("~/.ares/cache/deterministic_market_data")
            os.makedirs(cache_dir, exist_ok=True)
            os.environ.setdefault("ARES_MARKET_DATA_CACHE_DIR", cache_dir)
            self.data._disk_cache_dir = cache_dir
            os.environ.setdefault("OMP_NUM_THREADS", "1")
            os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
            os.environ.setdefault("MKL_NUM_THREADS", "1")
            try:
                from threadpoolctl import threadpool_limits

                threadpool_limits(limits=1)
            except Exception:
                pass
            self.alpha_learner.reset_state()
            logger.info("Backtest deterministic mode ON (seed=%d)", seed)
        deterministic_mode = bool(self.cfg.model.backtest_deterministic)
        # Download data with enough history for training
        data_start = self._offset_date(start, -self.cfg.model.train_window_days - 30)
        asset_data: Dict[str, pd.DataFrame] = {}

        for asset in all_assets:
            try:
                df = self.data.ohlcv(asset, data_start, end)
                asset_data[asset] = df
            except Exception as exc:
                logger.warning("Skipping %s: %s", asset, exc)

        if not asset_data:
            raise RuntimeError("No asset data fetched — check network/tickers")

        # IV series — actifs core uniquement
        iv_data: Dict[str, pd.Series] = {}
        for asset in self.cfg.universe.assets:
            try:
                iv = self.data.iv_series(asset, data_start, end)
                iv_data[asset] = iv
            except Exception as exc:
                logger.warning("No IV for %s: %s", asset, exc)

        self._macro_proxy_data = self._prepare_macro_proxies(data_start, end)

        # --- Pre-compute full RV series once (O(N) not O(N²)) ---
        rv_precomputed: Dict[str, pd.Series] = {}
        for asset, df in asset_data.items():
            try:
                rv_full = self._compute_rv_series(df)
                rv_precomputed[asset] = rv_full
                logger.info("Pre-computed RV for %s: %d rows", asset, len(rv_full))
            except Exception as exc:
                logger.warning("RV precompute failed for %s: %s", asset, exc)

        # Build OOS date range
        oos_dates = pd.bdate_range(start=start, end=end)
        nav_ser = pd.Series(
            dtype=float,
            index=oos_dates,
            name="nav",
        )
        nav = self.cfg.capital.paper_capital
        nav_ser.iloc[0] = nav

        for i, date in enumerate(oos_dates):
            date_str = str(date.date())
            try:
                daily_pnl = self._step(
                    date=date,
                    asset_data=asset_data,
                    iv_data=iv_data,
                    rv_precomputed=rv_precomputed,
                    current_nav=nav,
                )
                if deterministic_mode:
                    # Stabilize floating-point accumulation so repeated runs are bit-stable.
                    daily_pnl = float(np.round(daily_pnl, 8))
                nav += daily_pnl
                if deterministic_mode:
                    nav = float(np.round(nav, 2))
                nav = max(nav, 0)
            except Exception as exc:
                logger.debug("Step error on %s: %s", date_str, exc)

            nav_ser.iloc[i] = nav
            self._nav_history.append(nav)
            self.risk.update_nav(nav)

            if self.risk.is_halted:
                logger.warning("HALT triggered on %s", date_str)
                nav_ser.iloc[i:] = nav
                break

            if verbose and i % 60 == 0:
                logger.info("Backtest progress: %s | NAV=$%s", date_str, format(nav, ",.0f"))

        nav_ser = nav_ser.dropna()
        trade_log = self.orders.order_summary()
        closed_trade_log = pd.DataFrame(self._closed_vrp_trades)

        # --- Build SPY buy-and-hold benchmark NAV series ---
        benchmark_nav: Optional[pd.Series] = None
        try:
            spy_df = asset_data.get("SPY")
            if spy_df is None:
                spy_df = asset_data.get(self.cfg.universe.benchmark)
            if spy_df is not None:
                spy_oos = spy_df[spy_df.index >= pd.Timestamp(start)]
                spy_oos = spy_oos[spy_oos.index <= pd.Timestamp(end or datetime.today().strftime("%Y-%m-%d"))]
                if not spy_oos.empty:
                    spy_close = spy_oos["close"].dropna()
                    spy_close = spy_close.reindex(nav_ser.index, method="ffill").dropna()
                    if not spy_close.empty:
                        benchmark_nav = (
                            spy_close / float(spy_close.iloc[0]) * self.cfg.capital.paper_capital
                        )
        except Exception as exc:
            logger.debug("Benchmark NAV build failed: %s", exc)

        report = self.dashboard.render(
            nav_history=nav_ser,
            trade_log=trade_log,
            closed_trade_log=closed_trade_log,
            active_positions=self.orders.open_positions(),
            pnl_by_asset=self._strategy_pnl_by_asset,
            regime=self._regime_state,
            geo_risk=self._geo_score,
            halt_triggered=self.risk.is_halted,
            initial_capital=self.cfg.capital.paper_capital,
            print_to_console=verbose,
            save_html=True,
            save_csv=True,
            benchmark_nav=benchmark_nav,
            benchmark_label="SPY B&H",
            report_status="COMPLETED",
        )

        return report

    # ------------------------------------------------------------------
    # Live / Paper loop
    # ------------------------------------------------------------------

    def run_live(
        self,
        poll_interval_seconds: int = 60,
        max_iterations: Optional[int] = None,
    ) -> None:
        """
        Run ARES in live or paper mode (infinite loop).
        Polls every poll_interval_seconds.
        Press Ctrl+C to stop.
        """
        if self.mode == "live" and (self.ibkr is None or not self.ibkr.is_connected):
            raise RuntimeError("Cannot run live without IBKR connection. Call connect_ibkr() first.")

        logger.info("ARES starting in %s mode", self.mode.upper())

        iteration = 0
        nav = self.cfg.capital.paper_capital

        try:
            while True:
                if max_iterations and iteration >= max_iterations:
                    break

                now = datetime.now()
                start = self._offset_date(now.strftime("%Y-%m-%d"), -self.cfg.model.train_window_days - 30)
                end = now.strftime("%Y-%m-%d")

                try:
                    asset_data = {
                        a: self.data.ohlcv(a, start, end)
                        for a in self.cfg.universe.assets
                    }
                    self._macro_proxy_data = self._prepare_macro_proxies(start, end)
                    iv_data = {}
                    for a in self.cfg.universe.assets:
                        try:
                            iv_data[a] = self.data.iv_series(a, start, end)
                        except Exception:
                            pass

                    today = pd.Timestamp(now.date())
                    rv_live: Dict[str, pd.Series] = {}
                    for a, df in asset_data.items():
                        try:
                            rv_live[a] = self._compute_rv_series(df)
                        except Exception:
                            pass
                    daily_pnl = self._step(
                        date=today,
                        asset_data=asset_data,
                        iv_data=iv_data,
                        rv_precomputed=rv_live,
                        current_nav=nav,
                    )
                    nav += daily_pnl
                    nav = max(nav, 0)
                    self._nav_history.append(nav)
                    self.risk.update_nav(nav)

                    if self.risk.is_halted:
                        logger.critical("HALT triggered — stopping live loop")
                        break

                except Exception as exc:
                    logger.error("Live loop error: %s", exc)

                # Report every 10 iterations
                if iteration % 10 == 0:
                    nav_ser = pd.Series(self._nav_history)
                    self.dashboard.render(
                        nav_history=nav_ser,
                        trade_log=self.orders.order_summary(),
                        closed_trade_log=pd.DataFrame(self._closed_vrp_trades),
                        active_positions=self.orders.open_positions(),
                        pnl_by_asset=self.orders.pnl_by_asset(),
                        regime=self._regime_state,
                        geo_risk=self._geo_score,
                        halt_triggered=self.risk.is_halted,
                        initial_capital=self.cfg.capital.paper_capital,
                        print_to_console=True,
                        save_html=True,
                    )

                iteration += 1
                time.sleep(poll_interval_seconds)

        except KeyboardInterrupt:
            logger.info("ARES stopped by user")

    # ------------------------------------------------------------------
    # Core daily step
    # ------------------------------------------------------------------

    def _step(
        self,
        date: pd.Timestamp,
        asset_data: Dict[str, pd.DataFrame],
        iv_data: Dict[str, pd.Series],
        rv_precomputed: Dict[str, pd.Series],
        current_nav: float,
    ) -> float:
        """
        Process one trading day. Return estimated daily P&L.
        """
        total_pnl = 0.0
        self._step_count += 1

        # --- Macro regime + geo-risk ---
        self._update_regime_and_geo(asset_data, iv_data, date)
        # Smooth regime scaling from posterior high-vol probability.
        regime_scale = float(np.clip(1.0 - 0.6 * self._regime_prob_high, 0.35, 1.0))
        composite_geo = max(self._geo_score, self._news_geo_score)
        geo_scale = max(1.0 - composite_geo, 0.1)

        for asset in list(asset_data.keys()):
            try:
                df = asset_data[asset]
                iv = iv_data.get(asset)

                if df.empty or len(df) < 126:
                    continue

                cooldown_left = self._entry_cooldown.get(asset, 0)
                if cooldown_left > 0:
                    self._entry_cooldown[asset] = cooldown_left - 1

                # Slice OHLCV to current date (for spot price lookups)
                hist = df[df.index <= date]

                # --- Daily mark-to-market for any held position ---
                if asset in self._active_vol_positions and len(hist) >= 2:
                    pos = self._active_vol_positions[asset]
                    spot_today = float(hist["close"].iloc[-1])
                    spot_prev = float(hist["close"].iloc[-2])
                    today_ret = (spot_today - spot_prev) / max(spot_prev, 1e-6)
                    # Delta-hedged ATM straddle gamma/theta P&L
                    T_expiry = 30.0 / 252
                    N_prime = 0.3989
                    entry_iv = pos["entry_iv"]
                    n_straddles = pos["contracts"]
                    theta_per_str = (spot_today * N_prime * entry_iv
                                     / (math.sqrt(T_expiry) * 252) * 100)
                    gamma_coeff = (2 * N_prime
                                   / (spot_today * entry_iv * math.sqrt(T_expiry)) * 100)
                    total_theta = theta_per_str * n_straddles
                    dollar_move = today_ret * spot_today
                    gamma_loss = 0.5 * gamma_coeff * n_straddles * dollar_move ** 2
                    pnl_today = pos["direction"] * (total_theta - gamma_loss)
                    total_pnl += pnl_today
                    self._strategy_pnl_by_asset[asset] = self._strategy_pnl_by_asset.get(asset, 0.0) + float(pnl_today)
                    pos["mtm_pnl"] += pnl_today
                    pos["days_held"] += 1
                    # Check exit: signal decays or max holding (21 trading days)
                    vrp_history_for_exit = self._current_signals.get(asset, {}).get("vrp_history", [])
                    current_vrp_z = vrp_history_for_exit[-1]["vrp_zscore"] if vrp_history_for_exit else 0.0
                    shock_mult = float(getattr(self.cfg.model, "vrp_shock_exit_multiplier", 2.0))
                    shock_threshold = shock_mult * entry_iv / math.sqrt(252)
                    signal_exit = pos["direction"] * current_vrp_z < self.cfg.model.vrp_zscore_exit
                    max_holding_exit = pos["days_held"] >= self.cfg.model.vrp_max_holding_days
                    stop_loss_exit = pos["mtm_pnl"] <= -self.cfg.model.vrp_stop_loss_notional_pct * pos["notional"]
                    take_profit_exit = pos["mtm_pnl"] >= getattr(self.cfg.model, "vrp_take_profit_notional_pct", 0.0) * pos["notional"]
                    shock_exit = pos["direction"] > 0 and abs(today_ret) > shock_threshold
                    should_exit = signal_exit or max_holding_exit or stop_loss_exit or take_profit_exit or shock_exit
                    if should_exit:
                        if shock_exit:
                            exit_reason = "shock_exit"
                        elif take_profit_exit:
                            exit_reason = "take_profit"
                        elif stop_loss_exit:
                            exit_reason = "stop_loss"
                        elif max_holding_exit:
                            exit_reason = "max_holding"
                        else:
                            exit_reason = "signal_decay"
                        spot_exit = float(hist["close"].iloc[-1])
                        entry_iv_exit = pos["entry_iv"]
                        self.orders.submit(
                            asset=asset,
                            symbol=asset,
                            action="BUY" if pos["direction"] > 0 else "SELL",
                            quantity=pos["contracts"],
                            target_price=entry_iv_exit * spot_exit * 0.14,
                            sec_type="OPT",
                            note=f"exit={exit_reason} vrp_z={current_vrp_z:.2f} days={pos['days_held']}",
                        )
                        self.alpha_learner.record_trade(TradeFeedback(
                            timestamp=str(date),
                            asset=asset,
                            direction=int(pos["direction"]),
                            pnl=float(pos["mtm_pnl"]),
                            notional=float(pos["notional"]),
                            holding_days=int(pos["days_held"]),
                            entry_vrp_z=float(pos.get("entry_vrp_z", 0.0)),
                            entry_vol_spread=float(pos.get("entry_vol_spread", 0.0)),
                            entry_regime=str(pos.get("entry_regime", "unknown")),
                            entry_geo=float(pos.get("entry_geo", 0.0)),
                            exit_reason=exit_reason,
                        ))
                        self._closed_vrp_trades.append({
                            "timestamp": str(date),
                            "asset": asset,
                            "direction": int(pos["direction"]),
                            "entry_date": str(pos.get("entry_date", date)),
                            "exit_date": str(date),
                            "holding_days": int(pos["days_held"]),
                            "entry_vrp_z": float(pos.get("entry_vrp_z", 0.0)),
                            "exit_vrp_z": float(current_vrp_z),
                            "entry_vol_spread": float(pos.get("entry_vol_spread", 0.0)),
                            "entry_regime": str(pos.get("entry_regime", "unknown")),
                            "entry_geo": float(pos.get("entry_geo", 0.0)),
                            "exit_reason": exit_reason,
                            "notional": float(pos["notional"]),
                            "pnl": float(pos["mtm_pnl"]),
                        })
                        del self._active_vol_positions[asset]
                        self._entry_cooldown[asset] = self.cfg.model.vrp_cooldown_days
                if len(hist) < 126:
                    continue

                # Use precomputed RV series, sliced to current date
                rv_full = rv_precomputed.get(asset)
                if rv_full is None or rv_full.empty:
                    continue
                rv_series = rv_full[rv_full.index <= date]
                if len(rv_series) < 63:
                    continue

                # HAR-RV-CJ forecast
                har_forecast = self._har_forecast(rv_series)

                # AI ensemble forecast (LGBM if fitted)
                ai_forecast = self._ai_forecast(rv_series, iv, asset, date)

                # Blend: 60% HAR + 40% AI (if available)
                if ai_forecast is not None and np.isfinite(ai_forecast):
                    rv_forecast = 0.60 * har_forecast + 0.40 * ai_forecast
                else:
                    rv_forecast = har_forecast

                # VRP signal
                if iv is not None:
                    # Correct IV lookup: slice up to current date, take last value
                    iv_slice = iv[iv.index <= date]
                    if not iv_slice.empty and not np.isnan(float(iv_slice.iloc[-1])):
                        iv_val = float(iv_slice.iloc[-1])  # re-binds below
                        vrp = iv_val ** 2 - rv_forecast ** 2
                        vrp_history = self._current_signals.get(asset, {}).get("vrp_history", [])
                        vrp_z = self.vrp_calc._zscore(vrp, vrp_history)
                        vol_spread = iv_val - rv_forecast
                        rv_rank_window = rv_series.iloc[-252:] if len(rv_series) >= 252 else rv_series
                        rv_percentile = float((rv_rank_window <= rv_forecast).mean()) if len(rv_rank_window) > 0 else 0.5

                        # Regime-aware signal direction
                        # ── Recherche académique (Carr & Wu 2009, Bondarenko 2014) : la VRP
                        #    est POSITIVE dans tous les régimes. En high_vol, IV sur-estime RV
                        #    par une marge encore plus grande. Limiter au low_vol détruisait
                        #    2/3 des opportunités, dont toute la période 2022-2024.
                        direction = 0
                        _zscore_max = getattr(self.cfg.model, "vrp_zscore_max", 3.5)
                        # Z-entry adapté au régime : bar plus haute en high_vol pour plus de sécurité
                        _regime_z_mult = (
                            1.25 if self._regime_state == "high_vol"
                            else 1.10 if self._regime_state == "mid_vol"
                            else 1.00
                        )
                        _effective_entry_z = self.cfg.model.vrp_zscore_entry * _regime_z_mult
                        short_setup = (
                            bool(getattr(self.cfg.model, "vrp_trading_enabled", True))
                            and vrp_z >= _effective_entry_z
                            and vrp_z <= _zscore_max   # avoid entering during active vol spike
                            and vol_spread >= self.cfg.model.vrp_min_vol_spread_short
                            # Tous régimes — VRP est positive partout, sizing réduit en high_vol
                            # via regime_scale (déjà calculé au-dessus)
                            and rv_percentile <= self.cfg.model.vrp_max_rv_percentile_short
                            and composite_geo < 0.65   # élargi de 0.50 → 0.65
                        )
                        long_setup = (
                            self.cfg.model.vrp_allow_long_vol_hedge
                            and vrp_z <= -self.cfg.model.vrp_zscore_entry
                            and (-vol_spread) >= self.cfg.model.vrp_min_vol_spread_long
                            and (self._regime_state in ("high_vol", "mid_vol") or rv_percentile >= 0.85)
                            and composite_geo < 0.65
                        )
                        if short_setup:
                            direction = 1
                        elif long_setup:
                            direction = -1

                        if direction != 0:
                            # Kelly sizing — edge estimé à 35% du vol_spread
                            edge = max(abs(vol_spread) * 0.35, 0.001)
                            daily_vol = rv_forecast / np.sqrt(252)
                            sizing = self.risk.size_position(
                                asset=asset,
                                direction=direction,
                                edge_estimate=edge * self._last_meta_vrp_weight,
                                vol_estimate=daily_vol,
                                regime_scale=regime_scale,
                                geo_scale=geo_scale,
                                spot_price=float(hist["close"].iloc[-1]),
                            )

                            if sizing.approved and sizing.contracts > 0:
                                spot = float(hist["close"].iloc[-1])
                                # ATM straddle price proxy: S * IV * sqrt(T) * 0.798
                                T_expiry = 30.0 / 252
                                atm_price = iv_val * spot * math.sqrt(T_expiry) * 0.798
                                if self.mode in ("paper", "live"):
                                    submitted = self._submit_live_option_structure(
                                        asset=asset,
                                        direction=direction,
                                        quantity=sizing.contracts,
                                        fallback_target_price=atm_price,
                                        note=f"vrp_z={vrp_z:.2f} dir={direction}",
                                    )
                                    if not submitted and self.mode == "live":
                                        logger.error("Skipped live trade for %s: no executable option structure resolved", asset)
                                        continue
                                else:
                                    # Only enter if not already holding this asset
                                    if self._entry_cooldown.get(asset, 0) <= 0 and asset not in self._active_vol_positions:
                                        self.orders.submit(
                                            asset=asset,
                                            symbol=asset,
                                            action="SELL" if direction > 0 else "BUY",
                                            quantity=sizing.contracts,
                                            target_price=atm_price,
                                            sec_type="OPT",
                                            note=f"vrp_z={vrp_z:.2f} spread={vol_spread:.3f} dir={direction}",
                                        )
                                        self._active_vol_positions[asset] = {
                                            "direction": direction,
                                            "contracts": sizing.contracts,
                                            "notional": sizing.notional,
                                            "entry_iv": iv_val,
                                            "entry_date": str(date),
                                            "entry_vrp_z": vrp_z,
                                            "entry_vol_spread": vol_spread,
                                            "entry_regime": self._regime_state,
                                            "entry_geo": composite_geo,
                                            "days_held": 0,
                                            "mtm_pnl": 0.0,
                                        }

                        # Store VRP history
                        if asset not in self._current_signals:
                            self._current_signals[asset] = {"vrp_history": []}
                        self._current_signals[asset]["vrp_history"].append({"vrp": vrp, "vrp_zscore": vrp_z, "signal": direction if abs(vrp_z) >= self.cfg.model.vrp_zscore_entry else 0})

            except Exception as exc:
                logger.debug("Step error for %s on %s: %s", asset, date, exc)

        if (
            self.cfg.model.adaptive_learning_enabled
            and self._step_count % max(1, self.cfg.model.adaptive_recalibration_days) == 0
        ):
            updates = self.alpha_learner.adapt(
                self.cfg.model,
                self.cfg.risk,
                min_trades=self.cfg.model.adaptive_min_trades,
                lookback_trades=self.cfg.model.adaptive_lookback_trades,
            )
            if updates:
                self.vrp_calc.entry_z = self.cfg.model.vrp_zscore_entry
                logger.info("Adaptive alpha recalibration: %s", updates)

        # --- Multi-strategy meta overlay (momentum + carry) ---
        try:
            meta = self.meta_strategy.evaluate(
                date=date,
                asset_data=asset_data,
                iv_data=iv_data,
                rv_precomputed=rv_precomputed,
                regime=self._regime_state,
                geo_score=max(self._geo_score, self._news_geo_score),
                nav=current_nav,
                vrp_pnl_today=total_pnl,
            )
            self._last_meta_vrp_weight = meta.vrp_weight
            total_pnl += meta.momentum_pnl_overlay + meta.carry_pnl_overlay
            self._strategy_pnl_by_asset["META_MOMENTUM"] = (
                self._strategy_pnl_by_asset.get("META_MOMENTUM", 0.0) + float(meta.momentum_pnl_overlay)
            )
            self._strategy_pnl_by_asset["META_CARRY"] = (
                self._strategy_pnl_by_asset.get("META_CARRY", 0.0) + float(meta.carry_pnl_overlay)
            )
            if self._step_count % 21 == 0:
                logger.info(
                    "Meta-strategy [%s]: vrp_w=%.2f mom_pnl=%.1f carry_pnl=%.1f rw={vrp=%.2f,mom=%.2f,carry=%.2f} | %s",
                    date.date(),
                    meta.vrp_weight,
                    meta.momentum_pnl_overlay,
                    meta.carry_pnl_overlay,
                    meta.sleeve_weights.get("vrp", 0.0),
                    meta.sleeve_weights.get("momentum", 0.0),
                    meta.sleeve_weights.get("carry", 0.0),
                    meta.debug,
                )
        except Exception as exc:
            logger.debug("Meta-strategy error on %s: %s", date, exc)

        return total_pnl

    # ------------------------------------------------------------------
    # Model helpers
    # ------------------------------------------------------------------

    def _compute_rv_series(self, price_df: pd.DataFrame) -> pd.Series:
        """Compute daily Garman-Klass RV for every row in price_df."""
        rv_vals = []
        dates = []
        for i in range(len(price_df)):
            row = price_df.iloc[i]
            try:
                gk = self.rv.garman_klass(
                    open_=[row["open"]],
                    high=[row["high"]],
                    low=[row["low"]],
                    close=[row["close"]],
                )
                rv_vals.append(gk)
                dates.append(price_df.index[i])
            except Exception:
                pass
        return pd.Series(rv_vals, index=dates, name="rv")

    def _har_forecast(self, rv_series: pd.Series) -> float:
        """HAR-RV-CJ one-step forecast in vol space."""
        rv_var = rv_series.values ** 2
        n = len(rv_var)
        if n < 30:
            return float(rv_series.iloc[-22:].mean()) if len(rv_series) >= 22 else float(rv_series.mean())

        # Build CJ regressors using last 22 obs for monthly component
        bv_var = np.minimum(rv_var, pd.Series(rv_var).rolling(2).min().values)
        j_var = np.maximum(rv_var - bv_var, 0)
        w = pd.Series(rv_var).rolling(5).mean().values
        m = pd.Series(rv_var).rolling(21).mean().values

        valid = 21
        X = np.column_stack([
            np.ones(n - valid - 1),
            rv_var[valid:-1],
            w[valid:-1],
            m[valid:-1],
            bv_var[valid:-1],
            j_var[valid:-1],
        ])
        y = rv_var[valid + 1:]

        if len(y) < 20:
            return float(np.sqrt(np.mean(rv_var[-22:])))

        try:
            beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            x_now = np.array([1.0, rv_var[-1], w[-1], m[-1], bv_var[-1], j_var[-1]])
            fc_var = float(np.clip(x_now @ beta, 1e-8, None))
            return float(np.sqrt(fc_var))
        except Exception:
            return float(np.sqrt(np.mean(rv_var[-22:])))

    def _ai_forecast(
        self,
        rv_series: pd.Series,
        iv_series: Optional[pd.Series],
        asset: str,
        date: pd.Timestamp,
    ) -> Optional[float]:
        """LGBM forecast if model is fitted (warm-up required)."""
        if not self._lgbm_fitted:
            if len(rv_series) >= self.cfg.model.train_window_days:
                try:
                    features = self.lgbm.build_features(
                        rv=rv_series,
                        iv=iv_series.reindex(rv_series.index).ffill() if iv_series is not None else None,
                    )
                    self.lgbm.fit(features, rv_series)
                    self._lgbm_fitted = True
                    logger.info("LightGBM fitted on %s data through %s", asset, date)
                except Exception as exc:
                    logger.debug("LGBM fit failed: %s", exc)
            return None

        try:
            features = self.lgbm.build_features(
                rv=rv_series,
                iv=iv_series.reindex(rv_series.index).ffill() if iv_series is not None else None,
            )
            preds = self.lgbm.predict(features.iloc[-1:])
            return float(preds[0]) if len(preds) > 0 else None
        except Exception:
            return None

    def _update_regime_and_geo(
        self,
        asset_data: Dict[str, pd.DataFrame],
        iv_data: Dict[str, pd.Series],
        date: pd.Timestamp,
    ) -> None:
        """Update regime and geo-risk scores (SPY-based)."""
        spy_df = asset_data.get("SPY")
        if spy_df is None:
            spy_df = next(iter(asset_data.values()), pd.DataFrame())
        if spy_df.empty:
            return

        hist = spy_df[spy_df.index <= date]
        if len(hist) < 63:
            return

        rv_series = self._compute_rv_series(hist)
        if len(rv_series) < 63:
            return

        # Regime (refit every 63 days)
        if not self._regime_fitted or len(rv_series) % 63 == 0:
            try:
                iv_spy = iv_data.get("SPY", iv_data.get(next(iter(iv_data), None)))
                self.regime.fit(rv_series, iv_series=iv_spy)
                self._regime_fitted = True
            except Exception as exc:
                logger.debug("Regime fit failed: %s", exc)

        if self._regime_fitted:
            try:
                iv_spy = iv_data.get("SPY")
                state_df = self.regime.predict(rv_series, iv_series=iv_spy)
                if not state_df.empty:
                    self._regime_state = str(state_df["label"].iloc[-1])
                    self._regime_prob_high = float(np.clip(state_df["prob_high"].iloc[-1], 0.0, 1.0))
            except Exception:
                pass

        # Geo-risk
        try:
            vix = iv_data.get("SPY")
            if vix is None:
                for v in iv_data.values():
                    if v is not None:
                        vix = v
                        break
            if vix is not None:
                vix_aligned = vix.reindex(rv_series.index).ffill()
                geo_df = self.geo_risk.compute(
                    vix=vix_aligned * 100,
                    hyg=self._macro_proxy_data.get("HYG"),
                    dxy=self._macro_proxy_data.get("DXY"),
                    put_call=self._macro_proxy_data.get("PUTCALL"),
                )
                if not geo_df.empty:
                    self._geo_score = float(geo_df["composite"].iloc[-1])
        except Exception:
            pass

        # Live macro/geopolitical news overlay
        try:
            if self._world_monitor is not None and self.mode in ("paper", "live"):
                alerts = self._world_monitor.fetch_current_alerts()
                severities = {
                    "LOW": 0.10,
                    "MODERATE": 0.25,
                    "ELEVATED": 0.45,
                    "HIGH": 0.70,
                    "CRITICAL": 1.00,
                }
                if alerts:
                    self._news_geo_score = max(
                        severities.get(str(alert.get("severity", "MODERATE")).upper(), 0.25)
                        for alert in alerts
                    )
                else:
                    self._news_geo_score *= 0.85
        except Exception:
            pass

    def _prepare_macro_proxies(self, start: str, end: str) -> Dict[str, pd.Series]:
        proxies: Dict[str, pd.Series] = {}
        candidates = {
            "HYG": ["HYG"],
            "DXY": ["DX-Y.NYB", "DX=F", "UUP"],
        }
        for key, tickers in candidates.items():
            for ticker in tickers:
                try:
                    series = self.data.close(ticker, start, end)
                    if not series.empty:
                        proxies[key] = series
                        break
                except Exception:
                    continue
        return proxies

    def _submit_live_option_structure(
        self,
        asset: str,
        direction: int,
        quantity: int,
        fallback_target_price: float,
        note: str,
    ) -> bool:
        legs = self._resolve_atm_straddle(asset)
        if not legs:
            if self.mode == "paper":
                self.orders.submit(
                    asset=asset,
                    symbol=asset,
                    action="SELL" if direction > 0 else "BUY",
                    quantity=quantity,
                    target_price=fallback_target_price,
                    sec_type="OPT",
                    note=note,
                )
                return True
            return False

        current_signal = self._current_option_signal(asset)
        if current_signal == direction:
            return True
        if current_signal != 0:
            self._close_existing_option_positions(asset, legs, note=f"close_before_reopen {note}")

        action = "SELL" if direction > 0 else "BUY"
        for leg in legs:
            self.orders.submit(
                asset=self._option_asset_key(asset, leg),
                symbol=asset,
                action=action,
                quantity=quantity,
                target_price=float(leg.mid),
                sec_type="OPT",
                expiry=leg.expiry.replace("-", ""),
                strike=float(leg.strike),
                right="C" if leg.option_type == "call" else "P",
                note=note,
            )
        return True

    def _resolve_atm_straddle(self, asset: str) -> List[OptionContract]:
        try:
            chain = self.option_chain.fetch(asset, max_expiries=1)
        except Exception as exc:
            logger.debug("Option chain fetch failed for %s: %s", asset, exc)
            return []

        if not chain:
            return []
        expiry, contracts = next(iter(chain.items()))
        if not contracts:
            return []

        spot = float(contracts[0].S)
        calls = [c for c in contracts if c.option_type == "call" and c.mid > 0]
        puts = [c for c in contracts if c.option_type == "put" and c.mid > 0]
        if not calls or not puts:
            return []

        best_call = min(calls, key=lambda c: abs(c.strike - spot))
        best_put = min(puts, key=lambda c: abs(c.strike - spot))
        return [best_call, best_put]

    def _current_option_signal(self, asset: str) -> int:
        total = 0
        prefix = f"{asset}|"
        for key, qty in self.orders.open_positions().items():
            if key.startswith(prefix):
                total += qty
        if total < 0:
            return 1
        if total > 0:
            return -1
        return 0

    def _close_existing_option_positions(self, asset: str, live_legs: List[OptionContract], note: str) -> None:
        leg_map = {
            (leg.expiry.replace("-", ""), f"{leg.strike:.2f}", "C" if leg.option_type == "call" else "P"): leg
            for leg in live_legs
        }
        prefix = f"{asset}|"
        for key, qty in self.orders.open_positions().items():
            if not key.startswith(prefix):
                continue
            symbol, expiry, strike, right = key.split("|")
            leg = leg_map.get((expiry, strike, right))
            if leg is None:
                continue
            self.orders.submit(
                asset=key,
                symbol=symbol,
                action="BUY" if qty < 0 else "SELL",
                quantity=abs(int(qty)),
                target_price=float(leg.mid),
                sec_type="OPT",
                expiry=expiry,
                strike=float(strike),
                right=right,
                note=note,
            )

    @staticmethod
    def _option_asset_key(asset: str, leg: OptionContract) -> str:
        right = "C" if leg.option_type == "call" else "P"
        return f"{asset}|{leg.expiry.replace('-', '')}|{leg.strike:.2f}|{right}"

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _offset_date(date_str: str, days: int) -> str:
        dt = datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=days)
        return dt.strftime("%Y-%m-%d")

    def status(self) -> dict:
        """Return current engine status as a dict."""
        metrics = self.risk.portfolio_metrics(
            regime=self._regime_state,
            geo_risk=max(self._geo_score, self._news_geo_score),
        )
        return {
            "mode": self.mode,
            "nav": metrics.nav,
            "regime": self._regime_state,
            "regime_prob_high": self._regime_prob_high,
            "geo_risk": max(self._geo_score, self._news_geo_score),
            "halt": self.risk.is_halted,
            "ibkr_connected": self.ibkr.is_connected if self.ibkr else False,
            "n_fills": len(self.orders.get_fills()),
            "open_positions": self.orders.open_positions(),
        }
