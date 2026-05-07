"""
ARES command-line entry point.

Examples
--------
Backtest:
  python main.py --mode backtest --start 2020-01-01 --end 2024-12-31

Paper loop:
  python main.py --mode paper --poll 300

Live with IBKR paper gateway:
  python main.py --mode live --ibkr-host 127.0.0.1 --ibkr-port 7497 --client-id 1
"""
from __future__ import annotations

import argparse
import copy
import logging

from ares import DEFAULT_CONFIG, StrategyEngine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ARES - Autonomous Realized-vol Edge System")
    parser.add_argument("--mode", default="backtest", choices=["backtest", "paper", "live"], help="Run mode")
    parser.add_argument("--start", default="2020-01-01", help="Backtest start date")
    parser.add_argument("--end", default=None, help="Backtest end date")
    parser.add_argument("--poll", type=int, default=300, help="Live/paper poll interval in seconds")
    parser.add_argument("--iterations", type=int, default=None, help="Optional max iterations for paper/live")
    parser.add_argument("--ibkr-host", default="127.0.0.1", help="IBKR TWS/Gateway host")
    parser.add_argument("--ibkr-port", type=int, default=7497, help="IBKR port (7497 paper, 7496 live)")
    parser.add_argument("--client-id", type=int, default=1, help="IBKR client id")
    parser.add_argument("--live-account", action="store_true", help="Enable live trading mode (requires port 7496)")
    parser.add_argument("--leverage-multiplier", type=float, default=None, help="Global sizing multiplier across VRP and meta sleeves")
    parser.add_argument("--kelly-fraction", type=float, default=None, help="Override Kelly fraction")
    parser.add_argument("--max-gross-leverage", type=float, default=None, help="Override gross leverage cap")
    parser.add_argument("--max-position-pct", type=float, default=None, help="Override max per-position NAV fraction")
    parser.add_argument("--momentum-pct", type=float, default=None, help="Override momentum sleeve NAV fraction")
    parser.add_argument("--carry-pct", type=float, default=None, help="Override carry sleeve NAV fraction")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if args.leverage_multiplier is not None:
        cfg.risk.strategy_leverage_multiplier = max(args.leverage_multiplier, 0.0)
    if args.kelly_fraction is not None:
        cfg.risk.kelly_fraction = max(args.kelly_fraction, 0.0)
    if args.max_gross_leverage is not None:
        cfg.risk.max_gross_leverage = max(args.max_gross_leverage, 0.0)
    if args.max_position_pct is not None:
        cfg.risk.max_position_pct = max(args.max_position_pct, 0.0)
    if args.momentum_pct is not None:
        cfg.model.meta_momentum_position_pct = max(args.momentum_pct, 0.0)
    if args.carry_pct is not None:
        cfg.model.meta_carry_position_pct = max(args.carry_pct, 0.0)

    engine = StrategyEngine(config=cfg, mode=args.mode, log_level=args.log_level)

    logging.info(
        "Runtime overrides | lev_mult=%.2f | kelly=%.3f | gross_lev=%.2f | pos_cap=%.2f | mom_pct=%.3f | carry_pct=%.3f",
        cfg.risk.strategy_leverage_multiplier,
        cfg.risk.kelly_fraction,
        cfg.risk.max_gross_leverage,
        cfg.risk.max_position_pct,
        cfg.model.meta_momentum_position_pct,
        cfg.model.meta_carry_position_pct,
    )

    if args.mode == "backtest":
        engine.run(start=args.start, end=args.end, verbose=True)
        return

    if args.mode == "live":
        engine.connect_ibkr(
            host=args.ibkr_host,
            port=args.ibkr_port,
            client_id=args.client_id,
            live=args.live_account,
        )

    engine.run_live(
        poll_interval_seconds=args.poll,
        max_iterations=args.iterations,
    )


if __name__ == "__main__":
    main()
