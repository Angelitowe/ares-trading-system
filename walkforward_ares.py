from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ares import DEFAULT_CONFIG, StrategyEngine


@dataclass
class SegmentSpec:
    label: str
    start: str
    end: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ARES strict walk-forward validation")
    parser.add_argument("--train-start", default="2015-01-01")
    parser.add_argument("--train-end", default="2019-12-31")
    parser.add_argument("--val-start", default="2020-01-01")
    parser.add_argument("--val-end", default="2021-12-31")
    parser.add_argument("--test-start", default="2022-01-01")
    parser.add_argument("--test-end", default="2024-12-31")
    parser.add_argument("--leverage-multiplier", type=float, default=5.0)
    parser.add_argument("--log-level", default="ERROR")
    parser.add_argument("--output-csv", default="~/.ares/reports/walkforward_x5_segments.csv")
    parser.add_argument("--output-summary", default="~/.ares/reports/walkforward_x5_summary.txt")
    return parser


def _get_risk_free_rate(start_date: str, rf_by_period: dict) -> float:
    """Retourne le taux sans risque annualisé pour la période débutant à start_date."""
    applicable_rate = 0.025  # défaut
    for period_start in sorted(rf_by_period.keys()):
        if start_date >= period_start:
            applicable_rate = rf_by_period[period_start]
    return applicable_rate


def compute_nav_metrics(
    nav_values: list[float],
    initial_capital: float,
    risk_free_rate_ann: float = 0.0,
) -> dict[str, float]:
    """
    Calcule les métriques de performance à partir d'une série NAV.

    Le Sharpe est calculé en soustrayant le taux sans risque (risk_free_rate_ann)
    du rendement annualisé — conformément à la définition académique standard.
    """
    nav = np.asarray(nav_values, dtype=float)
    if len(nav) < 2:
        return {
            "nav": initial_capital,
            "total_return_pct": 0.0,
            "ann_return_pct": 0.0,
            "ann_vol_pct": 0.0,
            "sharpe": 0.0,
            "max_drawdown_pct": 0.0,
            "cvar_95_ann_pct": 0.0,
            "trading_days": 0,
        }
    total_return_pct = (nav[-1] / initial_capital - 1.0) * 100.0
    trading_days = max(len(nav) - 1, 1)
    ann_factor = 252 / trading_days
    ann_return_pct = ((nav[-1] / initial_capital) ** ann_factor - 1.0) * 100.0
    daily_ret = np.diff(nav) / nav[:-1]
    ann_vol_pct = float(np.std(daily_ret) * np.sqrt(252) * 100.0)
    # Sharpe ajusté au taux sans risque (Lo 2002 / CFA standard)
    excess_return_pct = ann_return_pct - risk_free_rate_ann * 100.0
    sharpe = float(excess_return_pct / ann_vol_pct) if ann_vol_pct > 1e-8 else 0.0
    peak = np.maximum.accumulate(nav)
    max_dd_pct = float(np.max((peak - nav) / (peak + 1e-10)) * 100.0)
    # CVaR 95% (Expected Shortfall) annualisé
    if len(daily_ret) >= 10:
        threshold = np.percentile(daily_ret, 5.0)
        tail = daily_ret[daily_ret <= threshold]
        cvar_daily = float(np.mean(tail)) if len(tail) > 0 else float(threshold)
        cvar_ann_pct = cvar_daily * np.sqrt(252) * 100.0
    else:
        cvar_ann_pct = 0.0
    return {
        "nav": float(nav[-1]),
        "total_return_pct": float(total_return_pct),
        "ann_return_pct": float(ann_return_pct),
        "ann_vol_pct": float(ann_vol_pct),
        "sharpe": sharpe,
        "max_drawdown_pct": max_dd_pct,
        "cvar_95_ann_pct": float(cvar_ann_pct),
        "trading_days": trading_days,
    }


def bootstrap_sharpe_ci(
    nav_values: list[float],
    risk_free_rate_ann: float = 0.0,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Bootstrap 95% CI sur le Sharpe annualisé ajusté RF.
    Utilise un bloc bootstrap stationnaire (~1 mois) pour préserver
    l'autocorrélation des rendements volatilité.
    Retourne (lo, hi).
    """
    rng = np.random.default_rng(seed)
    nav = np.asarray(nav_values, dtype=float)
    if len(nav) < 4:
        return (0.0, 0.0)
    daily_ret = np.diff(nav) / nav[:-1]
    rf_daily = risk_free_rate_ann / 252.0
    excess_daily = daily_ret - rf_daily
    n = len(excess_daily)
    block = max(1, min(21, n // 4))  # ~1 mois
    sharpes = []
    for _ in range(n_boot):
        indices: list[int] = []
        while len(indices) < n:
            start = int(rng.integers(0, n))
            length = int(rng.geometric(1.0 / block))
            indices.extend(range(start, min(start + length, n)))
        sample = excess_daily[indices[:n]]
        s_std = sample.std(ddof=1)
        if s_std > 1e-12:
            sharpes.append(float(sample.mean() / s_std * np.sqrt(252)))
    if not sharpes:
        return (0.0, 0.0)
    sharpes_arr = np.sort(sharpes)
    alpha = (1.0 - ci) / 2.0
    return (float(np.quantile(sharpes_arr, alpha)), float(np.quantile(sharpes_arr, 1.0 - alpha)))


def sharpe_pvalue(sharpe_ann: float, trading_days: int) -> float:
    """
    P-value unilatérale pour H0 : Sharpe ≤ 0.
    Utilise la statistique t de Lo (2002) qui corrige pour le biais
    des rendements non-gaussiens (VRP strategies).
    """
    from scipy import stats
    n = max(trading_days, 2)
    s_daily = sharpe_ann / np.sqrt(252.0)
    se = np.sqrt((1.0 + 0.5 * s_daily ** 2) / n)
    t_stat = s_daily / se
    return float(stats.t.sf(t_stat, df=n - 1))


def _build_row_base(
    segment_label: str,
    start: str,
    end: str,
    leverage_multiplier: float,
    report,
    engine,
    rf_rate: float,
) -> dict:
    """Construit le dict de résultats d'un segment avec toutes les métriques jury."""
    lev = leverage_multiplier
    nav_history = list(engine._nav_history)

    # Bootstrap CI sur Sharpe (utilise la série NAV brute du moteur)
    ci_lo, ci_hi = bootstrap_sharpe_ci(nav_history, risk_free_rate_ann=rf_rate)
    tdays = max(len(nav_history) - 1, 1)
    pval = sharpe_pvalue(report.sharpe, tdays)

    # CVaR depuis la série NAV
    nav_arr = np.asarray(nav_history, dtype=float)
    if len(nav_arr) >= 10:
        dr = np.diff(nav_arr) / nav_arr[:-1]
        thr = np.percentile(dr, 5.0)
        tail = dr[dr <= thr]
        cvar_ann_pct = float(np.mean(tail)) * np.sqrt(252) * 100.0 if len(tail) > 0 else float(thr) * np.sqrt(252) * 100.0
    else:
        cvar_ann_pct = 0.0

    # Métriques non-leveragées (analytique — linéaire car DD << seuil halt)
    unlevered_ann_return = report.ann_return_pct / lev if lev > 0 else 0.0
    unlevered_ann_vol = report.ann_vol_pct / lev if lev > 0 else 0.0
    unlevered_sharpe = (
        (unlevered_ann_return - rf_rate * 100.0) / unlevered_ann_vol
        if unlevered_ann_vol > 1e-8
        else 0.0
    )

    # Attribution P&L par sleeve
    pnl_attribution = getattr(engine, "_strategy_pnl_by_asset", {})
    total_pnl = sum(abs(v) for v in pnl_attribution.values()) or 1.0
    vrp_pnl = sum(v for k, v in pnl_attribution.items() if not k.startswith("META_"))
    momentum_pnl = pnl_attribution.get("META_MOMENTUM", 0.0)
    carry_pnl = pnl_attribution.get("META_CARRY", 0.0)

    # Trades par an
    years = max((tdays / 252.0), 0.01)
    trades_per_year = report.n_trades / years

    return {
        "segment": segment_label,
        "start": start,
        "end": end,
        "leverage_multiplier": lev,
        "risk_free_rate_ann": round(rf_rate, 4),
        "nav": round(report.nav, 2),
        "total_return_pct": round(report.total_return_pct, 4),
        "ann_return_pct": round(report.ann_return_pct, 4),
        "ann_vol_pct": round(report.ann_vol_pct, 4),
        "sharpe": round(report.sharpe, 4),
        "sharpe_rf_adjusted": round(report.sharpe, 4),  # déjà ajusté dans StrategyEngine si RF passé
        "sharpe_ci_lo": round(ci_lo, 4),
        "sharpe_ci_hi": round(ci_hi, 4),
        "sharpe_pvalue": round(pval, 4),
        "sortino": round(report.sortino, 4),
        "calmar": round(report.calmar, 4),
        "max_drawdown_pct": round(report.max_drawdown_pct, 4),
        "cvar_95_ann_pct": round(cvar_ann_pct, 4),
        "trades": report.n_trades,
        "trades_per_year": round(trades_per_year, 2),
        "profit_factor": round(report.profit_factor, 4),
        "alpha_ann_pct": round(report.alpha_ann_pct or 0.0, 4),
        "active_return_ann_pct": round(report.active_return_ann_pct or 0.0, 4),
        "information_ratio": round(report.information_ratio or 0.0, 4),
        # Métriques non-leveragées
        "ann_return_unlevered_pct": round(unlevered_ann_return, 4),
        "ann_vol_unlevered_pct": round(unlevered_ann_vol, 4),
        "sharpe_unlevered": round(unlevered_sharpe, 4),
        "max_dd_unlevered_pct": round(report.max_drawdown_pct / lev if lev > 0 else 0.0, 4),
        # Attribution P&L
        "vrp_pnl_usd": round(vrp_pnl, 2),
        "momentum_pnl_usd": round(momentum_pnl, 2),
        "carry_pnl_usd": round(carry_pnl, 2),
        "vrp_pct_of_total": round(abs(vrp_pnl) / total_pnl * 100, 2),
        "momentum_pct_of_total": round(abs(momentum_pnl) / total_pnl * 100, 2),
        "carry_pct_of_total": round(abs(carry_pnl) / total_pnl * 100, 2),
    }


def main() -> None:
    args = build_parser().parse_args()

    rf_by_period = DEFAULT_CONFIG.model.risk_free_rate_by_period

    segments = [
        SegmentSpec("train", args.train_start, args.train_end),
        SegmentSpec("validation", args.val_start, args.val_end),
        SegmentSpec("test", args.test_start, args.test_end),
    ]

    rows: list[dict] = []
    oos_nav_path: list[float] = [DEFAULT_CONFIG.capital.paper_capital]
    segment_nav_histories: dict[str, list[float]] = {}

    for segment in segments:
        cfg = copy.deepcopy(DEFAULT_CONFIG)
        cfg.model.backtest_deterministic = True
        cfg.risk.strategy_leverage_multiplier = args.leverage_multiplier

        rf_rate = _get_risk_free_rate(segment.start, rf_by_period)

        engine = StrategyEngine(config=cfg, mode="backtest", log_level=args.log_level)
        report = engine.run(start=segment.start, end=segment.end, verbose=False)

        row = _build_row_base(
            segment_label=segment.label,
            start=segment.start,
            end=segment.end,
            leverage_multiplier=args.leverage_multiplier,
            report=report,
            engine=engine,
            rf_rate=rf_rate,
        )
        rows.append(row)

        nav_hist = list(engine._nav_history)
        segment_nav_histories[segment.label] = nav_hist

        if segment.label in {"validation", "test"}:
            oos_segment_nav = nav_hist[1:]
            if oos_segment_nav:
                scale = oos_nav_path[-1] / DEFAULT_CONFIG.capital.paper_capital
                oos_nav_path.extend([value * scale for value in oos_segment_nav])

        jury_flag = " ⚠ trades/an < 40" if row["trades_per_year"] < 40.0 else ""
        print(
            f"SEGMENT {segment.label} | ann={row['ann_return_pct']:+.2f}% "
            f"(unlev={row['ann_return_unlevered_pct']:+.2f}%) | "
            f"sharpe={row['sharpe']:.3f} [CI {row['sharpe_ci_lo']:.2f};{row['sharpe_ci_hi']:.2f}] "
            f"p={row['sharpe_pvalue']:.3f} | "
            f"max_dd={row['max_drawdown_pct']:.2f}% | CVaR95={row['cvar_95_ann_pct']:.2f}% | "
            f"trades/an={row['trades_per_year']:.1f} | "
            f"VRP={row['vrp_pct_of_total']:.0f}% mom={row['momentum_pct_of_total']:.0f}% carry={row['carry_pct_of_total']:.0f}%"
            f"{jury_flag}"
        )

    # ── OOS combiné ──────────────────────────────────────────────────────────
    rf_oos = _get_risk_free_rate(args.val_start, rf_by_period)
    oos_metrics = compute_nav_metrics(oos_nav_path, DEFAULT_CONFIG.capital.paper_capital, risk_free_rate_ann=rf_oos)
    oos_ci_lo, oos_ci_hi = bootstrap_sharpe_ci(oos_nav_path, risk_free_rate_ann=rf_oos)
    oos_pval = sharpe_pvalue(oos_metrics["sharpe"], int(oos_metrics["trading_days"]))
    oos_trades = int(sum(int(r["trades"]) for r in rows if r["segment"] in {"validation", "test"}))
    oos_years = max(oos_metrics["trading_days"] / 252.0, 0.01)

    lev = args.leverage_multiplier
    oos_unlev_ret = oos_metrics["ann_return_pct"] / lev if lev > 0 else 0.0
    oos_unlev_vol = oos_metrics["ann_vol_pct"] / lev if lev > 0 else 0.0
    oos_unlev_sharpe = (oos_unlev_ret - rf_oos * 100.0) / oos_unlev_vol if oos_unlev_vol > 1e-8 else 0.0

    # Gap de généralisation
    val_row = next((r for r in rows if r["segment"] == "validation"), None)
    test_row = next((r for r in rows if r["segment"] == "test"), None)
    generalization_gap_pct = 0.0
    generalization_warning = ""
    if val_row and test_row:
        val_sharpe = float(val_row["sharpe"])
        test_sharpe = float(test_row["sharpe"])
        if val_sharpe > 1e-6:
            generalization_gap_pct = (val_sharpe - test_sharpe) / val_sharpe * 100.0
            if generalization_gap_pct > 30.0:
                generalization_warning = (
                    f"JURY_WARNING: Sharpe drop {generalization_gap_pct:.1f}% "
                    f"(val={val_sharpe:.3f} → test={test_sharpe:.3f}) > seuil 30%."
                )
                print(generalization_warning)

    oos_row = {
        "segment": "oos_combined",
        "start": args.val_start,
        "end": args.test_end,
        "leverage_multiplier": lev,
        "risk_free_rate_ann": round(rf_oos, 4),
        "nav": round(oos_metrics["nav"], 2),
        "total_return_pct": round(oos_metrics["total_return_pct"], 4),
        "ann_return_pct": round(oos_metrics["ann_return_pct"], 4),
        "ann_vol_pct": round(oos_metrics["ann_vol_pct"], 4),
        "sharpe": round(oos_metrics["sharpe"], 4),
        "sharpe_rf_adjusted": round(oos_metrics["sharpe"], 4),
        "sharpe_ci_lo": round(oos_ci_lo, 4),
        "sharpe_ci_hi": round(oos_ci_hi, 4),
        "sharpe_pvalue": round(oos_pval, 4),
        "sortino": 0.0,
        "calmar": round(oos_metrics["ann_return_pct"] / oos_metrics["max_drawdown_pct"], 4)
        if oos_metrics["max_drawdown_pct"] > 1e-8
        else 0.0,
        "max_drawdown_pct": round(oos_metrics["max_drawdown_pct"], 4),
        "cvar_95_ann_pct": round(oos_metrics["cvar_95_ann_pct"], 4),
        "trades": oos_trades,
        "trades_per_year": round(oos_trades / oos_years, 2),
        "profit_factor": 0.0,
        "alpha_ann_pct": 0.0,
        "active_return_ann_pct": 0.0,
        "information_ratio": 0.0,
        "ann_return_unlevered_pct": round(oos_unlev_ret, 4),
        "ann_vol_unlevered_pct": round(oos_unlev_vol, 4),
        "sharpe_unlevered": round(oos_unlev_sharpe, 4),
        "max_dd_unlevered_pct": round(oos_metrics["max_drawdown_pct"] / lev if lev > 0 else 0.0, 4),
        "vrp_pnl_usd": 0.0,
        "momentum_pnl_usd": 0.0,
        "carry_pnl_usd": 0.0,
        "vrp_pct_of_total": 0.0,
        "momentum_pct_of_total": 0.0,
        "carry_pct_of_total": 0.0,
        "generalization_gap_pct": round(generalization_gap_pct, 2),
        "generalization_warning": generalization_warning,
    }
    rows.append(oos_row)
    print(
        f"SEGMENT oos_combined | ann={oos_row['ann_return_pct']:+.2f}% "
        f"(unlev={oos_row['ann_return_unlevered_pct']:+.2f}%) | "
        f"sharpe={oos_row['sharpe']:.3f} [CI {oos_row['sharpe_ci_lo']:.2f};{oos_row['sharpe_ci_hi']:.2f}] "
        f"p={oos_row['sharpe_pvalue']:.3f} | "
        f"max_dd={oos_row['max_drawdown_pct']:.2f}% | CVaR95={oos_row['cvar_95_ann_pct']:.2f}% | "
        f"gap_gen={oos_row['generalization_gap_pct']:.1f}%"
    )

    # ── Segments de stress explicites ────────────────────────────────────────
    stress_specs = [
        SegmentSpec("stress_covid_2020", "2020-01-20", "2020-04-30"),
        SegmentSpec("stress_bear_2022", "2022-01-01", "2022-12-31"),
    ]
    for stress_seg in stress_specs:
        cfg = copy.deepcopy(DEFAULT_CONFIG)
        cfg.model.backtest_deterministic = True
        cfg.risk.strategy_leverage_multiplier = args.leverage_multiplier
        rf_stress = _get_risk_free_rate(stress_seg.start, rf_by_period)
        engine = StrategyEngine(config=cfg, mode="backtest", log_level=args.log_level)
        report = engine.run(start=stress_seg.start, end=stress_seg.end, verbose=False)
        srow = _build_row_base(
            segment_label=stress_seg.label,
            start=stress_seg.start,
            end=stress_seg.end,
            leverage_multiplier=args.leverage_multiplier,
            report=report,
            engine=engine,
            rf_rate=rf_stress,
        )
        # Champs OOS-only absents dans stress
        srow.setdefault("generalization_gap_pct", 0.0)
        srow.setdefault("generalization_warning", "")
        rows.append(srow)
        print(
            f"STRESS {stress_seg.label} | ann={srow['ann_return_pct']:+.2f}% | "
            f"sharpe={srow['sharpe']:.3f} | max_dd={srow['max_drawdown_pct']:.2f}% | "
            f"trades={srow['trades']}"
        )

    # Assurer que toutes les lignes ont les mêmes colonnes
    all_keys: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                all_keys.append(k)
                seen.add(k)
    for r in rows:
        for k in all_keys:
            r.setdefault(k, "")

    output_csv = Path(args.output_csv).expanduser()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(rows)

    summary_lines = [
        f"WALKFORWARD leverage={args.leverage_multiplier:.2f}",
        *[
            (
                f"{r['segment']} {r['start']} {r['end']} "
                f"ann={r['ann_return_pct']:+.2f}% (unlev={r['ann_return_unlevered_pct']:+.2f}%) "
                f"sharpe={r['sharpe']:.3f} [CI {r['sharpe_ci_lo']:.2f};{r['sharpe_ci_hi']:.2f}] p={r['sharpe_pvalue']:.3f} "
                f"max_dd={r['max_drawdown_pct']:.2f}% CVaR95={r['cvar_95_ann_pct']:.2f}% "
                f"trades/an={r['trades_per_year']:.1f} ir={r['information_ratio']:.3f} nav=${r['nav']:,.2f}"
            )
            for r in rows
        ],
    ]
    if generalization_warning:
        summary_lines.append(generalization_warning)

    output_summary = Path(args.output_summary).expanduser()
    output_summary.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"CSV {output_csv}")
    print(f"SUMMARY {output_summary}")


if __name__ == "__main__":
    main()
