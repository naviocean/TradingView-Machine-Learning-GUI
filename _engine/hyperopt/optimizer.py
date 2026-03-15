from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

from .._utils import _format_time, build_risk
from ..backtest.engine import TradingViewLikeBacktester
from ..models import BacktestMetrics, CandleRequest, Mode, Objective, OptimizationBundle, OptimizationRequest, RiskParameters
from strategy import BaseStrategy


def _print_progress(current: int, total: int, start_time: float, phase: str) -> None:
    elapsed = time.time() - start_time
    pct = current / total * 100
    bar_width = 30
    filled = int(bar_width * current // total)
    bar = "\u2588" * filled + "\u2591" * (bar_width - filled)
    if current > 0:
        eta = elapsed / current * (total - current)
        eta_str = f"ETA {_format_time(eta)}"
    else:
        eta_str = "ETA --"
    sys.stdout.write(
        f"\r  {phase}: |{bar}| {pct:5.1f}%  [{current}/{total}]  "
        f"elapsed {_format_time(elapsed)}  {eta_str}   "
    )
    sys.stdout.flush()
    if current == total:
        sys.stdout.write("\n")


def run_optimization(
    signal_frame: pd.DataFrame,
    candle_request: CandleRequest,
    strategy: BaseStrategy,
    request: OptimizationRequest,
    output_path: Path,
    initial_equity: float = 100_000.0,
) -> OptimizationBundle:
    """Top-level entry point for SL/TP optimization."""
    optimizer = _Optimizer(
        signal_frame=signal_frame,
        candle_request=candle_request,
        strategy=strategy,
        initial_equity=initial_equity,
    )
    return optimizer.optimize(request, output_path)


class _Optimizer:
    def __init__(
        self,
        signal_frame: pd.DataFrame,
        candle_request: CandleRequest,
        strategy: BaseStrategy,
        initial_equity: float = 100_000.0,
    ) -> None:
        self.signal_frame = signal_frame
        self.candle_request = candle_request
        self.strategy = strategy
        self.initial_equity = initial_equity
        self.backtester = TradingViewLikeBacktester(
            candle_request=candle_request,
            initial_equity=initial_equity,
        )

    def optimize(self, request: OptimizationRequest, output_path: Path) -> OptimizationBundle:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if request.search_method == "bayesian":
            return self._optimize_bayesian(request, output_path)
        return self._optimize_grid(request, output_path)

    def _optimize_grid(self, request: OptimizationRequest, output_path: Path) -> OptimizationBundle:
        coarse_sl_step = _auto_step(request.sl_min, request.sl_max)
        coarse_tp_step = _auto_step(request.tp_min, request.tp_max)
        sl_values = _build_range(request.sl_min, request.sl_max, coarse_sl_step)
        tp_values = _build_range(request.tp_min, request.tp_max, coarse_tp_step)
        print(f"\n[Coarse grid] {len(sl_values)} SL x {len(tp_values)} TP = {len(sl_values) * len(tp_values)} combinations")
        coarse_results = self._evaluate_grid(
            mode=request.mode,
            objective=request.objective,
            sl_values=sl_values,
            tp_values=tp_values,
            phase="Coarse",
        )

        best_coarse = coarse_results[0]
        fine_sl_step = max(coarse_sl_step / 2.0, 0.0001)
        fine_tp_step = max(coarse_tp_step / 2.0, 0.0001)
        fine_sl = _refine_range(best_coarse.sl_pct, request.sl_min, request.sl_max, coarse_sl_step, fine_sl_step)
        fine_tp = _refine_range(best_coarse.tp_pct, request.tp_min, request.tp_max, coarse_tp_step, fine_tp_step)
        print(f"\n[Fine grid] {len(fine_sl)} SL x {len(fine_tp)} TP = {len(fine_sl) * len(fine_tp)} combinations  (best coarse SL={best_coarse.sl_pct:.4f}% TP={best_coarse.tp_pct:.4f}%)")
        fine_results = self._evaluate_grid(
            mode=request.mode,
            objective=request.objective,
            sl_values=fine_sl,
            tp_values=fine_tp,
            phase="Fine",
        )

        combined = _deduplicate_and_rank(coarse_results + fine_results, request.objective)
        top_results = combined[: request.top_n]
        bundle = OptimizationBundle(
            request=request,
            results=top_results,
            coarse_results=coarse_results[: request.top_n],
            output_path=output_path,
        )
        _write_output(bundle)
        return bundle

    def _optimize_bayesian(self, request: OptimizationRequest, output_path: Path) -> OptimizationBundle:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        n_trials = request.n_trials
        objective_name = request.objective
        minimize = objective_name == "max_drawdown_pct"

        all_results: list[BacktestMetrics] = []
        start_time = time.time()

        print(f"\n[Bayesian] {n_trials} trials via Optuna TPE  (SL {request.sl_min}–{request.sl_max}, TP {request.tp_min}–{request.tp_max})")

        def objective(trial: optuna.Trial) -> float:
            sl_value = trial.suggest_float("sl_pct", request.sl_min, request.sl_max, step=0.01)
            tp_value = trial.suggest_float("tp_pct", request.tp_min, request.tp_max, step=0.01)
            risk = build_risk(request.mode, sl_value, tp_value)
            result = self.backtester.run(self.signal_frame, risk, request.mode)
            all_results.append(result.metrics)
            _print_progress(len(all_results), n_trials, start_time, "Bayesian")
            return _objective_value(result.metrics, objective_name)

        direction = "minimize" if minimize else "maximize"
        study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials)

        ranked = _rank_results(all_results, objective_name)
        top_results = ranked[: request.top_n]
        bundle = OptimizationBundle(
            request=request,
            results=top_results,
            coarse_results=[],
            output_path=output_path,
        )
        _write_output(bundle)
        return bundle

    def _evaluate_grid(
        self,
        mode: Mode,
        objective: Objective,
        sl_values: list[float],
        tp_values: list[float],
        phase: str = "Grid",
    ) -> list[BacktestMetrics]:
        results: list[BacktestMetrics] = []
        total = len(sl_values) * len(tp_values)
        start_time = time.time()
        count = 0
        for sl_value in sl_values:
            for tp_value in tp_values:
                risk = build_risk(mode, sl_value, tp_value)
                result = self.backtester.run(self.signal_frame, risk, mode)
                results.append(result.metrics)
                count += 1
                _print_progress(count, total, start_time, phase)
        return _rank_results(results, objective)


# ------------------------------------------------------------------
# Stateless helpers
# ------------------------------------------------------------------

def _write_output(bundle: OptimizationBundle) -> None:
    bundle.output_path.write_text(
        json.dumps(bundle.to_dict(), indent=2),
        encoding="utf-8",
    )


def _deduplicate_and_rank(results: list[BacktestMetrics], objective: Objective) -> list[BacktestMetrics]:
    deduped: dict[tuple[float, float, str], BacktestMetrics] = {}
    for result in results:
        key = (result.sl_pct, result.tp_pct, result.mode)
        existing = deduped.get(key)
        if existing is None:
            deduped[key] = result
            continue
        if objective == "max_drawdown_pct":
            if result.max_drawdown_pct < existing.max_drawdown_pct:
                deduped[key] = result
        elif _objective_value(result, objective) > _objective_value(existing, objective):
            deduped[key] = result
    return _rank_results(list(deduped.values()), objective)


def _rank_results(results: list[BacktestMetrics], objective: Objective) -> list[BacktestMetrics]:
    reverse = objective != "max_drawdown_pct"
    ranked = sorted(results, key=lambda item: _objective_value(item, objective), reverse=reverse)
    for index, item in enumerate(ranked, start=1):
        item.rank = index
    return ranked


def _objective_value(result: BacktestMetrics, objective: Objective) -> float:
    if objective == "net_profit_pct":
        return result.net_profit_pct
    if objective == "profit_factor":
        return result.profit_factor
    if objective == "win_rate_pct":
        return result.win_rate_pct
    if objective == "trade_count":
        return float(result.trade_count)
    return result.max_drawdown_pct


_GRID_POINTS = 29


def _auto_step(low: float, high: float, points: int = _GRID_POINTS) -> float:
    """Derive a step size that divides [low, high] into *points* intervals."""
    span = high - low
    if span <= 0 or points < 2:
        return max(span, 0.0001)
    return round(span / (points - 1), 8)


def _build_range(start: float, end: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError("Step must be greater than zero.")
    values: list[float] = []
    current = start
    while current <= end + (step / 10.0):
        values.append(round(current, 8))
        current += step
    return values


def _refine_range(center: float, lower_bound: float, upper_bound: float, coarse_step: float, fine_step: float) -> list[float]:
    start = max(lower_bound, center - coarse_step)
    end = min(upper_bound, center + coarse_step)
    return _build_range(start, end, fine_step)
