from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from ..models import BacktestMetrics, CandleRequest, OptimizationRequest
from ..presets import save_best_preset, strategy_preset_path
from ._helpers import _resolve, generate_signals, load_candles, print_hyperopt_table, resolve_pairlist
from strategy import list_strategies

from ..hyperopt import run_optimization


def _signal_density_suffix(signal_frame: Any) -> str:
    """Return a density string fragment like ' | High density (86.98/1k bars)'."""
    buy_count = int(signal_frame["buy_signal"].sum())
    sell_count = int(signal_frame["sell_signal"].sum())
    total_signals = buy_count + sell_count
    in_range_bars = int(signal_frame["in_date_range"].sum())

    density_per_1k = (total_signals / max(in_range_bars, 1)) * 1000

    if density_per_1k < 5:
        density_label = "Low"
    elif density_per_1k < 20:
        density_label = "Medium"
    else:
        density_label = "High"

    return f" | {density_label} density ({density_per_1k:.2f}/1k bars)"


def _run_single_hyperopt(
    symbol: str,
    exchange: str,
    timeframe: str,
    session_type: str,
    adjustment: str,
    strategy_name: str,
    initial_capital: float,
    data_dir: str,
    output_dir_base: str,
    mode: str,
    start: str | None,
    end: str | None,
    sl_min: float,
    sl_max: float,
    tp_min: float,
    tp_max: float,
    objective: str,
    top_n: int,
    search_method: str,
    n_trials: int,
) -> tuple[int, tuple[float, float] | None]:
    candle_request = CandleRequest(
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        start=start,
        end=end,
        session=session_type,
        adjustment=adjustment,
    )

    try:
        candles = load_candles(candle_request, data_dir, compact=True)
        if candles is None or len(candles) == 0:
            print("  Error: No candle data found. Optimization aborted.")
            return 1, None

        signal_frame, strategy = generate_signals(
            strategy_name,
            candles,
            mode,
            start,
            end,
            quiet=True,
        )
        if signal_frame is None or len(signal_frame) == 0:
            print("  Error: No signals generated. Optimization aborted.")
            return 1, None

        buy_count = int(signal_frame["buy_signal"].sum())
        sell_count = int(signal_frame["sell_signal"].sum())
        density = _signal_density_suffix(signal_frame)
        print(f"   \u2022 Signals : {buy_count} buy / {sell_count} sell ({density.lstrip(' | ')})")

        request = OptimizationRequest(
            candle_request=candle_request,
            mode=mode,
            objective=objective,
            sl_min=sl_min,
            sl_max=sl_max,
            tp_min=tp_min,
            tp_max=tp_max,
            top_n=top_n,
            search_method=search_method,
            n_trials=n_trials,
            initial_equity=initial_capital,
        )

        bundle = run_optimization(
            signal_frame=signal_frame,
            candle_request=candle_request,
            strategy=strategy,
            request=request,
            output_path=strategy_preset_path(output_dir_base, strategy_name),
            initial_equity=initial_capital,
        )

        print_hyperopt_table(bundle.results, top_n)

        if not bundle.results:
            print("  Error: Optimization produced no ranked results.")
            return 1, None

        best = bundle.results[0]
        preset_path = _save_best_result_to_presets(
            output_dir_base=output_dir_base,
            exchange=exchange,
            symbol=symbol,
            strategy_name=strategy_name,
            timeframe=timeframe,
            session=session_type,
            adjustment=adjustment,
            mode=mode,
            sl=best.sl_pct,
            tp=best.tp_pct,
            objective=objective,
            search_method=search_method,
            best_metrics=best,
        )
        from rich.console import Console as _Con
        _Con().print(f"   [green]✔[/green] Best preset saved to: {preset_path}")
        return 0, (best.sl_pct, best.tp_pct)

    except Exception as e:
        print(f"\n  Error: Optimization failed ({e})")
        logging.exception(f"Exception during single hyperopt for {exchange}:{symbol}")
        return 1, None


def run_hyperopt(args: argparse.Namespace, config: dict) -> int:
    strategy_name = _resolve(args, config, "strategy")
    available_strategies = list_strategies()

    if not strategy_name or strategy_name not in available_strategies:
        print("Error: Invalid or no strategy specified.")
        print(f"Available strategies: {', '.join(available_strategies) or '(none)'}")
        return 1

    timeframe = _resolve(args, config, "timeframe")
    session_type = _resolve(args, config, "session")
    mode = _resolve(args, config, "mode", "long")
    adjustment = args.adjustment
    initial_capital = config.get("initial_capital", 1000.0)
    data_dir = config.get("data_dir", "./data")

    output_dir_base = config.get("output_dir", "./outputs")
    Path(output_dir_base).mkdir(parents=True, exist_ok=True)

    opt_cfg = config.get("optimization", {})
    sl_range = opt_cfg.get("sl_range", {})
    tp_range = opt_cfg.get("tp_range", {})

    sl_min = args.sl_min if args.sl_min is not None else sl_range.get("min", 0.5)
    sl_max = args.sl_max if args.sl_max is not None else sl_range.get("max", 5.0)
    tp_min = args.tp_min if args.tp_min is not None else tp_range.get("min", 1.0)
    tp_max = args.tp_max if args.tp_max is not None else tp_range.get("max", 10.0)

    objective = args.objective or opt_cfg.get("objective", "net_profit_pct")
    top_n = args.top_n if args.top_n is not None else opt_cfg.get("top_n", 5)
    search_method = args.search_method or opt_cfg.get("search_method", "grid")
    n_trials = args.n_trials if args.n_trials is not None else opt_cfg.get("n_trials", 100)

    symbols = resolve_pairlist(args, config)
    if not symbols:
        print("Error: No trading pairs resolved. Please check your config or arguments.")
        return 1

    from rich.console import Console
    from rich.panel import Panel
    console = Console()

    header_text = (
        f"⚙ HYPERVIEW - BATCH HYPER-OPTIMIZATION\n"
        f"  Strategy : {strategy_name} | Mode: {mode} | Timeframe: {timeframe}\n"
        f"  Targets  : {len(symbols)} pair(s)"
    )
    console.print()
    console.print(Panel(header_text, expand=True))

    rc = 0
    for index, (symbol, pair_exchange) in enumerate(symbols, start=1):
        line = "─" * 72
        console.print(f"\n 📊 [{index}/{len(symbols)}] {pair_exchange}:{symbol}")
        console.print(f" {line}")

        pair_rc, best_values = _run_single_hyperopt(
            symbol=symbol,
            exchange=pair_exchange,
            timeframe=timeframe,
            session_type=session_type,
            adjustment=adjustment,
            strategy_name=strategy_name,
            initial_capital=initial_capital,
            data_dir=data_dir,
            output_dir_base=output_dir_base,
            mode=mode,
            start=args.start,
            end=args.end,
            sl_min=sl_min,
            sl_max=sl_max,
            tp_min=tp_min,
            tp_max=tp_max,
            objective=objective,
            top_n=top_n,
            search_method=search_method,
            n_trials=n_trials,
        )
        if pair_rc != 0:
            rc = 1
            continue

    return rc


def _extract_metrics(best: BacktestMetrics) -> dict[str, object]:
    return {
        "net_profit_pct": round(best.net_profit_pct, 2),
        "max_drawdown_pct": round(best.max_drawdown_pct, 2),
        "win_rate_pct": round(best.win_rate_pct, 2),
        "profit_factor": round(best.profit_factor, 2),
        "trade_count": best.trade_count,
        "equity_final": round(best.equity_final, 2),
        "sharpe_ratio": round(best.sharpe_ratio, 2),
        "calmar_ratio": round(best.calmar_ratio, 2),
        "expectancy_pct": round(best.expectancy_pct, 2),
        "avg_win_pct": round(best.avg_win_pct, 2),
        "avg_loss_pct": round(best.avg_loss_pct, 2),
        "worst_trade_pct": round(best.worst_trade_pct, 2),
        "max_consec_losses": best.max_consec_losses,
        "sl_exit_pct": round(best.sl_exit_pct, 2),
        "tp_exit_pct": round(best.tp_exit_pct, 2),
        "signal_exit_pct": round(best.signal_exit_pct, 2),
    }


def _save_best_result_to_presets(
    *,
    output_dir_base: str,
    exchange: str,
    symbol: str,
    strategy_name: str,
    timeframe: str,
    session: str,
    adjustment: str,
    mode: str,
    sl: float,
    tp: float,
    objective: str,
    search_method: str,
    best_metrics: BacktestMetrics,
) -> Path:
    preset_path = strategy_preset_path(output_dir_base, strategy_name)
    return save_best_preset(
        preset_path,
        strategy=strategy_name,
        pair=f"{exchange}:{symbol}",
        timeframe=timeframe,
        session=session,
        adjustment=adjustment,
        mode=mode,
        sl=sl,
        tp=tp,
        objective=objective,
        search_method=search_method,
        metrics=_extract_metrics(best_metrics),
    )
