from __future__ import annotations

import argparse
import time
from pathlib import Path

from .._utils import _format_time
from ..models import CandleRequest, OptimizationRequest
from ._helpers import _resolve, load_candles, generate_signals, resolve_pairlist
from strategy import list_strategies


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
) -> int:
    from ..hyperopt import run_optimization

    candle_request = CandleRequest(
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        start=start,
        end=end,
        session=session_type,
        adjustment=adjustment,
    )

    print(f"\n{'='*60}")
    print("  HyperView - Hyper-Optimization")
    print(f"  {exchange}:{symbol} | {timeframe} | mode={mode}")
    print(f"  Strategy: {strategy_name}")
    print(f"  SL range: {sl_min}-{sl_max}")
    print(f"  TP range: {tp_min}-{tp_max}")
    method_label = "Bayesian (Optuna TPE)" if search_method == "bayesian" else "Two-stage grid"
    print(f"  Search:   {method_label}")
    if search_method == "bayesian":
        print(f"  Trials:   {n_trials}")
    print(f"{'='*60}")

    # 1. Load data
    candles = load_candles(candle_request, data_dir, step="1/4")

    # 2. Generate signals
    signal_frame, strategy = generate_signals(
        strategy_name, candles, mode, start, end, step="2/4",
    )
    buy_count = int(signal_frame["buy_signal"].sum())
    sell_count = int(signal_frame["sell_signal"].sum())
    total_signals = buy_count + sell_count
    in_range_bars = int(signal_frame["in_date_range"].sum())
    density_per_1k = total_signals / max(in_range_bars, 1) * 1000
    if density_per_1k < 5:
        density_label = "low"
    elif density_per_1k < 20:
        density_label = "medium"
    else:
        density_label = "high"
    print(f"      Signal density: {density_label} ({density_per_1k:.2f}/1k bars) "
          f"over {in_range_bars} in-range bars")

    print("\n[3/4] Running optimization...")
    t_opt = time.time()
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
    output_path = Path(output_dir_base) / f"{strategy_name}_{exchange}_{symbol}_{timeframe}_{mode}.json"
    bundle = run_optimization(
        signal_frame=signal_frame,
        candle_request=candle_request,
        strategy=strategy,
        request=request,
        output_path=output_path,
        initial_equity=initial_capital,
    )
    print(f"      Optimization complete ({_format_time(time.time() - t_opt)})")

    print("\n[4/4] Results")
    print("\nTop candidates:")
    for rank, metrics in enumerate(bundle.results, start=1):
        print(
            f"{rank:>2}. SL={metrics.sl_pct:.4f}% TP={metrics.tp_pct:.4f}% "
            f"net={metrics.net_profit_pct:.4f}% dd={metrics.max_drawdown_pct:.4f}% "
            f"win={metrics.win_rate_pct:.4f}% pf={metrics.profit_factor} trades={metrics.trade_count}"
        )
    print(f"\nResults written to {output_path}")
    return 0


def run_hyperopt(args: argparse.Namespace, config: dict) -> int:
    timeframe = _resolve(args, config, "timeframe")
    session_type = _resolve(args, config, "session")
    adjustment = args.adjustment
    strategy_name = _resolve(args, config, "strategy")
    if strategy_name is None:
        available = ", ".join(list_strategies()) or "(none)"
        print(f"Error: No strategy specified. Use --strategy NAME or set 'strategy' in config.json.")
        print(f"Available strategies: {available}")
        return 1
    initial_capital = config["initial_capital"]
    data_dir = config["data_dir"]
    output_dir_base = config["output_dir"]

    opt_cfg = config.get("optimization", {})
    sl_range = opt_cfg.get("sl_range", {})
    tp_range = opt_cfg.get("tp_range", {})
    sl_min = args.sl_min if args.sl_min is not None else sl_range.get("min")
    sl_max = args.sl_max if args.sl_max is not None else sl_range.get("max")
    tp_min = args.tp_min if args.tp_min is not None else tp_range.get("min")
    tp_max = args.tp_max if args.tp_max is not None else tp_range.get("max")
    objective = args.objective or opt_cfg.get("objective")
    top_n = args.top_n if args.top_n is not None else opt_cfg.get("top_n")
    search_method = args.search_method or opt_cfg.get("search_method")
    n_trials = args.n_trials if args.n_trials is not None else opt_cfg.get("n_trials")
    symbols = resolve_pairlist(args, config)

    if len(symbols) > 1:
        print(f"\n{'#'*60}")
        print(f"  Running hyperopt for {len(symbols)} pair(s) from pairlist")
        print(f"{'#'*60}")

    rc = 0
    for symbol, pair_exchange in symbols:
        result = _run_single_hyperopt(
            symbol=symbol,
            exchange=pair_exchange,
            timeframe=timeframe,
            session_type=session_type,
            adjustment=adjustment,
            strategy_name=strategy_name,
            initial_capital=initial_capital,
            data_dir=data_dir,
            output_dir_base=output_dir_base,
            mode=args.mode,
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
        if result != 0:
            rc = result
    return rc
