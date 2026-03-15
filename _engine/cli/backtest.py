from __future__ import annotations

import argparse
import time

from .._utils import _format_time
from ..models import CandleRequest
from ._helpers import _resolve, load_candles, generate_signals, resolve_pairlist
from strategy import list_strategies


def _run_single_backtest(
    symbol: str,
    exchange: str,
    timeframe: str,
    session: str,
    adjustment: str,
    strategy_name: str,
    initial_capital: float,
    data_dir: str,
    mode: str,
    sl: float,
    tp: float,
    start: str | None,
    end: str | None,
) -> int:
    from ..backtest.engine import TradingViewLikeBacktester
    from .._utils import build_risk

    candle_request = CandleRequest(
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        start=start,
        end=end,
        session=session,
        adjustment=adjustment,
    )

    print(f"\n{'='*60}")
    print("  HyperView - Backtest")
    print(f"  {exchange}:{symbol} | {timeframe} | mode={mode}")
    print(f"  Strategy: {strategy_name} | SL={sl}% TP={tp}%")
    print(f"{'='*60}")

    # 1. Load data
    candles = load_candles(candle_request, data_dir, step="1/3")

    # 2. Generate signals
    signal_frame, strategy = generate_signals(
        strategy_name, candles, mode, start, end, step="2/3",
    )

    # 3. Run backtest
    print("\n[3/3] Running backtest...")
    t0 = time.time()
    risk = build_risk(mode, sl, tp)
    backtester = TradingViewLikeBacktester(candle_request=candle_request, initial_equity=initial_capital)
    result = backtester.run(signal_frame, risk, mode)
    m = result.metrics
    print(f"      Done ({_format_time(time.time() - t0)})")

    print(f"\n{'-'*60}")
    print(f"  Net Profit:    {m.net_profit_pct:>10.2f}%")
    print(f"  Max Drawdown:  {m.max_drawdown_pct:>10.2f}%")
    print(f"  Win Rate:      {m.win_rate_pct:>10.2f}%")
    print(f"  Profit Factor: {m.profit_factor:>10.2f}")
    print(f"  Trades:        {m.trade_count:>10d}")
    print(f"  Final Equity:  ${m.equity_final:>12,.2f}")
    print(f"{'-'*60}")
    return 0


def run_backtest(args: argparse.Namespace, config: dict) -> int:
    timeframe = _resolve(args, config, "timeframe")
    session = _resolve(args, config, "session")
    adjustment = args.adjustment
    strategy_name = _resolve(args, config, "strategy")
    if strategy_name is None:
        available = ", ".join(list_strategies()) or "(none)"
        print(f"Error: No strategy specified. Use --strategy NAME or set 'strategy' in config.json.")
        print(f"Available strategies: {available}")
        return 1
    initial_capital = config["initial_capital"]
    data_dir = config["data_dir"]

    pairs = resolve_pairlist(args, config)

    if len(pairs) > 1:
        print(f"\n{'#'*60}")
        print(f"  Running backtest for {len(pairs)} pair(s) from pairlist")
        print(f"{'#'*60}")

    rc = 0
    for symbol, pair_exchange in pairs:
        result = _run_single_backtest(
            symbol=symbol,
            exchange=pair_exchange,
            timeframe=timeframe,
            session=session,
            adjustment=adjustment,
            strategy_name=strategy_name,
            initial_capital=initial_capital,
            data_dir=data_dir,
            mode=args.mode,
            sl=args.sl,
            tp=args.tp,
            start=args.start,
            end=args.end,
        )
        if result != 0:
            rc = result
    return rc
