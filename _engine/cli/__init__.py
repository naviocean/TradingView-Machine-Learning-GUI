from __future__ import annotations

import argparse

from .._runtime import configure_pycache
from ..config import load_config
from .download import run_download_data
from .backtest import run_backtest
from .hyperopt import run_hyperopt
from .list import run_list_data, run_list_strategies


configure_pycache()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hyperview",
        description="HyperView - CLI-driven TradingView strategy backtester and hyper-optimizer",
    )
    parser.add_argument("--config", default=None, help="Path to config.json (default: auto-detect)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ------ download-data ------
    dl = subparsers.add_parser("download-data", help="Download candle data for one or more pairs")
    dl.add_argument("--pairs", nargs="+", default=None, help="Pairs to download (e.g. NASDAQ:NFLX COINBASE:BTCUSD); falls back to config pairlist")
    dl.add_argument("--timeframe", default=None)
    dl.add_argument("--start", default=None)
    dl.add_argument("--end", default=None)
    dl.add_argument("--session", default=None)
    dl.add_argument("--adjustment", default="splits", help="Price adjustment: splits, dividends, none (default: splits)")

    # ------ backtest ------
    bt = subparsers.add_parser("backtest", help="Run a single backtest with fixed SL/TP")
    bt.add_argument("--symbol", default=None, help="Single pair to backtest (e.g. NASDAQ:TSLA); falls back to config pairlist")
    bt.add_argument("--timeframe", default=None)
    bt.add_argument("--start", default=None)
    bt.add_argument("--end", default=None)
    bt.add_argument("--session", default=None)
    bt.add_argument("--adjustment", default="splits", help="Price adjustment: splits, dividends, none (default: splits)")
    bt.add_argument("--strategy", default=None, help="Strategy name (default: from config)")
    bt.add_argument("--sl", type=float, required=True, help="Stop-loss %%")
    bt.add_argument("--tp", type=float, required=True, help="Take-profit %%")
    bt.add_argument("--mode", choices=["long", "short", "both"], default="long")

    # ------ hyperopt ------
    ho = subparsers.add_parser("hyperopt", help="Hyper-optimize SL/TP parameters")
    ho.add_argument("--symbol", default=None, help="Single pair to optimize (e.g. NASDAQ:TSLA); falls back to config pairlist")
    ho.add_argument("--timeframe", default=None)
    ho.add_argument("--start", default=None)
    ho.add_argument("--end", default=None)
    ho.add_argument("--session", default=None)
    ho.add_argument("--adjustment", default="splits", help="Price adjustment: splits, dividends, none (default: splits)")
    ho.add_argument("--strategy", default=None, help="Strategy name (default: from config)")
    ho.add_argument("--sl-min", type=float, default=None)
    ho.add_argument("--sl-max", type=float, default=None)
    ho.add_argument("--tp-min", type=float, default=None)
    ho.add_argument("--tp-max", type=float, default=None)
    ho.add_argument("--mode", choices=["long", "short", "both"], default="long")
    ho.add_argument(
        "--objective",
        choices=["net_profit_pct", "profit_factor", "win_rate_pct", "max_drawdown_pct", "trade_count"],
        default=None,
    )
    ho.add_argument("--top-n", type=int, default=None)
    ho.add_argument(
        "--search-method",
        choices=["grid", "bayesian"],
        default=None,
        help="Search strategy: 'grid' (two-stage coarse+fine) or 'bayesian' (Optuna TPE)",
    )
    ho.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="Number of Bayesian optimization trials (ignored for grid search)",
    )

    # ------ list-data ------
    subparsers.add_parser("list-data", help="List cached candle datasets")

    # ------ list-strategies ------
    subparsers.add_parser("list-strategies", help="List registered strategies")

    return parser


_COMMANDS = {
    "download-data": run_download_data,
    "backtest": run_backtest,
    "hyperopt": run_hyperopt,
    "list-data": run_list_data,
    "list-strategies": run_list_strategies,
}


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args.config)
    handler = _COMMANDS.get(args.command)
    if handler is None:
        parser.error(f"unknown command: {args.command}")
        return 1
    return handler(args, config)
