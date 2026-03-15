from __future__ import annotations

import argparse
import time
from typing import TYPE_CHECKING

import pandas as pd

from .._utils import _format_time
from ..downloader import TradingViewDataClient
from strategy import get_strategy

if TYPE_CHECKING:
    from ..models import CandleRequest


def _resolve(args: argparse.Namespace, config: dict, key: str, default=None):
    """Return CLI override if set, else config value, else default."""
    cli_value = getattr(args, key, None)
    if cli_value is not None:
        return cli_value
    return config.get(key, default)


def parse_pair(entry: str) -> tuple[str, str]:
    """Parse 'EXCHANGE:SYMBOL' into (symbol, exchange)."""
    if ":" not in entry:
        raise SystemExit(
            f"Error: Invalid pair format '{entry}'. "
            f"Expected 'EXCHANGE:SYMBOL' (e.g. 'NASDAQ:AAPL')."
        )
    exchange, symbol = entry.split(":", 1)
    exchange, symbol = exchange.strip(), symbol.strip()
    if not exchange or not symbol:
        raise SystemExit(f"Error: Invalid pair format '{entry}'. Expected 'EXCHANGE:SYMBOL'.")
    return symbol, exchange


def resolve_pairlist(args: argparse.Namespace, config: dict) -> list[tuple[str, str]]:
    """Return list of (symbol, exchange) tuples from --pairs, --symbol, or config pairlist."""
    # Explicit --pairs (download-data)
    pairs_arg = getattr(args, "pairs", None)
    if pairs_arg:
        return [parse_pair(p) for p in pairs_arg]
    # Explicit --symbol (backtest / hyperopt)
    symbol = getattr(args, "symbol", None)
    if symbol:
        return [parse_pair(symbol)]
    # Fall back to config pairlist
    entries = config.get("pairlist", [])
    if not entries:
        raise SystemExit(
            "Error: No pairs specified. Either use --pairs / --symbol on the CLI,\n"
            "or define a 'pairlist' in your config file."
        )
    return [parse_pair(e) for e in entries]


def load_candles(candle_request: "CandleRequest", data_dir: str, *, step: str = "") -> pd.DataFrame:
    """Load cached candle data and print progress."""
    print(f"\n[{step}] Loading candle data...")
    t0 = time.time()
    client = TradingViewDataClient(cache_dir=data_dir)
    candles = client.get_history(candle_request, cache_only=True)
    print(f"      {len(candles)} candles loaded ({_format_time(time.time() - t0)})")
    return candles


def generate_signals(
    strategy_name: str,
    candles: pd.DataFrame,
    mode: str,
    start: str | None,
    end: str | None,
    *,
    step: str = "",
) -> tuple[pd.DataFrame, "get_strategy"]:
    """Instantiate strategy, generate signals, and print progress."""
    print(f"\n[{step}] Generating signals...")
    t0 = time.time()
    strategy = get_strategy(strategy_name)
    signal_settings = {
        "enable_long": mode in {"long", "both"},
        "enable_short": mode in {"short", "both"},
        "start": start,
        "end": end,
    }
    signal_frame = strategy.generate_signals(candles, signal_settings)
    buy_count = int(signal_frame["buy_signal"].sum())
    sell_count = int(signal_frame["sell_signal"].sum())
    print(f"      {buy_count} buy / {sell_count} sell signals ({_format_time(time.time() - t0)})")
    return signal_frame, strategy
