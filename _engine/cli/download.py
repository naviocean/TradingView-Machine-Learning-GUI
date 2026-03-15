from __future__ import annotations

import argparse
import time

from .._utils import _format_time
from ..downloader import TradingViewDataClient
from ._helpers import _resolve, resolve_pairlist


def run_download_data(args: argparse.Namespace, config: dict) -> int:
    timeframe = _resolve(args, config, "timeframe")
    session = _resolve(args, config, "session")
    adjustment = args.adjustment
    data_dir = config["data_dir"]

    pairs = resolve_pairlist(args, config)

    print(f"\n{'='*60}")
    print("  HyperView - Download Data")
    print(f"  {len(pairs)} pair(s) | {timeframe}")
    for symbol, exch in pairs:
        print(f"    {exch}:{symbol}")
    if args.start:
        print(f"  Range: {args.start} -> {args.end or 'now'}")
    print(f"{'='*60}\n")

    client = TradingViewDataClient(cache_dir=data_dir)
    t0 = time.time()
    results = client.download_pairs(
        pairs=pairs,
        timeframe=timeframe,
        start=args.start,
        end=args.end,
        session=session,
        adjustment=adjustment,
    )
    print(f"\nDone - {len(results)} pair(s) downloaded ({_format_time(time.time() - t0)})")
    return 0
