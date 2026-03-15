from __future__ import annotations

from pathlib import Path

import pandas as pd

from ...models import CandleRequest
from .timeframes import to_timestamp


class CandleCache:
    """CSV-backed candle cache keyed by exchange/symbol/timeframe."""

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load(self, request: CandleRequest) -> pd.DataFrame | None:
        path = self._resolve_existing_path(request)
        if not path.exists():
            return None
        return pd.read_csv(path)

    def write(self, request: CandleRequest, frame: pd.DataFrame) -> None:
        path = self._path(request)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        legacy_path = self._legacy_path(request)
        if legacy_path != path and legacy_path.exists():
            legacy_path.unlink()

    def covers_range(self, frame: pd.DataFrame, request: CandleRequest) -> bool:
        if frame.empty:
            return False
        start = to_timestamp(request.start)
        end = to_timestamp(request.end)
        if start is None and end is None:
            # No explicit range requested — cache is valid as-is.
            return True
        start_ok = True if start is None else int(frame["time"].min()) <= start
        end_ok = True if end is None else int(frame["time"].max()) >= end
        return start_ok and end_ok

    def _path(self, request: CandleRequest) -> Path:
        return self.cache_dir / f"{self._stem(request)}.csv"

    def _resolve_existing_path(self, request: CandleRequest) -> Path:
        preferred = self._path(request)
        if preferred.exists():
            return preferred
        legacy = self._legacy_path(request)
        if legacy.exists():
            return legacy
        return preferred

    def _stem(self, request: CandleRequest) -> str:
        timeframe = _sanitize_cache_component(request.timeframe)
        session = _sanitize_cache_component(request.session)
        exchange = _sanitize_cache_component(request.exchange)
        symbol = _sanitize_cache_component(request.symbol)
        adjustment = _sanitize_cache_component(request.adjustment)
        filename = f"{timeframe}-{session}-{exchange}-{symbol}"
        # Keep non-default adjustments distinct without bloating the common case.
        if adjustment != "splits":
            filename += f"-{adjustment}"
        return filename

    def _legacy_path(self, request: CandleRequest) -> Path:
        exchange = request.exchange.replace(":", "_")
        symbol = request.symbol.replace(":", "_")
        timeframe = request.timeframe.replace(":", "_")
        session = request.session.replace(":", "_")
        adjustment = request.adjustment.replace(":", "_")
        filename = f"{exchange}_{symbol}_{timeframe}_{session}"
        if adjustment != "splits":
            filename += f"_{adjustment}"
        return self.cache_dir / f"{filename}.csv"


def _sanitize_cache_component(value: str) -> str:
    return value.replace(":", "_")


# ------------------------------------------------------------------
# Stateless DataFrame utilities
# ------------------------------------------------------------------

def slice_frame(frame: pd.DataFrame, request: CandleRequest) -> pd.DataFrame:
    dataframe = frame.copy()
    start = to_timestamp(request.start)
    end = to_timestamp(request.end)
    if start is not None:
        dataframe = dataframe[dataframe["time"] >= start]
    if end is not None:
        dataframe = dataframe[dataframe["time"] < end]
    return dataframe.reset_index(drop=True)


def merge_frames(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    merged = pd.concat([left, right], ignore_index=True)
    merged = merged.sort_values("time").drop_duplicates(subset=["time"]).reset_index(drop=True)
    return merged
