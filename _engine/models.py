from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Mode = Literal["long", "short", "both"]
Objective = Literal[
    "net_profit_pct",
    "profit_factor",
    "win_rate_pct",
    "max_drawdown_pct",
    "trade_count",
]
SearchMethod = Literal["grid", "bayesian"]


# ---------------------------------------------------------------------------
# Request and input models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CandleRequest:
    symbol: str
    exchange: str
    timeframe: str
    start: str | None = None
    end: str | None = None
    session: str = "regular"
    adjustment: str = "splits"
    mintick: float = 0.01


@dataclass(frozen=True)
class RiskParameters:
    long_stoploss_pct: float
    long_takeprofit_pct: float
    short_stoploss_pct: float
    short_takeprofit_pct: float


@dataclass(frozen=True)
class OptimizationRequest:
    candle_request: CandleRequest
    mode: Mode
    objective: Objective
    sl_min: float
    sl_max: float
    sl_step: float
    tp_min: float
    tp_max: float
    tp_step: float
    top_n: int = 10
    initial_equity: float = 100_000.0
    fine_factor: int = 2
    search_method: SearchMethod = "grid"
    n_trials: int = 200


# ---------------------------------------------------------------------------
# Runtime and result models
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    entry_time: int
    exit_time: int
    direction: Literal["long", "short"]
    entry_price: float
    exit_price: float
    exit_reason: str
    return_pct: float
    equity_before: float
    equity_after: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BacktestMetrics:
    symbol: str
    timeframe: str
    start: str | None
    end: str | None
    mode: Mode
    sl_pct: float
    tp_pct: float
    net_profit_pct: float
    max_drawdown_pct: float
    win_rate_pct: float
    profit_factor: float
    trade_count: int
    equity_final: float
    rank: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return _rounded_metrics_dict(self)


@dataclass
class BacktestResult:
    metrics: BacktestMetrics
    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = self.metrics.to_dict()
        payload["trades"] = [trade.to_dict() for trade in self.trades]
        payload["equity_curve"] = self.equity_curve
        return payload


@dataclass
class OptimizationBundle:
    request: OptimizationRequest
    results: list[BacktestMetrics]
    coarse_results: list[BacktestMetrics]
    output_path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "request": {
                "candle_request": asdict(self.request.candle_request),
                "mode": self.request.mode,
                "objective": self.request.objective,
                "sl_min": self.request.sl_min,
                "sl_max": self.request.sl_max,
                "sl_step": self.request.sl_step,
                "tp_min": self.request.tp_min,
                "tp_max": self.request.tp_max,
                "tp_step": self.request.tp_step,
                "top_n": self.request.top_n,
                "initial_equity": self.request.initial_equity,
                "fine_factor": self.request.fine_factor,
                "search_method": self.request.search_method,
                "n_trials": self.request.n_trials,
            },
            "results": [result.to_dict() for result in self.results],
            "coarse_results": [result.to_dict() for result in self.coarse_results],
        }


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

_ROUNDED_METRIC_FIELDS = (
    "sl_pct",
    "tp_pct",
    "net_profit_pct",
    "max_drawdown_pct",
    "win_rate_pct",
    "profit_factor",
    "equity_final",
)


def _rounded_metrics_dict(metrics: BacktestMetrics) -> dict[str, Any]:
    data = asdict(metrics)
    for key in _ROUNDED_METRIC_FIELDS:
        if isinstance(data.get(key), float):
            data[key] = round(data[key], 2)
    return data
