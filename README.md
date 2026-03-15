# HyperView

**Turn TradingView ideas into testable, terminal-speed trading systems.**

HyperView is for the moment a TradingView strategy stops being a chart experiment and starts needing hard evidence. It pulls historical candles straight from TradingView's websocket, runs your strategy logic in Python, and backtests with fill behavior designed to closely mirror Pine Script, so the results you tune locally in python still match the results you'll see on TradingView's strategy tester.

Instead of bouncing between Pine scripts, CSV exports, and improvised notebooks, HyperView gives you one clean loop: pull up to 40K bars, build on [TA-Lib](https://github.com/TA-Lib/ta-lib-python)'s 150+ indicators, simulate realistic SL/TP execution, and let Bayesian optimization hunt for better parameter ranges. No API keys. No browser automation. No spreadsheet cleanup. Just faster iteration, sharper validation, and a workflow built for traders who want to develop strategies like engineers.

## Prerequisites

- **Python 3.11+**
- **TA-Lib** — installed automatically by `pip install`. Pre-built wheels ship for Python 3.9–3.14 on Windows, macOS, and Linux.
- **Firefox** *(optional)* — If you have a TradingView paid plan, HyperView can read your Firefox session cookies to download up to **40K candles**. Without it, the websocket still downloads up to **5K candles** anonymously. To use this, just log in to [tradingview.com](https://www.tradingview.com) in Firefox before downloading data.

## Quick Start

```bash
# Install in editable mode (creates the `hyperview` CLI command, installs all dependencies including TA-Lib)
pip install -e .

# Download data for specific pairs
hyperview download-data --pairs NASDAQ:NFLX NASDAQ:AAPL --timeframe 1h --start 2023-01-03 --session extended

# Or define your pairs in config.json and just run:
hyperview download-data

# Run a single backtest (uses config pairlist)
hyperview backtest --sl 3.23 --tp 13.06 --mode long --start 2023-01-03

# Or target a specific symbol
hyperview backtest --symbol NASDAQ:NFLX --sl 3.23 --tp 13.06 --mode long --start 2023-01-03

# Hyper-optimize SL/TP across all pairs in config
hyperview hyperopt --mode long --start 2023-01-03

# List cached data and registered strategies
hyperview list-data
hyperview list-strategies
```

You can also run via `python -m _engine` instead of `hyperview`.

Python bytecode is redirected into the project-level `.pycache/` directory, so runtime imports do not create scattered `__pycache__` folders under `_engine/` or `strategy/`.

## How It Works

1. **Download** — Connects to TradingView's websocket using your existing Firefox session cookies. Supports up to 40K historical bars on paid plans with automatic backfill.
2. **Signal** — Runs a pluggable strategy (e.g. the included MACD+RSI or ADX+Stochastic) in pure Python with TA-Lib indicator parity.
3. **Backtest** — Simulates trades bar-by-bar using TradingView-parity fill assumptions (next-bar-open entry, intrabar SL/TP exit ordering).
4. **Hyper-Optimize** — Runs coarse+fine grid search or Bayesian TPE across SL/TP combinations, ranking by your chosen objective.

## Repository Layout

```
pyproject.toml                    Package metadata & `hyperview` CLI entry point
config.json                       Default configuration (timeframe, pairlist, opt ranges)
data/                             Cached candle data (CSV, auto-generated)
results/                          Optimization & backtest results (auto-generated)
strategy/                         Pluggable strategy framework (user-facing)
├── __init__.py                   Plugin registry & auto-discovery
├── macd_rsi.py                   MACD+RSI strategy (registered as "macd_rsi")
├── adx_stochastic.py             ADX+Stochastic strategy (registered as "adx_stochastic")
└── _lib/                         Shared strategy library
    ├── base.py                   BaseStrategy ABC & prepare_candles()
    └── indicators.py             TA-Lib wrappers, conversion helpers & signal toolkit
_engine/                          Internal processing logic
├── __main__.py                   Module entry point (python -m _engine)
├── cli/                          CLI router & subcommand handlers
│   ├── backtest.py               backtest command
│   ├── download.py               download-data command
│   ├── hyperopt.py               hyperopt command
│   └── list.py                   list-data & list-strategies commands
├── config.py                     Config loader (JSON merged with CLI overrides)
├── models.py                     Shared dataclasses (CandleRequest, Trade, BacktestMetrics, …)
├── backtest/
│   └── engine.py                 TradingView-parity OHLC simulator
├── downloader/
│   ├── client.py                 TradingView websocket downloader & multi-pair support
│   └── _tv/                      WebSocket internals (session, protocol, cache, credentials)
└── hyperopt/
    └── optimizer.py              Grid search + Bayesian (Optuna TPE)
```

## Configuration

HyperView loads defaults from `config.json` at the project root. CLI flags always override config values.

```json
{
    "timeframe": "1h",
    "session": "extended",
    "strategy": "macd_rsi",
    "initial_capital": 100000,
    "data_dir": "data",
    "output_dir": "results",
    "pairlist": [
        "NASDAQ:NFLX",
        "NASDAQ:TSLA",
        "COINBASE:BTCUSD"
    ],
    "optimization": {
        "search_method": "bayesian",
        "n_trials": 200,
        "objective": "net_profit_pct",
        "top_n": 10,
        "sl_range": { "min": 1.0, "max": 15.0 },
        "tp_range": { "min": 1.0, "max": 15.0 }
    }
}
```

Use `--config /path/to/custom.json` to load a different file.

### Pairlist

The `pairlist` array defines the symbols you want to work with. Every entry must use the `EXCHANGE:SYMBOL` format — this lets you mix pairs from different exchanges in a single config:

```json
"pairlist": [
    "NASDAQ:NFLX",
    "NASDAQ:TSLA",
    "NASDAQ:AAPL",
    "COINBASE:BTCUSD"
]
```

When you run a command without `--pairs` or `--symbol`, HyperView automatically uses the config pairlist — downloading, backtesting, or optimizing every pair in sequence. If you pass `--pairs` or `--symbol` on the CLI, the config pairlist is ignored for that run.

You can maintain separate config files for different asset classes:

```bash
hyperview --config stocks.json download-data
hyperview --config crypto.json hyperopt --mode long
```

## CLI Reference

### `download-data` — Fetch Candle Data

```bash
# Download all pairs from config pairlist
hyperview download-data

# Or specify pairs directly
hyperview download-data --pairs NASDAQ:NFLX NASDAQ:AAPL NASDAQ:TSLA --timeframe 1h --start 2023-01-03
```

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--pairs` | No | config pairlist | One or more `EXCHANGE:SYMBOL` pairs (overrides pairlist) |
| `--timeframe` | No | config | Bar interval: `1m` `5m` `15m` `1h` `4h` `1d` etc. |
| `--start` / `--end` | No | — | Date range (ISO format) |
| `--session` | No | config | `regular` or `extended` |
| `--adjustment` | No | `splits` | Price adjustment (`splits`, `dividends`, `none`) |

### `backtest` — Single Strategy Evaluation

```bash
# Backtest all pairs from config pairlist
hyperview backtest --sl 5.0 --tp 5.0 --mode long --start 2023-01-03

# Or target a specific symbol
hyperview backtest --symbol NASDAQ:NFLX --sl 5.0 --tp 5.0 --mode long --start 2023-01-03
```

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--symbol` | No | config pairlist | `EXCHANGE:SYMBOL` pair (overrides pairlist) |
| `--sl` | Yes | — | Stop-loss % |
| `--tp` | Yes | — | Take-profit % |
| `--strategy` | No | config | Strategy name (e.g. `macd_rsi`, `adx_stochastic`) |
| `--mode` | No | `long` | `long`, `short`, or `both` |
| `--timeframe`, `--session`, `--adjustment`, `--start`, `--end` | No | config / defaults | Standard filters |

### `hyperopt` — Hyper-Optimize SL/TP

```bash
# Optimize all pairs from config pairlist
hyperview hyperopt --search-method bayesian --n-trials 300

# Or target a specific symbol
hyperview hyperopt --symbol NASDAQ:NFLX --search-method bayesian --n-trials 300
```

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--symbol` | No | config pairlist | `EXCHANGE:SYMBOL` pair (overrides pairlist) |
| `--sl-min/max/step` | No | config | Stop-loss % search range |
| `--tp-min/max/step` | No | config | Take-profit % search range |
| `--search-method` | No | config | `grid` or `bayesian` |
| `--n-trials` | No | config | Bayesian trial count |
| `--objective` | No | config | `net_profit_pct` `profit_factor` `win_rate_pct` `max_drawdown_pct` `trade_count` |
| `--top-n` | No | config | Number of top candidates to keep |
| `--strategy`, `--mode`, `--timeframe`, `--adjustment`, etc. | No | config / defaults | Standard filters |

### `list-data` — Show Cached Datasets

```bash
hyperview list-data
```

### `list-strategies` — Show Available Strategies

```bash
hyperview list-strategies
```

## Indicators

HyperView ships with **20 wrapped indicators** backed by TA-Lib, plus **4 signal helpers**. You also have direct access to all **150+ TA-Lib functions** via the `to_numpy` / `wrap` conversion helpers.

### Wrapped Indicators

| Category | Functions |
|----------|-----------|
| **Moving Averages** | `ema`, `sma`, `wma` |
| **Momentum** | `rsi`, `macd`, `stochastic`, `stochastic_rsi`, `cci`, `williams_r`, `momentum`, `roc` |
| **Trend** | `adx` (returns ADX, +DI, −DI), `aroon` (returns down, up), `psar` |
| **Volatility** | `atr`, `bollinger_bands` (returns upper, middle, lower) |
| **Volume** | `obv`, `mfi`, `ad`, `vwap` |

### Signal Helpers

| Function | Description |
|----------|-------------|
| `crossed_above(a, b)` | True on bars where `a` crosses above `b` |
| `crossed_below(a, b)` | True on bars where `a` crosses below `b` |
| `barssince(cond)` | Bars since condition was last True |
| `to_unix_timestamp(dt)` | Convert ISO date string to UTC unix timestamp |

### Using TA-Lib Directly

For any of TA-Lib's 150+ functions not wrapped above, call `talib` directly and use the conversion helpers:

```python
import talib
from strategy._lib.indicators import to_numpy, wrap

df["cci"] = wrap(talib.CCI(to_numpy(df["high"]),
                            to_numpy(df["low"]),
                            to_numpy(df["close"]), timeperiod=20), df.index)
```

## Adding Custom Strategies

1. Create a new file in `strategy/` (e.g. `my_strategy.py`)
2. Subclass `BaseStrategy` and implement `generate_signals()`, `default_settings()`, `required_columns()`
3. Decorate the class with `@register_strategy`

Strategies are auto-discovered at startup — no manual imports needed.

```python
from strategy import register_strategy
from strategy._lib.base import BaseStrategy
from strategy._lib.indicators import ema, crossed_above

@register_strategy
class MyStrategy(BaseStrategy):
    strategy_name = "my_strategy"

    def default_settings(self):
        return {"fast_period": 10, "slow_period": 20}

    def required_columns(self):
        return ["time", "open", "high", "low", "close"]

    def generate_signals(self, candles, settings):
        df = self.prepare_candles(candles)

        fast = ema(df["close"], settings["fast_period"])
        slow = ema(df["close"], settings["slow_period"])

        df["buy_signal"] = crossed_above(fast, slow)
        df["sell_signal"] = crossed_above(slow, fast)
        df["in_date_range"] = True
        df["enable_long"] = True
        df["enable_short"] = False

        return df
```

Then use it: `hyperview backtest --symbol NASDAQ:NFLX --strategy my_strategy --sl 5 --tp 5`

## Output Files

Hyperopt writes a JSON file to `results/`:

```
results/macd_rsi_NASDAQ_NFLX_1h_long.json
```

The filename encodes `{strategy}_{exchange}_{symbol}_{timeframe}_{mode}` so each
combination produces its own file. The JSON contains the full request parameters,
ranked results, and coarse grid results (grid mode only).

## Backtest Assumptions

The simulator approximates TradingView's intrabar fill behavior:

- **Entry**: Signal-generated market orders fill on the **next bar open**
- **Intrabar path**: If a bar opens closer to its high, path is `open → high → low → close`; closer to its low, path is `open → low → high → close`
- **Position sizing**: 100% of equity per trade, no pyramiding
- **SL/TP exits**: Checked against the intrabar price path within the same bar
