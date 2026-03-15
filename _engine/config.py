from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any


_DEFAULTS: dict[str, Any] = {
    "exchange": "NASDAQ",
    "timeframe": "1h",
    "session": "regular",
    "adjustment": "splits",
    "initial_capital": 100_000,
    "data_dir": "data",
    "output_dir": "results",
    "optimization": {
        "search_method": "grid",
        "n_trials": 200,
        "objective": "net_profit_pct",
        "top_n": 10,
        "fine_factor": 2,
        "sl_range": {"min": 1.0, "max": 15.0, "step": 0.5},
        "tp_range": {"min": 1.0, "max": 15.0, "step": 0.5},
    },
}

_SEARCH_PATHS = [
    Path("config.json"),
    Path("hyperview.json"),
]

_ALLOWED_SESSIONS = {"regular", "extended"}
_ALLOWED_ADJUSTMENTS = {"splits", "dividends", "none"}
_ALLOWED_OBJECTIVES = {
    "net_profit_pct",
    "profit_factor",
    "win_rate_pct",
    "max_drawdown_pct",
    "trade_count",
}
_ALLOWED_SEARCH_METHODS = {"grid", "bayesian"}


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load configuration from a JSON file, falling back to built-in defaults."""
    config = deepcopy(_DEFAULTS)
    file_path = _resolve_config_path(path)
    if file_path is not None:
        with open(file_path, encoding="utf-8") as fh:
            user_config = json.load(fh)
        if not isinstance(user_config, dict):
            raise ValueError(f"Config file must contain a JSON object: {file_path}")
        _deep_merge(config, user_config)
    normalized = _normalize_config(config)
    _validate_config(normalized)
    return normalized


def _resolve_config_path(path: str | Path | None) -> Path | None:
    if path is not None:
        candidate = Path(path)
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(f"Config file not found: {candidate}")
    for candidate in _SEARCH_PATHS:
        if candidate.is_file():
            return candidate
    return None


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> None:
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def _normalize_config(config: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(config)
    optimization = normalized.get("optimization", {})
    if isinstance(optimization, dict):
        normalized["optimization"] = dict(optimization)
        for key in ("sl_range", "tp_range"):
            range_config = normalized["optimization"].get(key, {})
            if isinstance(range_config, dict):
                normalized["optimization"][key] = dict(range_config)
    return normalized


def _validate_config(config: dict[str, Any]) -> None:
    _require_non_empty_string(config, "exchange")
    _require_non_empty_string(config, "timeframe")
    _require_choice(config, "session", _ALLOWED_SESSIONS)
    _require_choice(config, "adjustment", _ALLOWED_ADJUSTMENTS)
    _require_positive_number(config, "initial_capital")
    _require_non_empty_string(config, "data_dir")
    _require_non_empty_string(config, "output_dir")

    if "strategy" in config and config["strategy"] is not None:
        _require_non_empty_string(config, "strategy")

    optimization = config.get("optimization")
    if not isinstance(optimization, dict):
        raise ValueError("Config value 'optimization' must be an object")

    _require_choice(optimization, "search_method", _ALLOWED_SEARCH_METHODS, prefix="optimization.")
    _require_positive_int(optimization, "n_trials", prefix="optimization.")
    _require_choice(optimization, "objective", _ALLOWED_OBJECTIVES, prefix="optimization.")
    _require_positive_int(optimization, "top_n", prefix="optimization.")
    _require_positive_int(optimization, "fine_factor", prefix="optimization.")

    _validate_range_config(optimization, "sl_range")
    _validate_range_config(optimization, "tp_range")


def _validate_range_config(optimization: dict[str, Any], key: str) -> None:
    range_config = optimization.get(key)
    prefix = f"optimization.{key}."
    if not isinstance(range_config, dict):
        raise ValueError(f"Config value 'optimization.{key}' must be an object")

    _require_number(range_config, "min", prefix=prefix)
    _require_number(range_config, "max", prefix=prefix)
    _require_positive_number(range_config, "step", prefix=prefix)

    min_value = float(range_config["min"])
    max_value = float(range_config["max"])
    if min_value > max_value:
        raise ValueError(
            f"Config value 'optimization.{key}.min' must be less than or equal to "
            f"'optimization.{key}.max'"
        )


def _require_choice(
    config: dict[str, Any],
    key: str,
    allowed: set[str],
    *,
    prefix: str = "",
) -> None:
    value = config.get(key)
    if not isinstance(value, str) or value not in allowed:
        choices = ", ".join(sorted(allowed))
        raise ValueError(f"Config value '{prefix}{key}' must be one of: {choices}")


def _require_non_empty_string(config: dict[str, Any], key: str, *, prefix: str = "") -> None:
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Config value '{prefix}{key}' must be a non-empty string")


def _require_positive_int(config: dict[str, Any], key: str, *, prefix: str = "") -> None:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"Config value '{prefix}{key}' must be a positive integer")


def _require_number(config: dict[str, Any], key: str, *, prefix: str = "") -> None:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Config value '{prefix}{key}' must be a number")


def _require_positive_number(config: dict[str, Any], key: str, *, prefix: str = "") -> None:
    _require_number(config, key, prefix=prefix)
    if float(config[key]) <= 0:
        raise ValueError(f"Config value '{prefix}{key}' must be greater than 0")
