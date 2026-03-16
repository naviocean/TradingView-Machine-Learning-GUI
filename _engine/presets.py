from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_ALLOWED_SESSIONS = {"regular", "extended"}
_ALLOWED_MODES = {"long", "short", "both"}
_ALLOWED_ADJUSTMENTS = {"splits", "dividends", "none"}
_ALLOWED_OBJECTIVES = {
    "net_profit_pct",
    "profit_factor",
    "win_rate_pct",
    "max_drawdown_pct",
    "trade_count",
}
_ALLOWED_SEARCH_METHODS = {"grid", "bayesian"}
_PRESET_SCOPE_KEYS = ("pair", "timeframe", "session", "adjustment", "mode")

# Built once at import time — avoids recreating these sets on every _validate_payload call.
_ALLOWED_TOP_LEVEL_KEYS: frozenset[str] = frozenset({"strategy", "updated_at", "presets"})
_ALLOWED_PRESET_KEYS: frozenset[str] = frozenset({
    "pair", "timeframe", "session", "adjustment", "mode",
    "sl", "tp", "objective", "search_method", "updated_at", "metrics",
})


def strategy_preset_path(output_dir: str | Path, strategy: str) -> Path:
    return Path(output_dir) / f"{strategy}_presets.json"


def load_strategy_presets(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Preset file not found: {file_path}")
    with open(file_path, encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Preset file must contain a JSON object: {file_path}")
    normalized = _normalize_payload(data)
    _validate_payload(normalized)
    return normalized


def find_preset(
    path: str | Path,
    *,
    strategy: str,
    pair: str,
    timeframe: str,
    session: str,
    adjustment: str,
    mode: str,
) -> dict[str, Any] | None:
    payload = load_strategy_presets(path)
    if payload.get("strategy") != strategy:
        raise ValueError(
            f"Preset file strategy mismatch: expected '{strategy}', found '{payload.get('strategy')}'."
        )
    scope = {
        "pair": pair,
        "timeframe": timeframe,
        "session": session,
        "adjustment": adjustment,
        "mode": mode,
    }
    for preset in payload["presets"]:
        if all(preset.get(key) == value for key, value in scope.items()):
            return dict(preset)
    return None


def save_best_preset(
    path: str | Path,
    *,
    strategy: str,
    pair: str,
    timeframe: str,
    session: str,
    adjustment: str,
    mode: str,
    sl: float,
    tp: float,
    objective: str,
    search_method: str,
    metrics: dict[str, Any] | None = None,
) -> Path:
    file_path = Path(path)
    payload = _read_existing_payload(file_path, strategy)
    preset: dict[str, Any] = {
        "pair": pair,
        "timeframe": timeframe,
        "session": session,
        "adjustment": adjustment,
        "mode": mode,
        "sl": float(sl),
        "tp": float(tp),
        "objective": objective,
        "search_method": search_method,
        "updated_at": _utc_now(),
    }
    if metrics is not None:
        preset["metrics"] = metrics

    payload["updated_at"] = _utc_now()
    payload["presets"] = [
        item for item in payload["presets"] if not _matches_scope(item, preset)
    ]
    payload["presets"].append(preset)
    payload["presets"] = sorted(
        payload["presets"],
        key=lambda item: (
            item["pair"],
            item["timeframe"],
            item["session"],
            item["adjustment"],
            item["mode"],
        ),
    )

    _validate_payload(payload)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(payload, indent=4) + "\n", encoding="utf-8")
    return file_path


def _read_existing_payload(file_path: Path, strategy: str) -> dict[str, Any]:
    if not file_path.exists():
        return {
            "strategy": strategy,
            "updated_at": _utc_now(),
            "presets": [],
        }

    payload = load_strategy_presets(file_path)
    if payload.get("strategy") != strategy:
        raise ValueError(
            f"Preset file strategy mismatch: expected '{strategy}', found '{payload.get('strategy')}'."
        )
    return deepcopy(payload)


def _normalize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    # deepcopy already produces independent copies of all nested dicts — no further wrapping needed.
    return deepcopy(payload)


def _validate_payload(payload: dict[str, Any]) -> None:
    unknown_top_level = sorted(set(payload) - _ALLOWED_TOP_LEVEL_KEYS)
    if unknown_top_level:
        raise ValueError(
            f"Preset file contains unsupported top-level keys: {', '.join(unknown_top_level)}"
        )

    _require_non_empty_string(payload.get("strategy"), "strategy")
    _require_non_empty_string(payload.get("updated_at"), "updated_at")
    presets = payload.get("presets")
    if not isinstance(presets, list):
        raise ValueError("Preset file value 'presets' must be a list")

    for index, preset in enumerate(presets):
        prefix = f"presets[{index}]"
        if not isinstance(preset, dict):
            raise ValueError(f"Preset file value '{prefix}' must be an object")
        unknown_keys = sorted(set(preset) - _ALLOWED_PRESET_KEYS)
        if unknown_keys:
            raise ValueError(
                f"Preset file value '{prefix}' contains unsupported keys: {', '.join(unknown_keys)}"
            )

        _require_pair_string(preset.get("pair"), f"{prefix}.pair")
        _require_non_empty_string(preset.get("timeframe"), f"{prefix}.timeframe")
        _require_choice(preset.get("session"), _ALLOWED_SESSIONS, f"{prefix}.session")
        _require_choice(preset.get("adjustment"), _ALLOWED_ADJUSTMENTS, f"{prefix}.adjustment")
        _require_choice(preset.get("mode"), _ALLOWED_MODES, f"{prefix}.mode")
        _require_positive_number(preset.get("sl"), f"{prefix}.sl")
        _require_positive_number(preset.get("tp"), f"{prefix}.tp")
        _require_choice(preset.get("objective"), _ALLOWED_OBJECTIVES, f"{prefix}.objective")
        _require_choice(preset.get("search_method"), _ALLOWED_SEARCH_METHODS, f"{prefix}.search_method")
        _require_non_empty_string(preset.get("updated_at"), f"{prefix}.updated_at")
        if "metrics" in preset:
            _validate_metrics(preset["metrics"], f"{prefix}.metrics")


def _matches_scope(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return all(left.get(key) == right.get(key) for key in _PRESET_SCOPE_KEYS)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _require_choice(value: Any, allowed: set[str], key: str) -> None:
    if not isinstance(value, str) or value not in allowed:
        choices = ", ".join(sorted(allowed))
        raise ValueError(f"Preset file value '{key}' must be one of: {choices}")


def _require_non_empty_string(value: Any, key: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Preset file value '{key}' must be a non-empty string")


def _require_positive_number(value: Any, key: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) <= 0:
        raise ValueError(f"Preset file value '{key}' must be greater than 0")


def _validate_metrics(value: Any, key: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"Preset file value '{key}' must be an object")


def _require_pair_string(value: Any, key: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Preset file value '{key}' must be a non-empty string")
    parts = value.split(":")
    if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
        raise ValueError(
            f"Preset file value '{key}' has invalid format '{value}'. Expected 'EXCHANGE:SYMBOL'."
        )
