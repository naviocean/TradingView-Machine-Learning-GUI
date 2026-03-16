from __future__ import annotations

from .models import Mode, RiskParameters


def _format_time(seconds: float) -> str:
    """Format seconds into a human-readable string."""
    total_seconds = max(0, round(seconds))
    if total_seconds < 60:
        return f"{total_seconds}s"

    minutes, secs = divmod(total_seconds, 60)
    if minutes < 60:
        return f"{minutes}m {secs:02d}s"

    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m {secs:02d}s"


def build_risk(mode: Mode, sl_value: float, tp_value: float) -> RiskParameters:
    """Construct RiskParameters based on trade mode and SL/TP percentages."""
    is_short = mode == "short"
    return RiskParameters(
        long_stoploss_pct=0.0 if is_short else sl_value,
        long_takeprofit_pct=0.0 if is_short else tp_value,
        short_stoploss_pct=sl_value,
        short_takeprofit_pct=tp_value,
    )
